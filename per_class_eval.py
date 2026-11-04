#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对单个 checkpoint 在患者级测试集上做逐类别评估，输出画图 7/图 8 所需的原料：
  1) predictions.npz      —— logits / labels / label_names，本地画混淆矩阵与 ROC 用
  2) per_class_summary.json —— 逐类 TP/FP/TN/FN、灵敏度/特异度/P/R/F1/AUC/ROC 点/支持 + micro/macro

与训练/测试共用同一套数据加载与模型构造逻辑，并同样加载 EMA 权重（与 main.py --mode test 一致）。

用法（在 code1021 目录下，服务器上执行）：
  --config 用「对应 seed 目录下的 config.yaml」（与 main.py --mode test 相同），
  这样 seed / save_dir / split_file 才与 checkpoint 严格一致。

  python per_class_eval.py \
      --config ../results1021/ablation_06_no_contrastive/seed_42/config.yaml \
      --checkpoint ../results1021/ablation_06_no_contrastive/seed_42/checkpoints/best_model.pth \
      --output ../results1021/ablation_06_no_contrastive/seed_42/per_class \
      --device cuda

一次性跑 5 个 seed：
  for s in 42 123 2026 3407 9001; do
    python per_class_eval.py \
      --config ../results1021/ablation_06_no_contrastive/seed_${s}/config.yaml \
      --checkpoint ../results1021/ablation_06_no_contrastive/seed_${s}/checkpoints/best_model.pth \
      --output ../results1021/ablation_06_no_contrastive/seed_${s}/per_class \
      --device cuda
  done

若结果根目录不是 ../results1021，替换成实际绝对路径即可。
"""
import os
import sys
import json
import argparse

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.eye_mil_dataset import (
    build_eye_records, EyeMILDataset, mil_collate_fn,
    build_transforms, BorderCrop, CLAHE, SpeckleNoise,
)
from datasets.roi_usaug import ROICrop, UltrasoundInterferenceAugment
from models.model import CLSTransformerMIL
from engine.ema import ModelEMA
from utils.metrics import compute_metrics, format_metrics, _roc_auc
from utils.protocol import validate_patient_split, validate_checkpoint_metadata


def parse_args():
    p = argparse.ArgumentParser(description="Per-class evaluation on the patient-level test split.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def roc_curve_points(y_true, y_score):
    """简化 ROC 曲线点（降序累计），返回 (fpr, tpr)。无正/负样本返回空。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [], []
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    tpr = np.cumsum(yt) / n_pos
    fpr = np.cumsum(1 - yt) / n_neg
    return np.concatenate([[0.0], fpr]).tolist(), np.concatenate([[0.0], tpr]).tolist()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[PerClass] device: {device}")

    # ---------------- 数据集（与 main.py 一致） ----------------
    crop_cfg = cfg["data"].get("crop", {})
    save_dir = cfg["logging"]["save_dir"]
    split_file = cfg["logging"].get("split_file") or os.path.join(save_dir, "splits.json")

    _, _, test_records, label_names = build_eye_records(
        excel_path=cfg["data"]["excel_path"],
        image_root=cfg["data"]["image_root"],
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"].get("test_ratio", 0.2),
        seed=cfg["seed"],
        split_file=split_file,
        split_save_dir=save_dir,
        crop_cfg=crop_cfg,
        require_patient_split=cfg["data"].get("require_patient_split", False),
    )
    num_classes = len(label_names)
    split_audit = None
    if cfg["data"].get("require_patient_split", False):
        _, split_audit = validate_patient_split(split_file)
    print(f"[PerClass] classes: {num_classes} | labels: {label_names}")

    image_size = cfg["data"]["image_size"]
    roi_cfg = cfg["data"].get("roi", {})
    border_crop = None
    if roi_cfg.get("enabled", False):
        rcrop = roi_cfg.get("crop", {})
        border_crop = ROICrop(left=rcrop.get("left", 0.2), right=rcrop.get("right", 0.3),
                              top=rcrop.get("top", 0.25), bottom=rcrop.get("bottom", 0.25),
                              darken=roi_cfg.get("darken", 0.22))
    elif crop_cfg:
        ratios = {k: crop_cfg.get(k, 0.0) for k in ["top", "bottom", "left", "right"]}
        if any(v > 0.0 for v in ratios.values()):
            border_crop = BorderCrop(**ratios)

    val_transform = build_transforms(image_size=image_size, is_train=False, border_crop=border_crop)
    test_dataset = EyeMILDataset(test_records, label_names, transform=val_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             num_workers=cfg["data"]["num_workers"], pin_memory=True,
                             collate_fn=mil_collate_fn)

    # ---------------- 模型（与 main.py 一致） ----------------
    model_cfg = cfg["model"]
    contrastive_cfg = cfg.get("contrastive", {})
    label_adj = None
    if model_cfg.get("use_label_gcn", False):
        from models.label_gcn import build_label_adjacency
        label_adj = build_label_adjacency(train_records, num_classes,
                                          threshold=model_cfg.get("label_gcn_threshold", 0.4))

    model = CLSTransformerMIL(
        num_classes=num_classes,
        pretrained=False,
        embed_dim=model_cfg["embed_dim"],
        fusion_heads=model_cfg["fusion_heads"],
        fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 128),
        dropout=model_cfg["dropout"],
        backbone_name=model_cfg.get("backbone", "convnext_tiny"),
        use_contrastive=contrastive_cfg.get("enabled", True),
        contrastive_proj_dim=contrastive_cfg.get("proj_dim", 128),
        attn_dropout=model_cfg.get("attn_dropout", 0.0),
        use_cross_view=model_cfg.get("use_cross_view", True),
        cross_view_layers=model_cfg.get("cross_view_layers", 1),
        cross_view_heads=model_cfg.get("cross_view_heads", 4),
        use_view_gate=model_cfg.get("use_view_gate", True),
        use_cross_view_gate=model_cfg.get("use_cross_view_gate", True),
        use_label_gcn=model_cfg.get("use_label_gcn", False),
        label_adj=label_adj,
        evidence_config=model_cfg.get("evidence", {}),
    ).to(device)

    # ---------------- 加载 checkpoint（含 EMA，与 main.py test 一致） ----------------
    print(f"[PerClass] loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    validate_checkpoint_metadata(ckpt, label_names, split_audit, model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    if "ema_state_dict" in ckpt:
        ema_ld = ModelEMA(model, decay=ckpt["ema_state_dict"].get("decay", 0.999))
        ema_ld.load_state_dict(ckpt["ema_state_dict"])
        ema_ld.apply_shadow(model)
    model.eval()

    # ---------------- 推理，收集 logits / labels ----------------
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, view_mask, labels in test_loader:
            images = images.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            logits, _ = model(images, view_mask)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))  # sigmoid，等价于 metrics._to_probs

    threshold = float(cfg["eval"]["threshold"])
    preds = (probs >= threshold).astype(np.float32)
    n_test = labels.shape[0]

    # ---------------- 汇总指标 ----------------
    summary_metrics = compute_metrics(logits, labels, threshold=threshold)

    per_class = []
    for c, name in enumerate(label_names):
        y = labels[:, c].astype(int)
        p = preds[:, c].astype(int)
        tp = int((p * y).sum())
        fp = int((p * (1 - y)).sum())
        fn = int(((1 - p) * y).sum())
        tn = int(((1 - p) * (1 - y)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        auc = _roc_auc(y.astype(np.float64), probs[:, c].astype(np.float64))
        fpr, tpr = roc_curve_points(y.astype(np.float64), probs[:, c].astype(np.float64))
        per_class.append({
            "label": name, "support": int(y.sum()),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": round(float(sens), 4), "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4), "recall": round(float(rec), 4),
            "f1": round(float(f1), 4), "auc": round(float(auc), 4),
            "roc": {"fpr": [round(float(x), 5) for x in fpr],
                    "tpr": [round(float(x), 5) for x in tpr]},
        })

    result = {
        "seed": cfg.get("seed"),
        "checkpoint": os.path.abspath(args.checkpoint),
        "threshold": threshold,
        "label_names": list(label_names),
        "n_test_eyes": n_test,
        "micro": {k: round(float(v), 4) for k, v in summary_metrics.items()
                  if k.startswith("micro_")},
        "macro": {k: round(float(v), 4) for k, v in summary_metrics.items()
                  if k.startswith("macro_")},
        "exact_match_acc": round(float(summary_metrics["exact_match_acc"]), 4),
        "hamming_loss": round(float(summary_metrics["hamming_loss"]), 4),
        "per_class": per_class,
    }

    # ---------------- 输出 ----------------
    os.makedirs(args.output, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.output, "predictions.npz"),
        logits=logits, labels=labels, probs=probs,
        label_names=np.asarray(label_names, dtype=str))
    with open(os.path.join(args.output, "per_class_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------------- 打印 ----------------
    print("\n" + format_metrics(summary_metrics, label_names))
    print("\nPer-class confusion (threshold=%.2f):" % threshold)
    print(f"{'Class':<10}{'Sup':>5}{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}{'Sens':>8}{'Spec':>8}{'F1':>8}{'AUC':>8}")
    for r in per_class:
        print(f"{r['label']:<10}{r['support']:>5}{r['tp']:>5}{r['fp']:>5}{r['tn']:>5}{r['fn']:>5}"
              f"{r['sensitivity']:>8.3f}{r['specificity']:>8.3f}{r['f1']:>8.3f}{r['auc']:>8.3f}")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
