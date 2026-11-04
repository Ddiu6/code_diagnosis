"""
OURS: View-Gated Multi-Head Attention MIL — Main Entry Point (Phase 2).

Phase 2 改进:
    1. Multi-Head Attention Pooling (MAD-MIL 风格) — K 头并行注意力
    2. View Gate — 逐视图维度门控，抑制噪声
    3. Cross-View Contrastive Loss — 拉近同眼视图、推远异眼视图

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint checkpoints/best_model.pth
    python main.py --mode test --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import time
import argparse
import json
import random
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.eye_mil_dataset import (
    build_eye_records, EyeMILDataset, mil_collate_fn,
    build_transforms, BorderCrop, CLAHE, SpeckleNoise,
)
from datasets.roi_usaug import ROICrop, UltrasoundInterferenceAugment
from models.model import CLSTransformerMIL
from models.losses import build_loss
from engine.train import train_one_epoch, validate, build_scheduler, build_optimizer
from engine.ema import ModelEMA
from engine.early_stopping import EarlyStopping
from engine.evaluate import evaluate
from utils.metrics import compute_metrics, format_metrics
from utils.logger import Logger
from utils.loss_config import resolve_training_loss
from utils.protocol import validate_patient_split, validate_checkpoint_metadata


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="OURS Phase 2: View-Gated Multi-Head Attention MIL")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "test", "audit_ema"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--audit-output-dir", type=str, default=None,
                        help="Separate output directory for validation-only raw/EMA audit.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")
    set_seed(cfg["seed"])

    # ================================================================
    # Build eye-level dataset records
    # ================================================================
    print("\n[Main] Building eye-level dataset records...")
    crop_cfg = cfg["data"].get("crop", {})
    # split 文件：留空则自动使用 save_dir/splits.json（训练时生成、测试时复用，保证划分一致）
    save_dir = cfg["logging"]["save_dir"]
    split_file = cfg["logging"].get("split_file") or os.path.join(save_dir, "splits.json")
    train_records, val_records, test_records, label_names = build_eye_records(
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
    print(f"[Main] Classes: {num_classes} | Labels: {label_names}")

    image_size = cfg["data"]["image_size"]

    # Border crop（ROI 启用时用 ROICrop 替代 BorderCrop：固定裁边+内容框收紧+椭圆约束）
    roi_cfg = cfg["data"].get("roi", {})
    border_crop = None
    if roi_cfg.get("enabled", False):
        rcrop = roi_cfg.get("crop", {})
        border_crop = ROICrop(
            left=rcrop.get("left", 0.2), right=rcrop.get("right", 0.3),
            top=rcrop.get("top", 0.25), bottom=rcrop.get("bottom", 0.25),
            darken=roi_cfg.get("darken", 0.22),
        )
        print(f"[Main] ROI crop: {border_crop}")
    elif crop_cfg:
        ratios = {k: crop_cfg.get(k, 0.0) for k in ["top", "bottom", "left", "right"]}
        if any(v > 0.0 for v in ratios.values()):
            border_crop = BorderCrop(**ratios)
            print(f"[Main] Border crop: {border_crop}")

    # Augmentations
    aug_cfg = cfg["data"].get("augmentation", {})
    clahe = None
    if aug_cfg.get("clahe", {}).get("enabled", False):
        c = aug_cfg["clahe"]
        clahe = CLAHE(clip_limit=c.get("clip_limit", 2.0),
                      tile_grid_size=tuple(c.get("tile_grid_size", [8, 8])),
                      p=c.get("p", 0.5))
        print(f"[Main] CLAHE: {clahe}")

    speckle_noise = None
    if aug_cfg.get("speckle_noise", {}).get("enabled", False):
        s = aug_cfg["speckle_noise"]
        speckle_noise = SpeckleNoise(sigma=s.get("sigma", 0.05), p=s.get("p", 0.5))
        print(f"[Main] SpeckleNoise: {speckle_noise}")

    # 超声干扰仿真增强（Rayleigh散斑/回声增强/设备伪影，仅训练期生效）
    us_augment = None
    usa_cfg = cfg["data"].get("us_augment", {})
    if usa_cfg.get("enabled", False):
        us_augment = UltrasoundInterferenceAugment(
            speckle_p=usa_cfg.get("speckle_p", 0.30),
            echo_p=usa_cfg.get("echo_p", 0.25),
            artifact_p=usa_cfg.get("artifact_p", 0.20),
            max_ops=usa_cfg.get("max_ops", 2),
        )
        print(f"[Main] Ultrasound interference augment: {us_augment}")

    train_transform = build_transforms(
        image_size=image_size, is_train=True,
        border_crop=border_crop, clahe=clahe, speckle_noise=speckle_noise,
        us_augment=us_augment,
        rotation_degrees=aug_cfg.get("rotation_degrees", 10),
        rotation_p=aug_cfg.get("rotation_p", 1.0),
    )
    val_transform = build_transforms(
        image_size=image_size, is_train=False, border_crop=border_crop,
    )

    train_dataset = EyeMILDataset(train_records, label_names, transform=train_transform, is_train=True)
    val_dataset = EyeMILDataset(val_records, label_names, transform=val_transform, is_train=False)
    test_dataset = EyeMILDataset(test_records, label_names, transform=val_transform, is_train=False)

    batch_size = cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=nw, pin_memory=True, collate_fn=mil_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True, collate_fn=mil_collate_fn)

    # ================================================================
    # Build model
    # ================================================================
    model_cfg = cfg["model"]
    contrastive_cfg = cfg.get("contrastive", {})

    # 标签共现邻接矩阵（用于 Label GCN 模块）
    label_adj = None
    if model_cfg.get("use_label_gcn", False):
        from models.label_gcn import build_label_adjacency
        label_adj = build_label_adjacency(
            train_records, num_classes,
            threshold=model_cfg.get("label_gcn_threshold", 0.4),
        )
        print(f"[Main] Label GCN adjacency: {label_adj.shape}")

    model = CLSTransformerMIL(
        num_classes=num_classes,
        pretrained=model_cfg["pretrained"] and args.mode == "train",
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
    )
    model = model.to(device)

    tp = sum(p.numel() for p in model.parameters())
    trp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Main] Params: {tp:,} total, {trp:,} trainable")

    # ================================================================
    # Mode: Train
    # ================================================================
    if args.mode == "train":
        logger = Logger(cfg["logging"]["log_dir"], "train.log")
        # 保存本次实验的 config 快照到 logs 目录（便于复现）
        _cfg_snapshot = os.path.join(cfg["logging"]["log_dir"], "config.yaml")
        with open(_cfg_snapshot, "w", encoding="utf-8") as _f:
            yaml.dump(cfg, _f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.log(f"Config snapshot saved to: {_cfg_snapshot}")
        architecture = "Coverage evidence MIL" if model_cfg.get("evidence", {}).get("enabled", False) else "View-gated MHA MIL"
        logger.log(f"OURS: {architecture}")
        logger.log(f"  Contrastive: {'on' if contrastive_cfg.get('enabled', True) else 'off'}, "
                   f"λ={contrastive_cfg.get('weight', 0.1)}")
        logger.log(f"  Loss: {cfg['loss']['name']}")
        logger.log(f"  Image: {image_size}")

        extra = {
            "device": str(device), "num_classes": num_classes,
            "label_names": str(label_names),
            "train_eyes": len(train_dataset), "val_eyes": len(val_dataset),
            "test_eyes": len(test_dataset),
            "params": f"{tp:,}", "trainable": f"{trp:,}",
        }
        logger.log_config(cfg, extra)
        if split_audit is not None:
            logger.log(f"Fixed split audit: {split_audit}")

        # ── Loss ──
        loss_cfg, loss_audit = resolve_training_loss(
            cfg["loss"], np.stack([r["label"] for r in train_records]))
        loss_audit["label_names"] = label_names
        logger.log(f"Training positive counts: {loss_audit['positive_counts']}")
        if loss_cfg.get("pos_weight") is not None:
            logger.log(f"Train-only positive weights: {loss_cfg['pos_weight']}")
        with open(os.path.join(cfg["logging"]["log_dir"], "loss_config.json"), "w", encoding="utf-8") as f:
            json.dump(loss_audit, f, ensure_ascii=False, indent=2)
        criterion = build_loss(loss_cfg).to(device)
        print(f"[Main] Loss: {criterion}")

        # ── Optimizer ──
        optimizer = build_optimizer(model, cfg["train"])
        logger.log(f"Optimizer groups: {[(g['group_name'], g['lr'], g['weight_decay']) for g in optimizer.param_groups]}")

        # ── Scheduler ──
        scheduler_info = build_scheduler(
            optimizer, cfg["train"], steps_per_epoch=len(train_loader),
        )
        scheduler = scheduler_info["scheduler"]
        step_per_batch = scheduler_info["step_per_batch"]

        warmup_epochs = cfg["train"].get("warmup_epochs", 0)
        if warmup_epochs > 0:
            print(f"[Main] Warmup: {warmup_epochs} epochs")

        # ── EMA ──
        model_ema = None
        ema_decay = cfg["train"].get("ema_decay", 0.0)
        if ema_decay > 0:
            model_ema = ModelEMA(model, decay=ema_decay, device=device,
                                 warmup=cfg["train"].get("ema_warmup", False))
            print(f"[Main] EMA: decay={ema_decay}")

        # ── AMP / grad clip ──
        use_amp = cfg["train"].get("use_amp", False)
        grad_clip_norm = cfg["train"].get("grad_clip_norm", None)
        try:
            from torch.amp import GradScaler
        except ImportError:
            from torch.cuda.amp import GradScaler
        scaler = GradScaler() if use_amp and device.type == "cuda" else None
        if use_amp:
            print("[Main] AMP enabled")
        if grad_clip_norm:
            print(f"[Main] GradClip: max_norm={grad_clip_norm}")

        # ── Phase 2: Contrastive params ──
        ctr_weight = contrastive_cfg.get("weight", 0.1) if contrastive_cfg.get("enabled", True) else 0.0
        ctr_temp = contrastive_cfg.get("temperature", 0.07)

        epochs = cfg["train"]["epochs"]
        best_val_f1 = -float("inf")
        selection_metric = cfg["train"].get("selection_metric", "micro_f1")
        best_epoch = 0
        stopping = EarlyStopping(cfg["train"].get("early_stopping_patience", 0))
        logger.log(f"Training control: max_epochs={epochs}, "
                   f"lr_decay_epochs={cfg['train'].get('lr_decay_epochs', epochs)}, "
                   f"early_stopping_patience={stopping.patience}, metric={selection_metric}; "
                   "strict improvement, ties do not reset patience")
        save_dir = cfg["logging"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"OURS: {architecture}")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            train_avg, train_cls, train_ctr = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch=epoch, total_epochs=epochs,
                ema=model_ema,
                grad_clip_norm=grad_clip_norm,
                scaler=scaler, use_amp=use_amp,
                scheduler=scheduler if step_per_batch else None,
                contrastive_weight=ctr_weight,
                contrastive_temperature=ctr_temp,
                regularization_config=cfg.get("regularization", {}),
            )

            # ── Validation with EMA ──
            if model_ema is not None:
                model_ema.apply_shadow(model)
            val_logits, val_labels = validate(model, val_loader, device)
            if model_ema is not None:
                model_ema.restore(model)

            val_metrics = compute_metrics(val_logits, val_labels, threshold=cfg["eval"]["threshold"])
            # Unweighted validation BCE diagnoses confidence without changing selection.
            val_bce = float(np.mean(np.logaddexp(0.0, val_logits.astype(np.float64))
                                    - val_labels * val_logits))

            if not step_per_batch:
                scheduler.step()

            logger.log_metrics(epoch, train_avg, val_metrics, prefix="Val")
            logger.log(f"Train components: cls={train_cls:.6f} ctr={train_ctr:.6f}; "
                       f"next_lr={[g['lr'] for g in optimizer.param_groups]}")
            # Preserve every epoch's validation class metrics for later diagnosis.
            with open(os.path.join(cfg["logging"]["log_dir"], "validation_metrics.jsonl"), "a", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "label_names": label_names, "metrics": val_metrics,
                           "validation_bce": val_bce}, f,
                          ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
                f.write("\n")
            print(f"  -> Train avg={train_avg:.4f} cls={train_cls:.4f} ctr={train_ctr:.4f}  "
                  f"Val microF1={val_metrics['micro_f1']:.4f} macroF1={val_metrics['macro_f1']:.4f} "
                  f"AUC={val_metrics['micro_auc']:.4f}/{val_metrics['macro_auc']:.4f}")

            if selection_metric not in val_metrics or not np.isfinite(val_metrics[selection_metric]):
                raise ValueError(f"Checkpoint metric {selection_metric} is missing or non-finite")
            if stopping.update(val_metrics[selection_metric], epoch):
                best_val_f1 = float(val_metrics[selection_metric])
                best_epoch = epoch
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "label_names": label_names,
                    "num_classes": num_classes,
                    "selection_metric": selection_metric,
                    "config": cfg,
                    "resolved_loss": loss_cfg,
                    "split_audit": split_audit,
                }
                if model_ema is not None:
                    checkpoint["ema_state_dict"] = model_ema.state_dict()
                torch.save(checkpoint, os.path.join(save_dir, cfg["logging"]["best_model_name"]))
                if cfg["logging"].get("save_best_val_predictions", False):
                    np.savez_compressed(
                        os.path.join(save_dir, "best_val_predictions.npz"),
                        logits=val_logits, labels=val_labels,
                        label_names=np.asarray(label_names, dtype=str),
                        epoch=np.asarray(epoch), threshold=np.asarray(cfg["eval"]["threshold"]))
                print(f"  >>> Best saved ({selection_metric}={best_val_f1:.4f})")

            if stopping.patience:
                logger.log(f"Early stopping: {stopping.bad_epochs}/{stopping.patience} "
                           f"epochs without improvement; best_epoch={best_epoch}")
            if stopping.should_stop:
                logger.log(f"Early stopping triggered at epoch {epoch}; "
                           f"using best checkpoint from epoch {best_epoch}")
                break

        training_summary = {
            "planned_epochs": epochs, "completed_epochs": epoch,
            "best_epoch": best_epoch, "selection_metric": selection_metric,
            "best_score": best_val_f1, "early_stopping_patience": stopping.patience,
            "epochs_without_improvement": stopping.bad_epochs,
            "stopped_early": epoch < epochs,
            "stop_reason": "early_stopping" if stopping.should_stop else "max_epochs",
            "lr_decay_epochs": cfg["train"].get("lr_decay_epochs", epochs),
        }
        with open(os.path.join(save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(training_summary, f, ensure_ascii=False, indent=2)

        logger.log(f"\nDone! Best {selection_metric}={best_val_f1:.4f} at epoch {best_epoch}")

        # Final eval with EMA
        print("\n" + "=" * 60)
        print("Final Evaluation (Best Model)")
        print("=" * 60)
        best_ckpt = torch.load(os.path.join(save_dir, cfg["logging"]["best_model_name"]), map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        if model_ema is not None and "ema_state_dict" in best_ckpt:
            model_ema.load_state_dict(best_ckpt["ema_state_dict"])
            model_ema.apply_shadow(model)
        final_metrics = evaluate(model, val_loader, device, label_names, threshold=cfg["eval"]["threshold"], tta=cfg["eval"].get("tta", False))
        logger.log("\nFinal:\n" + format_metrics(final_metrics, label_names))
        logger.close()

    # ================================================================
    # Mode: Eval
    # ================================================================
    elif args.mode == "audit_ema":
        if not args.audit_output_dir:
            raise ValueError("audit_ema requires --audit-output-dir")
        if cfg["eval"].get("tta", False):
            raise ValueError("audit_ema currently requires eval.tta=false")
        checkpoint_path = args.checkpoint or os.path.join(
            cfg["logging"]["save_dir"], cfg["logging"]["best_model_name"])
        ckpt = torch.load(checkpoint_path, map_location=device)
        validate_checkpoint_metadata(ckpt, label_names, split_audit, model_cfg)
        if "ema_state_dict" not in ckpt:
            raise ValueError("Checkpoint has no EMA weights to compare")
        output_dir = args.audit_output_dir
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, "ema_audit.json")
        if os.path.exists(result_path):
            raise FileExistsError(f"Audit already exists; choose a new output directory: {output_dir}")
        result = {"seed": cfg["seed"], "checkpoint": os.path.abspath(checkpoint_path),
                  "epoch": ckpt["epoch"], "split": "validation", "tta": False,
                  "threshold": cfg["eval"]["threshold"], "label_names": label_names,
                  "split_audit": split_audit, "weights": {}}
        model.load_state_dict(ckpt["model_state_dict"])
        for variant in ("raw", "ema"):
            if variant == "ema":
                ema_audit = ModelEMA(model, decay=ckpt["ema_state_dict"].get("decay", .999))
                ema_audit.load_state_dict(ckpt["ema_state_dict"])
                ema_audit.apply_shadow(model)
            logits, labels = validate(model, val_loader, device)
            metrics = compute_metrics(logits, labels, threshold=cfg["eval"]["threshold"])
            metrics["validation_bce"] = float(np.mean(
                np.logaddexp(0., logits.astype(np.float64)) - labels * logits))
            result["weights"][variant] = metrics
            np.savez_compressed(os.path.join(output_dir, f"{variant}_val_predictions.npz"),
                                logits=logits, labels=labels, label_names=np.asarray(label_names, dtype=str))
            print(f"\nValidation audit: {variant}, checkpoint epoch {ckpt['epoch']}")
            print(format_metrics(metrics, label_names))
            print(f"Validation BCE: {metrics['validation_bce']:.6f}")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2,
                      default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        print(f"Audit saved to: {result_path}")

    elif args.mode == "eval":
        if args.checkpoint is None:
            args.checkpoint = os.path.join(cfg["logging"]["save_dir"], cfg["logging"]["best_model_name"])
        print(f"[Main] Loading: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        validate_checkpoint_metadata(ckpt, label_names, split_audit, model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            ema_ld = ModelEMA(model, decay=ckpt["ema_state_dict"].get("decay", 0.999))
            ema_ld.load_state_dict(ckpt["ema_state_dict"])
            ema_ld.apply_shadow(model)
        evaluate(model, val_loader, device, label_names, threshold=cfg["eval"]["threshold"], tta=cfg["eval"].get("tta", False))

    # ================================================================
    # Mode: Test
    # ================================================================
    elif args.mode == "test":
        if args.checkpoint is None:
            args.checkpoint = os.path.join(cfg["logging"]["save_dir"], cfg["logging"]["best_model_name"])
        print(f"[Main] Loading: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        validate_checkpoint_metadata(ckpt, label_names, split_audit, model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            ema_ld = ModelEMA(model, decay=ckpt["ema_state_dict"].get("decay", 0.999))
            ema_ld.load_state_dict(ckpt["ema_state_dict"])
            ema_ld.apply_shadow(model)
        print(f"[Main] Testing on {len(test_dataset)} eyes...")
        test_metrics = evaluate(model, test_loader, device, label_names, threshold=cfg["eval"]["threshold"], tta=cfg["eval"].get("tta", False))
        test_path = os.path.join(cfg["logging"]["save_dir"], "test_results.txt")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(f"Test Results ({args.checkpoint})\n")
            f.write(f"Test eyes: {len(test_dataset)}\n")
            f.write(format_metrics(test_metrics, label_names))
        print(f"[Main] Results saved to: {test_path}")


if __name__ == "__main__":
    main()
