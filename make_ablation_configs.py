"""生成减法消融实验 config（多种子 + 早停）。

用法:
    python make_ablation_configs.py

输出:
    configs/ablation/ablation_XX_name.yaml  共 7 组

每组结果目录（配合 main_seed.py 跑，每个 seed 一个子目录）:
    ../results1021/ablation_XX_name/seed_<n>/{checkpoints,logs,config.yaml,run.log}
    聚合: ../results1021/ablation_XX_name/seed_summary.json + seed_metrics.csv
"""

import copy
import os
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "configs" / "ablation"
RESULTS_ROOT = "../results1021"          # 相对 code1021，落到 exp1021/results1021

# ---------------------------------------------------------------------------
# Base: 完整模型（对齐 patient_A_es40，含患者级划分校验 + 早停 + 选择指标）
# ---------------------------------------------------------------------------
BASE = {
    "data": {
        "excel_path": "/mnt/data/zhouwenrui/projects/datasets/0831.xlsx",
        "image_root": "/mnt/data/zhouwenrui/projects/image_root_5",
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "image_size": 224,
        "num_workers": 4,
        "crop": {"top": 0.15, "bottom": 0.15, "left": 0.2, "right": 0.2, "show_sample": False},
        "augmentation": {
            "rotation_degrees": 10,
            "rotation_p": 0.1,
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8], "p": 0.5},
            "speckle_noise": {"enabled": False, "sigma": 0.05, "p": 0.5},
        },
        "roi": {"enabled": False,
                "crop": {"left": 0.2, "right": 0.3, "top": 0.25, "bottom": 0.25},
                "darken": 0.22},
        "us_augment": {"enabled": True, "speckle_p": 0.3, "echo_p": 0.25, "artifact_p": 0.2, "max_ops": 2},
        "require_patient_split": True,
    },
    "model": {
        "backbone": "convnext_tiny",
        "pretrained": True,
        "embed_dim": 256,
        "fusion_heads": 4,
        "fusion_hidden_dim": 128,
        "attn_dropout": 0.2,
        "dropout": 0.2,
        "use_cross_view": True,
        "cross_view_layers": 1,
        "cross_view_heads": 4,
        "use_view_gate": True,
        "use_cross_view_gate": True,
        "use_label_gcn": False,
        "label_gcn_threshold": 0.05,
    },
    "contrastive": {"enabled": True, "weight": 0.03, "temperature": 0.07, "proj_dim": 128},
    "train": {
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "scheduler": "cosine_epoch",
        "min_lr": 1.0e-6,
        "warmup_epochs": 5,
        "warmup_start_lr": 1.0e-6,
        "ema_decay": 0.999,
        "grad_clip_norm": 1.0,
        "use_amp": False,
        "early_stopping_patience": 40,
        "selection_metric": "micro_f1",
    },
    "loss": {"name": "bce"},
    "eval": {"threshold": 0.5, "tta": False},
    "logging": {
        "best_model_name": "best_model.pth",
        "split_file": "/mnt/data/zhouwenrui/projects/eyeClassfier_exp/split_files/split_p.json",
        "save_best_val_predictions": True,
    },
    "seed": 42,
}

# ---------------------------------------------------------------------------
# 减法消融组定义: (目录名, 说明, model 段覆盖, 是否关对比损失)
#   model 覆盖直接写进 model dict；contrastive_off=True 关掉对比损失
# ---------------------------------------------------------------------------
ABLATIONS = [
    ("01_full",            "完整模型（OURS，全模块开）", {}, False),
    ("02_no_cross_view",   "删跨视图交互 CrossViewAttention", {"use_cross_view": False}, False),
    ("03_no_cv_gate",      "删交互标量门控（保留交互，直接采纳）", {"use_cross_view_gate": False}, False),
    ("04_no_view_gate",    "删视图门控 ViewGate", {"use_view_gate": False}, False),
    ("05_single_head",     "多头→单头注意力（=Attention_MIL）", {"fusion_heads": 1}, False),
    ("06_no_contrastive",  "删跨视图对比损失", {}, True),
    ("07_minimal",         "最简基线（删交互+删门控+单头+无对比）",
     {"use_cross_view": False, "use_view_gate": False, "fusion_heads": 1}, True),
]


def apply_overrides(cfg, model_overrides, contrastive_off):
    if model_overrides:
        cfg["model"].update(model_overrides)
    if contrastive_off:
        cfg["contrastive"]["enabled"] = False
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for name, desc, model_ov, ctr_off in ABLATIONS:
        cfg = copy.deepcopy(BASE)
        apply_overrides(cfg, model_ov, ctr_off)
        cfg["logging"]["log_dir"] = f"{RESULTS_ROOT}/ablation_{name}/logs"
        cfg["logging"]["save_dir"] = f"{RESULTS_ROOT}/ablation_{name}/checkpoints"

        out_path = OUT_DIR / f"ablation_{name}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        m = cfg["model"]
        summary.append(
            f"ablation_{name:<24} | 交互={m['use_cross_view']} 交互门={m['use_cross_view_gate']} "
            f"视图门={m['use_view_gate']} 头数={m['fusion_heads']} 对比={cfg['contrastive']['enabled']} | {desc}"
        )

    print(f"已生成 {len(ABLATIONS)} 组 config 到: {OUT_DIR}\n")
    print("=" * 100)
    for line in summary:
        print(line)
    print("=" * 100)
    print("\n运行方式（在 code1021 目录下，多种子 + 早停）:")
    print("  python main_seed.py --config configs/ablation/ablation_XX_name.yaml")
    print("\n结果目录: ../results1021/ablation_XX_name/seed_<n>/...")


if __name__ == "__main__":
    main()
