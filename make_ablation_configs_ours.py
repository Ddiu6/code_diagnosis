"""生成以 OURS(06) 为唯一锚点的 4 组干净消融 config（无对比损失 + 单变量删减）。

背景：原 02/03/04/05 是在"含对比损失"的 01 基础上做的删减，与方法章（第四章）
所介绍的 OURS（=06，仅 BCE、无辅助分支）不一致，存在双重变量。
本脚本补跑 4 组"无对比损失 + 删组件"的干净版本，使整个消融统一以 OURS 为锚。

用法（在 code1021 目录下）:
    python make_ablation_configs_ours.py
    python main_seed.py --config configs/ablation/ablation_XX_name.yaml

结果目录: ../results1021/ablation_XX_name/seed_<n>/...
"""
import copy

import yaml

# 复用现有 BASE 与 apply_overrides
from make_ablation_configs import BASE, apply_overrides, OUT_DIR, RESULTS_ROOT

# (目录名, 说明, model 覆盖, 是否关对比损失=True 表示无对比损失)
OURS_ABLATIONS = [
    ("08_ours_no_cross_view",    "OURS 去跨视图交互",        {"use_cross_view": False},      True),
    ("09_ours_no_interaction_gate", "OURS 去交互门控(标量)", {"use_cross_view_gate": False}, True),
    ("10_ours_no_channel_gate",  "OURS 去通道门控",          {"use_view_gate": False},       True),
    ("11_ours_single_head",      "OURS 多头→单头",           {"fusion_heads": 1},            True),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, desc, model_ov, ctr_off in OURS_ABLATIONS:
        cfg = apply_overrides(copy.deepcopy(BASE), model_ov, ctr_off)
        cfg["logging"]["log_dir"] = f"{RESULTS_ROOT}/ablation_{name}/logs"
        cfg["logging"]["save_dir"] = f"{RESULTS_ROOT}/ablation_{name}/checkpoints"

        out_path = OUT_DIR / f"ablation_{name}.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        m = cfg["model"]
        summary.append(
            f"ablation_{name:<32} | 交互={m['use_cross_view']} 交互门={m['use_cross_view_gate']} "
            f"通道门={m['use_view_gate']} 头数={m['fusion_heads']} 对比={cfg['contrastive']['enabled']} | {desc}"
        )

    print(f"已生成 {len(OURS_ABLATIONS)} 组干净 config 到: {OUT_DIR}\n")
    print("=" * 100)
    for line in summary:
        print(line)
    print("=" * 100)
    print("\n运行方式（在 code1021 目录下，多种子 + 早停）:")
    print("  python main_seed.py --config configs/ablation/ablation_XX_name.yaml")


if __name__ == "__main__":
    main()
