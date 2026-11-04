#  眼B超多视图诊断

## 目录结构

```
code1021/
├── main.py                 # 入口：train / eval / test / audit_ema
├── main_seed.py            # 多种子训练脚本
├── make_ablation_configs.py  # 生成减法消融 config
├── configs/
│   ├── config.yaml         # 主配置
│   └── ablation/           # 7 组减法消融 config（ablation_01_full ~ ablation_07_minimal）
├── models/
│   ├── model.py            # 顶层 CLSTransformerMIL（对外统一接口）
│   ├── diagnosis.py        # DiagnosisModel 组装
│   ├── backbone.py         # ConvNeXt 骨干 + 投影层
│   ├── fusion.py           # 多视图融合：跨视图交互/视图门控/多头注意力池化/对比损失
│   ├── losses.py           # BCE / ASL / Focal 损失
│   ├── label_gcn.py        # 标签共现 GCN
│   └── evidence.py         # CoverageEvidenceHead
├── engine/
│   ├── train.py            # 训练/验证循环、优化器、调度器
│   ├── evaluate.py         # 评估 + TTA
│   ├── ema.py              # 指数滑动平均
│   └── early_stopping.py   # 早停
├── utils/
│   ├── metrics.py          # micro/macro F1、AUC 等指标
│   ├── loss_config.py      # 训练损失解析
│   ├── logger.py           # 日志
│   └── protocol.py         # 患者级划分校验、checkpoint 元数据校验
└── datasets/
    ├── eye_mil_dataset.py  # 眼级 MIL 数据集、划分、变换（含描述/诊断字段读取）
    └── roi_usaug.py        # ROI 裁边 + 超声干扰仿真增强
```

## 模型数据流

```
images [B, N, 3, H, W]
  → ImageBackbone (ConvNeXt-Tiny + proj)      → view_feats [B, N, D]
  → CrossViewAttention    → interacted [B, N, D]
  → ViewGate 
  → MultiHeadAttentionPooling      → pooled [B, D]
  → DiagnosisHead              → logits [B, C]
  + ContrastiveProjectionHead (对比损失, 可选)
```

## 「删模块」的抓手

### 用 config 开关做消融（不改代码，改 yaml）

| config 开关 | 对应模块 | 默认 | 说明 |
|---|---|---|---|
| `model.use_cross_view` | CrossViewAttention | true | 关掉 = 无跨视图交互 |
| `model.use_cross_view_gate` | 交互标量门控 | true | 关掉 = 直接采纳交互结果 |
| `model.use_view_gate` | ViewGate 逐维度门控 | true | 关掉 = 无视图门控 |
| `model.fusion_heads` | 多头注意力头数 | 4 | 设 1 = 单头 GatedAttention（即 Attention_MIL） |
| `contrastive.enabled` | 对比损失 | true | 关掉 = 纯分类损失 |
| `model.use_label_gcn` | 标签 GCN | false | 默认已关 |
| `model.evidence.enabled` | CoverageEvidenceHead | false | 默认已关 |

## 减法消融实验

`make_ablation_configs.py` 生成 7 组 config（`configs/ablation/ablation_XX_name.yaml`），
每次只删一个模块，最后加一个「最简基线」，用于定位哪个模块真正有用。

| config | 删了什么 | 交互 | 交互门 | 视图门 | 头数 | 对比 |
|---|---|---|---|---|---|---|
| ablation_01_full | （完整模型） | ✓ | ✓ | ✓ | 4 | ✓ |
| ablation_02_no_cross_view | 跨视图交互 | ✗ | - | ✓ | 4 | ✓ |
| ablation_03_no_cv_gate | 交互标量门控 | ✓ | ✗ | ✓ | 4 | ✓ |
| ablation_04_no_view_gate | 视图门控 | ✓ | ✓ | ✗ | 4 | ✓ |
| ablation_05_single_head | 多头→单头 | ✓ | ✓ | ✓ | 1 | ✓ |
| ablation_06_no_contrastive | 对比损失 | ✓ | ✓ | ✓ | 4 | ✗ |
| ablation_07_minimal | 全部删（最简基线） | ✗ | - | ✗ | 1 | ✗ |

## 运行环境

- 训练在服务器（CUDA），本地 `E:\anacond\python.exe`（torch 2.2.2 CPU）可用于 import 验证。
- 数据路径在 `configs/config.yaml` 的 `data.excel_path` / `data.image_root`（当前为服务器路径 `/mnt/...`）。
