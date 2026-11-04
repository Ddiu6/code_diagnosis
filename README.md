# View-Gated Multi-Head Attention MIL for Multi-View Ocular B-Scan Diagnosis

Reference implementation of the framework and the ablation configurations reported in

> **MI-OBUD: A Multi-View Ocular B-Scan Ultrasound Dataset and View-Gated Multi-Head Attention MIL
> for Eye-Level Disease Diagnosis** (submitted to *Biomedical Signal Processing and Control*)

One eye is treated as an unordered bag of a variable number of B-scan views, and the model predicts
a six-dimensional eye-level multi-label vector. A shared encoder produces per-view features; a
view-level scalar gate decides how much cross-view context each view accepts; a channel gate
filters each view dimension by dimension; multi-head attention pooling aggregates evidence across
views; and a residual head outputs non-mutually-exclusive label probabilities.

## Repository layout

```
code_diagnosis/
├── main.py                    # entry point: train / eval / test / audit_ema
├── main_seed.py               # multi-seed training driver
├── make_ablation_configs.py   # generates the subtractive ablation configs
├── make_ablation_configs_ours.py
├── per_class_eval.py          # per-class metrics and confusion matrix
├── configs/
│   ├── config.yaml            # main configuration
│   └── ablation/              # ablation configs (see table below)
├── models/
│   ├── model.py               # CLSTransformerMIL, top-level interface
│   ├── diagnosis.py           # DiagnosisModel assembly
│   ├── backbone.py            # ConvNeXt backbone + projection layer
│   ├── fusion.py              # cross-view interaction / gates / multi-head pooling / contrastive head
│   ├── losses.py              # BCE / ASL / focal losses
│   ├── label_gcn.py           # optional label-co-occurrence GCN head
│   └── evidence.py            # optional coverage-evidence head
├── engine/
│   ├── train.py               # training and validation loop, optimizer, scheduler
│   ├── evaluate.py            # evaluation, optional TTA
│   ├── ema.py                 # exponential moving average
│   └── early_stopping.py
├── utils/
│   ├── metrics.py             # micro/macro F1, AUC, ExactMatch, Hamming loss
│   ├── loss_config.py
│   ├── logger.py
│   └── protocol.py            # patient-level split and checkpoint metadata checks
└── datasets/
    ├── eye_mil_dataset.py     # eye-level MIL dataset, splits, transforms
    └── roi_usaug.py           # ROI cropping and ultrasound-interference simulation augmentation
```

## Data flow

```
images [B, N, 3, H, W]
  → ImageBackbone (ConvNeXt-Tiny + projection)   → view_feats [B, N, D]
  → CrossViewAttention                           → interacted [B, N, D]
  → ViewGate (scalar gate per view)
  → ChannelGate (per-view, per-channel gate)
  → MultiHeadAttentionPooling                    → pooled [B, D]
  → DiagnosisHead (residual MLP)                 → logits [B, C]
  + ContrastiveProjectionHead (optional, disabled in the final model)
```

`N` varies across eyes. Padded views are excluded by a view-validity mask in the cross-view
attention, in the pooling step and in the loss.

## Configuration switches

Every architectural component can be removed through `configs/config.yaml` without touching the
code, which is how the ablation study is reproduced.

| Config key | Component | Default | Effect when switched off |
|---|---|---|---|
| `model.use_cross_view` | Cross-view interaction | `true` | no cross-view context at all |
| `model.use_cross_view_gate` | Interaction gate (scalar, per view) | `true` | interaction adopted unconditionally |
| `model.use_view_gate` | Channel gating (per view, per channel) | `true` | no channel-level filtering |
| `model.fusion_heads` | Number of attention heads | `4` | `1` gives single-head gated attention (Attention-MIL) |
| `contrastive.enabled` | Cross-view contrastive loss | **`false`** | the final model in the paper: classification loss only |
| `model.use_label_gcn` | Label-co-occurrence GCN head | `false` | not used in the paper |
| `model.evidence.enabled` | Coverage-evidence head | `false` | not used in the paper |

> **Note on the default.** `contrastive.enabled` defaults to `false`, matching the final model
> reported in the paper. Setting it to `true` reproduces the "+Contrastive Loss" variant, which the
> ablation shows to be worse at this data scale.

## Reproducing the ablation table

| Paper variant | Config file |
|---|---|
| Ours (final model) | `configs/ablation/ablation_06_no_contrastive.yaml` |
| + Contrastive Loss | `configs/ablation/ablation_01_full.yaml` |
| w/o Cross-View Interaction | `configs/ablation/ablation_08_ours_no_cross_view.yaml` |
| w/o Interaction Gate | `configs/ablation/ablation_09_ours_no_interaction_gate.yaml` |
| w/o Channel Gating | `configs/ablation/ablation_10_ours_no_channel_gate.yaml` |
| w/ Single-Head Pooling | `configs/ablation/ablation_11_ours_single_head.yaml` |
| Minimal model (lower bound) | `configs/ablation/ablation_07_minimal.yaml` |

`ablation_02`–`ablation_05` belong to the earlier full-model series (contrastive loss enabled) and
are kept for completeness; they are not part of the final table.

## Environment

Tested with Python 3.10, PyTorch 2.2.2 + CUDA 11.8 on a single GPU; CPU is sufficient for import
checks.

```bash
pip install -r requirements.txt
```

## Data

The MI-OBUD images and the label table are **not** distributed with this repository, because the
B-scan images contain sensitive clinical information that cannot be released under the current
ethics approval. They are available from the first author upon reasonable request by e-mail,
subject to approval by the institutional ethics committee. See the data and code availability
statement in the paper.

To run the code on other data, set `data.excel_path` and `data.image_root` in
`configs/config.yaml`. The label table expects one row per eye with patient identifier, eye
identifier and six binary labels. Paths in the released configs point to an internal cluster and
must be replaced.

## Usage

```bash
# training
python main.py --mode train --config configs/config.yaml --device cuda

# evaluation with a saved checkpoint
python main.py --mode eval --config configs/config.yaml --checkpoint <path/to/best_model.pth>

# test set only
python main.py --mode test --config configs/config.yaml --checkpoint <path/to/best_model.pth>

# multi-seed runs (seeds 42, 123, 2026, 3407, 9001)
python main_seed.py --config configs/config.yaml --device cuda
```

An ablation run uses the same entry point with the corresponding config, for example:

```bash
python main.py --mode train --config configs/ablation/ablation_10_ours_no_channel_gate.yaml
```

## License

Released under the MIT License; see [LICENSE](LICENSE).

## Contact

Wenrui Zhou — celeste_zhou@foxmail.com
