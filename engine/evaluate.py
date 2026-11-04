"""
Evaluation engine for OURS: CLS Token Transformer MIL.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np

from utils.metrics import compute_metrics, format_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: List[str],
    threshold: float = 0.5,
    verbose: bool = True,
    tta: bool = False,
) -> Dict[str, float]:
    """Evaluate model on a dataloader and compute all metrics.

    Args:
        tta: 若为 True，对每张视图额外做水平翻转，平均两次前向的 logits
             （Test-Time Augmentation，稳定预测，通常提升 AUC/F1）。
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, view_mask, labels in dataloader:
        images = images.to(device, non_blocking=True)
        view_mask = view_mask.to(device, non_blocking=True)

        logits, _ = model(images, view_mask)

        if tta:
            # 水平翻转 TTA：原图 + 翻转，平均 logits
            images_flip = torch.flip(images, dims=[-1])
            logits_flip, _ = model(images_flip, view_mask)
            logits = (logits + logits_flip) / 2.0

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_logits, all_labels, threshold=threshold)

    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(format_metrics(metrics, label_names))
        print("=" * 50 + "\n")

    return metrics
