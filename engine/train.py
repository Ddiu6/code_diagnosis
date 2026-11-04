"""
Training engine for OURS (Phase 2).
Supports: combined loss (classification + contrastive), EMA, warmup.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Any
import numpy as np
import math
from contextlib import nullcontext

from engine.ema import ModelEMA
from models.fusion import cross_view_contrastive_loss
from models.evidence import sample_view_subset, coverage_consistency

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    ema: Optional[ModelEMA] = None,
    grad_clip_norm: Optional[float] = None,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
    scheduler: Optional[Any] = None,
    # Phase 2: contrastive loss
    contrastive_weight: float = 0.1,
    contrastive_temperature: float = 0.07,
    regularization_config: Optional[dict] = None,
) -> float:
    """
    Run one training epoch.

    Loss = L_cls + λ * L_contrastive
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_ctr_loss = 0.0
    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("Training loader is empty")
    reg = regularization_config or {}

    for batch_idx, (images, view_mask, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        view_mask = view_mask.to(device, non_blocking=True).bool()
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and scaler is not None
        context = torch.autocast(device_type=device.type) if amp_enabled else nullcontext()
        with context:
            output = model(images, view_mask, return_full=True)
            cls_loss = criterion(output["diag_logits"], labels)
            diag = model.diagnosis_model
            ctr_loss = cls_loss.new_zeros(())
            if contrastive_weight > 0 and diag.use_contrastive and view_mask.sum() > 1:
                projected = diag.get_contrastive_features(output["view_features"], view_mask)
                padded = projected.new_zeros(*view_mask.shape, projected.shape[-1])
                padded[view_mask] = projected
                ctr_loss = cross_view_contrastive_loss(
                    padded, view_mask, temperature=contrastive_temperature,
                    labels=labels if reg.get("exclude_same_label_negatives", True) else None,
                )
            aux_loss = cls_loss.new_zeros(())
            consistency_loss = cls_loss.new_zeros(())
            if "evidence_logits" in output:
                aux_loss = criterion(output["evidence_logits"], labels)
                if reg.get("consistency_weight", 0) > 0:
                    subset = sample_view_subset(view_mask, reg.get("view_drop_probability", 0.25))
                    student = diag.forward_features(output["view_features"], subset)
                    consistency_loss = coverage_consistency(
                        student["diag_logits"], output, labels, subset, view_mask,
                        min_coverage=reg.get("min_coverage", 0.7),
                        confidence=reg.get("teacher_confidence", 0.7),
                    )
            ramp = min(1.0, max(0.0, (epoch - 1) / max(1, reg.get("warmup_epochs", 5))))
            loss = (cls_loss + contrastive_weight * ctr_loss
                    + reg.get("evidence_aux_weight", 0.1) * aux_loss
                    + ramp * reg.get("consistency_weight", 0.0) * consistency_loss)

        if amp_enabled:
            previous_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            did_step = scaler.get_scale() >= previous_scale
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            did_step = True

        # ── EMA update ──
        if ema is not None and did_step:
            ema.update(model)

        # ── Per-batch scheduler step ──
        if scheduler is not None and did_step:
            scheduler.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_ctr_loss += ctr_loss.item()

        if (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.4f} (cls={cls_loss.item():.4f} ctr={ctr_loss.item():.4f} "
                f"aux={aux_loss.item():.4f} consistency={consistency_loss.item():.4f}) "
                f"LR: {current_lr:.2e}"
            )

    avg_loss = total_loss / num_batches
    avg_cls = total_cls_loss / num_batches
    avg_ctr = total_ctr_loss / num_batches
    return avg_loss, avg_cls, avg_ctr


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate: collect all logits and labels."""
    model.eval()
    all_logits = []
    all_labels = []

    for images, view_mask, labels in dataloader:
        images = images.to(device, non_blocking=True)
        view_mask = view_mask.to(device, non_blocking=True)

        logits, _ = model(images, view_mask)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> dict:
    """Use one update unit throughout; warmup and decay are optimizer steps.

    A lambda per group preserves backbone/head LR ratios, including the floor.
    """
    train_cfg = cfg.get("train", cfg)
    sched_name = train_cfg.get("scheduler", "cosine_epoch")
    epochs = train_cfg.get("epochs", 100)
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    warmup_start_lr = train_cfg.get("warmup_start_lr", 1e-6)
    base_lr = train_cfg.get("learning_rate", 1e-4)
    min_lr = train_cfg.get("min_lr", 1e-6)

    if sched_name not in ("cosine", "cosine_epoch", "step"):
        raise ValueError(f"Unknown scheduler: {sched_name}")
    if steps_per_epoch < 1 or epochs < 1 or not 0 <= warmup_epochs < epochs:
        raise ValueError("Require positive steps/epochs and 0 <= warmup_epochs < epochs")
    if base_lr <= 0 or not 0 <= min_lr <= base_lr or not 0 <= warmup_start_lr <= base_lr:
        raise ValueError("Invalid learning-rate bounds")
    # Decouple the cosine horizon from the maximum training budget. Once the
    # horizon is reached, clamp to min_lr instead of restarting the cosine.
    decay_epochs = train_cfg.get("lr_decay_epochs", epochs)
    if not warmup_epochs < decay_epochs <= epochs:
        raise ValueError("Require warmup_epochs < lr_decay_epochs <= epochs")
    if sched_name == "step" and "lr_decay_epochs" in train_cfg:
        raise ValueError("lr_decay_epochs is only supported for cosine schedulers")
    total_steps = decay_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    start_ratio, floor_ratio = warmup_start_lr / base_lr, min_lr / base_lr

    def lr_factor(step):
        if step < warmup_steps:
            return start_ratio + (1 - start_ratio) * step / warmup_steps
        elapsed = step - warmup_steps
        if sched_name == "step":
            interval = train_cfg.get("step_size", 30) * steps_per_epoch
            if interval <= 0:
                raise ValueError("step_size must be positive")
            return max(floor_ratio, train_cfg.get("step_gamma", 0.1) ** (elapsed // interval))
        progress = min(1.0, elapsed / max(1, total_steps - warmup_steps))
        return floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    return {"scheduler": scheduler, "step_per_batch": True}


def build_optimizer(model, cfg):
    """Lower LR for pretrained convolution weights; no decay on bias/norm."""
    if cfg.get("optimizer", "adamw").lower() != "adamw":
        raise ValueError("Only adamw is supported")
    multiplier = cfg.get("backbone_lr_multiplier", 1.0)
    if multiplier <= 0:
        raise ValueError("backbone_lr_multiplier must be positive")
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith("diagnosis_model.backbone.backbone.")
        decay = not cfg.get("exclude_norm_bias_decay", False) or parameter.ndim > 1
        key = (backbone, decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([
        {"params": params,
         "lr": cfg["learning_rate"] * (multiplier if backbone else 1.0),
         "weight_decay": cfg["weight_decay"] if decay else 0.0,
         "group_name": ("backbone" if backbone else "head") + ("_decay" if decay else "_no_decay")}
        for (backbone, decay), params in groups.items()
    ])
