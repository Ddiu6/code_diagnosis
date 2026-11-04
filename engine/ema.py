"""
Exponential Moving Average (EMA) for model weights.

EMA 对模型参数做指数移动平均，推理时使用 EMA 权重替代原始权重。
在大量图像分类任务中，EMA 能带来 0.5~2% 的稳定提升。

Reference:
    - Tarvainen & Valpola, "Mean teachers are better role models", NeurIPS 2017.
    - Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018.

Usage:
    ema = ModelEMA(model, decay=0.999)
    for epoch in range(epochs):
        train_one_epoch(...)
        ema.update(model)
    ema.apply_shadow(model)   # 推理前应用 EMA 权重
    evaluate(model, ...)
    ema.restore(model)         # 恢复原始权重
"""

from __future__ import annotations

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class ModelEMA:
    """
    Exponential Moving Average for model parameters.

    Args:
        model:   nn.Module to track
        decay:   EMA decay rate (default: 0.999)
                 越接近 1，EMA 越平滑但更新越慢
                 常用值: 0.999 ~ 0.9999
        device:  device for shadow parameters, None = same as model
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        warmup: bool = False,
    ):
        self.model = model
        self.decay = decay
        self.device = device
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.warmup = warmup
        self.num_updates = 0

        # Deep copy of all trainable parameters
        self.shadow = {}
        self.backup = {}

        self._register()

    def _register(self):
        """Register shadow parameters (deep copy of model state_dict)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: Optional[nn.Module] = None):
        """
        Update shadow parameters: shadow = decay * shadow + (1-decay) * model.

        Args:
            model: if None, use self.model
        """
        if model is None:
            model = self.model
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates)) if self.warmup else self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone().detach()

    @torch.no_grad()
    def apply_shadow(self, model: Optional[nn.Module] = None):
        """
        Replace model parameters with EMA shadow (for evaluation).

        Args:
            model: if None, use self.model
        """
        if model is None:
            model = self.model

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: Optional[nn.Module] = None):
        """
        Restore original model parameters from backup.

        Args:
            model: if None, use self.model
        """
        if model is None:
            model = self.model

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        """Return shadow parameters for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow,
                "warmup": self.warmup, "num_updates": self.num_updates}

    def load_state_dict(self, state_dict: dict):
        """Load shadow parameters from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
        self.warmup = state_dict.get("warmup", False)
        self.num_updates = state_dict.get("num_updates", 0)
