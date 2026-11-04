"""ConvNeXt 多规格 Backbone + 投影层。"""

from __future__ import annotations
import warnings
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

# ConvNeXt 各变体的特征维度（after avgpool）
_CONVNEXT_DIMS = {
    "convnext_tiny":   768,
    "convnext_small":  768,
    "convnext_base":  1024,
    "convnext_large": 1536,
    "convnext_v2_tiny":   768,
    "convnext_v2_base":  1024,
    "convnext_v2_large": 1536,
}


def build_convnext(name: str = "convnext_tiny", pretrained: bool = True) -> nn.Module:
    """
    构建指定规格的 ConvNeXt backbone。

    Args:
        name: "convnext_tiny" | "convnext_small" | "convnext_base" | "convnext_large"
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        ConvNeXt 模型实例
    """
    model_builders = {
        "convnext_tiny":  models.convnext_tiny,
        "convnext_small": models.convnext_small,
        "convnext_base":  models.convnext_base,
        "convnext_large": models.convnext_large,
    }

    # V2 weights (optional, torchvision >= 0.17)
    v2_weights = {}
    try:
        v2_weights["convnext_tiny"]  = models.ConvNeXt_Tiny_Weights.DEFAULT
    except AttributeError:
        pass
    try:
        v2_weights["convnext_small"] = models.ConvNeXt_Small_Weights.DEFAULT
    except AttributeError:
        pass
    try:
        v2_weights["convnext_base"]  = models.ConvNeXt_Base_Weights.DEFAULT
    except AttributeError:
        pass
    try:
        v2_weights["convnext_large"] = models.ConvNeXt_Large_Weights.DEFAULT
    except AttributeError:
        pass

    builder = model_builders.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown backbone: {name}. "
            f"Available: {list(model_builders.keys())}"
        )

    if pretrained:
        weights = v2_weights.get(name)
        if weights is not None:
            return builder(weights=weights)
        else:
            # Fallback: torchvision < 0.15 不支持 Weights enum
            try:
                return builder(weights="DEFAULT")
            except (TypeError, ValueError):
                warnings.warn(
                    f"无法加载 {name} 预训练权重（torchvision 版本过旧），使用随机初始化。"
                )

    return builder(weights=None)


class ImageBackbone(nn.Module):
    """逐视图提取特征 → (B, N, embed_dim)。"""

    def __init__(
        self,
        embed_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        backbone_name: str = "convnext_tiny",
    ):
        super().__init__()
        convnext = build_convnext(name=backbone_name, pretrained=pretrained)
        self.backbone = nn.Sequential(convnext.features, convnext.avgpool)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feature_dim = convnext.classifier[2].in_features
        print(f"[Backbone] {backbone_name}: feature_dim={feature_dim}, "
              f"project to embed_dim={embed_dim}")

        self.proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, images: torch.Tensor, view_mask=None) -> torch.Tensor:
        B, N, C, H, W = images.shape
        flat = images.reshape(B * N, C, H, W)
        if view_mask is None:
            view_mask = torch.ones(B, N, dtype=torch.bool, device=images.device)
        if view_mask.shape != (B, N) or not view_mask.bool().any(-1).all():
            raise ValueError("Invalid view mask or empty eye bag")
        valid = view_mask.bool().reshape(-1)
        features = self.backbone(flat[valid]).flatten(1)
        features = self.proj(features)
        padded = features.new_zeros(B * N, features.shape[-1])
        padded[valid] = features
        return padded.view(B, N, -1)
