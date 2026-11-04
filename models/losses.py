"""
Asymmetric Loss (ASL) for Multi-Label Classification.

Reference:
    Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    https://arxiv.org/abs/2009.14119

ASL 对正负样本使用不同的聚焦参数:
    - 正样本: (1-p)^gamma_plus  * log(p)
    - 负样本: pm^gamma_minus * log(1-pm),  pm = max(p-m, 0)

其中 gamma_minus > gamma_plus 意味着对负样本施加更强的难例聚焦,
同时 probability shift m 会丢弃极容易的负样本（概率低于 m 的不参与训练）。

在多标签眼底 B 超诊断场景中，7 类疾病天然分布不均衡（如"正常"远多于"眼球萎缩"），
ASL 能有效缓解长尾类别被淹没的问题。

参考 ARVO 2024 多视图 OCT 多病种诊断论文中使用的 asymmetry-balanced focal loss，
其在 8 类不平衡多标签任务上验证有效。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Args:
        gamma_pos: 正样本聚焦参数 (default: 1.0, 推荐 0~2)
        gamma_neg: 负样本聚焦参数 (default: 4.0, 推荐 2~5)
        m:        概率偏移量, 用于丢弃非常容易的负样本 (default: 0.05, 推荐 0~0.1)
        eps:      数值稳定性常数 (default: 1e-8)
        reduction: 'mean' | 'sum' | 'none' (default: 'mean')
        clip:     是否在取 log 前对概率做 clamp (default: 0.0，0 表示不 clamp)

    Suggested hyperparameters:
        - General multi-label:  gamma_pos=1, gamma_neg=4, m=0.05
        - Mild imbalance:       gamma_pos=0, gamma_neg=2, m=0.05
        - Severe imbalance:     gamma_pos=1, gamma_neg=5, m=0.10
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        m: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        clip: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.m = m
        self.eps = eps
        self.reduction = reduction
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw logits (before sigmoid)
            targets: [B, C] ground-truth multi-hot labels (0 or 1)
        Returns:
            loss: scalar if reduction='mean'/'sum', else [B, C]
        """
        # ── Sigmoid to get probabilities ──
        probs = torch.sigmoid(logits)

        # ── Numerical stability: clamp probs away from 0/1 ──
        if self.clip > 0:
            probs = probs.clamp(self.clip, 1.0 - self.clip)

        # ── Positive samples (targets == 1) ──
        # L+ = (1 - p)^gamma_pos * log(p)
        pos_term = (1.0 - probs).pow(self.gamma_pos) * torch.log(probs + self.eps)

        # ── Negative samples (targets == 0) ──
        # Probability shifting: 丢弃 p < m 的极容易负样本
        pm = (probs - self.m).clamp(min=0.0)  # max(p - m, 0)
        # L- = pm^gamma_neg * log(1 - pm)
        # 注意: 当 pm==0 时, log(1-pm)=0, 且 pm^gamma_neg=0, loss=0
        neg_term = pm.pow(self.gamma_neg) * torch.log(1.0 - pm + self.eps)

        # ── Combine ──
        # targets=1 用 pos_term, targets=0 用 neg_term
        loss_matrix = -targets * pos_term - (1.0 - targets) * neg_term

        if self.reduction == "mean":
            return loss_matrix.mean()
        elif self.reduction == "sum":
            return loss_matrix.sum()
        return loss_matrix


class AsymmetricLossOptimized(AsymmetricLoss):
    """
    针对正样本做了更精细的聚焦处理，同时继承 ASL 的核心逻辑。

    与 ASL 的区别：
        - 正样本使用 Focal Loss 风格: -(1-p)^gamma_pos * log(p)
        - 负样本与 ASL 一致: -pm^gamma_neg * log(1-pm)
        - 额外支持 label smoothing

    参考 ARVO 2024 多视图 OCT 多病种论文中的 asymmetry-balanced focal loss:
        gamma_pos=1, gamma_neg=4, m=0.05
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        m: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        clip: float = 1e-7,
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            gamma_pos=gamma_pos, gamma_neg=gamma_neg,
            m=m, eps=eps, reduction=reduction, clip=clip,
        )
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ── Apply label smoothing if requested ──
        if self.label_smoothing > 0 and self.training:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        return super().forward(logits, targets)


# ── Helper: build loss from config ──

class FocalLoss(nn.Module):
    """
    标准 Focal Loss（多标签 sigmoid 版）。

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    公式:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        p_t = p        (y=1)
            = 1 - p    (y=0)

    Args:
        gamma:      聚焦参数，越大越聚焦难例（default 2.0，标准值）
        alpha:      正样本缩放因子，标量（default None，即不做类别平衡；
                    标准 focal 常用 0.25，适合正样本稀少的场景）
        pos_weight: 每类正样本权重 [C]，比 alpha 更精细地处理类别不平衡，
                    可直接给稀有类（后巩膜）更高权重。与 alpha 二选一。
        reduction:  'mean' | 'sum'

    注意：gamma 只解决"难例 vs 易例"，不解决"正样本 vs 负样本数量不平衡"。
          针对稀有类（后巩膜葡萄肿）漏检，应使用 pos_weight 给该类别加权。
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = None,
        pos_weight=None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # focal 调制因子 (1 - p_t)^gamma
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce * ((1.0 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_t

        if self.pos_weight is not None:
            w = targets * self.pos_weight + (1.0 - targets)
            loss = loss * w

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def build_loss(cfg: dict) -> nn.Module:
    """
    Build loss function from configuration.

    Config keys (under cfg['loss']):
        name:            "bce" | "asl" | "asl_optimized"
        gamma_pos:       (ASL) positive focusing parameter, default 1.0
        gamma_neg:       (ASL) negative focusing parameter, default 4.0
        m:               (ASL) probability shift, default 0.05
        label_smoothing: (ASL_optimized) label smoothing, default 0.0
        pos_weight:      (BCE) optional per-class positive weight

    Returns:
        loss module
    """
    loss_name = cfg.get("name", "bce")

    if loss_name == "bce" or loss_name == "bce_with_logits":
        pos_weight = cfg.get("pos_weight", None)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    elif loss_name == "asl":
        return AsymmetricLoss(
            gamma_pos=cfg.get("gamma_pos", 1.0),
            gamma_neg=cfg.get("gamma_neg", 4.0),
            m=cfg.get("m", 0.05),
            reduction=cfg.get("reduction", "mean"),
        )

    elif loss_name == "asl_optimized":
        return AsymmetricLossOptimized(
            gamma_pos=cfg.get("gamma_pos", 1.0),
            gamma_neg=cfg.get("gamma_neg", 4.0),
            m=cfg.get("m", 0.05),
            reduction=cfg.get("reduction", "mean"),
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )

    elif loss_name == "focal":
        return FocalLoss(
            gamma=cfg.get("gamma", 2.0),
            alpha=cfg.get("alpha", None),
            pos_weight=cfg.get("pos_weight", None),
            reduction=cfg.get("reduction", "mean"),
        )

    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
