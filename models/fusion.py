"""多视图融合模块（Phase 2 升级：Multi-Head Attention Pooling + View Gate）。"""

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Cross-View Interaction (Cross-View Transformer, MICCAI 2021)
# ============================================================

class CrossViewAttention(nn.Module):
    """
    门控选择性跨视图交互模块 (Cross-View Transformer, MICCAI 2021 + 门控残差)。

    核心：把跨视图交互做成"门控残差"，为每个视图学一个交互强度 g∈[0,1]：
        output = x + g ⊙ (interacted - x)
    - g≈0：视图保持原样（不交互），适合"单视图冗余"的易类（如玻璃体混浊/积血）
    - g≈1：视图完全采纳交互，适合"信息分散多视图"的难类（如后巩膜葡萄肿）

    与 ViewGate 的关系（两个门控不冲突，作用层级不同）：
        ViewGate       = 特征级逐维度门控（交互之后、pooling 之前，筛选哪些 channel 有用）
        本模块的 gate   = 视图级标量门控（控制"要不要交互、交互多少"）
        两者串联使用，语义不重叠。

    初始化：gate 最后一层 bias 置为 -2，使初始 g≈sigmoid(-2)≈0.12（接近"不交互"），
    训练从接近基线处起步，逐步学会选择性交互，避免早期交互扰动。

    保持 permutation invariance（不加位置编码），用 key_padding_mask 屏蔽 padding。

    Args:
        embed_dim:   特征维度
        num_heads:   注意力头数（需整除 embed_dim）
        num_layers:  Transformer 层数 (default: 1)
        dropout:     dropout
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 1, dropout: float = 0.1, use_gate: bool = True):
        super().__init__()
        self.num_layers = num_layers

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(embed_dim)

        # ── 视图级门控：每个视图一个标量 g，控制跨视图交互的增量 ──
        self.use_gate = use_gate
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
            # 初始化 bias=-2，使初始 g≈sigmoid(-2)≈0.12，训练从接近"不交互"起步
            nn.init.constant_(self.gate_net[-1].bias, -2.0)

    def forward(self, x: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         [B, N, D] 视图特征
            view_mask: [B, N] bool, True=real view, False=padding
        Returns:
            [B, N, D] 门控交互后的视图特征（维度不变）
        """
        key_padding_mask = ~view_mask.bool()   # [B, N] True=padding
        x_orig = x                              # 交互前特征，用于门控残差

        # ── 跨视图交互（多层 Pre-LN Transformer）──
        for i in range(self.num_layers):
            # Pre-LN self-attention
            normed = self.norm1[i](x)
            attn_out, _ = self.attn[i](
                normed, normed, normed, key_padding_mask=key_padding_mask,
            )
            x = x + attn_out
            # Pre-LN FFN
            x = x + self.ffn[i](self.norm2[i](x))

        interacted = self.out_norm(x)           # [B, N, D]

        # ── 视图级门控残差：g 控制"采纳多少交互增量" ──
        if self.use_gate:
            gate = torch.sigmoid(self.gate_net(x_orig))   # [B, N, 1]
            output = x_orig + gate * (interacted - x_orig)
        else:
            output = interacted                          # 无门控：直接采纳交互结果

        return output


# ============================================================
# View Gate: 动态筛选关键视图维度
# ============================================================

class ViewGate(nn.Module):
    """
    对每个视图的每个特征维度学习一个门控系数。
    gate = sigmoid(W * x)  →  gated = x * gate

    直觉：B 超图像中不同视图包含的病灶信息量不同，
    门控机制让模型学会抑制无用视图的噪声维度。
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        return view_features * self.gate(view_features)


# ============================================================
# Multi-Head Attention Pooling (MAD-MIL 风格)
# ============================================================

class GatedAttention(nn.Module):
    """
    单头门控注意力，用于加权聚合视图特征。

    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128, attn_dropout: float = 0.0):
        super().__init__()
        self.att_V = nn.Linear(embed_dim, hidden_dim)
        self.att_U = nn.Linear(embed_dim, hidden_dim)
        self.att_w = nn.Linear(hidden_dim, 1)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gated attention: w^T * (tanh(Vh) ⊙ sigm(Uh))
        a_v = torch.tanh(self.att_V(features))       # [B, N, H]
        a_u = torch.sigmoid(self.att_U(features))     # [B, N, H]
        a = self.att_w(a_v * a_u).squeeze(-1)         # [B, N]

        # Mask padding views
        a = a.masked_fill(~mask, float("-inf"))
        weights = F.softmax(a, dim=1)                 # [B, N]
        weights = weights.masked_fill(~mask, 0.0)
        weights = self.attn_dropout(weights)           # dropout on attention weights

        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, weights


class MultiHeadAttentionPooling(nn.Module):
    """
    多头注意力池化 (MAD-MIL 风格)。

    K 个并行注意力头，每个头学习不同的视图加权策略。
    拼接 K 个头的结果后投影回 embed_dim。

    Reference: Keshvarikhojasteh et al., "MAD-MIL", 2024.

    直觉：B 超的不同视图可能侧重于不同的病灶特征
    （如某个视图突出玻璃体混浊，另一个视图突出视网膜脱离），
    多头机制让模型同时捕捉多种病理模式。

    Args:
        embed_dim:   特征维度
        num_heads:   注意力头数 (default: 4)
        hidden_dim:  注意力隐层维度 (default: 128)
        dropout:     projection dropout (default: 0.1)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.attention_heads = nn.ModuleList([
            GatedAttention(embed_dim, hidden_dim, attn_dropout) for _ in range(num_heads)
        ])

        # Project concatenated heads back to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * num_heads, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, N, D]
            mask:     [B, N] bool
        Returns:
            pooled:   [B, D]
            weights:  [B, K, N] 每个头的注意力权重
        """
        head_outputs = []
        head_weights = []

        for head in self.attention_heads:
            p, w = head(features, mask)        # p: [B, D], w: [B, N]
            head_outputs.append(p)
            head_weights.append(w)

        # Stack and project
        pooled = self.proj(torch.cat(head_outputs, dim=-1))  # [B, D]
        weights = torch.stack(head_weights, dim=1)            # [B, K, N]

        return pooled, weights


# ============================================================
# 组合模块: View-Gated Multi-Head Attention Fusion
# ============================================================

class ViewGatedMHAFusion(nn.Module):
    """
    视图门控 + 多头注意力融合的组合模块。

    流程:
        view_features [B, N, D]
          → ViewGate: 逐视图维度门控
          → MultiHeadAttentionPooling: K 头注意力聚合
          → pooled [B, D] + view_weights [B, K, N]

    Args:
        embed_dim:   特征维度
        num_heads:   注意力头数
        hidden_dim:  注意力隐层维度
        dropout:     dropout
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_view_gate: bool = True,
    ):
        super().__init__()
        self.use_view_gate = use_view_gate
        if use_view_gate:
            self.view_gate = ViewGate(embed_dim)
        self.attn_pool = MultiHeadAttentionPooling(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(
        self, view_features: torch.Tensor, view_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_view_gate:
            gated = self.view_gate(view_features)
        else:
            gated = view_features
        pooled, weights = self.attn_pool(gated, view_mask)
        return pooled, weights


# ============================================================
# 对比投影头（用于 Cross-View Contrastive Loss）
# ============================================================

class ContrastiveProjectionHead(nn.Module):
    """
    将特征投影到对比学习空间。

    两层 MLP + BN + ReLU，输出归一化到单位超球面。
    SimCLR 风格 projection head。

    Args:
        input_dim:  输入维度（view feature dim）
        proj_dim:   投影维度 (default: 128)
    """

    def __init__(self, input_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [total_views, D] or [B*N, D]
        Returns:
            [total_views, proj_dim] normalized
        """
        return F.normalize(self.net(x), dim=-1)


# ============================================================
# Cross-View Contrastive Loss
# ============================================================

def cross_view_contrastive_loss(
    view_features: torch.Tensor,
    view_mask: torch.Tensor,
    temperature: float = 0.07,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    跨视图对比损失 (InfoNCE)。

    对每只眼睛：
      - 正样本对：同一只眼睛的不同视图
      - 负样本对：不同眼睛的视图
    目标是拉近同眼视图、推远异眼视图。

    实现细节：
      - 只有 ≥2 个视图的眼睛参与损失计算
      - 对每个有效视图作为 anchor，同眼其他视图 as positive，异眼全部 as negative

    Args:
        view_features: [B, N, D] 各视图特征（backbone 输出，未池化）
        view_mask:     [B, N] bool
        temperature:   温度系数 (default: 0.07)

    Returns:
        scalar loss, 如果没有有效正样本对则返回 0
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    B, N, D = view_features.shape

    # Flatten real views: [total_real, D]
    flat = view_features[view_mask]                 # [M, D]
    M = flat.shape[0]
    if M < 2:
        return view_features.sum() * 0.0

    # L2 normalize
    flat = F.normalize(flat.float(), dim=-1)

    # Cosine similarity matrix: [M, M]
    with torch.autocast(device_type=flat.device.type, enabled=False):
        sim = torch.mm(flat, flat.t()) / temperature

    # Build eye id per flattened view
    eye_ids = torch.arange(B, device=view_features.device)
    eye_ids = eye_ids.unsqueeze(1).expand(B, N)     # [B, N]
    eye_ids = eye_ids[view_mask]                     # [M]

    # Positive mask: same eye, different view
    same_eye = eye_ids.unsqueeze(0) == eye_ids.unsqueeze(1)   # [M, M]
    self_mask = torch.eye(M, dtype=torch.bool, device=view_features.device)
    pos_mask = same_eye & ~self_mask

    # Filter anchors that have at least one positive
    has_pos = pos_mask.sum(dim=1) > 0
    if not has_pos.any():
        return view_features.sum() * 0.0

    candidates = ~self_mask
    if labels is not None:
        # Do not repel different eyes that share any known positive label.
        overlap = (labels.float() @ labels.float().T) > 0
        candidates &= same_eye | ~overlap[eye_ids[:, None], eye_ids[None, :]]
    # Select valid rows BEFORE logsumexp to avoid all-masked NaN gradients.
    positive_logsum = sim[has_pos].masked_fill(~pos_mask[has_pos], -torch.inf).logsumexp(-1)
    denominator = sim[has_pos].masked_fill(~candidates[has_pos], -torch.inf).logsumexp(-1)
    return (denominator - positive_logsum).mean()


# ============================================================
# 保留旧的 CLS Token Fusion 作为可选方案
# ============================================================

class TransformerFusionEncoder(nn.Module):
    """带 CLS Token 的 Transformer Encoder 多视图融合（保留兼容）。"""

    def __init__(self, embed_dim: int = 256, num_layers: int = 2,
                 num_heads: int = 4, max_views: int = 16, dropout: float = 0.2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.view_pos = nn.Parameter(torch.randn(1, max_views + 1, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, view_features: torch.Tensor, view_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = view_features.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, view_features], dim=1)
        tokens = tokens + self.view_pos[:, :(N + 1), :]
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=view_mask.device)
        full_mask = torch.cat([cls_mask, view_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        encoded = self.out_norm(encoded)
        return encoded[:, 0, :], encoded[:, 1:, :]
