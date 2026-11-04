"""诊断模型（多标签分类）—— Phase 2: View-Gated Multi-Head Attention Fusion。"""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ImageBackbone
from .fusion import ViewGatedMHAFusion, ContrastiveProjectionHead, CrossViewAttention
from .label_gcn import LabelCooccurrenceGCN
from .evidence import CoverageEvidenceHead


class DiagnosisHead(nn.Module):
    """带 skip-connection 的深层分类头。"""

    def __init__(self, embed_dim: int, num_labels: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 2)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc_out = nn.Linear(embed_dim, num_labels)
        self.norm1 = nn.LayerNorm(embed_dim * 2)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.norm1(self.fc1(pooled))))
        x = self.dropout(F.gelu(self.norm2(self.fc2(x) + pooled)))
        return self.fc_out(x)


class DiagnosisModel(nn.Module):
    """
    Phase 2 多视图诊断模型。

    流程:
        images [B, N, 3, H, W]
          → ConvNeXt Backbone → view_features [B, N, D]
          → Cross-View Attention (可选) → 视图交互特征 [B, N, D]
          → ViewGate + Multi-Head Attention Pooling → pooled [B, D]
          → DiagnosisHead → logits [B, num_labels]

    额外输出（用于 Cross-View Contrastive Loss）:
        - view_features: 原始视图特征（未池化）
        - view_weights:   各注意力头的权重 [B, K, N]
    """

    def __init__(
        self,
        num_labels: int,
        embed_dim: int = 256,
        fusion_heads: int = 4,      # 多头注意力头数
        fusion_hidden_dim: int = 128,
        max_views: int = 16,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        backbone_name: str = "convnext_tiny",
        # Phase 2: Cross-view contrastive
        use_contrastive: bool = True,
        contrastive_proj_dim: int = 128,
        attn_dropout: float = 0.0,
        # Cross-View Transformer 跨视图交互
        use_cross_view: bool = True,
        cross_view_layers: int = 1,
        cross_view_heads: int = 4,
        # 门控开关
        use_view_gate: bool = True,
        use_cross_view_gate: bool = True,
        # Label GCN 开关
        use_label_gcn: bool = False,
        label_adj=None,
        evidence_config=None,
    ):
        super().__init__()
        evidence_config = evidence_config or {}
        self.use_evidence = evidence_config.get("enabled", False)
        if self.use_evidence and use_label_gcn:
            raise ValueError("Evidence head and label GCN are alternative heads")

        # ── Backbone ──
        self.backbone = ImageBackbone(
            embed_dim=embed_dim, pretrained=pretrained,
            dropout=dropout, freeze_backbone=freeze_backbone,
            backbone_name=backbone_name,
        )

        # ── Phase 2: View-Gated Multi-Head Attention Fusion ──
        self.fusion = None if self.use_evidence else ViewGatedMHAFusion(
            embed_dim=embed_dim,
            num_heads=fusion_heads,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_view_gate=use_view_gate,
        )

        # ── Cross-View Interaction (Cross-View Transformer) ──
        self.use_cross_view = use_cross_view
        if use_cross_view:
            self.cross_view = CrossViewAttention(
                embed_dim=embed_dim,
                num_heads=cross_view_heads,
                num_layers=cross_view_layers,
                dropout=dropout,
                use_gate=use_cross_view_gate,
            )

        # ── Contrastive projection head ──
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_head = ContrastiveProjectionHead(
                input_dim=embed_dim,
                proj_dim=contrastive_proj_dim,
            )

        # ── Classification head（或 Label GCN）──
        self.use_label_gcn = use_label_gcn
        if self.use_evidence:
            self.evidence_head = CoverageEvidenceHead(
                embed_dim, num_labels, dropout=dropout,
                temperature=evidence_config.get("temperature", 0.5),
                redundancy_temperature=evidence_config.get("redundancy_temperature", 0.1),
                use_redundancy=evidence_config.get("use_redundancy", True),
                use_focal=evidence_config.get("use_focal", True),
                mean_only=evidence_config.get("mean_only", False),
            )
        elif use_label_gcn:
            self.label_gcn = LabelCooccurrenceGCN(
                num_classes=num_labels,
                embed_dim=embed_dim,
                adj=label_adj,
                dropout=dropout,
            )
        else:
            self.head = DiagnosisHead(embed_dim, num_labels, dropout=dropout)

    def forward(
        self, images: torch.Tensor, view_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            diag_logits:   [B, num_labels]
            view_features: [B, N, D] 原始视图特征（用于对比损失）
            pooled:        [B, D] 池化特征
            view_weights:  [B, K, N] 各注意力头权重
        """
        # ── Extract per-view features ──
        view_feats = self.backbone(images, view_mask)         # [B, N, D]
        return self.forward_features(view_feats, view_mask)

    def forward_features(self, view_feats, view_mask):
        """Re-pool raw features; recompute interactions for each subset mask."""
        view_mask = view_mask.bool()
        if not view_mask.any(-1).all():
            raise ValueError("Every eye must have at least one valid view")

        # ── Cross-view interaction (optional) ──
        # 视图间交换互补信息；原始 view_feats 仍保留用于对比损失
        if self.use_cross_view:
            interacted = self.cross_view(view_feats, view_mask)  # [B, N, D]
        else:
            interacted = view_feats

        # ── View-Gated Multi-Head Attention Pooling ──
        if self.use_evidence:
            output = self.evidence_head(interacted, view_mask)
            output["view_features"] = view_feats
            return output
        pooled, attn_weights = self.fusion(interacted, view_mask)  # [B,D], [B,K,N]

        # ── Diagnosis prediction（Label GCN 或 MLP head）──
        if self.use_label_gcn:
            logits = self.label_gcn(pooled)
        else:
            logits = self.head(pooled)

        return {
            "diag_logits": logits,
            "view_features": view_feats,
            "pooled": pooled,
            "view_weights": attn_weights,
        }

    def get_contrastive_features(
        self, view_features: torch.Tensor, view_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        从视图特征中提取对比学习的投影特征。

        Args:
            view_features: [B, N, D]
            view_mask:     [B, N] bool
        Returns:
            proj_features: [total_real_views, proj_dim] L2 normalized
        """
        if not self.use_contrastive:
            return view_features[view_mask]

        flat = view_features[view_mask]                       # [M, D]
        return self.contrastive_head(flat)                     # [M, proj_dim]

    @torch.no_grad()
    def inference(self, images, view_mask):
        return self.forward(images, view_mask)
