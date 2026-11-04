"""
OURS: View-Gated Multi-Head Attention MIL (Phase 2).

Architecture:
    Multiple B-scan views → Shared ConvNeXt Tiny
    → View Gate + Multi-Head Attention Pooling
    → Deep Diagnosis Head + Cross-View Contrastive Loss
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from models.diagnosis import DiagnosisModel


class CLSTransformerMIL(nn.Module):
    """
    Phase 2: View-Gated Multi-Head Attention MIL.

    Input:
        images:     [B, max_views, 3, H, W]  (padded batch)
        view_mask:  [B, max_views]  (1=real, 0=padding)

    Output modes:
        - Standard: (logits [B, C], None)
        - With attention: (logits, view_weights [B, K, N])
        - With contrastive: dict with diag_logits, view_features, pooled, view_weights
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        embed_dim: int = 256,
        fusion_heads: int = 4,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.2,
        backbone_name: str = "convnext_tiny",
        use_contrastive: bool = True,
        contrastive_proj_dim: int = 128,
        attn_dropout: float = 0.0,
        use_cross_view: bool = True,
        cross_view_layers: int = 1,
        cross_view_heads: int = 4,
        use_view_gate: bool = True,
        use_cross_view_gate: bool = True,
        use_label_gcn: bool = False,
        label_adj=None,
        evidence_config=None,
    ):
        super().__init__()

        self.diagnosis_model = DiagnosisModel(
            num_labels=num_classes,
            embed_dim=embed_dim,
            fusion_heads=fusion_heads,
            fusion_hidden_dim=fusion_hidden_dim,
            max_views=16,
            pretrained=pretrained,
            dropout=dropout,
            backbone_name=backbone_name,
            use_contrastive=use_contrastive,
            contrastive_proj_dim=contrastive_proj_dim,
            attn_dropout=attn_dropout,
            use_cross_view=use_cross_view,
            cross_view_layers=cross_view_layers,
            cross_view_heads=cross_view_heads,
            use_view_gate=use_view_gate,
            use_cross_view_gate=use_cross_view_gate,
            use_label_gcn=use_label_gcn,
            label_adj=label_adj,
            evidence_config=evidence_config,
        )

        self.num_classes = num_classes

    def forward(
        self, images: torch.Tensor, view_mask: torch.Tensor,
        return_attention: bool = False,
        return_full: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            images:           [B, N, 3, H, W]
            view_mask:        [B, N]
            return_attention: return view attention weights
            return_full:      return full dict (for training with contrastive loss)

        Returns:
            if return_full:   Dict with diag_logits, view_features, pooled, view_weights
            else:             (logits [B, C], attn_weights [B, K, N] or None)
        """
        if view_mask.dtype != torch.bool:
            view_mask = view_mask.bool()

        output = self.diagnosis_model(images, view_mask)

        if return_full:
            return output

        logits = output["diag_logits"]
        attn = output.get("view_weights", None) if return_attention else None
        return logits, attn

    def predict(self, images: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """Inference: return sigmoid probabilities."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(images, view_mask)
            probs = torch.sigmoid(logits)
        return probs
