"""Coverage-aware, redundancy-corrected evidence pooling (research hypothesis).

Class-specific pooling and dual streams are established ideas, not novelty
claims. See PATIENT_SPLIT_ANALYSIS.md for the proposed contribution and controls.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CoverageEvidenceHead(nn.Module):
    def __init__(self, embed_dim, num_labels, dropout=0.1,
                 temperature=0.5, redundancy_temperature=0.1,
                 use_redundancy=True, use_focal=True, mean_only=False):
        super().__init__()
        if temperature <= 0 or redundancy_temperature <= 0:
            raise ValueError("Evidence temperatures must be positive")
        self.temperature = temperature
        self.redundancy_temperature = redundancy_temperature
        self.use_redundancy = use_redundancy
        self.use_focal = use_focal
        self.mean_only = mean_only
        self.global_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, num_labels),
        )
        if mean_only:
            return
        self.instance_classifier = nn.Linear(embed_dim, num_labels)
        # Start near mean pooling. Each disease learns how focal its evidence is.
        self.focal_mix = nn.Parameter(torch.zeros(num_labels))
        self.evidence_mix = nn.Parameter(torch.full((num_labels,), -2.0))

    def forward(self, features, mask):
        mask = mask.bool()
        if mask.ndim != 2 or features.shape[:2] != mask.shape:
            raise ValueError("features and mask dimensions do not match")
        if not mask.any(dim=1).all():
            raise ValueError("Every eye must have at least one valid view")
        # Accumulate density/logsumexp in fp32, including under autocast.
        x = features.float().masked_fill(~mask.unsqueeze(-1), 0)
        mean_features = x.sum(1) / mask.sum(1, keepdim=True)
        global_logits = self.global_head(mean_features).float()
        if self.mean_only:
            uniform = mask.float() / mask.sum(-1, keepdim=True)
            weights = uniform[:, None, :].expand(-1, global_logits.shape[1], -1)
            return {"diag_logits": global_logits, "global_logits": global_logits,
                    "pooled": mean_features, "view_weights": weights,
                    "coverage_weights": weights, "view_prior": uniform}
        if self.use_redundancy:
            with torch.no_grad():
                unit = F.normalize(x.detach(), dim=-1)
                with torch.autocast(device_type=x.device.type, enabled=False):
                    sim = torch.bmm(unit, unit.transpose(1, 2)).clamp(-1, 1)
                kernel = torch.exp((sim - 1) / self.redundancy_temperature)
                density = (kernel * mask[:, None, :]).sum(-1).clamp_min(1e-6)
                prior = mask.float() / density
        else:
            prior = mask.float()
        prior = prior / prior.sum(-1, keepdim=True)
        log_prior = prior.clamp_min(1e-30).log().masked_fill(~mask, -torch.inf)
        scores = self.instance_classifier(x).float()  # B,N,C
        diffuse = (scores * prior.unsqueeze(-1)).sum(1)
        scaled = scores / self.temperature + log_prior.unsqueeze(-1)
        focal = self.temperature * torch.logsumexp(scaled, dim=1)
        weights = torch.softmax(scaled, dim=1).transpose(1, 2)  # B,C,N
        if self.use_focal:
            mix = self.focal_mix.sigmoid()
            evidence = diffuse + mix * (focal - diffuse)
            coverage_weights = ((1 - mix)[None, :, None] * prior[:, None, :]
                                + mix[None, :, None] * weights)
        else:
            evidence = diffuse
            coverage_weights = prior[:, None, :].expand_as(weights)
        alpha = self.evidence_mix.sigmoid()
        logits = global_logits + alpha * (evidence - global_logits)
        # Contribution weights include the mean path, used by subset consistency.
        uniform = mask.float() / mask.sum(-1, keepdim=True)
        coverage_weights = ((1 - alpha)[None, :, None] * uniform[:, None, :]
                            + alpha[None, :, None] * coverage_weights)
        return {"diag_logits": logits, "evidence_logits": evidence,
                "global_logits": global_logits, "view_logits": scores,
                "pooled": mean_features, "view_weights": weights,
                "coverage_weights": coverage_weights, "view_prior": prior}


def sample_view_subset(mask, drop_probability):
    """Drop real views, always keeping at least one per eye (no label access)."""
    if not 0 <= drop_probability < 1:
        raise ValueError("view_drop_probability must be in [0, 1)")
    keep = (torch.rand(mask.shape, device=mask.device) >= drop_probability) & mask
    empty = ~keep.any(-1)
    if empty.any():
        index = torch.multinomial(mask[empty].float(), 1)
        restored = torch.zeros_like(keep[empty]).scatter_(1, index, True)
        keep[empty] = restored
    return keep


def coverage_consistency(student_logits, teacher, labels, subset_mask,
                         full_mask, min_coverage=0.7, confidence=0.7):
    """No invariance target for positive labels whose supporting views were lost.

    Teacher is a detached full-bag prediction, not an EMA teacher. This is a
    feature-subset regularizer, not a calibrated uncertainty estimate.
    """
    if not 0 <= min_coverage <= 1 or not 0.5 <= confidence <= 1:
        raise ValueError("Invalid consistency thresholds")
    with torch.no_grad():
        target = teacher["diag_logits"].float().sigmoid()
        coverage = (teacher["coverage_weights"].detach()
                    * subset_mask[:, None, :]).sum(-1)
        changed = (subset_mask != full_mask).any(-1, keepdim=True)
        negative = (labels == 0) & (target <= 1 - confidence)
        positive = ((labels == 1) & (target >= confidence)
                    & (coverage >= min_coverage))
        eligible = (negative | positive) & changed
    # Bernoulli KL: zero for matching predictions, with a detached target.
    entropy = -(torch.special.xlogy(target, target)
                + torch.special.xlogy(1 - target, 1 - target))
    kl = F.binary_cross_entropy_with_logits(
        student_logits.float(), target, reduction="none") - entropy
    return (kl * eligible).sum() / eligible.sum().clamp_min(1)
