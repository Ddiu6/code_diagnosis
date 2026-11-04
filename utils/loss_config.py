"""Resolve positive-class weights exclusively from training eye labels."""
import math
import numpy as np


def resolve_training_loss(loss_cfg, train_labels):
    cfg = dict(loss_cfg)
    labels = np.asarray(train_labels, dtype=np.float64)
    if labels.ndim != 2 or min(labels.shape) == 0:
        raise ValueError("Training labels must be a nonempty [eyes, classes] matrix")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must be finite binary multi-hot values")
    counts = labels.sum(axis=0)
    audit = {"source": "train_only", "num_train_eyes": len(labels),
             "positive_counts": counts.astype(int).tolist(),
             "negative_counts": (len(labels) - counts).astype(int).tolist()}
    automatic = cfg.pop("auto_pos_weight", False)
    if automatic:
        if cfg.get("name", "bce") not in ("bce", "bce_with_logits", "focal"):
            raise ValueError("Automatic positive weights require BCE or focal loss")
        if cfg.get("pos_weight") is not None:
            raise ValueError("Specify either auto_pos_weight or explicit pos_weight, not both")
        if (counts == 0).any():
            raise ValueError("Training split has a class with no positive examples")
        # Defaults reproduce the previous automatic sqrt weights (cap 3).
        power = float(cfg.pop("pos_weight_power", 0.5))
        cap = float(cfg.pop("max_pos_weight", 3.0))
        if not math.isfinite(power) or not 0 < power <= 1:
            raise ValueError("pos_weight_power must lie in (0, 1]")
        if not math.isfinite(cap) or cap < 1:
            raise ValueError("max_pos_weight must be finite and >= 1")
        ratios = (len(labels) - counts) / counts
        cfg["pos_weight"] = np.clip(ratios ** power, 1.0, cap).tolist()
        audit.update(power=power, max_pos_weight=cap)
    elif "pos_weight_power" in cfg:
        raise ValueError("pos_weight_power requires auto_pos_weight=true")
    audit["resolved_loss"] = cfg
    return cfg, audit
