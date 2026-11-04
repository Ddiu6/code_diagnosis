"""Fail-closed validation of a fixed patient-disjoint experimental split."""
import hashlib
import json
from pathlib import Path


def validate_patient_split(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fixed patient split required: {path}")
    raw = path.read_bytes()
    split = json.loads(raw.decode("utf-8-sig"))
    owners, assignments, counts = {}, {}, {}
    for subset in ("train", "val", "test"):
        eyes = split.get(subset + "_eyes")
        if not isinstance(eyes, list) or not eyes:
            raise ValueError(f"Missing or empty {subset}_eyes")
        for item in eyes:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Each split entry must be [patient, eye]")
            patient, eye = map(str, item)
            normalized_patient = patient.strip().casefold()
            if not normalized_patient:
                raise ValueError("Empty patient ID")
            if normalized_patient in owners and owners[normalized_patient] != subset:
                raise ValueError("Patient leakage: a patient appears in multiple subsets")
            owners[normalized_patient] = subset
            key = (patient, eye)
            if key in assignments:
                raise ValueError("Duplicate eye entry in split")
            assignments[key] = subset
        counts[subset] = {"eyes": len(eyes),
                          "patients": len({str(p).strip().casefold() for p, _ in eyes})}
    return assignments, {"sha256": hashlib.sha256(raw).hexdigest(), "counts": counts}


def validate_checkpoint_metadata(checkpoint, label_names, split_audit, model_config):
    """Reject silent label-order, frozen-split or architecture changes at eval."""
    if "label_names" in checkpoint and list(checkpoint["label_names"]) != list(label_names):
        raise ValueError("Checkpoint label order differs from the current dataset")
    saved_split = checkpoint.get("split_audit")
    if saved_split is not None:
        if split_audit is None or saved_split["sha256"] != split_audit["sha256"]:
            raise ValueError("Checkpoint requires the same verified patient split")
    saved_model = checkpoint.get("config", {}).get("model")
    if saved_model is not None and saved_model != model_config:
        raise ValueError("Use the checkpoint's saved model configuration for evaluation")
