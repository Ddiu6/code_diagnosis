"""
Run the existing OURS training pipeline with five random seeds.

The script creates one isolated config/checkpoint/log directory per seed and
forces all runs to reuse the same patient-level split.json.  It then runs the
existing ``main.py --mode test`` command and aggregates the test metrics.

Example:
    python main_seed.py
    python main_seed.py --output-dir ./OURS/3 --split-file split.json \
        --seeds 42,123,2026,3407,9001 --device cuda

The data paths still come from the selected YAML config.  ``split_file`` is
overridden only in the per-seed config so the original config is unchanged.
If ``--output-dir`` is omitted, ``logging.save_dir`` is used to derive the
output root.  For example, ``./OURS/3/checkpoints`` becomes ``./OURS/3`` and
each run is stored under ``./OURS/3/seed_<seed>/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import yaml
from engine.early_stopping import EarlyStopping


DEFAULT_SEEDS = (42, 123, 2026, 3407, 9001)
METRIC_KEYS = (
    "micro_f1",
    "macro_f1",
    "micro_auc",
    "macro_auc",
    "micro_precision",
    "micro_recall",
    "macro_precision",
    "macro_recall",
    "exact_match_acc",
    "hamming_loss",
)
METRIC_LABELS = {
    "micro_f1": "micro-F1",
    "macro_f1": "macro-F1",
    "micro_auc": "micro-AUC",
    "macro_auc": "macro-AUC",
    "micro_precision": "micro-Precision",
    "micro_recall": "micro-Recall",
    "macro_precision": "macro-Precision",
    "macro_recall": "macro-Recall",
    "exact_match_acc": "ExactMatch Acc",
    "hamming_loss": "Hamming Loss",
}


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train OURS with five fixed random seeds.")
    parser.add_argument(
        "--config",
        default=str(script_dir / "configs" / "config.yaml"),
        help="Base YAML config used by main.py.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional split JSON override; default is logging.split_file from the YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output root. By default it is derived from logging.save_dir; "
            "e.g. ./OURS/3/checkpoints -> ./OURS/3."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated integer seeds; default is 5 seeds: 42,123,2026,3407,9001.",
    )
    parser.add_argument("--device", default="cuda", help="Device passed to main.py, e.g. cuda or cpu.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Consecutive epochs without a new best checkpoint. "
                             "Default: config value, otherwise 40. Use 0 to disable.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a seed if its test_results.txt already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed seed; default is to continue and summarize failures.",
    )
    return parser.parse_args()


def _resolve_path(value: str, base_dir: Path) -> Path:
    value = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_patience(cli_value, train_cfg):
    value = cli_value if cli_value is not None else train_cfg.get("early_stopping_patience", 40)
    return EarlyStopping(value).patience


def _resolve_output_dir(
    cli_output_dir: str | None,
    configured_save_dir: str | None,
    script_dir: Path,
) -> Path:
    """Resolve the common root that will contain all seed directories."""
    if cli_output_dir:
        return _resolve_path(cli_output_dir, script_dir)
    if configured_save_dir:
        configured_path = _resolve_path(configured_save_dir, script_dir)
        # A normal single-run config points to .../checkpoints.  For multiple
        # seeds, place seed_<n> beside that directory and give each seed its
        # own checkpoints/logs subdirectories.
        if configured_path.name.lower() == "checkpoints":
            return configured_path.parent
        return configured_path
    return script_dir / "seed_runs"


def _parse_seeds(text: str) -> List[int]:
    try:
        seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --seeds value: {text!r}") from exc
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique: {seeds}")
    return seeds


def _read_metrics(result_path: Path) -> Dict[str, float]:
    """Parse the text written by utils.metrics.format_metrics()."""
    metrics: Dict[str, float] = {}
    if not result_path.is_file():
        return metrics
    text = result_path.read_text(encoding="utf-8", errors="replace")
    for key, label in METRIC_LABELS.items():
        match = re.search(rf"^\s*{re.escape(label)}:\s*(N/A|[-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", text, re.MULTILINE)
        if match:
            value = match.group(1)
            metrics[key] = float("nan") if value == "N/A" else float(value)
    return metrics


def _run(command: Sequence[str], cwd: Path, log_path: Path) -> int:
    """Run a child command and stream output to both terminal and seed log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        command_line = "$ " + " ".join(command)
        print(command_line, flush=True)
        log.write("\n" + command_line + "\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to capture child-process output.")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def _write_config(path: Path, cfg: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(cfg), handle, allow_unicode=True, sort_keys=False)


def _aggregate(results: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
    aggregate: Dict[str, Dict[str, float]] = {}
    successful = [item for item in results if item.get("status") == "success"]
    for key in METRIC_KEYS:
        values = [item.get("metrics", {}).get(key, float("nan")) for item in successful]
        values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
        if not values:
            aggregate[key] = {"n": 0, "mean": float("nan"), "std": float("nan")}
            continue
        aggregate[key] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregate


def _write_outputs(output_dir: Path, results: Sequence[Mapping], aggregate: Mapping) -> None:
    summary = {
        "num_requested_seeds": len(results),
        "num_successful_seeds": sum(item.get("status") == "success" for item in results),
        "results": list(results),
        "aggregate": aggregate,
    }
    with (output_dir / "seed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)

    csv_path = output_dir / "seed_metrics.csv"
    fieldnames = ["seed", "status", *METRIC_KEYS]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {"seed": item.get("seed"), "status": item.get("status")}
            row.update(item.get("metrics", {}))
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    config_path = _resolve_path(args.config, script_dir)
    main_py = script_dir / "main.py"
    seeds = _parse_seeds(args.seeds)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not main_py.is_file():
        raise FileNotFoundError(f"Training entry point does not exist: {main_py}")

    with config_path.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)
    if not isinstance(base_cfg, Mapping):
        raise ValueError(f"Invalid YAML config: {config_path}")
    patience = _resolve_patience(args.early_stopping_patience, base_cfg.get("train", {}))

    configured_split = base_cfg.get("logging", {}).get("split_file")
    split_value = args.split_file or configured_split or str(script_dir / "split.json")
    split_file = _resolve_path(split_value, script_dir)
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Patient-level split file does not exist: {split_file}\n"
            "Run split_data.py first, set logging.split_file, or pass --split-file explicitly."
        )

    configured_save_dir = base_cfg.get("logging", {}).get("save_dir")
    output_dir = _resolve_output_dir(args.output_dir, configured_save_dir, script_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    print(f"[Seed] Base config: {config_path}")
    print(f"[Seed] Shared split: {split_file}")
    print(f"[Seed] Seeds: {seeds}")
    print(f"[Seed] Output: {output_dir}")
    print(f"[Seed] Early stopping patience: {patience} (0 = disabled)")

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        checkpoint_dir = seed_dir / "checkpoints"
        log_dir = seed_dir / "logs"
        seed_config_path = seed_dir / "config.yaml"
        test_result_path = checkpoint_dir / "test_results.txt"
        child_log = seed_dir / "run.log"

        if args.skip_existing and test_result_path.is_file():
            metrics = _read_metrics(test_result_path)
            results.append({
                "seed": seed,
                "status": "success",
                "skipped": True,
                "metrics": metrics,
                "test_results": str(test_result_path),
            })
            print(f"[Seed {seed}] Existing results found; skipped.")
            continue

        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = int(seed)
        cfg.setdefault("train", {})["early_stopping_patience"] = patience
        cfg.setdefault("logging", {})
        cfg["logging"]["save_dir"] = str(checkpoint_dir)
        cfg["logging"]["log_dir"] = str(log_dir)
        cfg["logging"]["split_file"] = str(split_file)
        # Avoid five repeated diagnostic images when the source config enables it.
        cfg.setdefault("data", {}).setdefault("crop", {})["show_sample"] = False
        _write_config(seed_config_path, cfg)

        train_command = [
            sys.executable,
            str(main_py),
            "--mode",
            "train",
            "--config",
            str(seed_config_path),
            "--device",
            args.device,
        ]
        test_command = [
            sys.executable,
            str(main_py),
            "--mode",
            "test",
            "--config",
            str(seed_config_path),
            "--device",
            args.device,
            "--checkpoint",
            str(checkpoint_dir / cfg["logging"].get("best_model_name", "best_model.pth")),
        ]

        print(f"\n[Seed {seed}] Training started...")
        train_code = _run(train_command, script_dir, child_log)
        if train_code != 0:
            item = {
                "seed": seed,
                "status": "train_failed",
                "returncode": train_code,
                "run_log": str(child_log),
            }
            results.append(item)
            print(f"[Seed {seed}] Training failed with return code {train_code}. See {child_log}")
            if args.fail_fast:
                break
            continue

        print(f"[Seed {seed}] Testing started...")
        test_code = _run(test_command, script_dir, child_log)
        if test_code != 0:
            item = {
                "seed": seed,
                "status": "test_failed",
                "returncode": test_code,
                "run_log": str(child_log),
            }
            results.append(item)
            print(f"[Seed {seed}] Testing failed with return code {test_code}. See {child_log}")
            if args.fail_fast:
                break
            continue

        metrics = _read_metrics(test_result_path)
        item = {
            "seed": seed,
            "status": "success",
            "metrics": metrics,
            "test_results": str(test_result_path),
            "checkpoint": str(checkpoint_dir / cfg["logging"].get("best_model_name", "best_model.pth")),
            "run_log": str(child_log),
        }
        results.append(item)
        print(f"[Seed {seed}] Finished. Metrics parsed: {sorted(metrics)}")
        training_summary = checkpoint_dir / "training_summary.json"
        if training_summary.is_file():
            item["training"] = json.loads(training_summary.read_text(encoding="utf-8"))

    aggregate = _aggregate(results)
    _write_outputs(output_dir, results, aggregate)

    print("\n[Seed] Summary")
    for key in METRIC_KEYS:
        item = aggregate[key]
        if item["n"] == 0:
            print(f"  {METRIC_LABELS[key]:<16} unavailable")
        else:
            print(f"  {METRIC_LABELS[key]:<16} {item['mean']:.4f} ± {item['std']:.4f} (n={item['n']})")
    print(f"[Seed] JSON summary: {output_dir / 'seed_summary.json'}")
    print(f"[Seed] CSV summary:  {output_dir / 'seed_metrics.csv'}")


if __name__ == "__main__":
    main()
