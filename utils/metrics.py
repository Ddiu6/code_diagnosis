"""
Evaluation metrics for multi-label classification.

Supports:
  - micro-F1, macro-F1
  - micro precision, micro recall
  - exact match accuracy (EMR / subset accuracy)
  - per-class F1, precision, recall
"""

import numpy as np
from typing import Dict, Tuple, Optional


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 ROC-AUC（Mann-Whitney U 统计量），无需 sklearn。
    y_true: [N] 0/1 标签；y_score: [N] 预测概率。
    若某类无正样本或无负样本，返回 nan。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    i, n = 0, len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = np.mean(ranks[i:j + 1])
        i = j + 1

    sum_rank_pos = ranks[sorted_labels == 1].sum()
    return float((sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _to_probs(logits: np.ndarray, from_logits: bool = True) -> np.ndarray:
    """Input type must be explicit; small logits are still logits."""
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain non-finite values")
    if from_logits:
        return np.exp(-np.logaddexp(0, -values))
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    return values


def find_optimal_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
    n_steps: int = 200,
    search_range: Tuple[float, float] = (0.05, 0.95),
    from_logits: bool = True,
) -> np.ndarray:
    """
    对每个类别独立搜索最优阈值（在验证集上）。

    Args:
        logits: [N, C] 模型输出（logits 或概率均可，自动判断）
        labels: [N, C] multi-hot 标签
        metric: 搜索目标，'f1'（F1 最大）或 'youden'（约登指数 J = Sen + Spe - 1 最大）
        n_steps: 在 search_range 内采样的阈值个数
        search_range: 阈值搜索区间 (low, high)

    Returns:
        thresholds: [C] 每类最优阈值；某类无正样本时返回 0.5（默认值）。
    """
    probs = _to_probs(logits, from_logits=from_logits)
    labels = np.asarray(labels, dtype=np.float32)
    C = labels.shape[1]

    cand = np.linspace(search_range[0], search_range[1], n_steps)
    thresholds = np.full(C, 0.5, dtype=np.float64)

    for c in range(C):
        y = labels[:, c]
        p = probs[:, c]
        n_pos = int(y.sum())
        if n_pos == 0:
            thresholds[c] = 0.5  # 无正样本，用默认
            continue

        best_t, best_score = 0.5, -1.0
        for t in cand:
            pred = (p >= t).astype(np.float32)
            tp = float((pred * y).sum())
            fp = float((pred * (1 - y)).sum())
            fn = float(((1 - pred) * y).sum())
            tn = float(((1 - pred) * (1 - y)).sum())

            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)

            if metric == "f1":
                score = 2 * prec * rec / (prec + rec + 1e-8)
            elif metric == "youden":
                score = rec + specificity - 1.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_t = float(t)

        thresholds[c] = best_t

    return thresholds


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> Dict[str, float]:
    """Compute multi-label classification metrics.

    threshold 可以是标量（所有类共用），也可以是 [C] 数组（逐类阈值）。
    """
    probs = _to_probs(logits, from_logits=from_logits)

    if np.isscalar(threshold):
        preds = (probs >= threshold).astype(np.float32)
    else:
        thr = np.asarray(threshold, dtype=np.float64).reshape(1, -1)
        preds = (probs >= thr).astype(np.float32)
    labels = labels.astype(np.float32)

    N, C = labels.shape

    # Micro metrics
    tp_micro = np.sum(preds * labels)
    fp_micro = np.sum(preds * (1 - labels))
    fn_micro = np.sum((1 - preds) * labels)

    micro_precision = tp_micro / (tp_micro + fp_micro + 1e-8)
    micro_recall = tp_micro / (tp_micro + fn_micro + 1e-8)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)

    # Per-class metrics
    per_class_f1, per_class_precision, per_class_recall = [], [], []
    for c in range(C):
        tp_c = np.sum(preds[:, c] * labels[:, c])
        fp_c = np.sum(preds[:, c] * (1 - labels[:, c]))
        fn_c = np.sum((1 - preds[:, c]) * labels[:, c])
        prec_c = tp_c / (tp_c + fp_c + 1e-8)
        rec_c = tp_c / (tp_c + fn_c + 1e-8)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-8)
        per_class_f1.append(f1_c)
        per_class_precision.append(prec_c)
        per_class_recall.append(rec_c)

    macro_f1 = np.mean(per_class_f1)
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    exact_match = np.mean(np.all(preds == labels, axis=1).astype(np.float32))
    hamming_loss = np.mean(preds != labels)

    # Per-class support (positive sample count per class)
    per_class_support = [float(np.sum(labels[:, c])) for c in range(C)]

    # AUC metrics (ROC-AUC, 无需 sklearn)
    per_class_auc = [_roc_auc(labels[:, c], probs[:, c]) for c in range(C)]
    valid_auc = [a for a in per_class_auc if not np.isnan(a)]
    macro_auc = float(np.mean(valid_auc)) if valid_auc else float("nan")
    micro_auc = _roc_auc(labels.ravel(), probs.ravel())

    return {
        "micro_f1": float(micro_f1), "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision), "micro_recall": float(micro_recall),
        "macro_precision": float(macro_precision), "macro_recall": float(macro_recall),
        "exact_match_acc": float(exact_match), "hamming_loss": float(hamming_loss),
        "micro_auc": float(micro_auc), "macro_auc": float(macro_auc),
        "per_class_f1": [float(v) for v in per_class_f1],
        "per_class_precision": [float(v) for v in per_class_precision],
        "per_class_recall": [float(v) for v in per_class_recall],
        "per_class_auc": [float(v) for v in per_class_auc],
        "per_class_support": per_class_support,
    }


def format_metrics(metrics: Dict[str, float], label_names: Optional[list] = None) -> str:
    """Format metrics dict into a human-readable string."""
    def _auc_str(v):
        return "N/A" if (v != v) else f"{v:.4f}"

    lines = [
        "=" * 70,
        "  Evaluation Results",
        "=" * 70,
        f"  micro-F1:        {metrics['micro_f1']:.4f}",
        f"  macro-F1:        {metrics['macro_f1']:.4f}",
        f"  micro-AUC:       {_auc_str(metrics.get('micro_auc', float('nan')))}",
        f"  macro-AUC:       {_auc_str(metrics.get('macro_auc', float('nan')))}",
        f"  micro-Precision: {metrics['micro_precision']:.4f}",
        f"  micro-Recall:    {metrics['micro_recall']:.4f}",
        f"  macro-Precision: {metrics['macro_precision']:.4f}",
        f"  macro-Recall:    {metrics['macro_recall']:.4f}",
        f"  ExactMatch Acc:  {metrics['exact_match_acc']:.4f}",
        f"  Hamming Loss:    {metrics['hamming_loss']:.4f}",
    ]
    if label_names and "per_class_f1" in metrics:
        lines.append("")
        lines.append("  Per-Class Metrics:")
        lines.append(f"  {'Class':<16} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8} {'Sup':<6}")
        lines.append(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for i, name in enumerate(label_names):
            prec = metrics.get("per_class_precision", metrics["per_class_f1"])[i]
            rec = metrics.get("per_class_recall", metrics["per_class_f1"])[i]
            f1 = metrics["per_class_f1"][i]
            sup = int(metrics.get("per_class_support", [0] * len(label_names))[i])
            auc = metrics.get("per_class_auc", [float("nan")] * len(label_names))[i]
            auc_str = "N/A" if (auc != auc) else f"{auc:.4f}"
            lines.append(f"  {name:<16} {prec:<8.4f} {rec:<8.4f} {f1:<8.4f} {auc_str:<8} {sup:<6}")
    lines.append("=" * 70)
    return "\n".join(lines)
