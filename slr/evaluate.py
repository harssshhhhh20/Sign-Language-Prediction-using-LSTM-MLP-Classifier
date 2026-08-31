"""Metrics.

Accuracy alone is not reportable on a 24-sample test set: one sequence is
4.2 percentage points, so 91.7% and 95.8% are the same measurement. Every
result therefore carries a Wilson confidence interval alongside the point
estimate, and macro-F1 alongside accuracy so that a model which collapses
onto the majority class cannot hide behind a balanced test set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1]
    and remains sensible at the small sample sizes this study operates at.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class FoldResult:
    """Metrics for one train/test fold."""

    protocol: str
    fold_id: str
    model: str
    feature_group: str
    normalisation: str
    seed: int

    n_train: int
    n_test: int
    n_features: int

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    top2_accuracy: float
    ci_low: float
    ci_high: float

    n_params: int = -1
    size_bytes: int = -1
    fit_seconds: float = 0.0
    held_out: str = ""
    per_class_f1: dict = field(default_factory=dict)
    confusion: list = field(default_factory=list)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("confusion", None)
        d.pop("per_class_f1", None)
        return d


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int = 2) -> float:
    if proba.shape[1] <= k:
        return 1.0
    topk = np.argsort(-proba, axis=1)[:, :k]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


def evaluate_fold(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    **meta,
) -> FoldResult:
    """Compute the full metric set for one fold."""
    y_true = np.asarray(y_true)
    y_pred = np.argmax(proba, axis=1)
    n = len(y_true)
    n_classes = len(labels)
    class_ids = list(range(n_classes))

    acc = float(accuracy_score(y_true, y_pred))
    lo, hi = wilson_interval(int(round(acc * n)), n)

    _, _, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0
    )

    return FoldResult(
        accuracy=acc,
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=class_ids,
                                average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=class_ids,
                                   average="weighted", zero_division=0)),
        top2_accuracy=top_k_accuracy(y_true, proba, k=2),
        ci_low=lo,
        ci_high=hi,
        n_test=n,
        per_class_f1={labels[i]: float(f1_per[i]) for i in range(n_classes)},
        confusion=confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        **meta,
    )


def aggregate(results: list[FoldResult], metric: str = "accuracy") -> dict:
    """Mean, std and range across folds/seeds.

    Reporting a single run's accuracy from a stochastic training procedure is
    how the original project arrived at numbers between 0.41 and 0.87 for the
    same configuration. Aggregate everything.
    """
    vals = np.asarray([getattr(r, metric) for r in results], dtype=float)
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
    }


def format_mean_std(agg: dict, pct: bool = True) -> str:
    if agg.get("n", 0) == 0:
        return "n/a"
    scale = 100.0 if pct else 1.0
    suffix = "" if pct else ""
    return f"{agg['mean'] * scale:.1f} +/- {agg['std'] * scale:.1f}{suffix}"
