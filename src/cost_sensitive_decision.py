"""Developer 3: Cost-Sensitive Decision Rule & Cost Trade-Off Analysis for DIFW-KNN.

Implements the Bayes-optimal cost-sensitive threshold:
    tau* = C_FP / (C_FP + C_FN)
    y_hat(x) = 1 if P_hat(yes | x) >= tau* else 0
and total expected misclassification cost evaluation:
    Total Cost = C_FP * FP + C_FN * FN.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.baseline_knn import compute_classification_metrics


def compute_optimal_threshold(c_fp: float = 1.0, c_fn: float = 5.0) -> float:
    """Compute Bayes cost-optimal probability threshold tau* = C_FP / (C_FP + C_FN)."""
    if c_fp < 0.0 or c_fn < 0.0:
        raise ValueError(f"Biaya kesalahan harus non-negatif: c_fp={c_fp}, c_fn={c_fn}.")
    denom = float(c_fp + c_fn)
    if denom <= 0.0:
        raise ValueError("Jumlah c_fp + c_fn harus > 0.")
    return float(c_fp) / denom


class CostSensitiveDecisionRule:
    """Apply cost-sensitive decision thresholding and compute cost metrics."""

    def __init__(self, c_fp: float = 1.0, c_fn: float = 5.0) -> None:
        self._threshold = compute_optimal_threshold(c_fp=c_fp, c_fn=c_fn)
        self.c_fp = float(c_fp)
        self.c_fn = float(c_fn)

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict(
        self,
        y_prob: np.ndarray | Sequence[float],
        threshold: float | None = None,
    ) -> np.ndarray:
        """Convert positive-class probabilities into binary predictions {0, 1}."""
        p = np.asarray(y_prob, dtype=np.float64)
        tau = self._threshold if threshold is None else float(threshold)
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"threshold harus dalam [0, 1], diterima {tau}.")
        return (p >= tau).astype(np.int64)

    def evaluate(
        self,
        y_true: np.ndarray | Sequence[int],
        y_prob: np.ndarray | Sequence[float],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate classification and misclassification-cost metrics at threshold."""
        y_t = np.asarray(y_true, dtype=np.int64)
        p = np.asarray(y_prob, dtype=np.float64)
        tau = self._threshold if threshold is None else float(threshold)
        if not (0.0 <= tau <= 1.0):
            raise ValueError(f"threshold harus dalam [0, 1], diterima {tau}.")

        metrics = compute_classification_metrics(y_t, p, threshold=tau)
        total_cost = float(self.c_fp * metrics["fp"] + self.c_fn * metrics["fn"])
        mean_cost = total_cost / float(len(y_t)) if len(y_t) > 0 else 0.0

        base_metrics = compute_classification_metrics(y_t, p, threshold=0.5)
        default_cost = float(self.c_fp * base_metrics["fp"] + self.c_fn * base_metrics["fn"])
        cost_reduction_pct = (
            100.0 * (default_cost - total_cost) / default_cost if default_cost > 0.0 else 0.0
        )

        return {
            **metrics,
            "threshold": float(tau),
            "c_fp": float(self.c_fp),
            "c_fn": float(self.c_fn),
            "total_cost": total_cost,
            "mean_cost": float(mean_cost),
            "default_05_total_cost": default_cost,
            "cost_reduction_pct": float(cost_reduction_pct),
        }


def evaluate_cost_ratio_grid(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    c_fn_grid: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 7.55, 10.0),
    c_fp: float = 1.0,
) -> pd.DataFrame:
    """Evaluate precision/recall/F1/cost trade-off across candidate C_FN ratios."""
    rows: list[dict[str, Any]] = []
    for c_fn in c_fn_grid:
        rule = CostSensitiveDecisionRule(c_fp=c_fp, c_fn=float(c_fn))
        res = rule.evaluate(y_true, y_prob)
        rows.append(res)
    return pd.DataFrame(rows)
