"""Reusable module for Baseline KNN experiments (E0 Standard & E1 Optimized).

Provides:
- Comprehensive imbalanced classification metrics (Accuracy, Precision, Recall,
  F1, PR-AUC, ROC-AUC, MCC, G-Mean, confusion counts) and 95% bootstrap CIs.
- Leak-free 5-fold CV grid search over (k, weights) reusing NearestNeighbors once
  per fold, verified for exact probability equivalence against KNeighborsClassifier.
- Final training on full 36,168 training split and hold-out test evaluation.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from src.preprocessing import BaselineEuclideanPreprocessor


METRIC_KEYS = (
    "accuracy",
    "precision_yes",
    "recall_yes",
    "f1_yes",
    "pr_auc",
    "roc_auc",
    "mcc",
    "g_mean",
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary classification metrics tailored for imbalanced evaluation."""
    y_t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (p >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_t, y_pred)),
        "precision_yes": float(precision_score(y_t, y_pred, pos_label=1, zero_division=0)),
        "recall_yes": float(recall_score(y_t, y_pred, pos_label=1, zero_division=0)),
        "f1_yes": float(f1_score(y_t, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_t, p)),
        "roc_auc": float(roc_auc_score(y_t, p)),
        "mcc": float(matthews_corrcoef(y_t, y_pred)),
        "g_mean": float(np.sqrt(tpr * tnr)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Estimate 95% percentile bootstrap confidence intervals for key metrics."""
    y_t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    n = len(y_t)
    rng = np.random.default_rng(random_state)

    collected: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_t[idx]
        if yb.min() == yb.max():
            continue
        pb = p[idx]
        m = compute_classification_metrics(yb, pb, threshold=threshold)
        for k in METRIC_KEYS:
            collected[k].append(m[k])

    cis: dict[str, tuple[float, float]] = {}
    for k in METRIC_KEYS:
        vals = np.asarray(collected[k], dtype=np.float64)
        cis[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return cis


def predict_proba_from_neighbors(
    y_train: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    k: int,
    weights: str,
) -> np.ndarray:
    """Compute positive-class probabilities from precomputed k-nearest neighbors."""
    if weights not in {"uniform", "distance"}:
        raise ValueError(f"weights tidak valid: {weights}")

    y_nn = y_train[indices[:, :k]].astype(np.float64)
    if weights == "uniform":
        return y_nn.mean(axis=1)

    d_nn = distances[:, :k].astype(np.float64)
    zero_mask = d_nn == 0.0
    has_zero = zero_mask.any(axis=1)

    w = np.empty_like(d_nn, dtype=np.float64)
    if has_zero.any():
        w[has_zero] = zero_mask[has_zero].astype(np.float64)
    if (~has_zero).any():
        w[~has_zero] = 1.0 / d_nn[~has_zero]

    return (w * y_nn).sum(axis=1) / w.sum(axis=1)


def verify_knn_probability_equivalence(
    X_train_euc: np.ndarray,
    y_train: np.ndarray,
    X_val_euc: np.ndarray,
    k_values: Sequence[int] = (5, 21),
    weights_list: Sequence[str] = ("uniform", "distance"),
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Verify NearestNeighbors probability helper matches KNeighborsClassifier and distance truncation."""
    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean", n_jobs=-1)
    nn.fit(X_train_euc)
    distances_max, indices_max = nn.kneighbors(X_val_euc, n_neighbors=max_k)

    rows: list[dict[str, Any]] = []
    for k in k_values:
        distances_k, indices_k = nn.kneighbors(X_val_euc, n_neighbors=k)
        max_dist_diff = float(np.max(np.abs(distances_max[:, :k] - distances_k)))
        if max_dist_diff > tol:
            raise AssertionError(
                f"Perbedaan vektor jarak terpotong melebihi toleransi ({tol}) pada k={k}: {max_dist_diff}"
            )
        for w in weights_list:
            p_exact_k = predict_proba_from_neighbors(y_train, distances_k, indices_k, k=k, weights=w)
            p_trunc_k = predict_proba_from_neighbors(y_train, distances_max, indices_max, k=k, weights=w)
            clf = KNeighborsClassifier(n_neighbors=k, weights=w, metric="euclidean", n_jobs=-1)
            clf.fit(X_train_euc, y_train)
            p_sklearn = clf.predict_proba(X_val_euc)[:, 1]
            max_prob_diff = float(np.max(np.abs(p_exact_k - p_sklearn)))
            n_tie_diff_rows = int(np.sum(np.abs(p_trunc_k - p_sklearn) > tol))
            if max_prob_diff > tol:
                raise AssertionError(
                    f"Perbedaan probabilitas melebihi toleransi ({tol}) pada k={k}, weights={w}: {max_prob_diff}"
                )
            rows.append(
                {
                    "k": int(k),
                    "weights": w,
                    "max_prob_abs_diff_k": max_prob_diff,
                    "max_dist_trunc_diff": max_dist_diff,
                    "n_boundary_tie_rows_maxk": n_tie_diff_rows,
                    "status": "PASS",
                }
            )
    return pd.DataFrame(rows)


def run_stratified_cv_grid(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    fold_ids: np.ndarray,
    k_grid: Sequence[int],
    weights_grid: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run leak-free 5-fold CV grid search for Euclidean Baseline KNN."""
    max_k = max(k_grid)
    per_fold_rows: list[dict[str, Any]] = []
    unique_folds = sorted(int(f) for f in np.unique(fold_ids))

    for fold_id in unique_folds:
        val_mask = fold_ids == fold_id
        tr_mask = ~val_mask

        X_tr_fold = X_train_df.loc[tr_mask].reset_index(drop=True)
        y_tr_fold = y_train[tr_mask]
        X_va_fold = X_train_df.loc[val_mask].reset_index(drop=True)
        y_va_fold = y_train[val_mask]

        t0 = time.perf_counter()
        prep = BaselineEuclideanPreprocessor()
        X_tr_euc = prep.fit_transform(X_tr_fold)
        X_va_euc = prep.transform(X_va_fold)

        nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean", n_jobs=-1)
        nn.fit(X_tr_euc)
        distances, indices = nn.kneighbors(X_va_euc)
        fold_nn_sec = time.perf_counter() - t0

        for w in weights_grid:
            for k in k_grid:
                y_prob = predict_proba_from_neighbors(y_tr_fold, distances, indices, k=k, weights=w)
                metrics = compute_classification_metrics(y_va_fold, y_prob, threshold=0.5)
                per_fold_rows.append(
                    {
                        "fold_id": fold_id,
                        "k": int(k),
                        "weights": w,
                        "fold_prep_nn_sec": round(fold_nn_sec, 4),
                        **metrics,
                    }
                )

    per_fold_df = pd.DataFrame(per_fold_rows)
    agg_spec: dict[str, tuple[str, str]] = {}
    for m in METRIC_KEYS:
        agg_spec[f"{m}_mean"] = (m, "mean")
        agg_spec[f"{m}_std"] = (m, "std")

    summary_df = (
        per_fold_df.groupby(["weights", "k"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["weights", "k"])
        .reset_index(drop=True)
    )
    return per_fold_df, summary_df


def evaluate_holdout_model(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    k: int,
    weights: str,
    model_id: str,
    model_name: str,
    n_boot: int = 1000,
    random_state: int = 42,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit BaselineEuclideanPreprocessor + KNeighborsClassifier on train and evaluate on hold-out test."""
    t_fit_start = time.perf_counter()
    prep = BaselineEuclideanPreprocessor()
    X_train_euc = prep.fit_transform(X_train_df)
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="euclidean", n_jobs=-1)
    clf.fit(X_train_euc, y_train)
    fit_sec = time.perf_counter() - t_fit_start

    t_pred_start = time.perf_counter()
    X_test_euc = prep.transform(X_test_df)
    y_prob = clf.predict_proba(X_test_euc)[:, 1]
    pred_sec = time.perf_counter() - t_pred_start

    metrics = compute_classification_metrics(y_test, y_prob, threshold=0.5)
    cis = bootstrap_confidence_intervals(
        y_test,
        y_prob,
        threshold=0.5,
        n_boot=n_boot,
        random_state=random_state,
    )

    result: dict[str, Any] = {
        "model_id": model_id,
        "model_name": model_name,
        "k": int(k),
        "weights": weights,
        "fit_sec": round(fit_sec, 4),
        "predict_sec": round(pred_sec, 4),
        **metrics,
    }
    for m_key in METRIC_KEYS:
        lo, hi = cis[m_key]
        result[f"{m_key}_ci95_low"] = lo
        result[f"{m_key}_ci95_high"] = hi
    return result, y_prob
