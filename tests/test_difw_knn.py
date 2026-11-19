"""Unit and integration tests for Developer 3: Cost-Sensitive Rule & DIFW-KNN."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cost_sensitive_decision import (
    CostSensitiveDecisionRule,
    compute_optimal_threshold,
    evaluate_cost_ratio_grid,
)
from src.difw_knn import DIFWKNNClassifier, evaluate_ablation_suite


def _make_bank_like_dataset(n_samples: int = 300, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_pos = int(round(n_samples * 0.2))
    y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_samples - n_pos, dtype=np.int64)])
    y = y[rng.permutation(n_samples)]

    pdays = np.where(
        y == 1,
        rng.choice([-1, 30, 90, 180], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        rng.choice([-1, 180, 360], size=n_samples, p=[0.85, 0.1, 0.05]),
    )
    poutcome = np.where(
        pdays > -1,
        np.where(y == 1, "success", "failure"),
        "unknown",
    )
    contact = np.where(
        y == 1,
        rng.choice(["cellular", "telephone"], size=n_samples, p=[0.85, 0.15]),
        rng.choice(["cellular", "unknown"], size=n_samples, p=[0.5, 0.5]),
    )

    df = pd.DataFrame(
        {
            "age": np.where(y == 1, rng.integers(30, 68, n_samples), rng.integers(22, 55, n_samples)),
            "balance": np.where(y == 1, rng.normal(2400, 800, n_samples), rng.normal(400, 600, n_samples)),
            "day": rng.integers(1, 29, n_samples),
            "campaign": np.where(y == 1, rng.integers(1, 3, n_samples), rng.integers(1, 8, n_samples)),
            "pdays": pdays,
            "previous": np.where(pdays > -1, rng.integers(1, 5, n_samples), 0),
            "job": rng.choice(["admin.", "technician", "management", "services", "retired"], n_samples),
            "marital": rng.choice(["single", "married", "divorced"], n_samples),
            "education": rng.choice(["primary", "secondary", "tertiary"], n_samples),
            "default": rng.choice(["no", "yes"], n_samples, p=[0.95, 0.05]),
            "housing": rng.choice(["yes", "no"], n_samples),
            "loan": rng.choice(["no", "yes"], n_samples, p=[0.85, 0.15]),
            "contact": contact,
            "month": np.where(
                y == 1,
                rng.choice(["mar", "sep", "oct"], n_samples),
                rng.choice(["may", "jun", "jul", "aug"], n_samples),
            ),
            "poutcome": poutcome,
        }
    )
    return df, y


def test_cost_sensitive_decision_rule_threshold_and_cost() -> None:
    tau = compute_optimal_threshold(c_fp=1.0, c_fn=4.0)
    assert tau == pytest.approx(0.2)

    rule = CostSensitiveDecisionRule(c_fp=1.0, c_fn=4.0)
    assert rule.threshold == pytest.approx(0.2)

    y_true = np.array([1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.35, 0.10, 0.25, 0.05, 0.19, 0.80])
    # Predictions at tau=0.20: [1, 0, 1, 0, 0, 1] -> TP=2, FN=1, FP=1, TN=2
    preds = rule.predict(y_prob)
    assert np.array_equal(preds, np.array([1, 0, 1, 0, 0, 1]))

    metrics = rule.evaluate(y_true, y_prob)
    assert metrics["tp"] == 2
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 2
    assert metrics["total_cost"] == pytest.approx(1.0 * 1 + 4.0 * 1)

    grid_df = evaluate_cost_ratio_grid(y_true, y_prob, c_fn_grid=[1.0, 4.0, 9.0])
    assert len(grid_df) == 3
    assert list(grid_df["threshold"]) == pytest.approx([0.5, 0.2, 0.1])


@pytest.mark.parametrize(
    "learn_weights,use_cost",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_difw_knn_variants_predict_and_proba(learn_weights: bool, use_cost: bool) -> None:
    df, y = _make_bank_like_dataset(n_samples=180, seed=11)
    X_tr, X_te = df.iloc[:130].reset_index(drop=True), df.iloc[130:].reset_index(drop=True)
    y_tr, y_te = y[:130], y[130:]

    clf = DIFWKNNClassifier(
        k=7,
        learn_feature_weights=learn_weights,
        use_cost_sensitive=use_cost,
        c_fp=1.0,
        c_fn=4.0,
        n_triplets=300,
        max_iter=80,
        batch_size=16,
        random_state=42,
    ).fit(X_tr, y_tr)

    dists, idxs = clf.kneighbors(X_te)
    assert dists.shape == (len(X_te), 7)
    assert idxs.shape == (len(X_te), 7)
    # Distances per row are sorted ascending
    assert np.all(np.diff(dists, axis=1) >= -1e-12)

    probs = clf.predict_proba(X_te)
    assert probs.shape == (len(X_te), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)

    preds = clf.predict(X_te)
    assert preds.shape == (len(X_te),)
    assert set(np.unique(preds)).issubset({0, 1})

    eval_dict = clf.evaluate(X_te, y_te)
    assert "f1_yes" in eval_dict and "total_cost" in eval_dict


def test_ablation_suite_end_to_end() -> None:
    df, y = _make_bank_like_dataset(n_samples=260, seed=99)
    X_tr, X_te = df.iloc[:190].reset_index(drop=True), df.iloc[190:].reset_index(drop=True)
    y_tr, y_te = y[:190], y[190:]

    results_df, artifacts = evaluate_ablation_suite(
        X_train_df=X_tr,
        y_train=y_tr,
        X_test_df=X_te,
        y_test=y_te,
        k=9,
        c_fp=1.0,
        c_fn=5.0,
        n_triplets=400,
        random_state=42,
        batch_size=32,
    )

    assert list(results_df["model_id"]) == ["E2", "E3", "E4", "E5"]
    for key in (
        "model_e2",
        "model_e3",
        "model_e4",
        "model_e5",
        "y_prob_uniform",
        "y_prob_weighted",
        "weights_table",
        "loss_history",
    ):
        assert key in artifacts

    e2_row = results_df.loc[results_df["model_id"] == "E2"].iloc[0]
    e5_row = results_df.loc[results_df["model_id"] == "E5"].iloc[0]
    assert e5_row["recall_yes"] >= e2_row["recall_yes"]
    assert e5_row["total_cost"] <= e2_row["total_cost"]
