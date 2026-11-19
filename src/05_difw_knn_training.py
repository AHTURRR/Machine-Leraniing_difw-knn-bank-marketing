"""Stage 05: Proposed DIFW-KNN Development, Component Validation, and 5-Fold CV Training.

Runs strictly on Training / Development Data (36,168 observations) and keeps
Hold-out Test Data (9,043 observations) untouched for final evaluation:
1. Executes 8 component validation checks (distance axioms, simplex constraints,
   balanced triplet validity, analytic-vs-numeric gradient check, loss convergence,
   and 5-seed weight stability).
2. Runs leak-free Stratified 5-Fold CV hyperparameter tuning over Mixed-Type Uniform
   (E2/E4) and Triplet Feature-Weighted Mixed-Type (E3/E5) across (margin, reg_lambda, k, q).
3. Evaluates cost-sensitive threshold trade-offs across ratios C_FN:C_FP in
   {1:1, 2:1, 3:1, 4:1, 5:1, 7.55:1} on 5-fold out-of-fold predictions.
4. Trains final DIFW-KNN feature weights on the full 36,168 training set using the
   CV-optimal hyperparameters and saves tables, plots, and Laporan_Training_DIFW_KNN.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_knn import METRIC_KEYS, compute_classification_metrics
from src.difw_knn import (
    TripletFeatureWeightLearner,
    compute_expected_cost,
    cost_threshold_from_ratio,
    predict_local_probability,
    query_mixed_knn_neighbors,
    sample_balanced_triplets,
)
from src.preprocessing import (
    DOMAIN_FEATURE_COLS,
    MixedTypeEncoded,
    MixedTypePreprocessor,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "baseline_knn"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "difw_knn"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Training_DIFW_KNN.md"

RANDOM_STATE = 42
STABILITY_SEEDS = [42, 100, 2026, 7, 999]
N_TRIPLETS = 30000
N_EPOCHS = 220
LR = 0.20

MARGIN_GRID = [0.05, 0.10, 0.20]
LAMBDA_GRID = [0.05, 0.15, 0.30]
K_GRID = [11, 21, 31, 51, 71]
Q_GRID = [0.0, 1.0, 2.0]

COST_RATIOS = [1.0, 2.0, 3.0, 4.0, 5.0, 7.55]
PRIMARY_COST_RATIO = 5.0


def configure_plot_style() -> None:
    """Configure clean matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
        }
    )


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render DataFrame as GitHub-flavored Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_component_validation(
    X_tr_fold: pd.DataFrame,
    y_tr_fold: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all 8 component validation tests and 5-seed weight stability analysis."""
    prep = MixedTypePreprocessor().fit(X_tr_fold)
    X_enc = prep.transform(X_tr_fold)

    rng = np.random.default_rng(RANDOM_STATE)
    idx_sub = rng.choice(X_enc.n_samples, size=180, replace=False)
    sub_enc = MixedTypeEncoded(
        numeric=X_enc.numeric[idx_sub],
        categorical=X_enc.categorical[idx_sub],
        feature_names=X_enc.feature_names,
    )
    d_pair = prep.compute_pairwise_feature_distances(sub_enc, sub_enc)

    # 1. Self-distance == 0
    diag_max = float(np.max(np.abs(np.diagonal(d_pair, axis1=0, axis2=1))))
    c1_pass = diag_max == 0.0

    # 2. Symmetry
    sym_max = float(np.max(np.abs(d_pair - np.transpose(d_pair, (1, 0, 2)))))
    c2_pass = sym_max <= 1e-12

    # 3. Distance bounds [0, 1]
    d_min, d_max = float(d_pair.min()), float(d_pair.max())
    c3_pass = (d_min >= 0.0) and (d_max <= 1.0)

    # 5. Balanced triplet sampling check
    triplets = sample_balanced_triplets(prep, X_enc, y_tr_fold, n_triplets=N_TRIPLETS, random_state=RANDOM_STATE)
    pos_anchor_frac = float(np.mean(triplets.anchor_labels == 1))
    same_pos_ok = bool(np.all(y_tr_fold[triplets.anchor_idx] == y_tr_fold[triplets.positive_idx]))
    not_self_ok = bool(np.all(triplets.anchor_idx != triplets.positive_idx))
    diff_neg_ok = bool(np.all(y_tr_fold[triplets.anchor_idx] != y_tr_fold[triplets.negative_idx]))
    c5_pass = abs(pos_anchor_frac - 0.5) < 1e-6 and same_pos_ok and not_self_ok and diff_neg_ok

    # 6. Analytic vs numerical gradient check
    learner = TripletFeatureWeightLearner(margin=0.10, reg_lambda=0.15, lr=LR, n_epochs=N_EPOCHS)
    gc = learner.gradient_check(triplets.delta_d)
    c6_pass = bool(gc["passed"])

    # 7. Loss convergence & 4. Simplex constraints
    learner.fit(triplets.delta_d)
    w_fit = learner.weights_
    assert w_fit is not None
    w_min, w_sum = float(w_fit.min()), float(w_fit.sum())
    c4_pass = (w_min >= 0.0) and (abs(w_sum - 1.0) <= 1e-9)

    loss_start = float(learner.loss_history_[0]["total_loss"])
    loss_end = float(learner.loss_history_[-1]["total_loss"])
    c7_pass = loss_end < loss_start

    # 8. 5-seed stability test
    seed_weights: list[np.ndarray] = []
    for seed in STABILITY_SEEDS:
        tb = sample_balanced_triplets(prep, X_enc, y_tr_fold, n_triplets=N_TRIPLETS, random_state=seed)
        lrn = TripletFeatureWeightLearner(margin=0.10, reg_lambda=0.15, lr=LR, n_epochs=N_EPOCHS).fit(tb.delta_d)
        assert lrn.weights_ is not None
        seed_weights.append(lrn.weights_)

    w_mat = np.vstack(seed_weights)
    pairwise_corrs: list[float] = []
    for i in range(len(STABILITY_SEEDS)):
        for j in range(i + 1, len(STABILITY_SEEDS)):
            rho, _ = spearmanr(w_mat[i], w_mat[j])
            pairwise_corrs.append(float(rho))
    mean_spearman = float(np.mean(pairwise_corrs))
    max_w_std = float(np.max(np.std(w_mat, axis=0, ddof=1)))
    c8_pass = mean_spearman >= 0.90

    stability_rows = []
    for j, feat in enumerate(DOMAIN_FEATURE_COLS):
        col_w = w_mat[:, j]
        row_d: dict[str, Any] = {
            "feature": feat,
            "mean_weight": round(float(np.mean(col_w)), 5),
            "std_weight": round(float(np.std(col_w, ddof=1)), 5),
            "min_weight": round(float(np.min(col_w)), 5),
            "max_weight": round(float(np.max(col_w)), 5),
        }
        for s_idx, seed in enumerate(STABILITY_SEEDS):
            row_d[f"seed_{seed}"] = round(float(w_mat[s_idx, j]), 5)
        stability_rows.append(row_d)
    stability_df = pd.DataFrame(stability_rows).sort_values("mean_weight", ascending=False).reset_index(drop=True)

    checks = [
        {"check_id": 1, "component_check": "Self-distance d_j(x, x) = 0", "measured_value": f"max_diag={diag_max:.1e}", "status": "PASS" if c1_pass else "FAIL"},
        {"check_id": 2, "component_check": "Simetri jarak d_j(x, z) = d_j(z, x)", "measured_value": f"max_diff={sym_max:.1e}", "status": "PASS" if c2_pass else "FAIL"},
        {"check_id": 3, "component_check": "Batas jarak fitur d_j(x, z) dalam [0, 1]", "measured_value": f"[{d_min:.4f}, {d_max:.4f}]", "status": "PASS" if c3_pass else "FAIL"},
        {"check_id": 4, "component_check": "Kendala simpleks bobot (w_j >= 0, sum(w) = 1)", "measured_value": f"min={w_min:.4f}, sum={w_sum:.12f}", "status": "PASS" if c4_pass else "FAIL"},
        {"check_id": 5, "component_check": "Validitas Balanced Triplet Sampling (50:50)", "measured_value": f"pos_anchor={pos_anchor_frac:.3f}, labels_ok={same_pos_ok and not_self_ok and diff_neg_ok}", "status": "PASS" if c5_pass else "FAIL"},
        {"check_id": 6, "component_check": "Gradient check analitik vs finite-difference (<1e-7)", "measured_value": f"max_abs_err={gc['max_abs_error']:.3e}", "status": "PASS" if c6_pass else "FAIL"},
        {"check_id": 7, "component_check": "Penurunan Triplet Margin Loss saat training", "measured_value": f"{loss_start:.5f} -> {loss_end:.5f}", "status": "PASS" if c7_pass else "FAIL"},
        {"check_id": 8, "component_check": "Stabilitas bobot fitur lintas 5 random seed", "measured_value": f"mean_spearman={mean_spearman:.4f}, max_std={max_w_std:.5f}", "status": "PASS" if c8_pass else "FAIL"},
    ]
    checks_df = pd.DataFrame(checks)
    if (checks_df["status"] != "PASS").any():
        failed = checks_df.loc[checks_df["status"] != "PASS", "component_check"].tolist()
        raise AssertionError(f"Uji validasi komponen gagal pada: {failed}")

    return checks_df, stability_df


def run_difw_cv_tuning(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Run leak-free 5-fold CV for Uniform Mixed-Type (E2/E4) and Weighted DIFW-KNN (E3/E5)."""
    unique_folds = sorted(int(f) for f in np.unique(fold_ids))
    max_k = max(K_GRID)
    tau_primary = cost_threshold_from_ratio(PRIMARY_COST_RATIO)

    # Pre-encode each fold once (strictly training-only fit per fold)
    fold_cache: list[dict[str, Any]] = []
    for fid in unique_folds:
        val_mask = fold_ids == fid
        tr_mask = ~val_mask
        X_tr_f = X_train_df.loc[tr_mask].reset_index(drop=True)
        y_tr_f = y_train[tr_mask]
        X_va_f = X_train_df.loc[val_mask].reset_index(drop=True)
        y_va_f = y_train[val_mask]

        prep = MixedTypePreprocessor().fit(X_tr_f)
        X_tr_enc = prep.transform(X_tr_f)
        X_va_enc = prep.transform(X_va_f)
        triplets = sample_balanced_triplets(
            prep,
            X_tr_enc,
            y_tr_f,
            n_triplets=N_TRIPLETS,
            random_state=RANDOM_STATE + fid,
        )
        fold_cache.append(
            {
                "fold_id": fid,
                "val_idx": np.where(val_mask)[0],
                "prep": prep,
                "X_tr_enc": X_tr_enc,
                "y_tr": y_tr_f,
                "X_va_enc": X_va_enc,
                "y_va": y_va_f,
                "triplets": triplets,
            }
        )

    # 1. Mixed-Type Uniform (E2 & E4)
    uniform_rows: list[dict[str, Any]] = []
    uniform_oof_cache: dict[tuple[int, float], np.ndarray] = {
        (k, q): np.zeros(len(y_train), dtype=np.float64)
        for k in K_GRID
        for q in Q_GRID
    }

    for fc in fold_cache:
        t0 = time.perf_counter()
        dists, idxs = query_mixed_knn_neighbors(
            fc["prep"],
            fc["X_va_enc"],
            fc["X_tr_enc"],
            weights=None,
            max_k=max_k,
            batch_size=512,
        )
        q_sec = time.perf_counter() - t0
        for k in K_GRID:
            for q in Q_GRID:
                y_prob = predict_local_probability(fc["y_tr"], dists, idxs, k=k, q=q)
                uniform_oof_cache[(k, q)][fc["val_idx"]] = y_prob
                m_std = compute_classification_metrics(fc["y_va"], y_prob, threshold=0.5)
                m_cs = compute_classification_metrics(fc["y_va"], y_prob, threshold=tau_primary)
                uniform_rows.append(
                    {
                        "fold_id": fc["fold_id"],
                        "k": k,
                        "q": q,
                        "query_sec": round(q_sec, 4),
                        "pr_auc": m_std["pr_auc"],
                        "roc_auc": m_std["roc_auc"],
                        "e2_accuracy": m_std["accuracy"],
                        "e2_precision": m_std["precision_yes"],
                        "e2_recall": m_std["recall_yes"],
                        "e2_f1": m_std["f1_yes"],
                        "e2_mcc": m_std["mcc"],
                        "e2_gmean": m_std["g_mean"],
                        "e4_accuracy": m_cs["accuracy"],
                        "e4_precision": m_cs["precision_yes"],
                        "e4_recall": m_cs["recall_yes"],
                        "e4_f1": m_cs["f1_yes"],
                        "e4_mcc": m_cs["mcc"],
                        "e4_gmean": m_cs["g_mean"],
                    }
                )

    uniform_per_fold = pd.DataFrame(uniform_rows)
    uniform_agg_cols = [c for c in uniform_per_fold.columns if c not in {"fold_id", "k", "q", "query_sec"}]
    uniform_summary = (
        uniform_per_fold.groupby(["k", "q"], as_index=False)
        .agg(**{f"{c}_mean": (c, "mean") for c in uniform_agg_cols}, **{f"{c}_std": (c, "std") for c in uniform_agg_cols})
        .sort_values(["pr_auc_mean", "e4_f1_mean"], ascending=False)
        .reset_index(drop=True)
    )

    # 2. Triplet Feature-Weighted Mixed-Type (E3 & E5) across (margin, reg_lambda)
    weighted_rows: list[dict[str, Any]] = []
    oof_probs_by_combo: dict[tuple[float, float, int, float], np.ndarray] = {}
    learned_weights_by_combo: dict[tuple[float, float], list[np.ndarray]] = {}

    for margin in MARGIN_GRID:
        for reg_lam in LAMBDA_GRID:
            combo_w_list: list[np.ndarray] = []
            for k in K_GRID:
                for q in Q_GRID:
                    oof_probs_by_combo[(margin, reg_lam, k, q)] = np.zeros(len(y_train), dtype=np.float64)

            for fc in fold_cache:
                t0 = time.perf_counter()
                learner = TripletFeatureWeightLearner(
                    margin=margin,
                    reg_lambda=reg_lam,
                    lr=LR,
                    n_epochs=N_EPOCHS,
                ).fit(fc["triplets"].delta_d)
                assert learner.weights_ is not None
                combo_w_list.append(learner.weights_)
                fit_sec = time.perf_counter() - t0

                t1 = time.perf_counter()
                dists, idxs = query_mixed_knn_neighbors(
                    fc["prep"],
                    fc["X_va_enc"],
                    fc["X_tr_enc"],
                    weights=learner.weights_,
                    max_k=max_k,
                    batch_size=512,
                )
                q_sec = time.perf_counter() - t1

                for k in K_GRID:
                    for q in Q_GRID:
                        y_prob = predict_local_probability(fc["y_tr"], dists, idxs, k=k, q=q)
                        oof_probs_by_combo[(margin, reg_lam, k, q)][fc["val_idx"]] = y_prob
                        m_std = compute_classification_metrics(fc["y_va"], y_prob, threshold=0.5)
                        m_cs = compute_classification_metrics(fc["y_va"], y_prob, threshold=tau_primary)
                        weighted_rows.append(
                            {
                                "fold_id": fc["fold_id"],
                                "margin": margin,
                                "reg_lambda": reg_lam,
                                "k": k,
                                "q": q,
                                "fit_sec": round(fit_sec, 4),
                                "query_sec": round(q_sec, 4),
                                "pr_auc": m_std["pr_auc"],
                                "roc_auc": m_std["roc_auc"],
                                "e3_accuracy": m_std["accuracy"],
                                "e3_precision": m_std["precision_yes"],
                                "e3_recall": m_std["recall_yes"],
                                "e3_f1": m_std["f1_yes"],
                                "e3_mcc": m_std["mcc"],
                                "e3_gmean": m_std["g_mean"],
                                "e5_accuracy": m_cs["accuracy"],
                                "e5_precision": m_cs["precision_yes"],
                                "e5_recall": m_cs["recall_yes"],
                                "e5_f1": m_cs["f1_yes"],
                                "e5_mcc": m_cs["mcc"],
                                "e5_gmean": m_cs["g_mean"],
                            }
                        )
            learned_weights_by_combo[(margin, reg_lam)] = combo_w_list

    weighted_per_fold = pd.DataFrame(weighted_rows)
    w_agg_cols = [c for c in weighted_per_fold.columns if c not in {"fold_id", "margin", "reg_lambda", "k", "q", "fit_sec", "query_sec"}]
    weighted_summary = (
        weighted_per_fold.groupby(["margin", "reg_lambda", "k", "q"], as_index=False)
        .agg(**{f"{c}_mean": (c, "mean") for c in w_agg_cols}, **{f"{c}_std": (c, "std") for c in w_agg_cols})
        .sort_values(["pr_auc_mean", "e5_mcc_mean", "e5_f1_mean"], ascending=False)
        .reset_index(drop=True)
    )

    best_u = uniform_summary.iloc[0]
    best_w = weighted_summary.iloc[0]
    best_oof = {
        "uniform": uniform_oof_cache[(int(best_u["k"]), float(best_u["q"]))],
        "weighted": oof_probs_by_combo[(float(best_w["margin"]), float(best_w["reg_lambda"]), int(best_w["k"]), float(best_w["q"]))],
        "cv_mean_weights": np.mean(np.vstack(learned_weights_by_combo[(float(best_w["margin"]), float(best_w["reg_lambda"]))]), axis=0),
    }

    return uniform_summary, weighted_summary, weighted_per_fold, best_oof


def analyze_cost_tradeoffs(
    y_train: np.ndarray,
    oof_uniform: np.ndarray,
    oof_weighted: np.ndarray,
) -> pd.DataFrame:
    """Evaluate precision/recall/F1/MCC/G-Mean and expected cost across cost ratios."""
    rows: list[dict[str, Any]] = []
    for variant, y_prob in [("E4: Cost-Sensitive Mixed KNN (Uniform)", oof_uniform), ("E5: Full DIFW-KNN (Weighted)", oof_weighted)]:
        for ratio in COST_RATIOS:
            tau = cost_threshold_from_ratio(ratio)
            m = compute_classification_metrics(y_train, y_prob, threshold=tau)
            # Compute expected cost under the scenario ratio itself AND under default 5:1 cost scenario
            c_scen = compute_expected_cost(y_train, y_prob, cost_fn=ratio, cost_fp=1.0, threshold=tau)
            c_ref5 = compute_expected_cost(y_train, y_prob, cost_fn=PRIMARY_COST_RATIO, cost_fp=1.0, threshold=tau)
            rows.append(
                {
                    "model_variant": variant,
                    "cost_ratio_fn_fp": ratio,
                    "threshold_tau": round(tau, 4),
                    "accuracy": round(m["accuracy"], 4),
                    "precision_yes": round(m["precision_yes"], 4),
                    "recall_yes": round(m["recall_yes"], 4),
                    "f1_yes": round(m["f1_yes"], 4),
                    "mcc": round(m["mcc"], 4),
                    "g_mean": round(m["g_mean"], 4),
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "avg_cost_scenario": round(c_scen["avg_cost_per_sample"], 4),
                    "avg_cost_at_5_to_1": round(c_ref5["avg_cost_per_sample"], 4),
                }
            )
    return pd.DataFrame(rows)


def create_training_figures(
    loss_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> list[Path]:
    """Generate 3 publication-quality training & CV figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Triplet Margin Loss convergence & Learned Feature Weights
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(loss_df["epoch"], loss_df["total_loss"], color="#4C72B0", linewidth=2.2, label="Total Loss L(w)")
    axes[0].plot(loss_df["epoch"], loss_df["hinge_loss"], color="#DD8452", linestyle="--", linewidth=1.8, label="Triplet Hinge Loss")
    axes[0].plot(loss_df["epoch"], loss_df["reg_loss"], color="#55A868", linestyle=":", linewidth=1.8, label="Regularisasi L2 ||w - u||^2")
    axes[0].set_title("Konvergensi Triplet Margin Loss (Full Training 36,168)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Nilai Objective Loss")
    axes[0].legend(fontsize=9)

    w_plot = weights_df.sort_values("weight_full_train", ascending=True)
    u_val = 1.0 / len(DOMAIN_FEATURE_COLS)
    bar_cols = ["#55A868" if v >= u_val else "#8C96A5" for v in w_plot["weight_full_train"]]
    bars = axes[1].barh(w_plot["feature"], w_plot["weight_full_train"], color=bar_cols)
    axes[1].axvline(u_val, color="#D62728", linestyle="--", linewidth=1.3, label=f"Bobot Seragam u = 1/16 ({u_val:.4f})")
    axes[1].set_title("Bobot Fitur Hasil Triplet Weight Learning (DIFW-KNN)")
    axes[1].set_xlabel("Bobot Fitur w_j (sum = 1)")
    axes[1].set_xlim(0, max(w_plot["weight_full_train"].max() * 1.24, 0.25))
    axes[1].legend(fontsize=8.5, loc="lower right")
    for bar, val in zip(bars, w_plot["weight_full_train"]):
        axes[1].text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8.5)

    fig.tight_layout()
    p1 = FIGURES_DIR / "01_triplet_loss_and_learned_weights.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2. 5-Fold CV Ablation Progression E0 -> E5
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    models = ablation_df["model_id"].tolist()
    x = np.arange(len(models))
    w_bar = 0.24

    axes[0].bar(x - w_bar, ablation_df["f1_yes"], width=w_bar, color="#4C72B0", label="F1 (yes)")
    axes[0].bar(x, ablation_df["mcc"], width=w_bar, color="#55A868", label="MCC")
    axes[0].bar(x + w_bar, ablation_df["g_mean"], width=w_bar, color="#DD8452", label="G-Mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 0.85)
    axes[0].set_title("Progresi Metrik Klasifikasi 5-Fold CV (E0 -> E5)")
    axes[0].set_ylabel("Skor Rata-Rata 5-Fold CV")
    axes[0].legend(fontsize=9)
    for idx_m, row in ablation_df.iterrows():
        axes[0].text(idx_m - w_bar, row["f1_yes"] + 0.012, f"{row['f1_yes']:.3f}", ha="center", fontsize=7.5, rotation=90)
        axes[0].text(idx_m, row["mcc"] + 0.012, f"{row['mcc']:.3f}", ha="center", fontsize=7.5, rotation=90)
        axes[0].text(idx_m + w_bar, row["g_mean"] + 0.012, f"{row['g_mean']:.3f}", ha="center", fontsize=7.5, rotation=90)

    axes[1].bar(x - 0.18, ablation_df["pr_auc"], width=0.36, color="#8172B3", label="PR-AUC")
    axes[1].bar(x + 0.18, ablation_df["roc_auc"], width=0.36, color="#CCB974", label="ROC-AUC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0, 0.92)
    axes[1].set_title("Progresi Kualitas Ranking Probabilitas (PR-AUC & ROC-AUC)")
    axes[1].set_ylabel("Skor Rata-Rata 5-Fold CV")
    axes[1].legend(fontsize=9)
    for idx_m, row in ablation_df.iterrows():
        axes[1].text(idx_m - 0.18, row["pr_auc"] + 0.012, f"{row['pr_auc']:.3f}", ha="center", fontsize=8)
        axes[1].text(idx_m + 0.18, row["roc_auc"] + 0.012, f"{row['roc_auc']:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_cv_ablation_progression_e0_to_e5.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # 3. Cost-Sensitive Threshold Trade-off
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    e5_cost = cost_df[cost_df["model_variant"].str.startswith("E5")].sort_values("cost_ratio_fn_fp")
    e4_cost = cost_df[cost_df["model_variant"].str.startswith("E4")].sort_values("cost_ratio_fn_fp")

    ratios = e5_cost["cost_ratio_fn_fp"].to_numpy()
    axes[0].plot(ratios, e5_cost["precision_yes"], marker="o", linewidth=2, color="#4C72B0", label="E5 Precision (yes)")
    axes[0].plot(ratios, e5_cost["recall_yes"], marker="s", linewidth=2, color="#DD8452", label="E5 Recall (yes)")
    axes[0].plot(ratios, e5_cost["f1_yes"], marker="^", linewidth=2, color="#55A868", label="E5 F1 (yes)")
    axes[0].plot(ratios, e5_cost["mcc"], marker="d", linewidth=2, color="#8172B3", label="E5 MCC")
    axes[0].axvline(PRIMARY_COST_RATIO, color="#333333", linestyle="--", linewidth=1.2, label="Rasio Default 5:1 (tau=0.1667)")
    axes[0].set_title("Trade-off Metrik DIFW-KNN (E5) per Rasio Biaya C_FN : C_FP")
    axes[0].set_xlabel("Rasio Biaya C_FN / C_FP")
    axes[0].set_ylabel("Skor Out-of-Fold CV")
    axes[0].set_xticks(ratios)
    axes[0].legend(fontsize=8.2)

    axes[1].plot(ratios, e4_cost["avg_cost_at_5_to_1"], marker="o", linestyle="--", linewidth=2, color="#7F7F7F", label="E4 (Mixed Uniform) Avg Cost @ 5:1")
    axes[1].plot(ratios, e5_cost["avg_cost_at_5_to_1"], marker="s", linewidth=2.2, color="#C44E52", label="E5 (Full DIFW-KNN) Avg Cost @ 5:1")
    axes[1].axvline(PRIMARY_COST_RATIO, color="#333333", linestyle="--", linewidth=1.2, label="Threshold Optimal Bayes (5:1)")
    axes[1].set_title("Biaya Kesalahan per Nasabah (Skenario Referensi 5:1)")
    axes[1].set_xlabel("Rasio Biaya Desain Threshold (C_FN / C_FP)")
    axes[1].set_ylabel("Rata-Rata Biaya per Nasabah")
    axes[1].set_xticks(ratios)
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    p3 = FIGURES_DIR / "03_cost_sensitive_threshold_tradeoff.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    return saved


def build_report(
    checks_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    uniform_summary: pd.DataFrame,
    weighted_summary: pd.DataFrame,
    weights_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    best_params: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    """Build Laporan_Training_DIFW_KNN.md."""
    top_weighted = weighted_summary.head(10).round(4)[
        ["margin", "reg_lambda", "k", "q", "pr_auc_mean", "roc_auc_mean", "e3_f1_mean", "e3_mcc_mean", "e5_f1_mean", "e5_mcc_mean", "e5_gmean_mean"]
    ]
    top_uniform = uniform_summary.head(8).round(4)[
        ["k", "q", "pr_auc_mean", "roc_auc_mean", "e2_f1_mean", "e2_mcc_mean", "e4_f1_mean", "e4_mcc_mean", "e4_gmean_mean"]
    ]

    lines = [
        "# LAPORAN TAHAP 05 — PELATIHAN PROPOSED `DIFW-KNN` (TRAINING / DEVELOPMENT DATA)",
        "",
        "**Modul Inti:** [`src/difw_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/difw_knn.py), [`src/05_difw_knn_training.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/05_difw_knn_training.py)  ",
        "**Notebook:** [`notebooks/05_Proposed_DIFW_KNN_Training.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/05_Proposed_DIFW_KNN_Training.ipynb)  ",
        "**Protokol Data:** Seluruh pelatihan, validasi komponen, dan tuning hyperparameter dilakukan **eksklusif pada `Training / Development Data` (`36.168` baris)**; `Hold-out Test Data` (`9.043` baris) tetap dikunci untuk tahap evaluasi akhir.",
        "",
        "---",
        "",
        "## 1. Uji Validasi Komponen & Stabilitas Bobot (Bab 6 Desain Penelitian)",
        "",
        df_to_markdown_table(checks_df),
        "",
        "### Stabilitas Bobot Fitur Lintas 5 Random Seed (`[42, 100, 2026, 7, 999]`)",
        "",
        df_to_markdown_table(stability_df[["feature", "mean_weight", "std_weight", "min_weight", "max_weight"]]),
        "",
        "---",
        "",
        "## 2. Hasil Hyperparameter Tuning 5-Fold CV (`E2`/`E4` vs `E3`/`E5`)",
        "",
        f"- **Konfigurasi Optimal Mixed-Type Uniform (`E2`/`E4`):** `k = {best_params['uniform']['k']}`, `q = {best_params['uniform']['q']}`.",
        f"- **Konfigurasi Optimal Full DIFW-KNN (`E3`/`E5`):** `margin = {best_params['difw']['margin']}`, `reg_lambda = {best_params['difw']['reg_lambda']}`, `k = {best_params['difw']['k']}`, `q = {best_params['difw']['q']}`.",
        "",
        "### Top 8 Konfigurasi Mixed-Type Uniform (`E2` & `E4`)",
        "",
        df_to_markdown_table(top_uniform),
        "",
        "### Top 10 Konfigurasi Triplet Feature-Weighted (`E3` & `E5`)",
        "",
        df_to_markdown_table(top_weighted),
        "",
        "---",
        "",
        "## 3. Bobot Fitur Terpelajar pada Full Training Set (`36.168` Baris)",
        "",
        df_to_markdown_table(weights_df),
        "",
        "---",
        "",
        "## 4. Progresi Ablation Study pada 5-Fold Cross-Validation (`E0` $\\rightarrow$ `E5`)",
        "",
        "*Catatan: `E4` dan `E5` dievaluasi pada rasio biaya default $C_{FN}:C_{FP} = 5:1$ ($\\tau = 1/6 \\approx 0.1667$).*",
        "",
        df_to_markdown_table(ablation_df.round(4)),
        "",
        "---",
        "",
        "## 5. Analisis Trade-off Rasio Biaya ($C_{FN}:C_{FP}$) pada Out-of-Fold CV",
        "",
        df_to_markdown_table(cost_df),
        "",
        "---",
        "",
        "## 6. Artefak Visual",
        "",
    ]
    for p in figure_paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"- [{p.name}](file:///{p.as_posix()}) (`{rel}`)")
    return "\n".join(lines)


def main() -> None:
    """Execute Stage 05 component checks, 5-fold CV tuning, and final weight training."""
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    folds_df = pd.read_csv(PROCESSED_DIR / "train_cv_folds.csv")
    baseline_cv = pd.read_csv(BASELINE_DIR / "cv_grid_results.csv")

    X_train_df = train_df.drop(columns=["y"])
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    fold_ids = folds_df["fold_id"].to_numpy(dtype=np.int64)

    # 1. Component Validation & 5-Seed Stability on Fold 0 training split
    tr0_mask = fold_ids != 0
    checks_df, stability_df = run_component_validation(
        X_train_df.loc[tr0_mask].reset_index(drop=True),
        y_train[tr0_mask],
    )
    checks_df.to_csv(OUTPUT_DIR / "component_validation_checks.csv", index=False)
    stability_df.to_csv(OUTPUT_DIR / "weight_seed_stability.csv", index=False)

    # 2. Leak-free 5-Fold CV Hyperparameter Tuning
    uniform_summary, weighted_summary, weighted_per_fold, best_oof = run_difw_cv_tuning(
        X_train_df=X_train_df,
        y_train=y_train,
        fold_ids=fold_ids,
    )
    uniform_summary.to_csv(OUTPUT_DIR / "cv_uniform_grid_results.csv", index=False)
    weighted_summary.to_csv(OUTPUT_DIR / "cv_triplet_grid_results.csv", index=False)
    weighted_per_fold.to_csv(OUTPUT_DIR / "cv_triplet_per_fold_results.csv", index=False)

    best_u = uniform_summary.iloc[0]
    best_w = weighted_summary.iloc[0]
    best_params = {
        "uniform": {"k": int(best_u["k"]), "q": float(best_u["q"])},
        "difw": {
            "margin": float(best_w["margin"]),
            "reg_lambda": float(best_w["reg_lambda"]),
            "k": int(best_w["k"]),
            "q": float(best_w["q"]),
            "cost_ratio_fn_fp": PRIMARY_COST_RATIO,
            "threshold_tau": round(cost_threshold_from_ratio(PRIMARY_COST_RATIO), 6),
        },
        "n_triplets": N_TRIPLETS,
        "n_epochs": N_EPOCHS,
        "lr": LR,
    }

    # Check hypothesis criterion on CV before final full-train weight fit
    e1_best_cv = baseline_cv.sort_values(["f1_yes_mean", "mcc_mean"], ascending=False).iloc[0]
    e1_pr_cv = baseline_cv.sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False).iloc[0]
    if float(best_w["pr_auc_mean"]) <= float(e1_pr_cv["pr_auc_mean"]):
        raise RuntimeError(
            f"CV PR-AUC DIFW-KNN ({best_w['pr_auc_mean']:.4f}) tidak melebihi Euclidean terbaik ({e1_pr_cv['pr_auc_mean']:.4f})."
        )

    # 3. Cost-sensitive threshold trade-off on 5-fold OOF predictions
    cost_df = analyze_cost_tradeoffs(
        y_train=y_train,
        oof_uniform=best_oof["uniform"],
        oof_weighted=best_oof["weighted"],
    )
    cost_df.to_csv(OUTPUT_DIR / "cost_tradeoff_oof_results.csv", index=False)

    # 4. Fit final Triplet Feature Weights on full 36,168 training set
    t_full0 = time.perf_counter()
    full_prep = MixedTypePreprocessor().fit(X_train_df)
    X_train_enc_full = full_prep.transform(X_train_df)
    full_triplets = sample_balanced_triplets(
        full_prep,
        X_train_enc_full,
        y_train,
        n_triplets=N_TRIPLETS,
        random_state=RANDOM_STATE,
    )
    full_learner = TripletFeatureWeightLearner(
        margin=best_params["difw"]["margin"],
        reg_lambda=best_params["difw"]["reg_lambda"],
        lr=LR,
        n_epochs=N_EPOCHS,
    ).fit(full_triplets.delta_d)
    full_fit_sec = time.perf_counter() - t_full0
    assert full_learner.weights_ is not None
    best_params["full_train_fit_sec"] = round(full_fit_sec, 4)

    (OUTPUT_DIR / "best_hyperparameters.json").write_text(
        json.dumps(best_params, indent=2),
        encoding="utf-8",
    )

    loss_df = pd.DataFrame(full_learner.loss_history_)
    loss_df.to_csv(OUTPUT_DIR / "triplet_loss_history.csv", index=False)

    u_val = 1.0 / len(DOMAIN_FEATURE_COLS)
    weights_df = pd.DataFrame(
        {
            "feature": DOMAIN_FEATURE_COLS,
            "weight_full_train": np.round(full_learner.weights_, 6),
            "weight_cv_5fold_mean": np.round(best_oof["cv_mean_weights"], 6),
            "uniform_prior": round(u_val, 6),
            "ratio_vs_uniform": np.round(full_learner.weights_ / u_val, 3),
        }
    ).sort_values("weight_full_train", ascending=False).reset_index(drop=True)
    weights_df.to_csv(OUTPUT_DIR / "learned_feature_weights_full_train.csv", index=False)

    # 5. Build 5-Fold CV Ablation Progression table (E0 -> E5)
    e0_cv = baseline_cv[(baseline_cv["k"] == 5) & (baseline_cv["weights"] == "uniform")].iloc[0]
    ablation_rows = [
        {
            "model_id": "E0",
            "model_name": "Standard KNN (k=5, uniform)",
            "accuracy": e0_cv["accuracy_mean"],
            "precision_yes": e0_cv["precision_yes_mean"],
            "recall_yes": e0_cv["recall_yes_mean"],
            "f1_yes": e0_cv["f1_yes_mean"],
            "pr_auc": e0_cv["pr_auc_mean"],
            "roc_auc": e0_cv["roc_auc_mean"],
            "mcc": e0_cv["mcc_mean"],
            "g_mean": e0_cv["g_mean_mean"],
        },
        {
            "model_id": "E1",
            "model_name": f"Optimized KNN (k={int(e1_best_cv['k'])}, {e1_best_cv['weights']})",
            "accuracy": e1_best_cv["accuracy_mean"],
            "precision_yes": e1_best_cv["precision_yes_mean"],
            "recall_yes": e1_best_cv["recall_yes_mean"],
            "f1_yes": e1_best_cv["f1_yes_mean"],
            "pr_auc": e1_best_cv["pr_auc_mean"],
            "roc_auc": e1_best_cv["roc_auc_mean"],
            "mcc": e1_best_cv["mcc_mean"],
            "g_mean": e1_best_cv["g_mean_mean"],
        },
        {
            "model_id": "E2",
            "model_name": f"Mixed-Type KNN (k={int(best_u['k'])}, q={best_u['q']})",
            "accuracy": best_u["e2_accuracy_mean"],
            "precision_yes": best_u["e2_precision_mean"],
            "recall_yes": best_u["e2_recall_mean"],
            "f1_yes": best_u["e2_f1_mean"],
            "pr_auc": best_u["pr_auc_mean"],
            "roc_auc": best_u["roc_auc_mean"],
            "mcc": best_u["e2_mcc_mean"],
            "g_mean": best_u["e2_gmean_mean"],
        },
        {
            "model_id": "E3",
            "model_name": f"Feature-Weighted KNN (m={best_w['margin']}, lam={best_w['reg_lambda']}, k={int(best_w['k'])}, q={best_w['q']})",
            "accuracy": best_w["e3_accuracy_mean"],
            "precision_yes": best_w["e3_precision_mean"],
            "recall_yes": best_w["e3_recall_mean"],
            "f1_yes": best_w["e3_f1_mean"],
            "pr_auc": best_w["pr_auc_mean"],
            "roc_auc": best_w["roc_auc_mean"],
            "mcc": best_w["e3_mcc_mean"],
            "g_mean": best_w["e3_gmean_mean"],
        },
        {
            "model_id": "E4",
            "model_name": f"Cost-Sensitive Mixed KNN (k={int(best_u['k'])}, q={best_u['q']}, 5:1)",
            "accuracy": best_u["e4_accuracy_mean"],
            "precision_yes": best_u["e4_precision_mean"],
            "recall_yes": best_u["e4_recall_mean"],
            "f1_yes": best_u["e4_f1_mean"],
            "pr_auc": best_u["pr_auc_mean"],
            "roc_auc": best_u["roc_auc_mean"],
            "mcc": best_u["e4_mcc_mean"],
            "g_mean": best_u["e4_gmean_mean"],
        },
        {
            "model_id": "E5",
            "model_name": f"DIFW-KNN Lengkap (m={best_w['margin']}, lam={best_w['reg_lambda']}, k={int(best_w['k'])}, q={best_w['q']}, 5:1)",
            "accuracy": best_w["e5_accuracy_mean"],
            "precision_yes": best_w["e5_precision_mean"],
            "recall_yes": best_w["e5_recall_mean"],
            "f1_yes": best_w["e5_f1_mean"],
            "pr_auc": best_w["pr_auc_mean"],
            "roc_auc": best_w["roc_auc_mean"],
            "mcc": best_w["e5_mcc_mean"],
            "g_mean": best_w["e5_gmean_mean"],
        },
    ]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUTPUT_DIR / "cv_ablation_progression.csv", index=False)

    figure_paths = create_training_figures(
        loss_df=loss_df,
        weights_df=weights_df,
        ablation_df=ablation_df,
        cost_df=cost_df,
    )

    report = build_report(
        checks_df=checks_df,
        stability_df=stability_df,
        uniform_summary=uniform_summary,
        weighted_summary=weighted_summary,
        weights_df=weights_df,
        ablation_df=ablation_df,
        cost_df=cost_df,
        best_params=best_params,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Tahap 05 (Proposed DIFW-KNN Training & 5-Fold CV) selesai.")
    print(f"Best Uniform (E2/E4): k={best_params['uniform']['k']}, q={best_params['uniform']['q']}")
    print(f"Best DIFW-KNN (E3/E5): margin={best_params['difw']['margin']}, reg_lambda={best_params['difw']['reg_lambda']}, k={best_params['difw']['k']}, q={best_params['difw']['q']}")
    print(ablation_df[["model_id", "accuracy", "precision_yes", "recall_yes", "f1_yes", "pr_auc", "roc_auc", "mcc", "g_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
