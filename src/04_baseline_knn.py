"""Stage 04: Baseline KNN Evaluation (E0 Standard KNN & E1 Optimized KNN).

Executes:
1. Unit check verifying exact probability equivalence against KNeighborsClassifier.
2. Leak-free Stratified 5-Fold CV grid search on Training/Development data
   across k in {3, 5, 7, 11, 15, 21, 31, 51, 71, 101} and weights in {'uniform', 'distance'}.
3. Final training on full 36,168 training set and evaluation on 9,043 Hold-out Test
   for E0 (Standard KNN: k=5, uniform) and E1 (Optimized KNN), including 95% bootstrap CIs.
4. Generates figures under outputs/baseline_knn/figures/ and writes
   Laporan_Baseline_KNN_DIFW_KNN.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_knn import (
    METRIC_KEYS,
    evaluate_holdout_model,
    run_stratified_cv_grid,
    verify_knn_probability_equivalence,
)
from src.preprocessing import BaselineEuclideanPreprocessor


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_knn"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Baseline_KNN_DIFW_KNN.md"

K_GRID = [3, 5, 7, 11, 15, 21, 31, 51, 71, 101]
WEIGHTS_GRID = ["uniform", "distance"]
RANDOM_STATE = 42
N_BOOTSTRAP = 1000


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


def create_baseline_figures(
    cv_summary: pd.DataFrame,
    best_f1_row: pd.Series,
    best_prauc_row: pd.Series,
    holdout_results: list[dict[str, Any]],
    y_test: np.ndarray,
    probs_map: dict[str, np.ndarray],
) -> list[Path]:
    """Generate CV tuning curves, ROC/PR curves, and confusion matrix comparisons."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. CV hyperparameter tuning curves (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6))
    panels = [
        ("f1_yes", "5-Fold CV: F1-Score (yes)"),
        ("mcc", "5-Fold CV: Matthews Correlation Coefficient (MCC)"),
        ("pr_auc", "5-Fold CV: PR-AUC"),
        ("roc_auc", "5-Fold CV: ROC-AUC"),
    ]
    colors = {"uniform": "#4C72B0", "distance": "#DD8452"}
    markers = {"uniform": "o", "distance": "s"}

    for ax, (metric, title) in zip(axes.flat, panels):
        for w in WEIGHTS_GRID:
            sub = cv_summary[cv_summary["weights"] == w].sort_values("k")
            k_vals = sub["k"].to_numpy()
            mean_v = sub[f"{metric}_mean"].to_numpy()
            std_v = sub[f"{metric}_std"].to_numpy()
            ax.plot(k_vals, mean_v, marker=markers[w], color=colors[w], linewidth=2, label=f"weights='{w}'")
            ax.fill_between(k_vals, mean_v - std_v, mean_v + std_v, color=colors[w], alpha=0.14)

        # Highlight E0 (k=5, uniform)
        e0_sub = cv_summary[(cv_summary["k"] == 5) & (cv_summary["weights"] == "uniform")].iloc[0]
        ax.scatter([5], [e0_sub[f"{metric}_mean"]], color="#333333", s=70, zorder=5, label="E0 (k=5, uniform)")

        # Highlight E1 best F1
        ax.axvline(int(best_f1_row["k"]), color="#2CA02C", linestyle="--", linewidth=1.2,
                   label=f"Best F1/MCC: k={int(best_f1_row['k'])}, {best_f1_row['weights']}")
        if int(best_prauc_row["k"]) != int(best_f1_row["k"]) or best_prauc_row["weights"] != best_f1_row["weights"]:
            ax.axvline(int(best_prauc_row["k"]), color="#9467BD", linestyle=":", linewidth=1.4,
                       label=f"Best PR-AUC: k={int(best_prauc_row['k'])}, {best_prauc_row['weights']}")

        ax.set_title(title)
        ax.set_xlabel("Nilai k (Jumlah Tetangga)")
        ax.set_ylabel(metric)
        ax.set_xticks(K_GRID)
        ax.legend(fontsize=8)

    fig.suptitle("Hyperparameter Tuning Baseline KNN pada 5-Fold Stratified CV (Data Training)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = FIGURES_DIR / "01_cv_hyperparameter_tuning.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2. Hold-out ROC & PR curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    palette = ["#4C72B0", "#55A868", "#8172B3"]
    for idx, res in enumerate(holdout_results):
        mid = res["model_id"]
        label_base = f"{mid}: {res['model_name']} (k={res['k']}, {res['weights']})"
        y_prob = probs_map[mid]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=palette[idx % len(palette)], linewidth=2.0,
                     label=f"{label_base} | AUC={res['roc_auc']:.4f}")

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(rec, prec, color=palette[idx % len(palette)], linewidth=2.0,
                     label=f"{label_base} | PR-AUC={res['pr_auc']:.4f}")

    axes[0].plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1, label="Random (0.5000)")
    axes[0].set_title("Kurva ROC — Hold-out Test Set (9,043)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(fontsize=8, loc="lower right")

    pos_base = float(y_test.mean())
    axes[1].axhline(pos_base, color="#666666", linestyle="--", linewidth=1, label=f"No-Skill ({pos_base:.4f})")
    axes[1].set_title("Kurva Precision–Recall — Hold-out Test Set (9,043)")
    axes[1].set_xlabel("Recall (yes)")
    axes[1].set_ylabel("Precision (yes)")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_e0_vs_e1_roc_pr_curves.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # 3. Metric bar comparison with 95% CI + Confusion Matrices for E0 and primary E1
    e0_res = holdout_results[0]
    e1_res = holdout_results[1]

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 1.0, 1.0])
    ax_bar = fig.add_subplot(gs[0, 0])

    x = np.arange(len(METRIC_KEYS))
    w = 0.36
    e0_vals = [e0_res[m] for m in METRIC_KEYS]
    e1_vals = [e1_res[m] for m in METRIC_KEYS]
    e0_err = np.array([[e0_res[m] - e0_res[f"{m}_ci95_low"], e0_res[f"{m}_ci95_high"] - e0_res[m]] for m in METRIC_KEYS]).T
    e1_err = np.array([[e1_res[m] - e1_res[f"{m}_ci95_low"], e1_res[f"{m}_ci95_high"] - e1_res[m]] for m in METRIC_KEYS]).T

    ax_bar.bar(x - w / 2, e0_vals, width=w, yerr=e0_err, capsize=3, color="#4C72B0",
               label=f"E0 Standard (k={e0_res['k']}, {e0_res['weights']})")
    ax_bar.bar(x + w / 2, e1_vals, width=w, yerr=e1_err, capsize=3, color="#55A868",
               label=f"E1 Optimized (k={e1_res['k']}, {e1_res['weights']})")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["Acc", "Prec", "Rec", "F1", "PR-AUC", "ROC-AUC", "MCC", "G-Mean"], rotation=20)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Perbandingan Metrik Hold-out Test (±95% Bootstrap CI)")
    ax_bar.set_ylabel("Skor Metrik")
    ax_bar.legend(fontsize=8.5, loc="upper right")

    for idx_cm, res in enumerate([e0_res, e1_res]):
        ax_cm = fig.add_subplot(gs[0, 1 + idx_cm])
        cm = np.array([[res["tn"], res["fp"]], [res["fn"], res["tp"]]])
        ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred no", "Pred yes"])
        ax_cm.set_yticklabels(["Actual no", "Actual yes"])
        ax_cm.set_title(f"Confusion Matrix {res['model_id']}\n(k={res['k']}, {res['weights']})")
        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                ax_cm.text(
                    c,
                    r,
                    f"{cm[r, c]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontweight="bold",
                    fontsize=10.5,
                )

    fig.tight_layout()
    p3 = FIGURES_DIR / "03_e0_vs_e1_confusion_and_metrics.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    return saved


def build_report(
    eq_df: pd.DataFrame,
    cv_summary: pd.DataFrame,
    best_f1_row: pd.Series,
    best_prauc_row: pd.Series,
    holdout_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Build Laporan_Baseline_KNN_DIFW_KNN.md."""
    cv_compact = cv_summary.copy()
    for col in cv_compact.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            cv_compact[col] = cv_compact[col].round(4)

    cv_display_cols = [
        "weights",
        "k",
        "accuracy_mean",
        "precision_yes_mean",
        "recall_yes_mean",
        "f1_yes_mean",
        "pr_auc_mean",
        "roc_auc_mean",
        "mcc_mean",
        "g_mean_mean",
    ]

    holdout_compact = holdout_df[
        [
            "model_id",
            "model_name",
            "k",
            "weights",
            "accuracy",
            "precision_yes",
            "recall_yes",
            "f1_yes",
            "pr_auc",
            "roc_auc",
            "mcc",
            "g_mean",
            "fit_sec",
            "predict_sec",
        ]
    ].copy()
    for c in METRIC_KEYS:
        holdout_compact[c] = holdout_compact[c].round(4)

    e0 = holdout_df.iloc[0]
    e1 = holdout_df.iloc[1]

    ci_rows = []
    for _, r in holdout_df.iterrows():
        row_ci = {"model_id": r["model_id"], "config": f"k={r['k']}, {r['weights']}"}
        for m in METRIC_KEYS:
            row_ci[m] = f"{r[m]:.4f} [{r[f'{m}_ci95_low']:.4f}, {r[f'{m}_ci95_high']:.4f}]"
        ci_rows.append(row_ci)
    ci_df = pd.DataFrame(ci_rows)

    lines = [
        "# LAPORAN TAHAP 04 — BASELINE KNN (`E0` STANDARD & `E1` OPTIMIZED)",
        "",
        "**Modul Kode:** [`src/baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/baseline_knn.py), [`src/04_baseline_knn.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/04_baseline_knn.py)  ",
        "**Notebook:** [`notebooks/04_Baseline_KNN.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/04_Baseline_KNN.ipynb)",
        "",
        "---",
        "",
        "## 1. Unit Test Ekuivalensi Pencarian Tetangga",
        "",
        "Memastikan implementasi cepat berbasis `NearestNeighbors` menghasilkan probabilitas prediksi yang **identik secara matematis** dengan `sklearn.neighbors.KNeighborsClassifier` (toleransi $< 10^{-9}$) pada Fold 0:",
        "",
        df_to_markdown_table(eq_df),
        "",
        "---",
        "",
        "## 2. Hasil Tuning Hyperparameter pada Stratified 5-Fold CV (`36.168` Data Training)",
        "",
        f"- **Kandidat Terbaik berdasarkan `F1 (yes)` & `MCC` (dipilih sebagai `E1` utama):** `k = {int(best_f1_row['k'])}`, `weights = '{best_f1_row['weights']}'` -> CV `F1 = {best_f1_row['f1_yes_mean']:.4f}`, `MCC = {best_f1_row['mcc_mean']:.4f}`, `PR-AUC = {best_f1_row['pr_auc_mean']:.4f}`, `ROC-AUC = {best_f1_row['roc_auc_mean']:.4f}`.",
        f"- **Kandidat Terbaik berdasarkan `PR-AUC`:** `k = {int(best_prauc_row['k'])}`, `weights = '{best_prauc_row['weights']}'` -> CV `PR-AUC = {best_prauc_row['pr_auc_mean']:.4f}`, `ROC-AUC = {best_prauc_row['roc_auc_mean']:.4f}`, `F1 = {best_prauc_row['f1_yes_mean']:.4f}`, `MCC = {best_prauc_row['mcc_mean']:.4f}`.",
        "",
        df_to_markdown_table(cv_compact[cv_display_cols]),
        "",
        "> **Observasi Peneliti (Fenomena Klasik Imbalanced KNN pada Euclidean):**",
        "> Semakin besar nilai $k$, probabilitas lokal $\\hat{P}(\\text{yes}|x)$ menjadi lebih halus sehingga metrik berbasis ranking (`PR-AUC` dan `ROC-AUC`) meningkat hingga sekitar $k=51$–$101$. Namun pada *decision threshold* standar `0.50`, nilai $k$ besar menyebabkan kelas minoritas (`11,70%`) kalah suara sehingga `Recall (yes)` turun. Ini menjadi bukti empiris awal mengapa penelitian ini membutuhkan **Mixed-Type Distance + Supervised Feature Weighting (`E3`) + Cost-Sensitive Decision Rule (`E4`/`E5`)**.",
        "",
        "---",
        "",
        "## 3. Hasil Evaluasi pada Hold-out Test Set (`9.043` Observasi)",
        "",
        df_to_markdown_table(holdout_compact),
        "",
        "### Interval Kepercayaan Bootstrap 95% (`n_boot = 1.000`)",
        "",
        df_to_markdown_table(ci_df),
        "",
        "### Confusion Matrix Hold-out Test",
        "",
        df_to_markdown_table(cm_df),
        "",
        "> **Ringkasan Kinerja Baseline (`E0` vs `E1`):**",
        f"> - **`E0 (Standard KNN, k=5, uniform)`:** `Accuracy = {e0['accuracy']:.4f}`, `Precision = {e0['precision_yes']:.4f}`, `Recall = {e0['recall_yes']:.4f}`, `F1 = {e0['f1_yes']:.4f}`, `PR-AUC = {e0['pr_auc']:.4f}`, `ROC-AUC = {e0['roc_auc']:.4f}`, `MCC = {e0['mcc']:.4f}`, `G-Mean = {e0['g_mean']:.4f}`.",
        f"> - **`E1 (Optimized KNN, k={int(e1['k'])}, {e1['weights']})`:** `Accuracy = {e1['accuracy']:.4f}`, `Precision = {e1['precision_yes']:.4f}`, `Recall = {e1['recall_yes']:.4f}`, `F1 = {e1['f1_yes']:.4f}`, `PR-AUC = {e1['pr_auc']:.4f}`, `ROC-AUC = {e1['roc_auc']:.4f}`, `MCC = {e1['mcc']:.4f}`, `G-Mean = {e1['g_mean']:.4f}`.",
        "",
        "---",
        "",
        "## 4. Artefak Visual",
        "",
    ]
    for p in figure_paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"- [{p.name}](file:///{p.as_posix()}) (`{rel}`)")
    return "\n".join(lines)


def main() -> None:
    """Execute Stage 04 Baseline KNN CV tuning and Hold-out evaluation."""
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_split.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_split.csv")
    folds_df = pd.read_csv(PROCESSED_DIR / "train_cv_folds.csv")

    X_train_df = train_df.drop(columns=["y"])
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    X_test_df = test_df.drop(columns=["y"])
    y_test = test_df["y"].to_numpy(dtype=np.int64)
    fold_ids = folds_df["fold_id"].to_numpy(dtype=np.int64)

    # 1. Equivalence unit check on Fold 0
    val0 = fold_ids == 0
    prep0 = BaselineEuclideanPreprocessor()
    X_tr0 = prep0.fit_transform(X_train_df.loc[~val0].reset_index(drop=True))
    X_va0 = prep0.transform(X_train_df.loc[val0].reset_index(drop=True))
    eq_df = verify_knn_probability_equivalence(X_tr0, y_train[~val0], X_va0)
    eq_df.to_csv(OUTPUT_DIR / "unit_test_equivalence.csv", index=False)

    # 2. Stratified 5-fold CV grid search
    per_fold_df, cv_summary = run_stratified_cv_grid(
        X_train_df=X_train_df,
        y_train=y_train,
        fold_ids=fold_ids,
        k_grid=K_GRID,
        weights_grid=WEIGHTS_GRID,
    )
    per_fold_df.to_csv(OUTPUT_DIR / "cv_per_fold_results.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "cv_grid_results.csv", index=False)

    # 3. Select E1 best configs from CV
    best_f1_row = cv_summary.sort_values(
        ["f1_yes_mean", "mcc_mean", "pr_auc_mean"],
        ascending=False,
    ).iloc[0]
    best_prauc_row = cv_summary.sort_values(
        ["pr_auc_mean", "roc_auc_mean", "f1_yes_mean"],
        ascending=False,
    ).iloc[0]

    # 4. Evaluate E0 & E1 on Hold-out Test Set
    models_to_eval = [
        ("E0", "Standard KNN", 5, "uniform"),
        ("E1", "Optimized KNN", int(best_f1_row["k"]), str(best_f1_row["weights"])),
    ]
    if (int(best_prauc_row["k"]) != int(best_f1_row["k"])) or (
        str(best_prauc_row["weights"]) != str(best_f1_row["weights"])
    ):
        models_to_eval.append(
            ("E1-PR", "Optimized KNN (Best CV PR-AUC)", int(best_prauc_row["k"]), str(best_prauc_row["weights"]))
        )

    holdout_results: list[dict[str, Any]] = []
    probs_map: dict[str, np.ndarray] = {}
    for mid, mname, k_val, w_val in models_to_eval:
        res, y_prob = evaluate_holdout_model(
            X_train_df=X_train_df,
            y_train=y_train,
            X_test_df=X_test_df,
            y_test=y_test,
            k=k_val,
            weights=w_val,
            model_id=mid,
            model_name=mname,
            n_boot=N_BOOTSTRAP,
            random_state=RANDOM_STATE,
        )
        holdout_results.append(res)
        probs_map[mid] = y_prob

    holdout_df = pd.DataFrame(holdout_results)
    holdout_df.to_csv(OUTPUT_DIR / "holdout_test_results.csv", index=False)

    cm_df = holdout_df[["model_id", "model_name", "k", "weights", "tn", "fp", "fn", "tp"]].copy()
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)

    pred_out = pd.DataFrame({"y_true": y_test})
    for mid, p_arr in probs_map.items():
        pred_out[f"prob_{mid.lower().replace('-', '_')}"] = p_arr
    pred_out.to_csv(OUTPUT_DIR / "test_predictions_e0_e1.csv", index=False)

    figure_paths = create_baseline_figures(
        cv_summary=cv_summary,
        best_f1_row=best_f1_row,
        best_prauc_row=best_prauc_row,
        holdout_results=holdout_results,
        y_test=y_test,
        probs_map=probs_map,
    )

    report = build_report(
        eq_df=eq_df,
        cv_summary=cv_summary,
        best_f1_row=best_f1_row,
        best_prauc_row=best_prauc_row,
        holdout_df=holdout_df,
        cm_df=cm_df,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Tahap 04 (Baseline KNN E0 & E1) selesai.")
    for r in holdout_results:
        print(
            f"[{r['model_id']}] {r['model_name']} (k={r['k']}, {r['weights']}): "
            f"Acc={r['accuracy']:.4f}, Prec={r['precision_yes']:.4f}, Rec={r['recall_yes']:.4f}, "
            f"F1={r['f1_yes']:.4f}, PR-AUC={r['pr_auc']:.4f}, ROC-AUC={r['roc_auc']:.4f}, "
            f"MCC={r['mcc']:.4f}, G-Mean={r['g_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
