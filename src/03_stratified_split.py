"""Stage 03: Stratified Train-Test Split & 5-Fold Development CV Setup.

Creates canonical 80:20 stratified train/hold-out test splits, performs
numeric (Kolmogorov-Smirnov) and categorical (Chi-Square) covariate balance
checks between Train and Hold-out Test, builds canonical Stratified 5-Fold CV
assignments on the training set, generates diagnostic figures, and writes
Laporan_Stratified_Split_DIFW_KNN.md.
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
from scipy import stats
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    RAW_CATEGORICAL_COLS,
    RAW_NUMERIC_COLS,
    clean_deterministic,
    load_raw_dataset,
    stratified_split_data,
)


RAW_DATA_PATH = PROJECT_ROOT / "bank-full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "split"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Stratified_Split_DIFW_KNN.md"
N_CV_FOLDS = 5


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


def check_numeric_covariate_balance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Run two-sample Kolmogorov-Smirnov tests on numeric features across Train vs Test."""
    rows: list[dict[str, Any]] = []
    for col in RAW_NUMERIC_COLS:
        tr = X_train[col].astype(float).to_numpy()
        te = X_test[col].astype(float).to_numpy()
        ks_stat, p_val = stats.ks_2samp(tr, te)
        rows.append(
            {
                "feature": col,
                "train_mean": round(float(np.mean(tr)), 4),
                "test_mean": round(float(np.mean(te)), 4),
                "train_median": round(float(np.median(tr)), 4),
                "test_median": round(float(np.median(te)), 4),
                "ks_stat": round(float(ks_stat), 5),
                "p_value": round(float(p_val), 4),
                "balanced_at_alpha_0_05": bool(p_val >= 0.05),
            }
        )
    return pd.DataFrame(rows)


def check_categorical_covariate_balance(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Run Chi-Square homogeneity tests on categorical features across Train vs Test."""
    rows: list[dict[str, Any]] = []
    for col in RAW_CATEGORICAL_COLS:
        tr_counts = X_train[col].astype(str).value_counts()
        te_counts = X_test[col].astype(str).value_counts()
        cats = sorted(set(tr_counts.index) | set(te_counts.index))
        contingency = np.vstack(
            [
                [int(tr_counts.get(c, 0)) for c in cats],
                [int(te_counts.get(c, 0)) for c in cats],
            ]
        )
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        rows.append(
            {
                "feature": col,
                "n_categories": len(cats),
                "chi2_stat": round(float(chi2), 4),
                "dof": int(dof),
                "p_value": round(float(p_val), 4),
                "balanced_at_alpha_0_05": bool(p_val >= 0.05),
            }
        )
    return pd.DataFrame(rows)


def build_cv_folds(
    y_train: pd.Series,
    n_splits: int = N_CV_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign deterministic Stratified K-Fold IDs on training rows."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_ids = np.full(len(y_train), -1, dtype=np.int64)
    dummy_x = np.zeros(len(y_train), dtype=np.int8)

    summary_rows: list[dict[str, Any]] = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(dummy_x, y_train)):
        fold_ids[va_idx] = fold_id
        y_tr = y_train.iloc[tr_idx]
        y_va = y_train.iloc[va_idx]
        summary_rows.append(
            {
                "fold_id": fold_id,
                "n_train_fold": int(len(tr_idx)),
                "n_val_fold": int(len(va_idx)),
                "n_train_yes": int(y_tr.sum()),
                "pct_train_yes": round(100.0 * float(y_tr.mean()), 4),
                "n_val_yes": int(y_va.sum()),
                "pct_val_yes": round(100.0 * float(y_va.mean()), 4),
            }
        )

    if (fold_ids < 0).any():
        raise RuntimeError("Ada baris training yang belum mendapat assignment fold_id.")

    folds_df = pd.DataFrame(
        {
            "row_id": np.arange(len(y_train), dtype=np.int64),
            "y": y_train.to_numpy(dtype=np.int64),
            "fold_id": fold_ids,
        }
    )
    return folds_df, pd.DataFrame(summary_rows)


def create_split_figures(
    y_clean: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_summary: pd.DataFrame,
    num_balance: pd.DataFrame,
    cat_balance: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> list[Path]:
    """Generate split balance and covariate alignment figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Train/Test split & 5-fold stratification balance
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    split_labels = [
        f"Full\n({len(y_clean):,})",
        f"Train / Dev\n({len(y_train):,})",
        f"Hold-out Test\n({len(y_test):,})",
    ]
    pct_yes_splits = [100.0 * float(y_clean.mean()), 100.0 * float(y_train.mean()), 100.0 * float(y_test.mean())]
    bars_s = axes[0].bar(split_labels, pct_yes_splits, color=["#4C72B0", "#55A868", "#DD8452"], width=0.55)
    axes[0].axhline(pct_yes_splits[0], color="#333333", linestyle="--", linewidth=1, label=f"Target Global ({pct_yes_splits[0]:.2f}%)")
    axes[0].set_title("Proporsi Kelas Positif (y=yes): Train vs Hold-out Test")
    axes[0].set_ylabel("Persentase Kelas 'yes' (%)")
    axes[0].set_ylim(0, 15.5)
    axes[0].legend(loc="upper right", fontsize=8.5)
    for bar, val in zip(bars_s, pct_yes_splits):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.4f}%", ha="center", fontweight="bold", fontsize=9)

    fold_labels = [f"Fold {int(f)}\n(n={int(n):,})" for f, n in zip(cv_summary["fold_id"], cv_summary["n_val_fold"])]
    bars_f = axes[1].bar(fold_labels, cv_summary["pct_val_yes"], color="#8172B3", width=0.58)
    axes[1].axhline(pct_yes_splits[1], color="#333333", linestyle="--", linewidth=1, label=f"Train Mean ({pct_yes_splits[1]:.2f}%)")
    axes[1].set_title("Keseimbangan Stratifikasi 5-Fold Validation Set")
    axes[1].set_ylabel("Persentase Kelas 'yes' (%)")
    axes[1].set_ylim(0, 15.5)
    axes[1].legend(loc="upper right", fontsize=8.5)
    for bar, val in zip(bars_f, cv_summary["pct_val_yes"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}%", ha="center", fontweight="bold", fontsize=8.5)

    fig.tight_layout()
    p1 = FIGURES_DIR / "01_train_test_and_5fold_balance.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2. Covariate balance p-values and sample distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    combined_p = pd.concat(
        [
            num_balance[["feature", "p_value"]].assign(kind="KS (Numerik)"),
            cat_balance[["feature", "p_value"]].assign(kind="Chi2 (Kategorikal)"),
        ],
        ignore_index=True,
    ).sort_values("p_value", ascending=True)
    bar_colors = ["#4C72B0" if k.startswith("KS") else "#55A868" for k in combined_p["kind"]]
    axes[0].barh(combined_p["feature"], combined_p["p_value"], color=bar_colors, alpha=0.88)
    axes[0].axvline(0.05, color="#D62728", linestyle="--", linewidth=1.3, label="alpha = 0.05")
    axes[0].set_title("Uji Keseimbangan Fitur Train vs Test\n(Semua p-value >= 0.05 = Seimbang)")
    axes[0].set_xlabel("p-value")
    axes[0].set_xlim(0, 1.08)
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].tick_params(axis="y", labelsize=8.5)

    # Panel B: Empirical CDF of age (Train vs Test)
    tr_age = np.sort(X_train["age"].astype(float).to_numpy())
    te_age = np.sort(X_test["age"].astype(float).to_numpy())
    axes[1].plot(tr_age, np.linspace(0, 1, len(tr_age)), label="Train (36,168)", color="#4C72B0", linewidth=2)
    axes[1].plot(te_age, np.linspace(0, 1, len(te_age)), label="Hold-out Test (9,043)", color="#DD8452", linestyle="--", linewidth=1.8)
    axes[1].set_title("Alignment Distribusi Fitur 'age' (ECDF)")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("Probabilitas Kumulatif")
    axes[1].legend(fontsize=8.5)

    # Panel C: Category proportion alignment for poutcome
    tr_pout = 100.0 * X_train["poutcome"].value_counts(normalize=True)
    te_pout = 100.0 * X_test["poutcome"].value_counts(normalize=True)
    cats = ["unknown", "failure", "other", "success"]
    x_c = np.arange(len(cats))
    axes[2].bar(x_c - 0.18, [tr_pout.get(c, 0.0) for c in cats], width=0.36, label="Train", color="#4C72B0")
    axes[2].bar(x_c + 0.18, [te_pout.get(c, 0.0) for c in cats], width=0.36, label="Hold-out Test", color="#DD8452")
    axes[2].set_xticks(x_c)
    axes[2].set_xticklabels(cats)
    axes[2].set_title("Alignment Proporsi Kategori 'poutcome'")
    axes[2].set_ylabel("Proporsi (%)")
    axes[2].legend(fontsize=8.5)

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_covariate_distribution_alignment.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    return saved


def build_report(
    n_total: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_balance: pd.DataFrame,
    cat_balance: pd.DataFrame,
    cv_summary: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Build Laporan_Stratified_Split_DIFW_KNN.md."""
    split_overview = pd.DataFrame(
        [
            {
                "subset": "Full (Cleaned, Pre-Split)",
                "n_rows": n_total,
                "n_no": int((y_train == 0).sum() + (y_test == 0).sum()),
                "n_yes": int(y_train.sum() + y_test.sum()),
                "pct_yes": round(100.0 * float(y_train.sum() + y_test.sum()) / n_total, 4),
            },
            {
                "subset": "Training / Development (80%)",
                "n_rows": len(X_train),
                "n_no": int((y_train == 0).sum()),
                "n_yes": int(y_train.sum()),
                "pct_yes": round(100.0 * float(y_train.mean()), 4),
            },
            {
                "subset": "Hold-out Test (20%)",
                "n_rows": len(X_test),
                "n_no": int((y_test == 0).sum()),
                "n_yes": int(y_test.sum()),
                "pct_yes": round(100.0 * float(y_test.mean()), 4),
            },
        ]
    )

    n_num_ok = int(num_balance["balanced_at_alpha_0_05"].sum())
    n_cat_ok = int(cat_balance["balanced_at_alpha_0_05"].sum())
    min_p_num = float(num_balance["p_value"].min())
    min_p_cat = float(cat_balance["p_value"].min())

    lines = [
        "# LAPORAN TAHAP 03 — STRATIFIED TRAIN-TEST SPLIT & 5-FOLD CV SETUP",
        "",
        "**Skrip:** [`src/03_stratified_split.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/03_stratified_split.py)  ",
        "**Notebook:** [`notebooks/03_Stratified_Train_Test_Split.ipynb`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/notebooks/03_Stratified_Train_Test_Split.ipynb)",
        "",
        "---",
        "",
        "## 1. Pembagian Stratified Train-Test Split (80:20, `random_state=42`)",
        "",
        df_to_markdown_table(split_overview),
        "",
        "Artefak split kanonik tersimpan di:",
        "- [`data/processed/train_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/train_split.csv) (`36.168` observasi — Training / Development Data)",
        "- [`data/processed/test_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/test_split.csv) (`9.043` observasi — Hold-out Test Data, dikunci hingga evaluasi akhir)",
        "",
        "---",
        "",
        "## 2. Uji Keseimbangan Distribusi Fitur (*Covariate Balance Check*)",
        "",
        f"Dilakukan uji **Kolmogorov-Smirnov (KS)** pada `{len(num_balance)}` fitur numerik dan **Chi-Square Homogeneity** pada `{len(cat_balance)}` fitur kategorikal untuk memastikan distribusi kovariat antara Training dan Hold-out Test setara:",
        f"- **Fitur Numerik seimbang ($p \\ge 0,05$):** `{n_num_ok}/{len(num_balance)}` (p-value minimum = `{min_p_num:.4f}`)",
        f"- **Fitur Kategorikal seimbang ($p \\ge 0,05$):** `{n_cat_ok}/{len(cat_balance)}` (p-value minimum = `{min_p_cat:.4f}`)",
        "",
        "### a. Uji Kolmogorov-Smirnov (Fitur Numerik)",
        "",
        df_to_markdown_table(num_balance),
        "",
        "### b. Uji Chi-Square Homogeneity (Fitur Kategorikal)",
        "",
        df_to_markdown_table(cat_balance),
        "",
        "---",
        "",
        "## 3. Penyiapan Stratified 5-Fold Cross-Validation (`Training / Development`)",
        "",
        "Untuk menjaga perbandingan yang **100% adil** antara seluruh model (`E0`–`E5`), indeks lipatan *Stratified 5-Fold CV* (`shuffle=True`, `random_state=42`) disimpan secara eksplisit pada [`data/processed/train_cv_folds.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/train_cv_folds.csv).",
        "",
        df_to_markdown_table(cv_summary),
        "",
        "---",
        "",
        "## 4. Visualisasi Validasi Split",
        "",
    ]
    for p in figure_paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"- [{p.name}](file:///{p.as_posix()}) (`{rel}`)")
    return "\n".join(lines)


def main() -> None:
    """Execute Stage 03 stratified split, balance audit, and CV fold setup."""
    configure_plot_style()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset(RAW_DATA_PATH)
    X_clean, y_clean = clean_deterministic(raw_df)
    X_train, X_test, y_train, y_test = stratified_split_data(
        X_clean,
        y_clean,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )

    X_train.assign(y=y_train).to_csv(PROCESSED_DIR / "train_split.csv", index=False)
    X_test.assign(y=y_test).to_csv(PROCESSED_DIR / "test_split.csv", index=False)

    num_balance = check_numeric_covariate_balance(X_train, X_test)
    cat_balance = check_categorical_covariate_balance(X_train, X_test)
    folds_df, cv_summary = build_cv_folds(y_train, n_splits=N_CV_FOLDS, random_state=DEFAULT_RANDOM_STATE)

    folds_df.to_csv(PROCESSED_DIR / "train_cv_folds.csv", index=False)
    num_balance.to_csv(OUTPUT_DIR / "covariate_balance_numeric.csv", index=False)
    cat_balance.to_csv(OUTPUT_DIR / "covariate_balance_categorical.csv", index=False)
    cv_summary.to_csv(OUTPUT_DIR / "cv_fold_summary.csv", index=False)

    split_overview = pd.DataFrame(
        [
            {"subset": "full", "n_rows": len(X_clean), "n_yes": int(y_clean.sum()), "pct_yes": round(100.0 * float(y_clean.mean()), 4)},
            {"subset": "train", "n_rows": len(X_train), "n_yes": int(y_train.sum()), "pct_yes": round(100.0 * float(y_train.mean()), 4)},
            {"subset": "test", "n_rows": len(X_test), "n_yes": int(y_test.sum()), "pct_yes": round(100.0 * float(y_test.mean()), 4)},
        ]
    )
    split_overview.to_csv(OUTPUT_DIR / "split_overview.csv", index=False)

    figure_paths = create_split_figures(
        y_clean=y_clean,
        y_train=y_train,
        y_test=y_test,
        cv_summary=cv_summary,
        num_balance=num_balance,
        cat_balance=cat_balance,
        X_train=X_train,
        X_test=X_test,
    )

    report = build_report(
        n_total=len(X_clean),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        num_balance=num_balance,
        cat_balance=cat_balance,
        cv_summary=cv_summary,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Tahap 03 (Stratified Train-Test Split & 5-Fold Setup) selesai.")
    print(f"Train: {X_train.shape} (yes={100*y_train.mean():.4f}%), Hold-out Test: {X_test.shape} (yes={100*y_test.mean():.4f}%)")
    print(f"Numeric balance passed (p>=0.05): {int(num_balance['balanced_at_alpha_0_05'].sum())}/{len(num_balance)}")
    print(f"Categorical balance passed (p>=0.05): {int(cat_balance['balanced_at_alpha_0_05'].sum())}/{len(cat_balance)}")
    print(f"CV folds saved: {PROCESSED_DIR / 'train_cv_folds.csv'}")


if __name__ == "__main__":
    main()
