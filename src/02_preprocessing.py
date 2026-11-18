"""Execution script for Preprocessing & Feature Engineering (Stage 2).

Runs the leak-free preprocessing pipeline on bank-full.csv, persists stratified
train/test splits, verifies all distance and data-integrity unit checks,
generates diagnostic visualizations, and writes Laporan_Preprocessing_DIFW_KNN.md.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DOMAIN_CATEGORICAL_COLS,
    DOMAIN_FEATURE_COLS,
    DOMAIN_NUMERIC_COLS,
    BaselineEuclideanPreprocessor,
    MixedTypeEncoded,
    MixedTypePreprocessor,
    clean_deterministic,
    load_raw_dataset,
    stratified_split_data,
)


RAW_DATA_PATH = PROJECT_ROOT / "bank-full.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_Preprocessing_DIFW_KNN.md"


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
    """Render a DataFrame as a GitHub-flavored Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def check_unseen_categories(
    X_train_dom: pd.DataFrame,
    X_test_dom: pd.DataFrame,
) -> pd.DataFrame:
    """Audit unseen categorical values in test relative to training vocabulary."""
    rows = []
    for col in DOMAIN_CATEGORICAL_COLS:
        train_cats = set(X_train_dom[col].astype(str).unique())
        test_cats = set(X_test_dom[col].astype(str).unique())
        unseen = sorted(test_cats - train_cats)
        rows.append(
            {
                "feature": col,
                "n_train_categories": len(train_cats),
                "n_test_categories": len(test_cats),
                "n_unseen_in_test": len(unseen),
                "unseen_labels": ", ".join(unseen) if unseen else "-",
            }
        )
    return pd.DataFrame(rows)


def run_distance_unit_checks(
    mixed_prep: MixedTypePreprocessor,
    X_train_enc: MixedTypeEncoded,
    X_test_enc: MixedTypeEncoded,
    X_train_euc: np.ndarray,
    X_test_euc: np.ndarray,
    rng_seed: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Verify data integrity and formal metric properties on transformed samples."""
    rng = np.random.default_rng(rng_seed)
    idx_a = rng.choice(X_train_enc.n_samples, size=200, replace=False)
    idx_b = rng.choice(X_test_enc.n_samples, size=200, replace=False)

    sub_a = MixedTypeEncoded(
        numeric=X_train_enc.numeric[idx_a],
        categorical=X_train_enc.categorical[idx_a],
        feature_names=X_train_enc.feature_names,
    )
    sub_b = MixedTypeEncoded(
        numeric=X_test_enc.numeric[idx_b],
        categorical=X_test_enc.categorical[idx_b],
        feature_names=X_test_enc.feature_names,
    )

    # 1. NaN / Inf checks
    euc_finite = bool(np.isfinite(X_train_euc).all() and np.isfinite(X_test_euc).all())
    num_finite = bool(
        np.isfinite(X_train_enc.numeric).all() and np.isfinite(X_test_enc.numeric).all()
    )

    # 2. Per-feature distances
    d_aa = mixed_prep.compute_pairwise_feature_distances(sub_a, sub_a)
    d_ab = mixed_prep.compute_pairwise_feature_distances(sub_a, sub_b)
    d_ba = mixed_prep.compute_pairwise_feature_distances(sub_b, sub_a)

    # Self-distance == 0 on diagonal for all 16 features
    diag_max = float(np.max(np.abs(np.diagonal(d_aa, axis1=0, axis2=1))))
    self_dist_ok = diag_max == 0.0

    # Symmetry: d_ab[i, j, f] == d_ba[j, i, f]
    sym_diff = float(np.max(np.abs(d_ab - np.transpose(d_ba, (1, 0, 2)))))
    symmetry_ok = sym_diff <= 1e-12

    # Bounds [0, 1]
    min_val = float(min(d_aa.min(), d_ab.min()))
    max_val = float(max(d_aa.max(), d_ab.max()))
    bounds_ok = (min_val >= 0.0) and (max_val <= 1.0)

    # 3. Weighted matrix consistency vs direct pairwise mean
    u = np.full(len(DOMAIN_FEATURE_COLS), 1.0 / len(DOMAIN_FEATURE_COLS))
    dw_direct = (d_ab * u[None, None, :]).sum(axis=2)
    dw_batched = mixed_prep.compute_weighted_distance_matrix(sub_a, sub_b, weights=u, batch_size=64)
    batch_diff = float(np.max(np.abs(dw_direct - dw_batched)))
    batch_ok = batch_diff <= 1e-12

    # Assert critical properties so any violation stops execution immediately
    assert euc_finite and num_finite, "Ditemukan NaN atau Inf pada hasil preprocessing!"
    assert self_dist_ok, f"Self-distance != 0 (max diagonal={diag_max})"
    assert symmetry_ok, f"Asimetri terdeteksi (max diff={sym_diff})"
    assert bounds_ok, f"Jarak keluar dari rentang [0, 1]: [{min_val}, {max_val}]"
    assert batch_ok, f"Perbedaan weighted matrix batched vs direct={batch_diff}"

    checks = [
        {
            "check_name": "Finite check (tanpa NaN/Inf pada Euclidean & Mixed)",
            "status": "PASS",
            "measured_value": f"euc={euc_finite}, mixed={num_finite}",
        },
        {
            "check_name": "Self-distance d_j(x, x) = 0",
            "status": "PASS",
            "measured_value": f"max_diag={diag_max:.1e}",
        },
        {
            "check_name": "Simetri d_j(x, z) = d_j(z, x)",
            "status": "PASS",
            "measured_value": f"max_diff={sym_diff:.1e}",
        },
        {
            "check_name": "Rentang jarak fitur d_j(x, z) dalam [0, 1]",
            "status": "PASS",
            "measured_value": f"[{min_val:.4f}, {max_val:.4f}]",
        },
        {
            "check_name": "Konsistensi batched weighted distance vs exact pairwise",
            "status": "PASS",
            "measured_value": f"max_diff={batch_diff:.1e}",
        },
    ]
    return pd.DataFrame(checks)


def create_preprocessing_figures(
    y_full: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_raw: pd.DataFrame,
    X_train_dom: pd.DataFrame,
    mixed_prep: MixedTypePreprocessor,
    X_train_enc: MixedTypeEncoded,
    X_train_euc: np.ndarray,
) -> list[Path]:
    """Generate diagnostic figures illustrating split balance and transformations."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 1. Stratified split verification
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    splits = ["Full Dataset\n(45,211)", f"Train Split\n({len(y_train):,})", f"Test Split\n({len(y_test):,})"]
    counts_no = [int((y_full == 0).sum()), int((y_train == 0).sum()), int((y_test == 0).sum())]
    counts_yes = [int((y_full == 1).sum()), int((y_train == 1).sum()), int((y_test == 1).sum())]
    pct_yes = [100.0 * y_full.mean(), 100.0 * y_train.mean(), 100.0 * y_test.mean()]
    pct_no = [100.0 - p for p in pct_yes]

    x = np.arange(len(splits))
    w = 0.35
    b1 = axes[0].bar(x - w / 2, counts_no, width=w, label="no (0)", color="#4C72B0")
    b2 = axes[0].bar(x + w / 2, counts_yes, width=w, label="yes (1)", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].set_title("Jumlah Observasi per Split (80:20 Stratified)")
    axes[0].set_ylabel("Jumlah Baris")
    axes[0].set_ylim(0, max(counts_no) * 1.16)
    axes[0].legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 500, f"{int(h):,}", ha="center", fontsize=8.5)

    axes[1].bar(splits, pct_no, label="no (0)", color="#4C72B0")
    axes[1].bar(splits, pct_yes, bottom=pct_no, label="yes (1)", color="#DD8452")
    axes[1].set_title("Verifikasi Proporsi Target Identik (Stratifikasi)")
    axes[1].set_ylabel("Persentase (%)")
    axes[1].set_ylim(0, 118)
    axes[1].legend(loc="upper center", ncol=2, frameon=True)
    for idx, p_y in enumerate(pct_yes):
        axes[1].text(idx, 100 - p_y / 2, f"{p_y:.2f}%", ha="center", va="center", color="white", fontweight="bold")
        axes[1].text(idx, (100 - p_y) / 2, f"{100 - p_y:.2f}%", ha="center", va="center", color="white", fontweight="bold")

    fig.tight_layout()
    p1 = FIGURES_DIR / "01_stratified_split_verification.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # 2. pdays transformation: raw vs pdays_recency + conversion by recency
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    pdays_pos_raw = X_train_raw.loc[X_train_raw["pdays"] > -1, "pdays"]
    axes[0].hist(pdays_pos_raw, bins=35, color="#8172B3", alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[0].axvline(mixed_prep.feature_engineer.pdays_median_pos_, color="#D62728", linestyle="--",
                    label=f"M_pos={mixed_prep.feature_engineer.pdays_median_pos_:.0f}")
    axes[0].set_title("Sebelum: pdays Mentah (Subset > -1)\n81.74% bernilai sentinel -1")
    axes[0].set_xlabel("pdays (hari)")
    axes[0].set_ylabel("Frekuensi (Train)")
    axes[0].legend(fontsize=8.5)

    rec_pos = X_train_dom.loc[X_train_dom["previously_contacted"] == "1", "pdays_recency"]
    axes[1].hist(rec_pos, bins=35, color="#55A868", alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[1].set_title("Sesudah: pdays_recency (Subset Kontak=1)\n0.0 khusus untuk belum pernah dihubungi")
    axes[1].set_xlabel("pdays_recency = 1 / (1 + pdays / M_pos)")
    axes[1].set_ylabel("Frekuensi (Train)")

    # Conversion rate across previously_contacted and pdays_recency quartiles
    rec_df = pd.DataFrame(
        {
            "prev_contacted": X_train_dom["previously_contacted"],
            "recency": X_train_dom["pdays_recency"],
            "y": y_train.to_numpy(),
        }
    )
    not_c = rec_df[rec_df["prev_contacted"] == "0"]
    yes_c = rec_df[rec_df["prev_contacted"] == "1"].copy()
    yes_c["q_bin"] = pd.qcut(yes_c["recency"], q=4, labels=["Q1 (Lama)", "Q2", "Q3", "Q4 (Baru)"])
    summary_rates = [100.0 * not_c["y"].mean()] + [
        100.0 * yes_c.loc[yes_c["q_bin"] == lbl, "y"].mean()
        for lbl in ["Q1 (Lama)", "Q2", "Q3", "Q4 (Baru)"]
    ]
    labels_rec = ["Belum Kontak\n(recency=0)", "Q1\n(Lama)", "Q2", "Q3", "Q4\n(Baru)"]
    bars_r = axes[2].bar(labels_rec, summary_rates, color=["#7F7F7F", "#4C72B0", "#55A868", "#CCB974", "#DD8452"])
    axes[2].set_title("Konversi (y=yes) pada Fitur Hasil Rekayasa\n(Data Training)")
    axes[2].set_ylabel("Tingkat Konversi (%)")
    axes[2].set_ylim(0, max(summary_rates) * 1.22)
    for bar, val in zip(bars_r, summary_rates):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{val:.1f}%", ha="center", fontsize=8.5)

    fig.tight_layout()
    p2 = FIGURES_DIR / "02_pdays_engineering_comparison.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    # 3. StandardScaler vs Mixed-Type Distance bounds (pair distances on training sample)
    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    sample_idx = rng.choice(X_train_enc.n_samples, size=250, replace=False)
    sample_mixed = MixedTypeEncoded(
        numeric=X_train_enc.numeric[sample_idx],
        categorical=X_train_enc.categorical[sample_idx],
        feature_names=X_train_enc.feature_names,
    )
    d_mixed = mixed_prep.compute_pairwise_feature_distances(sample_mixed, sample_mixed)
    iu = np.triu_indices(250, k=1)

    # Compute pairwise feature differences on the 6 standardized numeric features
    euc_num = X_train_euc[sample_idx, : len(DOMAIN_NUMERIC_COLS)]
    euc_diffs = [np.abs(euc_num[:, j][iu[0]] - euc_num[:, j][iu[1]]) for j in range(len(DOMAIN_NUMERIC_COLS))]
    mixed_diffs = [d_mixed[:, :, j][iu[0], iu[1]] for j in range(len(DOMAIN_NUMERIC_COLS))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].boxplot(euc_diffs, tick_labels=DOMAIN_NUMERIC_COLS, vert=True, showfliers=True)
    axes[0].set_title("Jalur Euclidean (E0–E1): |z_i - z_k| Setelah StandardScaler\n(Rentan Outlier Ekstrem)")
    axes[0].set_ylabel("Selisih Absolut (Satuan Std Dev)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].boxplot(mixed_diffs, tick_labels=DOMAIN_NUMERIC_COLS, vert=True, showfliers=True)
    axes[1].set_title("Jalur Mixed-Type (E2–E5): d_j(x, z) Berbasis Q95–Q05\n(Terikat Rapi pada [0, 1])")
    axes[1].set_ylabel("Jarak Fitur Terklip [0, 1]")
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    p3 = FIGURES_DIR / "03_quantile_scaling_vs_standard_scaling.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)

    return paths


def build_report(
    n_raw: int,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_dom: pd.DataFrame,
    X_test_dom: pd.DataFrame,
    mixed_prep: MixedTypePreprocessor,
    euc_prep: BaselineEuclideanPreprocessor,
    X_train_euc: np.ndarray,
    X_test_euc: np.ndarray,
    unseen_df: pd.DataFrame,
    unit_checks_df: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Build Laporan_Preprocessing_DIFW_KNN.md."""
    split_df = pd.DataFrame(
        [
            {
                "subset": "Full (Cleaned)",
                "n_rows": n_raw,
                "n_cols_pre_engineering": X_train_raw.shape[1],
                "n_no": int((y_train == 0).sum() + (y_test == 0).sum()),
                "n_yes": int((y_train == 1).sum() + (y_test == 1).sum()),
                "pct_yes": round(100.0 * (y_train.sum() + y_test.sum()) / n_raw, 4),
            },
            {
                "subset": "Train Split (80%)",
                "n_rows": len(X_train_raw),
                "n_cols_pre_engineering": X_train_raw.shape[1],
                "n_no": int((y_train == 0).sum()),
                "n_yes": int((y_train == 1).sum()),
                "pct_yes": round(100.0 * y_train.mean(), 4),
            },
            {
                "subset": "Test Split (20%)",
                "n_rows": len(X_test_raw),
                "n_cols_pre_engineering": X_test_raw.shape[1],
                "n_no": int((y_test == 0).sum()),
                "n_yes": int((y_test == 1).sum()),
                "pct_yes": round(100.0 * y_test.mean(), 4),
            },
        ]
    )

    quantile_df = mixed_prep.get_numeric_params_table()
    m_pos = mixed_prep.feature_engineer.pdays_median_pos_
    rec_train_pos = X_train_dom.loc[X_train_dom["previously_contacted"] == "1", "pdays_recency"]
    rec_test_pos = X_test_dom.loc[X_test_dom["previously_contacted"] == "1", "pdays_recency"]

    pdays_summary_df = pd.DataFrame(
        [
            {
                "subset": "Train",
                "n_contacted": int((X_train_dom["previously_contacted"] == "1").sum()),
                "pct_contacted": round(100.0 * (X_train_dom["previously_contacted"] == "1").mean(), 2),
                "recency_min_pos": round(float(rec_train_pos.min()), 4),
                "recency_median_pos": round(float(rec_train_pos.median()), 4),
                "recency_max_pos": round(float(rec_train_pos.max()), 4),
            },
            {
                "subset": "Test",
                "n_contacted": int((X_test_dom["previously_contacted"] == "1").sum()),
                "pct_contacted": round(100.0 * (X_test_dom["previously_contacted"] == "1").mean(), 2),
                "recency_min_pos": round(float(rec_test_pos.min()), 4),
                "recency_median_pos": round(float(rec_test_pos.median()), 4),
                "recency_max_pos": round(float(rec_test_pos.max()), 4),
            },
        ]
    )

    lines = [
        "# LAPORAN PREPROCESSING & FEATURE ENGINEERING (DIFW-KNN)",
        "",
        "**Sumber Data:** `bank-full.csv` (UCI Bank Marketing — 45.211 observasi)  ",
        "**Modul Kode:** [`src/preprocessing.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/preprocessing.py) dan [`src/02_preprocessing.py`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/src/02_preprocessing.py)",
        "",
        "---",
        "",
        "## 1. Pembersihan Deterministik & Penghapusan Target Leakage",
        "",
        "- **Penghapusan `duration`:** Kolom `duration` dihapus sebelum *split* karena nilainya hanya diketahui setelah panggilan selesai (*post-call leakage*).",
        "- **Pelestarian Kategori `\"unknown\"`:** Nilai `\"unknown\"` pada `job`, `education`, `contact`, dan `poutcome` dipertahankan sebagai kategori valid (tidak diimputasi dengan modus).",
        "- **Encoding Target:** Target `y` dipetakan deterministik menjadi `yes -> 1` dan `no -> 0`.",
        "",
        "---",
        "",
        "## 2. Stratified Train-Test Split (80:20, `random_state=42`)",
        "",
        df_to_markdown_table(split_df),
        "",
        "Kedua file *split* bersih (sebelum transformasi `pdays` agar *Cross-Validation* per *fold* tetap bebas *leakage*) disimpan di:",
        "- [`data/processed/train_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/train_split.csv) (`36.168` baris)",
        "- [`data/processed/test_split.csv`](file:///c:/Users/Advan%20Workplus/Downloads/jurnak/data/processed/test_split.csv) (`9.043` baris)",
        "",
        "---",
        "",
        "## 3. Rekayasa Fitur `pdays` Berbasis Training-Only ($M_{\\text{pos, train}}$)",
        "",
        f"Dari subset `pdays > -1` pada data training (`{int((X_train_raw['pdays'] > -1).sum()):,}` observasi), diperoleh median training **$M_{{\\text{{pos, train}}}} = {m_pos:.1f}$ hari**. Fitur `pdays` mentah kemudian dipecah menjadi:",
        "1. `previously_contacted`: `1` jika `pdays > -1`, dan `0` jika `pdays == -1`.",
        "2. `pdays_recency`: $\\frac{1}{1 + \\text{pdays} / M_{\\text{pos, train}}}$ jika `pdays > -1`, dan `0.0` jika `pdays == -1`.",
        "",
        df_to_markdown_table(pdays_summary_df),
        "",
        "Setelah transformasi ini dan penghapusan `pdays` mentah, terbentuk **16 fitur domain**:",
        f"- **6 Fitur Numerik:** {', '.join(f'`{c}`' for c in DOMAIN_NUMERIC_COLS)}",
        f"- **10 Fitur Kategorikal/Biner:** {', '.join(f'`{c}`' for c in DOMAIN_CATEGORICAL_COLS)}",
        "",
        "---",
        "",
        "## 4. Parameter Dua Jalur Preprocessor (Training-Only Fit)",
        "",
        "### a. Jalur Baseline Euclidean (`E0`, `E1`) — `BaselineEuclideanPreprocessor`",
        f"- `StandardScaler` pada 6 fitur numerik + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` pada 10 fitur kategorikal/biner.",
        f"- **Dimensi Matriks Output:** `X_train_euc = {X_train_euc.shape}`, `X_test_euc = {X_test_euc.shape}` (`{len(euc_prep.output_feature_names_)}` fitur hasil encoding).",
        "",
        "### b. Jalur Proposed Mixed-Type (`E2`–`E5`) — `MixedTypePreprocessor`",
        "- Parameter kuantil $(Q_{05,j}, Q_{95,j})$ dan rentang $R_j = Q_{95,j} - Q_{05,j}$ dihitung **hanya dari training set**:",
        "",
        df_to_markdown_table(quantile_df),
        "",
        "### Audit Kategori Unseen pada Test Set",
        "",
        df_to_markdown_table(unseen_df),
        "",
        "---",
        "",
        "## 5. Verifikasi Kualitas Data & Properti Metrik Jarak",
        "",
        df_to_markdown_table(unit_checks_df),
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
    """Execute Stage 2 preprocessing pipeline, checks, and reporting."""
    configure_plot_style()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset(RAW_DATA_PATH)
    X_clean, y_clean = clean_deterministic(raw_df)
    X_train_raw, X_test_raw, y_train, y_test = stratified_split_data(
        X_clean,
        y_clean,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # Save canonical leak-free splits (duration removed, raw pdays preserved for CV-fold fitting)
    train_df_out = X_train_raw.copy()
    train_df_out["y"] = y_train
    test_df_out = X_test_raw.copy()
    test_df_out["y"] = y_test
    train_df_out.to_csv(PROCESSED_DIR / "train_split.csv", index=False)
    test_df_out.to_csv(PROCESSED_DIR / "test_split.csv", index=False)

    # Fit both preprocessors ONLY on training set
    euc_prep = BaselineEuclideanPreprocessor()
    X_train_euc = euc_prep.fit_transform(X_train_raw)
    X_test_euc = euc_prep.transform(X_test_raw)

    mixed_prep = MixedTypePreprocessor()
    X_train_enc = mixed_prep.fit_transform(X_train_raw)
    X_test_enc = mixed_prep.transform(X_test_raw)

    # Engineered domain DataFrames for inspection and audit
    X_train_dom = mixed_prep.feature_engineer.transform(X_train_raw)
    X_test_dom = mixed_prep.feature_engineer.transform(X_test_raw)

    unseen_df = check_unseen_categories(X_train_dom, X_test_dom)
    unit_checks_df = run_distance_unit_checks(
        mixed_prep=mixed_prep,
        X_train_enc=X_train_enc,
        X_test_enc=X_test_enc,
        X_train_euc=X_train_euc,
        X_test_euc=X_test_euc,
    )

    # Persist summary tables
    quantile_df = mixed_prep.get_numeric_params_table()
    quantile_df.to_csv(OUTPUT_DIR / "numeric_quantile_params.csv", index=False)
    unseen_df.to_csv(OUTPUT_DIR / "unseen_category_audit.csv", index=False)
    unit_checks_df.to_csv(OUTPUT_DIR / "preprocessing_unit_checks.csv", index=False)
    pd.DataFrame({"feature_name": euc_prep.output_feature_names_}).to_csv(
        OUTPUT_DIR / "baseline_euclidean_features.csv", index=False
    )

    split_summary = pd.DataFrame(
        [
            {"subset": "train", "n_rows": len(X_train_raw), "n_yes": int(y_train.sum()), "pct_yes": round(100.0 * y_train.mean(), 4)},
            {"subset": "test", "n_rows": len(X_test_raw), "n_yes": int(y_test.sum()), "pct_yes": round(100.0 * y_test.mean(), 4)},
        ]
    )
    split_summary.to_csv(OUTPUT_DIR / "split_summary.csv", index=False)

    figure_paths = create_preprocessing_figures(
        y_full=y_clean,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_train_dom=X_train_dom,
        mixed_prep=mixed_prep,
        X_train_enc=X_train_enc,
        X_train_euc=X_train_euc,
    )

    report_text = build_report(
        n_raw=len(raw_df),
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_test=y_test,
        X_train_dom=X_train_dom,
        X_test_dom=X_test_dom,
        mixed_prep=mixed_prep,
        euc_prep=euc_prep,
        X_train_euc=X_train_euc,
        X_test_euc=X_test_euc,
        unseen_df=unseen_df,
        unit_checks_df=unit_checks_df,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("Preprocessing & Feature Engineering selesai.")
    print(f"Train split: {X_train_raw.shape} (yes={100*y_train.mean():.2f}%), Test split: {X_test_raw.shape} (yes={100*y_test.mean():.2f}%)")
    print(f"M_pos_train: {mixed_prep.feature_engineer.pdays_median_pos_}")
    print(f"Euclidean shape: train={X_train_euc.shape}, test={X_test_euc.shape}")
    print(f"Mixed-Type numeric={X_train_enc.numeric.shape}, categorical={X_train_enc.categorical.shape}")
    print(f"Laporan disimpan di: {REPORT_PATH}")


if __name__ == "__main__":
    main()
