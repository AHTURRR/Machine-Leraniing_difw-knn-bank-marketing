"""Exploratory Data Analysis (EDA) and Data Understanding for UCI Bank Marketing.

Generates summary tables under outputs/eda/, figures under outputs/eda/figures/,
and a structured Markdown report at Laporan_EDA_DIFW_KNN.md in the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "bank-full.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "Laporan_EDA_DIFW_KNN.md"

NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
TARGET_COL = "y"
RANDOM_STATE = 42


def configure_plot_style() -> None:
    """Configure clean matplotlib defaults for publication-ready figures."""
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


def load_dataset(path: Path) -> pd.DataFrame:
    """Load bank-full.csv and validate expected schema."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {path}")
    df = pd.read_csv(path, sep=";")
    expected_cols = set(NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL])
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan pada dataset: {sorted(missing_cols)}")
    return df


def compute_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Compute basic dataset dimensions, duplicates, and type counts."""
    n_rows, n_cols = df.shape
    n_duplicates = int(df.duplicated().sum())
    n_nan = int(df.isna().sum().sum())
    target_counts = df[TARGET_COL].value_counts()
    n_yes = int(target_counts.get("yes", 0))
    n_no = int(target_counts.get("no", 0))
    imbalance_ratio = round(n_no / n_yes, 4) if n_yes > 0 else float("inf")
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_predictors": n_cols - 1,
        "n_numeric": len(NUMERIC_COLS),
        "n_categorical": len(CATEGORICAL_COLS),
        "n_explicit_nan": n_nan,
        "n_duplicate_rows": n_duplicates,
        "n_yes": n_yes,
        "pct_yes": round(100.0 * n_yes / n_rows, 4),
        "n_no": n_no,
        "pct_no": round(100.0 * n_no / n_rows, 4),
        "imbalance_ratio_no_per_yes": imbalance_ratio,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 3),
    }


def analyze_missing_and_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize explicit NaNs, unique values, and 'unknown' category counts."""
    rows: list[dict[str, Any]] = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        n_nan = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        if col in CATEGORICAL_COLS or col == TARGET_COL:
            n_unknown = int((series.astype(str).str.lower() == "unknown").sum())
            ftype = "Kategorikal" if col != TARGET_COL else "Target"
            top_val = str(series.value_counts().index[0])
            top_pct = round(100.0 * series.value_counts().iloc[0] / n_rows, 2)
        else:
            n_unknown = 0
            ftype = "Numerik"
            top_val = "-"
            top_pct = np.nan

        rows.append(
            {
                "feature": col,
                "role_type": ftype,
                "dtype": str(series.dtype),
                "n_nan": n_nan,
                "n_unique": n_unique,
                "n_unknown": n_unknown,
                "pct_unknown": round(100.0 * n_unknown / n_rows, 2),
                "mode_category": top_val,
                "mode_pct": top_pct,
            }
        )
    return pd.DataFrame(rows)


def compute_numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics including Q05, Q95, and Q95-Q05 range."""
    rows: list[dict[str, Any]] = []
    n_rows = len(df)
    for col in NUMERIC_COLS:
        s = df[col].astype(float)
        q05 = float(s.quantile(0.05))
        q25 = float(s.quantile(0.25))
        q50 = float(s.quantile(0.50))
        q75 = float(s.quantile(0.75))
        q95 = float(s.quantile(0.95))
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        n_outliers_iqr = int(((s < lower_fence) | (s > upper_fence)).sum())

        rows.append(
            {
                "feature": col,
                "count": int(s.count()),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std(ddof=1)), 4),
                "min": round(float(s.min()), 4),
                "q05": round(q05, 4),
                "q25": round(q25, 4),
                "median": round(q50, 4),
                "q75": round(q75, 4),
                "q95": round(q95, 4),
                "max": round(float(s.max()), 4),
                "iqr": round(iqr, 4),
                "q95_minus_q05": round(q95 - q05, 4),
                "skewness": round(float(s.skew()), 4),
                "n_negative": int((s < 0).sum()),
                "n_zero": int((s == 0).sum()),
                "pct_outliers_iqr": round(100.0 * n_outliers_iqr / n_rows, 2),
            }
        )
    return pd.DataFrame(rows)


def analyze_numeric_by_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compare numeric features across target classes with statistical tests."""
    y_bin = (df[TARGET_COL] == "yes").astype(int)
    rows: list[dict[str, Any]] = []
    for col in NUMERIC_COLS:
        s_yes = df.loc[y_bin == 1, col].astype(float)
        s_no = df.loc[y_bin == 0, col].astype(float)
        u_stat, u_pval = stats.mannwhitneyu(s_yes, s_no, alternative="two-sided")
        # Rank-biserial correlation: 1 - (2U)/(n1*n0) or signed difference
        n1, n0 = len(s_yes), len(s_no)
        auc_equiv = float(u_stat) / (n1 * n0)
        rank_biserial = 2.0 * auc_equiv - 1.0
        r_pb, pb_pval = stats.pointbiserialr(y_bin, df[col].astype(float))
        rows.append(
            {
                "feature": col,
                "mean_no": round(float(s_no.mean()), 4),
                "median_no": round(float(s_no.median()), 4),
                "q95_no": round(float(s_no.quantile(0.95)), 4),
                "mean_yes": round(float(s_yes.mean()), 4),
                "median_yes": round(float(s_yes.median()), 4),
                "q95_yes": round(float(s_yes.quantile(0.95)), 4),
                "point_biserial_r": round(float(r_pb), 4),
                "pb_pvalue": float(pb_pval),
                "mann_whitney_u": float(u_stat),
                "mw_pvalue": float(u_pval),
                "rank_biserial_r": round(rank_biserial, 4),
                "single_feature_auc": round(max(auc_equiv, 1.0 - auc_equiv), 4),
            }
        )
    return pd.DataFrame(rows)


def analyze_pdays(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deep-dive analysis on pdays = -1 vs contacted subset, previous, and poutcome."""
    y_bin = (df[TARGET_COL] == "yes").astype(int)
    not_contacted = df["pdays"] == -1
    contacted = df["pdays"] > -1
    pdays_zero = df["pdays"] == 0

    contacted_vals = df.loc[contacted, "pdays"].astype(float)
    q05_c = float(contacted_vals.quantile(0.05))
    q50_c = float(contacted_vals.quantile(0.50))
    q95_c = float(contacted_vals.quantile(0.95))

    # Cross-check consistency with previous and poutcome
    n_not_contacted = int(not_contacted.sum())
    n_prev_zero_when_neg1 = int((df.loc[not_contacted, "previous"] == 0).sum())
    n_pout_unk_when_neg1 = int((df.loc[not_contacted, "poutcome"] == "unknown").sum())
    n_contacted = int(contacted.sum())
    n_prev_gt0_when_contacted = int((df.loc[contacted, "previous"] > 0).sum())

    summary_meta = {
        "n_total": len(df),
        "n_not_contacted_neg1": n_not_contacted,
        "pct_not_contacted_neg1": round(100.0 * n_not_contacted / len(df), 2),
        "n_contacted_gt_neg1": n_contacted,
        "pct_contacted_gt_neg1": round(100.0 * n_contacted / len(df), 2),
        "n_pdays_zero": int(pdays_zero.sum()),
        "n_pdays_lt_neg1": int((df["pdays"] < -1).sum()),
        "prev_zero_given_neg1_match_pct": round(100.0 * n_prev_zero_when_neg1 / n_not_contacted, 2),
        "poutcome_unknown_given_neg1_pct": round(100.0 * n_pout_unk_when_neg1 / n_not_contacted, 2),
        "prev_gt0_given_contacted_pct": round(100.0 * n_prev_gt0_when_contacted / n_contacted, 2),
        "all_rows_q05": float(df["pdays"].quantile(0.05)),
        "all_rows_q95": float(df["pdays"].quantile(0.95)),
        "all_rows_q95_minus_q05": float(df["pdays"].quantile(0.95) - df["pdays"].quantile(0.05)),
        "contacted_min": float(contacted_vals.min()),
        "contacted_mean": round(float(contacted_vals.mean()), 2),
        "contacted_std": round(float(contacted_vals.std(ddof=1)), 2),
        "contacted_q05": round(q05_c, 2),
        "contacted_median": round(q50_c, 2),
        "contacted_q95": round(q95_c, 2),
        "contacted_max": float(contacted_vals.max()),
        "contacted_q95_minus_q05": round(q95_c - q05_c, 2),
        "yes_rate_not_contacted_pct": round(100.0 * y_bin[not_contacted].mean(), 2),
        "yes_rate_contacted_pct": round(100.0 * y_bin[contacted].mean(), 2),
    }

    # Bucketed analysis of pdays
    buckets = pd.Series("Belum pernah (-1)", index=df.index)
    buckets.loc[(df["pdays"] >= 1) & (df["pdays"] <= 90)] = "1-90 hari"
    buckets.loc[(df["pdays"] > 90) & (df["pdays"] <= 180)] = "91-180 hari"
    buckets.loc[(df["pdays"] > 180) & (df["pdays"] <= 365)] = "181-365 hari"
    buckets.loc[df["pdays"] > 365] = "> 365 hari"

    bucket_order = [
        "Belum pernah (-1)",
        "1-90 hari",
        "91-180 hari",
        "181-365 hari",
        "> 365 hari",
    ]
    tmp = pd.DataFrame({"pdays_bucket": buckets, "y_bin": y_bin, "pdays": df["pdays"]})
    grouped = (
        tmp.groupby("pdays_bucket", observed=False)
        .agg(
            count=("y_bin", "size"),
            yes_count=("y_bin", "sum"),
            yes_rate_pct=("y_bin", lambda x: round(100.0 * x.mean(), 2)),
            mean_pdays=("pdays", lambda x: round(float(x.mean()), 2)),
        )
        .reindex(bucket_order)
        .reset_index()
    )
    grouped["pct_total"] = (100.0 * grouped["count"] / len(df)).round(2)

    # Also compare poutcome breakdown within contacted vs non-contacted
    return grouped, summary_meta


def analyze_duration_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Quantify empirical target leakage of the duration feature."""
    y_bin = (df[TARGET_COL] == "yes").astype(int)
    dur = df["duration"].astype(float)
    auc_dur = float(roc_auc_score(y_bin, dur))
    r_pb, p_pb = stats.pointbiserialr(y_bin, dur)

    bins = [-0.1, 60, 120, 180, 300, 600, 1200, float("inf")]
    labels = [
        "0-60 dtk",
        "61-120 dtk",
        "121-180 dtk",
        "181-300 dtk",
        "301-600 dtk",
        "601-1200 dtk",
        "> 1200 dtk",
    ]
    dur_bucket = pd.cut(dur, bins=bins, labels=labels)
    tmp = pd.DataFrame({"duration_bucket": dur_bucket, "y_bin": y_bin})
    bucket_df = (
        tmp.groupby("duration_bucket", observed=False)
        .agg(
            count=("y_bin", "size"),
            yes_count=("y_bin", "sum"),
            yes_rate_pct=("y_bin", lambda x: round(100.0 * x.mean(), 2)),
        )
        .reset_index()
    )
    bucket_df["pct_total"] = (100.0 * bucket_df["count"] / len(df)).round(2)

    meta = {
        "n_duration_zero": int((dur == 0).sum()),
        "yes_count_duration_zero": int(y_bin[dur == 0].sum()),
        "single_feature_roc_auc": round(auc_dur, 4),
        "point_biserial_r": round(float(r_pb), 4),
        "pb_pvalue": float(p_pb),
        "median_no_sec": round(float(dur[y_bin == 0].median()), 2),
        "median_yes_sec": round(float(dur[y_bin == 1].median()), 2),
        "mean_no_sec": round(float(dur[y_bin == 0].mean()), 2),
        "mean_yes_sec": round(float(dur[y_bin == 1].mean()), 2),
    }
    return bucket_df, meta


def analyze_categorical_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Chi-Square, Cramer's V, and per-category conversion rates."""
    y_bin = (df[TARGET_COL] == "yes").astype(int)
    baseline_yes_rate = float(y_bin.mean())
    n_rows = len(df)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for col in CATEGORICAL_COLS:
        ct = pd.crosstab(df[col], df[TARGET_COL])
        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        min_dim = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n_rows * min_dim))) if min_dim > 0 else 0.0

        summary_rows.append(
            {
                "feature": col,
                "n_categories": int(df[col].nunique()),
                "chi2": round(float(chi2), 2),
                "dof": int(dof),
                "p_value": float(p_val),
                "cramers_v": round(cramers_v, 4),
            }
        )

        grouped = (
            df.assign(y_bin=y_bin)
            .groupby(col, observed=False)
            .agg(
                count=("y_bin", "size"),
                yes_count=("y_bin", "sum"),
                yes_rate=("y_bin", "mean"),
            )
            .reset_index()
            .sort_values("yes_rate", ascending=False)
        )
        for _, row in grouped.iterrows():
            yes_rate = float(row["yes_rate"])
            detail_rows.append(
                {
                    "feature": col,
                    "category": str(row[col]),
                    "count": int(row["count"]),
                    "pct_total": round(100.0 * row["count"] / n_rows, 2),
                    "yes_count": int(row["yes_count"]),
                    "yes_rate_pct": round(100.0 * yes_rate, 2),
                    "lift_vs_baseline": round(yes_rate / baseline_yes_rate, 3),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("cramers_v", ascending=False)
    detail_df = pd.DataFrame(detail_rows)
    return summary_df, detail_df


def compute_mutual_information(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Mutual Information across all original predictors (and leakage flag)."""
    y_bin = (df[TARGET_COL] == "yes").astype(int).to_numpy()
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS

    x_encoded = pd.DataFrame(index=df.index)
    discrete_mask: list[bool] = []

    for col in NUMERIC_COLS:
        x_encoded[col] = df[col].astype(float)
        discrete_mask.append(False)

    encoder = OrdinalEncoder(dtype=np.int64)
    cat_vals = encoder.fit_transform(df[CATEGORICAL_COLS].astype(str))
    for idx, col in enumerate(CATEGORICAL_COLS):
        x_encoded[col] = cat_vals[:, idx]
        discrete_mask.append(True)

    mi_scores = mutual_info_classif(
        x_encoded,
        y_bin,
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )

    rows = []
    for col, score in zip(feature_cols, mi_scores):
        rows.append(
            {
                "feature": col,
                "feature_type": "Numerik" if col in NUMERIC_COLS else "Kategorikal",
                "is_leakage": col == "duration",
                "mutual_information": round(float(score), 5),
            }
        )
    return pd.DataFrame(rows).sort_values("mutual_information", ascending=False)


def create_figures(
    df: pd.DataFrame,
    unknown_df: pd.DataFrame,
    pdays_buckets: pd.DataFrame,
    dur_buckets: pd.DataFrame,
    cat_summary: pd.DataFrame,
    cat_detail: pd.DataFrame,
    mi_df: pd.DataFrame,
) -> list[Path]:
    """Generate EDA visualizations and return saved file paths."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # 1. Target distribution & Unknown categories
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    target_counts = df[TARGET_COL].value_counts().reindex(["no", "yes"])
    colors = ["#4C72B0", "#DD8452"]
    bars = axes[0].bar(target_counts.index, target_counts.values, color=colors, width=0.55)
    axes[0].set_title("Distribusi Kelas Target (y)")
    axes[0].set_ylabel("Jumlah Observasi")
    axes[0].set_ylim(0, target_counts.max() * 1.18)
    for bar, count in zip(bars, target_counts.values):
        pct = 100.0 * count / len(df)
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 700,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    unk_sub = unknown_df[unknown_df["n_unknown"] > 0].sort_values("pct_unknown", ascending=True)
    bars_u = axes[1].barh(unk_sub["feature"], unk_sub["pct_unknown"], color="#8172B3")
    axes[1].set_title('Proporsi Kategori "unknown" per Fitur')
    axes[1].set_xlabel('Persentase "unknown" (%)')
    axes[1].set_xlim(0, max(unk_sub["pct_unknown"].max() * 1.25, 10))
    for bar, (_, r) in zip(bars_u, unk_sub.iterrows()):
        axes[1].text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{r['pct_unknown']:.2f}% ({int(r['n_unknown']):,})",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    p1 = FIGURES_DIR / "01_target_and_unknown_distribution.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # 2. Numeric distributions with Q05, Median, Q95 markers (6 non-pdays or 6 numeric features)
    plot_num_cols = ["age", "balance", "day", "campaign", "previous", "duration"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8))
    for ax, col in zip(axes.flat, plot_num_cols):
        s = df[col].astype(float)
        q05, q50, q95 = s.quantile([0.05, 0.50, 0.95])
        # Clip display at 99th percentile for readability on heavily skewed features
        lo = float(s.quantile(0.005)) if col == "balance" else float(s.min())
        hi = float(s.quantile(0.99)) if col in {"balance", "campaign", "previous", "duration"} else float(s.max())
        clipped = s.clip(lower=lo, upper=hi)
        ax.hist(clipped, bins=35, color="#4C72B0", alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(q05, color="#2CA02C", linestyle="--", linewidth=1.3, label=f"Q05={q05:.0f}")
        ax.axvline(q50, color="#D62728", linestyle="-", linewidth=1.5, label=f"Med={q50:.0f}")
        ax.axvline(q95, color="#FF7F0E", linestyle="--", linewidth=1.3, label=f"Q95={q95:.0f}")
        title_suffix = ""
        if col == "balance":
            title_suffix = " (clip P0.5–P99)"
        elif col in {"campaign", "previous", "duration"}:
            title_suffix = " (clip <=P99)"
        ax.set_title(f"{col}{title_suffix}\nskew={s.skew():.2f}, Q95-Q05={q95 - q05:.0f}")
        ax.set_ylabel("Frekuensi")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Distribusi Fitur Numerik & Rentang Kuantil (Q05–Q95)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = FIGURES_DIR / "02_numeric_distributions_and_outliers.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # 3. pdays deep dive & duration leakage proof
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # Panel A: pdays contacted histogram
    pdays_pos = df.loc[df["pdays"] > -1, "pdays"]
    axes[0].hist(pdays_pos, bins=35, color="#55A868", alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[0].axvline(pdays_pos.quantile(0.05), color="#2C3E50", linestyle="--", label=f"Q05={pdays_pos.quantile(0.05):.0f}")
    axes[0].axvline(pdays_pos.median(), color="#D62728", linestyle="-", label=f"Med={pdays_pos.median():.0f}")
    axes[0].axvline(pdays_pos.quantile(0.95), color="#FF7F0E", linestyle="--", label=f"Q95={pdays_pos.quantile(0.95):.0f}")
    pct_neg1 = 100.0 * (df["pdays"] == -1).mean()
    axes[0].set_title(f"Distribusi pdays (Subset > -1)\npdays=-1 mencapai {pct_neg1:.2f}% data")
    axes[0].set_xlabel("pdays (hari)")
    axes[0].set_ylabel("Frekuensi")
    axes[0].legend(fontsize=8)

    # Panel B: pdays bucket conversion rate
    baseline_pct = 100.0 * (df[TARGET_COL] == "yes").mean()
    b_p = axes[1].bar(
        range(len(pdays_buckets)),
        pdays_buckets["yes_rate_pct"],
        color=["#7F7F7F", "#55A868", "#4C72B0", "#8172B3", "#CCB974"],
    )
    axes[1].axhline(baseline_pct, color="black", linestyle="--", linewidth=1, label=f"Baseline ({baseline_pct:.2f}%)")
    axes[1].set_xticks(range(len(pdays_buckets)))
    axes[1].set_xticklabels(pdays_buckets["pdays_bucket"], rotation=22, ha="right", fontsize=8.5)
    axes[1].set_title("Konversi (y=yes) per Kelompok pdays")
    axes[1].set_ylabel("Tingkat Konversi (%)")
    axes[1].set_ylim(0, max(pdays_buckets["yes_rate_pct"].max() * 1.22, 35))
    axes[1].legend(fontsize=8)
    for bar, val in zip(b_p, pdays_buckets["yes_rate_pct"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{val:.1f}%", ha="center", fontsize=8.5)

    # Panel C: duration bucket conversion rate (leakage evidence)
    b_d = axes[2].bar(range(len(dur_buckets)), dur_buckets["yes_rate_pct"], color="#C44E52", alpha=0.88)
    axes[2].axhline(baseline_pct, color="black", linestyle="--", linewidth=1, label=f"Baseline ({baseline_pct:.2f}%)")
    axes[2].set_xticks(range(len(dur_buckets)))
    axes[2].set_xticklabels(dur_buckets["duration_bucket"], rotation=25, ha="right", fontsize=8.5)
    axes[2].set_title("Bukti Leakage: Konversi vs duration\n(Wajib Dihapus untuk Pre-Call)")
    axes[2].set_ylabel("Tingkat Konversi (%)")
    axes[2].set_ylim(0, max(dur_buckets["yes_rate_pct"].max() * 1.18, 70))
    axes[2].legend(fontsize=8)
    for bar, val in zip(b_d, dur_buckets["yes_rate_pct"]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{val:.1f}%", ha="center", fontsize=8.5)

    fig.tight_layout()
    p3 = FIGURES_DIR / "03_pdays_and_duration_leakage.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # 4. Feature discriminative power (Mutual Information & Cramer's V)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mi_plot = mi_df.sort_values("mutual_information", ascending=True)
    mi_colors = ["#C44E52" if lk else "#4C72B0" for lk in mi_plot["is_leakage"]]
    bars_mi = axes[0].barh(mi_plot["feature"], mi_plot["mutual_information"], color=mi_colors)
    axes[0].set_title("Mutual Information terhadap Target y\n(Merah = Target Leakage)")
    axes[0].set_xlabel("Mutual Information (nats)")
    axes[0].set_xlim(0, mi_plot["mutual_information"].max() * 1.22)
    for bar, val in zip(bars_mi, mi_plot["mutual_information"]):
        axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8.5)

    cv_plot = cat_summary.sort_values("cramers_v", ascending=True)
    bars_cv = axes[1].barh(cv_plot["feature"], cv_plot["cramers_v"], color="#55A868")
    axes[1].set_title("Kekuatan Asosiasi Fitur Kategorikal (Cramer's V)")
    axes[1].set_xlabel("Cramer's V")
    axes[1].set_xlim(0, cv_plot["cramers_v"].max() * 1.22)
    for bar, val in zip(bars_cv, cv_plot["cramers_v"]):
        axes[1].text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=8.5)

    fig.tight_layout()
    p4 = FIGURES_DIR / "04_feature_discriminative_power.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    # 5. Conversion rates for key categorical predictors
    top_cat_cols = ["poutcome", "month", "contact", "housing", "marital", "job"]
    baseline_pct = 100.0 * (df[TARGET_COL] == "yes").mean()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, col in zip(axes.flat, top_cat_cols):
        sub = cat_detail[cat_detail["feature"] == col].sort_values("yes_rate_pct", ascending=True)
        bars_c = ax.barh(sub["category"], sub["yes_rate_pct"], color="#4C72B0", alpha=0.88)
        ax.axvline(baseline_pct, color="#D62728", linestyle="--", linewidth=1.2, label=f"Baseline ({baseline_pct:.1f}%)")
        ax.set_title(f"Tingkat Konversi (y=yes) — {col}")
        ax.set_xlabel("Konversi (%)")
        ax.set_xlim(0, max(sub["yes_rate_pct"].max() * 1.22, 20))
        ax.tick_params(axis="y", labelsize=8.5)
        ax.legend(fontsize=8, loc="lower right")
        for bar, val in zip(bars_c, sub["yes_rate_pct"]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)
    fig.suptitle("Analisis Tingkat Konversi pada Fitur Kategorikal Utama", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p5 = FIGURES_DIR / "05_categorical_conversion_rates.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p5)

    return saved_paths


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table without extra deps."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float) and np.isnan(v):
                vals.append("-")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_markdown_report(
    overview: dict[str, Any],
    unknown_df: pd.DataFrame,
    numeric_profile: pd.DataFrame,
    numeric_by_target: pd.DataFrame,
    pdays_buckets: pd.DataFrame,
    pdays_meta: dict[str, Any],
    dur_buckets: pd.DataFrame,
    dur_meta: dict[str, Any],
    cat_summary: pd.DataFrame,
    cat_detail: pd.DataFrame,
    mi_df: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Build Laporan_EDA_DIFW_KNN.md combining empirical outputs and research implications."""
    unknown_only = unknown_df[unknown_df["n_unknown"] > 0][
        ["feature", "n_unique", "n_unknown", "pct_unknown", "mode_category", "mode_pct"]
    ]
    top_mi_noleak = mi_df[~mi_df["is_leakage"]].head(5)

    poutcome_sub = cat_detail[cat_detail["feature"] == "poutcome"][
        ["category", "count", "pct_total", "yes_count", "yes_rate_pct", "lift_vs_baseline"]
    ]
    month_sub = cat_detail[cat_detail["feature"] == "month"][
        ["category", "count", "pct_total", "yes_count", "yes_rate_pct", "lift_vs_baseline"]
    ]
    contact_sub = cat_detail[cat_detail["feature"] == "contact"][
        ["category", "count", "pct_total", "yes_count", "yes_rate_pct", "lift_vs_baseline"]
    ]
    job_sub = cat_detail[cat_detail["feature"] == "job"][
        ["category", "count", "pct_total", "yes_count", "yes_rate_pct", "lift_vs_baseline"]
    ]

    num_compact = numeric_profile[
        [
            "feature",
            "mean",
            "std",
            "min",
            "q05",
            "median",
            "q95",
            "max",
            "q95_minus_q05",
            "skewness",
            "pct_outliers_iqr",
        ]
    ]
    num_target_compact = numeric_by_target[
        [
            "feature",
            "mean_no",
            "median_no",
            "mean_yes",
            "median_yes",
            "point_biserial_r",
            "rank_biserial_r",
            "single_feature_auc",
            "mw_pvalue",
        ]
    ]

    bal_row = numeric_profile.loc[numeric_profile["feature"] == "balance"].iloc[0]
    poutcome_unk_pct = float(unknown_df.loc[unknown_df["feature"] == "poutcome", "pct_unknown"].iloc[0])
    contact_unk_pct = float(unknown_df.loc[unknown_df["feature"] == "contact", "pct_unknown"].iloc[0])
    contacted_multiplier = (
        pdays_meta["yes_rate_contacted_pct"] / pdays_meta["yes_rate_not_contacted_pct"]
        if pdays_meta["yes_rate_not_contacted_pct"] > 0
        else 0.0
    )
    dur_first_bucket = dur_buckets.iloc[0]
    dur_last_bucket = dur_buckets.iloc[-1]
    n_test_20 = int(np.ceil(overview["n_rows"] * 0.2))
    n_train_80 = overview["n_rows"] - n_test_20

    lines = [
        "# LAPORAN DATA UNDERSTANDING & EXPLORATORY DATA ANALYSIS (EDA)",
        "",
        "**Dataset:** `bank-full.csv` (UCI Bank Marketing Dataset — ID 222)  ",
        "**Tujuan:** Mendukung desain algoritma *Domain-Informed Feature-Weighted KNN* (`DIFW-KNN`) pada skenario prediksi pra-panggilan (*pre-call telemarketing prediction*).",
        "",
        "---",
        "",
        "## 1. Profil Umum & Kualitas Data (*Data Understanding*)",
        "",
        f"- **Dimensi Dataset:** `{overview['n_rows']:,}` observasi × `{overview['n_cols']}` kolom (`{overview['n_predictors']}` prediktor + 1 target `y`).",
        f"- **Komposisi Tipe Fitur:** `{overview['n_numeric']}` fitur numerik (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) dan `{overview['n_categorical']}` fitur kategorikal (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`).",
        f"- **Missing Values Eksplisit (`NaN`):** `{overview['n_explicit_nan']}` sel kosong.",
        f"- **Baris Duplikat Eksak:** `{overview['n_duplicate_rows']}` baris.",
        f"- **Distribusi Kelas Target (`y`):**",
        f"  - Kelas negatif (`no`): `{overview['n_no']:,}` (`{overview['pct_no']:.2f}%`)",
        f"  - Kelas positif (`yes`): `{overview['n_yes']:,}` (`{overview['pct_yes']:.2f}%`)",
        f"  - **Imbalance Ratio:** `{overview['imbalance_ratio_no_per_yes']:.2f} : 1` (kelas minoritas `yes` sebesar `{overview['pct_yes']:.2f}%`).",
        "",
        "### Ringkasan Kategori `unknown` (Implicit Missing Values)",
        "",
        f"Tidak terdapat nilai `NaN`, namun terdapat nilai string `\"unknown\"` pada {len(unknown_only)} fitur kategorikal:",
        "",
        df_to_markdown_table(unknown_only),
        "",
        f"> **Catatan Peneliti:** Pada `poutcome` (`{poutcome_unk_pct:.2f}%` `unknown`) dan `contact` (`{contact_unk_pct:.2f}%` `unknown`), label `\"unknown\"` membawa sinyal informatif yang kuat (misalnya ketiadaan riwayat kampanye sebelumnya atau tipe kanal lama), sehingga **tidak boleh diimputasi dengan modus atau dihapus**. Label `\"unknown\"` wajib dipertahankan sebagai kategori valid.",
        "",
        "---",
        "",
        "## 2. Profil Statistik Fitur Numerik & Implikasi *Mixed-Type Distance*",
        "",
        df_to_markdown_table(num_compact),
        "",
        "### Perbandingan Fitur Numerik Berdasarkan Kelas Target (`y=yes` vs `y=no`)",
        "",
        df_to_markdown_table(num_target_compact),
        "",
        "> **Temuan Kritis untuk Formulasi Jarak (`DIFW-KNN`):**",
        f"> 1. Fitur `balance` (skewness `{bal_row['skewness']}`), `campaign`, dan `previous` sangat *right-skewed* serta memiliki outlier ekstrem. Pengukuran jarak Euclidean/Min-Max standar akan terkompresi oleh nilai ekstrem (`balance` max `{bal_row['max']:,.0f}` vs `Q95-Q05` sebesar `{bal_row['q95_minus_q05']:,.1f}`). Oleh karena itu, penggunaan **normalisasi interkuantil terklip $\\min(1, |x_j - z_j| / (Q_{{95,j}} - Q_{{05,j}}))$** pada *Mixed-Type Distance* terbukti sangat tepat secara empiris.",
        f"> 2. Perhatikan `pdays` pada tabel profil global: karena `{pdays_meta['pct_not_contacted_neg1']:.2f}%` bernilai `-1`, nilai `Q05` dan `median` bernilai `-1`, sehingga **`Q95 - Q05` global terdistorsi oleh mayoritas nilai sentinel `-1`**, seperti dijelaskan lebih lanjut pada Bagian 3.",
        "",
        "---",
        "",
        "## 3. Analisis Khusus `pdays` (Bukti Empiris Perlunya Rekayasa Fitur)",
        "",
        f"- **Proporsi `pdays = -1` (belum pernah dihubungi):** `{pdays_meta['n_not_contacted_neg1']:,}` observasi (**`{pdays_meta['pct_not_contacted_neg1']}%`**).",
        f"- **Proporsi `pdays > -1` (pernah dihubungi sebelumnya):** `{pdays_meta['n_contacted_gt_neg1']:,}` observasi (**`{pdays_meta['pct_contacted_gt_neg1']}%`**), dengan `pdays = 0` sebanyak `{pdays_meta['n_pdays_zero']}` observasi (minimum pada subset pernah dihubungi adalah `{pdays_meta['contacted_min']:.0f}` hari).",
        f"- **Konsistensi Lintas Fitur:** Dari seluruh baris dengan `pdays = -1`, `{pdays_meta['prev_zero_given_neg1_match_pct']}%` memiliki `previous = 0` dan `{pdays_meta['poutcome_unknown_given_neg1_pct']}%` memiliki `poutcome = 'unknown'`. Sebaliknya, pada baris `pdays > -1`, `{pdays_meta['prev_gt0_given_contacted_pct']}%` memiliki `previous > 0`.",
        f"- **Distribusi `pdays` KHUSUS Subset Pernah Dihubungi (`pdays > -1`):**",
        f"  - Mean = `{pdays_meta['contacted_mean']}` hari, Std = `{pdays_meta['contacted_std']}` hari",
        f"  - `Q05` = `{pdays_meta['contacted_q05']}`, Median = `{pdays_meta['contacted_median']}`, `Q95` = `{pdays_meta['contacted_q95']}`, Max = `{pdays_meta['contacted_max']:.0f}`",
        f"  - Rentang `Q95 - Q05` pada subset pernah dihubungi = **`{pdays_meta['contacted_q95_minus_q05']}`** hari (berbeda jauh dari `Q95 - Q05` global = `{pdays_meta['all_rows_q95_minus_q05']}`).",
        f"- **Dampak terhadap Tingkat Konversi (`y=yes`):**",
        f"  - Belum pernah dihubungi (`pdays = -1`): **`{pdays_meta['yes_rate_not_contacted_pct']}%`**",
        f"  - Pernah dihubungi (`pdays > -1`): **`{pdays_meta['yes_rate_contacted_pct']}%`** (`{contacted_multiplier:.2f}x` lebih tinggi)",
        "",
        df_to_markdown_table(pdays_buckets),
        "",
        "> **Keputusan Metodologis (`Preprocessing`):**",
        "> Jika `pdays = -1` langsung diganti menjadi `pdays_recency = 0` seperti draf awal, nasabah yang baru dihubungi `1` hari lalu menjadi bertetangga dekat dengan nasabah yang belum pernah dihubungi (`0`). Sebaliknya, pada desain perbaikan: (1) buat fitur biner `previously_contacted` (`1` jika `pdays > -1`, `0` jika `-1`), dan (2) definisikan `pdays_recency = 1 / (1 + pdays)` jika pernah dihubungi dan `0` jika belum pernah dihubungi (sehingga `0` = tidak pernah dihubungi, mendekati `1` = baru saja dihubungi, sementara efek non-monoton pada kontak `> 365` hari tetap dapat tertangkap oleh ketetanggaan lokal KNN). Rentang `Q95 - Q05` untuk `pdays_recency` harus diperiksa dari fold training.",
        "",
        "---",
        "",
        "## 4. Analisis Fitur `duration` (Bukti Empiris *Target Leakage*)",
        "",
        f"- **Korelasi & Daya Pisah Tunggal:** Fitur `duration` sendiri menghasilkan **ROC-AUC = `{dur_meta['single_feature_roc_auc']}`** dan Point-Biserial $r$ = `{dur_meta['point_biserial_r']}`.",
        f"- **Median Durasi:** Kelas `no` = `{dur_meta['median_no_sec']}` detik (mean `{dur_meta['mean_no_sec']}` dtk) vs Kelas `yes` = `{dur_meta['median_yes_sec']}` detik (mean `{dur_meta['mean_yes_sec']}` dtk).",
        f"- **Durasi = 0 detik:** Terdapat `{dur_meta['n_duration_zero']}` observasi, dengan `{dur_meta['yes_count_duration_zero']}` berlabel `yes`.",
        "",
        df_to_markdown_table(dur_buckets),
        "",
        f"> **Validasi Keputusan Desain:** Tingkat konversi melonjak dari `{dur_first_bucket['yes_rate_pct']:.2f}%` (`{dur_first_bucket['duration_bucket']}`) menjadi `{dur_last_bucket['yes_rate_pct']:.2f}%` (`{dur_last_bucket['duration_bucket']}`) karena durasi panggilan adalah akibat langsung dari ketertarikan nasabah selama percakapan berlangsung. Untuk tujuan *pre-call targeting* (memilih siapa yang harus ditelepon sebelum panggilan dilakukan), **fitur `duration` wajib dihapus sepenuhnya sebelum pemodelan**.",
        "",
        "---",
        "",
        "## 5. Kekuatan Diskriminatif Fitur & Tingkat Konversi Kategorikal",
        "",
        "### Peringkat Mutual Information (Semua Prediktor Asli)",
        "",
        df_to_markdown_table(mi_df),
        "",
        "### Uji Asosiasi Chi-Square & Cramér's V Fitur Kategorikal",
        "",
        df_to_markdown_table(cat_summary),
        "",
        "### Rincian Konversi pada Fitur Kategorikal Berpengaruh",
        "",
        "**a. `poutcome` (Hasil Kampanye Sebelumnya)**",
        "",
        df_to_markdown_table(poutcome_sub),
        "",
        "**b. `contact` (Kanal Komunikasi)**",
        "",
        df_to_markdown_table(contact_sub),
        "",
        "**c. `month` (Bulan Kontak Terakhir)**",
        "",
        df_to_markdown_table(month_sub),
        "",
        "**d. `job` (Jenis Pekerjaan)**",
        "",
        df_to_markdown_table(job_sub),
        "",
        "> **Implikasi untuk *Feature Weighting* (`DIFW-KNN`):**",
        f"> Di luar `duration` (yang dihapus), prediktor pra-panggilan terkuat menurut Mutual Information adalah {', '.join(f'`{f}`' for f in top_mi_noleak['feature'])}. Sementara fitur dengan informasi marginal kecil akan menambah noise pada jarak seragam. Hal ini memvalidasi hipotesis bahwa **KNN dengan bobot seragam (`E0`–`E2`) rentan terdilusi oleh fitur berdaya pisah rendah**, sehingga *Supervised Triplet Feature Weighting* (`E3`, `E5`) diharapkan memberikan bobot lebih besar pada fitur relevan dan menekan fitur *noise*.",
        "",
        "---",
        "",
        "## 6. Ringkasan Artefak Visual & Kesiapan Tahap Berikutnya",
        "",
        "Seluruh grafik dan tabel mentah telah disimpan di folder `outputs/eda/`:",
    ]

    for p in figure_paths:
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        lines.append(f"- [{p.name}](file:///{p.as_posix()}) (`{rel}`)")

    lines.extend(
        [
            "",
            "### Kontrak *Preprocessing* Menuju Tahap Eksperimen (`02_preprocessing.py`)",
            "1. **Deterministic Cleaning (Pre-Split Safe):** Hapus kolom `duration`; pertahankan kategori `\"unknown\"`; buat fitur `previously_contacted` (`1` jika `pdays > -1`, `0` jika `-1`) dan `pdays_recency = 1 / (1 + pdays)` untuk `pdays > -1` (`0` jika `-1`), lalu hapus `pdays` mentah.",
            f"2. **Stratified Split (80:20):** Bagi data `{overview['n_rows']:,}` baris menjadi `{n_train_80:,}` data latih dan `{n_test_20:,}` data uji (`random_state=42`) dengan proporsi `yes = {overview['pct_yes']:.2f}%` identik di kedua subset.",
            f"3. **Training-Only Parameter Fitting:** Seluruh parameter `StandardScaler`/`OneHotEncoder` (jalur Euclidean `E0`–`E1`) dan kuantil `(Q05, Q95)` (jalur Mixed-Type `E2`–`E5`) hanya dihitung dari data latih (`{n_train_80:,}` baris atau per *fold* CV), tidak pernah menyentuh *test set*.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Execute complete EDA workflow and save all outputs."""
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)

    overview = compute_dataset_overview(df)
    unknown_df = analyze_missing_and_unknown(df)
    numeric_profile = compute_numeric_profile(df)
    numeric_by_target = analyze_numeric_by_target(df)
    pdays_buckets, pdays_meta = analyze_pdays(df)
    dur_buckets, dur_meta = analyze_duration_leakage(df)
    cat_summary, cat_detail = analyze_categorical_features(df)
    mi_df = compute_mutual_information(df)

    # Save CSV outputs
    pd.DataFrame([overview]).to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)
    unknown_df.to_csv(OUTPUT_DIR / "missing_and_unknown_summary.csv", index=False)
    df[TARGET_COL].value_counts().rename_axis("y").reset_index(name="count").to_csv(
        OUTPUT_DIR / "target_distribution.csv", index=False
    )
    numeric_profile.to_csv(OUTPUT_DIR / "numeric_profile.csv", index=False)
    numeric_by_target.to_csv(OUTPUT_DIR / "numeric_by_target.csv", index=False)
    pdays_buckets.to_csv(OUTPUT_DIR / "pdays_analysis.csv", index=False)
    pd.DataFrame([pdays_meta]).to_csv(OUTPUT_DIR / "pdays_metadata.csv", index=False)
    dur_buckets.to_csv(OUTPUT_DIR / "duration_leakage_analysis.csv", index=False)
    pd.DataFrame([dur_meta]).to_csv(OUTPUT_DIR / "duration_leakage_metadata.csv", index=False)
    cat_summary.to_csv(OUTPUT_DIR / "categorical_discriminative_summary.csv", index=False)
    cat_detail.to_csv(OUTPUT_DIR / "categorical_conversion_details.csv", index=False)
    mi_df.to_csv(OUTPUT_DIR / "feature_mutual_information.csv", index=False)

    figure_paths = create_figures(
        df=df,
        unknown_df=unknown_df,
        pdays_buckets=pdays_buckets,
        dur_buckets=dur_buckets,
        cat_summary=cat_summary,
        cat_detail=cat_detail,
        mi_df=mi_df,
    )

    report_text = build_markdown_report(
        overview=overview,
        unknown_df=unknown_df,
        numeric_profile=numeric_profile,
        numeric_by_target=numeric_by_target,
        pdays_buckets=pdays_buckets,
        pdays_meta=pdays_meta,
        dur_buckets=dur_buckets,
        dur_meta=dur_meta,
        cat_summary=cat_summary,
        cat_detail=cat_detail,
        mi_df=mi_df,
        figure_paths=figure_paths,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("EDA selesai dijalankan.")
    print(f"Rows: {overview['n_rows']}, Cols: {overview['n_cols']}, Duplicates: {overview['n_duplicate_rows']}, NaN: {overview['n_explicit_nan']}")
    print(f"Target yes: {overview['n_yes']} ({overview['pct_yes']:.2f}%), no: {overview['n_no']} ({overview['pct_no']:.2f}%)")
    print(f"Laporan disimpan di: {REPORT_PATH}")
    print(f"Output tabel & grafik di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
