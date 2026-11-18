"""Leak-free preprocessing and feature engineering for DIFW-KNN Bank Marketing.

Implements:
1. Deterministic pre-split cleaning (drop leakage feature 'duration', preserve
   'unknown' category labels, map target 'y' -> {0, 1}).
2. Stratified 80:20 train/test split (random_state=42).
3. Training-only domain feature engineering for 'pdays' -> ('previously_contacted',
   'pdays_recency') yielding 16 domain features (6 numeric + 10 categorical/binary).
4. BaselineEuclideanPreprocessor (StandardScaler + OneHotEncoder) for E0/E1.
5. MixedTypePreprocessor (Q05-Q95 range scaling + categorical mismatch) for E2-E5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_NUMERIC_COLS = ["age", "balance", "day", "campaign", "pdays", "previous"]
RAW_CATEGORICAL_COLS = [
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
LEAKAGE_COLS = ["duration"]
TARGET_COL = "y"

DOMAIN_NUMERIC_COLS = [
    "age",
    "balance",
    "day",
    "campaign",
    "previous",
    "pdays_recency",
]
DOMAIN_CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "previously_contacted",
]
DOMAIN_FEATURE_COLS = DOMAIN_NUMERIC_COLS + DOMAIN_CATEGORICAL_COLS

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load bank-full.csv using semicolon delimiter and validate core columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    required = set(RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS + LEAKAGE_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {sorted(missing)}")
    return df


def clean_deterministic(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Perform safe row-independent cleaning before train-test splitting.

    - Removes target-leakage column 'duration'.
    - Strips whitespace in categorical columns and preserves 'unknown' labels.
    - Maps target 'y' ('yes' -> 1, 'no' -> 0).
    """
    cleaned = df.drop(columns=LEAKAGE_COLS, errors="ignore").copy()
    for col in RAW_CATEGORICAL_COLS:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    y_raw = cleaned.pop(TARGET_COL).astype(str).str.strip().str.lower()
    invalid_targets = set(y_raw.unique()) - {"yes", "no"}
    if invalid_targets:
        raise ValueError(f"Nilai target tidak valid ditemukan: {sorted(invalid_targets)}")
    y = (y_raw == "yes").astype(np.int64)
    return cleaned, y


def stratified_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Training-only feature engineer for 'pdays' -> recency and contact flag.

    Learns M_pos_train = median(pdays | pdays > -1) strictly from training data,
    then transforms pdays into:
      - previously_contacted: '1' if pdays > -1 else '0'
      - pdays_recency: 1 / (1 + pdays / M_pos_train) if pdays > -1 else 0.0
    and drops raw 'pdays'.
    """

    def __init__(self) -> None:
        self.pdays_median_pos_: float | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> DomainFeatureEngineer:
        """Estimate median of positive pdays strictly on training rows."""
        if "pdays" not in X.columns:
            raise ValueError("Kolom 'pdays' wajib ada sebelum DomainFeatureEngineer.")
        pos_pdays = X.loc[X["pdays"] > -1, "pdays"].astype(float)
        if pos_pdays.empty:
            raise ValueError("Tidak ada observasi dengan pdays > -1 pada data training.")
        median_val = float(pos_pdays.median())
        if median_val <= 0:
            raise ValueError(f"Median pdays > -1 harus > 0, diperoleh {median_val}")
        self.pdays_median_pos_ = median_val
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform pdays and return DataFrame with ordered 16 domain features."""
        if self.pdays_median_pos_ is None:
            raise RuntimeError("DomainFeatureEngineer belum di-fit pada data training.")
        out = X.copy()
        pdays_vals = out["pdays"].astype(float)
        contacted_mask = pdays_vals > -1

        out["previously_contacted"] = np.where(contacted_mask, "1", "0")
        recency = np.zeros(len(out), dtype=np.float64)
        recency[contacted_mask.to_numpy()] = 1.0 / (
            1.0 + (pdays_vals[contacted_mask].to_numpy() / self.pdays_median_pos_)
        )
        out["pdays_recency"] = recency
        out = out.drop(columns=["pdays"])

        for col in DOMAIN_CATEGORICAL_COLS:
            out[col] = out[col].astype(str)
        for col in DOMAIN_NUMERIC_COLS:
            out[col] = out[col].astype(np.float64)
        return out[DOMAIN_FEATURE_COLS]


class BaselineEuclideanPreprocessor(BaseEstimator, TransformerMixin):
    """Leak-free preprocessing pipeline for Standard/Optimized KNN (E0, E1).

    Applies training-only DomainFeatureEngineer, then StandardScaler on the
    6 numeric features and OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    on the 10 categorical/binary features.
    """

    def __init__(self) -> None:
        self.feature_engineer = DomainFeatureEngineer()
        self.column_transformer: ColumnTransformer | None = None
        self.output_feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> BaselineEuclideanPreprocessor:
        """Fit domain engineering, StandardScaler, and OneHotEncoder on training X."""
        X_dom = self.feature_engineer.fit_transform(X)
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), DOMAIN_NUMERIC_COLS),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    DOMAIN_CATEGORICAL_COLS,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.column_transformer.fit(X_dom)
        self.output_feature_names_ = list(self.column_transformer.get_feature_names_out())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame into standardized + one-hot Euclidean feature matrix."""
        if self.column_transformer is None:
            raise RuntimeError("BaselineEuclideanPreprocessor belum di-fit.")
        X_dom = self.feature_engineer.transform(X)
        return self.column_transformer.transform(X_dom).astype(np.float64)


@dataclass(frozen=True)
class MixedTypeEncoded:
    """Container holding separated numeric and integer-encoded categorical arrays."""

    numeric: np.ndarray
    categorical: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.numeric.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.numeric.shape[1] + self.categorical.shape[1])


class MixedTypePreprocessor(BaseEstimator, TransformerMixin):
    """Leak-free preprocessor and feature-distance calculator for DIFW-KNN (E2-E5).

    Learns strictly from training data:
      - M_pos_train for pdays_recency
      - Q05_j, Q95_j, and R_j = Q95_j - Q05_j for each of the 6 numeric features
      - Category-to-integer vocabulary for each of the 10 categorical/binary features

    Distance per feature:
      - Numeric: d_j(x, z) = min(1, |x_j - z_j| / R_j)
      - Categorical: d_j(x, z) = 1 if x_j != z_j else 0
    """

    def __init__(self) -> None:
        self.feature_engineer = DomainFeatureEngineer()
        self.q05_: np.ndarray | None = None
        self.q95_: np.ndarray | None = None
        self.ranges_: np.ndarray | None = None
        self.category_maps_: dict[str, dict[str, int]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> MixedTypePreprocessor:
        """Fit pdays median, Q05-Q95 numeric ranges, and category vocabularies."""
        X_dom = self.feature_engineer.fit_transform(X)

        num_arr = X_dom[DOMAIN_NUMERIC_COLS].to_numpy(dtype=np.float64)
        q05 = np.quantile(num_arr, 0.05, axis=0)
        q95 = np.quantile(num_arr, 0.95, axis=0)
        ranges = q95 - q05

        invalid_idx = np.where(ranges <= 0.0)[0]
        if len(invalid_idx) > 0:
            bad_cols = [
                f"{DOMAIN_NUMERIC_COLS[i]} (Q05={q05[i]:.6f}, Q95={q95[i]:.6f})"
                for i in invalid_idx
            ]
            raise ValueError(
                "Rentang kuantil Q95 - Q05 <= 0 ditemukan pada fitur numerik: "
                + ", ".join(bad_cols)
            )

        self.q05_ = q05
        self.q95_ = q95
        self.ranges_ = ranges
        self.category_maps_ = {}
        for col in DOMAIN_CATEGORICAL_COLS:
            cats = sorted(X_dom[col].astype(str).unique().tolist())
            self.category_maps_[col] = {cat: idx for idx, cat in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> MixedTypeEncoded:
        """Transform DataFrame into MixedTypeEncoded container."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypePreprocessor belum di-fit.")
        X_dom = self.feature_engineer.transform(X)
        num_arr = X_dom[DOMAIN_NUMERIC_COLS].to_numpy(dtype=np.float64)

        cat_arr = np.empty((len(X_dom), len(DOMAIN_CATEGORICAL_COLS)), dtype=np.int64)
        for j, col in enumerate(DOMAIN_CATEGORICAL_COLS):
            mapping = self.category_maps_[col]
            cat_arr[:, j] = [mapping.get(val, -1) for val in X_dom[col].astype(str)]

        return MixedTypeEncoded(
            numeric=num_arr,
            categorical=cat_arr,
            feature_names=tuple(DOMAIN_FEATURE_COLS),
        )

    def get_numeric_params_table(self) -> pd.DataFrame:
        """Return DataFrame of fitted Q05, Q95, and range parameters per numeric feature."""
        if self.q05_ is None or self.q95_ is None or self.ranges_ is None:
            raise RuntimeError("MixedTypePreprocessor belum di-fit.")
        return pd.DataFrame(
            {
                "feature": DOMAIN_NUMERIC_COLS,
                "q05_train": np.round(self.q05_, 6),
                "q95_train": np.round(self.q95_, 6),
                "range_q95_q05": np.round(self.ranges_, 6),
            }
        )

    def compute_pairwise_feature_distances(
        self,
        a: MixedTypeEncoded,
        b: MixedTypeEncoded,
    ) -> np.ndarray:
        """Compute per-feature distances d_j(x, z) in [0, 1] of shape (n_a, n_b, 16)."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypePreprocessor belum di-fit.")
        d_num = np.minimum(
            1.0,
            np.abs(a.numeric[:, None, :] - b.numeric[None, :, :]) / self.ranges_[None, None, :],
        )
        # Unseen category (-1) always yields mismatch distance 1 against any observation
        a_cat = a.categorical[:, None, :]
        b_cat = b.categorical[None, :, :]
        d_cat = ((a_cat != b_cat) | (a_cat < 0) | (b_cat < 0)).astype(np.float64)
        return np.concatenate([d_num, d_cat], axis=2)

    def compute_aligned_feature_distances(
        self,
        a: MixedTypeEncoded,
        b: MixedTypeEncoded,
    ) -> np.ndarray:
        """Compute element-wise per-feature distances for aligned pairs (shape: n, 16)."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypePreprocessor belum di-fit.")
        if a.n_samples != b.n_samples:
            raise ValueError("Aligned distance membutuhkan jumlah baris yang sama.")
        d_num = np.minimum(1.0, np.abs(a.numeric - b.numeric) / self.ranges_[None, :])
        d_cat = ((a.categorical != b.categorical) | (a.categorical < 0) | (b.categorical < 0)).astype(
            np.float64
        )
        return np.concatenate([d_num, d_cat], axis=1)

    def compute_weighted_distance_matrix(
        self,
        queries: MixedTypeEncoded,
        references: MixedTypeEncoded,
        weights: np.ndarray | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Compute weighted mixed-type distance matrix D_w(q, r) in memory-bounded batches."""
        if self.ranges_ is None:
            raise RuntimeError("MixedTypePreprocessor belum di-fit.")
        n_features = len(DOMAIN_FEATURE_COLS)
        if weights is None:
            w = np.full(n_features, 1.0 / n_features, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (n_features,):
                raise ValueError(f"Panjang weights harus {n_features}, diterima {w.shape}.")

        w_num = w[: len(DOMAIN_NUMERIC_COLS)]
        w_cat = w[len(DOMAIN_NUMERIC_COLS) :]
        n_q = queries.n_samples
        n_r = references.n_samples
        dist_mat = np.zeros((n_q, n_r), dtype=np.float64)

        for start in range(0, n_q, batch_size):
            end = min(start + batch_size, n_q)
            q_num = queries.numeric[start:end]
            q_cat = queries.categorical[start:end]
            acc = np.zeros((end - start, n_r), dtype=np.float64)

            for j in range(len(DOMAIN_NUMERIC_COLS)):
                if w_num[j] == 0.0:
                    continue
                diff = np.abs(q_num[:, j : j + 1] - references.numeric[:, j][None, :])
                acc += w_num[j] * np.minimum(1.0, diff / self.ranges_[j])

            for j in range(len(DOMAIN_CATEGORICAL_COLS)):
                if w_cat[j] == 0.0:
                    continue
                q_col = q_cat[:, j : j + 1]
                r_col = references.categorical[:, j][None, :]
                mismatch = (q_col != r_col) | (q_col < 0) | (r_col < 0)
                acc += w_cat[j] * mismatch

            dist_mat[start:end] = acc
        return dist_mat
