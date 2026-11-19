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


DEFAULT_RAW_DATA_PATH = Path(__file__).resolve(
).parent.parent / "data" / "bank-full.csv"


def load_raw_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load bank-full.csv using semicolon delimiter and validate core columns."""
    csv_path = DEFAULT_RAW_DATA_PATH if path is None else Path(path)
    if not csv_path.exists() and csv_path.name == "bank-full.csv":
        fallback_path = csv_path.parent / "data" / "bank-full.csv"
        if fallback_path.exists():
            csv_path = fallback_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    required = set(RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS +
                   LEAKAGE_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada dataset: {sorted(missing)}")
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
        raise ValueError(
            f"Nilai target tidak valid ditemukan: {sorted(invalid_targets)}")
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
            raise ValueError(
                "Kolom 'pdays' wajib ada sebelum DomainFeatureEngineer.")
        pos_pdays = X.loc[X["pdays"] > -1, "pdays"].astype(float)
        if pos_pdays.empty:
            raise ValueError(
                "Tidak ada observasi dengan pdays > -1 pada data training.")
        median_val = float(pos_pdays.median())
        if median_val <= 0:
            raise ValueError(
                f"Median pdays > -1 harus > 0, diperoleh {median_val}")
        self.pdays_median_pos_ = median_val
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform pdays and return DataFrame with ordered 16 domain features."""
        if self.pdays_median_pos_ is None:
            raise RuntimeError(
                "DomainFeatureEngineer belum di-fit pada data training.")
        out = X.copy()
        pdays_vals = out["pdays"].astype(float)
        contacted_mask = pdays_vals > -1

        out["previously_contacted"] = np.where(contacted_mask, "1", "0")
        recency = np.zeros(len(out), dtype=np.float64)
        recency[contacted_mask.to_numpy()] = 1.0 / (
            1.0 + (pdays_vals[contacted_mask].to_numpy() /
                   self.pdays_median_pos_)
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
                    OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False),
                    DOMAIN_CATEGORICAL_COLS,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.column_transformer.fit(X_dom)
        self.output_feature_names_ = list(
            self.column_transformer.get_feature_names_out())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame into standardized + one-hot Euclidean feature matrix."""
        if self.column_transformer is None:
            raise RuntimeError("BaselineEuclideanPreprocessor belum di-fit.")
        X_dom = self.feature_engineer.transform(X)
        return self.column_transformer.transform(X_dom).astype(np.float64)


# Re-export Developer 1 mixed-type distance classes for backward compatibility
from src.mixed_type_distance import (  # noqa: E402, F401
    MixedTypeDistanceMetric,
    MixedTypeEncoded,
    MixedTypePreprocessor,
)
