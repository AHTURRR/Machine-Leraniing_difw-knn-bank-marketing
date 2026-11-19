"""Developer 2: Simplex-Constrained Feature Weight Optimizer for DIFW-KNN.

Optimizes feature importance weights w in the probability simplex:
    {w in R^p : w_j >= min_weight, sum_j w_j = 1}
by minimizing the regularized Triplet Margin Loss via Projected Gradient Descent
with Armijo/monotonic backtracking line search.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.mixed_type_distance import MixedTypeDistanceMetric, MixedTypeEncoded
from src.triplet_learning import (
    BalancedTripletSampler,
    TripletBatch,
    triplet_margin_loss_and_grad,
)


def project_onto_simplex(v: np.ndarray | Sequence[float], min_weight: float = 0.0) -> np.ndarray:
    """Exact Euclidean projection onto {w : w_j >= min_weight, sum_j w_j = 1}.

    Uses Duchi et al. (2008) simplex projection on shifted vector (v - min_weight)
    with target sum (1 - p * min_weight).
    """
    vec = np.asarray(v, dtype=np.float64)
    if vec.ndim != 1 or len(vec) == 0:
        raise ValueError("Input vektor untuk proyeksi simplex harus 1D dan tidak kosong.")
    p = len(vec)
    if min_weight < 0.0 or min_weight * p > 1.0 + 1e-12:
        raise ValueError(
            f"min_weight={min_weight} tidak valid untuk p={p} fitur (harus 0 <= min_weight <= 1/p)."
        )
    if abs(min_weight * p - 1.0) <= 1e-12:
        return np.full(p, 1.0 / p, dtype=np.float64)

    target_sum = 1.0 - p * float(min_weight)
    shifted = vec - float(min_weight)
    sorted_s = np.sort(shifted)[::-1]
    cssv = np.cumsum(sorted_s) - target_sum
    ind = np.arange(1, p + 1)
    cond = sorted_s - cssv / ind > 0
    if not np.any(cond):
        return np.full(p, 1.0 / p, dtype=np.float64)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(shifted - theta, 0.0) + float(min_weight)
    # Normalize slight floating-point drift while preserving lower bound
    excess = w - float(min_weight)
    if excess.sum() > 0:
        w = float(min_weight) + excess * (target_sum / excess.sum())
    return w


class FeatureWeightOptimizer:
    """Projected Gradient Descent optimizer for supervised feature weights w."""

    def __init__(
        self,
        margin: float = 0.2,
        reg_lambda: float = 0.05,
        learning_rate: float = 0.5,
        max_iter: int = 200,
        tol: float = 1e-7,
        min_weight: float = 0.0,
        n_triplets: int = 6000,
        positive_anchor_ratio: float = 0.5,
        mining_strategy: str = "semi_hard",
        candidate_pool_size: int = 16,
        random_state: int = 42,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate harus > 0, diterima {learning_rate}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter harus > 0, diterima {max_iter}.")
        self.margin = float(margin)
        self.reg_lambda = float(reg_lambda)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_weight = float(min_weight)
        self.n_triplets = int(n_triplets)
        self.positive_anchor_ratio = float(positive_anchor_ratio)
        self.mining_strategy = mining_strategy
        self.candidate_pool_size = int(candidate_pool_size)
        self.random_state = int(random_state)

        self.weights_: np.ndarray | None = None
        self.uniform_weights_: np.ndarray | None = None
        self.feature_names_: tuple[str, ...] = ()
        self.loss_history_: list[float] = []
        self.hinge_loss_history_: list[float] = []
        self.reg_loss_history_: list[float] = []
        self.active_fraction_history_: list[float] = []
        self.initial_loss_: float | None = None
        self.final_loss_: float | None = None
        self.n_iter_: int = 0
        self.triplet_batch_: TripletBatch | None = None

    def fit(
        self,
        encoded_X: MixedTypeEncoded,
        y: np.ndarray,
        metric: MixedTypeDistanceMetric,
        triplet_batch: TripletBatch | None = None,
    ) -> FeatureWeightOptimizer:
        """Sample triplets (if not provided) and optimize feature weights on the simplex."""
        if triplet_batch is None:
            sampler = BalancedTripletSampler(
                n_triplets=self.n_triplets,
                positive_anchor_ratio=self.positive_anchor_ratio,
                mining_strategy=self.mining_strategy,
                candidate_pool_size=self.candidate_pool_size,
                margin=self.margin,
                random_state=self.random_state,
            )
            batch = sampler.sample(encoded_X, y, metric)
        else:
            batch = triplet_batch

        self.triplet_batch_ = batch
        self.feature_names_ = encoded_X.feature_names
        n_features = encoded_X.n_features
        delta_d = batch.delta_feature_distances

        w = np.full(n_features, 1.0 / n_features, dtype=np.float64)
        self.uniform_weights_ = w.copy()

        loss, grad, stats = triplet_margin_loss_and_grad(
            w, delta_d, margin=self.margin, reg_lambda=self.reg_lambda
        )
        self.initial_loss_ = loss
        self.loss_history_ = [loss]
        self.hinge_loss_history_ = [stats["hinge_loss"]]
        self.reg_loss_history_ = [stats["reg_loss"]]
        self.active_fraction_history_ = [stats["active_fraction"]]

        n_iter = 0
        for it in range(1, self.max_iter + 1):
            step = self.learning_rate
            accepted = False
            best_candidate: tuple[np.ndarray, float, np.ndarray, dict[str, float]] | None = None

            # Backtracking line search along projected gradient arc
            for _ in range(25):
                w_cand = project_onto_simplex(w - step * grad, min_weight=self.min_weight)
                cand_loss, cand_grad, cand_stats = triplet_margin_loss_and_grad(
                    w_cand, delta_d, margin=self.margin, reg_lambda=self.reg_lambda
                )
                if cand_loss <= loss - 1e-12:
                    best_candidate = (w_cand, cand_loss, cand_grad, cand_stats)
                    accepted = True
                    break
                step *= 0.5

            if not accepted or best_candidate is None:
                break

            w_next, next_loss, next_grad, next_stats = best_candidate
            loss_drop = loss - next_loss
            weight_shift = float(np.linalg.norm(w_next - w))

            w = w_next
            loss = next_loss
            grad = next_grad
            n_iter = it

            self.loss_history_.append(loss)
            self.hinge_loss_history_.append(next_stats["hinge_loss"])
            self.reg_loss_history_.append(next_stats["reg_loss"])
            self.active_fraction_history_.append(next_stats["active_fraction"])

            if loss_drop <= self.tol and weight_shift <= self.tol:
                break

        self.weights_ = w
        self.final_loss_ = loss
        self.n_iter_ = n_iter
        return self

    def get_weights_table(self, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
        """Return sorted DataFrame summarizing learned vs uniform feature weights."""
        if self.weights_ is None or self.uniform_weights_ is None:
            raise RuntimeError("FeatureWeightOptimizer belum di-fit.")
        names = list(feature_names) if feature_names is not None else list(self.feature_names_)
        if len(names) != len(self.weights_):
            raise ValueError("Jumlah nama fitur tidak sesuai dengan panjang bobot.")

        df = pd.DataFrame(
            {
                "feature": names,
                "weight": self.weights_,
                "uniform_weight": self.uniform_weights_,
                "relative_importance": self.weights_ / self.uniform_weights_,
                "weight_pct": self.weights_ * 100.0,
            }
        )
        return df.sort_values("weight", ascending=False).reset_index(drop=True)

    def evaluate_seed_stability(
        self,
        encoded_X: MixedTypeEncoded,
        y: np.ndarray,
        metric: MixedTypeDistanceMetric,
        seeds: Sequence[int] = (42, 100, 2026, 7, 99),
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Evaluate stability of learned feature weights across multiple random seeds."""
        if len(seeds) < 2:
            raise ValueError("Minimal 2 random seed diperlukan untuk uji stabilitas.")

        weight_matrix = np.empty((len(seeds), encoded_X.n_features), dtype=np.float64)
        losses: list[float] = []
        for idx, seed in enumerate(seeds):
            opt = FeatureWeightOptimizer(
                margin=self.margin,
                reg_lambda=self.reg_lambda,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol,
                min_weight=self.min_weight,
                n_triplets=self.n_triplets,
                positive_anchor_ratio=self.positive_anchor_ratio,
                mining_strategy=self.mining_strategy,
                candidate_pool_size=self.candidate_pool_size,
                random_state=int(seed),
            )
            opt.fit(encoded_X, y, metric)
            assert opt.weights_ is not None and opt.final_loss_ is not None
            weight_matrix[idx] = opt.weights_
            losses.append(opt.final_loss_)

        cos_sims: list[float] = []
        spearman_corrs: list[float] = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                wi, wj = weight_matrix[i], weight_matrix[j]
                denom = float(np.linalg.norm(wi) * np.linalg.norm(wj))
                cos_sims.append(float(np.dot(wi, wj) / denom) if denom > 0 else 0.0)
                rho = float(spearmanr(wi, wj).statistic)
                if np.isnan(rho):
                    rho = 1.0 if np.allclose(wi, wj) else 0.0
                spearman_corrs.append(rho)

        feature_names = list(encoded_X.feature_names)
        stability_df = pd.DataFrame(
            {
                "feature": feature_names,
                "weight_mean": weight_matrix.mean(axis=0),
                "weight_std": weight_matrix.std(axis=0),
                "weight_min": weight_matrix.min(axis=0),
                "weight_max": weight_matrix.max(axis=0),
            }
        ).sort_values("weight_mean", ascending=False).reset_index(drop=True)

        summary_stats = {
            "n_seeds": float(len(seeds)),
            "mean_pairwise_cosine": float(np.mean(cos_sims)),
            "min_pairwise_cosine": float(np.min(cos_sims)),
            "mean_pairwise_spearman": float(np.mean(spearman_corrs)),
            "min_pairwise_spearman": float(np.min(spearman_corrs)),
            "mean_final_loss": float(np.mean(losses)),
            "std_final_loss": float(np.std(losses)),
        }
        return stability_df, summary_stats
