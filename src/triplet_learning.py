"""Developer 2: Class-Balanced Triplet Sampling & Triplet Margin Loss for DIFW-KNN.

Implements:
  - BalancedTripletSampler: constructs class-balanced triplets (a, p, n) with
    y_a = y_p and y_a != y_n, supporting random and semi-hard mining.
  - triplet_margin_loss_and_grad: computes the regularized Triplet Margin Loss
        L(w) = mean(max(0, m + delta_d^T w)) + lambda * ||w - u||_2^2
    and its exact analytical gradient w.r.t. w.
  - numerical_gradient_check: central-difference verification of the analytical gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.mixed_type_distance import (
    MixedTypeDistanceMetric,
    MixedTypeEncoded,
    normalize_feature_weights,
)


@dataclass(frozen=True)
class TripletBatch:
    """Container storing sampled triplet indices, anchor labels, and delta_d vectors."""

    anchor_idx: np.ndarray
    positive_idx: np.ndarray
    negative_idx: np.ndarray
    anchor_labels: np.ndarray
    delta_feature_distances: np.ndarray

    @property
    def n_triplets(self) -> int:
        return int(len(self.anchor_idx))

    @property
    def n_features(self) -> int:
        return int(self.delta_feature_distances.shape[1])


class BalancedTripletSampler:
    """Construct class-balanced triplets for supervised feature weight learning."""

    def __init__(
        self,
        n_triplets: int = 6000,
        positive_anchor_ratio: float = 0.5,
        mining_strategy: str = "semi_hard",
        candidate_pool_size: int = 16,
        margin: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if n_triplets <= 0:
            raise ValueError(f"n_triplets harus > 0, diterima {n_triplets}.")
        if not (0.0 < positive_anchor_ratio < 1.0):
            raise ValueError(
                f"positive_anchor_ratio harus dalam (0, 1), diterima {positive_anchor_ratio}."
            )
        if mining_strategy not in {"random", "semi_hard"}:
            raise ValueError(
                f"mining_strategy harus 'random' atau 'semi_hard', diterima {mining_strategy!r}."
            )
        if candidate_pool_size <= 0:
            raise ValueError(f"candidate_pool_size harus > 0, diterima {candidate_pool_size}.")
        self.n_triplets = int(n_triplets)
        self.positive_anchor_ratio = float(positive_anchor_ratio)
        self.mining_strategy = mining_strategy
        self.candidate_pool_size = int(candidate_pool_size)
        self.margin = float(margin)
        self.random_state = int(random_state)

    @staticmethod
    def _slice_encoded(encoded: MixedTypeEncoded, idx: np.ndarray) -> MixedTypeEncoded:
        return MixedTypeEncoded(
            numeric=encoded.numeric[idx],
            categorical=encoded.categorical[idx],
            feature_names=encoded.feature_names,
        )

    def _sample_anchor_indices(
        self,
        pos_pool: np.ndarray,
        neg_pool: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_pos_anchors = int(round(self.n_triplets * self.positive_anchor_ratio))
        n_neg_anchors = self.n_triplets - n_pos_anchors

        pos_anchors = rng.choice(
            pos_pool,
            size=n_pos_anchors,
            replace=(n_pos_anchors > len(pos_pool)),
        )
        neg_anchors = rng.choice(
            neg_pool,
            size=n_neg_anchors,
            replace=(n_neg_anchors > len(neg_pool)),
        )

        anchors = np.concatenate([pos_anchors, neg_anchors])
        labels = np.concatenate(
            [
                np.ones(n_pos_anchors, dtype=np.int64),
                np.zeros(n_neg_anchors, dtype=np.int64),
            ]
        )
        perm = rng.permutation(self.n_triplets)
        return anchors[perm], labels[perm]

    @staticmethod
    def _sample_excluding_self(
        pool: np.ndarray,
        exclude: np.ndarray,
        size_per_row: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample candidate indices from pool ensuring no element equals exclude[i]."""
        n_rows = len(exclude)
        n_pool = len(pool)
        # Draw positions in [0, n_pool - 1]; shift if colliding with exclude[i]
        cand = rng.choice(pool, size=(n_rows, size_per_row), replace=True)
        collision = cand == exclude[:, None]
        if collision.any():
            # Build lookup from pool value to position for fast collision resolution
            pos_map = {int(val): idx for idx, val in enumerate(pool)}
            row_idx, col_idx = np.where(collision)
            shifts = rng.integers(1, n_pool, size=len(row_idx))
            for r, c, sh in zip(row_idx, col_idx, shifts):
                ex_pos = pos_map[int(exclude[r])]
                cand[r, c] = pool[(ex_pos + int(sh)) % n_pool]
        return cand

    def sample(
        self,
        encoded_X: MixedTypeEncoded,
        y: np.ndarray,
        metric: MixedTypeDistanceMetric,
        mining_weights: Sequence[float] | np.ndarray | None = None,
        batch_size: int = 512,
    ) -> TripletBatch:
        """Sample class-balanced triplets and precompute delta_d = d(a, p) - d(a, n)."""
        y_arr = np.asarray(y, dtype=np.int64)
        if y_arr.shape != (encoded_X.n_samples,):
            raise ValueError(
                f"Panjang y ({len(y_arr)}) tidak cocok dengan jumlah sampel ({encoded_X.n_samples})."
            )
        unique_classes = set(np.unique(y_arr).tolist())
        if unique_classes != {0, 1}:
            raise ValueError(f"Target y harus biner {{0, 1}}, ditemukan {sorted(unique_classes)}.")

        pos_pool = np.where(y_arr == 1)[0]
        neg_pool = np.where(y_arr == 0)[0]
        if len(pos_pool) < 2 or len(neg_pool) < 2:
            raise ValueError("Masing-masing kelas minimal membutuhkan 2 sampel untuk sampling triplet.")

        rng = np.random.default_rng(self.random_state)
        anchor_idx, anchor_labels = self._sample_anchor_indices(pos_pool, neg_pool, rng)

        n_features = encoded_X.n_features
        w_mine = normalize_feature_weights(mining_weights, n_features=n_features)

        positive_idx = np.empty(self.n_triplets, dtype=np.int64)
        negative_idx = np.empty(self.n_triplets, dtype=np.int64)

        is_pos_anchor = anchor_labels == 1
        n_pos = int(is_pos_anchor.sum())
        n_neg = self.n_triplets - n_pos

        if self.mining_strategy == "random":
            positive_idx[is_pos_anchor] = self._sample_excluding_self(
                pos_pool, anchor_idx[is_pos_anchor], size_per_row=1, rng=rng
            )[:, 0]
            positive_idx[~is_pos_anchor] = self._sample_excluding_self(
                neg_pool, anchor_idx[~is_pos_anchor], size_per_row=1, rng=rng
            )[:, 0]
            negative_idx[is_pos_anchor] = rng.choice(neg_pool, size=n_pos, replace=True)
            negative_idx[~is_pos_anchor] = rng.choice(pos_pool, size=n_neg, replace=True)

            delta_d = np.empty((self.n_triplets, n_features), dtype=np.float64)
            for start in range(0, self.n_triplets, batch_size):
                end = min(start + batch_size, self.n_triplets)
                enc_a = self._slice_encoded(encoded_X, anchor_idx[start:end])
                enc_p = self._slice_encoded(encoded_X, positive_idx[start:end])
                enc_n = self._slice_encoded(encoded_X, negative_idx[start:end])
                d_ap = metric.compute_aligned_feature_distances(enc_a, enc_p)
                d_an = metric.compute_aligned_feature_distances(enc_a, enc_n)
                delta_d[start:end] = d_ap - d_an

            return TripletBatch(
                anchor_idx=anchor_idx,
                positive_idx=positive_idx,
                negative_idx=negative_idx,
                anchor_labels=anchor_labels,
                delta_feature_distances=delta_d,
            )

        # Semi-hard mining over random candidate pools
        pool_k = self.candidate_pool_size
        pos_cands = np.empty((self.n_triplets, pool_k), dtype=np.int64)
        neg_cands = np.empty((self.n_triplets, pool_k), dtype=np.int64)

        pos_cands[is_pos_anchor] = self._sample_excluding_self(
            pos_pool, anchor_idx[is_pos_anchor], size_per_row=pool_k, rng=rng
        )
        pos_cands[~is_pos_anchor] = self._sample_excluding_self(
            neg_pool, anchor_idx[~is_pos_anchor], size_per_row=pool_k, rng=rng
        )
        neg_cands[is_pos_anchor] = rng.choice(neg_pool, size=(n_pos, pool_k), replace=True)
        neg_cands[~is_pos_anchor] = rng.choice(pos_pool, size=(n_neg, pool_k), replace=True)

        delta_d = np.empty((self.n_triplets, n_features), dtype=np.float64)
        for start in range(0, self.n_triplets, batch_size):
            end = min(start + batch_size, self.n_triplets)
            bs = end - start

            a_rep = np.repeat(anchor_idx[start:end], pool_k)
            p_flat = pos_cands[start:end].ravel()
            n_flat = neg_cands[start:end].ravel()

            enc_a = self._slice_encoded(encoded_X, a_rep)
            enc_p = self._slice_encoded(encoded_X, p_flat)
            enc_n = self._slice_encoded(encoded_X, n_flat)

            d_ap_feat = metric.compute_aligned_feature_distances(enc_a, enc_p).reshape(
                bs, pool_k, n_features
            )
            d_an_feat = metric.compute_aligned_feature_distances(enc_a, enc_n).reshape(
                bs, pool_k, n_features
            )

            d_ap = d_ap_feat @ w_mine
            d_an = d_an_feat @ w_mine

            # Choose a random or close-to-median positive from candidate pool to preserve local clusters
            p_choice = np.argsort(d_ap, axis=1)[:, min(pool_k // 4, pool_k - 1)]
            row_ar = np.arange(bs)
            chosen_ap_dist = d_ap[row_ar, p_choice][:, None]

            # Semi-hard negative: D(a, p) < D(a, n) < D(a, p) + margin (closest to D(a, p))
            semi_hard_mask = (d_an > chosen_ap_dist) & (d_an < chosen_ap_dist + self.margin)
            beyond_mask = d_an > chosen_ap_dist

            n_choice = np.empty(bs, dtype=np.int64)
            for i in range(bs):
                if semi_hard_mask[i].any():
                    masked = np.where(semi_hard_mask[i], d_an[i], np.inf)
                    n_choice[i] = int(np.argmin(masked))
                elif beyond_mask[i].any():
                    masked = np.where(beyond_mask[i], d_an[i], np.inf)
                    n_choice[i] = int(np.argmin(masked))
                else:
                    # All negatives closer than positive: pick negative closest to positive boundary
                    n_choice[i] = int(np.argmax(d_an[i]))

            positive_idx[start:end] = pos_cands[start:end][row_ar, p_choice]
            negative_idx[start:end] = neg_cands[start:end][row_ar, n_choice]
            delta_d[start:end] = d_ap_feat[row_ar, p_choice] - d_an_feat[row_ar, n_choice]

        return TripletBatch(
            anchor_idx=anchor_idx,
            positive_idx=positive_idx,
            negative_idx=negative_idx,
            anchor_labels=anchor_labels,
            delta_feature_distances=delta_d,
        )


def triplet_margin_loss_and_grad(
    weights: np.ndarray | Sequence[float],
    delta_d: np.ndarray,
    margin: float = 0.2,
    reg_lambda: float = 0.05,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Compute regularized Triplet Margin Loss and its exact analytical gradient.

    L(w) = mean(max(0, m + delta_d @ w)) + reg_lambda * ||w - u||_2^2
    where u = (1/p, ..., 1/p).
    """
    w = np.asarray(weights, dtype=np.float64)
    dd = np.asarray(delta_d, dtype=np.float64)
    if dd.ndim != 2 or dd.shape[0] == 0:
        raise ValueError(f"delta_d harus berdimensi (n_triplets, n_features), diterima {dd.shape}.")
    n_triplets, n_features = dd.shape
    if w.shape != (n_features,):
        raise ValueError(f"Panjang weights harus ({n_features},), diterima {w.shape}.")
    if margin < 0.0:
        raise ValueError(f"margin harus >= 0, diterima {margin}.")
    if reg_lambda < 0.0:
        raise ValueError(f"reg_lambda harus >= 0, diterima {reg_lambda}.")

    u = np.full(n_features, 1.0 / n_features, dtype=np.float64)
    raw_hinge = margin + (dd @ w)
    active_mask = raw_hinge > 0.0

    hinge_loss = float(np.where(active_mask, raw_hinge, 0.0).mean())
    diff_u = w - u
    reg_loss = float(reg_lambda * np.dot(diff_u, diff_u))
    total_loss = hinge_loss + reg_loss

    if active_mask.any():
        grad_hinge = dd[active_mask].sum(axis=0) / float(n_triplets)
    else:
        grad_hinge = np.zeros(n_features, dtype=np.float64)
    grad = grad_hinge + 2.0 * reg_lambda * diff_u

    stats = {
        "total_loss": total_loss,
        "hinge_loss": hinge_loss,
        "reg_loss": reg_loss,
        "active_fraction": float(active_mask.mean()),
        "n_active": int(active_mask.sum()),
    }
    return total_loss, grad, stats


def numerical_gradient_check(
    weights: np.ndarray | Sequence[float],
    delta_d: np.ndarray,
    margin: float = 0.2,
    reg_lambda: float = 0.05,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Verify analytical gradient against central finite differences."""
    w = np.asarray(weights, dtype=np.float64).copy()
    _, analytical_grad, _ = triplet_margin_loss_and_grad(
        w, delta_d, margin=margin, reg_lambda=reg_lambda
    )

    numerical_grad = np.zeros_like(w)
    for j in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[j] += eps
        w_minus[j] -= eps
        loss_plus, _, _ = triplet_margin_loss_and_grad(
            w_plus, delta_d, margin=margin, reg_lambda=reg_lambda
        )
        loss_minus, _, _ = triplet_margin_loss_and_grad(
            w_minus, delta_d, margin=margin, reg_lambda=reg_lambda
        )
        numerical_grad[j] = (loss_plus - loss_minus) / (2.0 * eps)

    abs_diff = np.abs(analytical_grad - numerical_grad)
    max_abs_err = float(np.max(abs_diff))
    denom = np.maximum(1.0, np.abs(analytical_grad) + np.abs(numerical_grad))
    max_rel_err = float(np.max(abs_diff / denom))

    return {
        "analytical_grad": analytical_grad,
        "numerical_grad": numerical_grad,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "passed": bool(max_abs_err <= tol),
    }
