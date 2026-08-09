"""Synthetic data and evaluation helpers for content/tag fusion experiments.

The experiment deliberately keeps latent truth separate from observed tags.  A
dataset therefore contains the true soft root memberships, clean tag vectors,
corrupted observed tag vectors, and per-row corruption counts.  The clustering
helpers use the repository's normalized PCA projection and spherical FCM
implementation so that synthetic results exercise the same geometry as the
production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from fcm_core import spherical_fcm
from pca_projection import (
    fit_normalized_pca_projection,
    transform_normalized_pca_projection,
)
from tag_embedding_features import build_tag_augmented_features


FusionVariant = Literal[
    "content_only",
    "additive",
    "concat",
    "same_pca_additive",
]

TAG_CORRUPTION_NAMES = (
    "missing_tag",
    "wrong_tag",
    "general_tag",
    "extra_tag",
)
DEFAULT_TAG_CORRUPTION_RATES = (0.15, 0.10, 0.20, 0.25)


@dataclass(frozen=True)
class SyntheticTagConfig:
    """Configuration for one deterministic synthetic content/tag dataset."""

    n_samples: int = 600
    n_roots: int = 10
    embedding_dim: int = 64
    factor_dim: int = 5
    seed: int = 42
    content_noise: float = 0.10
    note_specific_scale: float = 0.15
    tag_noise: float = 0.03
    tag_corruption: float = 1.0
    membership_alpha: float = 0.90
    boundary_threshold: float = 0.60
    min_active_roots: int = 2
    max_active_roots: int = 3
    common_root_strength: float = 0.20
    related_root_strength: float = 0.55
    tag_corruption_rates: tuple[float, ...] = DEFAULT_TAG_CORRUPTION_RATES

    def validate(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if self.n_roots < 1:
            raise ValueError("n_roots must be at least 1")
        if self.embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        if self.factor_dim < 1:
            raise ValueError("factor_dim must be at least 1")
        if self.content_noise < 0.0:
            raise ValueError("content_noise must be non-negative")
        if self.note_specific_scale < 0.0 or self.tag_noise < 0.0:
            raise ValueError("noise scales must be non-negative")
        if self.tag_corruption < 0.0:
            raise ValueError("tag_corruption must be non-negative")
        if self.membership_alpha <= 0.0:
            raise ValueError("membership_alpha must be positive")
        if not 0.0 < self.boundary_threshold <= 1.0:
            raise ValueError("boundary_threshold must be in (0, 1]")
        if not 1 <= self.min_active_roots <= self.max_active_roots:
            raise ValueError(
                "active-root bounds must satisfy 1 <= min <= max"
            )
        if self.max_active_roots > self.n_roots:
            raise ValueError("max_active_roots cannot exceed n_roots")
        if self.common_root_strength < 0.0 or self.related_root_strength < 0.0:
            raise ValueError("root factor strengths must be non-negative")
        if (
            self.common_root_strength**2 + self.related_root_strength**2
            >= 1.0
        ):
            raise ValueError(
                "common_root_strength^2 + related_root_strength^2 must be < 1"
            )
        if len(self.tag_corruption_rates) != len(TAG_CORRUPTION_NAMES):
            raise ValueError(
                "tag_corruption_rates must contain one value per corruption type"
            )
        if any(
            rate < 0.0 or rate > 1.0 for rate in self.tag_corruption_rates
        ):
            raise ValueError("tag corruption rates must be between 0 and 1")


@dataclass(frozen=True)
class SyntheticTagDataset:
    """Latent truth and observed vectors for one synthetic experiment cell."""

    content_embeddings: np.ndarray
    clean_tag_embeddings: np.ndarray
    observed_tag_embeddings: np.ndarray
    true_memberships: np.ndarray
    root_embeddings: np.ndarray
    root_groups: np.ndarray
    metadata: pd.DataFrame
    corruption_flags: pd.DataFrame
    config: SyntheticTagConfig


@dataclass(frozen=True)
class FusionClusterResult:
    """Fixed-K clustering output and aligned quality metrics."""

    variant: str
    tag_source: str
    tag_weight: float
    projected: np.ndarray
    memberships: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    pca_components: int
    metrics: dict[str, Any]
    alignment: dict[int, int]


def _normalize_rows(values: np.ndarray, *, allow_zero: bool = False) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("values must be a 2D matrix with non-zero width")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("values must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not allow_zero and np.any(norms <= 1e-12):
        raise ValueError("values must not contain zero-length rows")
    return np.divide(
        matrix,
        np.maximum(norms, 1e-12),
        out=np.zeros_like(matrix),
        where=norms > 1e-12,
    )


def _probability_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    sums = matrix.sum(axis=1, keepdims=True)
    if np.any(sums <= 1e-12):
        raise ValueError(f"{name} must have positive row sums")
    return matrix / sums


def _make_correlated_roots(
    rng: np.random.Generator,
    config: SyntheticTagConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Create root vectors with shared and unrelated semantic factors."""

    factor_count = min(config.factor_dim, config.n_roots)
    common = _normalize_rows(rng.normal(size=(1, config.embedding_dim)))[0]
    factors = _normalize_rows(
        rng.normal(size=(factor_count, config.embedding_dim))
    )
    unique_strength = np.sqrt(
        1.0
        - config.common_root_strength**2
        - config.related_root_strength**2
    )
    roots: list[np.ndarray] = []
    groups: list[int] = []
    for root_id in range(config.n_roots):
        group = root_id % factor_count
        unique = _normalize_rows(
            rng.normal(size=(1, config.embedding_dim))
        )[0]
        raw = (
            config.common_root_strength * common
            + config.related_root_strength * factors[group]
            + unique_strength * unique
        )
        roots.append(raw)
        groups.append(group)
    return _normalize_rows(np.asarray(roots)), np.asarray(groups, dtype=int)


def _make_soft_memberships(
    rng: np.random.Generator,
    config: SyntheticTagConfig,
) -> np.ndarray:
    memberships = np.zeros((config.n_samples, config.n_roots), dtype=np.float64)
    for row_index in range(config.n_samples):
        active_count = int(
            rng.integers(config.min_active_roots, config.max_active_roots + 1)
        )
        dominant = int(rng.integers(0, config.n_roots))
        remaining = np.delete(np.arange(config.n_roots), dominant)
        if active_count == 1:
            active = np.asarray([dominant], dtype=int)
        else:
            active = np.concatenate(
                ([dominant], rng.choice(remaining, active_count - 1, replace=False))
            )
        weights = rng.dirichlet(
            np.full(active_count, config.membership_alpha, dtype=np.float64)
        )
        memberships[row_index, active] = weights
    return memberships


def _tag_corruption_probabilities(config: SyntheticTagConfig) -> dict[str, float]:
    return {
        name: min(1.0, config.tag_corruption * rate)
        for name, rate in zip(
            TAG_CORRUPTION_NAMES,
            config.tag_corruption_rates,
            strict=True,
        )
    }


def _make_observed_tags(
    rng: np.random.Generator,
    config: SyntheticTagConfig,
    roots: np.ndarray,
    memberships: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply independent missing, wrong, general, and extra-tag corruption."""

    probabilities = _tag_corruption_probabilities(config)
    general = _normalize_rows(rng.normal(size=(1, config.embedding_dim)))[0]
    observed = np.zeros_like(memberships @ roots)
    rows: list[dict[str, Any]] = []
    for row_index, membership in enumerate(memberships):
        active = np.flatnonzero(membership > 0.0)
        vectors: list[np.ndarray] = []
        missing_count = 0
        wrong_count = 0
        general_count = 0
        extra_count = 0
        for root_id in active:
            if rng.random() < probabilities["missing_tag"]:
                missing_count += 1
                continue
            selected_root = int(root_id)
            if rng.random() < probabilities["wrong_tag"]:
                candidates = np.delete(np.arange(config.n_roots), active)
                if candidates.size:
                    selected_root = int(rng.choice(candidates))
                wrong_count += 1
            vector = roots[selected_root]
            if rng.random() < probabilities["general_tag"]:
                vector = general
                general_count += 1
            vector = vector + config.tag_noise * rng.normal(
                size=config.embedding_dim
            )
            vectors.append(float(membership[root_id]) * vector)

        if rng.random() < probabilities["extra_tag"]:
            extra_count = int(rng.integers(1, 3))
            for _ in range(extra_count):
                if rng.random() < 0.5:
                    vector = general
                else:
                    vector = roots[int(rng.integers(0, config.n_roots))]
                vectors.append(
                    0.5
                    * (vector + config.tag_noise * rng.normal(size=config.embedding_dim))
                )

        if vectors:
            observed[row_index] = _normalize_rows(
                np.sum(np.asarray(vectors), axis=0, keepdims=True)
            )[0]
        rows.append(
            {
                "id": f"synthetic-{row_index}",
                "missing_tag_count": missing_count,
                "wrong_tag_count": wrong_count,
                "general_tag_count": general_count,
                "extra_tag_count": extra_count,
                "observed_tag_count": len(vectors),
                "is_corrupted": bool(
                    missing_count + wrong_count + general_count + extra_count
                ),
            }
        )
    return observed, pd.DataFrame(rows)


def generate_synthetic_tag_dataset(
    config: SyntheticTagConfig | None = None,
) -> SyntheticTagDataset:
    """Generate deterministic latent content and corrupted tag observations."""

    config = config or SyntheticTagConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)
    roots, root_groups = _make_correlated_roots(rng, config)
    memberships = _make_soft_memberships(rng, config)
    semantic = memberships @ roots
    content = semantic + config.note_specific_scale * rng.normal(
        size=(config.n_samples, config.embedding_dim)
    )
    content += config.content_noise * rng.normal(
        size=(config.n_samples, config.embedding_dim)
    )
    content = _normalize_rows(content)

    clean_tags = memberships @ roots
    clean_tags += config.tag_noise * rng.normal(
        size=(config.n_samples, config.embedding_dim)
    )
    clean_tags = _normalize_rows(clean_tags)
    observed_tags, corruption_flags = _make_observed_tags(
        rng,
        config,
        roots,
        memberships,
    )
    dominant_root = memberships.argmax(axis=1).astype(int)
    metadata = pd.DataFrame(
        {
            "id": [f"synthetic-{index}" for index in range(config.n_samples)],
            "dominant_root": dominant_root,
            "active_root_count": np.count_nonzero(memberships > 0.0, axis=1),
            "is_boundary": memberships.max(axis=1) < config.boundary_threshold,
            "content_noise": float(config.content_noise),
            "tag_corruption": float(config.tag_corruption),
        }
    )
    return SyntheticTagDataset(
        content_embeddings=content,
        clean_tag_embeddings=clean_tags,
        observed_tag_embeddings=observed_tags,
        true_memberships=memberships,
        root_embeddings=roots,
        root_groups=root_groups,
        metadata=metadata,
        corruption_flags=corruption_flags,
        config=config,
    )


def shuffle_tag_embeddings(
    tag_embeddings: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Permute tag rows while preserving their marginal vector distribution."""

    values = np.asarray(tag_embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("tag_embeddings must be a non-empty 2D matrix")
    permutation = np.random.default_rng(seed).permutation(values.shape[0])
    return values[permutation].copy()


def build_additive_fusion(
    content_embeddings: np.ndarray,
    tag_embeddings: np.ndarray,
    *,
    tag_weight: float = 1.0,
) -> np.ndarray:
    """Aggregate normalized tags into the content direction additively."""

    if not np.isfinite(tag_weight) or tag_weight < 0.0:
        raise ValueError("tag_weight must be finite and non-negative")
    content = np.asarray(content_embeddings, dtype=np.float64)
    tags = np.asarray(tag_embeddings, dtype=np.float64)
    if content.ndim != 2 or tags.ndim != 2 or content.shape != tags.shape:
        raise ValueError("content and tag embeddings must have the same 2D shape")
    fused = _normalize_rows(content) + tag_weight * _normalize_rows(
        tags,
        allow_zero=True,
    )
    return _normalize_rows(fused)


def build_fusion_features(
    content_embeddings: np.ndarray,
    tag_embeddings: np.ndarray,
    *,
    variant: FusionVariant,
    tag_weight: float = 1.0,
) -> np.ndarray:
    """Build one of the pre-PCA content/tag representations."""

    if variant == "content_only":
        return _normalize_rows(np.asarray(content_embeddings, dtype=np.float64))
    if variant == "additive":
        return build_additive_fusion(
            content_embeddings,
            tag_embeddings,
            tag_weight=tag_weight,
        )
    if variant == "concat":
        if tag_weight <= 0.0:
            raise ValueError("concat fusion requires a positive tag_weight")
        return build_tag_augmented_features(
            content_embeddings,
            tag_embeddings,
            tag_weight=tag_weight,
        )
    if variant == "same_pca_additive":
        raise ValueError(
            "same_pca_additive requires a fitted PCA; use cluster_fusion_dataset"
        )
    raise ValueError(f"Unknown fusion variant: {variant}")


def align_predicted_memberships(
    true_memberships: np.ndarray,
    predicted_memberships: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], float]:
    """Match predicted clusters to roots using maximum membership cosine."""

    truth = _probability_rows(true_memberships, name="true_memberships")
    predicted = _probability_rows(
        predicted_memberships,
        name="predicted_memberships",
    )
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError("true and predicted membership rows must align")
    truth_unit = _normalize_rows(truth)
    predicted_unit = _normalize_rows(predicted)
    similarity = truth_unit.T @ predicted_unit
    true_indices, predicted_indices = linear_sum_assignment(-similarity)
    aligned = np.zeros_like(truth)
    mapping: dict[int, int] = {}
    for true_index, predicted_index in zip(
        true_indices,
        predicted_indices,
        strict=True,
    ):
        aligned[:, true_index] = predicted[:, predicted_index]
        mapping[int(true_index)] = int(predicted_index)
    matched = set(int(index) for index in predicted_indices)
    unmatched_mass = float(
        predicted[:, [
            index for index in range(predicted.shape[1]) if index not in matched
        ]].sum(axis=1).mean()
    ) if len(matched) < predicted.shape[1] else 0.0
    aligned = _probability_rows(aligned, name="aligned_memberships")
    return aligned, mapping, unmatched_mass


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    result = np.zeros(left.shape[0], dtype=np.float64)
    for values in (left, right):
        positive = values > 0.0
        ratio = np.ones_like(values)
        np.divide(values, midpoint, out=ratio, where=positive)
        contribution = np.zeros_like(values)
        contribution[positive] = values[positive] * np.log(ratio[positive])
        result += contribution.sum(axis=1)
    return 0.5 * result


def _membership_subset_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    if not np.any(mask):
        return {
            "membership_cosine": None,
            "membership_js_divergence": None,
            "membership_mae": None,
            "membership_mse": None,
            "dominant_1_accuracy": None,
            "top_2_root_recall": None,
        }
    truth_subset = truth[mask]
    predicted_subset = predicted[mask]
    truth_unit = _normalize_rows(truth_subset)
    predicted_unit = _normalize_rows(predicted_subset)
    top_k = min(2, truth.shape[1], predicted.shape[1])
    top_2_recall = []
    for true_row, predicted_row in zip(
        truth_subset,
        predicted_subset,
        strict=True,
    ):
        truth_top = set(np.argsort(true_row)[-top_k:])
        predicted_top = set(np.argsort(predicted_row)[-top_k:])
        top_2_recall.append(len(truth_top & predicted_top) / top_k)
    return {
        "membership_cosine": float(
            np.mean(np.sum(truth_unit * predicted_unit, axis=1))
        ),
        "membership_js_divergence": float(
            np.mean(_js_divergence(truth_subset, predicted_subset))
        ),
        "membership_mae": float(np.mean(np.abs(truth_subset - predicted_subset))),
        "membership_mse": float(np.mean((truth_subset - predicted_subset) ** 2)),
        "dominant_1_accuracy": float(
            np.mean(truth_subset.argmax(axis=1) == predicted_subset.argmax(axis=1))
        ),
        "top_2_root_recall": float(np.mean(top_2_recall)),
    }


def evaluate_soft_memberships(
    true_memberships: np.ndarray,
    predicted_memberships: np.ndarray,
    *,
    boundary_threshold: float = 0.60,
) -> tuple[dict[str, Any], dict[int, int]]:
    """Return hard/soft metrics for all rows and true boundary rows."""

    truth = _probability_rows(true_memberships, name="true_memberships")
    aligned, mapping, unmatched_mass = align_predicted_memberships(
        truth,
        predicted_memberships,
    )
    boundary = truth.max(axis=1) < boundary_threshold
    metrics: dict[str, Any] = {
        "hard_ari": float(
            adjusted_rand_score(truth.argmax(axis=1), aligned.argmax(axis=1))
        ),
        "hard_nmi": float(
            normalized_mutual_info_score(
                truth.argmax(axis=1),
                aligned.argmax(axis=1),
            )
        ),
        "boundary_count": int(np.sum(boundary)),
        "boundary_rate": float(np.mean(boundary)),
        "unmatched_predicted_mass": unmatched_mass,
    }
    metrics.update(_membership_subset_metrics(truth, aligned, np.ones(len(truth), dtype=bool)))
    boundary_metrics = _membership_subset_metrics(truth, aligned, boundary)
    metrics.update(
        {
            f"boundary_{key}": value
            for key, value in boundary_metrics.items()
        }
    )
    return metrics, mapping


def cluster_fusion_dataset(
    dataset: SyntheticTagDataset,
    *,
    variant: FusionVariant = "content_only",
    tag_embeddings: np.ndarray | None = None,
    tag_source: str = "observed",
    tag_weight: float = 1.0,
    n_clusters: int | None = None,
    pca_components: int = 32,
    seed: int = 42,
    n_init: int = 3,
    max_iter: int = 200,
) -> FusionClusterResult:
    """Fit fixed-K SFCM through one content/tag fusion and score it."""

    if n_clusters is None:
        n_clusters = dataset.true_memberships.shape[1]
    if n_clusters < 1 or n_clusters > len(dataset.content_embeddings):
        raise ValueError("n_clusters must be within the sample count")
    tags = (
        dataset.observed_tag_embeddings
        if tag_embeddings is None
        else np.asarray(tag_embeddings, dtype=np.float64)
    )
    if tags.shape != dataset.content_embeddings.shape:
        raise ValueError("tag_embeddings must align with content_embeddings")
    if variant == "same_pca_additive":
        fitted = fit_normalized_pca_projection(
            dataset.content_embeddings,
            n_components=pca_components,
            seed=seed,
            name="content_embeddings",
        )
        content_projected = fitted.normalized_prefix()
        tag_projected = transform_normalized_pca_projection(
            tags,
            fitted.pca,
            dimension=content_projected.shape[1],
            name="tag_embeddings",
        )
        projected = build_additive_fusion(
            content_projected,
            tag_projected,
            tag_weight=tag_weight,
        )
    else:
        features = build_fusion_features(
            dataset.content_embeddings,
            tags,
            variant=variant,
            tag_weight=tag_weight,
        )
        fitted = fit_normalized_pca_projection(
            features,
            n_components=pca_components,
            seed=seed,
            name=f"{variant}_features",
        )
        projected = fitted.normalized_prefix()

    fcm_result = spherical_fcm(
        projected,
        n_clusters=n_clusters,
        seed=seed,
        n_init=n_init,
        max_attempts=max(n_init * 2, n_init),
        max_iter=max_iter,
        min_cluster_size=1,
        min_center_separation=0.0,
    )
    metrics, alignment = evaluate_soft_memberships(
        dataset.true_memberships,
        fcm_result.memberships,
        boundary_threshold=dataset.config.boundary_threshold,
    )
    return FusionClusterResult(
        variant=variant,
        tag_source=tag_source,
        tag_weight=float(tag_weight),
        projected=projected,
        memberships=fcm_result.memberships,
        labels=fcm_result.labels,
        centers=fcm_result.centers,
        pca_components=int(projected.shape[1]),
        metrics=metrics,
        alignment=alignment,
    )


__all__ = [
    "DEFAULT_TAG_CORRUPTION_RATES",
    "FusionClusterResult",
    "FusionVariant",
    "SyntheticTagConfig",
    "SyntheticTagDataset",
    "TAG_CORRUPTION_NAMES",
    "align_predicted_memberships",
    "build_additive_fusion",
    "build_fusion_features",
    "cluster_fusion_dataset",
    "evaluate_soft_memberships",
    "generate_synthetic_tag_dataset",
    "shuffle_tag_embeddings",
]
