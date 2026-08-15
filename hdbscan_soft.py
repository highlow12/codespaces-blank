"""Distance-based soft assignments for an HDBSCAN clustering.

The scores in this module are a post-hoc diagnostic.  They are deliberately
not presented as HDBSCAN's own probabilities: HDBSCAN first supplies hard
non-noise members, then cosine distances in the PCA-normalized space supply
two comparable soft assignment rules for its original noise points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN

from fcm_core import fit_clustering_pca


DEFAULT_MIN_CLUSTER_SIZE = 15
DEFAULT_MIN_SAMPLES = 5
DEFAULT_REASSIGNMENT_THRESHOLD = 0.60
DEFAULT_NEIGHBOR_COUNT = 5
DEFAULT_MEDOID_CANDIDATE_BUDGET = 256
DEFAULT_MEDOID_EVALUATION_BUDGET = 1024


@dataclass(frozen=True)
class HdbscanSoftResult:
    """HDBSCAN labels and the two distance-based membership estimates."""

    features: np.ndarray
    original_labels: np.ndarray
    labels: np.ndarray
    medoid_memberships: np.ndarray
    neighbor_memberships: np.ndarray
    medoid_recommended_labels: np.ndarray
    neighbor_recommended_labels: np.ndarray
    medoid_confidences: np.ndarray
    neighbor_confidences: np.ndarray
    medoid_indices: np.ndarray
    cluster_radii: np.ndarray
    neighbor_member_counts: np.ndarray

    @property
    def cluster_count(self) -> int:
        return int(self.medoid_memberships.shape[1])


def _uniform_indices(size: int, budget: int) -> np.ndarray:
    if size <= budget:
        return np.arange(size, dtype=int)
    # linspace is deterministic and samples both ends of the ordered members.
    return np.unique(np.linspace(0, size - 1, num=budget, dtype=int))


def _softmax_negative_distances(distances: np.ndarray) -> np.ndarray:
    if distances.shape[1] == 0:
        return np.zeros_like(distances, dtype=np.float64)
    logits = -np.asarray(distances, dtype=np.float64)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _normalize_hdbscan_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=int)
    normalized = np.full(values.shape, -1, dtype=int)
    source_labels = np.unique(values[values >= 0])
    for target, source in enumerate(source_labels):
        normalized[values == source] = target
    return normalized


def _cluster_representatives(
    features: np.ndarray,
    labels: np.ndarray,
    cluster_count: int,
    *,
    candidate_budget: int,
    evaluation_budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select actual-document approximate medoids and their 90% radii."""

    medoids = np.empty(cluster_count, dtype=int)
    radii = np.empty(cluster_count, dtype=np.float64)
    for cluster in range(cluster_count):
        members = np.flatnonzero(labels == cluster)
        candidates = members[_uniform_indices(len(members), candidate_budget)]
        evaluation = members[_uniform_indices(len(members), evaluation_budget)]
        distances = 1.0 - np.clip(
            features[candidates] @ features[evaluation].T,
            -1.0,
            1.0,
        )
        means = distances.mean(axis=1)
        # np.argmin gives the first candidate, making ties deterministic.
        medoid = int(candidates[int(np.argmin(means))])
        medoids[cluster] = medoid
        member_distances = 1.0 - np.clip(features[members] @ features[medoid], -1.0, 1.0)
        radii[cluster] = max(float(np.percentile(member_distances, 90)), 1e-12)
    return medoids, radii


def _build_distance_soft_result(
    features: np.ndarray,
    original_labels: np.ndarray,
    *,
    reassignment_threshold: float,
    neighbor_count: int,
    medoid_candidate_budget: int,
    medoid_evaluation_budget: int,
) -> HdbscanSoftResult:
    """Build post-hoc memberships for already normalized HDBSCAN labels."""

    cluster_count = int(np.max(original_labels)) + 1 if np.any(original_labels >= 0) else 0
    n_rows = len(features)
    if cluster_count == 0:
        empty = np.zeros((n_rows, 0), dtype=np.float64)
        rejected = np.full(n_rows, -1, dtype=int)
        return HdbscanSoftResult(features, original_labels, original_labels.copy(), empty, empty.copy(), rejected, rejected.copy(), np.zeros(n_rows), np.zeros(n_rows), np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int))

    medoids, radii = _cluster_representatives(
        features, original_labels, cluster_count,
        candidate_budget=medoid_candidate_budget,
        evaluation_budget=medoid_evaluation_budget,
    )
    medoid_distances = (1.0 - np.clip(features @ features[medoids].T, -1.0, 1.0)) / radii
    neighbor_distances = np.empty((n_rows, cluster_count), dtype=np.float64)
    neighbor_member_counts = np.empty(cluster_count, dtype=int)
    for cluster in range(cluster_count):
        members = np.flatnonzero(original_labels == cluster)
        count = min(neighbor_count, len(members))
        neighbor_member_counts[cluster] = count
        distances = 1.0 - np.clip(features @ features[members].T, -1.0, 1.0)
        nearest = np.partition(distances, kth=count - 1, axis=1)[:, :count]
        neighbor_distances[:, cluster] = nearest.mean(axis=1) / radii[cluster]

    medoid_memberships = _softmax_negative_distances(medoid_distances)
    neighbor_memberships = _softmax_negative_distances(neighbor_distances)
    non_noise = original_labels >= 0
    medoid_memberships[non_noise] = 0.0
    neighbor_memberships[non_noise] = 0.0
    medoid_memberships[non_noise, original_labels[non_noise]] = 1.0
    neighbor_memberships[non_noise, original_labels[non_noise]] = 1.0
    medoid_recommended = np.argmax(medoid_memberships, axis=1).astype(int)
    neighbor_recommended = np.argmax(neighbor_memberships, axis=1).astype(int)
    medoid_confidence = np.max(medoid_memberships, axis=1)
    neighbor_confidence = np.max(neighbor_memberships, axis=1)
    final_labels = original_labels.copy()
    noise = ~non_noise
    final_labels[noise & (medoid_confidence >= reassignment_threshold)] = medoid_recommended[
        noise & (medoid_confidence >= reassignment_threshold)
    ]
    return HdbscanSoftResult(features, original_labels, final_labels, medoid_memberships, neighbor_memberships, medoid_recommended, neighbor_recommended, medoid_confidence, neighbor_confidence, medoids, radii, neighbor_member_counts)


def fit_hdbscan_soft(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    cluster_selection_method: str = "eom",
    reassignment_threshold: float = DEFAULT_REASSIGNMENT_THRESHOLD,
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
    pca_components: int | None = None,
    medoid_candidate_budget: int = DEFAULT_MEDOID_CANDIDATE_BUDGET,
    medoid_evaluation_budget: int = DEFAULT_MEDOID_EVALUATION_BUDGET,
    seed: int = 42,
) -> HdbscanSoftResult:
    """Fit HDBSCAN and compare medoid and nearest-member noise assignments."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("embeddings must be a non-empty finite 2D array")
    if min_cluster_size < 2 or min_samples < 1 or neighbor_count < 1:
        raise ValueError("min_cluster_size >= 2, min_samples >= 1, and neighbor_count >= 1 are required")
    if not 0.0 <= reassignment_threshold <= 1.0:
        raise ValueError("reassignment_threshold must be between 0 and 1")
    if medoid_candidate_budget < 1 or medoid_evaluation_budget < 1:
        raise ValueError("medoid budgets must be positive")

    features, _pca, _selection = fit_clustering_pca(
        values, n_components=pca_components, seed=seed
    )
    raw_labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
    ).fit_predict(features)
    original_labels = _normalize_hdbscan_labels(raw_labels)
    return _build_distance_soft_result(
        features, original_labels,
        reassignment_threshold=reassignment_threshold,
        neighbor_count=neighbor_count,
        medoid_candidate_budget=medoid_candidate_budget,
        medoid_evaluation_budget=medoid_evaluation_budget,
    )
