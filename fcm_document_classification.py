"""Boundary and noise classification for fuzzy cluster assignments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clustering_types import FCMResult
from fcm_core import sfcm_memberships_from_centers
from hierarchical_assignments import (
    DOCUMENT_TYPE_BOUNDARY,
    DOCUMENT_TYPE_CORE,
    DOCUMENT_TYPE_NOISE,
)


DEFAULT_MAX_MEMBERSHIP_GAP = 0.10
DEFAULT_FORCED_NOISE_RATIO = 0.0




def fcm_membership_boundary_mask(
    memberships: np.ndarray,
    *,
    min_membership: float = 0.40,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
) -> np.ndarray:
    """Identify points with both low confidence and a small top-two gap.

    A point is marked only when its largest membership is below
    ``min_membership`` and the gap between its largest and second-largest
    memberships is below ``max_membership_gap``.
    """

    values = np.asarray(memberships, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("memberships must contain only finite values")
    if not 0.0 <= min_membership <= 1.0:
        raise ValueError("min_membership must be between 0 and 1")
    if not 0.0 <= max_membership_gap <= 1.0:
        raise ValueError("max_membership_gap must be between 0 and 1")
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=bool)

    top_two = np.partition(values, -2, axis=1)[:, -2:]
    largest = top_two.max(axis=1)
    second_largest = top_two.min(axis=1)
    return (largest < min_membership) & (
        largest - second_largest < max_membership_gap
    )


def classify_fcm_documents(
    memberships: np.ndarray,
    assigned_distances: np.ndarray,
    distance_thresholds: np.ndarray,
    *,
    min_membership: float = 0.40,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
) -> np.ndarray:
    """Classify documents as core, boundary, or noise from three signals.

    Low maximum membership and a small top-two gap form an ambiguous boundary
    candidate. A candidate is noise only when it is also farther from its
    assigned center than the supplied distance threshold.
    """

    values = np.asarray(memberships, dtype=np.float64)
    distances = np.asarray(assigned_distances, dtype=np.float64)
    thresholds = np.asarray(distance_thresholds, dtype=np.float64)
    if distances.ndim != 1 or thresholds.ndim != 1:
        raise ValueError("assigned distances and thresholds must be 1D arrays")
    if distances.shape[0] != values.shape[0] or thresholds.shape[0] != values.shape[0]:
        raise ValueError("distance arrays must align with membership rows")
    if not np.all(np.isfinite(distances)):
        raise ValueError("assigned distances must contain only finite values")
    if np.any(np.isnan(thresholds)):
        raise ValueError("distance thresholds must not contain NaN")

    boundary_candidates = fcm_membership_boundary_mask(
        values,
        min_membership=min_membership,
        max_membership_gap=max_membership_gap,
    )
    far_from_center = distances > thresholds
    document_types = np.full(values.shape[0], DOCUMENT_TYPE_CORE, dtype=object)
    document_types[boundary_candidates] = DOCUMENT_TYPE_BOUNDARY
    document_types[boundary_candidates & far_from_center] = DOCUMENT_TYPE_NOISE
    return document_types


def fcm_noise_scores(
    memberships: np.ndarray,
    assigned_distances: np.ndarray,
    assigned_labels: np.ndarray,
) -> np.ndarray:
    """Rank noise risk from confidence, ambiguity, and center distance."""

    values = np.asarray(memberships, dtype=np.float64)
    distances = np.asarray(assigned_distances, dtype=np.float64)
    labels = np.asarray(assigned_labels)
    if values.ndim != 2:
        raise ValueError("memberships must be a 2D array")
    if distances.ndim != 1 or labels.ndim != 1:
        raise ValueError("assigned distances and labels must be 1D arrays")
    if distances.shape[0] != values.shape[0] or labels.shape[0] != values.shape[0]:
        raise ValueError("assigned arrays must align with membership rows")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(distances)):
        raise ValueError("score inputs must contain only finite values")
    if values.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=np.float64)

    top_two = np.partition(values, -2, axis=1)[:, -2:]
    largest = top_two.max(axis=1)
    membership_gap = largest - top_two.min(axis=1)

    def percentile_rank(signal: np.ndarray) -> np.ndarray:
        return pd.Series(signal).rank(method="average", pct=True).to_numpy()

    low_confidence_rank = percentile_rank(1.0 - largest)
    ambiguity_rank = percentile_rank(1.0 - membership_gap)
    distance_rank = np.zeros(values.shape[0], dtype=np.float64)
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        distance_rank[cluster_mask] = percentile_rank(distances[cluster_mask])

    return np.cbrt(low_confidence_rank * ambiguity_rank * distance_rank)


def forced_noise_mask(
    noise_scores: np.ndarray,
    document_ids: np.ndarray,
    *,
    forced_noise_ratio: float = DEFAULT_FORCED_NOISE_RATIO,
) -> np.ndarray:
    """Select the highest-risk fraction, breaking equal scores by document ID."""

    scores = np.asarray(noise_scores, dtype=np.float64)
    ids = np.asarray(document_ids)
    if scores.ndim != 1 or ids.ndim != 1 or scores.shape[0] != ids.shape[0]:
        raise ValueError("noise scores and document IDs must be aligned 1D arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("noise scores must contain only finite values")
    if not 0.0 <= forced_noise_ratio <= 1.0:
        raise ValueError("forced_noise_ratio must be between 0 and 1")

    selected = np.zeros(scores.shape[0], dtype=bool)
    selected_count = min(
        scores.shape[0],
        int(np.ceil(scores.shape[0] * forced_noise_ratio)),
    )
    if selected_count == 0:
        return selected

    row_indices = np.arange(scores.shape[0])
    order = np.lexsort((row_indices, ids.astype(str), -scores))
    selected[order[:selected_count]] = True
    return selected


def merge_forced_noise(
    natural_noise: np.ndarray,
    noise_scores: np.ndarray,
    document_ids: np.ndarray,
    document_types: np.ndarray,
    noise_level: np.ndarray,
    *,
    forced_noise_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge the global forced-noise quota into natural classifications."""

    forced_noise = forced_noise_mask(
        noise_scores,
        document_ids,
        forced_noise_ratio=forced_noise_ratio,
    )
    natural_noise = np.asarray(natural_noise, dtype=bool)
    combined_noise = natural_noise | forced_noise
    forced_only = forced_noise & ~natural_noise
    updated_types = np.asarray(document_types, dtype=object).copy()
    updated_types[forced_noise] = DOCUMENT_TYPE_NOISE
    updated_noise_level = np.asarray(noise_level, dtype=int).copy()
    updated_noise_level[forced_only] = 0
    return (
        combined_noise,
        forced_noise,
        forced_only,
        updated_types,
        updated_noise_level,
    )


def fcm_document_types(
    X: np.ndarray,
    result: FCMResult,
    *,
    min_membership: float = 0.40,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
    distance_z: float = 3.5,
) -> np.ndarray:
    """Classify fitted FCM samples using robust per-cluster distances."""

    if distance_z < 0.0:
        raise ValueError("distance_z must be non-negative")

    labels = result.labels
    _, distances = sfcm_memberships_from_centers(X, result.centers)
    row_indices = np.arange(X.shape[0])
    assigned_distances = distances[row_indices, labels]
    assigned_thresholds = np.full(X.shape[0], float("inf"), dtype=np.float64)

    for cluster_id in range(result.memberships.shape[1]):
        cluster_mask = labels == cluster_id
        cluster_distances = assigned_distances[cluster_mask]
        if cluster_distances.size < 4:
            continue

        median = float(np.median(cluster_distances))
        mad = float(np.median(np.abs(cluster_distances - median)))
        if mad <= 1e-12:
            continue
        assigned_thresholds[cluster_mask] = median + distance_z * 1.4826 * mad

    return classify_fcm_documents(
        result.memberships,
        assigned_distances,
        assigned_thresholds,
        min_membership=min_membership,
        max_membership_gap=max_membership_gap,
    )


def fcm_noise_mask(
    X: np.ndarray,
    result: FCMResult,
    *,
    min_membership: float = 0.40,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
    distance_z: float = 3.5,
) -> np.ndarray:
    """Return documents satisfying all membership and distance noise rules."""

    return fcm_document_types(
        X,
        result,
        min_membership=min_membership,
        max_membership_gap=max_membership_gap,
        distance_z=distance_z,
    ) == DOCUMENT_TYPE_NOISE
