from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from clustering_types import (
    FCMKCandidate,
    FCMResult,
    HierarchicalModel,
    HierarchicalResult,
    HierarchyNodeModel,
)


DEFAULT_CLUSTERING_PCA_COMPONENTS = 256
DEFAULT_MAX_MEMBERSHIP_GAP = 0.10
DEFAULT_FORCED_NOISE_RATIO = 0.01
DOCUMENT_TYPE_CORE = "core"
DOCUMENT_TYPE_BOUNDARY = "boundary"
DOCUMENT_TYPE_NOISE = "noise"


def spherical_fcm(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> FCMResult:
    """Run FCM on the unit sphere using Euclidean distances.

    Every invocation re-normalizes its input. The weighted FCM centers are
    projected back to unit length after each update, so both samples and
    centers remain on the same sphere while the distance calculation stays
    ordinary Euclidean distance.
    """

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if m <= 1.0:
        raise ValueError("m must be greater than 1")

    X = normalize(X, norm="l2")
    rng = np.random.default_rng(seed)
    memberships = rng.random((X.shape[0], n_clusters))
    memberships /= memberships.sum(axis=1, keepdims=True)

    epsilon = 1e-12
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    for iteration in range(1, max_iter + 1):
        previous = memberships.copy()
        weights = memberships**m
        centers = weights.T @ X
        centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        zero_center_mask = centers_norm[:, 0] < epsilon
        if np.any(zero_center_mask):
            replacement_indices = rng.integers(
                0,
                X.shape[0],
                size=int(np.sum(zero_center_mask)),
            )
            centers[zero_center_mask] = X[replacement_indices]
            centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(centers_norm, epsilon)

        distances = euclidean_distances(X, centers)
        distances = np.maximum(distances, epsilon)

        exponent = 2.0 / (m - 1.0)
        ratios = (distances[:, :, None] / distances[:, None, :]) ** exponent
        memberships = 1.0 / ratios.sum(axis=2)

        change = np.max(np.abs(memberships - previous))
        if change < tol:
            break

    labels = memberships.argmax(axis=1)
    return FCMResult(
        labels=labels,
        memberships=memberships,
        centers=centers,
        iterations=iteration,
    )


def xie_beni_index(X: np.ndarray, result: FCMResult) -> float:
    X = normalize(X, norm="l2")
    memberships = result.memberships
    centers = result.centers
    distances = euclidean_distances(X, centers)
    numerator = np.sum((memberships**2) * (distances**2))
    center_distances = euclidean_distances(centers, centers)
    np.fill_diagonal(center_distances, np.inf)
    denominator = X.shape[0] * np.min(center_distances) ** 2
    return float(numerator / max(denominator, 1e-12))


def partition_coefficient(result: FCMResult) -> float:
    """Return the FCM partition coefficient (higher is crisper)."""

    memberships = np.asarray(result.memberships, dtype=np.float64)
    if memberships.ndim != 2 or memberships.shape[0] == 0:
        raise ValueError("memberships must be a non-empty 2D array")
    return float(np.mean(np.sum(memberships**2, axis=1)))


def modified_partition_coefficient(result: FCMResult) -> float:
    """Remove the raw partition coefficient's 1/k baseline."""

    cluster_count = result.memberships.shape[1]
    if cluster_count < 2:
        return 1.0
    coefficient = partition_coefficient(result)
    baseline = 1.0 / cluster_count
    return float((coefficient - baseline) / (1.0 - baseline))


def partition_entropy(result: FCMResult) -> float:
    """Return fuzzy partition entropy (lower is crisper)."""

    memberships = np.asarray(result.memberships, dtype=np.float64)
    if memberships.ndim != 2 or memberships.shape[0] == 0:
        raise ValueError("memberships must be a non-empty 2D array")
    safe_memberships = np.maximum(memberships, 1e-12)
    return float(-np.mean(np.sum(memberships * np.log(safe_memberships), axis=1)))


def normalized_partition_entropy(result: FCMResult) -> float:
    """Normalize partition entropy to [0, 1] using log(k)."""

    cluster_count = result.memberships.shape[1]
    if cluster_count < 2:
        return 0.0
    return float(partition_entropy(result) / np.log(cluster_count))


def fuzzy_silhouette_proxy(
    X: np.ndarray,
    result: FCMResult,
    *,
    m: float = 2.0,
) -> float:
    X = normalize(X, norm="l2")
    memberships = result.memberships
    centers = result.centers
    distances = euclidean_distances(X, centers)
    weights = memberships**m
    a = np.sum(weights * distances, axis=1) / np.sum(weights, axis=1)
    b = np.partition(distances, 1, axis=1)[:, 1]
    scores = (b - a) / np.maximum(a, b)
    return float(np.mean(scores))


def pca_normalized_features(
    X: np.ndarray,
    *,
    n_components: int = DEFAULT_CLUSTERING_PCA_COMPONENTS,
    seed: int = 42,
) -> np.ndarray:
    """Create the same normalized PCA representation used by PCA+FCM."""

    projected, _ = fit_pca_normalized_features(
        X,
        n_components=n_components,
        seed=seed,
    )
    return projected


def fit_pca_normalized_features(
    X: np.ndarray,
    *,
    n_components: int = DEFAULT_CLUSTERING_PCA_COMPONENTS,
    seed: int = 42,
) -> tuple[np.ndarray, PCA]:
    """Fit the PCA representation and return its transformer for later batches."""

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if n_components < 1:
        raise ValueError("n_components must be at least 1")

    component_count = min(n_components, X.shape[0], X.shape[1])
    normalized_input = normalize(X, norm="l2")
    pca = PCA(n_components=component_count, random_state=seed).fit(
        normalized_input
    )
    projected = pca.transform(normalized_input)
    return normalize(projected, norm="l2"), pca


def transform_pca_normalized_features(X: np.ndarray, pca: PCA) -> np.ndarray:
    """Transform a future batch with a previously fitted PCA representation."""

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    normalized_input = normalize(X, norm="l2")
    projected = pca.transform(normalized_input)
    return normalize(projected, norm="l2")


def fcm_memberships_from_centers(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate FCM memberships for new points using fixed centers."""

    if m <= 1.0:
        raise ValueError("m must be greater than 1")
    if X.ndim != 2 or centers.ndim != 2:
        raise ValueError("X and centers must be 2D arrays")
    if centers.shape[0] < 1 or X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers have incompatible shapes")

    normalized_X = normalize(X, norm="l2")
    normalized_centers = normalize(centers, norm="l2")
    distances = euclidean_distances(normalized_X, normalized_centers)
    distances = np.maximum(distances, 1e-12)
    exponent = 2.0 / (m - 1.0)
    ratios = (distances[:, :, None] / distances[:, None, :]) ** exponent
    memberships = 1.0 / ratios.sum(axis=2)
    return memberships, distances


def path_membership_column(path: str) -> str:
    """Return a stable assignment-column name for a hierarchy path."""

    if not path:
        raise ValueError("path membership columns require a non-root path")
    parts = path.split("/")
    return f"level_{len(parts)}_path_membership_{'_'.join(parts)}"


def conditional_memberships_from_projected(
    projected: np.ndarray,
    hierarchy_model: HierarchicalModel,
    *,
    m: float = 2.0,
) -> dict[str, np.ndarray]:
    """Propagate local FCM memberships into conditional path probabilities.

    Each node's FCM membership is conditional on its parent. The returned
    values are therefore comparable only after multiplying by the probability
    of reaching that parent. No raw memberships from unrelated parent nodes
    are compared directly.
    """

    if projected.ndim != 2 or projected.shape[0] == 0:
        raise ValueError("projected must be a non-empty 2D array")

    root_probability = np.ones(projected.shape[0], dtype=np.float64)
    if hierarchy_model.fallback_single_cluster:
        return {"0": root_probability}

    probabilities: dict[str, np.ndarray] = {"": root_probability}
    ordered_nodes = sorted(
        hierarchy_model.nodes.items(),
        key=lambda item: (item[0].count("/"), item[0]),
    )
    for parent_path, node_model in ordered_nodes:
        parent_probability = probabilities.get(parent_path)
        if parent_probability is None:
            continue
        local_memberships, _ = fcm_memberships_from_centers(
            projected,
            node_model.centers,
            m=m,
        )
        for cluster_id in range(local_memberships.shape[1]):
            child_path = (
                f"{parent_path}/{cluster_id}"
                if parent_path
                else str(cluster_id)
            )
            probabilities[child_path] = (
                parent_probability * local_memberships[:, cluster_id]
            )

    probabilities.pop("", None)
    return probabilities


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

    Xn = normalize(X, norm="l2")
    labels = result.labels
    distances = euclidean_distances(Xn, result.centers)
    row_indices = np.arange(Xn.shape[0])
    assigned_distances = distances[row_indices, labels]
    assigned_thresholds = np.full(Xn.shape[0], float("inf"), dtype=np.float64)

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


def spherical_fcm_objective(
    X: np.ndarray,
    result: FCMResult,
    *,
    m: float = 2.0,
) -> float:
    """Return the Euclidean fuzzy compactness objective on the unit sphere."""

    Xn = normalize(X, norm="l2")
    distances = euclidean_distances(Xn, result.centers)
    return float(np.sum((result.memberships**m) * (distances**2)) / Xn.shape[0])


def _filter_fcm_labels(
    X: np.ndarray,
    result: FCMResult,
    *,
    min_child_size: int,
    min_membership: float,
    max_membership_gap: float,
    distance_z: float,
) -> tuple[np.ndarray, list[int]]:
    """Apply noise rules and remap surviving FCM labels to contiguous IDs."""

    labels = result.labels.copy()
    labels[
        fcm_noise_mask(
            X,
            result,
            min_membership=min_membership,
            max_membership_gap=max_membership_gap,
            distance_z=distance_z,
        )
    ] = -1

    surviving_old_labels = [
        cluster_id
        for cluster_id in range(result.memberships.shape[1])
        if int(np.sum(labels == cluster_id)) >= min_child_size
    ]
    filtered = np.full(labels.shape, -1, dtype=int)
    cluster_sizes: list[int] = []
    for new_label, old_label in enumerate(surviving_old_labels):
        mask = labels == old_label
        filtered[mask] = new_label
        cluster_sizes.append(int(np.sum(mask)))
    return filtered, cluster_sizes


def _candidate_to_record(candidate: FCMKCandidate) -> dict[str, Any]:
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "k": int(candidate.n_clusters),
        "silhouette": finite_or_none(candidate.silhouette),
        "xie_beni": finite_or_none(candidate.xie_beni),
        "xb_relative_improvement": (
            finite_or_none(candidate.xb_relative_improvement)
            if candidate.xb_relative_improvement is not None
            else None
        ),
        "partition_coefficient": finite_or_none(
            candidate.partition_coefficient
        ),
        "modified_partition_coefficient": finite_or_none(
            candidate.modified_partition_coefficient
        ),
        "partition_entropy": finite_or_none(candidate.partition_entropy),
        "normalized_partition_entropy": finite_or_none(
            candidate.normalized_partition_entropy
        ),
        "selection_score": (
            finite_or_none(candidate.selection_score)
            if candidate.selection_score is not None
            else None
        ),
        "objective": finite_or_none(candidate.objective),
        "valid_clusters": int(len(candidate.cluster_sizes)),
        "noise_count": int(candidate.noise_count),
        "cluster_sizes": [int(size) for size in candidate.cluster_sizes],
    }


def _rank_desirability(
    values: np.ndarray,
    *,
    higher_is_better: bool,
) -> np.ndarray:
    """Convert metric ranks to [0, 1] desirability with averaged ties."""

    if not np.all(np.isfinite(values)):
        raise ValueError("rank desirability requires finite values")
    if values.size == 1:
        return np.ones(values.shape, dtype=np.float64)

    sort_values = -values if higher_is_better else values
    order = np.argsort(sort_values, kind="stable")
    ordered_values = sort_values[order]
    ranks = np.empty(values.shape, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and np.isclose(
            ordered_values[end],
            ordered_values[start],
            rtol=1e-12,
            atol=1e-12,
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return 1.0 - (ranks - 1.0) / (values.size - 1.0)


def _score_multi_metric_candidates(candidates: list[FCMKCandidate]) -> None:
    """Set an equal-weight validity score across three fuzzy metrics."""

    if not candidates:
        return
    metric_specs = (
        ("xie_beni", False, 0.50),
        ("modified_partition_coefficient", True, 0.25),
        ("normalized_partition_entropy", False, 0.25),
    )
    weighted_desirabilities = []
    for attribute, higher_is_better, weight in metric_specs:
        values = np.asarray(
            [getattr(candidate, attribute) for candidate in candidates],
            dtype=np.float64,
        )
        weighted_desirabilities.append(
            weight
            * _rank_desirability(
                values,
                higher_is_better=higher_is_better,
            )
        )
    scores = np.sum(np.vstack(weighted_desirabilities), axis=0)
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.selection_score = float(score)


def _choose_knee_candidate(candidates: list[FCMKCandidate]) -> FCMKCandidate:
    """Choose the largest curvature point of the FCM objective curve."""

    if len(candidates) <= 2:
        return candidates[0]

    objectives = np.asarray([candidate.objective for candidate in candidates], dtype=float)
    if not np.all(np.isfinite(objectives)) or np.ptp(objectives) <= 1e-12:
        return candidates[0]

    x = np.linspace(0.0, 1.0, len(candidates))
    y = (objectives - objectives.min()) / np.ptp(objectives)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 1e-12:
        return candidates[0]

    points = np.column_stack([x, y])
    distances = (
        np.abs(
            line[0] * (start[1] - points[:, 1])
            - line[1] * (start[0] - points[:, 0])
        )
        / line_norm
    )
    best_distance = float(np.max(distances))
    best_indices = np.flatnonzero(np.isclose(distances, best_distance))
    return candidates[int(best_indices[0])]


def select_fcm_cluster_count(
    X: np.ndarray,
    *,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 20,
    min_membership: float = 0.40,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
    distance_z: float = 3.5,
    selection_method: str = "multi_metric",
    min_xb_relative_improvement: float = 0.05,
    xb_worsening_patience: int = 2,
    seed: int = 42,
) -> tuple[FCMKCandidate | None, list[dict[str, Any]], str]:
    """Evaluate increasing k values and return the best FCM split.

    With ``selection_method="multi_metric"``, candidates are evaluated from
    the configured minimum k upward. After XB first worsens, two additional k
    values are evaluated by default. XB, modified partition coefficient, and
    normalized partition entropy are converted to rank desirabilities. They
    are combined with weights 0.50, 0.25, and 0.25 respectively. Silhouette is
    retained only for diagnostics and legacy selection methods. Raw PC and PE
    are retained for reporting.
    """

    if not 0.0 <= min_membership <= 1.0:
        raise ValueError("min_membership must be between 0 and 1")
    if not 0.0 <= max_membership_gap <= 1.0:
        raise ValueError("max_membership_gap must be between 0 and 1")
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be at least min_clusters")
    if min_child_size < 2:
        raise ValueError("min_child_size must be at least 2")
    if selection_method not in {
        "silhouette",
        "knee",
        "xie_beni",
        "multi_metric",
    }:
        raise ValueError(
            "selection_method must be 'silhouette', 'knee', 'xie_beni', "
            "or 'multi_metric'"
        )
    if not 0.0 <= min_xb_relative_improvement <= 1.0:
        raise ValueError("min_xb_relative_improvement must be between 0 and 1")
    if xb_worsening_patience < 0:
        raise ValueError("xb_worsening_patience must be non-negative")

    Xn = normalize(X, norm="l2")
    node_size = Xn.shape[0]
    maximum_k = min(max_clusters, node_size // min_child_size)
    if maximum_k < min_clusters:
        return None, [], "too_few_samples_for_two_valid_children"

    candidates: list[FCMKCandidate] = []
    xb_stop_candidate: FCMKCandidate | None = None
    multi_metric_stop_k: int | None = None
    for candidate_k in range(min_clusters, maximum_k + 1):
        result = spherical_fcm(
            Xn,
            n_clusters=candidate_k,
            seed=seed + candidate_k * 1009,
        )
        if selection_method == "multi_metric":
            labels = result.memberships.argmax(axis=1)
            cluster_sizes = [
                int(np.sum(labels == cluster_id))
                for cluster_id in range(candidate_k)
            ]
        else:
            labels, cluster_sizes = _filter_fcm_labels(
                Xn,
                result,
                min_child_size=min_child_size,
                min_membership=min_membership,
                max_membership_gap=max_membership_gap,
                distance_z=distance_z,
            )
        non_noise = labels != -1
        valid_cluster_count = len(cluster_sizes)
        silhouette = float("nan")
        if valid_cluster_count >= 2 and int(np.sum(non_noise)) >= 2:
            try:
                silhouette = float(
                    silhouette_score(
                        Xn[non_noise],
                        labels[non_noise],
                        metric="euclidean",
                    )
                )
            except Exception:
                silhouette = float("nan")

        xie_beni = xie_beni_index(Xn, result)
        xb_relative_improvement: float | None = None
        if (
            candidates
            and np.isfinite(candidates[-1].xie_beni)
            and np.isfinite(xie_beni)
        ):
            previous_xb = candidates[-1].xie_beni
            xb_relative_improvement = float(
                (previous_xb - xie_beni) / max(abs(previous_xb), 1e-12)
            )

        candidate = FCMKCandidate(
            n_clusters=candidate_k,
            result=result,
            labels=labels,
            silhouette=silhouette,
            xie_beni=xie_beni,
            xb_relative_improvement=xb_relative_improvement,
            partition_coefficient=partition_coefficient(result),
            modified_partition_coefficient=modified_partition_coefficient(
                result
            ),
            partition_entropy=partition_entropy(result),
            normalized_partition_entropy=normalized_partition_entropy(result),
            selection_score=None,
            objective=spherical_fcm_objective(Xn, result),
            noise_count=int(np.sum(~non_noise)),
            cluster_sizes=cluster_sizes,
        )
        candidates.append(candidate)

        previous_candidate = candidates[-2] if len(candidates) >= 2 else None
        current_is_valid = (
            len(candidate.cluster_sizes) >= 2
            and min(candidate.cluster_sizes) >= min_child_size
            and (
                selection_method == "multi_metric"
                or np.isfinite(candidate.silhouette)
            )
            and np.isfinite(candidate.xie_beni)
        )
        previous_is_valid = (
            previous_candidate is not None
            and len(previous_candidate.cluster_sizes) >= 2
            and min(previous_candidate.cluster_sizes) >= min_child_size
            and (
                selection_method == "multi_metric"
                or np.isfinite(previous_candidate.silhouette)
            )
            and np.isfinite(previous_candidate.xie_beni)
        )
        if (
            selection_method in {"xie_beni", "multi_metric"}
            and current_is_valid
            and previous_is_valid
            and xb_relative_improvement is not None
            and (
                (
                    selection_method == "xie_beni"
                    and xb_relative_improvement < min_xb_relative_improvement
                )
                or (
                    selection_method == "multi_metric"
                    and xb_relative_improvement < 0.0
                )
            )
        ):
            if selection_method == "xie_beni":
                eligible = [
                    evaluated
                    for evaluated in candidates[:-1]
                    if len(evaluated.cluster_sizes) >= 2
                    and min(evaluated.cluster_sizes) >= min_child_size
                    and np.isfinite(evaluated.silhouette)
                    and np.isfinite(evaluated.xie_beni)
                ]
                xb_stop_candidate = min(
                    eligible,
                    key=lambda evaluated: (
                        evaluated.xie_beni,
                        evaluated.n_clusters,
                    ),
                )
                break
            if multi_metric_stop_k is None:
                multi_metric_stop_k = min(
                    maximum_k,
                    candidate_k + xb_worsening_patience,
                )

        if (
            selection_method == "multi_metric"
            and multi_metric_stop_k is not None
            and candidate_k >= multi_metric_stop_k
        ):
            break

    valid_candidates = [
        candidate
        for candidate in candidates
        if len(candidate.cluster_sizes) >= 2
        and min(candidate.cluster_sizes) >= min_child_size
        and (
            selection_method == "multi_metric"
            or np.isfinite(candidate.silhouette)
        )
        and (
            selection_method not in {"xie_beni", "multi_metric"}
            or np.isfinite(candidate.xie_beni)
        )
        and (
            selection_method != "multi_metric"
            or (
                np.isfinite(candidate.modified_partition_coefficient)
                and np.isfinite(candidate.normalized_partition_entropy)
            )
        )
    ]
    if not valid_candidates:
        invalid_reason = (
            "no_valid_xie_beni_split"
            if selection_method in {"xie_beni", "multi_metric"}
            else "no_valid_silhouette_split"
        )
        return (
            None,
            [_candidate_to_record(candidate) for candidate in candidates],
            invalid_reason,
        )

    if selection_method == "silhouette":
        best = max(
            valid_candidates,
            key=lambda candidate: (
                candidate.silhouette,
                -candidate.xie_beni
                if np.isfinite(candidate.xie_beni)
                else float("-inf"),
                -candidate.n_clusters,
            ),
        )
    elif selection_method == "knee":
        best = _choose_knee_candidate(valid_candidates)
    elif selection_method == "multi_metric":
        _score_multi_metric_candidates(valid_candidates)
        best = max(
            valid_candidates,
            key=lambda candidate: (
                candidate.selection_score,
                -candidate.xie_beni,
                candidate.modified_partition_coefficient,
                -candidate.normalized_partition_entropy,
                -candidate.n_clusters,
            ),
        )
    elif xb_stop_candidate is not None:
        best = xb_stop_candidate
    else:
        best = min(
            valid_candidates,
            key=lambda candidate: (candidate.xie_beni, candidate.n_clusters),
        )

    if selection_method == "multi_metric":
        selection_reason = (
            "selected_multi_metric_xb_worsening_patience"
            if multi_metric_stop_k is not None
            else "selected_multi_metric_max_k"
        )
    elif selection_method == "xie_beni":
        selection_reason = (
            "selected_xb_relative_improvement"
            if xb_stop_candidate is not None
            else "selected_xb_minimum"
        )
    else:
        selection_reason = "selected"
    return (
        best,
        [_candidate_to_record(candidate) for candidate in candidates],
        selection_reason,
    )


def run_hierarchical_pca_fcm(
    X: np.ndarray,
    metadata: pd.DataFrame | None = None,
    *,
    max_depth: int = 4,
    min_node_size: int = 60,
    min_child_size: int = 20,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_membership: float = 0.40,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
    forced_noise_ratio: float = DEFAULT_FORCED_NOISE_RATIO,
    distance_z: float = 3.5,
    selection_method: str = "multi_metric",
    min_xb_relative_improvement: float = 0.05,
    xb_worsening_patience: int = 2,
    min_split_silhouette: float = 0.05,
    pca_components: int = DEFAULT_CLUSTERING_PCA_COMPONENTS,
    seed: int = 42,
) -> HierarchicalResult:
    """Recursively split a dataset with spherical PCA+FCM."""

    started_at = time.perf_counter()
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if metadata is not None and len(metadata) != X.shape[0]:
        raise ValueError("metadata must contain exactly one row per embedding")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if min_node_size < min_child_size:
        raise ValueError("min_node_size must be at least min_child_size")
    if min_child_size < 2:
        raise ValueError("min_child_size must be at least 2")
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be at least min_clusters")
    if not 0.0 <= min_membership <= 1.0:
        raise ValueError("min_membership must be between 0 and 1")
    if not 0.0 <= max_membership_gap <= 1.0:
        raise ValueError("max_membership_gap must be between 0 and 1")
    if not 0.0 <= forced_noise_ratio <= 1.0:
        raise ValueError("forced_noise_ratio must be between 0 and 1")
    if selection_method not in {
        "silhouette",
        "knee",
        "xie_beni",
        "multi_metric",
    }:
        raise ValueError(
            "selection_method must be 'silhouette', 'knee', 'xie_beni', "
            "or 'multi_metric'"
        )
    if not 0.0 <= min_xb_relative_improvement <= 1.0:
        raise ValueError("min_xb_relative_improvement must be between 0 and 1")
    if xb_worsening_patience < 0:
        raise ValueError("xb_worsening_patience must be non-negative")
    if min_split_silhouette < -1.0 or min_split_silhouette > 1.0:
        raise ValueError("min_split_silhouette must be between -1 and 1")

    if metadata is None:
        metadata = pd.DataFrame({"id": np.arange(X.shape[0])})
    else:
        metadata = metadata.copy()

    Xp, pca = fit_pca_normalized_features(
        X,
        n_components=pca_components,
        seed=seed,
    )
    labels_by_level = np.full((X.shape[0], max_depth), -1, dtype=int)
    soft_memberships_by_level = [
        np.full((X.shape[0], max_clusters), np.nan, dtype=np.float64)
        for _ in range(max_depth)
    ]
    is_noise = np.zeros(X.shape[0], dtype=bool)
    document_types = np.full(X.shape[0], DOCUMENT_TYPE_CORE, dtype=object)
    noise_scores = np.zeros(X.shape[0], dtype=np.float64)
    boundary_level = np.full(X.shape[0], -1, dtype=int)
    noise_level = np.full(X.shape[0], -1, dtype=int)
    node_models: dict[str, HierarchyNodeModel] = {}

    def node_template(
        *,
        node_id: str,
        parent_id: str | None,
        path: str,
        depth: int,
        size: int,
    ) -> dict[str, Any]:
        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "path": path,
            "depth": depth,
            "size": int(size),
            "selected_k": None,
            "selected_silhouette": None,
            "selected_xie_beni": None,
            "selected_partition_coefficient": None,
            "selected_partition_entropy": None,
            "selected_selection_score": None,
            "selected_valid_clusters": 0,
            "selection_reason": None,
            "noise_count": 0,
            "boundary_count": 0,
            "candidate_metrics": [],
            "stop_reason": None,
            "children": [],
        }

    root = node_template(
        node_id="root",
        parent_id=None,
        path="",
        depth=0,
        size=X.shape[0],
    )

    def make_root_fallback(reason: str) -> None:
        labels_by_level[:, 0] = 0
        soft_memberships_by_level[0][:, 0] = 1.0
        root["stop_reason"] = reason
        root["fallback_single_cluster"] = True
        child = node_template(
            node_id="0",
            parent_id="root",
            path="0",
            depth=1,
            size=X.shape[0],
        )
        child["stop_reason"] = f"root_not_split:{reason}"
        root["children"].append(child)

    def recurse(indices: np.ndarray, node: dict[str, Any], depth: int) -> None:
        if depth >= max_depth:
            node["stop_reason"] = "max_depth_reached"
            return
        if indices.size < min_node_size:
            node["stop_reason"] = "node_too_small"
            return

        best, candidate_metrics, reason = select_fcm_cluster_count(
            Xp[indices],
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_child_size=min_child_size,
            min_membership=min_membership,
            max_membership_gap=max_membership_gap,
            distance_z=distance_z,
            selection_method=selection_method,
            min_xb_relative_improvement=min_xb_relative_improvement,
            xb_worsening_patience=xb_worsening_patience,
            seed=seed + depth * 100_003 + indices.size,
        )
        node["candidate_metrics"] = candidate_metrics
        node["selection_reason"] = reason
        if best is None:
            node["stop_reason"] = reason
            if depth == 0:
                make_root_fallback(reason)
            return
        if (
            selection_method != "multi_metric"
            and best.silhouette < min_split_silhouette
        ):
            node["stop_reason"] = "silhouette_below_threshold"
            node["selected_silhouette"] = float(best.silhouette)
            if depth == 0:
                make_root_fallback("silhouette_below_threshold")
            return

        if selection_method == "multi_metric":
            local_labels, effective_cluster_sizes = _filter_fcm_labels(
                Xp[indices],
                best.result,
                min_child_size=min_child_size,
                min_membership=min_membership,
                max_membership_gap=max_membership_gap,
                distance_z=distance_z,
            )
        else:
            local_labels = best.labels.copy()
            effective_cluster_sizes = list(best.cluster_sizes)
        if len(effective_cluster_sizes) < 2:
            node["stop_reason"] = "noise_filter_left_fewer_than_two_clusters"
            if depth == 0:
                make_root_fallback(node["stop_reason"])
            return

        node["selected_k"] = int(best.n_clusters)
        node["selected_silhouette"] = float(best.silhouette)
        node["selected_xie_beni"] = float(best.xie_beni)
        node["selected_partition_coefficient"] = float(
            best.partition_coefficient
        )
        node["selected_partition_entropy"] = float(best.partition_entropy)
        node["selected_selection_score"] = (
            float(best.selection_score)
            if best.selection_score is not None
            else None
        )
        node["selected_valid_clusters"] = int(len(effective_cluster_sizes))
        node["noise_count"] = int(np.sum(local_labels == -1))

        current_level = depth
        local_document_types = fcm_document_types(
            Xp[indices],
            best.result,
            min_membership=min_membership,
            max_membership_gap=max_membership_gap,
            distance_z=distance_z,
        )
        local_distances = euclidean_distances(
            Xp[indices],
            best.result.centers,
        )[np.arange(indices.size), best.result.labels]
        local_noise_scores = fcm_noise_scores(
            best.result.memberships,
            local_distances,
            best.result.labels,
        )
        noise_scores[indices] = np.maximum(
            noise_scores[indices],
            local_noise_scores,
        )
        local_document_types[local_labels == -1] = DOCUMENT_TYPE_NOISE
        boundary_rows = (
            (local_labels >= 0)
            & (local_document_types == DOCUMENT_TYPE_BOUNDARY)
        )
        first_boundary_rows = boundary_rows & (
            document_types[indices] == DOCUMENT_TYPE_CORE
        )
        document_types[indices[boundary_rows]] = DOCUMENT_TYPE_BOUNDARY
        boundary_level[indices[first_boundary_rows]] = current_level + 1
        node["boundary_count"] = int(np.sum(boundary_rows))

        model_centers: list[np.ndarray] = []
        distance_thresholds: list[float] = []
        surviving_source_labels: list[int] = []
        node_features = Xp[indices]
        for cluster_id in range(len(effective_cluster_sizes)):
            cluster_mask = local_labels == cluster_id
            source_labels = best.result.labels[cluster_mask]
            if source_labels.size == 0:
                raise RuntimeError("Selected cluster has no surviving samples")
            source_label = int(np.bincount(source_labels).argmax())
            surviving_source_labels.append(source_label)
            center = best.result.centers[source_label]
            model_centers.append(center.copy())

            cluster_distances = euclidean_distances(
                node_features[cluster_mask],
                center.reshape(1, -1),
            ).ravel()
            if cluster_distances.size < 4:
                distance_thresholds.append(float("inf"))
            else:
                median = float(np.median(cluster_distances))
                mad = float(np.median(np.abs(cluster_distances - median)))
                if mad <= 1e-12:
                    distance_thresholds.append(float("inf"))
                else:
                    distance_thresholds.append(
                        median + distance_z * 1.4826 * mad
                    )

        selected_memberships = best.result.memberships[:, surviving_source_labels]
        membership_sums = selected_memberships.sum(axis=1, keepdims=True)
        normalized_memberships = np.divide(
            selected_memberships,
            membership_sums,
            out=np.zeros_like(selected_memberships),
            where=membership_sums > 1e-12,
        )
        valid_membership_rows = local_labels >= 0
        soft_memberships_by_level[current_level][
            indices[valid_membership_rows], : len(surviving_source_labels)
        ] = normalized_memberships[valid_membership_rows]

        node_models[str(node["path"])] = HierarchyNodeModel(
            path=str(node["path"]),
            depth=current_level,
            centers=np.vstack(model_centers),
            distance_thresholds=np.asarray(distance_thresholds, dtype=np.float64),
        )

        noise_indices = indices[local_labels == -1]
        if noise_indices.size:
            is_noise[noise_indices] = True
            document_types[noise_indices] = DOCUMENT_TYPE_NOISE
            boundary_level[noise_indices] = -1
            noise_level[noise_indices] = current_level + 1

        for cluster_id, cluster_size in enumerate(effective_cluster_sizes):
            child_indices = indices[local_labels == cluster_id]
            if child_indices.size != cluster_size:
                raise RuntimeError("FCM cluster size bookkeeping is inconsistent")
            labels_by_level[child_indices, current_level] = cluster_id
            child_path = (
                f"{node['path']}/{cluster_id}"
                if node["path"]
                else str(cluster_id)
            )
            child = node_template(
                node_id=child_path,
                parent_id=str(node["node_id"]),
                path=child_path,
                depth=current_level + 1,
                size=child_indices.size,
            )
            node["children"].append(child)
            recurse(child_indices, child, depth + 1)

    recurse(np.arange(X.shape[0], dtype=int), root, 0)

    is_natural_noise = is_noise.copy()
    document_ids = (
        metadata["id"].to_numpy()
        if "id" in metadata.columns
        else np.arange(X.shape[0])
    )
    is_forced_noise = forced_noise_mask(
        noise_scores,
        document_ids,
        forced_noise_ratio=forced_noise_ratio,
    )
    is_noise |= is_forced_noise
    forced_only = is_forced_noise & ~is_natural_noise
    document_types[is_forced_noise] = DOCUMENT_TYPE_NOISE
    noise_level[forced_only] = 0

    assigned_depth = np.sum(labels_by_level >= 0, axis=1)
    leaf_level = np.where(assigned_depth > 0, assigned_depth, -1).astype(int)
    leaf_cluster = np.full(X.shape[0], -1, dtype=int)
    has_leaf = assigned_depth > 0
    row_indices = np.arange(X.shape[0])
    leaf_cluster[has_leaf] = labels_by_level[
        row_indices[has_leaf], assigned_depth[has_leaf] - 1
    ]
    leaf_cluster[is_noise] = -1

    model = HierarchicalModel(
        pca=pca,
        nodes=node_models,
        max_depth=max_depth,
        fallback_single_cluster=bool(root.get("fallback_single_cluster", False)),
    )
    conditional_memberships = conditional_memberships_from_projected(Xp, model)

    assignments = metadata.copy()
    for level in range(max_depth):
        assignments[f"level_{level + 1}_cluster"] = labels_by_level[:, level]
        for cluster_id in range(max_clusters):
            assignments[f"level_{level + 1}_membership_{cluster_id}"] = (
                soft_memberships_by_level[level][:, cluster_id]
            )
    assignments["cluster"] = leaf_cluster
    for path, path_membership in conditional_memberships.items():
        assignments[path_membership_column(path)] = path_membership

    cluster_paths: list[str] = []
    for row in range(X.shape[0]):
        path_parts = [
            str(int(label)) for label in labels_by_level[row] if label >= 0
        ]
        if is_noise[row]:
            cluster_paths.append(
                "/".join(path_parts + ["noise"]) if path_parts else "noise"
            )
        else:
            cluster_paths.append("/".join(path_parts) if path_parts else "root")
    assignments["cluster_path"] = cluster_paths
    assignments["is_noise"] = is_noise
    assignments["is_natural_noise"] = is_natural_noise
    assignments["is_forced_noise"] = is_forced_noise
    assignments["is_boundary"] = document_types == DOCUMENT_TYPE_BOUNDARY
    assignments["document_type"] = document_types
    assignments["noise_score"] = noise_scores
    assignments["boundary_level"] = boundary_level
    assignments["noise_level"] = noise_level
    assignments["leaf_level"] = leaf_level

    def collect_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = [node]
        for child in node["children"]:
            nodes.extend(collect_nodes(child))
        return nodes

    nodes = collect_nodes(root)
    non_root_nodes = [node for node in nodes if node["depth"] > 0]
    levels_reached = int(np.max(assigned_depth)) if np.any(assigned_depth) else 0
    summary: dict[str, Any] = {
        "method": f"recursive_pca{Xp.shape[1]}_spherical_fcm",
        "samples": int(X.shape[0]),
        "pca_components": int(Xp.shape[1]),
        "levels_requested": int(max_depth),
        "levels_reached": levels_reached,
        "node_count": int(len(non_root_nodes)),
        "leaf_count": int(sum(not node["children"] for node in nodes)),
        "noise_count": int(np.sum(is_noise)),
        "natural_noise_count": int(np.sum(is_natural_noise)),
        "forced_noise_count": int(np.sum(is_forced_noise)),
        "forced_only_noise_count": int(np.sum(forced_only)),
        "boundary_count": int(
            np.sum(document_types == DOCUMENT_TYPE_BOUNDARY)
        ),
        "core_count": int(np.sum(document_types == DOCUMENT_TYPE_CORE)),
        "noise_by_level": {
            str(level): int(np.sum(noise_level == level))
            for level in range(1, max_depth + 1)
        },
        "leaf_cluster_count": int(
            assignments.loc[~assignments["is_noise"], "cluster_path"].nunique()
        ),
        "runtime_sec": float(time.perf_counter() - started_at),
    }
    config = {
        "max_depth": int(max_depth),
        "min_node_size": int(min_node_size),
        "min_child_size": int(min_child_size),
        "min_clusters": int(min_clusters),
        "max_clusters": int(max_clusters),
        "selection_method": selection_method,
        "min_xb_relative_improvement": float(min_xb_relative_improvement),
        "xb_worsening_patience": int(xb_worsening_patience),
        "multi_metric_weights": {
            "xie_beni": 0.50,
            "modified_partition_coefficient": 0.25,
            "normalized_partition_entropy": 0.25,
        },
        "multi_metric_normalization": "rank_average_ties",
        "multi_metric_candidate_metrics_include_all_samples": True,
        "multi_metric_assign_all_samples": False,
        "min_split_silhouette": float(min_split_silhouette),
        "min_membership": float(min_membership),
        "max_membership_gap": float(max_membership_gap),
        "forced_noise_ratio": float(forced_noise_ratio),
        "distance_z": float(distance_z),
        "pca_components_requested": int(pca_components),
        "seed": int(seed),
    }
    tree = {"config": config, "summary": summary, "root": root}
    return HierarchicalResult(
        assignments=assignments,
        tree=tree,
        summary=summary,
        memberships={
            level + 1: level_memberships
            for level, level_memberships in enumerate(soft_memberships_by_level)
        },
        conditional_memberships=conditional_memberships,
        model=model,
    )
