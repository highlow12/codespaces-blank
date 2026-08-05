"""Core spherical FCM and normalized PCA transforms."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from clustering_types import FCMResult, HierarchicalModel
from pca_dimension_search import (
    DEFAULT_K_VALUES,
    DEFAULT_MAX_COMPONENTS,
    DEFAULT_MINIMUM_PRESERVATION_GAIN,
)
from pca_dimension_selection import (
    DEFAULT_COMPONENT_STEP,
    DEFAULT_MIN_COMPONENTS,
    PcaDimensionSelection,
    select_pca_dimension_for_data,
)
from pca_projection import (
    PcaPrefixTransformer,
    fit_normalized_pca_projection,
    transform_normalized_pca_projection,
)


DEFAULT_CLUSTERING_PCA_COMPONENTS = 256
DEFAULT_FCM_N_INIT = 10
DEFAULT_FCM_MIN_CENTER_SEPARATION = 1e-3


def _memberships_from_distances(
    distances: np.ndarray,
    *,
    m: float,
) -> np.ndarray:
    epsilon = 1e-12
    zero_rows = np.any(distances <= epsilon, axis=1)
    memberships = np.empty_like(distances, dtype=np.float64)

    regular_distances = np.maximum(distances[~zero_rows], epsilon)
    if regular_distances.size:
        exponent = 2.0 / (m - 1.0)
        ratios = (
            regular_distances[:, :, None]
            / regular_distances[:, None, :]
        ) ** exponent
        memberships[~zero_rows] = 1.0 / ratios.sum(axis=2)

    if np.any(zero_rows):
        exact_matches = distances[zero_rows] <= epsilon
        memberships[zero_rows] = exact_matches / exact_matches.sum(
            axis=1,
            keepdims=True,
        )
    return memberships


def _minimum_center_distance(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return float("inf")
    distances = euclidean_distances(centers, centers)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _fcm_objective(
    memberships: np.ndarray,
    distances: np.ndarray,
    *,
    m: float,
) -> float:
    return float(np.sum((memberships**m) * (distances**2)) / len(memberships))


def _restart_stability(results: list[FCMResult]) -> float:
    if len(results) < 2:
        return 0.0
    scores = [
        adjusted_rand_score(results[first].labels, results[second].labels)
        for first in range(len(results))
        for second in range(first + 1, len(results))
    ]
    return float(np.mean(scores))


def _spherical_fcm_once(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float,
    max_iter: int,
    tol: float,
    seed: int,
    collapse_center_separation: float | None = None,
) -> FCMResult:
    initial_centers, _ = kmeans_plusplus(
        X,
        n_clusters=n_clusters,
        random_state=seed,
    )
    initial_centers = normalize(initial_centers, norm="l2")
    memberships = _memberships_from_distances(
        euclidean_distances(X, initial_centers),
        m=m,
    )

    epsilon = 1e-12
    centers = initial_centers
    distances = euclidean_distances(X, centers)
    for iteration in range(1, max_iter + 1):
        previous = memberships.copy()
        weights = memberships**m
        centers = weights.T @ X
        centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        zero_center_mask = centers_norm[:, 0] < epsilon
        if np.any(zero_center_mask):
            replacement_rng = np.random.default_rng(seed + iteration)
            replacement_indices = replacement_rng.integers(
                0,
                X.shape[0],
                size=int(np.sum(zero_center_mask)),
            )
            centers[zero_center_mask] = X[replacement_indices]
            centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(centers_norm, epsilon)

        distances = euclidean_distances(X, centers)
        memberships = _memberships_from_distances(distances, m=m)

        change = np.max(np.abs(memberships - previous))
        if (
            collapse_center_separation is not None
            and iteration >= 5
            and _minimum_center_distance(centers)
            < collapse_center_separation
        ):
            break
        if change < tol:
            break

    labels = memberships.argmax(axis=1)
    return FCMResult(
        labels=labels,
        memberships=memberships,
        centers=centers,
        iterations=iteration,
        objective=_fcm_objective(memberships, distances, m=m),
        m=m,
        minimum_center_distance=_minimum_center_distance(centers),
    )




def spherical_fcm(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
    n_init: int = DEFAULT_FCM_N_INIT,
    max_attempts: int | None = None,
    min_cluster_size: int = 1,
    min_center_separation: float = DEFAULT_FCM_MIN_CENTER_SEPARATION,
    collapse_center_separation: float | None = None,
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
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")
    if min_center_separation < 0.0:
        raise ValueError("min_center_separation must be non-negative")
    if n_clusters > X.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of samples")

    X = normalize(X, norm="l2")
    attempt_limit = max_attempts if max_attempts is not None else n_init * 3
    if attempt_limit < n_init:
        raise ValueError("max_attempts must be at least n_init")

    valid_results: list[FCMResult] = []
    all_results: list[FCMResult] = []
    attempts = 0
    for attempt in range(attempt_limit):
        attempts = attempt + 1
        result = _spherical_fcm_once(
            X,
            n_clusters,
            m=m,
            max_iter=max_iter,
            tol=tol,
            seed=seed + attempt * 1009,
            collapse_center_separation=collapse_center_separation,
        )
        all_results.append(result)
        cluster_sizes = np.bincount(result.labels, minlength=n_clusters)
        center_separation = (
            result.minimum_center_distance
            if result.minimum_center_distance is not None
            else 0.0
        )
        if (
            int(cluster_sizes.min()) >= min_cluster_size
            and center_separation >= min_center_separation
        ):
            valid_results.append(result)
            if len(valid_results) >= n_init:
                break

    selected_pool = valid_results if valid_results else all_results
    best = min(
        selected_pool,
        key=lambda result: (
            float("inf") if result.objective is None else result.objective,
            -result.minimum_center_distance
            if result.minimum_center_distance is not None
            else float("inf"),
        ),
    )
    return FCMResult(
        labels=best.labels,
        memberships=best.memberships,
        centers=best.centers,
        iterations=best.iterations,
        objective=best.objective,
        m=m,
        n_init=n_init,
        attempts=attempts,
        valid_restarts=len(valid_results),
        restart_stability=_restart_stability(valid_results),
        minimum_center_distance=best.minimum_center_distance,
    )


def pca_normalized_features(
    X: np.ndarray,
    *,
    n_components: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Create the normalized PCA representation used by the default FCM path."""

    projected, _, _ = fit_clustering_pca(
        X,
        n_components=n_components,
        seed=seed,
    )
    return projected


def fit_clustering_pca(
    X: np.ndarray,
    *,
    n_components: int | None = None,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    min_components: int = DEFAULT_MIN_COMPONENTS,
    component_step: int = DEFAULT_COMPONENT_STEP,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    minimum_preservation_gain: float = DEFAULT_MINIMUM_PRESERVATION_GAIN,
    seed: int = 42,
) -> tuple[np.ndarray, PCA | PcaPrefixTransformer, PcaDimensionSelection | None]:
    """Fit clustering PCA, selecting its width automatically by default."""

    selection = None
    if n_components is None:
        selection = select_pca_dimension_for_data(
            X,
            max_components=max_components,
            min_components=min_components,
            component_step=component_step,
            k_values=k_values,
            minimum_preservation_gain=minimum_preservation_gain,
            seed=seed,
        )
        if selection is None:
            projected, pca = fit_pca_normalized_features(
                X,
                n_components=1,
                seed=seed,
            )
            return projected, pca, None
        return (
            selection.selected_features,
            PcaPrefixTransformer(
                selection.pca,
                selection.selected_dimension,
            ),
            selection,
        )

    projected, pca = fit_pca_normalized_features(
        X,
        n_components=n_components,
        seed=seed,
    )
    return projected, pca, selection


def fit_pca_normalized_features(
    X: np.ndarray,
    *,
    n_components: int = DEFAULT_CLUSTERING_PCA_COMPONENTS,
    seed: int = 42,
) -> tuple[np.ndarray, PCA]:
    """Fit the PCA representation and return its transformer for later batches."""

    fitted = fit_normalized_pca_projection(
        X,
        n_components=n_components,
        seed=seed,
    )
    return fitted.normalized_prefix(), fitted.pca


def transform_pca_normalized_features(X: np.ndarray, pca: PCA) -> np.ndarray:
    """Transform a future batch with a previously fitted PCA representation."""

    return transform_normalized_pca_projection(X, pca)


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
    memberships = _memberships_from_distances(distances, m=m)
    return memberships, distances


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
            m=float(getattr(node_model, "m", m)),
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
