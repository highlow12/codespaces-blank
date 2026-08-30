"""Phase 1-3 comparison of HDBSCAN native and PCA-space memberships.

The discovery branch deliberately uses UMAP, while the membership branch
stays in the unnormalized PCA prefix.  This module does not implement the
later adaptive-neighborhood, HNSW, hierarchy, or incremental stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import hdbscan
import numpy as np
from umap import UMAP

from pca_neighbor_search import PcaNeighborIndex, build_pca_neighbor_index

from pca_dimension_search import (
    DEFAULT_K_VALUES,
    DEFAULT_MINIMUM_PRESERVATION_GAIN,
)
from pca_dimension_selection import (
    DEFAULT_COMPONENT_STEP,
    DEFAULT_MIN_COMPONENTS,
    select_pca_dimension_for_data,
    PcaDimensionSelection,
)
from pca_projection import validate_embedding_matrix


DEFAULT_MAX_PCA_COMPONENTS = 512
DEFAULT_UMAP_COMPONENTS = 20
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3
DEFAULT_NEIGHBOR_COUNT = 15
DEFAULT_UMAP_N_NEIGHBORS = 15


@dataclass(frozen=True)
class ExactKnnPropagationResult:
    """Independent affinities produced by fixed-width exact kNN voting."""

    affinities: np.ndarray
    unexplained: np.ndarray
    max_affinity: np.ndarray
    recommended_labels: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    local_sigmas: np.ndarray


@dataclass(frozen=True)
class HdbscanMembershipComparisonResult:
    """All model outputs needed by the artifact and review pipelines."""

    pca_features: np.ndarray
    umap_features: np.ndarray
    leaf_labels: np.ndarray
    probabilities: np.ndarray
    outlier_scores: np.ndarray
    native_memberships: np.ndarray
    native_unexplained: np.ndarray
    native_max_affinity: np.ndarray
    native_recommended_labels: np.ndarray
    exact_knn: ExactKnnPropagationResult
    pca_selection: PcaDimensionSelection
    configuration: dict[str, Any]
    runtime_seconds: dict[str, float]
    neighbor_index: PcaNeighborIndex | None = None

    @property
    def cluster_count(self) -> int:
        return int(self.native_memberships.shape[1])


def _validate_probability_vector(
    values: np.ndarray,
    n_samples: int,
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (n_samples,):
        raise ValueError(f"{name} must have shape ({n_samples},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result < -1e-12) or np.any(result > 1.0 + 1e-12):
        raise ValueError(f"{name} must be between 0 and 1")
    return np.clip(result, 0.0, 1.0)


def validate_leaf_labels(labels: np.ndarray, n_samples: int) -> tuple[np.ndarray, int]:
    """Validate HDBSCAN's contiguous ``-1, 0, ..., C-1`` label contract."""

    result = np.asarray(labels)
    if result.shape != (n_samples,):
        raise ValueError(
            f"leaf_labels must have shape ({n_samples},), got {result.shape}"
        )
    if not np.issubdtype(result.dtype, np.integer):
        try:
            finite = np.all(np.isfinite(result))
            integral = np.all(result == result.astype(int))
        except (TypeError, ValueError) as error:
            raise ValueError("leaf_labels must contain integer values") from error
        if not finite or not integral:
            raise ValueError("leaf_labels must contain integer values")
        result = result.astype(np.int64)
    else:
        result = result.astype(np.int64, copy=False)
    if np.any(result < -1):
        raise ValueError("leaf_labels may only contain -1 or non-negative labels")
    non_noise = result[result >= 0]
    cluster_count = 0 if non_noise.size == 0 else int(non_noise.max()) + 1
    if cluster_count and not np.array_equal(
        np.unique(non_noise), np.arange(cluster_count, dtype=np.int64)
    ):
        raise ValueError("non-noise leaf_labels must be contiguous from zero")
    return result, cluster_count


def normalize_native_membership_vectors(
    raw_memberships: np.ndarray,
    *,
    n_samples: int,
    cluster_count: int,
) -> np.ndarray:
    """Normalize HDBSCAN's membership shape, including its no-cluster quirk.

    hdbscan 0.8.44 returns a one-dimensional all-zero vector when no leaf is
    discovered.  Internally that means ``(n_samples, 0)`` clusters, not one
    cluster per row, so the vector is converted explicitly and validated.
    """

    raw = np.asarray(raw_memberships, dtype=np.float64)
    if cluster_count == 0:
        if raw.ndim == 1:
            if raw.shape != (n_samples,):
                raise ValueError(
                    "HDBSCAN no-cluster membership must have shape "
                    f"({n_samples},), got {raw.shape}"
                )
            if not np.all(np.isfinite(raw)):
                raise ValueError("HDBSCAN no-cluster membership must be finite")
            if not np.allclose(raw, 0.0, atol=1e-12):
                raise ValueError(
                    "HDBSCAN returned non-zero one-dimensional membership "
                    "without any discovered clusters"
                )
            return np.zeros((n_samples, 0), dtype=np.float64)
        if raw.shape == (n_samples, 0):
            return np.zeros((n_samples, 0), dtype=np.float64)
        raise ValueError(
            "HDBSCAN no-cluster membership must be a zero vector or shape "
            f"({n_samples}, 0), got {raw.shape}"
        )

    if raw.ndim != 2 or raw.shape != (n_samples, cluster_count):
        raise ValueError(
            "HDBSCAN native membership must have shape "
            f"({n_samples}, {cluster_count}), got {raw.shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("HDBSCAN native memberships must contain only finite values")
    if np.any(raw < -1e-12) or np.any(raw > 1.0 + 1e-12):
        raise ValueError("HDBSCAN native memberships must be between 0 and 1")
    return np.clip(raw, 0.0, 1.0)


def _recommended_labels(affinities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(affinities, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("affinities must be a finite 2D array")
    if values.shape[1] == 0:
        return (
            np.zeros(values.shape[0], dtype=np.float64),
            np.full(values.shape[0], -1, dtype=np.int64),
        )
    maximum = np.max(values, axis=1)
    labels = np.argmax(values, axis=1).astype(np.int64)
    labels[maximum <= 0.0] = -1
    return maximum, labels


def _local_sigma(
    distances: np.ndarray,
    *,
    fallback: float,
) -> np.ndarray:
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    global_fallback = (
        float(np.median(positive))
        if positive.size and np.isfinite(np.median(positive))
        else fallback
    )
    if not np.isfinite(global_fallback) or global_fallback <= 0.0:
        global_fallback = fallback
    sigmas = np.empty(distances.shape[0], dtype=np.float64)
    for row_index, row in enumerate(distances):
        row_positive = row[np.isfinite(row) & (row > 0.0)]
        sigma = (
            float(np.median(row_positive))
            if row_positive.size
            else global_fallback
        )
        sigmas[row_index] = (
            sigma if np.isfinite(sigma) and sigma > 0.0 else global_fallback
        )
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("local neighborhood scales must be finite and positive")
    return sigmas


def propagate_exact_knn_memberships(
    pca_features: np.ndarray,
    leaf_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
    neighbor_backend: str = "exact",
    neighbor_index: PcaNeighborIndex | None = None,
    random_state: int = 42,
    graph_neighbors: int = 32,
    query_epsilon: float = 0.1,
) -> ExactKnnPropagationResult:
    """Propagate HDBSCAN leaf confidence through fixed PCA-space kNN neighborhoods.

    ``neighbor_backend='exact'`` preserves the historical brute-force path;
    ``'pynndescent'`` uses a deterministic Euclidean NNDescent index.  A
    supplied index is reused, which is important when evaluating several k
    values against the same discovery projection.
    """

    features = validate_embedding_matrix(pca_features, name="pca_features")
    n_samples = features.shape[0]
    labels, cluster_count = validate_leaf_labels(leaf_labels, n_samples)
    confidence = _validate_probability_vector(
        probabilities, n_samples, name="probabilities"
    )
    if neighbor_count < 1 or neighbor_count >= n_samples:
        raise ValueError(
            "neighbor_count must be between 1 and n_samples - 1; "
            f"got {neighbor_count} for {n_samples} samples"
        )

    if neighbor_index is None:
        neighbor_index = build_pca_neighbor_index(
            features,
            backend=neighbor_backend,
            max_neighbors=neighbor_count,
            graph_neighbors=graph_neighbors,
            random_state=random_state,
            query_epsilon=query_epsilon,
        )
    elif neighbor_index.backend != neighbor_backend:
        raise ValueError("neighbor_index backend does not match neighbor_backend")
    selected_distances, selected_indices = neighbor_index.query(
        features, neighbor_count, exclude_self=True
    )
    if not np.all(np.isfinite(selected_distances)):
        raise ValueError("exact kNN distances must contain only finite values")

    sigmas = _local_sigma(selected_distances, fallback=1.0)
    weights = np.exp(-np.square(selected_distances / sigmas[:, None]))
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("exact kNN distance weights must be finite and non-negative")
    denominators = np.sum(weights, axis=1)
    if not np.all(np.isfinite(denominators)) or np.any(denominators <= 0.0):
        raise ValueError("exact kNN denominators must be finite and positive")

    affinities = np.zeros((n_samples, cluster_count), dtype=np.float64)
    for row_index, neighbor_row in enumerate(selected_indices):
        neighbor_weights = weights[row_index]
        neighbor_labels = labels[neighbor_row]
        votes = neighbor_weights * confidence[neighbor_row]
        for cluster in range(cluster_count):
            affinities[row_index, cluster] = np.sum(
                votes[neighbor_labels == cluster]
            ) / denominators[row_index]
    affinities = np.clip(affinities, 0.0, 1.0)
    unexplained = np.clip(1.0 - np.sum(affinities, axis=1), 0.0, 1.0)
    maximum, recommended = _recommended_labels(affinities)
    return ExactKnnPropagationResult(
        affinities=affinities,
        unexplained=unexplained,
        max_affinity=maximum,
        recommended_labels=recommended,
        neighbor_indices=selected_indices,
        neighbor_distances=selected_distances,
        local_sigmas=sigmas,
    )


def _raw_selected_pca_prefix(selection: PcaDimensionSelection) -> np.ndarray:
    """Recover the unnormalized PCA prefix from the shared PCA fit."""

    raw_prefix = np.asarray(selection.pca.transform(selection.normalized_input))[
        :, : selection.selected_dimension
    ]
    return validate_embedding_matrix(raw_prefix, name="raw_pca_features")


def _fit_pca_selection(
    embeddings: np.ndarray,
    *,
    pca_components: int | None,
    pca_max_components: int,
    pca_min_components: int,
    pca_component_step: int,
    k_values: Sequence[int],
    minimum_preservation_gain: float,
    seed: int,
) -> PcaDimensionSelection:
    if pca_components is not None:
        if pca_components < 1:
            raise ValueError("pca_components must be at least 1")
        max_components = pca_components
        min_components = pca_components
        component_step = 1
    else:
        max_components = pca_max_components
        min_components = pca_min_components
        component_step = pca_component_step
    selection = select_pca_dimension_for_data(
        embeddings,
        max_components=max_components,
        min_components=min_components,
        component_step=component_step,
        k_values=k_values,
        minimum_preservation_gain=minimum_preservation_gain,
        seed=seed,
    )
    if selection is None:
        raise ValueError("PCA selection requires at least two samples")
    return selection


def fit_hdbscan_membership_comparison(
    embeddings: np.ndarray,
    *,
    pca_components: int | None = None,
    pca_max_components: int = DEFAULT_MAX_PCA_COMPONENTS,
    pca_min_components: int = DEFAULT_MIN_COMPONENTS,
    pca_component_step: int = DEFAULT_COMPONENT_STEP,
    pca_k_values: Sequence[int] = DEFAULT_K_VALUES,
    minimum_preservation_gain: float = DEFAULT_MINIMUM_PRESERVATION_GAIN,
    umap_components: int = DEFAULT_UMAP_COMPONENTS,
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
    neighbor_backend: str = "exact",
    neighbor_graph_neighbors: int = 32,
    neighbor_query_epsilon: float = 0.1,
    seed: int = 42,
) -> HdbscanMembershipComparisonResult:
    """Fit discovery and both membership methods on one embedding matrix."""

    matrix = validate_embedding_matrix(embeddings, name="embeddings")
    if matrix.shape[0] < 3:
        raise ValueError("comparison requires at least 3 samples")
    if np.any(np.linalg.norm(matrix, axis=1) <= 1e-12):
        raise ValueError("embeddings must have non-zero L2 norm per row")
    if umap_components < 1:
        raise ValueError("umap_components must be at least 1")
    if umap_n_neighbors < 2:
        raise ValueError("umap_n_neighbors must be at least 2")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    if neighbor_count < 1 or neighbor_count >= matrix.shape[0]:
        raise ValueError("neighbor_count must be between 1 and n_samples - 1")

    started = time.perf_counter()
    selection = _fit_pca_selection(
        matrix,
        pca_components=pca_components,
        pca_max_components=pca_max_components,
        pca_min_components=pca_min_components,
        pca_component_step=pca_component_step,
        k_values=pca_k_values,
        minimum_preservation_gain=minimum_preservation_gain,
        seed=seed,
    )
    pca_features = _raw_selected_pca_prefix(selection)
    pca_seconds = time.perf_counter() - started

    discovery_started = time.perf_counter()
    effective_umap_neighbors = min(umap_n_neighbors, matrix.shape[0] - 1)
    umap_model = UMAP(
        n_components=umap_components,
        n_neighbors=effective_umap_neighbors,
        init="random",
        random_state=seed,
        n_jobs=1,
    )
    umap_features = np.asarray(umap_model.fit_transform(pca_features), dtype=np.float64)
    if umap_features.shape != (matrix.shape[0], umap_components):
        raise ValueError(
            "UMAP output has unexpected shape "
            f"{umap_features.shape}; expected ({matrix.shape[0]}, {umap_components})"
        )
    if not np.all(np.isfinite(umap_features)):
        raise ValueError("UMAP output must contain only finite values")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    ).fit(umap_features)
    leaf_labels, cluster_count = validate_leaf_labels(
        np.asarray(clusterer.labels_), matrix.shape[0]
    )
    probabilities = _validate_probability_vector(
        clusterer.probabilities_, matrix.shape[0], name="hdbscan_probabilities"
    )
    outlier_scores = _validate_probability_vector(
        clusterer.outlier_scores_, matrix.shape[0], name="hdbscan_outlier_scores"
    )
    native_raw = hdbscan.all_points_membership_vectors(clusterer)
    native_memberships = normalize_native_membership_vectors(
        native_raw,
        n_samples=matrix.shape[0],
        cluster_count=cluster_count,
    )
    native_unexplained = np.clip(
        1.0 - np.sum(native_memberships, axis=1), 0.0, 1.0
    )
    native_max, native_recommended = _recommended_labels(native_memberships)
    discovery_seconds = time.perf_counter() - discovery_started

    propagation_started = time.perf_counter()
    neighbor_index = build_pca_neighbor_index(
        pca_features,
        backend=neighbor_backend,
        max_neighbors=neighbor_count,
        graph_neighbors=neighbor_graph_neighbors,
        random_state=42,
        query_epsilon=neighbor_query_epsilon,
    )
    exact_knn = propagate_exact_knn_memberships(
        pca_features,
        leaf_labels,
        probabilities,
        neighbor_count=neighbor_count,
        neighbor_backend=neighbor_backend,
        neighbor_index=neighbor_index,
        random_state=42,
        graph_neighbors=neighbor_graph_neighbors,
        query_epsilon=neighbor_query_epsilon,
    )
    propagation_seconds = time.perf_counter() - propagation_started
    runtime_seconds = {
        "pca_selection": float(pca_seconds),
        "umap_hdbscan_discovery": float(discovery_seconds),
        "exact_knn_propagation": float(propagation_seconds),
        "total": float(time.perf_counter() - started),
    }
    configuration = {
        "input_l2_normalization": True,
        "post_pca_l2_normalization": False,
        "pca_components_requested": pca_components,
        "pca_components_selected": int(selection.selected_dimension),
        "umap_components": int(umap_components),
        "umap_n_neighbors": int(effective_umap_neighbors),
        "umap_random_state": int(seed),
        "umap_n_jobs": 1,
        "umap_init": "random",
        "hdbscan_cluster_selection_method": "leaf",
        "hdbscan_prediction_data": True,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "neighbor_count": int(neighbor_count),
        "neighbor_search": neighbor_backend,
        "neighbor_graph_neighbors": int(neighbor_graph_neighbors),
        "neighbor_random_state": 42,
        "neighbor_query_epsilon": float(neighbor_query_epsilon),
        "distance_metric": "euclidean",
        "local_sigma": "median_positive_selected_neighbor_distance",
        "membership_weight": "exp(-(distance / sigma)^2)",
        "membership_normalization": "independent_affinity",
        "pca_selection": selection.to_dict(),
    }
    return HdbscanMembershipComparisonResult(
        pca_features=pca_features,
        umap_features=umap_features,
        leaf_labels=leaf_labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
        native_memberships=native_memberships,
        native_unexplained=native_unexplained,
        native_max_affinity=native_max,
        native_recommended_labels=native_recommended,
        exact_knn=exact_knn,
        pca_selection=selection,
        configuration=configuration,
        runtime_seconds=runtime_seconds,
        neighbor_index=neighbor_index,
    )


# Split-aware API ---------------------------------------------------------
# Kept in this established module so callers of the original comparison API
# can migrate to out-of-sample prediction without changing their import path.
def fit_discovery_state(
    embeddings: np.ndarray,
    discovery_metadata: Sequence[dict[str, Any]],
    **kwargs: Any,
) -> Any:
    """Fit PCA -> UMAP -> leaf-HDBSCAN using discovery rows only."""
    from wikipedia_soft_benchmark.hierarchy_benchmark import fit_discovery

    return fit_discovery(embeddings, discovery_metadata, **kwargs)


def predict_native_memberships(
    discovery_state: Any,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Return HDBSCAN native memberships for rows absent from discovery."""
    from wikipedia_soft_benchmark.hierarchy_benchmark import predict_memberships

    return predict_memberships(discovery_state, embeddings, neighbor_count=1).native


def predict_exact_knn_memberships(
    discovery_state: Any,
    embeddings: np.ndarray,
    *,
    neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
) -> np.ndarray:
    """Return independent PCA exact-kNN affinities against discovery rows."""
    from wikipedia_soft_benchmark.hierarchy_benchmark import predict_memberships

    return predict_memberships(discovery_state, embeddings, neighbor_count=neighbor_count).exact_knn
