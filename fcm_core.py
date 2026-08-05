"""Core spherical FCM and normalized PCA transforms."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from clustering_types import FCMResult, HierarchicalModel
from fuzzy_cmeans import (
    SphericalFuzzyCMeans,
    SphericalGeometry,
    memberships_from_squared_dissimilarities,
)
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




def spherical_fcm(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> FCMResult:
    """Run spherical FCM with cosine dissimilarity on unit vectors.

    This compatibility function delegates to the reusable SFCM optimizer.
    """

    return SphericalFuzzyCMeans(
        m=m,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    ).fit(X, n_clusters)


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


def sfcm_memberships_from_centers(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate spherical-FCM memberships from fixed unit prototypes."""

    values = np.asarray(X, dtype=np.float64)
    prototypes = np.asarray(centers, dtype=np.float64)
    if values.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("X and centers must be 2D arrays")
    if prototypes.shape[0] < 1 or values.shape[1] != prototypes.shape[1]:
        raise ValueError("X and centers have incompatible shapes")

    geometry = SphericalGeometry()
    normalized_X = geometry.prepare_samples(values)
    normalized_centers = geometry.prepare_samples(prototypes)
    squared_dissimilarities = geometry.squared_dissimilarities(
        normalized_X,
        normalized_centers,
    )
    memberships = memberships_from_squared_dissimilarities(
        squared_dissimilarities,
        m=m,
    )
    return memberships, np.sqrt(squared_dissimilarities)


def fcm_memberships_from_centers(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for :func:`sfcm_memberships_from_centers`."""

    return sfcm_memberships_from_centers(X, centers, m=m)


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
        local_memberships, _ = sfcm_memberships_from_centers(
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
