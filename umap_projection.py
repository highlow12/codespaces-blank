"""PCA + UMAP fitting and reusable projection transforms."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "codex-numba"),
)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from embedding_data import load_embeddings_from_json
from pca_projection import (
    PcaPrefixTransformer,
    fit_normalized_pca_projection,
    transform_normalized_pca_projection,
    validate_embedding_matrix,
)
from visualization_constants import (
    DEFAULT_CLUSTER_TARGET_WEIGHT,
    DEFAULT_VISUAL_PCA_COMPONENTS,
)


def _load_umap() -> Any:
    """Load UMAP with a writable Numba cache in restricted environments."""

    from umap import UMAP

    return UMAP


def load_embeddings(json_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    return load_embeddings_from_json(json_path)


def project_embeddings(
    embeddings: np.ndarray,
    *,
    seed: int,
    pca_components: int | None = None,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
    normalize_pca_output: bool = False,
    cluster_target: np.ndarray | None = None,
    cluster_target_metric: str | None = None,
    cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> np.ndarray:
    _, _, coordinates = fit_projection_model(
        embeddings,
        seed=seed,
        pca_components=pca_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
        normalize_pca_output=normalize_pca_output,
        cluster_target=cluster_target,
        cluster_target_metric=cluster_target_metric,
        cluster_target_weight=cluster_target_weight,
    )
    return coordinates


def fit_projection_model(
    embeddings: np.ndarray,
    *,
    seed: int,
    pca_components: int | None = None,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
    normalize_pca_output: bool = False,
    cluster_target: np.ndarray | None = None,
    cluster_target_metric: str | None = None,
    cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> tuple[PCA | PcaPrefixTransformer, Any, np.ndarray]:
    """Fit PCA+UMAP once and return the model for future point transforms."""

    if pca_components is None:
        from visualization_pca_dimension_selection import (
            select_visualization_pca_dimension_for_data,
        )

        selection = select_visualization_pca_dimension_for_data(
            embeddings,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            spread=spread,
            densmap=densmap,
            normalize_pca_output=normalize_pca_output,
            cluster_target=cluster_target,
            cluster_target_metric=cluster_target_metric,
            cluster_target_weight=cluster_target_weight,
            seed=seed,
        )
        setattr(selection.umap, "_visualization_normalize_pca_output", normalize_pca_output)
        return (
            PcaPrefixTransformer(
                selection.pca,
                selection.selected_dimension,
            ),
            selection.umap,
            selection.selected_coordinates,
        )

    fitted = fit_normalized_pca_projection(
        embeddings,
        n_components=pca_components,
        seed=seed,
        name="embeddings",
    )
    target, target_metric, target_weight = _validate_cluster_target(
        cluster_target,
        cluster_target_metric,
        cluster_target_weight,
        n_samples=fitted.normalized_input.shape[0],
    )
    reducer = _make_umap_reducer(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
        random_state=seed,
        target_metric=target_metric,
        target_weight=target_weight,
    )
    pca_features = (
        fitted.normalized_prefix()
        if normalize_pca_output
        else fitted.projected
    )
    setattr(reducer, "_visualization_normalize_pca_output", normalize_pca_output)
    reduced = reducer.fit_transform(pca_features, y=target)
    return fitted.pca, reducer, reduced


def _validate_cluster_target(
    cluster_target: np.ndarray | None,
    cluster_target_metric: str | None,
    cluster_target_weight: float,
    *,
    n_samples: int,
) -> tuple[np.ndarray | None, str | None, float]:
    """Validate and normalize the optional weakly supervised UMAP target."""

    weight = float(cluster_target_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("cluster_target_weight must be between 0 and 1")
    if cluster_target is None or weight == 0.0:
        return None, None, 0.0
    if not cluster_target_metric:
        raise ValueError("cluster_target_metric is required when a target is used")

    target = np.asarray(cluster_target)
    if target.shape[0] != n_samples:
        raise ValueError("cluster_target must contain one value per embedding")
    if cluster_target_metric == "categorical":
        if target.ndim != 1:
            raise ValueError("categorical cluster_target must be one-dimensional")
        target = target.astype(np.int32, copy=False)
    else:
        if target.ndim not in {1, 2}:
            raise ValueError("continuous cluster_target must be one- or two-dimensional")
        target = target.astype(np.float64, copy=False)
        if target.ndim == 1:
            target = target.reshape(-1, 1)
    if not np.all(np.isfinite(target)):
        raise ValueError("cluster_target must contain only finite values")

    unique_count = (
        np.unique(target, axis=0).shape[0]
        if target.ndim == 2
        else np.unique(target).size
    )
    if unique_count < 2:
        return None, None, 0.0
    return target, cluster_target_metric, weight


def _make_umap_reducer(
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
    random_state: int,
    target_metric: str | None,
    target_weight: float,
) -> Any:
    kwargs: dict[str, Any] = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "spread": spread,
        "densmap": densmap,
        "random_state": random_state,
    }
    if target_metric is not None and target_weight > 0.0:
        kwargs["target_metric"] = target_metric
        kwargs["target_weight"] = target_weight
    return _load_umap()(**kwargs)


def transform_projection(
    embeddings: np.ndarray,
    *,
    pca: PCA | PcaPrefixTransformer,
    reducer: Any,
) -> np.ndarray:
    """Project a new batch using a previously fitted PCA+UMAP model."""

    normalize_pca_output = bool(
        getattr(reducer, "_visualization_normalize_pca_output", True)
    )
    if normalize_pca_output:
        pca_features = transform_normalized_pca_projection(
            embeddings,
            pca,
            name="embeddings",
        )
    else:
        matrix = validate_embedding_matrix(
            embeddings,
            name="embeddings",
            expected_features=int(pca.n_features_in_),
        )
        pca_features = np.asarray(pca.transform(normalize(matrix, norm="l2")))
    reduced = np.asarray(reducer.transform(pca_features), dtype=np.float64)
    if reduced.ndim != 2 or reduced.shape[1] != 2:
        raise ValueError("UMAP transform must return two-dimensional coordinates")
    if not np.all(np.isfinite(reduced)):
        raise ValueError("UMAP transform returned non-finite coordinates")
    return reduced
