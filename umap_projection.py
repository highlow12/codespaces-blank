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

from embedding_data import load_embeddings_from_json
from pca_projection import (
    fit_normalized_pca_projection,
    transform_normalized_pca_projection,
)


DEFAULT_VISUAL_PCA_COMPONENTS = 64
DEFAULT_CLUSTER_TARGET_WEIGHT = 0.01


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
    pca_components: int = DEFAULT_VISUAL_PCA_COMPONENTS,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
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
        cluster_target=cluster_target,
        cluster_target_metric=cluster_target_metric,
        cluster_target_weight=cluster_target_weight,
    )
    return coordinates


def fit_projection_model(
    embeddings: np.ndarray,
    *,
    seed: int,
    pca_components: int = DEFAULT_VISUAL_PCA_COMPONENTS,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
    cluster_target: np.ndarray | None = None,
    cluster_target_metric: str | None = None,
    cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> tuple[PCA, Any, np.ndarray]:
    """Fit PCA+UMAP once and return the model for future point transforms."""

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
    reduced = reducer.fit_transform(fitted.normalized_prefix(), y=target)
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
    pca: PCA,
    reducer: Any,
) -> np.ndarray:
    """Project a new batch using a previously fitted PCA+UMAP model."""

    pca_features = transform_normalized_pca_projection(
        embeddings,
        pca,
        name="embeddings",
    )
    reduced = np.asarray(reducer.transform(pca_features), dtype=np.float64)
    if reduced.ndim != 2 or reduced.shape[1] != 2:
        raise ValueError("UMAP transform must return two-dimensional coordinates")
    if not np.all(np.isfinite(reduced)):
        raise ValueError("UMAP transform returned non-finite coordinates")
    return reduced


