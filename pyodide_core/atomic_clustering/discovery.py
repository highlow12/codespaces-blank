"""Optional UMAP/HDBSCAN discovery boundary.

Pyodide does not ship these two native-extension packages in its standard
package set.  Imports therefore happen only when discovery is requested.
Tests and a future JavaScript implementation can inject a callable returning
the same small dictionary instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .types import DiscoveryDependencyError, DiscoveryOutput

DEFAULT_UMAP_COMPONENTS = 20
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3


def dependency_status() -> dict[str, Any]:
    """Report whether native UMAP/HDBSCAN discovery can run in this runtime."""

    status: dict[str, Any] = {"umap": False, "hdbscan": False, "available": False, "errors": {}}
    for name in ("umap", "hdbscan"):
        try:
            __import__(name)
            status[name] = True
        except Exception as error:  # pragma: no cover - depends on runtime
            status["errors"][name] = f"{type(error).__name__}: {error}"
    status["available"] = bool(status["umap"] and status["hdbscan"])
    status["runtime_note"] = (
        "Native discovery is available."
        if status["available"]
        else "Install Pyodide-compatible UMAP/HDBSCAN wheels or inject discovery outputs."
    )
    return status


def _validate_output(raw: Mapping[str, Any], n_samples: int, expected_umap_components: int | None) -> DiscoveryOutput:
    required = {"umap_features", "leaf_labels", "memberships"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"discovery output is missing: {', '.join(missing)}")
    umap_features = np.asarray(raw["umap_features"], dtype=np.float64)
    if umap_features.ndim != 2 or umap_features.shape[0] != n_samples:
        raise ValueError("umap_features must be a 2D array aligned with embeddings")
    if expected_umap_components is not None and umap_features.shape[1] != expected_umap_components:
        raise ValueError("umap_features has an unexpected component count")
    if not np.all(np.isfinite(umap_features)):
        raise ValueError("umap_features must contain only finite values")
    labels = np.asarray(raw["leaf_labels"], dtype=np.int64)
    if labels.shape != (n_samples,):
        raise ValueError("leaf_labels must contain one value per sample")
    if np.any(labels < -1):
        raise ValueError("leaf_labels may only contain -1 or non-negative labels")
    non_noise = labels[labels >= 0]
    cluster_count = 0 if not len(non_noise) else int(non_noise.max()) + 1
    if cluster_count and not np.array_equal(np.unique(non_noise), np.arange(cluster_count)):
        raise ValueError("non-noise leaf_labels must be contiguous from zero")
    memberships = np.asarray(raw["memberships"], dtype=np.float64)
    if memberships.shape != (n_samples, cluster_count):
        raise ValueError(f"memberships must have shape ({n_samples}, {cluster_count})")
    if not np.all(np.isfinite(memberships)) or np.any(memberships < -1e-12):
        raise ValueError("memberships must be finite and non-negative")
    probabilities = np.asarray(raw.get("probabilities", np.zeros(n_samples)), dtype=np.float64)
    outlier_scores = np.asarray(raw.get("outlier_scores", np.zeros(n_samples)), dtype=np.float64)
    for name, values in (("probabilities", probabilities), ("outlier_scores", outlier_scores)):
        if values.shape != (n_samples,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain one finite value per sample")
    return DiscoveryOutput(
        umap_features=umap_features,
        leaf_labels=labels,
        memberships=np.clip(memberships, 0.0, 1.0),
        probabilities=np.clip(probabilities, 0.0, 1.0),
        outlier_scores=np.clip(outlier_scores, 0.0, 1.0),
        configuration=dict(raw.get("configuration", {})),
    )


def _native_discovery(pca_features: np.ndarray, config: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        from umap import UMAP
        import hdbscan
    except Exception as error:  # pragma: no cover - environment dependent
        status = dependency_status()
        raise DiscoveryDependencyError(
            "UMAP/HDBSCAN discovery is unavailable in this runtime. "
            f"{status['runtime_note']} Details: {status['errors']}"
        ) from error

    n_samples = len(pca_features)
    umap_components = int(config.get("umap_components", DEFAULT_UMAP_COMPONENTS))
    n_neighbors = min(int(config.get("umap_n_neighbors", DEFAULT_UMAP_N_NEIGHBORS)), n_samples - 1)
    min_cluster_size = int(config.get("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE))
    min_samples = int(config.get("min_samples", DEFAULT_MIN_SAMPLES))
    seed = int(config.get("seed", 42))
    reduced = np.asarray(UMAP(n_components=umap_components, n_neighbors=n_neighbors, init="random", random_state=seed, n_jobs=1).fit_transform(pca_features), dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean", cluster_selection_method="leaf", prediction_data=True).fit(reduced)
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    count = 0 if not np.any(labels >= 0) else int(labels.max()) + 1
    if count:
        memberships = np.asarray(hdbscan.all_points_membership_vectors(clusterer), dtype=np.float64)
    else:
        memberships = np.zeros((n_samples, 0), dtype=np.float64)
    return {
        "umap_features": reduced,
        "leaf_labels": labels,
        "memberships": memberships,
        "probabilities": np.asarray(clusterer.probabilities_, dtype=np.float64),
        "outlier_scores": np.asarray(clusterer.outlier_scores_, dtype=np.float64),
        "configuration": {"runtime": "native", "umap_components": umap_components, "umap_n_neighbors": n_neighbors, "min_cluster_size": min_cluster_size, "min_samples": min_samples, "seed": seed},
    }


def run_discovery(
    pca_features: Any,
    *,
    config: Mapping[str, Any] | None = None,
    runner: Callable[[np.ndarray, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> DiscoveryOutput:
    """Run native discovery or validate injected UMAP/HDBSCAN outputs."""

    features = np.asarray(pca_features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 3:
        raise ValueError("discovery requires at least 3 PCA rows")
    options = dict(config or {})
    # Injected runners may be implemented in JavaScript and are free to pick
    # a different coordinate width.  Native discovery uses the configured
    # width and validates it below.
    expected = options.get("umap_components")
    if expected is None and runner is None:
        expected = DEFAULT_UMAP_COMPONENTS
    raw = (runner or _native_discovery)(features, options)
    return _validate_output(raw, len(features), int(expected) if expected is not None else None)
