"""Single JSON/JavaScript-friendly entry point for browser clustering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .discovery import run_discovery
from .hierarchy import build_hierarchy
from .pca import fit_pca
from .serialization import to_jsonable


def _pca_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Accept both compact package names and existing pipeline names."""

    aliases = {
        "pca_components": "components",
        "pca_max_components": "max_components",
        "pca_min_components": "min_components",
        "pca_component_step": "component_step",
        "pca_k_values": "k_values",
        "minimum_preservation_gain": "minimum_preservation_gain",
    }
    return {aliases.get(str(key), str(key)): value for key, value in options.items()}


def cluster_documents(
    embeddings: Any,
    *,
    ids: Any | None = None,
    config: Mapping[str, Any] | None = None,
    discovery_runner: Callable[[np.ndarray, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run PCA(auto) -> UMAP -> HDBSCAN -> bottom-up hierarchy.

    ``discovery_runner`` is the Pyodide migration seam: a JS-side UMAP or
    HDBSCAN implementation can return arrays without installing those Python
    packages.  The returned value contains only dictionaries, lists, numbers,
    and strings and is safe to pass through ``toJs()``.
    """

    options = dict(config or {})
    pca_options = _pca_options(options.get("pca", {}))
    discovery_options = dict(options.get("discovery", {}))
    matrix = np.asarray(embeddings, dtype=np.float64)
    selection = fit_pca(matrix, **pca_options)
    discovery = run_discovery(selection.features, config=discovery_options, runner=discovery_runner)
    hierarchy = build_hierarchy(
        selection.features,
        discovery.leaf_labels,
        discovery.memberships,
        probabilities=discovery.probabilities,
        outlier_scores=discovery.outlier_scores,
    )
    if ids is None:
        row_ids = list(range(len(matrix)))
    else:
        row_ids = list(ids)
        if len(row_ids) != len(matrix):
            raise ValueError("ids must contain one value per embedding")
    result = {
        "schema_version": 1,
        "pipeline": "pca_umap_hdbscan_bottom_up",
        "ids": row_ids,
        "pca": {**selection.to_dict(), "features": selection.features},
        "discovery": {
            "umap_features": discovery.umap_features,
            "leaf_labels": discovery.leaf_labels,
            "probabilities": discovery.probabilities,
            "outlier_scores": discovery.outlier_scores,
            "memberships": discovery.memberships,
            "cluster_count": int(discovery.memberships.shape[1]),
            "configuration": discovery.configuration,
        },
        "hierarchy": hierarchy,
    }
    return to_jsonable(result)
