from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clustering_types import PipelineResult
from extract_clustering_metrics import extract_metrics_from_frame
from fcm_hierarchy import (
    DEFAULT_CLUSTERING_PCA_COMPONENTS,
    fit_clustering_pca,
    spherical_fcm,
)
from hdbscan_membership_comparison import (
    DEFAULT_MIN_CLUSTER_SIZE as DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES as DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_UMAP_COMPONENTS,
    DEFAULT_UMAP_N_NEIGHBORS,
    fit_hdbscan_membership_comparison,
)
from hdbscan_bottom_up import build_hdbscan_hierarchy


DEFAULT_PIPELINE_NAME = "pca_umap_hdbscan"
# The first entry is the user-facing default.  The FCM names are retained as
# explicit compatibility/benchmark paths.
PIPELINE_NAMES = (
    DEFAULT_PIPELINE_NAME,
    "2_auto_pca_fcm",
    "2_pca256_fcm",
)


def evaluate_clustering(
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    X_for_silhouette: np.ndarray,
) -> dict[str, Any]:
    assignments = pd.DataFrame({"cluster": np.asarray(y_pred)})
    if y_true is not None:
        assignments["class"] = np.asarray(y_true)
    return extract_metrics_from_frame(
        assignments,
        source="<pipeline>",
        features=X_for_silhouette,
        feature_source="pipeline_features",
    )


def run_pipeline_2(
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
    *,
    pca_components: int | None = None,
    pipeline_name: str = "2_auto_pca_fcm",
) -> PipelineResult:
    start = time.perf_counter()
    Xn, _pca, pca_selection = fit_clustering_pca(
        X,
        n_components=pca_components,
        seed=42,
    )
    result = spherical_fcm(Xn, n_clusters=n_clusters, seed=42)
    elapsed = time.perf_counter() - start
    assignments = pd.DataFrame({"cluster": result.labels})
    if y is not None:
        assignments["class"] = np.asarray(y)
    for index in range(result.memberships.shape[1]):
        assignments[f"membership_{index}"] = result.memberships[:, index]
    extracted_metrics = extract_metrics_from_frame(
        assignments,
        source=pipeline_name,
        features=Xn,
        feature_source="post_pca_features",
        centers=result.centers,
    )
    metrics = {
        "pipeline": pipeline_name,
        "pca_components_requested": (
            "auto" if pca_components is None else int(pca_components)
        ),
        "pca_components": int(Xn.shape[1]),
        "pca_selection": (
            None if pca_selection is None else pca_selection.to_dict()
        ),
        "runtime_sec": elapsed,
        **extracted_metrics,
        "iterations": result.iterations,
    }
    return PipelineResult(
        metrics=metrics,
        labels=result.labels,
        memberships=result.memberships,
    )


def run_pipeline_pca_umap_hdbscan(
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int | None = None,
    *,
    pca_components: int | None = None,
    pca_max_components: int = 512,
    pca_min_components: int = 32,
    pca_component_step: int = 32,
    umap_components: int = DEFAULT_UMAP_COMPONENTS,
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    seed: int = 42,
) -> PipelineResult:
    """Run the default ``PCA -> UMAP -> HDBSCAN`` discovery path.

    ``n_clusters`` is accepted for API compatibility with the old flat FCM
    runners.  HDBSCAN determines the number of clusters from the data, so it
    is intentionally not used as a forced K.
    """

    del n_clusters
    start = time.perf_counter()
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("PCA + UMAP + HDBSCAN requires at least 3 samples")
    # The comparison API also computes independent PCA-space affinities.  It
    # is useful here because it gives the default route stable soft
    # assignments, while the actual hard discovery labels come from UMAP.
    effective_neighbor_count = min(DEFAULT_UMAP_N_NEIGHBORS, values.shape[0] - 1)
    comparison = fit_hdbscan_membership_comparison(
        values,
        pca_components=pca_components,
        pca_max_components=pca_max_components,
        pca_min_components=pca_min_components,
        pca_component_step=pca_component_step,
        umap_components=umap_components,
        umap_n_neighbors=umap_n_neighbors,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        neighbor_count=effective_neighbor_count,
        seed=seed,
    )
    assignments = pd.DataFrame({"cluster": comparison.leaf_labels})
    if y is not None:
        assignments["class"] = np.asarray(y)
    memberships = comparison.native_memberships
    for index in range(memberships.shape[1]):
        assignments[f"membership_{index}"] = memberships[:, index]
    extracted_metrics = extract_metrics_from_frame(
        assignments,
        source=DEFAULT_PIPELINE_NAME,
        # HDBSCAN fits on UMAP coordinates, so internal metrics should use the
        # same space rather than the pre-discovery PCA representation.
        features=comparison.umap_features,
        feature_source="umap_features",
    )
    elapsed = time.perf_counter() - start
    metrics: dict[str, Any] = {
        "pipeline": DEFAULT_PIPELINE_NAME,
        "pca_components_requested": (
            "auto" if pca_components is None else int(pca_components)
        ),
        "pca_components": int(comparison.pca_features.shape[1]),
        "pca_selection": comparison.pca_selection.to_dict(),
        "umap_components": int(comparison.umap_features.shape[1]),
        "umap_n_neighbors": int(comparison.configuration["umap_n_neighbors"]),
        "hdbscan_min_cluster_size": int(min_cluster_size),
        "hdbscan_min_samples": int(min_samples),
        "runtime_sec": elapsed,
        "iterations": None,
        **extracted_metrics,
    }
    hierarchy = build_hdbscan_hierarchy(
        comparison.pca_features,
        comparison.leaf_labels,
        memberships,
        probabilities=getattr(comparison, "probabilities", None),
        outlier_scores=getattr(comparison, "outlier_scores", None),
    )
    return PipelineResult(
        metrics=metrics,
        labels=comparison.leaf_labels,
        memberships=memberships if memberships.shape[1] else None,
        hierarchy=hierarchy,
    )


def run_pipeline_by_name(
    pipeline_name: str,
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
) -> PipelineResult:
    if pipeline_name == DEFAULT_PIPELINE_NAME:
        return run_pipeline_pca_umap_hdbscan(X, y, n_clusters)
    if pipeline_name == "2_auto_pca_fcm":
        return run_pipeline_2(
            X,
            y,
            n_clusters,
            pca_components=None,
            pipeline_name=pipeline_name,
        )
    if pipeline_name == "2_pca256_fcm":
        return run_pipeline_2(
            X,
            y,
            n_clusters,
            pca_components=DEFAULT_CLUSTERING_PCA_COMPONENTS,
            pipeline_name=pipeline_name,
        )
    raise ValueError(f"Unknown pipeline: {pipeline_name}")


def run_selected_pipelines(
    pipeline_names: list[str],
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
) -> dict[str, PipelineResult]:
    unknown_pipelines = sorted(set(pipeline_names) - set(PIPELINE_NAMES))
    if unknown_pipelines:
        raise ValueError(f"Unknown pipelines: {unknown_pipelines}")

    return {
        pipeline_name: run_pipeline_by_name(pipeline_name, X, y, n_clusters)
        for pipeline_name in pipeline_names
    }


def choose_best_pipeline(frame: pd.DataFrame, has_ground_truth: bool) -> pd.Series:
    sortable = frame.copy()
    if has_ground_truth:
        sortable = sortable.sort_values(
            ["nmi", "ari", "tag_fragmentation", "silhouette"],
            ascending=[False, False, True, False],
            na_position="last",
        )
    else:
        sortable = sortable.sort_values(
            ["silhouette", "noise_ratio", "clusters", "runtime_sec"],
            ascending=[False, True, True, True],
            na_position="last",
        )
    return sortable.iloc[0]


def pipeline_to_filename(pipeline: str) -> str:
    return pipeline.replace("/", "_").replace(" ", "_")


def build_soft_assignments(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    memberships: np.ndarray,
) -> pd.DataFrame:
    """Combine hard labels and soft memberships in one assignment frame."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != len(metadata):
        raise ValueError("Labels must be a 1D array aligned with metadata")

    memberships = np.asarray(memberships, dtype=np.float64)
    if memberships.ndim != 2 or memberships.shape[0] != len(metadata):
        raise ValueError("Soft memberships must be a 2D array aligned with metadata")

    assignments = metadata.copy()
    assignments["cluster"] = labels
    for index in range(memberships.shape[1]):
        assignments[f"membership_{index}"] = memberships[:, index]
    membership_sums = memberships.sum(axis=1)
    if np.any(membership_sums < 1.0 - 1e-8):
        assignments["membership_noise"] = np.clip(1.0 - membership_sums, 0.0, 1.0)
    return assignments


def save_soft_assignments(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    memberships: np.ndarray,
    output_path: Path,
) -> None:
    assignments = build_soft_assignments(metadata, labels, memberships)
    assignments.to_csv(output_path, index=False)
