from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score

from clustering_types import PipelineResult
from fcm_hierarchy import (
    DEFAULT_CLUSTERING_PCA_COMPONENTS,
    fit_clustering_pca,
    fuzzy_silhouette_proxy,
    spherical_fcm,
    xie_beni_index,
)


PIPELINE_NAMES = (
    "2_auto_pca_fcm",
    "2_pca256_fcm",
)


def evaluate_clustering(
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    X_for_silhouette: np.ndarray,
) -> dict[str, Any]:
    labels = np.unique(y_pred)
    non_noise = y_pred != -1
    cluster_count = int(np.sum(labels != -1))
    noise_ratio = float(np.mean(~non_noise))

    nmi = np.nan
    ari = np.nan
    if y_true is not None:
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))

    fragmentation = np.nan
    if y_true is not None:
        fragmentation_scores: list[float] = []
        for true_label in np.unique(y_true):
            mask = y_true == true_label
            assigned_clusters = y_pred[mask]
            assigned_clusters = assigned_clusters[assigned_clusters != -1]
            if assigned_clusters.size == 0:
                continue
            fragmentation_scores.append(
                float(pd.Series(assigned_clusters).nunique())
            )
        if fragmentation_scores:
            fragmentation = float(np.mean(fragmentation_scores))

    metrics: dict[str, Any] = {
        "nmi": nmi,
        "ari": ari,
        "clusters": cluster_count,
        "noise_ratio": noise_ratio,
        "tag_fragmentation": fragmentation,
    }

    if cluster_count >= 2 and np.sum(non_noise) >= 3:
        try:
            metrics["silhouette"] = float(
                silhouette_score(X_for_silhouette[non_noise], y_pred[non_noise])
            )
        except Exception:
            metrics["silhouette"] = np.nan
    else:
        metrics["silhouette"] = np.nan
    return metrics


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
        **evaluate_clustering(y, result.labels, Xn),
        "xie_beni": xie_beni_index(Xn, result),
        "fuzzy_silhouette": fuzzy_silhouette_proxy(Xn, result),
        "iterations": result.iterations,
    }
    return PipelineResult(
        metrics=metrics,
        labels=result.labels,
        memberships=result.memberships,
    )


def run_pipeline_by_name(
    pipeline_name: str,
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
) -> PipelineResult:
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
