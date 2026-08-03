from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from clustering_types import PipelineResult
from fcm_hierarchy import (
    fuzzy_silhouette_proxy,
    pca_normalized_features,
    spherical_fcm,
    xie_beni_index,
)


PIPELINE_NAMES = (
    "1_raw_fcm",
    "2_pca64_fcm",
    "2b_pca64_hdbscan",
    "3_pca50_umap2_hdbscan",
    "4_umap8_hdbscan",
    "5_pca64_gmm",
    "6_pca64_hdbscan_cosine",
)


def build_compact_umap(
    *,
    n_components: int,
    seed: int,
    n_neighbors: int = 15,
    min_dist: float = 0.02,
    metric: str = "cosine",
    spread: float = 0.8,
    densmap: bool = True,
) -> Any:
    from umap import UMAP

    return UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
        random_state=seed,
    )


def compact_umap_presets() -> list[dict[str, Any]]:
    return [
        {"name": "dense", "n_neighbors": 8, "min_dist": 0.0, "spread": 0.7, "densmap": True},
        {"name": "compact", "n_neighbors": 12, "min_dist": 0.01, "spread": 0.8, "densmap": True},
        {"name": "balanced", "n_neighbors": 15, "min_dist": 0.02, "spread": 0.85, "densmap": True},
        {"name": "local", "n_neighbors": 20, "min_dist": 0.03, "spread": 0.9, "densmap": False},
    ]


def sort_candidate_metrics(
    candidates: list[tuple[dict[str, Any], np.ndarray]],
    has_ground_truth: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    def candidate_key(
        item: tuple[dict[str, Any], np.ndarray],
    ) -> tuple[float, float, float, float]:
        metrics = item[0]
        if has_ground_truth:
            fragmentation = metrics.get("tag_fragmentation", np.nan)
            fragmentation_key = (
                float(fragmentation) if pd.notna(fragmentation) else float("inf")
            )
            return (
                float(metrics.get("nmi", np.nan)),
                float(metrics.get("ari", np.nan)),
                -fragmentation_key,
                float(metrics.get("silhouette", np.nan)),
            )
        return (
            float(metrics.get("silhouette", np.nan)),
            -float(metrics.get("noise_ratio", np.nan)),
            float(metrics.get("clusters", np.nan)),
            -float(metrics.get("runtime_sec", np.nan)),
        )

    return max(candidates, key=candidate_key)


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


def run_compact_umap_sweep(
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    n_components: int,
    seed: int,
    pipeline_name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    Xn = normalize(X, norm="l2")
    candidates: list[tuple[dict[str, Any], np.ndarray]] = []
    for preset in compact_umap_presets():
        start = time.perf_counter()
        Xu = build_compact_umap(
            n_components=n_components,
            seed=seed,
            n_neighbors=int(preset["n_neighbors"]),
            min_dist=float(preset["min_dist"]),
            spread=float(preset["spread"]),
            densmap=bool(preset["densmap"]),
        ).fit_transform(Xn)
        labels = HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(Xu)
        elapsed = time.perf_counter() - start
        metrics = evaluate_clustering(y, labels, Xu)
        metrics.update(
            {
                "pipeline": pipeline_name,
                "runtime_sec": elapsed,
                "xie_beni": np.nan,
                "fuzzy_silhouette": np.nan,
                "iterations": np.nan,
                "umap_preset": preset["name"],
                "umap_n_neighbors": int(preset["n_neighbors"]),
                "umap_min_dist": float(preset["min_dist"]),
                "umap_spread": float(preset["spread"]),
                "umap_densmap": bool(preset["densmap"]),
            }
        )
        candidates.append((metrics, labels))

    return sort_candidate_metrics(candidates, has_ground_truth=y is not None)


def run_pipeline_1(
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
) -> PipelineResult:
    start = time.perf_counter()
    Xn = normalize(X, norm="l2")
    result = spherical_fcm(Xn, n_clusters=n_clusters, seed=42)
    elapsed = time.perf_counter() - start
    metrics = {
        "pipeline": "1_raw_fcm",
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


def run_pipeline_2(
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
) -> PipelineResult:
    start = time.perf_counter()
    Xn = pca_normalized_features(X, n_components=64, seed=42)
    result = spherical_fcm(Xn, n_clusters=n_clusters, seed=42)
    elapsed = time.perf_counter() - start
    metrics = {
        "pipeline": "2_pca64_fcm",
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


def run_pipeline_2b(X: np.ndarray, y: np.ndarray | None) -> PipelineResult:
    start = time.perf_counter()
    Xp = PCA(n_components=64, random_state=42).fit_transform(X)
    Xn = normalize(Xp, norm="l2")
    labels = HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(Xn)
    elapsed = time.perf_counter() - start
    metrics = evaluate_clustering(y, labels, Xn)
    metrics.update({"xie_beni": np.nan, "fuzzy_silhouette": np.nan})
    return PipelineResult(
        metrics={
            "pipeline": "2b_pca64_hdbscan",
            "runtime_sec": elapsed,
            **metrics,
        },
        labels=labels,
    )


def run_pipeline_3(X: np.ndarray, y: np.ndarray | None) -> PipelineResult:
    Xp = PCA(n_components=50, random_state=42).fit_transform(X)
    metrics, labels = run_compact_umap_sweep(
        Xp,
        y,
        n_components=2,
        seed=42,
        pipeline_name="3_pca50_umap2_hdbscan_sweep",
    )
    return PipelineResult(metrics=metrics, labels=labels)


def run_pipeline_4(X: np.ndarray, y: np.ndarray | None) -> PipelineResult:
    metrics, labels = run_compact_umap_sweep(
        X,
        y,
        n_components=8,
        seed=42,
        pipeline_name="4_umap8_hdbscan_sweep",
    )
    return PipelineResult(metrics=metrics, labels=labels)


def run_pipeline_5(
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
) -> PipelineResult:
    start = time.perf_counter()
    Xp = PCA(n_components=64, random_state=42).fit_transform(X)
    Xn = normalize(Xp, norm="l2")
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        n_init=5,
        max_iter=200,
        reg_covar=1e-6,
        random_state=42,
    )
    model.fit(Xn)
    memberships = model.predict_proba(Xn)
    labels = memberships.argmax(axis=1)
    elapsed = time.perf_counter() - start
    metrics = {
        "pipeline": "5_pca64_gmm",
        "runtime_sec": elapsed,
        **evaluate_clustering(y, labels, Xn),
        "xie_beni": np.nan,
        "fuzzy_silhouette": np.nan,
        "iterations": model.n_iter_,
    }
    return PipelineResult(metrics=metrics, labels=labels, memberships=memberships)


def run_pipeline_6(X: np.ndarray, y: np.ndarray | None) -> PipelineResult:
    start = time.perf_counter()
    Xp = PCA(n_components=64, random_state=42).fit_transform(X)
    Xn = normalize(Xp, norm="l2")
    # On L2-normalized vectors, Euclidean distance is equivalent to cosine distance.
    clusterer = HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(Xn)
    memberships = all_points_membership_vectors(clusterer)
    elapsed = time.perf_counter() - start
    metrics = evaluate_clustering(y, labels, Xn)
    metrics.update({"xie_beni": np.nan, "fuzzy_silhouette": np.nan})
    return PipelineResult(
        metrics={
            "pipeline": "6_pca64_hdbscan_cosine",
            "runtime_sec": elapsed,
            **metrics,
        },
        labels=labels,
        memberships=memberships,
    )


def run_pipeline_by_name(
    pipeline_name: str,
    X: np.ndarray,
    y: np.ndarray | None,
    n_clusters: int,
) -> PipelineResult:
    if pipeline_name == "1_raw_fcm":
        return run_pipeline_1(X, y, n_clusters)
    if pipeline_name == "2_pca64_fcm":
        return run_pipeline_2(X, y, n_clusters)
    if pipeline_name == "2b_pca64_hdbscan":
        return run_pipeline_2b(X, y)
    if pipeline_name == "3_pca50_umap2_hdbscan":
        return run_pipeline_3(X, y)
    if pipeline_name == "4_umap8_hdbscan":
        return run_pipeline_4(X, y)
    if pipeline_name == "5_pca64_gmm":
        return run_pipeline_5(X, y, n_clusters)
    if pipeline_name == "6_pca64_hdbscan_cosine":
        return run_pipeline_6(X, y)
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
