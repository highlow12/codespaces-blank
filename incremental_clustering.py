"""Incremental hierarchical clustering and fixed-coordinate visualization.

The initial batch is clustered with the existing recursive PCA + spherical FCM
implementation. Later batches update fuzzy cluster centers online. Full
memberships are periodically refreshed, and XB degradation or excessive
new-batch noise triggers complete re-clustering while visualization stays fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cluster_visualization import (
    DEFAULT_CLUSTER_TARGET_WEIGHT,
    DEFAULT_VISUAL_PCA_COMPONENTS,
    build_cluster_supervision,
    fit_projection_model,
    make_fixed_coordinate_plot,
    transform_projection,
)
from clustering_types import HierarchicalModel
from embedding_data import load_embeddings_from_json
from fcm_hierarchy import (
    DEFAULT_CLUSTERING_PCA_COMPONENTS,
    DEFAULT_FORCED_NOISE_RATIO,
    DEFAULT_MAX_MEMBERSHIP_GAP,
    DOCUMENT_TYPE_BOUNDARY,
    DOCUMENT_TYPE_CORE,
    DOCUMENT_TYPE_NOISE,
    classify_fcm_documents,
    conditional_memberships_from_projected,
    fcm_memberships_from_centers,
    fcm_noise_scores,
    forced_noise_mask,
    merge_forced_noise,
    run_hierarchical_pca_fcm,
    transform_pca_normalized_features,
)
from hierarchical_assignments import (
    build_hierarchical_assignments as _assignments_from_labels,
)


DEFAULT_NOISE_THRESHOLD = 0.05
DEFAULT_CENTER_UPDATES_BEFORE_MEMBERSHIP_REFRESH = 10
DEFAULT_MAX_XB_RELATIVE_DEGRADATION = 0.05


@dataclass
class IncrementalClusterState:
    """All fitted state needed to process another embedding batch."""

    embeddings: np.ndarray
    metadata: pd.DataFrame
    assignments: pd.DataFrame
    coordinates: np.ndarray
    hierarchy_model: HierarchicalModel
    tree: dict[str, Any]
    config: dict[str, Any]
    visual_pca: Any
    visual_reducer: Any
    center_statistics: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict
    )


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings must contain only finite values")
    return values


def _validate_metadata(metadata: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    if len(metadata) != expected_rows:
        raise ValueError("metadata must contain exactly one row per embedding")
    if "id" not in metadata.columns:
        raise ValueError("metadata must contain an 'id' column")
    frame = metadata.copy().reset_index(drop=True)
    ids = frame["id"].tolist()
    try:
        if len(set(ids)) != len(ids):
            raise ValueError("embedding IDs must be unique")
    except TypeError as error:
        raise ValueError("embedding IDs must be hashable scalar values") from error
    if any(pd.isna(value) for value in ids):
        raise ValueError("embedding IDs must not be missing")
    return frame


def _validate_noise_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("noise_threshold must be between 0 and 1")
    return threshold


def _path_for_cluster(parent_path: str, cluster_id: int) -> str:
    return f"{parent_path}/{cluster_id}" if parent_path else str(cluster_id)


def assign_to_hierarchy(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    hierarchy_model: HierarchicalModel,
    *,
    min_membership: float,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
    forced_noise_ratio: float = DEFAULT_FORCED_NOISE_RATIO,
    m: float = 2.0,
) -> tuple[pd.DataFrame, float]:
    """Assign a batch to fixed hierarchy centers and return its noise ratio."""

    values = _validate_embeddings(embeddings)
    frame = _validate_metadata(metadata, len(values))
    if not 0.0 <= min_membership <= 1.0:
        raise ValueError("min_membership must be between 0 and 1")
    if not 0.0 <= max_membership_gap <= 1.0:
        raise ValueError("max_membership_gap must be between 0 and 1")
    if not 0.0 <= forced_noise_ratio <= 1.0:
        raise ValueError("forced_noise_ratio must be between 0 and 1")
    projected = transform_pca_normalized_features(values, hierarchy_model.pca)
    conditional_memberships = conditional_memberships_from_projected(
        projected,
        hierarchy_model,
        m=m,
    )
    labels_by_level = np.full(
        (len(values), hierarchy_model.max_depth),
        -1,
        dtype=int,
    )
    is_noise = np.zeros(len(values), dtype=bool)
    document_types = np.full(len(values), DOCUMENT_TYPE_CORE, dtype=object)
    noise_scores = np.zeros(len(values), dtype=np.float64)
    boundary_level = np.full(len(values), -1, dtype=int)
    noise_level = np.full(len(values), -1, dtype=int)
    max_cluster_count = max(
        (
            int(node_model.centers.shape[0])
            for node_model in hierarchy_model.nodes.values()
        ),
        default=1,
    )
    soft_memberships_by_level = [
        np.full(
            (len(values), max_cluster_count),
            np.nan,
            dtype=np.float64,
        )
        for _ in range(hierarchy_model.max_depth)
    ]

    if hierarchy_model.fallback_single_cluster:
        labels_by_level[:, 0] = 0
        soft_memberships_by_level[0][:, 0] = 1.0
    else:
        active: dict[str, np.ndarray] = {
            "": np.arange(len(values), dtype=int)
        }
        for depth in range(hierarchy_model.max_depth):
            next_active: dict[str, np.ndarray] = {}
            for parent_path, indices in active.items():
                node_model = hierarchy_model.nodes.get(parent_path)
                if node_model is None:
                    continue

                memberships, distances = fcm_memberships_from_centers(
                    projected[indices],
                    node_model.centers,
                    m=m,
                )
                local_labels = memberships.argmax(axis=1)
                row_indices = np.arange(len(indices))
                assigned_distances = distances[row_indices, local_labels]
                thresholds = node_model.distance_thresholds[local_labels]
                local_document_types = classify_fcm_documents(
                    memberships,
                    assigned_distances,
                    thresholds,
                    min_membership=min_membership,
                    max_membership_gap=max_membership_gap,
                )
                local_noise_scores = fcm_noise_scores(
                    memberships,
                    assigned_distances,
                    local_labels,
                )
                noise_scores[indices] = np.maximum(
                    noise_scores[indices],
                    local_noise_scores,
                )
                local_noise = local_document_types == DOCUMENT_TYPE_NOISE
                local_boundary = local_document_types == DOCUMENT_TYPE_BOUNDARY

                if np.any(local_boundary):
                    boundary_indices = indices[local_boundary]
                    first_boundary = (
                        document_types[boundary_indices] == DOCUMENT_TYPE_CORE
                    )
                    boundary_level[boundary_indices[first_boundary]] = depth + 1
                    document_types[boundary_indices] = DOCUMENT_TYPE_BOUNDARY

                if np.any(local_noise):
                    noise_indices = indices[local_noise]
                    is_noise[noise_indices] = True
                    document_types[noise_indices] = DOCUMENT_TYPE_NOISE
                    boundary_level[noise_indices] = -1
                    noise_level[noise_indices] = depth + 1

                valid_indices = indices[~local_noise]
                valid_labels = local_labels[~local_noise]
                if valid_indices.size == 0:
                    continue
                labels_by_level[valid_indices, depth] = valid_labels
                soft_memberships_by_level[depth][
                    valid_indices, : memberships.shape[1]
                ] = memberships[~local_noise]

                for cluster_id in range(node_model.centers.shape[0]):
                    child_indices = valid_indices[valid_labels == cluster_id]
                    if child_indices.size:
                        next_active[_path_for_cluster(parent_path, cluster_id)] = (
                            child_indices
                        )
            active = next_active

    is_natural_noise = is_noise.copy()
    (
        is_noise,
        is_forced_noise,
        _forced_only,
        document_types,
        noise_level,
    ) = merge_forced_noise(
        is_natural_noise,
        noise_scores,
        frame["id"].to_numpy(),
        document_types,
        noise_level,
        forced_noise_ratio=forced_noise_ratio,
    )

    assignments = _assignments_from_labels(
        frame,
        labels_by_level,
        is_noise,
        is_natural_noise,
        is_forced_noise,
        document_types,
        noise_scores,
        boundary_level,
        noise_level,
        soft_memberships_by_level,
        conditional_memberships,
    )
    return assignments, float(np.mean(is_noise))


def _center_statistics_for_batch(
    embeddings: np.ndarray,
    hierarchy_model: HierarchicalModel,
    *,
    min_membership: float,
    m: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Collect fuzzy weighted sums for every hierarchy node reached by a batch."""

    values = _validate_embeddings(embeddings)
    projected = transform_pca_normalized_features(values, hierarchy_model.pca)
    statistics: dict[str, dict[str, np.ndarray]] = {}
    if hierarchy_model.fallback_single_cluster:
        return statistics

    active: dict[str, np.ndarray] = {"": np.arange(len(values), dtype=int)}
    for _depth in range(hierarchy_model.max_depth):
        next_active: dict[str, np.ndarray] = {}
        for parent_path, indices in active.items():
            node_model = hierarchy_model.nodes.get(parent_path)
            if node_model is None or indices.size == 0:
                continue

            memberships, distances = fcm_memberships_from_centers(
                projected[indices],
                node_model.centers,
                m=m,
            )
            weights = memberships**m
            statistics[parent_path] = {
                "weighted_sum": weights.T @ projected[indices],
                "weight": weights.sum(axis=0),
            }

            local_labels = memberships.argmax(axis=1)
            row_indices = np.arange(len(indices))
            assigned_distances = distances[row_indices, local_labels]
            local_noise = (
                memberships.max(axis=1) < min_membership
            ) | (
                assigned_distances
                > node_model.distance_thresholds[local_labels]
            )
            valid_indices = indices[~local_noise]
            valid_labels = local_labels[~local_noise]
            for cluster_id in range(node_model.centers.shape[0]):
                child_indices = valid_indices[valid_labels == cluster_id]
                if child_indices.size:
                    next_active[_path_for_cluster(parent_path, cluster_id)] = (
                        child_indices
                    )
        active = next_active
    return statistics


def _build_center_statistics(
    embeddings: np.ndarray,
    hierarchy_model: HierarchicalModel,
    *,
    min_membership: float,
    m: float,
) -> dict[str, dict[str, np.ndarray]]:
    return _center_statistics_for_batch(
        embeddings,
        hierarchy_model,
        min_membership=min_membership,
        m=m,
    )


def _update_hierarchy_centers(
    hierarchy_model: HierarchicalModel,
    center_statistics: dict[str, dict[str, np.ndarray]],
    embeddings: np.ndarray,
    *,
    min_membership: float,
    m: float,
) -> tuple[HierarchicalModel, dict[str, dict[str, np.ndarray]], int]:
    """Update node centers from cumulative online fuzzy sufficient statistics."""

    updated_model = copy.deepcopy(hierarchy_model)
    accumulated = {
        path: {
            "weighted_sum": values["weighted_sum"].copy(),
            "weight": values["weight"].copy(),
        }
        for path, values in center_statistics.items()
    }
    batch_statistics = _center_statistics_for_batch(
        embeddings,
        hierarchy_model,
        min_membership=min_membership,
        m=m,
    )
    updated_node_count = 0
    for path, batch_values in batch_statistics.items():
        node_model = updated_model.nodes.get(path)
        if node_model is None:
            continue
        if path not in accumulated:
            accumulated[path] = {
                "weighted_sum": np.zeros_like(batch_values["weighted_sum"]),
                "weight": np.zeros_like(batch_values["weight"]),
            }
        stored = accumulated[path]
        if (
            stored["weighted_sum"].shape != batch_values["weighted_sum"].shape
            or stored["weight"].shape != batch_values["weight"].shape
        ):
            raise ValueError(f"Center statistics do not match hierarchy node: {path}")
        stored["weighted_sum"] += batch_values["weighted_sum"]
        stored["weight"] += batch_values["weight"]
        raw_centers = np.divide(
            stored["weighted_sum"],
            stored["weight"][:, None],
            out=node_model.centers.copy(),
            where=stored["weight"][:, None] > 1e-12,
        )
        center_norms = np.linalg.norm(raw_centers, axis=1, keepdims=True)
        node_model.centers = np.divide(
            raw_centers,
            center_norms,
            out=node_model.centers.copy(),
            where=center_norms > 1e-12,
        )
        updated_node_count += 1
    return updated_model, accumulated, updated_node_count


def _refresh_distance_thresholds(
    embeddings: np.ndarray,
    hierarchy_model: HierarchicalModel,
    *,
    min_membership: float,
    distance_z: float,
    m: float,
) -> HierarchicalModel:
    """Recompute node distance cutoffs after centers have moved."""

    updated_model = copy.deepcopy(hierarchy_model)
    projected = transform_pca_normalized_features(
        _validate_embeddings(embeddings),
        updated_model.pca,
    )
    if updated_model.fallback_single_cluster:
        return updated_model

    active: dict[str, np.ndarray] = {"": np.arange(len(projected), dtype=int)}
    for _depth in range(updated_model.max_depth):
        next_active: dict[str, np.ndarray] = {}
        for parent_path, indices in active.items():
            node_model = updated_model.nodes.get(parent_path)
            if node_model is None or indices.size == 0:
                continue
            memberships, distances = fcm_memberships_from_centers(
                projected[indices],
                node_model.centers,
                m=m,
            )
            local_labels = memberships.argmax(axis=1)
            thresholds = np.full(node_model.centers.shape[0], np.inf)
            for cluster_id in range(node_model.centers.shape[0]):
                cluster_distances = distances[
                    local_labels == cluster_id,
                    cluster_id,
                ]
                if cluster_distances.size < 4:
                    continue
                median = float(np.median(cluster_distances))
                mad = float(np.median(np.abs(cluster_distances - median)))
                if mad > 1e-12:
                    thresholds[cluster_id] = (
                        median + distance_z * 1.4826 * mad
                    )
            node_model.distance_thresholds = thresholds

            rows = np.arange(len(indices))
            local_noise = (
                memberships.max(axis=1) < min_membership
            ) | (distances[rows, local_labels] > thresholds[local_labels])
            valid_indices = indices[~local_noise]
            valid_labels = local_labels[~local_noise]
            for cluster_id in range(node_model.centers.shape[0]):
                child_indices = valid_indices[valid_labels == cluster_id]
                if child_indices.size:
                    next_active[_path_for_cluster(parent_path, cluster_id)] = (
                        child_indices
                    )
        active = next_active
    return updated_model


def hierarchy_xie_beni_index(
    embeddings: np.ndarray,
    hierarchy_model: HierarchicalModel,
    *,
    min_membership: float,
    m: float,
) -> float:
    """Return the sample-weighted XB index across fitted hierarchy nodes."""

    projected = transform_pca_normalized_features(
        _validate_embeddings(embeddings),
        hierarchy_model.pca,
    )
    if hierarchy_model.fallback_single_cluster:
        return float("nan")

    weighted_xb_sum = 0.0
    evaluated_samples = 0
    active: dict[str, np.ndarray] = {"": np.arange(len(projected), dtype=int)}
    for _depth in range(hierarchy_model.max_depth):
        next_active: dict[str, np.ndarray] = {}
        for parent_path, indices in active.items():
            node_model = hierarchy_model.nodes.get(parent_path)
            if node_model is None or indices.size == 0:
                continue
            memberships, distances = fcm_memberships_from_centers(
                projected[indices],
                node_model.centers,
                m=m,
            )
            center_differences = (
                node_model.centers[:, None, :] - node_model.centers[None, :, :]
            )
            center_distances_squared = np.sum(center_differences**2, axis=2)
            np.fill_diagonal(center_distances_squared, np.inf)
            minimum_separation_squared = float(
                np.min(center_distances_squared)
            )
            if np.isfinite(minimum_separation_squared):
                numerator = float(np.sum((memberships**m) * (distances**2)))
                node_xb = numerator / max(
                    len(indices) * minimum_separation_squared,
                    1e-12,
                )
                weighted_xb_sum += node_xb * len(indices)
                evaluated_samples += len(indices)

            local_labels = memberships.argmax(axis=1)
            rows = np.arange(len(indices))
            local_noise = (
                memberships.max(axis=1) < min_membership
            ) | (
                distances[rows, local_labels]
                > node_model.distance_thresholds[local_labels]
            )
            valid_indices = indices[~local_noise]
            valid_labels = local_labels[~local_noise]
            for cluster_id in range(node_model.centers.shape[0]):
                child_indices = valid_indices[valid_labels == cluster_id]
                if child_indices.size:
                    next_active[_path_for_cluster(parent_path, cluster_id)] = (
                        child_indices
                    )
        active = next_active

    if evaluated_samples == 0:
        return float("nan")
    return float(weighted_xb_sum / evaluated_samples)


def _cluster_config(
    *,
    max_depth: int,
    min_node_size: int,
    min_child_size: int,
    min_clusters: int,
    max_clusters: int,
    min_membership: float,
    max_membership_gap: float,
    forced_noise_ratio: float,
    distance_z: float,
    selection_method: str,
    min_xb_relative_improvement: float,
    xb_worsening_patience: int,
    min_split_silhouette: float,
    pca_components: int,
    seed: int,
    noise_threshold: float,
    visual_pca_components: int,
    visual_cluster_target_weight: float,
    visual_n_neighbors: int,
    visual_min_dist: float,
    visual_metric: str,
    visual_spread: float,
    visual_densmap: bool,
    center_updates_before_membership_refresh: int,
    max_xb_relative_degradation: float,
) -> dict[str, Any]:
    if center_updates_before_membership_refresh < 1:
        raise ValueError(
            "center_updates_before_membership_refresh must be at least 1"
        )
    if (
        not np.isfinite(max_xb_relative_degradation)
        or max_xb_relative_degradation < 0.0
    ):
        raise ValueError("max_xb_relative_degradation must be non-negative")
    return {
        "max_depth": int(max_depth),
        "min_node_size": int(min_node_size),
        "min_child_size": int(min_child_size),
        "min_clusters": int(min_clusters),
        "max_clusters": int(max_clusters),
        "min_membership": float(min_membership),
        "max_membership_gap": float(max_membership_gap),
        "forced_noise_ratio": float(forced_noise_ratio),
        "distance_z": float(distance_z),
        "selection_method": selection_method,
        "min_xb_relative_improvement": float(min_xb_relative_improvement),
        "xb_worsening_patience": int(xb_worsening_patience),
        "min_split_silhouette": float(min_split_silhouette),
        "pca_components": int(pca_components),
        "seed": int(seed),
        "m": 2.0,
        "noise_threshold": _validate_noise_threshold(noise_threshold),
        "visual_pca_components": int(visual_pca_components),
        "visual_cluster_target_weight": float(visual_cluster_target_weight),
        "visual_n_neighbors": int(visual_n_neighbors),
        "visual_min_dist": float(visual_min_dist),
        "visual_metric": visual_metric,
        "visual_spread": float(visual_spread),
        "visual_densmap": bool(visual_densmap),
        "update_count": 0,
        "center_updates_before_membership_refresh": int(
            center_updates_before_membership_refresh
        ),
        "max_xb_relative_degradation": float(max_xb_relative_degradation),
        "center_updates_since_membership_refresh": 0,
        "membership_refreshes_since_recluster": 0,
        "total_center_updates": 0,
        "total_membership_refreshes": 0,
        "total_reclusters": 0,
        "recluster_trigger_policy": "xb_and_noise_v2",
    }


def _fuzzy_parameters(config: dict[str, Any]) -> dict[str, float]:
    return {
        "min_membership": float(config["min_membership"]),
        "m": float(config["m"]),
    }


def _assignment_parameters(
    config: dict[str, Any],
    *,
    forced_noise_ratio: float | None = None,
) -> dict[str, float]:
    return {
        **_fuzzy_parameters(config),
        "max_membership_gap": float(
            config.get("max_membership_gap", DEFAULT_MAX_MEMBERSHIP_GAP)
        ),
        "forced_noise_ratio": (
            float(config.get("forced_noise_ratio", DEFAULT_FORCED_NOISE_RATIO))
            if forced_noise_ratio is None
            else float(forced_noise_ratio)
        ),
    }


def _hierarchy_fit_parameters(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_depth": int(config["max_depth"]),
        "min_node_size": int(config["min_node_size"]),
        "min_child_size": int(config["min_child_size"]),
        "min_clusters": int(config["min_clusters"]),
        "max_clusters": int(config["max_clusters"]),
        "distance_z": float(config["distance_z"]),
        "selection_method": str(config["selection_method"]),
        "min_xb_relative_improvement": float(
            config.get("min_xb_relative_improvement", 0.05)
        ),
        "xb_worsening_patience": int(config.get("xb_worsening_patience", 2)),
        "min_split_silhouette": float(config["min_split_silhouette"]),
        "pca_components": int(config["pca_components"]),
        "seed": int(config["seed"]),
        "min_membership": float(config["min_membership"]),
        "max_membership_gap": float(
            config.get("max_membership_gap", DEFAULT_MAX_MEMBERSHIP_GAP)
        ),
        "forced_noise_ratio": float(
            config.get("forced_noise_ratio", DEFAULT_FORCED_NOISE_RATIO)
        ),
    }


def _visualization_fit_parameters(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(config["seed"]),
        "pca_components": int(config["visual_pca_components"]),
        "n_neighbors": int(config["visual_n_neighbors"]),
        "min_dist": float(config["visual_min_dist"]),
        "metric": str(config["visual_metric"]),
        "spread": float(config["visual_spread"]),
        "densmap": bool(config["visual_densmap"]),
        "cluster_target_weight": float(config["visual_cluster_target_weight"]),
    }


def _initialize_update_config(config: dict[str, Any]) -> dict[str, Any]:
    initialized = dict(config)
    defaults = {
        "center_updates_before_membership_refresh": (
            DEFAULT_CENTER_UPDATES_BEFORE_MEMBERSHIP_REFRESH
        ),
        "max_xb_relative_degradation": DEFAULT_MAX_XB_RELATIVE_DEGRADATION,
        "center_updates_since_membership_refresh": 0,
        "membership_refreshes_since_recluster": 0,
        "total_center_updates": 0,
        "total_membership_refreshes": 0,
        "total_reclusters": 0,
    }
    for key, value in defaults.items():
        initialized.setdefault(key, value)
    return initialized


def _hierarchy_xb(
    embeddings: np.ndarray,
    hierarchy_model: HierarchicalModel,
    config: dict[str, Any],
) -> float:
    return hierarchy_xie_beni_index(
        embeddings,
        hierarchy_model,
        **_fuzzy_parameters(config),
    )


def _center_statistics(
    embeddings: np.ndarray,
    hierarchy_model: HierarchicalModel,
    config: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    return _build_center_statistics(
        embeddings,
        hierarchy_model,
        **_fuzzy_parameters(config),
    )


def _fit_hierarchy(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[HierarchicalModel, dict[str, Any], pd.DataFrame]:
    result = run_hierarchical_pca_fcm(
        embeddings,
        metadata,
        **_hierarchy_fit_parameters(config),
    )
    if result.model is None:
        raise RuntimeError("Hierarchical clustering did not return a reusable model")
    return result.model, result.tree, result.assignments


def _fit_visualization(
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Any, Any, np.ndarray]:
    cluster_target, cluster_target_metric, _ = build_cluster_supervision(assignments)
    return fit_projection_model(
        embeddings,
        cluster_target=cluster_target,
        cluster_target_metric=cluster_target_metric,
        **_visualization_fit_parameters(config),
    )


def fit_incremental_state(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    max_depth: int = 4,
    min_node_size: int = 60,
    min_child_size: int = 20,
    min_clusters: int = 2,
    max_clusters: int = 4,
    min_membership: float = 0.20,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
    forced_noise_ratio: float = DEFAULT_FORCED_NOISE_RATIO,
    distance_z: float = 3.5,
    selection_method: str = "multi_metric",
    min_xb_relative_improvement: float = 0.05,
    xb_worsening_patience: int = 2,
    min_split_silhouette: float = 0.05,
    pca_components: int = DEFAULT_CLUSTERING_PCA_COMPONENTS,
    seed: int = 42,
    noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
    visual_pca_components: int = DEFAULT_VISUAL_PCA_COMPONENTS,
    visual_cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
    visual_n_neighbors: int = 15,
    visual_min_dist: float = 0.02,
    visual_metric: str = "cosine",
    visual_spread: float = 0.85,
    visual_densmap: bool = False,
    center_updates_before_membership_refresh: int = (
        DEFAULT_CENTER_UPDATES_BEFORE_MEMBERSHIP_REFRESH
    ),
    max_xb_relative_degradation: float = DEFAULT_MAX_XB_RELATIVE_DEGRADATION,
) -> IncrementalClusterState:
    """Fit the initial batch and persist reusable clustering/visual models."""

    values = _validate_embeddings(embeddings)
    frame = _validate_metadata(metadata, len(values))
    if visual_densmap:
        warnings.warn(
            "UMAP densMAP does not support transforming new points; "
            "disabling densMAP for the incremental projection.",
            RuntimeWarning,
            stacklevel=2,
        )
        visual_densmap = False
    config = _cluster_config(
        max_depth=max_depth,
        min_node_size=min_node_size,
        min_child_size=min_child_size,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_membership=min_membership,
        max_membership_gap=max_membership_gap,
        forced_noise_ratio=forced_noise_ratio,
        distance_z=distance_z,
        selection_method=selection_method,
        min_xb_relative_improvement=min_xb_relative_improvement,
        xb_worsening_patience=xb_worsening_patience,
        min_split_silhouette=min_split_silhouette,
        pca_components=pca_components,
        seed=seed,
        noise_threshold=noise_threshold,
        visual_pca_components=visual_pca_components,
        visual_cluster_target_weight=visual_cluster_target_weight,
        visual_n_neighbors=visual_n_neighbors,
        visual_min_dist=visual_min_dist,
        visual_metric=visual_metric,
        visual_spread=visual_spread,
        visual_densmap=visual_densmap,
        center_updates_before_membership_refresh=(
            center_updates_before_membership_refresh
        ),
        max_xb_relative_degradation=max_xb_relative_degradation,
    )
    hierarchy_model, tree, assignments = _fit_hierarchy(values, frame, config)
    baseline_xie_beni = _hierarchy_xb(values, hierarchy_model, config)
    config["baseline_xie_beni"] = baseline_xie_beni
    config["current_xie_beni"] = baseline_xie_beni
    visual_pca, visual_reducer, coordinates = _fit_visualization(
        values,
        assignments,
        config,
    )
    return IncrementalClusterState(
        embeddings=values.copy(),
        metadata=frame,
        assignments=assignments,
        coordinates=coordinates,
        hierarchy_model=hierarchy_model,
        tree=tree,
        config=config,
        visual_pca=visual_pca,
        visual_reducer=visual_reducer,
        center_statistics=_center_statistics(values, hierarchy_model, config),
    )


def _append_assignments(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*first.columns, *second.columns]))
    return pd.concat(
        [first.reindex(columns=columns), second.reindex(columns=columns)],
        ignore_index=True,
    )


def _apply_global_forced_noise(
    assignments: pd.DataFrame,
    *,
    forced_noise_ratio: float,
) -> pd.DataFrame:
    """Re-rank all accumulated documents and apply one global noise quota."""

    updated = assignments.copy()
    stored_scores = updated.get(
        "noise_score",
        pd.Series(0.0, index=updated.index),
    )
    scores = pd.to_numeric(stored_scores, errors="coerce").fillna(0.0)
    natural_noise = updated.get(
        "is_natural_noise",
        updated["is_noise"],
    ).fillna(False).astype(bool)
    stored_boundary_level = updated.get(
        "boundary_level",
        pd.Series(-1, index=updated.index),
    )
    boundary_level = pd.to_numeric(
        stored_boundary_level,
        errors="coerce",
    ).fillna(-1).astype(int)
    forced_noise = forced_noise_mask(
        scores.to_numpy(),
        updated["id"].to_numpy(),
        forced_noise_ratio=forced_noise_ratio,
    )
    is_noise = natural_noise.to_numpy() | forced_noise
    is_boundary = (
        (boundary_level.to_numpy() >= 1)
        & ~natural_noise.to_numpy()
        & ~forced_noise
    )
    document_types = np.full(len(updated), DOCUMENT_TYPE_CORE, dtype=object)
    document_types[is_boundary] = DOCUMENT_TYPE_BOUNDARY
    document_types[is_noise] = DOCUMENT_TYPE_NOISE

    updated["is_noise"] = is_noise
    updated["is_natural_noise"] = natural_noise.to_numpy()
    updated["is_forced_noise"] = forced_noise
    updated["is_boundary"] = is_boundary
    updated["document_type"] = document_types
    updated["noise_score"] = scores.to_numpy()

    stored_noise_level = updated.get(
        "noise_level",
        pd.Series(-1, index=updated.index),
    )
    noise_level = pd.to_numeric(
        stored_noise_level,
        errors="coerce",
    ).fillna(-1).astype(int).to_numpy(copy=True)
    noise_level[~natural_noise.to_numpy()] = np.where(
        forced_noise[~natural_noise.to_numpy()],
        0,
        -1,
    )
    updated["noise_level"] = noise_level

    level_columns = sorted(
        [
            column
            for column in updated
            if column.startswith("level_") and column.endswith("_cluster")
        ],
        key=lambda column: int(column.split("_")[1]),
    )
    labels = updated[level_columns].fillna(-1).to_numpy(dtype=int)
    assigned_depth = np.sum(labels >= 0, axis=1)
    leaf_cluster = np.full(len(updated), -1, dtype=int)
    has_leaf = assigned_depth > 0
    rows = np.arange(len(updated))
    leaf_cluster[has_leaf] = labels[
        rows[has_leaf],
        assigned_depth[has_leaf] - 1,
    ]
    leaf_cluster[is_noise] = -1
    updated["cluster"] = leaf_cluster

    cluster_paths: list[str] = []
    for row, row_labels in enumerate(labels):
        parts = [str(int(label)) for label in row_labels if label >= 0]
        if is_noise[row]:
            cluster_paths.append("/".join(parts + ["noise"]) if parts else "noise")
        else:
            cluster_paths.append("/".join(parts) if parts else "root")
    updated["cluster_path"] = cluster_paths
    return updated


def _refresh_tree_after_append(
    tree: dict[str, Any],
    new_assignments: pd.DataFrame,
    all_assignments: pd.DataFrame,
    update_count: int,
) -> dict[str, Any]:
    """Update tree counts while preserving the fitted split definitions."""

    updated_tree = copy.deepcopy(tree)
    nodes_by_path: dict[str, dict[str, Any]] = {}

    def collect(node: dict[str, Any]) -> None:
        nodes_by_path[str(node["path"])] = node
        for child in node["children"]:
            collect(child)

    collect(updated_tree["root"])
    level_columns = sorted(
        [
            column
            for column in new_assignments
            if column.startswith("level_") and column.endswith("_cluster")
        ],
        key=lambda column: int(column.split("_")[1]),
    )

    for _, row in new_assignments.iterrows():
        root_node = nodes_by_path.get("")
        if root_node is not None:
            root_node["size"] += 1

        path_parts: list[str] = []
        for column in level_columns:
            label = int(row[column])
            if label < 0:
                break
            path_parts.append(str(label))
            path = "/".join(path_parts)
            node = nodes_by_path.get(path)
            if node is not None:
                node["size"] += 1

        if bool(row["is_noise"]) and not bool(row.get("is_forced_noise", False)):
            noise_level = int(row["noise_level"])
            parent_path = "/".join(path_parts[: max(noise_level - 1, 0)])
            node = nodes_by_path.get(parent_path)
            if node is not None:
                node["noise_count"] += 1
        elif (
            not bool(row.get("is_natural_noise", False))
            and int(row.get("boundary_level", -1)) >= 1
        ):
            boundary_level = max(int(row.get("boundary_level", 1)), 1)
            parent_path = "/".join(path_parts[: boundary_level - 1])
            node = nodes_by_path.get(parent_path)
            if node is not None:
                node["boundary_count"] = int(node.get("boundary_count", 0)) + 1

    summary = copy.deepcopy(updated_tree["summary"])
    summary["samples"] = int(len(all_assignments))
    summary["noise_count"] = int(all_assignments["is_noise"].sum())
    natural_noise = all_assignments.get(
        "is_natural_noise",
        all_assignments["is_noise"],
    ).fillna(False).astype(bool)
    forced_noise = all_assignments.get(
        "is_forced_noise",
        pd.Series(False, index=all_assignments.index),
    ).fillna(False).astype(bool)
    summary["natural_noise_count"] = int(natural_noise.sum())
    summary["forced_noise_count"] = int(forced_noise.sum())
    summary["forced_only_noise_count"] = int((forced_noise & ~natural_noise).sum())
    if "document_type" in all_assignments:
        stored_types = all_assignments["document_type"]
        document_types = stored_types.where(
            stored_types.notna(),
            np.where(
                all_assignments["is_noise"],
                DOCUMENT_TYPE_NOISE,
                DOCUMENT_TYPE_CORE,
            ),
        )
    else:
        document_types = pd.Series(
            np.where(
                all_assignments["is_noise"],
                DOCUMENT_TYPE_NOISE,
                DOCUMENT_TYPE_CORE,
            )
        )
    summary["boundary_count"] = int(
        (document_types == DOCUMENT_TYPE_BOUNDARY).sum()
    )
    summary["core_count"] = int((document_types == DOCUMENT_TYPE_CORE).sum())
    summary["noise_by_level"] = {
        str(level): int((all_assignments["noise_level"] == level).sum())
        for level in range(1, len(level_columns) + 1)
    }
    positive_leaf_levels = all_assignments.loc[
        ~all_assignments["is_noise"], "leaf_level"
    ]
    summary["levels_reached"] = (
        int(positive_leaf_levels.max()) if not positive_leaf_levels.empty else 0
    )
    summary["leaf_cluster_count"] = int(
        all_assignments.loc[
            ~all_assignments["is_noise"], "cluster_path"
        ].nunique()
    )
    summary["incremental_update_count"] = int(update_count)
    updated_tree["summary"] = summary
    updated_tree.setdefault("config", {})["incremental_update_count"] = int(
        update_count
    )
    return updated_tree


def _rebuild_tree_counts(
    tree: dict[str, Any],
    assignments: pd.DataFrame,
    update_count: int,
) -> dict[str, Any]:
    """Rebuild all mutable tree counts while preserving its split structure."""

    reset_tree = copy.deepcopy(tree)

    def reset(node: dict[str, Any]) -> None:
        node["size"] = 0
        node["noise_count"] = 0
        for child in node["children"]:
            reset(child)

    reset(reset_tree["root"])
    return _refresh_tree_after_append(
        reset_tree,
        assignments,
        assignments,
        update_count,
    )


def update_incremental_state(
    state: IncrementalClusterState,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    noise_threshold: float | None = None,
) -> tuple[IncrementalClusterState, dict[str, Any]]:
    """Update centers, memberships, and models on independent schedules."""

    values = _validate_embeddings(embeddings)
    frame = _validate_metadata(metadata, len(values))
    if values.shape[1] != state.embeddings.shape[1]:
        raise ValueError("new embeddings have a different dimensionality")

    existing_ids = set(state.metadata["id"].tolist())
    new_ids = set(frame["id"].tolist())
    if existing_ids.intersection(new_ids):
        raise ValueError("new batch contains IDs already present in the state")

    threshold = _validate_noise_threshold(
        state.config["noise_threshold"]
        if noise_threshold is None
        else noise_threshold
    )
    config = _initialize_update_config(state.config)
    baseline_xie_beni = float(config.get("baseline_xie_beni", np.nan))
    if not np.isfinite(baseline_xie_beni):
        baseline_xie_beni = _hierarchy_xb(
            state.embeddings,
            state.hierarchy_model,
            config,
        )
        config["baseline_xie_beni"] = baseline_xie_beni
    config.setdefault("current_xie_beni", baseline_xie_beni)

    forced_noise_ratio = float(
        config.get("forced_noise_ratio", DEFAULT_FORCED_NOISE_RATIO)
    )
    new_assignments, natural_noise_ratio = assign_to_hierarchy(
        values,
        frame,
        state.hierarchy_model,
        **_assignment_parameters(config, forced_noise_ratio=0.0),
    )
    emergency_recluster = natural_noise_ratio > threshold
    new_coordinates = transform_projection(
        values,
        pca=state.visual_pca,
        reducer=state.visual_reducer,
    )

    combined_embeddings = np.vstack([state.embeddings, values])
    combined_metadata = pd.concat(
        [state.metadata, frame],
        ignore_index=True,
    )
    combined_coordinates = np.vstack([state.coordinates, new_coordinates])
    config["noise_threshold"] = threshold
    config["update_count"] = int(config.get("update_count", 0)) + 1

    center_statistics = getattr(state, "center_statistics", None) or {}
    if not center_statistics:
        center_statistics = _center_statistics(
            state.embeddings,
            state.hierarchy_model,
            config,
        )
    hierarchy_model, center_statistics, updated_node_count = (
        _update_hierarchy_centers(
            state.hierarchy_model,
            center_statistics,
            values,
            **_fuzzy_parameters(config),
        )
    )
    config["center_updates_since_membership_refresh"] = (
        int(config["center_updates_since_membership_refresh"]) + 1
    )
    config["total_center_updates"] = int(config["total_center_updates"]) + 1

    membership_refreshed = (
        int(config["center_updates_since_membership_refresh"])
        >= int(config["center_updates_before_membership_refresh"])
    )
    if membership_refreshed:
        config["center_updates_since_membership_refresh"] = 0
        config["membership_refreshes_since_recluster"] = (
            int(config["membership_refreshes_since_recluster"]) + 1
        )
        config["total_membership_refreshes"] = (
            int(config["total_membership_refreshes"]) + 1
        )

    refreshed_assignments: pd.DataFrame | None = None
    refreshed_tree: dict[str, Any] | None = None
    current_xie_beni = float(config["current_xie_beni"])
    xb_relative_degradation: float | None = None
    xb_degradation_recluster = False
    if membership_refreshed:
        hierarchy_model = _refresh_distance_thresholds(
            combined_embeddings,
            hierarchy_model,
            distance_z=float(config["distance_z"]),
            **_fuzzy_parameters(config),
        )
        refreshed_assignments, _ = assign_to_hierarchy(
            combined_embeddings,
            combined_metadata,
            hierarchy_model,
            **_assignment_parameters(
                config,
                forced_noise_ratio=forced_noise_ratio,
            ),
        )
        refreshed_tree = _rebuild_tree_counts(
            state.tree,
            refreshed_assignments,
            int(config["update_count"]),
        )
        center_statistics = _center_statistics(
            combined_embeddings,
            hierarchy_model,
            config,
        )
        current_xie_beni = _hierarchy_xb(
            combined_embeddings,
            hierarchy_model,
            config,
        )
        config["current_xie_beni"] = current_xie_beni
        if np.isfinite(baseline_xie_beni) and np.isfinite(current_xie_beni):
            xb_relative_degradation = float(
                (current_xie_beni - baseline_xie_beni)
                / max(abs(baseline_xie_beni), 1e-12)
            )
            xb_degradation_recluster = (
                xb_relative_degradation
                >= float(config["max_xb_relative_degradation"])
            )

    should_recluster = emergency_recluster or xb_degradation_recluster

    visualization_refitted = False
    visual_pca = state.visual_pca
    visual_reducer = state.visual_reducer
    if should_recluster:
        hierarchy_model, tree, assignments = _fit_hierarchy(
            combined_embeddings,
            combined_metadata,
            config,
        )
        center_statistics = _center_statistics(
            combined_embeddings,
            hierarchy_model,
            config,
        )
        config["center_updates_since_membership_refresh"] = 0
        config["membership_refreshes_since_recluster"] = 0
        config["total_reclusters"] = int(config["total_reclusters"]) + 1
        baseline_xie_beni = _hierarchy_xb(
            combined_embeddings,
            hierarchy_model,
            config,
        )
        current_xie_beni = baseline_xie_beni
        config["baseline_xie_beni"] = baseline_xie_beni
        config["current_xie_beni"] = current_xie_beni
        reclustered = True
    elif membership_refreshed:
        if refreshed_assignments is None or refreshed_tree is None:
            raise RuntimeError("Membership refresh results are unavailable")
        assignments = refreshed_assignments
        tree = refreshed_tree
        reclustered = False
    else:
        assignments = _append_assignments(state.assignments, new_assignments)
        assignments = _apply_global_forced_noise(
            assignments,
            forced_noise_ratio=forced_noise_ratio,
        )
        effective_new_assignments = assignments.iloc[-len(values):].copy()
        tree = _refresh_tree_after_append(
            state.tree,
            effective_new_assignments,
            assignments,
            int(config["update_count"]),
        )
        reclustered = False

    effective_new_assignments = assignments.iloc[-len(values):].copy()
    new_noise_ratio = float(effective_new_assignments["is_noise"].mean())
    config["last_update_noise_ratio"] = float(new_noise_ratio)
    config["last_update_natural_noise_ratio"] = float(natural_noise_ratio)
    config["last_update_reclustered"] = reclustered
    config["last_update_membership_refreshed"] = membership_refreshed
    config["last_update_visualization_refitted"] = visualization_refitted
    config["last_xb_relative_degradation"] = xb_relative_degradation
    config["last_update_xb_degradation_recluster"] = xb_degradation_recluster
    tree.setdefault("config", {}).update(
        {
            "center_updates_before_membership_refresh": int(
                config["center_updates_before_membership_refresh"]
            ),
            "max_xb_relative_degradation": float(
                config["max_xb_relative_degradation"]
            ),
            "center_updates_since_membership_refresh": int(
                config["center_updates_since_membership_refresh"]
            ),
            "membership_refreshes_since_recluster": int(
                config["membership_refreshes_since_recluster"]
            ),
            "baseline_xie_beni": float(config["baseline_xie_beni"]),
            "current_xie_beni": float(config["current_xie_beni"]),
        }
    )
    tree.setdefault("summary", {}).update(
        {
            "total_center_updates": int(config["total_center_updates"]),
            "total_membership_refreshes": int(
                config["total_membership_refreshes"]
            ),
            "total_reclusters": int(config["total_reclusters"]),
        }
    )
    updated_state = IncrementalClusterState(
        embeddings=combined_embeddings,
        metadata=combined_metadata,
        assignments=assignments,
        coordinates=combined_coordinates,
        hierarchy_model=hierarchy_model,
        tree=tree,
        config=config,
        visual_pca=visual_pca,
        visual_reducer=visual_reducer,
        center_statistics=center_statistics,
    )
    summary = {
        "new_samples": int(len(values)),
        "new_noise_count": int(round(new_noise_ratio * len(values))),
        "new_natural_noise_count": int(
            effective_new_assignments["is_natural_noise"].sum()
        ),
        "new_forced_noise_count": int(
            effective_new_assignments["is_forced_noise"].sum()
        ),
        "new_boundary_count": int(effective_new_assignments["is_boundary"].sum()),
        "new_core_count": int(
            (
                effective_new_assignments["document_type"]
                == DOCUMENT_TYPE_CORE
            ).sum()
        ),
        "new_noise_ratio": float(new_noise_ratio),
        "new_natural_noise_ratio": float(natural_noise_ratio),
        "noise_threshold": float(threshold),
        "center_updated": updated_node_count > 0,
        "updated_center_nodes": int(updated_node_count),
        "membership_refreshed": membership_refreshed,
        "xie_beni": float(current_xie_beni),
        "xb_relative_degradation": xb_relative_degradation,
        "xb_degradation_recluster": xb_degradation_recluster,
        "emergency_recluster": emergency_recluster,
        "reclustered": reclustered,
        "visualization_refitted": visualization_refitted,
        "center_updates_since_membership_refresh": int(
            config["center_updates_since_membership_refresh"]
        ),
        "membership_refreshes_since_recluster": int(
            config["membership_refreshes_since_recluster"]
        ),
        "total_samples": int(len(combined_embeddings)),
    }
    return updated_state, summary


def coordinates_frame(state: IncrementalClusterState) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": state.metadata["id"].tolist(),
            "umap_1": state.coordinates[:, 0],
            "umap_2": state.coordinates[:, 1],
        }
    )


def save_state(state: IncrementalClusterState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    payload = {
        "version": 2,
        "embeddings": state.embeddings,
        "metadata": state.metadata,
        "assignments": state.assignments,
        "coordinates": state.coordinates,
        "hierarchy_model": state.hierarchy_model,
        "tree": state.tree,
        "config": state.config,
        "visual_pca": state.visual_pca,
        "visual_reducer": state.visual_reducer,
        "center_statistics": state.center_statistics,
    }
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def load_state(path: Path) -> IncrementalClusterState:
    with path.open("rb") as handle:
        try:
            payload = pickle.load(handle)
        except AttributeError as error:
            # States written by an earlier CLI invocation may refer to
            # IncrementalClusterState as __main__. Keep them readable once.
            setattr(sys.modules["__main__"], "IncrementalClusterState", IncrementalClusterState)
            handle.seek(0)
            try:
                payload = pickle.load(handle)
            except Exception:
                raise error

    if isinstance(payload, IncrementalClusterState):
        state = payload
    elif isinstance(payload, dict) and payload.get("version") in {1, 2}:
        required_fields = (
            "embeddings",
            "metadata",
            "assignments",
            "coordinates",
            "hierarchy_model",
            "tree",
            "config",
            "visual_pca",
            "visual_reducer",
        )
        if any(key not in payload for key in required_fields):
            raise ValueError(f"Invalid incremental state: {path}")
        fields = {key: payload[key] for key in required_fields}
        fields["center_statistics"] = payload.get("center_statistics", {})
        state = IncrementalClusterState(**fields)
    else:
        raise ValueError(f"Invalid incremental state: {path}")
    if not hasattr(state, "center_statistics"):
        state.center_statistics = {}
    if state.config.get("recluster_trigger_policy") != "xb_and_noise_v2":
        state.config["noise_threshold"] = DEFAULT_NOISE_THRESHOLD
        state.config.setdefault(
            "max_xb_relative_degradation",
            DEFAULT_MAX_XB_RELATIVE_DEGRADATION,
        )
        state.config["recluster_trigger_policy"] = "xb_and_noise_v2"
    _validate_embeddings(state.embeddings)
    _validate_metadata(state.metadata, len(state.embeddings))
    if len(state.assignments) != len(state.embeddings):
        raise ValueError("State assignments and embeddings do not align")
    if state.coordinates.shape != (len(state.embeddings), 2):
        raise ValueError("State coordinates must have shape (samples, 2)")
    return state


def write_outputs(
    state: IncrementalClusterState,
    *,
    assignments_output: Path | None = None,
    coordinates_output: Path | None = None,
    tree_output: Path | None = None,
    plot_output: Path | None = None,
    title: str = "Incremental Clustering",
    color_by: str = "auto",
) -> None:
    if assignments_output is not None:
        assignments_output.parent.mkdir(parents=True, exist_ok=True)
        state.assignments.to_csv(assignments_output, index=False)
    if coordinates_output is not None:
        coordinates_output.parent.mkdir(parents=True, exist_ok=True)
        coordinates_frame(state).to_csv(coordinates_output, index=False)
    if tree_output is not None:
        tree_output.parent.mkdir(parents=True, exist_ok=True)
        tree_output.write_text(
            json.dumps(state.tree, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if plot_output is not None:
        configured_target_weight = state.config.get("visual_cluster_target_weight")
        make_fixed_coordinate_plot(
            state.coordinates,
            state.assignments,
            plot_output,
            title=title,
            color_by=color_by,
            pca_components=int(
                state.config.get(
                    "visual_pca_components",
                    DEFAULT_VISUAL_PCA_COMPONENTS,
                )
            ),
            cluster_target_weight=(
                None
                if configured_target_weight is None
                else float(configured_target_weight)
            ),
        )


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--id-offset",
        type=int,
        default=0,
        help="Offset generated index IDs when records do not contain id/resource.",
    )


def _add_visual_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assignments-output", type=Path, default=None)
    parser.add_argument("--coordinates-output", type=Path, default=None)
    parser.add_argument("--tree-output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--title", type=str, default="Incremental Clustering")
    parser.add_argument(
        "--color-by",
        choices=["auto", "cluster"],
        default="auto",
    )


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-node-size", type=int, default=60)
    parser.add_argument("--min-child-size", type=int, default=20)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--min-membership", type=float, default=0.20)
    parser.add_argument(
        "--max-membership-gap",
        type=float,
        default=DEFAULT_MAX_MEMBERSHIP_GAP,
        help=(
            "Treat low-membership points as boundary candidates when the "
            "top-two membership gap is below this value (default: 0.10)."
        ),
    )
    parser.add_argument(
        "--forced-noise-ratio",
        type=float,
        default=DEFAULT_FORCED_NOISE_RATIO,
        help="Force the highest-risk fraction to noise (default: 0.01).",
    )
    parser.add_argument("--distance-z", type=float, default=3.5)
    parser.add_argument(
        "--selection-method",
        choices=["silhouette", "knee", "xie_beni", "multi_metric"],
        default="multi_metric",
    )
    parser.add_argument(
        "--min-xb-relative-improvement",
        type=float,
        default=0.05,
        help=(
            "Legacy xie_beni method: stop when XB relative improvement falls "
            "below this value (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--xb-worsening-patience",
        type=int,
        default=2,
        help=(
            "After XB first worsens, evaluate this many additional k values "
            "for multi-metric selection (default: 2)."
        ),
    )
    parser.add_argument("--min-split-silhouette", type=float, default=0.05)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_CLUSTERING_PCA_COMPONENTS,
        help="PCA dimensions used for clustering (default: 256).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=DEFAULT_NOISE_THRESHOLD,
        help="Re-cluster when the new batch noise ratio exceeds this value.",
    )
    parser.add_argument(
        "--visual-pca-components",
        type=int,
        default=DEFAULT_VISUAL_PCA_COMPONENTS,
        help="PCA dimensions before the 2D UMAP visualization (default: 64).",
    )
    parser.add_argument(
        "--visual-cluster-target-weight",
        type=float,
        default=DEFAULT_CLUSTER_TARGET_WEIGHT,
        help=(
            "Weak supervised UMAP weight for cluster membership; 0 disables it "
            "(default: 0.01)."
        ),
    )
    parser.add_argument("--visual-n-neighbors", type=int, default=15)
    parser.add_argument("--visual-min-dist", type=float, default=0.02)
    parser.add_argument("--visual-metric", type=str, default="cosine")
    parser.add_argument("--visual-spread", type=float, default=0.85)
    parser.add_argument(
        "--visual-densmap",
        action="store_true",
        default=False,
        help="Request densMAP; it is disabled because densMAP cannot transform new points.",
    )
    parser.add_argument(
        "--center-updates-before-membership-refresh",
        type=int,
        default=DEFAULT_CENTER_UPDATES_BEFORE_MEMBERSHIP_REFRESH,
        help="Recompute every document membership after this many batch center updates.",
    )
    parser.add_argument(
        "--max-xb-relative-degradation",
        type=float,
        default=DEFAULT_MAX_XB_RELATIVE_DEGRADATION,
        help="Re-cluster when the hierarchy XB index worsens by this fraction.",
    )


def _derived_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}")


def _run_fit(args: argparse.Namespace) -> None:
    embeddings, metadata = load_embeddings_from_json(
        args.input_json,
        start=args.start,
        limit=args.limit,
        id_offset=args.id_offset,
    )
    state = fit_incremental_state(
        embeddings,
        metadata,
        max_depth=args.max_depth,
        min_node_size=args.min_node_size,
        min_child_size=args.min_child_size,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_membership=args.min_membership,
        max_membership_gap=args.max_membership_gap,
        forced_noise_ratio=args.forced_noise_ratio,
        distance_z=args.distance_z,
        selection_method=args.selection_method,
        min_xb_relative_improvement=args.min_xb_relative_improvement,
        xb_worsening_patience=args.xb_worsening_patience,
        min_split_silhouette=args.min_split_silhouette,
        pca_components=args.pca_components,
        seed=args.seed,
        noise_threshold=args.noise_threshold,
        visual_pca_components=args.visual_pca_components,
        visual_cluster_target_weight=args.visual_cluster_target_weight,
        visual_n_neighbors=args.visual_n_neighbors,
        visual_min_dist=args.visual_min_dist,
        visual_metric=args.visual_metric,
        visual_spread=args.visual_spread,
        visual_densmap=args.visual_densmap,
        center_updates_before_membership_refresh=(
            args.center_updates_before_membership_refresh
        ),
        max_xb_relative_degradation=args.max_xb_relative_degradation,
    )
    save_state(state, args.state_output)
    write_outputs(
        state,
        assignments_output=args.assignments_output
        or _derived_path(args.state_output, "assignments.csv"),
        coordinates_output=args.coordinates_output
        or _derived_path(args.state_output, "coordinates.csv"),
        tree_output=args.tree_output or _derived_path(args.state_output, "tree.json"),
        plot_output=args.plot_output,
        title=args.title,
        color_by=args.color_by,
    )
    print(
        f"Initial state saved: {args.state_output} "
        f"({len(embeddings)} samples, "
        f"{int(state.assignments['is_noise'].sum())} noise)"
    )


def _run_update(args: argparse.Namespace) -> None:
    state = load_state(args.state)
    embeddings, metadata = load_embeddings_from_json(
        args.input_json,
        start=args.start,
        limit=args.limit,
        id_offset=args.id_offset,
    )
    updated_state, summary = update_incremental_state(
        state,
        embeddings,
        metadata,
        noise_threshold=args.noise_threshold,
    )
    output_state = args.state_output or args.state
    save_state(updated_state, output_state)
    write_outputs(
        updated_state,
        assignments_output=args.assignments_output
        or _derived_path(output_state, "assignments.csv"),
        coordinates_output=args.coordinates_output
        or _derived_path(output_state, "coordinates.csv"),
        tree_output=args.tree_output or _derived_path(output_state, "tree.json"),
        plot_output=args.plot_output,
        title=args.title,
        color_by=args.color_by,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and update an incremental hierarchical clustering state."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit the initial batch.")
    _add_input_args(fit_parser)
    fit_parser.add_argument("--state-output", type=Path, required=True)
    _add_visual_output_args(fit_parser)
    _add_cluster_args(fit_parser)
    fit_parser.set_defaults(handler=_run_fit)

    update_parser = subparsers.add_parser(
        "update",
        help=(
            "Update centers and periodically refresh memberships, clustering, "
            "while keeping visualization coordinates fixed."
        ),
    )
    _add_input_args(update_parser)
    update_parser.add_argument("--state", type=Path, required=True)
    update_parser.add_argument("--state-output", type=Path, default=None)
    update_parser.add_argument("--noise-threshold", type=float, default=None)
    _add_visual_output_args(update_parser)
    update_parser.set_defaults(handler=_run_update)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
