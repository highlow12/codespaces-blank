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
from embedding_data import load_embeddings_from_json, sample_embedding_batch
from fcm_hierarchy import (
    DEFAULT_FORCED_NOISE_RATIO,
    DEFAULT_MAX_MEMBERSHIP_GAP,
    DOCUMENT_TYPE_BOUNDARY,
    DOCUMENT_TYPE_CORE,
    DOCUMENT_TYPE_NOISE,
    classify_fcm_documents,
    conditional_memberships_from_projected,
    fcm_memberships_from_centers,
    sfcm_memberships_from_centers,
    fcm_noise_scores,
    forced_noise_mask,
    merge_forced_noise,
    run_hierarchical_pca_fcm,
    transform_pca_normalized_features,
)
from hierarchical_assignments import (
    build_hierarchical_assignments as _assignments_from_labels,
)
from pca_dimension_search import (
    DEFAULT_K_VALUES,
    DEFAULT_MAX_COMPONENTS,
    DEFAULT_MINIMUM_PRESERVATION_GAIN,
)
from pca_dimension_selection import (
    DEFAULT_COMPONENT_STEP,
    DEFAULT_MIN_COMPONENTS,
)
from pca_projection import pca_projection_support


DEFAULT_NOISE_THRESHOLD = 0.05
DEFAULT_DRIFT_MIN_SAMPLES = 20
DEFAULT_DRIFT_EWMA_ALPHA = 0.30
DEFAULT_NOISE_RELEASE_RATIO = 0.50
DEFAULT_RECLUSTER_COOLDOWN_UPDATES = 3
DEFAULT_MEMBERSHIP_REFRESH_MIN_CENTER_MOVEMENT = 0.01
DEFAULT_MEMBERSHIP_REFRESH_MIN_INFLUENCE = 0.0025
DEFAULT_CENTER_UPDATES_BEFORE_MEMBERSHIP_REFRESH = 10
DEFAULT_MAX_XB_RELATIVE_DEGRADATION = 0.05
CENTER_CONTRIBUTION_FORMAT = "compact_weights_v1"
RECLUSTER_TRIGGER_POLICY = "xb_and_noise_stable_v3"
STATE_VERSION = 6


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
    center_contributions: dict[Any, dict[str, Any]] = field(
        default_factory=dict
    )
    membership_reference_centers: dict[str, np.ndarray] = field(
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


def _validate_drift_settings(
    *,
    noise_threshold: float,
    noise_release_threshold: float,
    drift_min_samples: int,
    drift_ewma_alpha: float,
    recluster_cooldown_updates: int,
) -> tuple[float, float, int, float, int]:
    enter_threshold = _validate_noise_threshold(noise_threshold)
    release_threshold = _validate_noise_threshold(noise_release_threshold)
    if release_threshold > enter_threshold:
        raise ValueError(
            "noise_release_threshold must not exceed noise_threshold"
        )
    minimum_samples = int(drift_min_samples)
    if minimum_samples < 1:
        raise ValueError("drift_min_samples must be at least 1")
    alpha = float(drift_ewma_alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("drift_ewma_alpha must be between 0 (exclusive) and 1")
    cooldown = int(recluster_cooldown_updates)
    if cooldown < 0:
        raise ValueError("recluster_cooldown_updates must be non-negative")
    return (
        enter_threshold,
        release_threshold,
        minimum_samples,
        alpha,
        cooldown,
    )


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
    projection_support = pca_projection_support(values, hierarchy_model.pca)
    projection_support_threshold = float(
        getattr(hierarchy_model, "projection_support_threshold", 0.0)
    )
    projection_outliers = (
        projection_support_threshold > 0.0
    ) & (projection_support < projection_support_threshold)
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
    is_noise = projection_outliers.copy()
    document_types = np.full(len(values), DOCUMENT_TYPE_CORE, dtype=object)
    document_types[projection_outliers] = DOCUMENT_TYPE_NOISE
    noise_scores = np.clip(1.0 - projection_support, 0.0, 1.0)
    boundary_level = np.full(len(values), -1, dtype=int)
    noise_level = np.full(len(values), -1, dtype=int)
    noise_level[projection_outliers] = 1
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
        valid_rows = ~projection_outliers
        labels_by_level[valid_rows, 0] = 0
        soft_memberships_by_level[0][valid_rows, 0] = 1.0
    else:
        active: dict[str, np.ndarray] = {
            "": np.flatnonzero(~projection_outliers)
        }
        for depth in range(hierarchy_model.max_depth):
            next_active: dict[str, np.ndarray] = {}
            for parent_path, indices in active.items():
                node_model = hierarchy_model.nodes.get(parent_path)
                if node_model is None or indices.size == 0:
                    continue
                node_m = float(getattr(node_model, "m", m))

                memberships, distances = sfcm_memberships_from_centers(
                    projected[indices],
                    node_model.centers,
                    m=node_m,
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
        projection_support,
        projection_support_threshold,
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

    projection_support = pca_projection_support(values, hierarchy_model.pca)
    support_threshold = float(
        getattr(hierarchy_model, "projection_support_threshold", 0.0)
    )
    active: dict[str, np.ndarray] = {
        "": np.flatnonzero(
            (support_threshold <= 0.0)
            | (projection_support >= support_threshold)
        )
    }
    for _depth in range(hierarchy_model.max_depth):
        next_active: dict[str, np.ndarray] = {}
        for parent_path, indices in active.items():
            node_model = hierarchy_model.nodes.get(parent_path)
            if node_model is None or indices.size == 0:
                continue
            node_m = float(getattr(node_model, "m", m))

            memberships, distances = sfcm_memberships_from_centers(
                projected[indices],
                node_model.centers,
                m=node_m,
            )
            weights = memberships**node_m
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


def _center_contributions_for_batch(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    hierarchy_model: HierarchicalModel,
    *,
    min_membership: float,
    m: float,
) -> dict[Any, dict[str, Any]]:
    """Collect each document's fuzzy contribution to every reached node.

    Each document stores its projected vector once and only the fuzzy weights
    for hierarchy nodes it reached. The weighted outer products are rebuilt
    only while applying a delta or doing a scheduled full refresh. This avoids
    persisting one ``clusters x dimensions`` array per document and node.
    """

    values = _validate_embeddings(embeddings)
    frame = _validate_metadata(metadata, len(values))
    identifiers = frame["id"].tolist()
    projected = transform_pca_normalized_features(values, hierarchy_model.pca)
    contributions: dict[Any, dict[str, Any]] = {
        identifier: {
            "projected": projected[index].copy(),
            "weights_by_path": {},
        }
        for index, identifier in enumerate(identifiers)
    }
    if hierarchy_model.fallback_single_cluster:
        return contributions

    projection_support = pca_projection_support(values, hierarchy_model.pca)
    support_threshold = float(
        getattr(hierarchy_model, "projection_support_threshold", 0.0)
    )
    active: dict[str, np.ndarray] = {
        "": np.flatnonzero(
            (support_threshold <= 0.0)
            | (projection_support >= support_threshold)
        )
    }
    for _depth in range(hierarchy_model.max_depth):
        next_active: dict[str, np.ndarray] = {}
        for parent_path, indices in active.items():
            node_model = hierarchy_model.nodes.get(parent_path)
            if node_model is None or indices.size == 0:
                continue
            node_m = float(getattr(node_model, "m", m))

            memberships, distances = sfcm_memberships_from_centers(
                projected[indices],
                node_model.centers,
                m=node_m,
            )
            weights = memberships**node_m
            for local_index, global_index in enumerate(indices):
                identifier = identifiers[int(global_index)]
                contributions[identifier]["weights_by_path"][parent_path] = (
                    weights[local_index].copy()
                )

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
    return contributions


def _aggregate_center_contributions(
    contributions: dict[Any, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Sum per-document contributions into the persisted center statistics."""

    statistics: dict[str, dict[str, np.ndarray]] = {}
    for contribution in contributions.values():
        projected = np.asarray(contribution["projected"], dtype=np.float64)
        for path, stored_weights in contribution["weights_by_path"].items():
            weights = np.asarray(stored_weights, dtype=np.float64)
            if path not in statistics:
                statistics[path] = {
                    "weighted_sum": np.zeros(
                        (len(weights), len(projected)),
                        dtype=np.float64,
                    ),
                    "weight": np.zeros_like(weights),
                }
            statistics[path]["weighted_sum"] += np.outer(weights, projected)
            statistics[path]["weight"] += weights
    return statistics


def _legacy_center_contribution_to_compact(
    contribution: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Convert a pre-v4 outer-product contribution without losing precision."""

    weights_by_path: dict[str, np.ndarray] = {}
    projected: np.ndarray | None = None
    for path, values in contribution.items():
        weights = np.asarray(values["weight"], dtype=np.float64)
        weighted_sum = np.asarray(values["weighted_sum"], dtype=np.float64)
        if weighted_sum.ndim != 2 or weighted_sum.shape[0] != len(weights):
            raise ValueError("Invalid legacy center contribution")
        weights_by_path[path] = weights.copy()
        if projected is None:
            usable = np.flatnonzero(np.abs(weights) > 1e-12)
            if usable.size:
                projected = weighted_sum[int(usable[0])] / weights[int(usable[0])]
            else:
                projected = np.zeros(weighted_sum.shape[1], dtype=np.float64)
    if projected is None:
        projected = np.empty(0, dtype=np.float64)
    return {
        "projected": np.asarray(projected, dtype=np.float64).copy(),
        "weights_by_path": weights_by_path,
    }


def _compact_center_contributions(
    contributions: dict[Any, dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
    """Normalize legacy persisted contributions to the compact v4 schema."""

    compact: dict[Any, dict[str, Any]] = {}
    for identifier, contribution in contributions.items():
        if "projected" in contribution and "weights_by_path" in contribution:
            compact[identifier] = contribution
        else:
            compact[identifier] = _legacy_center_contribution_to_compact(
                contribution
            )
    return compact


def _copy_center_statistics(
    statistics: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Copy only the small per-node aggregates before applying one batch."""

    return {
        path: {
            "weighted_sum": values["weighted_sum"].copy(),
            "weight": values["weight"].copy(),
        }
        for path, values in statistics.items()
    }


def _apply_center_contribution_delta(
    statistics: dict[str, dict[str, np.ndarray]],
    contribution: dict[str, Any],
    *,
    sign: float,
) -> None:
    """Add or subtract one compact document contribution in place."""

    projected = np.asarray(contribution["projected"], dtype=np.float64)
    for path, stored_weights in contribution["weights_by_path"].items():
        weights = np.asarray(stored_weights, dtype=np.float64)
        if path not in statistics:
            if sign < 0.0:
                raise ValueError(
                    f"Center statistics are missing contribution path: {path}"
                )
            statistics[path] = {
                "weighted_sum": np.zeros(
                    (len(weights), len(projected)),
                    dtype=np.float64,
                ),
                "weight": np.zeros_like(weights),
            }
        values = statistics[path]
        expected_shape = (len(weights), len(projected))
        if (
            values["weighted_sum"].shape != expected_shape
            or values["weight"].shape != weights.shape
        ):
            raise ValueError(f"Center contribution shape mismatch at node: {path}")
        values["weighted_sum"] += sign * np.outer(weights, projected)
        values["weight"] += sign * weights
        values["weighted_sum"][np.abs(values["weighted_sum"]) < 1e-14] = 0.0
        values["weight"][np.abs(values["weight"]) < 1e-14] = 0.0
        if np.any(values["weight"] < -1e-10):
            raise ValueError(f"Center contribution underflow at node: {path}")


def _update_hierarchy_centers_from_statistics(
    hierarchy_model: HierarchicalModel,
    center_statistics: dict[str, dict[str, np.ndarray]],
) -> tuple[HierarchicalModel, int]:
    """Recompute centers from a complete set of cumulative statistics."""

    updated_model = copy.deepcopy(hierarchy_model)
    updated_node_count = 0
    for path, values in center_statistics.items():
        node_model = updated_model.nodes.get(path)
        if node_model is None:
            continue
        weighted_sum = values["weighted_sum"]
        weight = values["weight"]
        if (
            weighted_sum.shape
            != (node_model.centers.shape[0], node_model.centers.shape[1])
            or weight.shape != (node_model.centers.shape[0],)
        ):
            raise ValueError(
                f"Center statistics do not match hierarchy node: {path}"
            )
        raw_centers = np.divide(
            weighted_sum,
            weight[:, None],
            out=node_model.centers.copy(),
            where=weight[:, None] > 1e-12,
        )
        center_norms = np.linalg.norm(raw_centers, axis=1, keepdims=True)
        node_model.centers = np.divide(
            raw_centers,
            center_norms,
            out=node_model.centers.copy(),
            where=center_norms > 1e-12,
        )
        updated_node_count += 1
    return updated_model, updated_node_count


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
            node_m = float(getattr(node_model, "m", m))
            memberships, distances = fcm_memberships_from_centers(
                projected[indices],
                node_model.centers,
                m=node_m,
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
            node_m = float(getattr(node_model, "m", m))
            memberships, distances = fcm_memberships_from_centers(
                projected[indices],
                node_model.centers,
                m=node_m,
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
                numerator = float(
                    np.sum((memberships**node_m) * (distances**2))
                )
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
    pca_components: int | None,
    pca_max_components: int,
    pca_min_components: int,
    pca_component_step: int,
    pca_k_values: tuple[int, ...],
    pca_minimum_preservation_gain: float,
    seed: int,
    noise_threshold: float,
    noise_release_threshold: float | None,
    drift_min_samples: int,
    drift_ewma_alpha: float,
    recluster_cooldown_updates: int,
    visual_pca_components: int | None,
    visual_cluster_target_weight: float,
    visual_n_neighbors: int,
    visual_min_dist: float,
    visual_metric: str,
    visual_spread: float,
    visual_densmap: bool,
    center_updates_before_membership_refresh: int,
    selective_membership_refresh: bool,
    membership_refresh_min_center_movement: float,
    membership_refresh_min_influence: float,
    max_xb_relative_degradation: float,
    fuzzifier: float,
    max_fcm_iter: int,
    fcm_tol: float,
    fast_mode: bool,
    fast_sample_size: int,
    fast_scout_n_init: int,
    fast_refine_n_init: int,
    fast_refine_top_k: int,
    fast_stability_target: float,
    fast_m_values: tuple[float, ...],
) -> dict[str, Any]:
    if center_updates_before_membership_refresh < 1:
        raise ValueError(
            "center_updates_before_membership_refresh must be at least 1"
        )
    if (
        not np.isfinite(membership_refresh_min_center_movement)
        or membership_refresh_min_center_movement < 0.0
    ):
        raise ValueError(
            "membership_refresh_min_center_movement must be non-negative"
        )
    if (
        not np.isfinite(membership_refresh_min_influence)
        or membership_refresh_min_influence < 0.0
    ):
        raise ValueError(
            "membership_refresh_min_influence must be non-negative"
        )
    if (
        not np.isfinite(max_xb_relative_degradation)
        or max_xb_relative_degradation < 0.0
    ):
        raise ValueError("max_xb_relative_degradation must be non-negative")
    enter_threshold = _validate_noise_threshold(noise_threshold)
    release_threshold_auto = noise_release_threshold is None
    resolved_release_threshold = (
        enter_threshold * DEFAULT_NOISE_RELEASE_RATIO
        if noise_release_threshold is None
        else noise_release_threshold
    )
    (
        enter_threshold,
        resolved_release_threshold,
        drift_min_samples,
        drift_ewma_alpha,
        recluster_cooldown_updates,
    ) = _validate_drift_settings(
        noise_threshold=enter_threshold,
        noise_release_threshold=resolved_release_threshold,
        drift_min_samples=drift_min_samples,
        drift_ewma_alpha=drift_ewma_alpha,
        recluster_cooldown_updates=recluster_cooldown_updates,
    )
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
        "pca_components": (
            None if pca_components is None else int(pca_components)
        ),
        "pca_components_requested": (
            "auto" if pca_components is None else int(pca_components)
        ),
        "pca_components_auto": pca_components is None,
        "pca_max_components": int(pca_max_components),
        "pca_min_components": int(pca_min_components),
        "pca_component_step": int(pca_component_step),
        "pca_k_values": [int(value) for value in pca_k_values],
        "pca_minimum_preservation_gain": float(
            pca_minimum_preservation_gain
        ),
        "seed": int(seed),
        "m": float(fuzzifier),
        "max_fcm_iter": int(max_fcm_iter),
        "fcm_tol": float(fcm_tol),
        "fast_mode": bool(fast_mode),
        "fast_sample_size": int(fast_sample_size),
        "fast_scout_n_init": int(fast_scout_n_init),
        "fast_refine_n_init": int(fast_refine_n_init),
        "fast_refine_top_k": int(fast_refine_top_k),
        "fast_stability_target": float(fast_stability_target),
        "fast_m_values": [float(value) for value in fast_m_values],
        "noise_threshold": enter_threshold,
        "noise_release_threshold": resolved_release_threshold,
        "noise_release_threshold_auto": release_threshold_auto,
        "drift_min_samples": drift_min_samples,
        "drift_ewma_alpha": drift_ewma_alpha,
        "drift_pending_samples": 0,
        "drift_pending_natural_noise": 0,
        "drift_ewma_noise_ratio": None,
        "drift_alarm_active": False,
        "recluster_cooldown_updates": recluster_cooldown_updates,
        "recluster_cooldown_remaining": 0,
        "visual_pca_components": (
            None
            if visual_pca_components is None
            else int(visual_pca_components)
        ),
        "visual_pca_components_requested": (
            "auto"
            if visual_pca_components is None
            else int(visual_pca_components)
        ),
        "visual_pca_components_auto": visual_pca_components is None,
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
        "selective_membership_refresh": bool(selective_membership_refresh),
        "membership_refresh_min_center_movement": float(
            membership_refresh_min_center_movement
        ),
        "membership_refresh_min_influence": float(
            membership_refresh_min_influence
        ),
        "max_xb_relative_degradation": float(max_xb_relative_degradation),
        "center_updates_since_membership_refresh": 0,
        "membership_refreshes_since_recluster": 0,
        "total_center_updates": 0,
        "total_membership_refreshes": 0,
        "total_reclusters": 0,
        "recluster_trigger_policy": RECLUSTER_TRIGGER_POLICY,
        "center_contribution_format": CENTER_CONTRIBUTION_FORMAT,
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
    auto_select = bool(
        config.get(
            "pca_components_auto",
            config.get("pca_components") is None,
        )
    )
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
        "pca_components": (
            None if auto_select else int(config["pca_components"])
        ),
        "pca_max_components": int(
            config.get("pca_max_components", DEFAULT_MAX_COMPONENTS)
        ),
        "pca_min_components": int(
            config.get("pca_min_components", DEFAULT_MIN_COMPONENTS)
        ),
        "pca_component_step": int(
            config.get("pca_component_step", DEFAULT_COMPONENT_STEP)
        ),
        "pca_k_values": tuple(
            int(value)
            for value in config.get("pca_k_values", DEFAULT_K_VALUES)
        ),
        "pca_minimum_preservation_gain": float(
            config.get(
                "pca_minimum_preservation_gain",
                DEFAULT_MINIMUM_PRESERVATION_GAIN,
            )
        ),
        "seed": int(config["seed"]),
        "min_membership": float(config["min_membership"]),
        "max_membership_gap": float(
            config.get("max_membership_gap", DEFAULT_MAX_MEMBERSHIP_GAP)
        ),
        "forced_noise_ratio": float(
            config.get("forced_noise_ratio", DEFAULT_FORCED_NOISE_RATIO)
        ),
        "m": float(config.get("m", 2.0)),
        "max_fcm_iter": int(config.get("max_fcm_iter", 200)),
        "fcm_tol": float(config.get("fcm_tol", 1e-6)),
        "fast_mode": bool(config.get("fast_mode", False)),
        "fast_sample_size": int(config.get("fast_sample_size", 1000)),
        "fast_scout_n_init": int(config.get("fast_scout_n_init", 2)),
        "fast_refine_n_init": int(config.get("fast_refine_n_init", 3)),
        "fast_refine_top_k": int(config.get("fast_refine_top_k", 2)),
        "fast_stability_target": float(
            config.get("fast_stability_target", 0.85)
        ),
        "fast_m_values": tuple(
            float(value)
            for value in config.get(
                "fast_m_values", (2.0, 1.8, 1.6, 1.4)
            )
        ),
    }


def _visualization_fit_parameters(config: dict[str, Any]) -> dict[str, Any]:
    auto_select = bool(
        config.get(
            "visual_pca_components_auto",
            config.get("visual_pca_components") is None,
        )
    )
    return {
        "seed": int(config["seed"]),
        "pca_components": (
            None
            if auto_select
            else int(config["visual_pca_components"])
        ),
        "n_neighbors": int(config["visual_n_neighbors"]),
        "min_dist": float(config["visual_min_dist"]),
        "metric": str(config["visual_metric"]),
        "spread": float(config["visual_spread"]),
        "densmap": bool(config["visual_densmap"]),
        "cluster_target_weight": float(config["visual_cluster_target_weight"]),
    }


def _initialize_update_config(config: dict[str, Any]) -> dict[str, Any]:
    initialized = dict(config)
    legacy_policy = (
        initialized.get("recluster_trigger_policy")
        != RECLUSTER_TRIGGER_POLICY
    )
    threshold = _validate_noise_threshold(
        initialized.get("noise_threshold", DEFAULT_NOISE_THRESHOLD)
    )
    initialized["noise_threshold"] = threshold
    if legacy_policy:
        # Preserve the immediate per-batch behavior of v1-v4 states. Newly
        # fitted states use the stabilized defaults from _cluster_config.
        initialized.setdefault("noise_release_threshold", threshold)
        initialized.setdefault("noise_release_threshold_auto", False)
        initialized.setdefault("drift_min_samples", 1)
        initialized.setdefault("drift_ewma_alpha", 1.0)
        initialized.setdefault("recluster_cooldown_updates", 0)
        initialized.setdefault("selective_membership_refresh", False)
    else:
        initialized.setdefault(
            "noise_release_threshold",
            threshold * DEFAULT_NOISE_RELEASE_RATIO,
        )
        initialized.setdefault("noise_release_threshold_auto", True)
        initialized.setdefault("drift_min_samples", DEFAULT_DRIFT_MIN_SAMPLES)
        initialized.setdefault("drift_ewma_alpha", DEFAULT_DRIFT_EWMA_ALPHA)
        initialized.setdefault(
            "recluster_cooldown_updates",
            DEFAULT_RECLUSTER_COOLDOWN_UPDATES,
        )
        initialized.setdefault("selective_membership_refresh", True)
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
        "drift_pending_samples": 0,
        "drift_pending_natural_noise": 0,
        "drift_ewma_noise_ratio": None,
        "drift_alarm_active": False,
        "recluster_cooldown_remaining": 0,
        "membership_refresh_min_center_movement": (
            DEFAULT_MEMBERSHIP_REFRESH_MIN_CENTER_MOVEMENT
        ),
        "membership_refresh_min_influence": (
            DEFAULT_MEMBERSHIP_REFRESH_MIN_INFLUENCE
        ),
    }
    for key, value in defaults.items():
        initialized.setdefault(key, value)
    (
        initialized["noise_threshold"],
        initialized["noise_release_threshold"],
        initialized["drift_min_samples"],
        initialized["drift_ewma_alpha"],
        initialized["recluster_cooldown_updates"],
    ) = _validate_drift_settings(
        noise_threshold=initialized["noise_threshold"],
        noise_release_threshold=initialized["noise_release_threshold"],
        drift_min_samples=initialized["drift_min_samples"],
        drift_ewma_alpha=initialized["drift_ewma_alpha"],
        recluster_cooldown_updates=initialized[
            "recluster_cooldown_updates"
        ],
    )
    initialized["recluster_trigger_policy"] = RECLUSTER_TRIGGER_POLICY
    return initialized


def _evaluate_noise_drift(
    config: dict[str, Any],
    *,
    natural_noise_count: int,
    sample_count: int,
) -> dict[str, Any]:
    """Accumulate small batches and update the persisted EWMA alarm state."""

    pending_samples = int(config["drift_pending_samples"]) + int(sample_count)
    pending_noise = int(config["drift_pending_natural_noise"]) + int(
        natural_noise_count
    )
    config["drift_pending_samples"] = pending_samples
    config["drift_pending_natural_noise"] = pending_noise
    evaluated = pending_samples >= int(config["drift_min_samples"])
    observed_ratio: float | None = None
    if evaluated:
        observed_ratio = float(pending_noise / pending_samples)
        previous_ewma = config.get("drift_ewma_noise_ratio")
        if previous_ewma is None or not np.isfinite(float(previous_ewma)):
            smoothed_ratio = observed_ratio
        else:
            alpha = float(config["drift_ewma_alpha"])
            smoothed_ratio = (
                alpha * observed_ratio
                + (1.0 - alpha) * float(previous_ewma)
            )
        alarm_active = bool(config["drift_alarm_active"])
        if alarm_active:
            if smoothed_ratio <= float(config["noise_release_threshold"]):
                alarm_active = False
        elif smoothed_ratio > float(config["noise_threshold"]):
            alarm_active = True
        config["drift_ewma_noise_ratio"] = float(smoothed_ratio)
        config["drift_alarm_active"] = alarm_active
        config["drift_pending_samples"] = 0
        config["drift_pending_natural_noise"] = 0
    else:
        previous_ewma = config.get("drift_ewma_noise_ratio")
        smoothed_ratio = (
            None
            if previous_ewma is None
            else float(previous_ewma)
        )
        alarm_active = bool(config["drift_alarm_active"])
    return {
        "evaluated": evaluated,
        "evaluation_samples": pending_samples if evaluated else 0,
        "observed_ratio": observed_ratio,
        "smoothed_ratio": smoothed_ratio,
        "alarm_active": alarm_active,
        "pending_samples": int(config["drift_pending_samples"]),
    }


def _center_movement_diagnostics(
    previous: HierarchicalModel,
    current: HierarchicalModel,
) -> dict[str, float | int]:
    movements: list[np.ndarray] = []
    for path, previous_node in previous.nodes.items():
        current_node = current.nodes.get(path)
        if current_node is None or current_node.centers.shape != previous_node.centers.shape:
            continue
        movements.append(
            np.linalg.norm(current_node.centers - previous_node.centers, axis=1)
        )
    if not movements:
        return {
            "center_movement_mean": 0.0,
            "center_movement_max": 0.0,
            "center_movement_cluster_count": 0,
        }
    combined = np.concatenate(movements)
    return {
        "center_movement_mean": float(np.mean(combined)),
        "center_movement_max": float(np.max(combined)),
        "center_movement_cluster_count": int(len(combined)),
    }


def _snapshot_hierarchy_centers(
    hierarchy_model: HierarchicalModel,
) -> dict[str, np.ndarray]:
    return {
        path: node.centers.copy()
        for path, node in hierarchy_model.nodes.items()
    }


def _select_center_affected_ids(
    identifiers: list[Any],
    center_contributions: dict[Any, dict[str, Any]],
    reference_centers: dict[str, np.ndarray],
    hierarchy_model: HierarchicalModel,
    *,
    always_include: set[Any],
    min_center_movement: float,
    min_influence: float,
) -> tuple[list[Any], set[str], dict[str, Any]]:
    """Select notes whose stored fuzzy weights amplify moved centers."""

    moved_by_path: dict[str, np.ndarray] = {}
    moved_cluster_count = 0
    maximum_reference_movement = 0.0
    for path, node in hierarchy_model.nodes.items():
        reference = reference_centers.get(path)
        if reference is None or reference.shape != node.centers.shape:
            movements = np.full(node.centers.shape[0], np.inf)
        else:
            movements = np.linalg.norm(node.centers - reference, axis=1)
        finite_movements = movements[np.isfinite(movements)]
        if finite_movements.size:
            maximum_reference_movement = max(
                maximum_reference_movement,
                float(np.max(finite_movements)),
            )
        moved = movements >= float(min_center_movement)
        if np.any(moved):
            moved_by_path[path] = movements
            moved_cluster_count += int(np.sum(moved))

    selected: list[Any] = []
    for identifier in identifiers:
        if identifier in always_include:
            selected.append(identifier)
            continue
        contribution = center_contributions.get(identifier)
        if contribution is None:
            selected.append(identifier)
            continue
        for path, movements in moved_by_path.items():
            stored_weights = contribution["weights_by_path"].get(path)
            if stored_weights is None:
                continue
            weights = np.asarray(stored_weights, dtype=np.float64)
            if weights.shape != movements.shape:
                selected.append(identifier)
                break
            finite_movements = np.where(np.isfinite(movements), movements, np.inf)
            influence = float(np.max(weights * finite_movements))
            if influence >= float(min_influence):
                selected.append(identifier)
                break

    return selected, set(moved_by_path), {
        "affected_center_node_count": int(len(moved_by_path)),
        "affected_center_cluster_count": int(moved_cluster_count),
        "max_center_movement_since_membership_refresh": float(
            maximum_reference_movement
        ),
    }


def _select_embedding_rows_by_ids(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    identifiers: list[Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    positions = {
        identifier: index
        for index, identifier in enumerate(metadata["id"].tolist())
    }
    try:
        selected = [positions[identifier] for identifier in identifiers]
    except KeyError as error:
        raise ValueError("Metadata does not contain every selected ID") from error
    return (
        np.asarray(embeddings, dtype=np.float64)[selected].copy(),
        metadata.iloc[selected].reset_index(drop=True).copy(),
    )


def _refresh_distance_thresholds_from_contributions(
    hierarchy_model: HierarchicalModel,
    center_contributions: dict[Any, dict[str, Any]],
    affected_paths: set[str],
    *,
    distance_z: float,
) -> HierarchicalModel:
    """Refresh moved-node cutoffs without recomputing any memberships."""

    updated_model = copy.deepcopy(hierarchy_model)
    for path in affected_paths:
        node = updated_model.nodes.get(path)
        if node is None:
            continue
        projected_rows: list[np.ndarray] = []
        stored_weight_rows: list[np.ndarray] = []
        for contribution in center_contributions.values():
            stored_weights = contribution["weights_by_path"].get(path)
            if stored_weights is None:
                continue
            projected_rows.append(
                np.asarray(contribution["projected"], dtype=np.float64)
            )
            stored_weight_rows.append(
                np.asarray(stored_weights, dtype=np.float64)
            )
        if not projected_rows:
            continue
        projected = np.vstack(projected_rows)
        weights = np.vstack(stored_weight_rows)
        if weights.shape[1] != node.centers.shape[0]:
            raise ValueError(
                f"Center contribution shape mismatch at node: {path}"
            )
        labels = weights.argmax(axis=1)
        distances = np.linalg.norm(
            projected[:, None, :] - node.centers[None, :, :],
            axis=2,
        )
        thresholds = node.distance_thresholds.copy()
        for cluster_id in range(node.centers.shape[0]):
            cluster_distances = distances[
                labels == cluster_id,
                cluster_id,
            ]
            if cluster_distances.size < 4:
                continue
            median = float(np.median(cluster_distances))
            mad = float(np.median(np.abs(cluster_distances - median)))
            thresholds[cluster_id] = (
                np.inf
                if mad <= 1e-12
                else median + distance_z * 1.4826 * mad
            )
        node.distance_thresholds = thresholds
    return updated_model


def _hierarchy_xb_from_contributions(
    hierarchy_model: HierarchicalModel,
    center_contributions: dict[Any, dict[str, Any]],
) -> float:
    """Estimate hierarchy XB from persisted weights, without membership work."""

    weighted_xb_sum = 0.0
    evaluated_samples = 0
    for path, node in hierarchy_model.nodes.items():
        projected_rows: list[np.ndarray] = []
        stored_weight_rows: list[np.ndarray] = []
        for contribution in center_contributions.values():
            stored_weights = contribution["weights_by_path"].get(path)
            if stored_weights is None:
                continue
            projected_rows.append(
                np.asarray(contribution["projected"], dtype=np.float64)
            )
            stored_weight_rows.append(
                np.asarray(stored_weights, dtype=np.float64)
            )
        if not projected_rows or node.centers.shape[0] < 2:
            continue
        projected = np.vstack(projected_rows)
        weights = np.vstack(stored_weight_rows)
        if weights.shape[1] != node.centers.shape[0]:
            raise ValueError(
                f"Center contribution shape mismatch at node: {path}"
            )
        distances = np.linalg.norm(
            projected[:, None, :] - node.centers[None, :, :],
            axis=2,
        )
        center_differences = (
            node.centers[:, None, :] - node.centers[None, :, :]
        )
        separation_squared = np.sum(center_differences**2, axis=2)
        np.fill_diagonal(separation_squared, np.inf)
        minimum_separation_squared = float(np.min(separation_squared))
        if not np.isfinite(minimum_separation_squared):
            continue
        sample_count = len(projected)
        numerator = float(np.sum(weights * (distances**2)))
        node_xb = numerator / max(
            sample_count * minimum_separation_squared,
            1e-12,
        )
        weighted_xb_sum += node_xb * sample_count
        evaluated_samples += sample_count
    if evaluated_samples == 0:
        return float("nan")
    return float(weighted_xb_sum / evaluated_samples)


def _cluster_occupancy_change(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> float:
    previous_distribution = previous["cluster_path"].fillna("noise").value_counts(
        normalize=True
    )
    current_distribution = current["cluster_path"].fillna("noise").value_counts(
        normalize=True
    )
    labels = previous_distribution.index.union(current_distribution.index)
    return float(
        0.5
        * np.abs(
            previous_distribution.reindex(labels, fill_value=0.0)
            - current_distribution.reindex(labels, fill_value=0.0)
        ).sum()
    )


def _assignment_change_rate(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[float, int]:
    previous_by_id = previous.set_index("id")
    current_by_id = current.set_index("id")
    shared_ids = previous_by_id.index.intersection(current_by_id.index)
    if len(shared_ids) == 0:
        return 0.0, 0
    previous_paths = previous_by_id.loc[shared_ids, "cluster_path"].fillna("noise")
    current_paths = current_by_id.loc[shared_ids, "cluster_path"].fillna("noise")
    previous_noise = previous_by_id.loc[shared_ids, "is_noise"].astype(bool)
    current_noise = current_by_id.loc[shared_ids, "is_noise"].astype(bool)
    changed = (previous_paths.to_numpy() != current_paths.to_numpy()) | (
        previous_noise.to_numpy() != current_noise.to_numpy()
    )
    return float(np.mean(changed)), int(len(shared_ids))


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
    config["pca_components_selected"] = int(result.model.pca.n_components_)
    result_config = result.tree.get("config", {})
    if result_config.get("pca_selection") is not None:
        config["pca_selection"] = result_config["pca_selection"]
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
    pca_components: int | None = None,
    pca_max_components: int = DEFAULT_MAX_COMPONENTS,
    pca_min_components: int = DEFAULT_MIN_COMPONENTS,
    pca_component_step: int = DEFAULT_COMPONENT_STEP,
    pca_k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    pca_minimum_preservation_gain: float = DEFAULT_MINIMUM_PRESERVATION_GAIN,
    seed: int = 42,
    noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
    noise_release_threshold: float | None = None,
    drift_min_samples: int = DEFAULT_DRIFT_MIN_SAMPLES,
    drift_ewma_alpha: float = DEFAULT_DRIFT_EWMA_ALPHA,
    recluster_cooldown_updates: int = DEFAULT_RECLUSTER_COOLDOWN_UPDATES,
    visual_pca_components: int | None = None,
    visual_cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
    visual_n_neighbors: int = 15,
    visual_min_dist: float = 0.02,
    visual_metric: str = "cosine",
    visual_spread: float = 0.85,
    visual_densmap: bool = False,
    center_updates_before_membership_refresh: int = (
        DEFAULT_CENTER_UPDATES_BEFORE_MEMBERSHIP_REFRESH
    ),
    selective_membership_refresh: bool = True,
    membership_refresh_min_center_movement: float = (
        DEFAULT_MEMBERSHIP_REFRESH_MIN_CENTER_MOVEMENT
    ),
    membership_refresh_min_influence: float = (
        DEFAULT_MEMBERSHIP_REFRESH_MIN_INFLUENCE
    ),
    max_xb_relative_degradation: float = DEFAULT_MAX_XB_RELATIVE_DEGRADATION,
    fuzzifier: float = 2.0,
    max_fcm_iter: int = 200,
    fcm_tol: float = 1e-6,
    fast_mode: bool = False,
    fast_sample_size: int = 1000,
    fast_scout_n_init: int = 2,
    fast_refine_n_init: int = 3,
    fast_refine_top_k: int = 2,
    fast_stability_target: float = 0.85,
    fast_m_values: tuple[float, ...] = (2.0, 1.8, 1.6, 1.4),
    fit_visualization: bool = True,
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
        pca_max_components=pca_max_components,
        pca_min_components=pca_min_components,
        pca_component_step=pca_component_step,
        pca_k_values=pca_k_values,
        pca_minimum_preservation_gain=pca_minimum_preservation_gain,
        seed=seed,
        noise_threshold=noise_threshold,
        noise_release_threshold=noise_release_threshold,
        drift_min_samples=drift_min_samples,
        drift_ewma_alpha=drift_ewma_alpha,
        recluster_cooldown_updates=recluster_cooldown_updates,
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
        selective_membership_refresh=selective_membership_refresh,
        membership_refresh_min_center_movement=(
            membership_refresh_min_center_movement
        ),
        membership_refresh_min_influence=membership_refresh_min_influence,
        max_xb_relative_degradation=max_xb_relative_degradation,
        fuzzifier=fuzzifier,
        max_fcm_iter=max_fcm_iter,
        fcm_tol=fcm_tol,
        fast_mode=fast_mode,
        fast_sample_size=fast_sample_size,
        fast_scout_n_init=fast_scout_n_init,
        fast_refine_n_init=fast_refine_n_init,
        fast_refine_top_k=fast_refine_top_k,
        fast_stability_target=fast_stability_target,
        fast_m_values=fast_m_values,
    )
    config["visualization_deferred"] = not fit_visualization
    hierarchy_model, tree, assignments = _fit_hierarchy(values, frame, config)
    baseline_xie_beni = _hierarchy_xb(values, hierarchy_model, config)
    config["baseline_xie_beni"] = baseline_xie_beni
    config["current_xie_beni"] = baseline_xie_beni
    if fit_visualization:
        visual_pca, visual_reducer, coordinates = _fit_visualization(
            values,
            assignments,
            config,
        )
        config["visual_pca_components_selected"] = int(
            getattr(visual_pca, "n_components_", DEFAULT_VISUAL_PCA_COMPONENTS)
        )
        if config.get("visual_pca_components_auto", False):
            config["visual_pca_components"] = config["visual_pca_components_selected"]
    else:
        visual_pca = None
        visual_reducer = None
        coordinates = np.zeros((len(values), 2), dtype=np.float64)
        config["visual_pca_components_selected"] = None
    center_contributions = _center_contributions_for_batch(
        values,
        frame,
        hierarchy_model,
        **_fuzzy_parameters(config),
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
        center_statistics=_aggregate_center_contributions(center_contributions),
        center_contributions=center_contributions,
        membership_reference_centers=_snapshot_hierarchy_centers(
            hierarchy_model
        ),
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


def _merge_state_rows_by_id(
    existing_embeddings: np.ndarray,
    existing_metadata: pd.DataFrame,
    existing_coordinates: np.ndarray,
    incoming_embeddings: np.ndarray,
    incoming_metadata: pd.DataFrame,
    incoming_coordinates: np.ndarray,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    list[Any],
    list[Any],
]:
    """Replace existing rows by ID and append IDs not seen in the state."""

    existing_frame = _validate_metadata(
        existing_metadata,
        len(existing_embeddings),
    )
    incoming_frame = _validate_metadata(
        incoming_metadata,
        len(incoming_embeddings),
    )
    if existing_coordinates.shape != (len(existing_embeddings), 2):
        raise ValueError("State coordinates must have shape (samples, 2)")
    if incoming_coordinates.shape != (len(incoming_embeddings), 2):
        raise ValueError("Incoming coordinates must have shape (samples, 2)")

    existing_ids = existing_frame["id"].tolist()
    incoming_ids = incoming_frame["id"].tolist()
    existing_positions = {
        identifier: index for index, identifier in enumerate(existing_ids)
    }
    incoming_positions = {
        identifier: index for index, identifier in enumerate(incoming_ids)
    }
    replaced_ids = [
        identifier
        for identifier in incoming_ids
        if identifier in existing_positions
    ]
    appended_ids = [
        identifier
        for identifier in incoming_ids
        if identifier not in existing_positions
    ]

    merged_embeddings = np.asarray(existing_embeddings, dtype=np.float64).copy()
    merged_coordinates = np.asarray(existing_coordinates, dtype=np.float64).copy()
    for identifier in replaced_ids:
        existing_index = existing_positions[identifier]
        incoming_index = incoming_positions[identifier]
        merged_embeddings[existing_index] = incoming_embeddings[incoming_index]
        merged_coordinates[existing_index] = incoming_coordinates[incoming_index]

    if appended_ids:
        appended_indices = [
            incoming_positions[identifier] for identifier in appended_ids
        ]
        merged_embeddings = np.vstack(
            [merged_embeddings, incoming_embeddings[appended_indices]]
        )
        merged_coordinates = np.vstack(
            [merged_coordinates, incoming_coordinates[appended_indices]]
        )

    columns = list(dict.fromkeys([*existing_frame.columns, *incoming_frame.columns]))
    existing_aligned = existing_frame.reindex(columns=columns).astype(object)
    incoming_aligned = incoming_frame.reindex(columns=columns).astype(object)
    merged_metadata = existing_aligned.copy()
    for identifier in replaced_ids:
        existing_index = existing_positions[identifier]
        incoming_index = incoming_positions[identifier]
        merged_metadata.iloc[existing_index, :] = incoming_aligned.iloc[
            incoming_index
        ].to_numpy()
    if appended_ids:
        appended_indices = [
            incoming_positions[identifier] for identifier in appended_ids
        ]
        merged_metadata = pd.concat(
            [merged_metadata, incoming_aligned.iloc[appended_indices]],
            ignore_index=True,
        )
    return (
        merged_embeddings,
        merged_metadata.reset_index(drop=True),
        merged_coordinates,
        replaced_ids,
        appended_ids,
    )


def _merge_assignments_by_id(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    """Replace assignments by ID and append assignments for new documents."""

    if "id" not in existing.columns or "id" not in incoming.columns:
        raise ValueError("Assignments must contain an 'id' column")
    existing_ids = existing["id"].tolist()
    incoming_ids = incoming["id"].tolist()
    existing_positions = {
        identifier: index for index, identifier in enumerate(existing_ids)
    }
    incoming_positions = {
        identifier: index for index, identifier in enumerate(incoming_ids)
    }
    columns = list(dict.fromkeys([*existing.columns, *incoming.columns]))
    existing_aligned = (
        existing.reindex(columns=columns).astype(object).reset_index(drop=True)
    )
    incoming_aligned = (
        incoming.reindex(columns=columns).astype(object).reset_index(drop=True)
    )
    merged = existing_aligned.copy()
    replaced_ids = [
        identifier
        for identifier in incoming_ids
        if identifier in existing_positions
    ]
    for identifier in replaced_ids:
        merged.iloc[existing_positions[identifier], :] = incoming_aligned.iloc[
            incoming_positions[identifier]
        ].to_numpy()
    appended_ids = [
        identifier
        for identifier in incoming_ids
        if identifier not in existing_positions
    ]
    if appended_ids:
        appended_indices = [incoming_positions[identifier] for identifier in appended_ids]
        merged = pd.concat(
            [merged, incoming_aligned.iloc[appended_indices]],
            ignore_index=True,
        )
    return merged.reset_index(drop=True)


def _select_assignments_by_ids(
    assignments: pd.DataFrame,
    identifiers: list[Any],
) -> pd.DataFrame:
    """Return assignments in the same order as an incoming batch."""

    positions = {
        identifier: index
        for index, identifier in enumerate(assignments["id"].tolist())
    }
    try:
        selected = [positions[identifier] for identifier in identifiers]
    except KeyError as error:
        raise ValueError("Assignments do not contain every incoming ID") from error
    return assignments.iloc[selected].reset_index(drop=True)


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
        node["boundary_count"] = 0
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
    noise_release_threshold: float | None = None,
) -> tuple[IncrementalClusterState, dict[str, Any]]:
    """Update centers, memberships, and models on independent schedules."""

    values = _validate_embeddings(embeddings)
    frame = _validate_metadata(metadata, len(values))
    if values.shape[1] != state.embeddings.shape[1]:
        raise ValueError("new embeddings have a different dimensionality")
    if state.visual_pca is None or state.visual_reducer is None:
        raise ValueError(
            "This state deferred visualization; fit a final state without "
            "--skip-visualization before incremental updates."
        )

    existing_ids = set(state.metadata["id"].tolist())
    incoming_ids = frame["id"].tolist()
    replaced_ids = [
        identifier for identifier in incoming_ids if identifier in existing_ids
    ]
    appended_ids = [
        identifier for identifier in incoming_ids if identifier not in existing_ids
    ]

    config = _initialize_update_config(state.config)
    threshold = _validate_noise_threshold(
        config["noise_threshold"] if noise_threshold is None else noise_threshold
    )
    if noise_release_threshold is not None:
        release_threshold = _validate_noise_threshold(
            noise_release_threshold
        )
        config["noise_release_threshold_auto"] = False
    elif bool(config.get("noise_release_threshold_auto", False)):
        release_threshold = threshold * DEFAULT_NOISE_RELEASE_RATIO
    else:
        release_threshold = min(
            float(config["noise_release_threshold"]),
            threshold,
        )
    _validate_drift_settings(
        noise_threshold=threshold,
        noise_release_threshold=release_threshold,
        drift_min_samples=config["drift_min_samples"],
        drift_ewma_alpha=config["drift_ewma_alpha"],
        recluster_cooldown_updates=config["recluster_cooldown_updates"],
    )
    config["noise_threshold"] = threshold
    config["noise_release_threshold"] = release_threshold
    membership_reference_centers = getattr(
        state,
        "membership_reference_centers",
        None,
    )
    if not membership_reference_centers:
        membership_reference_centers = _snapshot_hierarchy_centers(
            state.hierarchy_model
        )
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
    natural_noise_count = int(new_assignments["is_natural_noise"].sum())
    drift = _evaluate_noise_drift(
        config,
        natural_noise_count=natural_noise_count,
        sample_count=len(values),
    )
    cooldown_remaining_before_update = int(
        config["recluster_cooldown_remaining"]
    )
    cooldown_active = cooldown_remaining_before_update > 0
    emergency_recluster_requested = bool(
        drift["evaluated"] and drift["alarm_active"]
    )
    emergency_recluster = emergency_recluster_requested and not cooldown_active
    new_coordinates = transform_projection(
        values,
        pca=state.visual_pca,
        reducer=state.visual_reducer,
    )

    (
        combined_embeddings,
        combined_metadata,
        combined_coordinates,
        _merged_replaced_ids,
        _merged_appended_ids,
    ) = _merge_state_rows_by_id(
        state.embeddings,
        state.metadata,
        state.coordinates,
        values,
        frame,
        new_coordinates,
    )
    if (
        _merged_replaced_ids != replaced_ids
        or _merged_appended_ids != appended_ids
    ):
        raise RuntimeError("State row merge did not preserve incoming ID order")
    config["update_count"] = int(config.get("update_count", 0)) + 1

    stored_contributions = getattr(state, "center_contributions", None)
    if not stored_contributions:
        stored_contributions = _center_contributions_for_batch(
            state.embeddings,
            state.metadata,
            state.hierarchy_model,
            **_fuzzy_parameters(config),
        )
        center_statistics = _aggregate_center_contributions(stored_contributions)
    else:
        if config.get("center_contribution_format") != CENTER_CONTRIBUTION_FORMAT:
            stored_contributions = _compact_center_contributions(
                stored_contributions
            )
        center_statistics = (
            _copy_center_statistics(state.center_statistics)
            if state.center_statistics
            else _aggregate_center_contributions(stored_contributions)
        )
    center_contributions = dict(stored_contributions)
    incoming_contributions = _center_contributions_for_batch(
        values,
        frame,
        state.hierarchy_model,
        **_fuzzy_parameters(config),
    )
    for identifier in incoming_ids:
        previous = center_contributions.get(identifier)
        if previous is not None:
            _apply_center_contribution_delta(
                center_statistics,
                previous,
                sign=-1.0,
            )
        replacement = incoming_contributions[identifier]
        _apply_center_contribution_delta(
            center_statistics,
            replacement,
            sign=1.0,
        )
        center_contributions[identifier] = replacement
    config["center_contribution_format"] = CENTER_CONTRIBUTION_FORMAT
    hierarchy_model, updated_node_count = _update_hierarchy_centers_from_statistics(
        state.hierarchy_model,
        center_statistics,
    )
    center_movement = _center_movement_diagnostics(
        state.hierarchy_model,
        hierarchy_model,
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
    membership_refresh_scope = "none"
    membership_refresh_sample_count = 0
    membership_refresh_skipped_count = int(len(combined_embeddings))
    membership_selection_diagnostics = {
        "affected_center_node_count": 0,
        "affected_center_cluster_count": 0,
        "max_center_movement_since_membership_refresh": 0.0,
    }
    current_xie_beni = float(config["current_xie_beni"])
    xb_relative_degradation: float | None = None
    xb_degradation_detected = False
    if membership_refreshed:
        if bool(config["selective_membership_refresh"]):
            membership_refresh_scope = "selective"
            combined_ids = combined_metadata["id"].tolist()
            (
                refresh_ids,
                affected_paths,
                membership_selection_diagnostics,
            ) = _select_center_affected_ids(
                combined_ids,
                center_contributions,
                membership_reference_centers,
                hierarchy_model,
                always_include=set(incoming_ids),
                min_center_movement=float(
                    config["membership_refresh_min_center_movement"]
                ),
                min_influence=float(
                    config["membership_refresh_min_influence"]
                ),
            )
            membership_refresh_sample_count = len(refresh_ids)
            membership_refresh_skipped_count = (
                len(combined_embeddings) - len(refresh_ids)
            )
            hierarchy_model = _refresh_distance_thresholds_from_contributions(
                hierarchy_model,
                center_contributions,
                affected_paths,
                distance_z=float(config["distance_z"]),
            )
            refresh_embeddings, refresh_metadata = _select_embedding_rows_by_ids(
                combined_embeddings,
                combined_metadata,
                refresh_ids,
            )
            refreshed_subset, _ = assign_to_hierarchy(
                refresh_embeddings,
                refresh_metadata,
                hierarchy_model,
                **_assignment_parameters(
                    config,
                    forced_noise_ratio=0.0,
                ),
            )
            refreshed_assignments = _merge_assignments_by_id(
                state.assignments,
                new_assignments,
            )
            refreshed_assignments = _merge_assignments_by_id(
                refreshed_assignments,
                refreshed_subset,
            )
            refreshed_assignments = _apply_global_forced_noise(
                refreshed_assignments,
                forced_noise_ratio=forced_noise_ratio,
            )
            refreshed_contributions = _center_contributions_for_batch(
                refresh_embeddings,
                refresh_metadata,
                hierarchy_model,
                **_fuzzy_parameters(config),
            )
            for identifier in refresh_ids:
                previous = center_contributions.get(identifier)
                if previous is not None:
                    _apply_center_contribution_delta(
                        center_statistics,
                        previous,
                        sign=-1.0,
                    )
                replacement = refreshed_contributions[identifier]
                _apply_center_contribution_delta(
                    center_statistics,
                    replacement,
                    sign=1.0,
                )
                center_contributions[identifier] = replacement
            hierarchy_model, _ = _update_hierarchy_centers_from_statistics(
                hierarchy_model,
                center_statistics,
            )
            current_xie_beni = _hierarchy_xb_from_contributions(
                hierarchy_model,
                center_contributions,
            )
        else:
            membership_refresh_scope = "full_legacy"
            membership_refresh_sample_count = len(combined_embeddings)
            membership_refresh_skipped_count = 0
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
            center_contributions = _center_contributions_for_batch(
                combined_embeddings,
                combined_metadata,
                hierarchy_model,
                **_fuzzy_parameters(config),
            )
            center_statistics = _aggregate_center_contributions(
                center_contributions
            )
            current_xie_beni = _hierarchy_xb(
                combined_embeddings,
                hierarchy_model,
                config,
            )
        if refreshed_assignments is None:
            raise RuntimeError("Membership refresh results are unavailable")
        refreshed_tree = _rebuild_tree_counts(
            state.tree,
            refreshed_assignments,
            int(config["update_count"]),
        )
        membership_reference_centers = _snapshot_hierarchy_centers(
            hierarchy_model
        )
        config["current_xie_beni"] = current_xie_beni
        if np.isfinite(baseline_xie_beni) and np.isfinite(current_xie_beni):
            xb_relative_degradation = float(
                (current_xie_beni - baseline_xie_beni)
                / max(abs(baseline_xie_beni), 1e-12)
            )
            xb_degradation_detected = (
                xb_relative_degradation
                >= float(config["max_xb_relative_degradation"])
            )

    xb_degradation_recluster = (
        xb_degradation_detected and not cooldown_active
    )
    recluster_suppressed_by_cooldown = bool(
        cooldown_active
        and (emergency_recluster_requested or xb_degradation_detected)
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
        center_contributions = _center_contributions_for_batch(
            combined_embeddings,
            combined_metadata,
            hierarchy_model,
            **_fuzzy_parameters(config),
        )
        center_statistics = _aggregate_center_contributions(center_contributions)
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
        config["recluster_cooldown_remaining"] = int(
            config["recluster_cooldown_updates"]
        )
        membership_reference_centers = _snapshot_hierarchy_centers(
            hierarchy_model
        )
        reclustered = True
    elif membership_refreshed:
        if refreshed_assignments is None or refreshed_tree is None:
            raise RuntimeError("Membership refresh results are unavailable")
        assignments = refreshed_assignments
        tree = refreshed_tree
        reclustered = False
    else:
        assignments = _merge_assignments_by_id(state.assignments, new_assignments)
        assignments = _apply_global_forced_noise(
            assignments,
            forced_noise_ratio=forced_noise_ratio,
        )
        effective_new_assignments = _select_assignments_by_ids(
            assignments,
            incoming_ids,
        )
        if replaced_ids:
            tree = _rebuild_tree_counts(
                state.tree,
                assignments,
                int(config["update_count"]),
            )
        else:
            tree = _refresh_tree_after_append(
                state.tree,
                effective_new_assignments,
                assignments,
                int(config["update_count"]),
            )
        reclustered = False

    if not should_recluster and cooldown_active:
        config["recluster_cooldown_remaining"] = max(
            cooldown_remaining_before_update - 1,
            0,
        )

    effective_new_assignments = _select_assignments_by_ids(
        assignments,
        incoming_ids,
    )
    new_noise_ratio = float(effective_new_assignments["is_noise"].mean())
    occupancy_change = _cluster_occupancy_change(
        state.assignments,
        assignments,
    )
    assignment_change_rate, compared_assignment_count = _assignment_change_rate(
        state.assignments,
        assignments,
    )
    config["last_update_noise_ratio"] = float(new_noise_ratio)
    config["last_update_natural_noise_ratio"] = float(natural_noise_ratio)
    config["last_update_reclustered"] = reclustered
    config["last_update_membership_refreshed"] = membership_refreshed
    config["last_update_visualization_refitted"] = visualization_refitted
    config["last_xb_relative_degradation"] = xb_relative_degradation
    config["last_update_xb_degradation_recluster"] = xb_degradation_recluster
    config["last_update_drift_evaluated"] = bool(drift["evaluated"])
    config["last_update_emergency_recluster"] = emergency_recluster
    config["last_center_movement_mean"] = center_movement[
        "center_movement_mean"
    ]
    config["last_center_movement_max"] = center_movement[
        "center_movement_max"
    ]
    config["last_cluster_occupancy_change"] = occupancy_change
    config["last_assignment_change_rate"] = assignment_change_rate
    config["last_membership_refresh_scope"] = membership_refresh_scope
    config["last_membership_refresh_sample_count"] = int(
        membership_refresh_sample_count
    )
    config["last_membership_refresh_skipped_count"] = int(
        membership_refresh_skipped_count
    )
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
            "membership_refresh_scope": membership_refresh_scope,
            "membership_refresh_sample_count": int(
                membership_refresh_sample_count
            ),
            "baseline_xie_beni": float(config["baseline_xie_beni"]),
            "current_xie_beni": float(config["current_xie_beni"]),
            "noise_threshold": float(config["noise_threshold"]),
            "noise_release_threshold": float(
                config["noise_release_threshold"]
            ),
            "drift_ewma_noise_ratio": config["drift_ewma_noise_ratio"],
            "drift_alarm_active": bool(config["drift_alarm_active"]),
            "recluster_cooldown_remaining": int(
                config["recluster_cooldown_remaining"]
            ),
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
        center_contributions=center_contributions,
        membership_reference_centers=membership_reference_centers,
    )
    summary = {
        "new_samples": int(len(values)),
        "replaced_samples": int(len(replaced_ids)),
        "appended_samples": int(len(appended_ids)),
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
        "noise_release_threshold": float(release_threshold),
        "drift_evaluated": bool(drift["evaluated"]),
        "drift_evaluation_samples": int(drift["evaluation_samples"]),
        "drift_observed_noise_ratio": drift["observed_ratio"],
        "drift_smoothed_noise_ratio": drift["smoothed_ratio"],
        "drift_alarm_active": bool(drift["alarm_active"]),
        "drift_pending_samples": int(drift["pending_samples"]),
        "center_updated": updated_node_count > 0,
        "updated_center_nodes": int(updated_node_count),
        **center_movement,
        "cluster_occupancy_change": occupancy_change,
        "assignment_change_rate": assignment_change_rate,
        "compared_assignment_count": compared_assignment_count,
        "membership_refreshed": membership_refreshed,
        "membership_refresh_scope": membership_refresh_scope,
        "membership_refresh_sample_count": int(
            membership_refresh_sample_count
        ),
        "membership_refresh_skipped_count": int(
            membership_refresh_skipped_count
        ),
        **membership_selection_diagnostics,
        "xie_beni": float(current_xie_beni),
        "xb_relative_degradation": xb_relative_degradation,
        "xb_degradation_recluster": xb_degradation_recluster,
        "xb_degradation_detected": xb_degradation_detected,
        "emergency_recluster_requested": emergency_recluster_requested,
        "emergency_recluster": emergency_recluster,
        "recluster_suppressed_by_cooldown": (
            recluster_suppressed_by_cooldown
        ),
        "reclustered": reclustered,
        "visualization_refitted": visualization_refitted,
        "center_updates_since_membership_refresh": int(
            config["center_updates_since_membership_refresh"]
        ),
        "membership_refreshes_since_recluster": int(
            config["membership_refreshes_since_recluster"]
        ),
        "recluster_cooldown_remaining": int(
            config["recluster_cooldown_remaining"]
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
    config = _initialize_update_config(state.config)
    center_contributions = getattr(state, "center_contributions", {})
    if center_contributions and (
        config.get("center_contribution_format")
        != CENTER_CONTRIBUTION_FORMAT
    ):
        center_contributions = _compact_center_contributions(
            center_contributions
        )
    config["center_contribution_format"] = CENTER_CONTRIBUTION_FORMAT
    payload = {
        "version": STATE_VERSION,
        "embeddings": state.embeddings,
        "metadata": state.metadata,
        "assignments": state.assignments,
        "coordinates": state.coordinates,
        "hierarchy_model": state.hierarchy_model,
        "tree": state.tree,
        "config": config,
        "visual_pca": state.visual_pca,
        "visual_reducer": state.visual_reducer,
        "center_statistics": state.center_statistics,
        "center_contributions": center_contributions,
        "membership_reference_centers": getattr(
            state,
            "membership_reference_centers",
            {},
        ),
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
    elif isinstance(payload, dict) and payload.get("version") in {
        1,
        2,
        3,
        4,
        5,
        STATE_VERSION,
    }:
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
        fields["center_contributions"] = payload.get("center_contributions", {})
        fields["membership_reference_centers"] = payload.get(
            "membership_reference_centers",
            {},
        )
        state = IncrementalClusterState(**fields)
    else:
        raise ValueError(f"Invalid incremental state: {path}")
    if not hasattr(state, "center_statistics"):
        state.center_statistics = {}
    if not hasattr(state, "center_contributions"):
        state.center_contributions = {}
    if (
        not hasattr(state, "membership_reference_centers")
        or not state.membership_reference_centers
    ):
        state.membership_reference_centers = _snapshot_hierarchy_centers(
            state.hierarchy_model
        )
    state.config = _initialize_update_config(state.config)
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
        if state.visual_pca is None or state.visual_reducer is None:
            raise ValueError(
                "Visualization was deferred; omit --plot-output or run a "
                "final fit without --skip-visualization."
            )
        configured_target_weight = state.config.get("visual_cluster_target_weight")
        configured_pca_components = state.config.get(
            "visual_pca_components",
            DEFAULT_VISUAL_PCA_COMPONENTS,
        )
        make_fixed_coordinate_plot(
            state.coordinates,
            state.assignments,
            plot_output,
            title=title,
            color_by=color_by,
            pca_components=(
                None
                if configured_pca_components is None
                else int(configured_pca_components)
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


def _add_dataset_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-sample-size",
        type=int,
        default=None,
        help=(
            "Randomly select this many documents from the loaded dataset "
            "without replacement before fitting."
        ),
    )
    parser.add_argument(
        "--dataset-sample-seed",
        type=int,
        default=None,
        help="Seed for dataset sampling; defaults to --seed.",
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
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help=(
            "Skip PCA+UMAP during fast iteration. The saved state is for "
            "clustering inspection and cannot process incremental updates "
            "until a visualization model is fitted."
        ),
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
        help=(
            "Legacy forced-noise quota; defaults to 0 so only natural "
            "membership, distance, and PCA-support evidence is used."
        ),
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
        "--fuzzifier",
        type=float,
        default=2.0,
        help="Default FCM fuzzifier m (fast mode can adapt it per node).",
    )
    parser.add_argument("--max-fcm-iter", type=int, default=200)
    parser.add_argument("--fcm-tol", type=float, default=1e-6)
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use sample-based K scouting, adaptive m, dynamic restarts, "
            "and bounded FCM iterations."
        ),
    )
    parser.add_argument("--fast-sample-size", type=int, default=1000)
    parser.add_argument("--fast-scout-n-init", type=int, default=2)
    parser.add_argument("--fast-refine-n-init", type=int, default=3)
    parser.add_argument("--fast-refine-top-k", type=int, default=2)
    parser.add_argument("--fast-stability-target", type=float, default=0.85)
    parser.add_argument(
        "--fast-m",
        type=float,
        nargs="+",
        default=[2.0, 1.8, 1.6, 1.4],
        help="Fuzzifier fallback schedule used by --fast.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "PCA dimensions used for clustering; omit to auto-select using "
            "normalized k-NN preservation."
        ),
    )
    parser.add_argument(
        "--pca-max-components",
        type=int,
        default=DEFAULT_MAX_COMPONENTS,
        help="Maximum PCA width considered by the automatic clustering selector.",
    )
    parser.add_argument(
        "--pca-min-components",
        type=int,
        default=DEFAULT_MIN_COMPONENTS,
        help="Minimum PCA width considered by the automatic clustering selector.",
    )
    parser.add_argument(
        "--pca-component-step",
        type=int,
        default=DEFAULT_COMPONENT_STEP,
        help="PCA width increment used by the automatic clustering selector.",
    )
    parser.add_argument(
        "--pca-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Neighbor counts used by the automatic clustering selector.",
    )
    parser.add_argument(
        "--pca-minimum-preservation-gain",
        type=float,
        default=DEFAULT_MINIMUM_PRESERVATION_GAIN,
        help="Minimum k-NN preservation gain before the selector stops.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=DEFAULT_NOISE_THRESHOLD,
        help="Activate the drift alarm when smoothed natural noise exceeds this value.",
    )
    parser.add_argument(
        "--noise-release-threshold",
        type=float,
        default=None,
        help=(
            "Release the drift alarm below this value; defaults to half of "
            "--noise-threshold."
        ),
    )
    parser.add_argument(
        "--drift-min-samples",
        type=int,
        default=DEFAULT_DRIFT_MIN_SAMPLES,
        help="Accumulate at least this many new samples before drift evaluation.",
    )
    parser.add_argument(
        "--drift-ewma-alpha",
        type=float,
        default=DEFAULT_DRIFT_EWMA_ALPHA,
        help="EWMA weight for the newest natural-noise observation.",
    )
    parser.add_argument(
        "--recluster-cooldown-updates",
        type=int,
        default=DEFAULT_RECLUSTER_COOLDOWN_UPDATES,
        help="Suppress repeated re-clustering for this many updates.",
    )
    parser.add_argument(
        "--visual-pca-components",
        type=int,
        default=None,
        help=(
            "PCA dimensions before the 2D UMAP visualization; omit to "
            "auto-select using UMAP k-NN preservation."
        ),
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
    dataset_total_rows = len(embeddings)
    dataset_sample_seed = (
        args.dataset_sample_seed
        if args.dataset_sample_seed is not None
        else args.seed
    )
    if args.dataset_sample_size is not None:
        embeddings, metadata = sample_embedding_batch(
            embeddings,
            metadata,
            sample_size=args.dataset_sample_size,
            seed=dataset_sample_seed,
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
        pca_max_components=args.pca_max_components,
        pca_min_components=args.pca_min_components,
        pca_component_step=args.pca_component_step,
        pca_k_values=tuple(args.pca_k),
        pca_minimum_preservation_gain=args.pca_minimum_preservation_gain,
        seed=args.seed,
        noise_threshold=args.noise_threshold,
        noise_release_threshold=args.noise_release_threshold,
        drift_min_samples=args.drift_min_samples,
        drift_ewma_alpha=args.drift_ewma_alpha,
        recluster_cooldown_updates=args.recluster_cooldown_updates,
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
        fuzzifier=args.fuzzifier,
        max_fcm_iter=args.max_fcm_iter,
        fcm_tol=args.fcm_tol,
        fast_mode=args.fast,
        fast_sample_size=args.fast_sample_size,
        fast_scout_n_init=args.fast_scout_n_init,
        fast_refine_n_init=args.fast_refine_n_init,
        fast_refine_top_k=args.fast_refine_top_k,
        fast_stability_target=args.fast_stability_target,
        fast_m_values=tuple(args.fast_m),
        fit_visualization=not args.skip_visualization,
    )
    if args.dataset_sample_size is not None:
        state.config["dataset_total_rows"] = int(dataset_total_rows)
        state.config["dataset_sample_size"] = int(args.dataset_sample_size)
        state.config["dataset_sample_seed"] = int(dataset_sample_seed)
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
    sampling_note = ""
    if args.dataset_sample_size is not None:
        sampling_note = (
            f" [random sample {len(embeddings)}/{dataset_total_rows}, "
            f"seed={dataset_sample_seed}]"
        )
    print(
        f"Initial state saved: {args.state_output} "
        f"({len(embeddings)} samples, "
        f"{int(state.assignments['is_noise'].sum())} noise)"
        f"{sampling_note}"
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
        noise_release_threshold=args.noise_release_threshold,
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
    _add_dataset_sampling_args(fit_parser)
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
    update_parser.add_argument(
        "--noise-release-threshold",
        type=float,
        default=None,
    )
    _add_visual_output_args(update_parser)
    update_parser.set_defaults(handler=_run_update)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
