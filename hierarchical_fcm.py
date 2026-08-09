"""Recursive hierarchical PCA + spherical FCM orchestration."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from clustering_types import HierarchicalModel, HierarchicalResult, HierarchyNodeModel
from fcm_core import (
    conditional_memberships_from_projected,
    fit_clustering_pca,
    sfcm_memberships_from_centers,
)
from fcm_document_classification import (
    DEFAULT_FORCED_NOISE_RATIO,
    DEFAULT_MAX_MEMBERSHIP_GAP,
    classify_fcm_documents,
    fcm_document_types,
    fcm_noise_scores,
    merge_forced_noise,
)
from fcm_validity import (
    _filter_fcm_labels,
    _validate_fcm_selection_parameters,
    select_fcm_cluster_count,
)
from fast_fcm import FastFcmConfig, select_fast_fcm_cluster_count
from pca_dimension_search import (
    DEFAULT_K_VALUES,
    DEFAULT_MAX_COMPONENTS,
    DEFAULT_MINIMUM_PRESERVATION_GAIN,
)
from pca_dimension_selection import (
    DEFAULT_COMPONENT_STEP,
    DEFAULT_MIN_COMPONENTS,
)
from pca_projection import (
    calibrate_pca_projection_support_threshold,
    pca_projection_support,
)
from hierarchical_assignments import (
    DOCUMENT_TYPE_BOUNDARY,
    DOCUMENT_TYPE_CORE,
    DOCUMENT_TYPE_NOISE,
    build_hierarchical_assignments,
)


def _sqrt_selected_squared_distances(
    squared_dissimilarities: np.ndarray,
    row_selector: np.ndarray,
    column_selector: int | np.ndarray,
) -> np.ndarray:
    """Return only selected distances from a squared distance matrix."""

    return np.sqrt(squared_dissimilarities[row_selector, column_selector])


def run_hierarchical_pca_fcm(
    X: np.ndarray,
    metadata: pd.DataFrame | None = None,
    *,
    max_depth: int = 4,
    min_node_size: int = 60,
    min_child_size: int = 20,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_membership: float = 0.40,
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
    m: float = 2.0,
    max_fcm_iter: int = 200,
    fcm_tol: float = 1e-6,
    include_conditional_memberships: bool = False,
    fast_mode: bool = False,
    fast_sample_size: int = 1000,
    fast_scout_n_init: int = 2,
    fast_refine_n_init: int = 3,
    fast_refine_top_k: int = 2,
    fast_stability_target: float = 0.85,
    fast_m_values: tuple[float, ...] = (2.0, 1.8, 1.6, 1.4),
    fast_reuse_scout_m: bool = True,
) -> HierarchicalResult:
    """Recursively split a dataset with spherical PCA+FCM."""

    started_at = time.perf_counter()
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if metadata is not None and len(metadata) != X.shape[0]:
        raise ValueError("metadata must contain exactly one row per embedding")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if min_node_size < min_child_size:
        raise ValueError("min_node_size must be at least min_child_size")
    _validate_fcm_selection_parameters(
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_child_size=min_child_size,
        min_membership=min_membership,
        max_membership_gap=max_membership_gap,
        selection_method=selection_method,
        min_xb_relative_improvement=min_xb_relative_improvement,
        xb_worsening_patience=xb_worsening_patience,
        m=m,
        max_iter=max_fcm_iter,
        tol=fcm_tol,
    )
    if not 0.0 <= forced_noise_ratio <= 1.0:
        raise ValueError("forced_noise_ratio must be between 0 and 1")
    if min_split_silhouette < -1.0 or min_split_silhouette > 1.0:
        raise ValueError("min_split_silhouette must be between -1 and 1")
    fast_config = FastFcmConfig(
        sample_size=fast_sample_size,
        scout_n_init=fast_scout_n_init,
        scout_max_attempts=max(fast_scout_n_init + 1, fast_scout_n_init * 2),
        scout_max_iter=min(max_fcm_iter, 60),
        scout_tol=max(fcm_tol, 1e-4),
        refine_top_k=fast_refine_top_k,
        refine_n_init=fast_refine_n_init,
        refine_max_attempts=max(fast_refine_n_init + 2, fast_refine_n_init * 2),
        refine_max_iter=min(max_fcm_iter, 60),
        refine_tol=max(fcm_tol, 1e-4),
        stability_target=fast_stability_target,
        m_values=tuple(float(value) for value in fast_m_values),
    )
    if fast_mode:
        fast_config.validate()

    if metadata is None:
        metadata = pd.DataFrame({"id": np.arange(X.shape[0])})
    else:
        metadata = metadata.copy()

    Xp, pca, pca_selection = fit_clustering_pca(
        X,
        n_components=pca_components,
        max_components=pca_max_components,
        min_components=pca_min_components,
        component_step=pca_component_step,
        k_values=pca_k_values,
        minimum_preservation_gain=pca_minimum_preservation_gain,
        seed=seed,
    )
    projection_support = pca_projection_support(X, pca)
    projection_support_threshold = calibrate_pca_projection_support_threshold(
        X,
        pca,
        distance_z=distance_z,
    )
    projection_outliers = (
        projection_support_threshold > 0.0
    ) & (projection_support < projection_support_threshold)
    labels_by_level = np.full((X.shape[0], max_depth), -1, dtype=int)
    soft_memberships_by_level = [
        np.full((X.shape[0], max_clusters), np.nan, dtype=np.float64)
        for _ in range(max_depth)
    ]
    is_noise = projection_outliers.copy()
    document_types = np.full(X.shape[0], DOCUMENT_TYPE_CORE, dtype=object)
    document_types[projection_outliers] = DOCUMENT_TYPE_NOISE
    noise_scores = np.clip(1.0 - projection_support, 0.0, 1.0)
    boundary_level = np.full(X.shape[0], -1, dtype=int)
    noise_level = np.full(X.shape[0], -1, dtype=int)
    noise_level[projection_outliers] = 1
    node_models: dict[str, HierarchyNodeModel] = {}

    def node_template(
        *,
        node_id: str,
        parent_id: str | None,
        path: str,
        depth: int,
        size: int,
    ) -> dict[str, Any]:
        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "path": path,
            "depth": depth,
            "size": int(size),
            "selected_k": None,
            "selected_silhouette": None,
            "selected_xie_beni": None,
            "selected_partition_coefficient": None,
            "selected_partition_entropy": None,
            "selected_selection_score": None,
            "selected_valid_clusters": 0,
            "selection_reason": None,
            "noise_count": 0,
            "boundary_count": 0,
            "candidate_metrics": [],
            "stop_reason": None,
            "children": [],
        }

    root = node_template(
        node_id="root",
        parent_id=None,
        path="",
        depth=0,
        size=X.shape[0],
    )

    def make_root_fallback(reason: str) -> None:
        valid_rows = ~projection_outliers
        labels_by_level[valid_rows, 0] = 0
        soft_memberships_by_level[0][valid_rows, 0] = 1.0
        root["stop_reason"] = reason
        root["fallback_single_cluster"] = True
        child = node_template(
            node_id="0",
            parent_id="root",
            path="0",
            depth=1,
            size=int(np.sum(valid_rows)),
        )
        child["stop_reason"] = f"root_not_split:{reason}"
        root["children"].append(child)

    def recurse(
        indices: np.ndarray,
        node: dict[str, Any],
        depth: int,
        inherited_m: float | None = None,
    ) -> None:
        if depth >= max_depth:
            node["stop_reason"] = "max_depth_reached"
            return
        if indices.size < min_node_size:
            node["stop_reason"] = "node_too_small"
            return

        selection_seed = seed + depth * 100_003 + indices.size
        if fast_mode:
            best, candidate_metrics, reason = select_fast_fcm_cluster_count(
                Xp[indices],
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                min_child_size=min_child_size,
                min_membership=min_membership,
                max_membership_gap=max_membership_gap,
                distance_z=distance_z,
                selection_method=selection_method,
                seed=selection_seed,
                config=fast_config,
                m_hint=(inherited_m if fast_reuse_scout_m else None),
            )
        else:
            best, candidate_metrics, reason = select_fcm_cluster_count(
                Xp[indices],
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                min_child_size=min_child_size,
                min_membership=min_membership,
                max_membership_gap=max_membership_gap,
                distance_z=distance_z,
                selection_method=selection_method,
                min_xb_relative_improvement=min_xb_relative_improvement,
                xb_worsening_patience=xb_worsening_patience,
                seed=selection_seed,
                m=m,
                max_iter=max_fcm_iter,
                tol=fcm_tol,
            )
        node["candidate_metrics"] = candidate_metrics
        node["selection_reason"] = reason
        if best is None:
            node["stop_reason"] = reason
            if depth == 0:
                make_root_fallback(reason)
            return
        if (
            selection_method != "multi_metric"
            and best.silhouette < min_split_silhouette
        ):
            node["stop_reason"] = "silhouette_below_threshold"
            node["selected_silhouette"] = float(best.silhouette)
            if depth == 0:
                make_root_fallback("silhouette_below_threshold")
            return

        if selection_method == "multi_metric":
            local_labels, effective_cluster_sizes = _filter_fcm_labels(
                Xp[indices],
                best.result,
                min_child_size=min_child_size,
                min_membership=min_membership,
                max_membership_gap=max_membership_gap,
                distance_z=distance_z,
            )
        else:
            local_labels = best.labels.copy()
            effective_cluster_sizes = list(best.cluster_sizes)
        if len(effective_cluster_sizes) < 2:
            node["stop_reason"] = "noise_filter_left_fewer_than_two_clusters"
            if depth == 0:
                make_root_fallback(node["stop_reason"])
            return

        node["selected_k"] = int(best.n_clusters)
        node["selected_silhouette"] = float(best.silhouette)
        node["selected_xie_beni"] = float(best.xie_beni)
        node["selected_partition_coefficient"] = float(
            best.partition_coefficient
        )
        node["selected_partition_entropy"] = float(best.partition_entropy)
        node["selected_selection_score"] = (
            float(best.selection_score)
            if best.selection_score is not None
            else None
        )
        node["selected_valid_clusters"] = int(len(effective_cluster_sizes))
        node["noise_count"] = int(np.sum(local_labels == -1))

        current_level = depth
        cached_squared_dissimilarities = best.result.squared_dissimilarities
        cached_squared = (
            None
            if cached_squared_dissimilarities is None
            else np.asarray(cached_squared_dissimilarities)
        )
        has_cached_squared = (
            cached_squared is not None
            and cached_squared.shape == best.result.memberships.shape
        )
        if not has_cached_squared:
            _, local_distance_matrix = sfcm_memberships_from_centers(
                Xp[indices],
                best.result.centers,
            )
            row_indices = np.arange(indices.size)
            local_distances = local_distance_matrix[
                row_indices,
                best.result.labels,
            ]
        else:
            row_indices = np.arange(indices.size)
            local_distances = _sqrt_selected_squared_distances(
                cached_squared,
                row_indices,
                best.result.labels,
            )
        local_document_types = fcm_document_types(
            Xp[indices],
            best.result,
            min_membership=min_membership,
            max_membership_gap=max_membership_gap,
            distance_z=distance_z,
            assigned_distances=local_distances,
        )
        local_noise_scores = fcm_noise_scores(
            best.result.memberships,
            local_distances,
            best.result.labels,
        )
        noise_scores[indices] = np.maximum(
            noise_scores[indices],
            local_noise_scores,
        )
        local_document_types[local_labels == -1] = DOCUMENT_TYPE_NOISE
        boundary_rows = (
            (local_labels >= 0)
            & (local_document_types == DOCUMENT_TYPE_BOUNDARY)
        )
        first_boundary_rows = boundary_rows & (
            document_types[indices] == DOCUMENT_TYPE_CORE
        )
        document_types[indices[boundary_rows]] = DOCUMENT_TYPE_BOUNDARY
        boundary_level[indices[first_boundary_rows]] = current_level + 1
        node["boundary_count"] = int(np.sum(boundary_rows))

        model_centers: list[np.ndarray] = []
        distance_thresholds: list[float] = []
        surviving_source_labels: list[int] = []
        node_features = Xp[indices]
        for cluster_id in range(len(effective_cluster_sizes)):
            cluster_mask = local_labels == cluster_id
            source_labels = best.result.labels[cluster_mask]
            if source_labels.size == 0:
                raise RuntimeError("Selected cluster has no surviving samples")
            source_label = int(np.bincount(source_labels).argmax())
            surviving_source_labels.append(source_label)
            center = best.result.centers[source_label]
            model_centers.append(center.copy())

            if not has_cached_squared:
                cluster_distances = local_distance_matrix[
                    cluster_mask,
                    source_label,
                ]
            else:
                assert cached_squared is not None
                cluster_distances = _sqrt_selected_squared_distances(
                    cached_squared,
                    cluster_mask,
                    source_label,
                )
            if cluster_distances.size < 4:
                distance_thresholds.append(float("inf"))
            else:
                median = float(np.median(cluster_distances))
                mad = float(np.median(np.abs(cluster_distances - median)))
                if mad <= 1e-12:
                    distance_thresholds.append(float("inf"))
                else:
                    distance_thresholds.append(
                        median + distance_z * 1.4826 * mad
                    )

        selected_memberships = best.result.memberships[:, surviving_source_labels]
        membership_sums = selected_memberships.sum(axis=1, keepdims=True)
        normalized_memberships = np.divide(
            selected_memberships,
            membership_sums,
            out=np.zeros_like(selected_memberships),
            where=membership_sums > 1e-12,
        )
        valid_membership_rows = local_labels >= 0
        soft_memberships_by_level[current_level][
            indices[valid_membership_rows], : len(surviving_source_labels)
        ] = normalized_memberships[valid_membership_rows]

        node_models[str(node["path"])] = HierarchyNodeModel(
            path=str(node["path"]),
            depth=current_level,
            centers=np.vstack(model_centers),
            distance_thresholds=np.asarray(distance_thresholds, dtype=np.float64),
            m=float(best.result.m),
        )

        noise_indices = indices[local_labels == -1]
        if noise_indices.size:
            is_noise[noise_indices] = True
            document_types[noise_indices] = DOCUMENT_TYPE_NOISE
            boundary_level[noise_indices] = -1
            noise_level[noise_indices] = current_level + 1

        for cluster_id, cluster_size in enumerate(effective_cluster_sizes):
            child_indices = indices[local_labels == cluster_id]
            if child_indices.size != cluster_size:
                raise RuntimeError("FCM cluster size bookkeeping is inconsistent")
            labels_by_level[child_indices, current_level] = cluster_id
            child_path = (
                f"{node['path']}/{cluster_id}"
                if node["path"]
                else str(cluster_id)
            )
            child = node_template(
                node_id=child_path,
                parent_id=str(node["node_id"]),
                path=child_path,
                depth=current_level + 1,
                size=child_indices.size,
            )
            node["children"].append(child)
            recurse(
                child_indices,
                child,
                depth + 1,
                float(best.m) if fast_mode and fast_reuse_scout_m else None,
            )

    recurse(np.flatnonzero(~projection_outliers), root, 0)
    root["projection_outlier_count"] = int(np.sum(projection_outliers))
    root["noise_count"] = int(root.get("noise_count", 0)) + int(
        np.sum(projection_outliers)
    )

    is_natural_noise = is_noise.copy()
    document_ids = (
        metadata["id"].to_numpy()
        if "id" in metadata.columns
        else np.arange(X.shape[0])
    )
    (
        is_noise,
        is_forced_noise,
        forced_only,
        document_types,
        noise_level,
    ) = merge_forced_noise(
        is_natural_noise,
        noise_scores,
        document_ids,
        document_types,
        noise_level,
        forced_noise_ratio=forced_noise_ratio,
    )

    assigned_depth = np.sum(labels_by_level >= 0, axis=1)
    model = HierarchicalModel(
        pca=pca,
        nodes=node_models,
        max_depth=max_depth,
        fallback_single_cluster=bool(root.get("fallback_single_cluster", False)),
        projection_support_threshold=projection_support_threshold,
    )
    conditional_memberships = None
    if include_conditional_memberships:
        conditional_memberships = conditional_memberships_from_projected(
            Xp,
            model,
        )
    assignments = build_hierarchical_assignments(
        metadata,
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

    def collect_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = [node]
        for child in node["children"]:
            nodes.extend(collect_nodes(child))
        return nodes

    nodes = collect_nodes(root)
    non_root_nodes = [node for node in nodes if node["depth"] > 0]
    levels_reached = int(np.max(assigned_depth)) if np.any(assigned_depth) else 0
    summary: dict[str, Any] = {
        "method": f"recursive_pca{Xp.shape[1]}_spherical_fcm",
        "samples": int(X.shape[0]),
        "pca_components": int(Xp.shape[1]),
        "levels_requested": int(max_depth),
        "levels_reached": levels_reached,
        "node_count": int(len(non_root_nodes)),
        "leaf_count": int(sum(not node["children"] for node in nodes)),
        "noise_count": int(np.sum(is_noise)),
        "natural_noise_count": int(np.sum(is_natural_noise)),
        "forced_noise_count": int(np.sum(is_forced_noise)),
        "forced_only_noise_count": int(np.sum(forced_only)),
        "projection_outlier_count": int(np.sum(projection_outliers)),
        "projection_support_threshold": float(projection_support_threshold),
        "boundary_count": int(
            np.sum(document_types == DOCUMENT_TYPE_BOUNDARY)
        ),
        "core_count": int(np.sum(document_types == DOCUMENT_TYPE_CORE)),
        "noise_by_level": {
            str(level): int(np.sum(noise_level == level))
            for level in range(1, max_depth + 1)
        },
        "leaf_cluster_count": int(
            assignments.loc[~assignments["is_noise"], "cluster_path"].nunique()
        ),
        "runtime_sec": float(time.perf_counter() - started_at),
    }
    config = {
        "max_depth": int(max_depth),
        "min_node_size": int(min_node_size),
        "min_child_size": int(min_child_size),
        "min_clusters": int(min_clusters),
        "max_clusters": int(max_clusters),
        "selection_method": selection_method,
        "min_xb_relative_improvement": float(min_xb_relative_improvement),
        "xb_worsening_patience": int(xb_worsening_patience),
        "multi_metric_weights": {
            "xie_beni": 0.25,
            "silhouette": 0.45,
            "restart_stability": 0.20,
            "modified_partition_coefficient": 0.10,
        },
        "multi_metric_normalization": "rank_average_ties",
        "multi_metric_candidate_metrics_include_all_samples": True,
        "multi_metric_assign_all_samples": False,
        "min_split_silhouette": float(min_split_silhouette),
        "min_membership": float(min_membership),
        "max_membership_gap": float(max_membership_gap),
        "forced_noise_ratio": float(forced_noise_ratio),
        "distance_z": float(distance_z),
        "projection_support_threshold": float(projection_support_threshold),
        "pca_components_requested": (
            "auto" if pca_components is None else int(pca_components)
        ),
        "pca_components_selected": int(Xp.shape[1]),
        "pca_selection": (
            None if pca_selection is None else pca_selection.to_dict()
        ),
        "seed": int(seed),
        "fuzzifier": float(m),
        "max_fcm_iter": int(max_fcm_iter),
        "fcm_tol": float(fcm_tol),
        "include_conditional_memberships": bool(
            include_conditional_memberships
        ),
        "fast_mode": bool(fast_mode),
        "fast_sample_size": int(fast_sample_size),
        "fast_scout_n_init": int(fast_scout_n_init),
        "fast_refine_n_init": int(fast_refine_n_init),
        "fast_refine_top_k": int(fast_refine_top_k),
        "fast_stability_target": float(fast_stability_target),
        "fast_m_values": [float(value) for value in fast_m_values],
        "fast_reuse_scout_m": bool(fast_reuse_scout_m),
    }
    tree = {"config": config, "summary": summary, "root": root}
    return HierarchicalResult(
        assignments=assignments,
        tree=tree,
        summary=summary,
        memberships={
            level + 1: level_memberships
            for level, level_memberships in enumerate(soft_memberships_by_level)
        },
        conditional_memberships=conditional_memberships,
        model=model,
    )
