"""Recursive hierarchical PCA + spherical FCM orchestration."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from clustering_types import HierarchicalModel, HierarchicalResult, HierarchyNodeModel
from fcm_core import (
    DEFAULT_CLUSTERING_PCA_COMPONENTS,
    conditional_memberships_from_projected,
    fcm_memberships_from_centers,
    fit_pca_normalized_features,
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
    _validate_fcm_selection_parameters,
    select_fcm_cluster_count,
)
from hierarchical_assignments import (
    DOCUMENT_TYPE_BOUNDARY,
    DOCUMENT_TYPE_CORE,
    DOCUMENT_TYPE_NOISE,
    build_hierarchical_assignments,
)




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
    pca_components: int = DEFAULT_CLUSTERING_PCA_COMPONENTS,
    seed: int = 42,
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
    )
    if not 0.0 <= forced_noise_ratio <= 1.0:
        raise ValueError("forced_noise_ratio must be between 0 and 1")
    if min_split_silhouette < -1.0 or min_split_silhouette > 1.0:
        raise ValueError("min_split_silhouette must be between -1 and 1")

    if metadata is None:
        metadata = pd.DataFrame({"id": np.arange(X.shape[0])})
    else:
        metadata = metadata.copy()

    Xp, pca = fit_pca_normalized_features(
        X,
        n_components=pca_components,
        seed=seed,
    )
    labels_by_level = np.full((X.shape[0], max_depth), -1, dtype=int)
    soft_memberships_by_level = [
        np.full((X.shape[0], max_clusters), np.nan, dtype=np.float64)
        for _ in range(max_depth)
    ]
    is_noise = np.zeros(X.shape[0], dtype=bool)
    document_types = np.full(X.shape[0], DOCUMENT_TYPE_CORE, dtype=object)
    noise_scores = np.zeros(X.shape[0], dtype=np.float64)
    boundary_level = np.full(X.shape[0], -1, dtype=int)
    noise_level = np.full(X.shape[0], -1, dtype=int)
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
        labels_by_level[:, 0] = 0
        soft_memberships_by_level[0][:, 0] = 1.0
        root["stop_reason"] = reason
        root["fallback_single_cluster"] = True
        child = node_template(
            node_id="0",
            parent_id="root",
            path="0",
            depth=1,
            size=X.shape[0],
        )
        child["stop_reason"] = f"root_not_split:{reason}"
        root["children"].append(child)

    def recurse(indices: np.ndarray, node: dict[str, Any], depth: int) -> None:
        if depth >= max_depth:
            node["stop_reason"] = "max_depth_reached"
            return
        if indices.size < min_node_size:
            node["stop_reason"] = "node_too_small"
            return

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
            seed=seed + depth * 100_003 + indices.size,
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
        local_document_types = fcm_document_types(
            Xp[indices],
            best.result,
            min_membership=min_membership,
            max_membership_gap=max_membership_gap,
            distance_z=distance_z,
        )
        local_distances = euclidean_distances(
            Xp[indices],
            best.result.centers,
        )[np.arange(indices.size), best.result.labels]
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

            cluster_distances = euclidean_distances(
                node_features[cluster_mask],
                center.reshape(1, -1),
            ).ravel()
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
            recurse(child_indices, child, depth + 1)

    recurse(np.arange(X.shape[0], dtype=int), root, 0)

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
    )
    conditional_memberships = conditional_memberships_from_projected(Xp, model)
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
            "xie_beni": 0.50,
            "modified_partition_coefficient": 0.25,
            "normalized_partition_entropy": 0.25,
        },
        "multi_metric_normalization": "rank_average_ties",
        "multi_metric_candidate_metrics_include_all_samples": True,
        "multi_metric_assign_all_samples": False,
        "min_split_silhouette": float(min_split_silhouette),
        "min_membership": float(min_membership),
        "max_membership_gap": float(max_membership_gap),
        "forced_noise_ratio": float(forced_noise_ratio),
        "distance_z": float(distance_z),
        "pca_components_requested": int(pca_components),
        "seed": int(seed),
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
