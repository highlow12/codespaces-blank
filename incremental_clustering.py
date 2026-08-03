"""Incremental hierarchical clustering and fixed-coordinate visualization.

The initial batch is clustered with the existing recursive PCA + spherical FCM
implementation. Later batches are transformed with the fitted PCA and assigned
to the stored hierarchy. If a new batch contains too much noise, the complete
accumulated dataset is clustered again while the original PCA + UMAP
coordinates remain fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
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
    conditional_memberships_from_projected,
    fcm_memberships_from_centers,
    path_membership_column,
    run_hierarchical_pca_fcm,
    transform_pca_normalized_features,
)


DEFAULT_NOISE_THRESHOLD = 0.30


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


def _assignments_from_labels(
    metadata: pd.DataFrame,
    labels_by_level: np.ndarray,
    is_noise: np.ndarray,
    noise_level: np.ndarray,
    soft_memberships_by_level: list[np.ndarray] | None = None,
    conditional_memberships: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    assignments = metadata.copy()
    max_depth = labels_by_level.shape[1]
    for level in range(max_depth):
        assignments[f"level_{level + 1}_cluster"] = labels_by_level[:, level]
        if soft_memberships_by_level is not None:
            level_memberships = soft_memberships_by_level[level]
            for cluster_id in range(level_memberships.shape[1]):
                assignments[
                    f"level_{level + 1}_membership_{cluster_id}"
                ] = level_memberships[:, cluster_id]
    if conditional_memberships is not None:
        for path, path_membership in conditional_memberships.items():
            assignments[path_membership_column(path)] = path_membership

    assigned_depth = np.sum(labels_by_level >= 0, axis=1)
    leaf_level = np.where(assigned_depth > 0, assigned_depth, -1).astype(int)
    leaf_cluster = np.full(len(metadata), -1, dtype=int)
    has_leaf = assigned_depth > 0
    row_indices = np.arange(len(metadata))
    leaf_cluster[has_leaf] = labels_by_level[
        row_indices[has_leaf], assigned_depth[has_leaf] - 1
    ]
    leaf_cluster[is_noise] = -1
    assignments["cluster"] = leaf_cluster

    cluster_paths: list[str] = []
    for row in range(len(metadata)):
        path_parts = [
            str(int(label))
            for label in labels_by_level[row]
            if label >= 0
        ]
        if is_noise[row]:
            cluster_paths.append(
                "/".join(path_parts + ["noise"]) if path_parts else "noise"
            )
        else:
            cluster_paths.append("/".join(path_parts) if path_parts else "root")

    assignments["cluster_path"] = cluster_paths
    assignments["is_noise"] = is_noise.astype(bool)
    assignments["noise_level"] = noise_level.astype(int)
    assignments["leaf_level"] = leaf_level
    return assignments


def assign_to_hierarchy(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    hierarchy_model: HierarchicalModel,
    *,
    min_membership: float,
    m: float = 2.0,
) -> tuple[pd.DataFrame, float]:
    """Assign a batch to fixed hierarchy centers and return its noise ratio."""

    values = _validate_embeddings(embeddings)
    frame = _validate_metadata(metadata, len(values))
    if not 0.0 <= min_membership <= 1.0:
        raise ValueError("min_membership must be between 0 and 1")

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
                local_noise = (
                    memberships.max(axis=1) < min_membership
                ) | (assigned_distances > thresholds)

                if np.any(local_noise):
                    noise_indices = indices[local_noise]
                    is_noise[noise_indices] = True
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

    assignments = _assignments_from_labels(
        frame,
        labels_by_level,
        is_noise,
        noise_level,
        soft_memberships_by_level,
        conditional_memberships,
    )
    return assignments, float(np.mean(is_noise))


def _cluster_config(
    *,
    max_depth: int,
    min_node_size: int,
    min_child_size: int,
    min_clusters: int,
    max_clusters: int,
    min_membership: float,
    distance_z: float,
    selection_method: str,
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
) -> dict[str, Any]:
    return {
        "max_depth": int(max_depth),
        "min_node_size": int(min_node_size),
        "min_child_size": int(min_child_size),
        "min_clusters": int(min_clusters),
        "max_clusters": int(max_clusters),
        "min_membership": float(min_membership),
        "distance_z": float(distance_z),
        "selection_method": selection_method,
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
    }


def _fit_hierarchy(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[HierarchicalModel, dict[str, Any], pd.DataFrame]:
    result = run_hierarchical_pca_fcm(
        embeddings,
        metadata,
        max_depth=int(config["max_depth"]),
        min_node_size=int(config["min_node_size"]),
        min_child_size=int(config["min_child_size"]),
        min_clusters=int(config["min_clusters"]),
        max_clusters=int(config["max_clusters"]),
        min_membership=float(config["min_membership"]),
        distance_z=float(config["distance_z"]),
        selection_method=str(config["selection_method"]),
        min_split_silhouette=float(config["min_split_silhouette"]),
        pca_components=int(config["pca_components"]),
        seed=int(config["seed"]),
    )
    if result.model is None:
        raise RuntimeError("Hierarchical clustering did not return a reusable model")
    return result.model, result.tree, result.assignments


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
    distance_z: float = 3.5,
    selection_method: str = "silhouette",
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
        distance_z=distance_z,
        selection_method=selection_method,
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
    )
    hierarchy_model, tree, assignments = _fit_hierarchy(values, frame, config)
    cluster_target, cluster_target_metric, _ = build_cluster_supervision(assignments)
    visual_pca, visual_reducer, coordinates = fit_projection_model(
        values,
        seed=seed,
        pca_components=visual_pca_components,
        n_neighbors=visual_n_neighbors,
        min_dist=visual_min_dist,
        metric=visual_metric,
        spread=visual_spread,
        densmap=visual_densmap,
        cluster_target=cluster_target,
        cluster_target_metric=cluster_target_metric,
        cluster_target_weight=visual_cluster_target_weight,
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

        if bool(row["is_noise"]):
            noise_level = int(row["noise_level"])
            parent_path = "/".join(path_parts[: max(noise_level - 1, 0)])
            node = nodes_by_path.get(parent_path)
            if node is not None:
                node["noise_count"] += 1

    summary = copy.deepcopy(updated_tree["summary"])
    summary["samples"] = int(len(all_assignments))
    summary["noise_count"] = int(all_assignments["is_noise"].sum())
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


def update_incremental_state(
    state: IncrementalClusterState,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    noise_threshold: float | None = None,
) -> tuple[IncrementalClusterState, dict[str, Any]]:
    """Assign a new batch and optionally re-cluster all accumulated samples."""

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
    new_assignments, new_noise_ratio = assign_to_hierarchy(
        values,
        frame,
        state.hierarchy_model,
        min_membership=float(state.config["min_membership"]),
        m=float(state.config["m"]),
    )
    should_recluster = new_noise_ratio > threshold
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
    config = dict(state.config)
    config["noise_threshold"] = threshold
    config["update_count"] = int(config.get("update_count", 0)) + 1

    if should_recluster:
        hierarchy_model, tree, assignments = _fit_hierarchy(
            combined_embeddings,
            combined_metadata,
            config,
        )
        reclustered = True
    else:
        hierarchy_model = state.hierarchy_model
        assignments = _append_assignments(state.assignments, new_assignments)
        tree = _refresh_tree_after_append(
            state.tree,
            new_assignments,
            assignments,
            int(config["update_count"]),
        )
        reclustered = False

    config["last_update_noise_ratio"] = float(new_noise_ratio)
    config["last_update_reclustered"] = reclustered
    updated_state = IncrementalClusterState(
        embeddings=combined_embeddings,
        metadata=combined_metadata,
        assignments=assignments,
        coordinates=combined_coordinates,
        hierarchy_model=hierarchy_model,
        tree=tree,
        config=config,
        visual_pca=state.visual_pca,
        visual_reducer=state.visual_reducer,
    )
    summary = {
        "new_samples": int(len(values)),
        "new_noise_count": int(round(new_noise_ratio * len(values))),
        "new_noise_ratio": float(new_noise_ratio),
        "noise_threshold": float(threshold),
        "reclustered": reclustered,
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
        "version": 1,
        "embeddings": state.embeddings,
        "metadata": state.metadata,
        "assignments": state.assignments,
        "coordinates": state.coordinates,
        "hierarchy_model": state.hierarchy_model,
        "tree": state.tree,
        "config": state.config,
        "visual_pca": state.visual_pca,
        "visual_reducer": state.visual_reducer,
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
    elif isinstance(payload, dict) and payload.get("version") == 1:
        fields = {
            key: payload[key]
            for key in (
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
            if key in payload
        }
        if len(fields) != 9:
            raise ValueError(f"Invalid incremental state: {path}")
        state = IncrementalClusterState(**fields)
    else:
        raise ValueError(f"Invalid incremental state: {path}")
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
    parser.add_argument("--distance-z", type=float, default=3.5)
    parser.add_argument(
        "--selection-method",
        choices=["silhouette", "knee"],
        default="silhouette",
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
        distance_z=args.distance_z,
        selection_method=args.selection_method,
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
        help="Assign a new batch and re-cluster when its noise is too high.",
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
