from __future__ import annotations

import numpy as np
import pandas as pd


DOCUMENT_TYPE_CORE = "core"
DOCUMENT_TYPE_BOUNDARY = "boundary"
DOCUMENT_TYPE_NOISE = "noise"


def path_membership_column(path: str) -> str:
    """Return a stable assignment-column name for a hierarchy path."""

    if not path:
        raise ValueError("path membership columns require a non-root path")
    parts = path.split("/")
    return f"level_{len(parts)}_path_membership_{'_'.join(parts)}"


def build_hierarchical_assignments(
    metadata: pd.DataFrame,
    labels_by_level: np.ndarray,
    is_noise: np.ndarray,
    is_natural_noise: np.ndarray,
    is_forced_noise: np.ndarray,
    document_types: np.ndarray,
    noise_scores: np.ndarray,
    boundary_level: np.ndarray,
    noise_level: np.ndarray,
    soft_memberships_by_level: list[np.ndarray] | None = None,
    conditional_memberships: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build the canonical flat assignment table for a hierarchy result."""

    assignments = metadata.copy()
    for level in range(labels_by_level.shape[1]):
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
    rows = np.arange(len(metadata))
    leaf_cluster[has_leaf] = labels_by_level[
        rows[has_leaf],
        assigned_depth[has_leaf] - 1,
    ]
    leaf_cluster[is_noise] = -1
    assignments["cluster"] = leaf_cluster

    cluster_paths: list[str] = []
    for row, row_labels in enumerate(labels_by_level):
        parts = [str(int(label)) for label in row_labels if label >= 0]
        if is_noise[row]:
            cluster_paths.append("/".join(parts + ["noise"]) if parts else "noise")
        else:
            cluster_paths.append("/".join(parts) if parts else "root")

    assignments["cluster_path"] = cluster_paths
    assignments["is_noise"] = is_noise.astype(bool)
    assignments["is_natural_noise"] = is_natural_noise.astype(bool)
    assignments["is_forced_noise"] = is_forced_noise.astype(bool)
    assignments["is_boundary"] = document_types == DOCUMENT_TYPE_BOUNDARY
    assignments["document_type"] = document_types
    assignments["noise_score"] = noise_scores
    assignments["boundary_level"] = boundary_level.astype(int)
    assignments["noise_level"] = noise_level.astype(int)
    assignments["leaf_level"] = leaf_level
    return assignments
