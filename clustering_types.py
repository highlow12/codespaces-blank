from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FCMResult:
    labels: np.ndarray
    memberships: np.ndarray
    centers: np.ndarray
    iterations: int
    objective: float | None = None
    m: float = 2.0
    n_init: int = 1
    attempts: int = 1
    valid_restarts: int = 1
    restart_stability: float = 1.0
    minimum_center_distance: float | None = None
    squared_dissimilarities: np.ndarray | None = None


@dataclass
class PipelineResult:
    metrics: dict[str, Any]
    labels: np.ndarray
    memberships: np.ndarray | None = None
    # Optional artifact produced by discovery pipelines.  This is separate
    # from the legacy recursive SFCM HierarchicalResult below.
    hierarchy: Any | None = None


@dataclass
class FCMKCandidate:
    """One candidate k evaluated while splitting a hierarchy node."""

    n_clusters: int
    result: FCMResult
    labels: np.ndarray
    silhouette: float
    xie_beni: float
    xb_relative_improvement: float | None
    partition_coefficient: float
    modified_partition_coefficient: float
    partition_entropy: float
    normalized_partition_entropy: float
    selection_score: float | None
    objective: float
    noise_count: int
    cluster_sizes: list[int]
    m: float = 2.0
    restart_stability: float = 1.0
    valid_restarts: int = 1
    attempts: int = 1
    minimum_center_distance: float | None = None


@dataclass
class HierarchicalResult:
    assignments: pd.DataFrame
    tree: dict[str, Any]
    summary: dict[str, Any]
    memberships: dict[int, np.ndarray] | None = None
    conditional_memberships: dict[str, np.ndarray] | None = None
    model: "HierarchicalModel | None" = None


@dataclass
class HierarchyNodeModel:
    """A fitted split that can assign future points without refitting."""

    path: str
    depth: int
    centers: np.ndarray
    distance_thresholds: np.ndarray
    m: float = 2.0


@dataclass
class HierarchicalModel:
    """Reusable PCA and hierarchy split models for incremental assignment."""

    pca: Any
    nodes: dict[str, HierarchyNodeModel]
    max_depth: int
    fallback_single_cluster: bool = False
    projection_support_threshold: float = 0.0
