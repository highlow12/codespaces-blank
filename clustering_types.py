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


@dataclass
class PipelineResult:
    metrics: dict[str, Any]
    labels: np.ndarray
    memberships: np.ndarray | None = None


@dataclass
class FCMKCandidate:
    """One candidate k evaluated while splitting a hierarchy node."""

    n_clusters: int
    result: FCMResult
    labels: np.ndarray
    silhouette: float
    xie_beni: float
    xb_relative_improvement: float | None
    objective: float
    noise_count: int
    cluster_sizes: list[int]


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


@dataclass
class HierarchicalModel:
    """Reusable PCA and hierarchy split models for incremental assignment."""

    pca: Any
    nodes: dict[str, HierarchyNodeModel]
    max_depth: int
    fallback_single_cluster: bool = False
