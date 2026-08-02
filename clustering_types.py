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
    objective: float
    noise_count: int
    cluster_sizes: list[int]


@dataclass
class HierarchicalResult:
    assignments: pd.DataFrame
    tree: dict[str, Any]
    summary: dict[str, Any]
