"""Small, dependency-light value objects used by the browser API.

The objects in this module are deliberately internal-facing.  The public
entry point converts them to ordinary dictionaries/lists before returning so
that Pyodide callers can use ``result.toJs()`` without knowing Python types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PcaSelection:
    selected_dimension: int
    fitted_dimension: int
    selection_reason: str
    candidates: tuple[dict[str, Any], ...]
    features: np.ndarray
    normalized_input: np.ndarray
    pca: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_dimension": int(self.selected_dimension),
            "fitted_dimension": int(self.fitted_dimension),
            "selection_reason": self.selection_reason,
            "candidates": [dict(candidate) for candidate in self.candidates],
            "configuration": {"input_normalized": True},
        }


@dataclass(frozen=True)
class DiscoveryOutput:
    """Validated output of the optional UMAP/HDBSCAN runtime boundary."""

    umap_features: np.ndarray
    leaf_labels: np.ndarray
    memberships: np.ndarray
    probabilities: np.ndarray
    outlier_scores: np.ndarray
    configuration: dict[str, Any]


class DiscoveryDependencyError(RuntimeError):
    """Raised when the browser runtime has no UMAP/HDBSCAN implementation."""
