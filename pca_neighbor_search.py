"""Reusable exact and PyNNDescent neighbor search in PCA space.

The clustering benchmarks query the same discovery projection several times
(calibration and held-out test predictions).  Keeping the index here makes
that reuse explicit and also gives the exact and approximate paths the same
distance/weighting contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


SUPPORTED_BACKENDS = ("exact", "pynndescent")


def _features(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 2D array")
    return result


@dataclass
class PcaNeighborIndex:
    """A discovery-projection neighbor index shared by all queries."""

    reference: np.ndarray
    backend: str = "exact"
    max_neighbors: int = 24
    graph_neighbors: int = 32
    random_state: int = 42
    query_epsilon: float = 0.1
    model: Any = None

    def __post_init__(self) -> None:
        self.reference = _features(self.reference, name="reference")
        if self.reference.shape[0] < 2:
            raise ValueError("reference must contain at least two rows")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}")
        if self.max_neighbors < 1 or self.max_neighbors >= self.reference.shape[0]:
            raise ValueError("max_neighbors must be between 1 and reference size - 1")
        if self.graph_neighbors < 1:
            raise ValueError("graph_neighbors must be positive")
        if self.query_epsilon < 0:
            raise ValueError("query_epsilon must be non-negative")
        if self.backend == "exact":
            self.model = NearestNeighbors(metric="euclidean", algorithm="brute", n_jobs=1).fit(self.reference)
        else:
            # NNDescent requires n_neighbors <= n_samples.  On the real
            # Wikipedia discovery set this is at least 32; tiny unit-test
            # fixtures necessarily use the largest valid graph.
            from pynndescent import NNDescent

            graph_k = min(
                self.reference.shape[0],
                max(32, self.graph_neighbors, self.max_neighbors + 1),
            )
            self.graph_neighbors = int(graph_k)
            self.model = NNDescent(
                self.reference,
                metric="euclidean",
                n_neighbors=graph_k,
                random_state=int(self.random_state),
                n_jobs=1,
            )

    def query(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude_self: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return sorted ``(distances, indices)`` for up to ``k`` neighbors.

        ``exclude_self`` is positional for a query that is the reference
        matrix (the normal discovery self-query use case).  Out-of-sample
        calibration and test queries leave it false.
        """

        values = _features(query, name="query")
        if values.shape[1] != self.reference.shape[1]:
            raise ValueError("query and reference dimensions must match")
        if k < 1 or k > self.max_neighbors:
            raise ValueError(f"k must be between 1 and max_neighbors ({self.max_neighbors})")
        positional_self = bool(
            exclude_self
            and values.shape == self.reference.shape
            and np.array_equal(values, self.reference)
        )
        request_k = min(self.reference.shape[0], k + 1 if positional_self else k)
        if self.backend == "exact":
            distances, indices = self.model.kneighbors(values, n_neighbors=request_k, return_distance=True)
        else:
            # NNDescent returns ``(indices, distances)`` (the opposite order
            # of sklearn's kneighbors API).
            indices, distances = self.model.query(values, k=request_k, epsilon=float(self.query_epsilon))
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        output_distances = np.empty((len(values), k), dtype=np.float64)
        output_indices = np.empty((len(values), k), dtype=np.int64)
        for row, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
            if positional_self:
                keep = row_indices != row
                row_indices = row_indices[keep]
                row_distances = row_distances[keep]
            if len(row_indices) < k:
                raise ValueError("neighbor search could not return enough neighbors")
            order = np.argsort(row_distances[:k], kind="mergesort")
            output_distances[row] = row_distances[:k][order]
            output_indices[row] = row_indices[:k][order]
        if not np.all(np.isfinite(output_distances)):
            raise ValueError("neighbor distances must be finite")
        return output_distances, output_indices


def build_pca_neighbor_index(
    reference: np.ndarray,
    *,
    backend: str = "exact",
    max_neighbors: int = 24,
    graph_neighbors: int = 32,
    random_state: int = 42,
    query_epsilon: float = 0.1,
) -> PcaNeighborIndex:
    """Build one reusable PCA-space index for a discovery projection."""

    return PcaNeighborIndex(
        reference,
        backend=backend,
        max_neighbors=int(max_neighbors),
        graph_neighbors=int(graph_neighbors),
        random_state=int(random_state),
        query_epsilon=float(query_epsilon),
    )
