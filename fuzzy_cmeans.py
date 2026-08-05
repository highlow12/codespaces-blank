"""Reusable fuzzy C-means optimizer and geometry strategies.

The optimizer owns the FCM iteration.  A geometry supplies the representation,
prototype update, and non-negative squared dissimilarities used by that
iteration.  This keeps the clustering mechanics reusable while making the
chosen geometry explicit at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.preprocessing import normalize

from clustering_types import FCMResult


DEFAULT_EPSILON = 1e-12


class FuzzyCMeansGeometry(Protocol):
    """Operations that specialize FCM for a feature geometry."""

    def prepare_samples(self, X: np.ndarray) -> np.ndarray:
        """Validate and map raw samples to the geometry's representation."""

    def update_centers(
        self,
        X: np.ndarray,
        memberships: np.ndarray,
        *,
        m: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return one prototype per membership column."""

    def squared_dissimilarities(
        self,
        X: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        """Return the sample-by-center squared dissimilarity matrix."""


def memberships_from_squared_dissimilarities(
    squared_dissimilarities: np.ndarray,
    *,
    m: float,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Compute FCM memberships, assigning tied exact matches equally.

    Squared dissimilarities avoid an unnecessary square root.  They produce
    the same memberships as ordinary Euclidean distances after adjusting the
    exponent from ``2 / (m - 1)`` to ``1 / (m - 1)``.
    """

    values = np.asarray(squared_dissimilarities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("squared_dissimilarities must be a non-empty 2D array")
    if not np.all(np.isfinite(values)) or np.any(values < -epsilon):
        raise ValueError("squared_dissimilarities must be finite and non-negative")
    if m <= 1.0:
        raise ValueError("m must be greater than 1")

    values = np.maximum(values, 0.0)
    memberships = np.zeros_like(values)
    exact_matches = values <= epsilon
    exact_rows = exact_matches.any(axis=1)
    if np.any(exact_rows):
        ties = exact_matches[exact_rows]
        memberships[exact_rows] = ties / ties.sum(axis=1, keepdims=True)

    regular_rows = ~exact_rows
    if np.any(regular_rows):
        regular = values[regular_rows]
        exponent = 1.0 / (m - 1.0)
        ratios = (regular[:, :, None] / regular[:, None, :]) ** exponent
        memberships[regular_rows] = 1.0 / ratios.sum(axis=2)
    return memberships


@dataclass(frozen=True)
class SphericalGeometry:
    """Cosine geometry represented by unit vectors and unit prototypes."""

    epsilon: float = DEFAULT_EPSILON

    def prepare_samples(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain only finite values")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= self.epsilon):
            raise ValueError("SFCM cannot cluster zero-length samples")
        return normalize(values, norm="l2")

    def update_centers(
        self,
        X: np.ndarray,
        memberships: np.ndarray,
        *,
        m: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        centers = (memberships**m).T @ X
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        empty = norms[:, 0] <= self.epsilon
        if np.any(empty):
            centers[empty] = X[
                rng.integers(0, X.shape[0], size=int(np.sum(empty)))
            ]
            norms = np.linalg.norm(centers, axis=1, keepdims=True)
        return centers / np.maximum(norms, self.epsilon)

    def squared_dissimilarities(
        self,
        X: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        # ||x - v||² = 2(1 - cos(x, v)) for unit x and v.
        return np.maximum(2.0 - 2.0 * (X @ centers.T), 0.0)


@dataclass
class FuzzyCMeans:
    """Geometry-independent FCM optimizer."""

    geometry: FuzzyCMeansGeometry
    m: float = 2.0
    max_iter: int = 200
    tol: float = 1e-6
    seed: int = 42

    def fit(self, X: np.ndarray, n_clusters: int) -> FCMResult:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if self.m <= 1.0:
            raise ValueError("m must be greater than 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative")

        samples = self.geometry.prepare_samples(X)
        rng = np.random.default_rng(self.seed)
        memberships = rng.random((samples.shape[0], n_clusters))
        memberships /= memberships.sum(axis=1, keepdims=True)

        for iteration in range(1, self.max_iter + 1):
            previous = memberships
            centers = self.geometry.update_centers(
                samples,
                memberships,
                m=self.m,
                rng=rng,
            )
            memberships = memberships_from_squared_dissimilarities(
                self.geometry.squared_dissimilarities(samples, centers),
                m=self.m,
            )
            if np.max(np.abs(memberships - previous)) < self.tol:
                break

        return FCMResult(
            labels=memberships.argmax(axis=1),
            memberships=memberships,
            centers=centers,
            iterations=iteration,
        )


@dataclass
class SphericalFuzzyCMeans:
    """SFCM facade using cosine dissimilarity on the unit sphere."""

    m: float = 2.0
    max_iter: int = 200
    tol: float = 1e-6
    seed: int = 42

    def fit(self, X: np.ndarray, n_clusters: int) -> FCMResult:
        return FuzzyCMeans(
            geometry=SphericalGeometry(),
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            seed=self.seed,
        ).fit(X, n_clusters)
