"""FCM variants that retain or discard PCA projection magnitude explicitly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from clustering_types import FCMResult
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fuzzy_cmeans import memberships_from_squared_dissimilarities


GeometryName = Literal["cosine_raw", "cosine_normalized", "euclidean"]


def _validate_samples(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain only finite values")
    return values


@dataclass(frozen=True)
class ExperimentalFcmGeometry:
    """Geometry used by the PCA normalization comparison benchmark."""

    name: GeometryName
    epsilon: float = 1e-12

    def prepare_samples(self, X: np.ndarray) -> np.ndarray:
        values = _validate_samples(X)
        if self.name == "cosine_normalized":
            return self._unit_rows(values)
        return values

    def initialization_basis(self, X: np.ndarray) -> np.ndarray:
        if self.name.startswith("cosine_"):
            return self._unit_rows(X)
        return X

    def update_centers(
        self,
        X: np.ndarray,
        memberships: np.ndarray,
        *,
        m: float,
    ) -> np.ndarray:
        weights = memberships**m
        denominators = np.maximum(weights.sum(axis=0)[:, None], self.epsilon)
        return (weights.T @ X) / denominators

    def squared_dissimilarities(
        self,
        X: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        if self.name.startswith("cosine_"):
            samples = self._unit_rows(X)
            prototypes = self._unit_rows(centers)
            # Match the existing SFCM chord-distance convention.
            return np.maximum(2.0 - 2.0 * (samples @ prototypes.T), 0.0)
        differences = X[:, None, :] - centers[None, :, :]
        return np.einsum("nkd,nkd->nk", differences, differences)

    def minimum_center_distance(self, centers: np.ndarray) -> float:
        if len(centers) < 2:
            return float("inf")
        squared = self.squared_dissimilarities(centers, centers)
        np.fill_diagonal(squared, np.inf)
        return float(np.sqrt(np.min(squared)))

    def silhouette_metric(self) -> str:
        return "cosine" if self.name.startswith("cosine_") else "euclidean"

    def _unit_rows(self, values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= self.epsilon):
            raise ValueError("cosine FCM cannot cluster zero-length samples")
        return normalize(values, norm="l2")


def _restart_stability(results: list[FCMResult]) -> float:
    if len(results) < 2:
        return 0.0
    values = [
        adjusted_rand_score(results[first].labels, results[second].labels)
        for first in range(len(results))
        for second in range(first + 1, len(results))
    ]
    return float(np.mean(values))


def _fit_once(
    X: np.ndarray,
    n_clusters: int,
    *,
    geometry: ExperimentalFcmGeometry,
    m: float,
    max_iter: int,
    tol: float,
    seed: int,
) -> FCMResult:
    basis = geometry.initialization_basis(X)
    _centers, center_indices = kmeans_plusplus(
        basis,
        n_clusters=n_clusters,
        random_state=seed,
    )
    centers = X[np.asarray(center_indices, dtype=int)].copy()
    squared = geometry.squared_dissimilarities(X, centers)
    memberships = memberships_from_squared_dissimilarities(squared, m=m)

    for iteration in range(1, max_iter + 1):
        previous = memberships.copy()
        centers = geometry.update_centers(X, memberships, m=m)
        squared = geometry.squared_dissimilarities(X, centers)
        memberships = memberships_from_squared_dissimilarities(squared, m=m)
        if np.max(np.abs(memberships - previous)) < tol:
            break

    objective = float(np.sum((memberships**m) * squared) / len(X))
    return FCMResult(
        labels=memberships.argmax(axis=1),
        memberships=memberships,
        centers=centers,
        iterations=iteration,
        objective=objective,
        m=m,
        minimum_center_distance=geometry.minimum_center_distance(centers),
        squared_dissimilarities=squared,
    )


def geometry_fcm(
    X: np.ndarray,
    n_clusters: int,
    *,
    geometry_name: GeometryName,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
    n_init: int = DEFAULT_FCM_N_INIT,
    max_attempts: int | None = None,
    min_cluster_size: int = 1,
    min_center_separation: float = DEFAULT_FCM_MIN_CENTER_SEPARATION,
) -> FCMResult:
    """Run multi-start FCM without hiding the selected geometry."""

    if n_clusters < 1 or n_init < 1:
        raise ValueError("n_clusters and n_init must be positive")
    if m <= 1.0:
        raise ValueError("m must be greater than 1")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")
    geometry = ExperimentalFcmGeometry(geometry_name)
    samples = geometry.prepare_samples(X)
    if n_clusters > len(samples):
        raise ValueError("n_clusters cannot exceed the number of samples")
    attempt_limit = n_init * 3 if max_attempts is None else max_attempts
    if attempt_limit < n_init:
        raise ValueError("max_attempts must be at least n_init")

    valid: list[FCMResult] = []
    attempted: list[FCMResult] = []
    for attempt in range(attempt_limit):
        result = _fit_once(
            samples,
            n_clusters,
            geometry=geometry,
            m=m,
            max_iter=max_iter,
            tol=tol,
            seed=seed + attempt * 1009,
        )
        attempted.append(result)
        sizes = np.bincount(result.labels, minlength=n_clusters)
        separation = result.minimum_center_distance or 0.0
        if sizes.min() >= min_cluster_size and separation >= min_center_separation:
            valid.append(result)
            if len(valid) >= n_init:
                break

    pool = valid if valid else attempted
    best = min(
        pool,
        key=lambda item: (
            float("inf") if item.objective is None else item.objective,
            -(item.minimum_center_distance or 0.0),
        ),
    )
    return FCMResult(
        labels=best.labels,
        memberships=best.memberships,
        centers=best.centers,
        iterations=best.iterations,
        objective=best.objective,
        m=m,
        n_init=n_init,
        attempts=len(attempted),
        valid_restarts=len(valid),
        restart_stability=_restart_stability(valid),
        minimum_center_distance=best.minimum_center_distance,
        squared_dissimilarities=best.squared_dissimilarities,
    )
