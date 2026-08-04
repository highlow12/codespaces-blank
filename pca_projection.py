from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class FittedPcaProjection:
    """A single normalized PCA fit and its full-width training projection."""

    normalized_input: np.ndarray
    pca: PCA
    projected: np.ndarray

    def normalized_prefix(self, dimension: int | None = None) -> np.ndarray:
        if dimension is not None and not 1 <= dimension <= self.projected.shape[1]:
            raise ValueError("dimension must be between 1 and the fitted PCA width")
        projected = (
            self.projected
            if dimension is None
            else self.projected[:, :dimension]
        )
        return normalize(projected, norm="l2")


def validate_embedding_matrix(
    values: np.ndarray,
    *,
    name: str = "X",
    expected_features: int | None = None,
) -> np.ndarray:
    """Return a finite float matrix suitable for PCA-based pipelines."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            f"{name} has {matrix.shape[1]} features; expected {expected_features}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def fit_normalized_pca_projection(
    values: np.ndarray,
    *,
    n_components: int,
    seed: int,
    svd_solver: str = "auto",
    name: str = "X",
) -> FittedPcaProjection:
    """Normalize inputs, fit PCA once, and retain the full projection."""

    matrix = validate_embedding_matrix(values, name=name)
    if n_components < 1:
        raise ValueError("n_components must be at least 1")
    component_count = min(n_components, matrix.shape[0], matrix.shape[1])
    normalized_input = normalize(matrix, norm="l2")
    pca = PCA(
        n_components=component_count,
        random_state=seed,
        svd_solver=svd_solver,
    ).fit(normalized_input)
    return FittedPcaProjection(
        normalized_input=normalized_input,
        pca=pca,
        projected=pca.transform(normalized_input),
    )


def transform_normalized_pca_projection(
    values: np.ndarray,
    pca: PCA,
    *,
    dimension: int | None = None,
    name: str = "X",
) -> np.ndarray:
    """Apply a fitted PCA and L2-normalize its full output or one prefix."""

    matrix = validate_embedding_matrix(
        values,
        name=name,
        expected_features=int(pca.n_features_in_),
    )
    if dimension is not None and not 1 <= dimension <= int(pca.n_components_):
        raise ValueError("dimension must be between 1 and the fitted PCA width")
    projected = pca.transform(normalize(matrix, norm="l2"))
    if dimension is not None:
        projected = projected[:, :dimension]
    return normalize(projected, norm="l2")
