from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


DEFAULT_PROJECTION_SUPPORT_GAP_FRACTION = 0.40


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


@dataclass(frozen=True)
class PcaPrefixTransformer:
    """Expose a selected PCA prefix as a reusable sklearn-like transformer."""

    base_pca: PCA
    dimension: int

    def __post_init__(self) -> None:
        fitted_width = int(self.base_pca.n_components_)
        if not 1 <= self.dimension <= fitted_width:
            raise ValueError(
                "dimension must be between 1 and the fitted PCA width"
            )

    @property
    def n_features_in_(self) -> int:
        return int(self.base_pca.n_features_in_)

    @property
    def n_components_(self) -> int:
        return int(self.dimension)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.base_pca.transform(values))[:, : self.dimension]


def _base_pca_and_dimension(
    pca: PCA | PcaPrefixTransformer,
) -> tuple[PCA, int]:
    if isinstance(pca, PcaPrefixTransformer):
        return pca.base_pca, int(pca.dimension)
    return pca, int(pca.n_components_)


def validate_embedding_matrix(
    values: np.ndarray,
    *,
    name: str = "X",
    expected_features: int | None = None,
) -> np.ndarray:
    """Return a finite float32/float64 matrix suitable for PCA pipelines."""

    matrix = np.asarray(values)
    if matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
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


def pca_projection_support(
    values: np.ndarray,
    pca: PCA | PcaPrefixTransformer,
    *,
    name: str = "X",
) -> np.ndarray:
    """Measure how much centered input energy lies in the selected PCA space.

    The clustering projection is normalized after PCA, which intentionally
    discards projection magnitude.  That magnitude is nevertheless useful for
    OOD detection: an unrelated embedding can point somewhere in the PCA space
    after normalization while having very little support in that space.
    """

    base_pca, dimension = _base_pca_and_dimension(pca)
    matrix = validate_embedding_matrix(
        values,
        name=name,
        expected_features=int(base_pca.n_features_in_),
    )
    normalized_input = normalize(matrix, norm="l2")
    centered_input = normalized_input - np.asarray(base_pca.mean_)
    projected = np.asarray(base_pca.transform(normalized_input))[:, :dimension]
    support = np.linalg.norm(projected, axis=1) / np.maximum(
        np.linalg.norm(centered_input, axis=1),
        1e-12,
    )
    return np.clip(support, 0.0, 1.0)


def calibrate_pca_projection_support_threshold(
    values: np.ndarray,
    pca: PCA | PcaPrefixTransformer,
    *,
    distance_z: float = 3.5,
    gap_fraction: float = DEFAULT_PROJECTION_SUPPORT_GAP_FRACTION,
) -> float:
    """Calibrate a conservative natural-OOD threshold without a fixed quota.

    The threshold sits partway between the expected support of an isotropic
    random direction and a robust lower fence of the fitted data.  If those
    references are not separated, projection-support rejection is disabled.
    """

    if distance_z < 0.0:
        raise ValueError("distance_z must be non-negative")
    if not 0.0 <= gap_fraction <= 1.0:
        raise ValueError("gap_fraction must be between 0 and 1")

    support = pca_projection_support(values, pca)
    median = float(np.median(support))
    mad = float(np.median(np.abs(support - median)))
    robust_lower = max(0.0, median - distance_z * 1.4826 * mad)
    base_pca, dimension = _base_pca_and_dimension(pca)
    isotropic_support = float(
        np.sqrt(dimension / int(base_pca.n_features_in_))
    )
    if robust_lower <= isotropic_support + 1e-12:
        return 0.0
    return float(
        isotropic_support
        + gap_fraction * (robust_lower - isotropic_support)
    )
