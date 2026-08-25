"""Portable normalized PCA and k-NN preservation based dimension selection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .types import PcaSelection

DEFAULT_MAX_COMPONENTS = 512
DEFAULT_MIN_COMPONENTS = 32
DEFAULT_COMPONENT_STEP = 32
DEFAULT_K_VALUES = (15, 30)
DEFAULT_MINIMUM_PRESERVATION_GAIN = 0.05


def validate_embeddings(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("embeddings must be a non-empty 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must contain only finite values")
    if np.any(np.linalg.norm(matrix, axis=1) <= 1e-12):
        raise ValueError("embeddings must have non-zero L2 norm per row")
    return matrix


def _neighbors(values: np.ndarray, k: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    raw = model.fit(values).kneighbors(values, return_distance=False)
    result = np.empty((len(values), k), dtype=np.int64)
    for row_index, row in enumerate(raw):
        row = row[row != row_index]
        if len(row) < k:
            raise RuntimeError("could not obtain enough non-self neighbors")
        result[row_index] = row[:k]
    return result


def _preservation(reference: np.ndarray, candidate: np.ndarray) -> float:
    total = 0
    for expected, actual in zip(reference, candidate):
        total += np.intersect1d(expected, actual, assume_unique=True).size
    return float(total / (reference.shape[0] * reference.shape[1]))


def _global_knee(candidates: list[dict[str, Any]]) -> int:
    if len(candidates) <= 2:
        return 0
    dimensions = np.asarray([item["dimension"] for item in candidates], dtype=float)
    curve = np.maximum.accumulate(
        np.asarray([item["mean_knn_preservation"] for item in candidates], dtype=float)
    )
    if np.ptp(dimensions) <= 1e-12 or np.ptp(curve) <= 1e-12:
        return 0
    score = (curve - curve[0]) / np.ptp(curve) - (dimensions - dimensions[0]) / np.ptp(dimensions)
    return int(np.flatnonzero(np.isclose(score, score.max(), atol=1e-12))[0])


def fit_pca(
    values: Any,
    *,
    components: int | None = None,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    min_components: int = DEFAULT_MIN_COMPONENTS,
    component_step: int = DEFAULT_COMPONENT_STEP,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    minimum_preservation_gain: float = DEFAULT_MINIMUM_PRESERVATION_GAIN,
    seed: int = 42,
) -> PcaSelection:
    """Fit normalized PCA and select a prefix by neighborhood preservation.

    This mirrors the research selector while keeping the implementation
    independent of the repository's CLI/data-loading modules.
    """

    matrix = validate_embeddings(values)
    if components is not None and components < 1:
        raise ValueError("components must be at least 1")
    if min_components < 1 or component_step < 1:
        raise ValueError("min_components and component_step must be positive")
    if not 0.0 <= minimum_preservation_gain <= 1.0:
        raise ValueError("minimum_preservation_gain must be between 0 and 1")

    normalized = normalize(matrix, norm="l2")
    fitted_dimension = min(
        components if components is not None else max_components,
        matrix.shape[0],
        matrix.shape[1],
    )
    if fitted_dimension < 1:
        raise ValueError("PCA cannot fit an empty feature space")
    pca = PCA(n_components=fitted_dimension, random_state=seed, svd_solver="full").fit(normalized)
    raw_features = np.asarray(pca.transform(normalized), dtype=np.float64)

    if components is not None:
        dimensions = (fitted_dimension,)
        reason = "fixed_dimension"
    else:
        effective_min = min(min_components, fitted_dimension)
        dimensions = tuple(range(effective_min, fitted_dimension + 1, component_step))
        if not dimensions:
            dimensions = (fitted_dimension,)
        elif dimensions[-1] != fitted_dimension:
            dimensions = (*dimensions, fitted_dimension)
        reason = "all_gains_meet_minimum"

    effective_k = tuple(dict.fromkeys(int(k) for k in k_values if 1 <= int(k) < len(matrix)))
    if not effective_k:
        effective_k = (max(1, min(len(matrix) - 1, len(matrix) // 2)),)
    references = {k: _neighbors(normalized, k) for k in effective_k}
    candidates: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    selected_index: int | None = None
    for dimension in dimensions:
        projected = normalize(raw_features[:, :dimension], norm="l2")
        scores = {
            k: _preservation(references[k], _neighbors(projected, k))
            for k in effective_k
        }
        mean_score = float(np.mean(list(scores.values())))
        candidate = {
            "dimension": int(dimension),
            "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_[:dimension])),
            "knn_preservation_by_k": {str(k): float(score) for k, score in scores.items()},
            "mean_knn_preservation": mean_score,
            "knn_preservation_gain": (
                None if previous is None else mean_score - previous["mean_knn_preservation"]
            ),
        }
        candidates.append(candidate)
        if (
            selected_index is None
            and previous is not None
            and candidate["knn_preservation_gain"] < minimum_preservation_gain
        ):
            selected_index = len(candidates) - 2
            reason = "first_below_minimum_gain_use_previous"
        previous = candidate

    if selected_index is None:
        selected_index = len(candidates) - 1
    elif components is None:
        knee_index = _global_knee(candidates)
        if knee_index > selected_index:
            selected_index = knee_index
            reason = "global_preservation_knee_after_local_plateau"

    selected_dimension = int(candidates[selected_index]["dimension"])
    return PcaSelection(
        selected_dimension=selected_dimension,
        fitted_dimension=int(fitted_dimension),
        selection_reason=reason,
        candidates=tuple(candidates),
        features=raw_features[:, :selected_dimension],
        normalized_input=normalized,
        pca=pca,
    )
