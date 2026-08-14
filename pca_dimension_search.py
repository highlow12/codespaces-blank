"""Shared PCA-prefix search and neighborhood-preservation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pca_projection import (
    FittedPcaProjection,
    fit_normalized_pca_projection,
    validate_embedding_matrix,
)


DEFAULT_MAX_COMPONENTS = 512
DEFAULT_K_VALUES = (15, 30)
DEFAULT_MINIMUM_PRESERVATION_GAIN = 0.05
ALL_GAINS_REASON = "all_gains_meet_minimum_use_maximum_dimension"
PLATEAU_REASON = "first_below_minimum_gain_use_previous_dimension"
GLOBAL_KNEE_REASON = "global_preservation_knee_after_local_plateau"

CandidatePayload = TypeVar("CandidatePayload")


@dataclass(frozen=True)
class PcaDimensionCandidate:
    dimension: int
    cumulative_explained_variance: float
    knn_preservation_by_k: dict[int, float]
    mean_knn_preservation: float
    explained_variance_gain: float | None
    knn_preservation_gain: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "cumulative_explained_variance": self.cumulative_explained_variance,
            "knn_preservation_by_k": {
                str(k): value for k, value in self.knn_preservation_by_k.items()
            },
            "mean_knn_preservation": self.mean_knn_preservation,
            "explained_variance_gain": self.explained_variance_gain,
            "knn_preservation_gain": self.knn_preservation_gain,
        }
@dataclass(frozen=True)
class PcaPrefixSearch:
    projection: FittedPcaProjection
    candidate_dimensions: tuple[int, ...]
    k_values: tuple[int, ...]
    cumulative_variance: np.ndarray
    reference_neighbors: dict[int, np.ndarray]

    @property
    def fitted_dimension(self) -> int:
        return int(self.projection.pca.n_components_)


@dataclass(frozen=True)
class PcaPrefixEvaluation(Generic[CandidatePayload]):
    candidate: PcaDimensionCandidate
    payload: CandidatePayload


@dataclass(frozen=True)
class PcaPrefixSearchResult(Generic[CandidatePayload]):
    selected_dimension: int
    selection_reason: str
    evaluations: tuple[PcaPrefixEvaluation[CandidatePayload], ...]
    selected_payload: CandidatePayload


def _global_preservation_knee(
    evaluations: Sequence[PcaPrefixEvaluation[CandidatePayload]],
) -> PcaPrefixEvaluation[CandidatePayload]:
    """Return the saturation knee of the complete preservation curve.

    A single noisy gain can make the local plateau rule stop too early. The
    knee uses the complete, monotonized preservation curve and selects the
    point with the largest distance above the line joining its endpoints.
    Flat curves deliberately choose the smallest candidate.
    """

    if not evaluations:
        raise ValueError("evaluations must not be empty")
    if len(evaluations) <= 2:
        return evaluations[0]

    dimensions = np.asarray(
        [evaluation.candidate.dimension for evaluation in evaluations],
        dtype=np.float64,
    )
    preservation = np.maximum.accumulate(
        np.asarray(
            [
                evaluation.candidate.mean_knn_preservation
                for evaluation in evaluations
            ],
            dtype=np.float64,
        )
    )
    dimension_range = float(np.ptp(dimensions))
    preservation_range = float(np.ptp(preservation))
    if dimension_range <= 1e-12 or preservation_range <= 1e-12:
        return evaluations[0]

    normalized_dimensions = (dimensions - dimensions[0]) / dimension_range
    normalized_preservation = (
        preservation - preservation[0]
    ) / preservation_range
    knee_strength = normalized_preservation - normalized_dimensions
    best_strength = float(np.max(knee_strength))
    best_indices = np.flatnonzero(
        np.isclose(knee_strength, best_strength, rtol=1e-12, atol=1e-12)
    )
    return evaluations[int(best_indices[0])]


def validate_dimension_selection_inputs(
    X: np.ndarray,
    *,
    max_components: int,
    min_components: int,
    component_step: int,
    k_values: Sequence[int],
    minimum_preservation_gain: float,
) -> tuple[np.ndarray, tuple[int, ...], int, tuple[int, ...]]:
    X = validate_embedding_matrix(X)
    if max_components < 1:
        raise ValueError("max_components must be at least 1")
    if min_components < 1:
        raise ValueError("min_components must be at least 1")
    if component_step < 1:
        raise ValueError("component_step must be at least 1")
    if minimum_preservation_gain < 0.0 or minimum_preservation_gain > 1.0:
        raise ValueError("minimum_preservation_gain must be between 0 and 1")

    normalized_k_values = tuple(dict.fromkeys(int(k) for k in k_values))
    if not normalized_k_values:
        raise ValueError("k_values must contain at least one value")
    if any(k < 1 or k >= X.shape[0] for k in normalized_k_values):
        raise ValueError("Each k must be between 1 and n_samples - 1")

    fitted_dimension = min(max_components, X.shape[0], X.shape[1])
    candidate_dimensions = tuple(
        range(min_components, fitted_dimension + 1, component_step)
    )
    if not candidate_dimensions:
        raise ValueError(
            "The input cannot support the minimum PCA dimension: "
            f"min_components={min_components}, available={fitted_dimension}"
        )
    return X, normalized_k_values, fitted_dimension, candidate_dimensions


def neighbor_indices(
    X: np.ndarray,
    k: int,
    *,
    metric: str = "cosine",
) -> np.ndarray:
    """Return exactly k non-self neighbors for every row."""

    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm="brute",
    ).fit(X)
    raw_neighbors = model.kneighbors(X, return_distance=False)
    neighbors = np.empty((X.shape[0], k), dtype=np.int64)
    for row_index, row in enumerate(raw_neighbors):
        without_self = row[row != row_index]
        if without_self.size < k:
            raise RuntimeError("Could not obtain enough non-self neighbors")
        neighbors[row_index] = without_self[:k]
    return neighbors


def neighbor_indices_by_k(
    X: np.ndarray,
    k_values: Sequence[int],
    *,
    metric: str = "cosine",
) -> dict[int, np.ndarray]:
    """Compute one maximum-neighbor search and expose the requested prefixes.

    ``NearestNeighbors.kneighbors`` repeats the pairwise search when called
    separately for each k. The first k non-self neighbors from a maximum-k
    search are the same neighbors needed by every smaller k for the continuous
    embedding data used by the clustering pipeline. Keeping the prefixes as
    views also avoids copying the small neighbor-index arrays.
    """

    normalized_k_values = tuple(dict.fromkeys(int(k) for k in k_values))
    if not normalized_k_values:
        raise ValueError("k_values must contain at least one value")
    maximum_k = max(normalized_k_values)
    neighbors = neighbor_indices(X, maximum_k, metric=metric)
    return {
        k: neighbors[:, :k]
        for k in normalized_k_values
    }


def mean_neighbor_preservation(
    reference_neighbors: np.ndarray,
    candidate_neighbors: np.ndarray,
) -> float:
    k = reference_neighbors.shape[1]
    preserved = 0
    for reference_row, candidate_row in zip(
        reference_neighbors,
        candidate_neighbors,
        strict=True,
    ):
        preserved += np.intersect1d(
            reference_row,
            candidate_row,
            assume_unique=True,
        ).size
    return float(preserved / (reference_neighbors.shape[0] * k))


def prepare_pca_prefix_search(
    X: np.ndarray,
    *,
    max_components: int,
    min_components: int,
    component_step: int,
    k_values: Sequence[int],
    minimum_preservation_gain: float,
    seed: int,
) -> PcaPrefixSearch:
    """Fit the shared maximum-width PCA and reference neighborhoods once."""

    (
        X,
        normalized_k_values,
        fitted_dimension,
        candidate_dimensions,
    ) = validate_dimension_selection_inputs(
        X,
        max_components=max_components,
        min_components=min_components,
        component_step=component_step,
        k_values=k_values,
        minimum_preservation_gain=minimum_preservation_gain,
    )
    projection = fit_normalized_pca_projection(
        X,
        n_components=fitted_dimension,
        seed=seed,
        svd_solver="full",
    )
    return PcaPrefixSearch(
        projection=projection,
        candidate_dimensions=candidate_dimensions,
        k_values=normalized_k_values,
        cumulative_variance=np.cumsum(projection.pca.explained_variance_ratio_),
        reference_neighbors=neighbor_indices_by_k(
            projection.normalized_input,
            normalized_k_values,
        ),
    )


def evaluate_pca_prefixes(
    search: PcaPrefixSearch,
    candidate_factory: Callable[
        [int, np.ndarray],
        tuple[np.ndarray, CandidatePayload],
    ],
    *,
    minimum_preservation_gain: float,
    neighbor_metric: str,
    stop_at_plateau: bool,
    normalize_pca_output: bool = True,
) -> PcaPrefixSearchResult[CandidatePayload]:
    """Score PCA prefixes and select a stable preservation saturation point.

    Callers that stop at the plateau retain the inexpensive local rule. Full
    searches also inspect the global preservation knee so a noisy early gain
    cannot prematurely discard useful dimensions.
    """

    evaluations: list[PcaPrefixEvaluation[CandidatePayload]] = []
    selected_evaluation: PcaPrefixEvaluation[CandidatePayload] | None = None
    previous_evaluation: PcaPrefixEvaluation[CandidatePayload] | None = None

    for dimension in search.candidate_dimensions:
        pca_features = (
            search.projection.normalized_prefix(dimension)
            if normalize_pca_output
            else search.projection.projected[:, :dimension]
        )
        score_features, payload = candidate_factory(dimension, pca_features)
        score_features = validate_embedding_matrix(
            score_features,
            name="candidate_features",
        )
        if score_features.shape[0] != search.projection.normalized_input.shape[0]:
            raise ValueError("candidate_features must contain one row per input")

        candidate_neighbors = neighbor_indices_by_k(
            score_features,
            search.k_values,
            metric=neighbor_metric,
        )
        preservation_by_k = {
            k: mean_neighbor_preservation(
                search.reference_neighbors[k],
                candidate_neighbors[k],
            )
            for k in search.k_values
        }
        mean_preservation = float(np.mean(list(preservation_by_k.values())))
        explained_variance = float(search.cumulative_variance[dimension - 1])
        previous_candidate = (
            None if previous_evaluation is None else previous_evaluation.candidate
        )
        candidate = PcaDimensionCandidate(
            dimension=dimension,
            cumulative_explained_variance=explained_variance,
            knn_preservation_by_k=preservation_by_k,
            mean_knn_preservation=mean_preservation,
            explained_variance_gain=(
                None
                if previous_candidate is None
                else explained_variance
                - previous_candidate.cumulative_explained_variance
            ),
            knn_preservation_gain=(
                None
                if previous_candidate is None
                else mean_preservation - previous_candidate.mean_knn_preservation
            ),
        )
        evaluation = PcaPrefixEvaluation(candidate=candidate, payload=payload)
        evaluations.append(evaluation)
        if (
            selected_evaluation is None
            and previous_evaluation is not None
            and candidate.knn_preservation_gain is not None
            and candidate.knn_preservation_gain < minimum_preservation_gain
        ):
            selected_evaluation = previous_evaluation
            if stop_at_plateau:
                break
        previous_evaluation = evaluation

    if not evaluations:
        raise RuntimeError("No PCA candidate was evaluated")
    if selected_evaluation is None:
        selected_evaluation = evaluations[-1]
        selection_reason = ALL_GAINS_REASON
    else:
        selection_reason = PLATEAU_REASON
        if not stop_at_plateau:
            knee_evaluation = _global_preservation_knee(evaluations)
            if (
                knee_evaluation.candidate.dimension
                > selected_evaluation.candidate.dimension
            ):
                selected_evaluation = knee_evaluation
                selection_reason = GLOBAL_KNEE_REASON
    return PcaPrefixSearchResult(
        selected_dimension=selected_evaluation.candidate.dimension,
        selection_reason=selection_reason,
        evaluations=tuple(evaluations),
        selected_payload=selected_evaluation.payload,
    )
