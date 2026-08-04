from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from embedding_data import load_embeddings_from_json
from pca_projection import (
    FittedPcaProjection,
    fit_normalized_pca_projection,
    transform_normalized_pca_projection,
    validate_embedding_matrix,
)


DEFAULT_MAX_COMPONENTS = 512
DEFAULT_MIN_COMPONENTS = 32
DEFAULT_COMPONENT_STEP = 32
DEFAULT_K_VALUES = (15, 30)
DEFAULT_MINIMUM_PRESERVATION_GAIN = 0.05
ALL_GAINS_REASON = "all_gains_meet_minimum_use_maximum_dimension"
PLATEAU_REASON = "first_below_minimum_gain_use_previous_dimension"

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
class PcaDimensionSelection:
    selected_dimension: int
    selection_reason: str
    fitted_dimension: int
    candidates: tuple[PcaDimensionCandidate, ...]
    pca: PCA
    selected_features: np.ndarray
    normalized_input: np.ndarray
    min_components: int
    component_step: int
    k_values: tuple[int, ...]
    minimum_preservation_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_dimension": self.selected_dimension,
            "selection_reason": self.selection_reason,
            "fitted_dimension": self.fitted_dimension,
            "configuration": {
                "min_components": self.min_components,
                "component_step": self.component_step,
                "k_values": list(self.k_values),
                "minimum_preservation_gain": self.minimum_preservation_gain,
                "input_normalized": True,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
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
        reference_neighbors={
            k: neighbor_indices(projection.normalized_input, k)
            for k in normalized_k_values
        },
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
) -> PcaPrefixSearchResult[CandidatePayload]:
    """Score PCA prefixes and select the last dimension before a plateau."""

    evaluations: list[PcaPrefixEvaluation[CandidatePayload]] = []
    selected_evaluation: PcaPrefixEvaluation[CandidatePayload] | None = None
    previous_evaluation: PcaPrefixEvaluation[CandidatePayload] | None = None

    for dimension in search.candidate_dimensions:
        pca_features = search.projection.normalized_prefix(dimension)
        score_features, payload = candidate_factory(dimension, pca_features)
        score_features = validate_embedding_matrix(
            score_features,
            name="candidate_features",
        )
        if score_features.shape[0] != search.projection.normalized_input.shape[0]:
            raise ValueError("candidate_features must contain one row per input")

        preservation_by_k = {
            k: mean_neighbor_preservation(
                search.reference_neighbors[k],
                neighbor_indices(score_features, k, metric=neighbor_metric),
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
    return PcaPrefixSearchResult(
        selected_dimension=selected_evaluation.candidate.dimension,
        selection_reason=selection_reason,
        evaluations=tuple(evaluations),
        selected_payload=selected_evaluation.payload,
    )


def select_pca_dimension(
    X: np.ndarray,
    *,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    min_components: int = DEFAULT_MIN_COMPONENTS,
    component_step: int = DEFAULT_COMPONENT_STEP,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    minimum_preservation_gain: float = DEFAULT_MINIMUM_PRESERVATION_GAIN,
    seed: int = 42,
) -> PcaDimensionSelection:
    """Select a clustering PCA width from one maximum-width PCA fit.

    The input is L2-normalized, PCA is fitted once at up to ``max_components``,
    and every candidate is made by slicing the same projection with ``[:, :d]``.
    Candidates increase by ``component_step``. At the first candidate whose
    mean k-NN preservation gain is less than ``minimum_preservation_gain``, the
    previous dimension is selected. This chooses the last dimension before the
    improvement plateaus. If every gain meets the minimum, the largest
    evaluated dimension is selected.
    """

    search = prepare_pca_prefix_search(
        X,
        max_components=max_components,
        min_components=min_components,
        component_step=component_step,
        k_values=k_values,
        minimum_preservation_gain=minimum_preservation_gain,
        seed=seed,
    )
    result = evaluate_pca_prefixes(
        search,
        lambda _dimension, features: (features, features),
        minimum_preservation_gain=minimum_preservation_gain,
        neighbor_metric="cosine",
        stop_at_plateau=False,
    )
    return PcaDimensionSelection(
        selected_dimension=result.selected_dimension,
        selection_reason=result.selection_reason,
        fitted_dimension=search.fitted_dimension,
        candidates=tuple(evaluation.candidate for evaluation in result.evaluations),
        pca=search.projection.pca,
        selected_features=result.selected_payload,
        normalized_input=search.projection.normalized_input,
        min_components=min_components,
        component_step=component_step,
        k_values=search.k_values,
        minimum_preservation_gain=minimum_preservation_gain,
    )


def transform_with_selected_dimension(
    X: np.ndarray,
    selection: PcaDimensionSelection,
) -> np.ndarray:
    """Apply the fitted maximum-width PCA and selected prefix to new rows."""

    return transform_normalized_pca_projection(
        X,
        selection.pca,
        dimension=selection.selected_dimension,
    )


def _print_selection(selection: PcaDimensionSelection) -> None:
    print("dimension  explained_variance  mean_knn_preservation  preservation_gain")
    for candidate in selection.candidates:
        gain = (
            "-"
            if candidate.knn_preservation_gain is None
            else f"{candidate.knn_preservation_gain:.4f}"
        )
        print(
            f"{candidate.dimension:9d}  "
            f"{candidate.cumulative_explained_variance:18.4f}  "
            f"{candidate.mean_knn_preservation:21.4f}  "
            f"{gain:>17}"
        )
    print(
        f"\nSelected PCA dimension: {selection.selected_dimension} "
        f"({selection.selection_reason})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a clustering PCA dimension using variance and k-NN preservation."
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-components", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument("--min-components", type=int, default=DEFAULT_MIN_COMPONENTS)
    parser.add_argument("--component-step", type=int, default=DEFAULT_COMPONENT_STEP)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--minimum-preservation-gain",
        type=float,
        default=DEFAULT_MINIMUM_PRESERVATION_GAIN,
        help=(
            "Keep increasing PCA width while mean k-NN preservation improves "
            "by at least this amount."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, _ = load_embeddings_from_json(args.input_json)
    selection = select_pca_dimension(
        X,
        max_components=args.max_components,
        min_components=args.min_components,
        component_step=args.component_step,
        k_values=args.k,
        minimum_preservation_gain=args.minimum_preservation_gain,
        seed=args.seed,
    )
    _print_selection(selection)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(selection.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Selection report saved to: {args.output}")


if __name__ == "__main__":
    main()
