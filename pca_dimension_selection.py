from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from embedding_data import load_embeddings_from_json


DEFAULT_MAX_COMPONENTS = 512
DEFAULT_MIN_COMPONENTS = 32
DEFAULT_COMPONENT_STEP = 32
DEFAULT_K_VALUES = (15, 30)
DEFAULT_SHARP_PRESERVATION_GAIN = 0.05


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
    sharp_preservation_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_dimension": self.selected_dimension,
            "selection_reason": self.selection_reason,
            "fitted_dimension": self.fitted_dimension,
            "configuration": {
                "min_components": self.min_components,
                "component_step": self.component_step,
                "k_values": list(self.k_values),
                "sharp_preservation_gain": self.sharp_preservation_gain,
                "input_normalized": True,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _validate_inputs(
    X: np.ndarray,
    *,
    max_components: int,
    min_components: int,
    component_step: int,
    k_values: Sequence[int],
    sharp_preservation_gain: float,
) -> tuple[np.ndarray, tuple[int, ...], int, tuple[int, ...]]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    if max_components < 1:
        raise ValueError("max_components must be at least 1")
    if min_components < 1:
        raise ValueError("min_components must be at least 1")
    if component_step < 1:
        raise ValueError("component_step must be at least 1")
    if sharp_preservation_gain < 0.0 or sharp_preservation_gain > 1.0:
        raise ValueError("sharp_preservation_gain must be between 0 and 1")

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


def _neighbor_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Return exactly k non-self cosine neighbors for every row."""

    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
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


def _mean_neighbor_preservation(
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


def select_pca_dimension(
    X: np.ndarray,
    *,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    min_components: int = DEFAULT_MIN_COMPONENTS,
    component_step: int = DEFAULT_COMPONENT_STEP,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    sharp_preservation_gain: float = DEFAULT_SHARP_PRESERVATION_GAIN,
    seed: int = 42,
) -> PcaDimensionSelection:
    """Select a clustering PCA width from one maximum-width PCA fit.

    The input is L2-normalized, PCA is fitted once at up to ``max_components``,
    and every candidate is made by slicing the same projection with ``[:, :d]``.
    The first candidate whose mean k-NN preservation improves by at least
    ``sharp_preservation_gain`` over the previous candidate is selected. If no
    such jump exists, the largest evaluated dimension is used as a conservative
    fallback.
    """

    (
        X,
        normalized_k_values,
        fitted_dimension,
        candidate_dimensions,
    ) = _validate_inputs(
        X,
        max_components=max_components,
        min_components=min_components,
        component_step=component_step,
        k_values=k_values,
        sharp_preservation_gain=sharp_preservation_gain,
    )

    normalized_input = normalize(X, norm="l2")
    pca = PCA(
        n_components=fitted_dimension,
        svd_solver="full",
        random_state=seed,
    ).fit(normalized_input)
    maximum_projection = pca.transform(normalized_input)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    reference_neighbors = {
        k: _neighbor_indices(normalized_input, k) for k in normalized_k_values
    }
    candidates: list[PcaDimensionCandidate] = []
    previous_variance: float | None = None
    previous_preservation: float | None = None
    selected_dimension: int | None = None

    for dimension in candidate_dimensions:
        candidate_features = normalize(
            maximum_projection[:, :dimension],
            norm="l2",
        )
        preservation_by_k = {
            k: _mean_neighbor_preservation(
                reference_neighbors[k],
                _neighbor_indices(candidate_features, k),
            )
            for k in normalized_k_values
        }
        mean_preservation = float(np.mean(list(preservation_by_k.values())))
        explained_variance = float(cumulative_variance[dimension - 1])
        variance_gain = (
            None
            if previous_variance is None
            else explained_variance - previous_variance
        )
        preservation_gain = (
            None
            if previous_preservation is None
            else mean_preservation - previous_preservation
        )
        candidates.append(
            PcaDimensionCandidate(
                dimension=dimension,
                cumulative_explained_variance=explained_variance,
                knn_preservation_by_k=preservation_by_k,
                mean_knn_preservation=mean_preservation,
                explained_variance_gain=variance_gain,
                knn_preservation_gain=preservation_gain,
            )
        )
        if (
            selected_dimension is None
            and preservation_gain is not None
            and preservation_gain >= sharp_preservation_gain
        ):
            selected_dimension = dimension

        previous_variance = explained_variance
        previous_preservation = mean_preservation

    if selected_dimension is None:
        selected_dimension = candidate_dimensions[-1]
        selection_reason = "no_sharp_gain_use_maximum_evaluated_dimension"
    else:
        selection_reason = "first_sharp_knn_preservation_gain"

    selected_features = normalize(
        maximum_projection[:, :selected_dimension],
        norm="l2",
    )
    return PcaDimensionSelection(
        selected_dimension=selected_dimension,
        selection_reason=selection_reason,
        fitted_dimension=fitted_dimension,
        candidates=tuple(candidates),
        pca=pca,
        selected_features=selected_features,
        normalized_input=normalized_input,
        min_components=min_components,
        component_step=component_step,
        k_values=normalized_k_values,
        sharp_preservation_gain=sharp_preservation_gain,
    )


def transform_with_selected_dimension(
    X: np.ndarray,
    selection: PcaDimensionSelection,
) -> np.ndarray:
    """Apply the fitted maximum-width PCA and selected prefix to new rows."""

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if X.shape[1] != selection.pca.n_features_in_:
        raise ValueError("X has a different embedding dimension from the fitted PCA")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    projected = selection.pca.transform(normalize(X, norm="l2"))
    return normalize(projected[:, : selection.selected_dimension], norm="l2")


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
        "--sharp-preservation-gain",
        type=float,
        default=DEFAULT_SHARP_PRESERVATION_GAIN,
        help="Minimum mean k-NN preservation increase that selects a dimension.",
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
        sharp_preservation_gain=args.sharp_preservation_gain,
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
