"""Clustering-specific PCA dimension selection and command-line entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA

from embedding_data import load_embeddings_from_json
from pca_dimension_search import (
    DEFAULT_K_VALUES,
    DEFAULT_MAX_COMPONENTS,
    DEFAULT_MINIMUM_PRESERVATION_GAIN,
    PcaDimensionCandidate,
    PcaPrefixEvaluation,
    PcaPrefixSearch,
    PcaPrefixSearchResult,
    evaluate_pca_prefixes,
    mean_neighbor_preservation,
    neighbor_indices,
    prepare_pca_prefix_search,
    validate_dimension_selection_inputs,
)
from pca_projection import transform_normalized_pca_projection


DEFAULT_MIN_COMPONENTS = 32
DEFAULT_COMPONENT_STEP = 32


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
