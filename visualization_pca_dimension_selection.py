from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.decomposition import PCA

from cluster_visualization import (
    DEFAULT_CLUSTER_TARGET_WEIGHT,
    _make_umap_reducer,
    _validate_cluster_target,
    build_cluster_supervision,
    load_assignments,
)
from embedding_data import load_embeddings_from_json
from pca_dimension_selection import (
    DEFAULT_K_VALUES,
    DEFAULT_MAX_COMPONENTS,
    DEFAULT_MINIMUM_PRESERVATION_GAIN,
    PcaDimensionCandidate,
    evaluate_pca_prefixes,
    prepare_pca_prefix_search,
)
from pca_projection import transform_normalized_pca_projection


UmapFactory = Callable[..., Any]
DEFAULT_VISUALIZATION_MIN_COMPONENTS = 16
DEFAULT_VISUALIZATION_COMPONENT_STEP = 16


@dataclass(frozen=True)
class VisualizationPcaDimensionSelection:
    selected_dimension: int
    selection_reason: str
    fitted_dimension: int
    candidates: tuple[PcaDimensionCandidate, ...]
    pca: PCA
    umap: Any
    selected_coordinates: np.ndarray
    min_components: int
    component_step: int
    k_values: tuple[int, ...]
    minimum_preservation_gain: float
    umap_configuration: dict[str, Any]

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
                "neighbor_reference": "original_embeddings",
                "neighbor_candidate": "umap_2d",
                "umap": self.umap_configuration,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def select_visualization_pca_dimension(
    X: np.ndarray,
    *,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    min_components: int = DEFAULT_VISUALIZATION_MIN_COMPONENTS,
    component_step: int = DEFAULT_VISUALIZATION_COMPONENT_STEP,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    minimum_preservation_gain: float = DEFAULT_MINIMUM_PRESERVATION_GAIN,
    n_neighbors: int = 15,
    min_dist: float = 0.02,
    metric: str = "cosine",
    spread: float = 0.85,
    densmap: bool = False,
    cluster_target: np.ndarray | None = None,
    cluster_target_metric: str | None = None,
    cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
    seed: int = 42,
    umap_factory: UmapFactory = _make_umap_reducer,
) -> VisualizationPcaDimensionSelection:
    """Select the PCA prefix used before UMAP-2 visualization.

    PCA is fitted once at the maximum supported width. Each configured
    prefix is normalized and passed through an identically configured UMAP-2.
    The score is k-NN preservation between the normalized original embeddings
    and the two-dimensional UMAP coordinates. When the score gain first falls
    below the configured minimum, the previous PCA dimension is selected.
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
    sample_count = search.projection.normalized_input.shape[0]
    if n_neighbors < 2 or n_neighbors >= sample_count:
        raise ValueError("n_neighbors must be between 2 and n_samples - 1")
    if min_dist < 0.0:
        raise ValueError("min_dist must be non-negative")
    if spread <= 0.0:
        raise ValueError("spread must be positive")

    target, target_metric, target_weight = _validate_cluster_target(
        cluster_target,
        cluster_target_metric,
        cluster_target_weight,
        n_samples=sample_count,
    )

    umap_configuration = {
        "n_components": 2,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "spread": spread,
        "densmap": densmap,
        "random_state": seed,
        "target_metric": target_metric,
        "target_weight": target_weight,
    }

    def project_candidate(
        _dimension: int,
        pca_features: np.ndarray,
    ) -> tuple[np.ndarray, tuple[Any, np.ndarray]]:
        reducer = umap_factory(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            spread=spread,
            densmap=densmap,
            random_state=seed,
            target_metric=target_metric,
            target_weight=target_weight,
        )
        coordinates = np.asarray(
            reducer.fit_transform(pca_features, y=target),
            dtype=np.float64,
        )
        if coordinates.shape != (sample_count, 2):
            raise ValueError("UMAP must return one two-dimensional row per sample")
        return coordinates, (reducer, coordinates)

    result = evaluate_pca_prefixes(
        search,
        project_candidate,
        minimum_preservation_gain=minimum_preservation_gain,
        neighbor_metric="euclidean",
        stop_at_plateau=True,
    )
    selected_umap, selected_coordinates = result.selected_payload

    return VisualizationPcaDimensionSelection(
        selected_dimension=result.selected_dimension,
        selection_reason=result.selection_reason,
        fitted_dimension=search.fitted_dimension,
        candidates=tuple(evaluation.candidate for evaluation in result.evaluations),
        pca=search.projection.pca,
        umap=selected_umap,
        selected_coordinates=selected_coordinates,
        min_components=min_components,
        component_step=component_step,
        k_values=search.k_values,
        minimum_preservation_gain=minimum_preservation_gain,
        umap_configuration=umap_configuration,
    )


def transform_with_selected_visualization(
    X: np.ndarray,
    selection: VisualizationPcaDimensionSelection,
) -> np.ndarray:
    """Transform new embeddings with the selected PCA prefix and UMAP."""

    features = transform_normalized_pca_projection(
        X,
        selection.pca,
        dimension=selection.selected_dimension,
    )
    return np.asarray(selection.umap.transform(features), dtype=np.float64)


def _print_selection(selection: VisualizationPcaDimensionSelection) -> None:
    print("dimension  explained_variance  umap_knn_preservation  preservation_gain")
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
        f"\nSelected visualization PCA dimension: {selection.selected_dimension} "
        f"({selection.selection_reason})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the PCA dimension before UMAP using explained variance "
            "and original-to-UMAP k-NN preservation."
        )
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--assignments-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-components", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument(
        "--min-components",
        type=int,
        default=DEFAULT_VISUALIZATION_MIN_COMPONENTS,
    )
    parser.add_argument(
        "--component-step",
        type=int,
        default=DEFAULT_VISUALIZATION_COMPONENT_STEP,
    )
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--minimum-preservation-gain",
        type=float,
        default=DEFAULT_MINIMUM_PRESERVATION_GAIN,
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.02)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--spread", type=float, default=0.85)
    parser.add_argument("--densmap", action="store_true", default=False)
    parser.add_argument(
        "--cluster-target-weight",
        type=float,
        default=DEFAULT_CLUSTER_TARGET_WEIGHT,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, metadata = load_embeddings_from_json(args.input_json)
    cluster_target = None
    cluster_target_metric = None
    target_description = "none"
    if args.assignments_csv is not None:
        assignments = load_assignments(args.assignments_csv)
        merged = metadata[["id"]].merge(
            assignments,
            on="id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if len(merged) != len(metadata) or not (merged["_merge"] == "both").all():
            raise ValueError("Assignments and embeddings do not align by id")
        merged = merged.drop(columns="_merge")
        cluster_target, cluster_target_metric, target_description = (
            build_cluster_supervision(merged)
        )

    selection = select_visualization_pca_dimension(
        X,
        max_components=args.max_components,
        min_components=args.min_components,
        component_step=args.component_step,
        k_values=args.k,
        minimum_preservation_gain=args.minimum_preservation_gain,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        spread=args.spread,
        densmap=args.densmap,
        cluster_target=cluster_target,
        cluster_target_metric=cluster_target_metric,
        cluster_target_weight=args.cluster_target_weight,
        seed=args.seed,
    )
    _print_selection(selection)
    report = selection.to_dict()
    report["configuration"]["cluster_target"] = target_description
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Selection report saved to: {args.output}")


if __name__ == "__main__":
    main()
