"""CLI and deterministic CSV/JSON artifacts for the Phase 1-3 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from embedding_data import load_embeddings_from_json, sample_embedding_batch
from hdbscan_membership_comparison import (
    DEFAULT_MAX_PCA_COMPONENTS,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_NEIGHBOR_COUNT,
    DEFAULT_UMAP_COMPONENTS,
    DEFAULT_UMAP_N_NEIGHBORS,
    HdbscanMembershipComparisonResult,
    fit_hdbscan_membership_comparison,
)


DEFAULT_MULTI_LEAF_AFFINITY_THRESHOLD = 0.25
DEFAULT_METHOD_DIFFERENCE_THRESHOLD = 0.10


def _validate_metadata(metadata: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    if len(metadata) != n_samples:
        raise ValueError(
            f"metadata must contain {n_samples} rows, got {len(metadata)}"
        )
    return metadata.reset_index(drop=True).copy()


def build_assignments(
    metadata: pd.DataFrame,
    result: HdbscanMembershipComparisonResult,
) -> pd.DataFrame:
    """Build the row-aligned comparison artifact without normalization."""

    frame = _validate_metadata(metadata, len(result.leaf_labels))
    frame["hdbscan_leaf_label"] = result.leaf_labels
    frame["hdbscan_probability"] = result.probabilities
    frame["hdbscan_outlier_score"] = result.outlier_scores
    frame["native_unexplained"] = result.native_unexplained
    frame["native_max_affinity"] = result.native_max_affinity
    frame["native_recommended_leaf"] = result.native_recommended_labels
    frame["pca_exact_knn_unexplained"] = result.exact_knn.unexplained
    frame["pca_exact_knn_max_affinity"] = result.exact_knn.max_affinity
    frame["pca_exact_knn_recommended_leaf"] = result.exact_knn.recommended_labels
    for cluster in range(result.cluster_count):
        frame[f"native_membership_{cluster}"] = result.native_memberships[:, cluster]
        frame[f"pca_exact_knn_affinity_{cluster}"] = (
            result.exact_knn.affinities[:, cluster]
        )
    return frame


def _second_largest(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=np.float64)
    return np.partition(values, -2, axis=1)[:, -2]


def build_boundary_cases(
    metadata: pd.DataFrame,
    result: HdbscanMembershipComparisonResult,
    *,
    multi_leaf_threshold: float = DEFAULT_MULTI_LEAF_AFFINITY_THRESHOLD,
    method_difference_threshold: float = DEFAULT_METHOD_DIFFERENCE_THRESHOLD,
) -> pd.DataFrame:
    """Rank review candidates with disagreements or independent overlap."""

    if not 0.0 <= multi_leaf_threshold <= 1.0:
        raise ValueError("multi_leaf_threshold must be between 0 and 1")
    if method_difference_threshold < 0.0:
        raise ValueError("method_difference_threshold must be non-negative")

    assignments = build_assignments(metadata, result)
    native = result.native_memberships
    exact = result.exact_knn.affinities
    native_second = _second_largest(native)
    exact_second = _second_largest(exact)
    max_gap = np.abs(result.native_max_affinity - result.exact_knn.max_affinity)
    unexplained_gap = np.abs(
        result.native_unexplained - result.exact_knn.unexplained
    )
    recommendation_disagreement = (
        result.native_recommended_labels != result.exact_knn.recommended_labels
    )
    exact_multi_leaf = exact_second >= multi_leaf_threshold
    method_disagreement = (
        recommendation_disagreement
        | (max_gap >= method_difference_threshold)
        | (unexplained_gap >= method_difference_threshold)
    )
    review_mask = method_disagreement | exact_multi_leaf
    score = max_gap + unexplained_gap + exact_second
    reasons = np.full(len(assignments), "", dtype=object)
    for row_index in np.flatnonzero(review_mask):
        row_reasons: list[str] = []
        if method_disagreement[row_index]:
            row_reasons.append("native_exact_disagreement")
        if exact_multi_leaf[row_index]:
            row_reasons.append("exact_multi_leaf_overlap")
        reasons[row_index] = ";".join(row_reasons)

    boundary = assignments.loc[review_mask].copy()
    boundary["native_second_affinity"] = native_second[review_mask]
    boundary["pca_exact_knn_second_affinity"] = exact_second[review_mask]
    boundary["native_exact_max_affinity_gap"] = max_gap[review_mask]
    boundary["native_exact_unexplained_gap"] = unexplained_gap[review_mask]
    boundary["exact_multi_leaf_count"] = np.sum(
        exact[review_mask] >= multi_leaf_threshold, axis=1
    ) if result.cluster_count else np.zeros(int(np.sum(review_mask)), dtype=np.int64)
    boundary["boundary_score"] = score[review_mask]
    boundary["boundary_reason"] = reasons[review_mask]
    boundary = boundary.assign(
        _original_row=np.flatnonzero(review_mask),
    ).sort_values(
        by=["boundary_score", "_original_row"],
        ascending=[False, True],
        kind="mergesort",
    )
    boundary.insert(0, "boundary_rank", np.arange(1, len(boundary) + 1))
    return boundary.drop(columns=["_original_row"])


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_summary(
    metadata: pd.DataFrame,
    result: HdbscanMembershipComparisonResult,
    *,
    input_json: Path,
    sample_seed: int | None,
    multi_leaf_threshold: float,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    native_row_sum = result.native_memberships.sum(axis=1)
    exact_row_sum = result.exact_knn.affinities.sum(axis=1)
    max_gap = np.abs(result.native_max_affinity - result.exact_knn.max_affinity)
    unexplained_gap = np.abs(
        result.native_unexplained - result.exact_knn.unexplained
    )
    exact_second = _second_largest(result.exact_knn.affinities)
    exact_multi_leaf = exact_second >= multi_leaf_threshold
    return {
        "pipeline": "hdbscan_native_vs_pca_exact_knn_membership_comparison",
        "phase": "Phase 1-3 validation",
        "samples": int(len(metadata)),
        "embedding_dimensions": int(result.pca_selection.pca.n_features_in_),
        "cluster_count": int(result.cluster_count),
        "noise_count": int(np.sum(result.leaf_labels == -1)),
        "cluster_sizes": {
            str(cluster): int(np.sum(result.leaf_labels == cluster))
            for cluster in range(result.cluster_count)
        },
        "configuration": {
            "input_json": str(input_json),
            "sample_seed": sample_seed,
            **result.configuration,
        },
        "pca_selection": result.pca_selection.to_dict(),
        "runtime_seconds": result.runtime_seconds,
        "method_comparison": {
            "native_membership_row_sum": _finite_stats(native_row_sum),
            "pca_exact_knn_affinity_row_sum": _finite_stats(exact_row_sum),
            "native_unexplained": _finite_stats(result.native_unexplained),
            "pca_exact_knn_unexplained": _finite_stats(result.exact_knn.unexplained),
            "native_exact_max_affinity_absolute_gap": _finite_stats(max_gap),
            "native_exact_unexplained_absolute_gap": _finite_stats(unexplained_gap),
            "recommended_leaf_agreement_rate": float(
                np.mean(
                    result.native_recommended_labels
                    == result.exact_knn.recommended_labels
                )
            ),
            "exact_multi_leaf_overlap_threshold": float(multi_leaf_threshold),
            "exact_multi_leaf_overlap_count": int(np.sum(exact_multi_leaf)),
            "exact_multi_leaf_overlap_rate": float(np.mean(exact_multi_leaf)),
        },
        "artifacts": artifact_paths,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    embeddings, metadata = load_embeddings_from_json(
        args.input_json,
        start=args.start,
        limit=args.limit,
    )
    sample_seed: int | None = None
    if args.dataset_sample_size is not None:
        sample_seed = (
            args.seed
            if args.dataset_sample_seed is None
            else args.dataset_sample_seed
        )
        embeddings, metadata = sample_embedding_batch(
            embeddings,
            metadata,
            sample_size=args.dataset_sample_size,
            seed=sample_seed,
        )

    result = fit_hdbscan_membership_comparison(
        embeddings,
        pca_components=args.pca_components,
        pca_max_components=args.pca_max_components,
        pca_min_components=args.pca_min_components,
        pca_component_step=args.pca_component_step,
        umap_components=args.umap_components,
        umap_n_neighbors=args.umap_n_neighbors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        neighbor_count=args.neighbor_count,
        neighbor_backend=args.neighbor_backend,
        neighbor_graph_neighbors=args.neighbor_graph_neighbors,
        neighbor_query_epsilon=args.neighbor_query_epsilon,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = args.output_dir / "assignments.csv"
    boundary_path = args.output_dir / "boundary_cases.csv"
    summary_path = args.output_dir / "summary.json"
    build_assignments(metadata, result).to_csv(
        assignments_path,
        index=False,
        float_format="%.10g",
    )
    build_boundary_cases(metadata, result).to_csv(
        boundary_path,
        index=False,
        float_format="%.10g",
    )
    artifact_paths = {
        "assignments": str(assignments_path),
        "boundary_cases": str(boundary_path),
        "summary": str(summary_path),
    }
    summary = build_summary(
        metadata,
        result,
        input_json=args.input_json,
        sample_seed=sample_seed,
        multi_leaf_threshold=DEFAULT_MULTI_LEAF_AFFINITY_THRESHOLD,
        artifact_paths=artifact_paths,
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HDBSCAN native soft membership with PCA-space exact-kNN "
            "independent affinities."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-sample-size", type=int)
    parser.add_argument("--dataset-sample-seed", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--pca-max-components", type=int, default=DEFAULT_MAX_PCA_COMPONENTS)
    parser.add_argument("--pca-min-components", type=int, default=32)
    parser.add_argument("--pca-component-step", type=int, default=32)
    parser.add_argument("--umap-components", type=int, default=DEFAULT_UMAP_COMPONENTS)
    parser.add_argument("--umap-n-neighbors", type=int, default=DEFAULT_UMAP_N_NEIGHBORS)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--neighbor-count", type=int, default=DEFAULT_NEIGHBOR_COUNT)
    parser.add_argument("--neighbor-backend", choices=("exact", "pynndescent"), default="exact")
    parser.add_argument("--neighbor-graph-neighbors", type=int, default=32)
    parser.add_argument("--neighbor-query-epsilon", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_pipeline(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
