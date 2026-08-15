"""CLI for comparing two post-hoc soft assignments for HDBSCAN noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from embedding_data import load_embeddings_from_json, sample_embedding_batch
from hdbscan_soft import (
    DEFAULT_MEDOID_CANDIDATE_BUDGET,
    DEFAULT_MEDOID_EVALUATION_BUDGET,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_NEIGHBOR_COUNT,
    DEFAULT_REASSIGNMENT_THRESHOLD,
    fit_hdbscan_soft,
)


def build_assignments(metadata: pd.DataFrame, result: Any) -> pd.DataFrame:
    frame = metadata.copy().reset_index(drop=True)
    frame["hdbscan_label"] = result.original_labels
    frame["cluster"] = result.labels
    frame["medoid_recommended_cluster"] = result.medoid_recommended_labels
    frame["neighbor_recommended_cluster"] = result.neighbor_recommended_labels
    frame["medoid_max_membership"] = result.medoid_confidences
    frame["neighbor_max_membership"] = result.neighbor_confidences
    frame["recommended_labels_agree"] = (
        result.medoid_recommended_labels == result.neighbor_recommended_labels
    )
    for cluster in range(result.cluster_count):
        frame[f"membership_medoid_{cluster}"] = result.medoid_memberships[:, cluster]
        frame[f"membership_neighbor_{cluster}"] = result.neighbor_memberships[:, cluster]
    return frame


def build_summary(metadata: pd.DataFrame, result: Any, *, threshold: float) -> dict[str, Any]:
    noise = result.original_labels == -1
    agreement = result.medoid_recommended_labels[noise] == result.neighbor_recommended_labels[noise]
    confidence_delta = result.medoid_confidences[noise] - result.neighbor_confidences[noise]
    clusters = []
    for cluster in range(result.cluster_count):
        clusters.append({
            "cluster": cluster,
            "size": int(np.sum(result.original_labels == cluster)),
            "medoid_document_id": metadata.iloc[int(result.medoid_indices[cluster])]["id"],
            "radius_90_cosine_distance": float(result.cluster_radii[cluster]),
            "neighbor_member_count": int(result.neighbor_member_counts[cluster]),
        })
    return {
        "pipeline": "hdbscan_distance_soft_comparison",
        "samples": int(len(metadata)),
        "clusters": clusters,
        "noise_count": int(np.sum(noise)),
        "medoid_reassigned_count": int(np.sum(noise & (result.labels != -1))),
        "neighbor_reassigned_count": int(np.sum(noise & (result.neighbor_confidences >= threshold))),
        "noise_recommended_label_agreement_rate": float(np.mean(agreement)) if len(agreement) else 1.0,
        "noise_confidence_difference": {
            "mean_medoid_minus_neighbor": float(np.mean(confidence_delta)) if len(confidence_delta) else 0.0,
            "mean_absolute": float(np.mean(np.abs(confidence_delta))) if len(confidence_delta) else 0.0,
        },
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    embeddings, metadata = load_embeddings_from_json(args.input_json, start=args.start, limit=args.limit)
    if args.dataset_sample_size is not None:
        embeddings, metadata = sample_embedding_batch(embeddings, metadata, sample_size=args.dataset_sample_size, seed=args.dataset_sample_seed if args.dataset_sample_seed is not None else args.seed)
    result = fit_hdbscan_soft(
        embeddings, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        reassignment_threshold=args.reassignment_threshold, neighbor_count=args.neighbor_count,
        pca_components=args.pca_components, medoid_candidate_budget=args.medoid_candidate_budget,
        medoid_evaluation_budget=args.medoid_evaluation_budget, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = args.output_dir / "assignments.csv"
    summary_path = args.output_dir / "summary.json"
    build_assignments(metadata, result).to_csv(assignments_path, index=False)
    summary = build_summary(metadata, result, threshold=args.reassignment_threshold)
    summary["config"] = {
        "min_cluster_size": args.min_cluster_size, "min_samples": args.min_samples,
        "cluster_selection_method": args.cluster_selection_method,
        "reassignment_threshold": args.reassignment_threshold, "neighbor_count": args.neighbor_count,
        "pca_components_requested": args.pca_components, "pca_components": int(result.features.shape[1]),
        "medoid_candidate_budget": args.medoid_candidate_budget,
        "medoid_evaluation_budget": args.medoid_evaluation_budget,
    }
    summary["artifacts"] = {"assignments": str(assignments_path), "summary": str(summary_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare distance-based soft memberships for HDBSCAN noise.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hdbscan_soft"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-sample-size", type=int)
    parser.add_argument("--dataset-sample-seed", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--cluster-selection-method", choices=("eom", "leaf"), default="eom")
    parser.add_argument("--reassignment-threshold", type=float, default=DEFAULT_REASSIGNMENT_THRESHOLD)
    parser.add_argument("--neighbor-count", type=int, default=DEFAULT_NEIGHBOR_COUNT)
    parser.add_argument("--pca-components", type=int)
    parser.add_argument("--medoid-candidate-budget", type=int, default=DEFAULT_MEDOID_CANDIDATE_BUDGET)
    parser.add_argument("--medoid-evaluation-budget", type=int, default=DEFAULT_MEDOID_EVALUATION_BUDGET)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_pipeline(args), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
