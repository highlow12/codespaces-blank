"""Benchmark automatic spherical-FCM K selection across PCA prefixes."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from embedding_data import load_embeddings_from_json
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fcm_validity import select_fcm_cluster_count
from pca_projection import fit_normalized_pca_projection


def _mean_pairwise_ari(labels: list[np.ndarray]) -> float | None:
    if len(labels) < 2:
        return None
    return float(
        np.mean(
            [
                adjusted_rand_score(first, second)
                for first, second in combinations(labels, 2)
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare automatic FCM K selection across PCA dimensions."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[96, 128, 192, 256],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=DEFAULT_FCM_N_INIT)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument(
        "--min-center-separation",
        type=float,
        default=DEFAULT_FCM_MIN_CENTER_SEPARATION,
    )
    parser.add_argument("--pca-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dimensions = sorted(dict.fromkeys(args.dimensions))
    seeds = list(dict.fromkeys(args.seeds))
    if not dimensions or min(dimensions) < 1:
        raise ValueError("--dimensions must contain positive values")
    if not seeds:
        raise ValueError("--seeds must contain at least one value")

    embeddings, metadata = load_embeddings_from_json(args.input_json)
    fitted = fit_normalized_pca_projection(
        embeddings,
        n_components=max(dimensions),
        seed=args.pca_seed,
    )
    max_supported = int(fitted.pca.n_components_)
    if max(dimensions) > max_supported:
        raise ValueError(
            f"Requested PCA {max(dimensions)} but only {max_supported} is supported"
        )

    rows: list[dict[str, Any]] = []
    candidates_by_run: list[dict[str, Any]] = []
    labels_by_dimension: dict[int, list[np.ndarray]] = {
        dimension: [] for dimension in dimensions
    }
    started_at = time.perf_counter()

    for dimension in dimensions:
        features = fitted.normalized_prefix(dimension)
        for seed in seeds:
            run_started_at = time.perf_counter()
            best, candidate_metrics, reason = select_fcm_cluster_count(
                features,
                min_clusters=args.min_clusters,
                max_clusters=args.max_clusters,
                min_child_size=args.min_child_size,
                min_membership=0.40,
                selection_method="multi_metric",
                seed=seed,
                n_init=args.n_init,
                max_attempts=args.max_attempts,
                min_center_separation=args.min_center_separation,
            )
            if best is None:
                rows.append(
                    {
                        "pca_dimension": dimension,
                        "seed": seed,
                        "selected_k": None,
                        "selection_reason": reason,
                        "runtime_sec": time.perf_counter() - run_started_at,
                    }
                )
                candidates_by_run.append(
                    {
                        "pca_dimension": dimension,
                        "seed": seed,
                        "selection_reason": reason,
                        "candidates": candidate_metrics,
                    }
                )
                continue

            labels_by_dimension[dimension].append(best.labels.copy())
            row: dict[str, Any] = {
                "pca_dimension": dimension,
                "seed": seed,
                "selected_k": best.n_clusters,
                "selection_reason": reason,
                "selection_score": best.selection_score,
                "xie_beni": best.xie_beni,
                "silhouette": best.silhouette,
                "modified_partition_coefficient": (
                    best.modified_partition_coefficient
                ),
                "restart_stability": best.restart_stability,
                "valid_restarts": best.valid_restarts,
                "attempts": best.attempts,
                "minimum_center_distance": best.minimum_center_distance,
                "runtime_sec": time.perf_counter() - run_started_at,
            }
            if "tag" in metadata.columns:
                row["top_nmi"] = normalized_mutual_info_score(
                    metadata["tag"], best.labels
                )
                row["top_ari"] = adjusted_rand_score(
                    metadata["tag"], best.labels
                )
            if "class" in metadata.columns:
                row["leaf_nmi"] = normalized_mutual_info_score(
                    metadata["class"], best.labels
                )
                row["leaf_ari"] = adjusted_rand_score(
                    metadata["class"], best.labels
                )
            rows.append(row)
            candidates_by_run.append(
                {
                    "pca_dimension": dimension,
                    "seed": seed,
                    "selected_k": best.n_clusters,
                    "selection_reason": reason,
                    "candidates": candidate_metrics,
                }
            )
            print(
                f"PCA-{dimension} seed={seed}: K={best.n_clusters}, "
                f"restart_stability={best.restart_stability:.4f}"
            )

    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for dimension in dimensions:
        dimension_rows = frame[frame["pca_dimension"] == dimension]
        selected = [
            int(value)
            for value in dimension_rows["selected_k"].dropna().tolist()
        ]
        counts = Counter(selected)
        summaries.append(
            {
                "pca_dimension": dimension,
                "selected_k_counts": {
                    str(k): int(count) for k, count in sorted(counts.items())
                },
                "selected_k_mode": (
                    counts.most_common(1)[0][0] if counts else None
                ),
                "between_seed_partition_ari": _mean_pairwise_ari(
                    labels_by_dimension[dimension]
                ),
                "mean_restart_stability": (
                    float(dimension_rows["restart_stability"].mean())
                    if "restart_stability" in dimension_rows
                    else None
                ),
                "mean_top_nmi": (
                    float(dimension_rows["top_nmi"].mean())
                    if "top_nmi" in dimension_rows
                    else None
                ),
                "mean_top_ari": (
                    float(dimension_rows["top_ari"].mean())
                    if "top_ari" in dimension_rows
                    else None
                ),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    args.output_json.write_text(
        json.dumps(
            {
                "configuration": {
                    "samples": int(len(embeddings)),
                    "embedding_dimension": int(embeddings.shape[1]),
                    "pca_dimensions": dimensions,
                    "seeds": seeds,
                    "min_clusters": args.min_clusters,
                    "max_clusters": args.max_clusters,
                    "min_child_size": args.min_child_size,
                    "n_init": args.n_init,
                    "max_attempts": args.max_attempts,
                    "min_center_separation": args.min_center_separation,
                    "score_weights": {
                        "xie_beni": 0.40,
                        "silhouette": 0.25,
                        "restart_stability": 0.25,
                        "modified_partition_coefficient": 0.10,
                    },
                },
                "summary": summaries,
                "runs": rows,
                "candidate_metrics": candidates_by_run,
                "runtime_sec": time.perf_counter() - started_at,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved benchmark JSON: {args.output_json}")
    print(f"Saved benchmark CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
