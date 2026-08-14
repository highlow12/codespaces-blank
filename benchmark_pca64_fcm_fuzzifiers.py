"""Sweep FCM fuzzifiers across PCA-64 geometry variants with exhaustive k."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_pca64_fcm_geometries import (
    VARIANTS,
    _external_metrics,
    _safe_silhouette,
)
from embedding_data import load_embeddings_from_json
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fcm_validity import select_fcm_cluster_count
from geometry_fcm_selection import select_geometry_fcm_cluster_count
from pca_projection import fit_normalized_pca_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fuzzifiers", type=float, nargs="+", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[name for name, _geometry, _normalized in VARIANTS],
        default=[name for name, _geometry, _normalized in VARIANTS],
    )
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=DEFAULT_FCM_N_INIT)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--min-center-separation",
        type=float,
        default=DEFAULT_FCM_MIN_CENTER_SEPARATION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(value <= 1.0 for value in args.fuzzifiers):
        raise ValueError("every fuzzifier must be greater than 1")
    started_at = time.perf_counter()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    fitted = fit_normalized_pca_projection(
        embeddings,
        n_components=args.pca_components,
        seed=args.seed,
    )
    raw_pca = np.asarray(fitted.projected, dtype=np.float64)
    normalized_pca = np.asarray(fitted.normalized_prefix(), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    candidate_runs: list[dict[str, Any]] = []
    assignments = metadata.copy().reset_index(drop=True)

    for fuzzifier in args.fuzzifiers:
        for variant_name, geometry_name, use_normalized in VARIANTS:
            if variant_name not in args.variants:
                continue
            features = normalized_pca if use_normalized else raw_pca
            run_started_at = time.perf_counter()
            kwargs = {
                "min_clusters": args.min_clusters,
                "max_clusters": args.max_clusters,
                "min_child_size": args.min_child_size,
                "seed": args.seed,
                "n_init": args.n_init,
                "max_attempts": args.max_attempts,
                "min_center_separation": args.min_center_separation,
                "m": fuzzifier,
                "max_iter": args.max_iter,
                "tol": args.tol,
                # Once worsening is detected, continue through max k.
                "xb_worsening_patience": args.max_clusters,
            }
            if geometry_name == "spherical":
                best, records, reason = select_fcm_cluster_count(
                    features,
                    selection_method="multi_metric",
                    **kwargs,
                )
            else:
                best, records, reason = select_geometry_fcm_cluster_count(
                    features,
                    geometry_name=geometry_name,
                    **kwargs,
                )
            run_id = f"{variant_name}_m{fuzzifier:g}".replace(".", "p")
            candidate_runs.append(
                {
                    "run_id": run_id,
                    "variant": variant_name,
                    "fuzzifier": fuzzifier,
                    "selection_reason": reason,
                    "selected_k": None if best is None else int(best.n_clusters),
                    "candidates": records,
                }
            )
            if best is None:
                assignments[f"{run_id}_label"] = -1
                rows.append(
                    {
                        "run_id": run_id,
                        "variant": variant_name,
                        "geometry": geometry_name,
                        "post_pca_normalized": use_normalized,
                        "fuzzifier": fuzzifier,
                        "selected_k": None,
                        "selection_reason": reason,
                        "runtime_sec": time.perf_counter() - run_started_at,
                    }
                )
                print(f"{run_id}: no valid k ({reason})", flush=True)
                continue

            labels = np.asarray(best.labels, dtype=int)
            assignments[f"{run_id}_label"] = labels
            assignments[f"{run_id}_max_membership"] = np.asarray(
                best.result.memberships
            ).max(axis=1)
            row = {
                "run_id": run_id,
                "variant": variant_name,
                "geometry": geometry_name,
                "post_pca_normalized": use_normalized,
                "fuzzifier": fuzzifier,
                "selected_k": int(best.n_clusters),
                "selection_reason": reason,
                "selection_score": best.selection_score,
                "xie_beni": best.xie_beni,
                "selection_silhouette": best.silhouette,
                "restart_stability": best.restart_stability,
                "modified_partition_coefficient": best.modified_partition_coefficient,
                "cluster_sizes": json.dumps(best.cluster_sizes),
                "raw_pca_euclidean_silhouette": _safe_silhouette(
                    raw_pca, labels, metric="euclidean"
                ),
                "pca_direction_cosine_silhouette": _safe_silhouette(
                    raw_pca, labels, metric="cosine"
                ),
                "runtime_sec": time.perf_counter() - run_started_at,
                **_external_metrics(metadata, labels),
            }
            rows.append(row)
            print(
                f"{run_id}: k={best.n_clusters}, "
                f"silhouette={best.silhouette:.6f}, "
                f"stability={best.restart_stability:.6f}",
                flush=True,
            )

    report = {
        "configuration": {
            "input_json": str(args.input_json),
            "samples": len(embeddings),
            "embedding_dimension": embeddings.shape[1],
            "pca_components": raw_pca.shape[1],
            "pre_pca_l2_normalized": True,
            "fuzzifiers": args.fuzzifiers,
            "variants": args.variants,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "exhaustive_k": True,
            "min_child_size": args.min_child_size,
            "seed": args.seed,
            "n_init": args.n_init,
            "max_attempts": args.max_attempts,
        },
        "runs": rows,
        "candidate_runs": candidate_runs,
        "runtime_sec": time.perf_counter() - started_at,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)
    assignments.to_csv(args.output_dir / "assignments.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved fuzzifier sweep to: {args.output_dir}")


if __name__ == "__main__":
    main()
