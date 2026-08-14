"""Compare four flat FCM variants on one shared 64-dimensional PCA fit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from embedding_data import load_embeddings_from_json
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fcm_validity import select_fcm_cluster_count
from geometry_fcm_selection import select_geometry_fcm_cluster_count
from pca_projection import fit_normalized_pca_projection


VARIANTS = (
    ("existing_sfcm", "spherical", True),
    ("raw_pca_cosine_fcm", "cosine_raw", False),
    ("normalized_pca_cosine_fcm", "cosine_normalized", True),
    ("raw_pca_euclidean_fcm", "euclidean", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=DEFAULT_FCM_N_INIT)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--fuzzifier", type=float, default=2.0)
    parser.add_argument(
        "--min-center-separation",
        type=float,
        default=DEFAULT_FCM_MIN_CENTER_SEPARATION,
    )
    return parser.parse_args()


def _safe_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
) -> float | None:
    try:
        return float(silhouette_score(features, labels, metric=metric))
    except Exception:
        return None


def _external_metrics(
    metadata: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "tag" in metadata.columns:
        metrics["top_nmi"] = float(normalized_mutual_info_score(metadata["tag"], labels))
        metrics["top_ari"] = float(adjusted_rand_score(metadata["tag"], labels))
    if "class" in metadata.columns:
        metrics["leaf_nmi"] = float(normalized_mutual_info_score(metadata["class"], labels))
        metrics["leaf_ari"] = float(adjusted_rand_score(metadata["class"], labels))
    return metrics


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    fitted = fit_normalized_pca_projection(
        embeddings,
        n_components=args.pca_components,
        seed=args.seed,
    )
    raw_pca = np.asarray(fitted.projected, dtype=np.float64)
    normalized_pca = np.asarray(fitted.normalized_prefix(), dtype=np.float64)

    run_rows: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    labels_by_variant: dict[str, np.ndarray] = {}
    assignments = metadata.copy().reset_index(drop=True)

    for variant_name, geometry_name, use_normalized in VARIANTS:
        features = normalized_pca if use_normalized else raw_pca
        run_started_at = time.perf_counter()
        selection_kwargs = {
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_child_size": args.min_child_size,
            "seed": args.seed,
            "n_init": args.n_init,
            "max_attempts": args.max_attempts,
            "min_center_separation": args.min_center_separation,
            "m": args.fuzzifier,
            "max_iter": args.max_iter,
            "tol": args.tol,
        }
        if geometry_name == "spherical":
            best, candidate_records, reason = select_fcm_cluster_count(
                features,
                selection_method="multi_metric",
                **selection_kwargs,
            )
        else:
            best, candidate_records, reason = select_geometry_fcm_cluster_count(
                features,
                geometry_name=geometry_name,
                **selection_kwargs,
            )
        if best is None:
            assignments[f"{variant_name}_label"] = -1
            assignments[f"{variant_name}_max_membership"] = np.nan
            run_rows.append(
                {
                    "variant": variant_name,
                    "geometry": geometry_name,
                    "post_pca_normalized": bool(use_normalized),
                    "selected_k": None,
                    "selection_reason": reason,
                    "selection_score": None,
                    "xie_beni": None,
                    "selection_silhouette": None,
                    "restart_stability": None,
                    "modified_partition_coefficient": None,
                    "cluster_sizes": None,
                    "raw_pca_euclidean_silhouette": None,
                    "pca_direction_cosine_silhouette": None,
                    "runtime_sec": float(time.perf_counter() - run_started_at),
                }
            )
            candidates[variant_name] = {
                "selection_reason": reason,
                "selected_k": None,
                "candidates": candidate_records,
            }
            print(f"{variant_name}: no valid k ({reason})")
            continue

        labels = np.asarray(best.labels, dtype=int)
        memberships = np.asarray(best.result.memberships, dtype=np.float64)
        labels_by_variant[variant_name] = labels
        assignments[f"{variant_name}_label"] = labels
        assignments[f"{variant_name}_max_membership"] = memberships.max(axis=1)
        row: dict[str, Any] = {
            "variant": variant_name,
            "geometry": geometry_name,
            "post_pca_normalized": bool(use_normalized),
            "selected_k": int(best.n_clusters),
            "selection_reason": reason,
            "selection_score": best.selection_score,
            "xie_beni": float(best.xie_beni),
            "selection_silhouette": float(best.silhouette),
            "restart_stability": float(best.restart_stability),
            "modified_partition_coefficient": float(best.modified_partition_coefficient),
            "cluster_sizes": json.dumps(best.cluster_sizes),
            "raw_pca_euclidean_silhouette": _safe_silhouette(
                raw_pca, labels, metric="euclidean"
            ),
            "pca_direction_cosine_silhouette": _safe_silhouette(
                raw_pca, labels, metric="cosine"
            ),
            "runtime_sec": float(time.perf_counter() - run_started_at),
            **_external_metrics(metadata, labels),
        }
        run_rows.append(row)
        candidates[variant_name] = {
            "selection_reason": reason,
            "selected_k": int(best.n_clusters),
            "candidates": candidate_records,
        }
        print(
            f"{variant_name}: k={best.n_clusters}, "
            f"silhouette={best.silhouette:.6f}, "
            f"stability={best.restart_stability:.6f}"
        )

    equivalence_ari = float(
        adjusted_rand_score(
            labels_by_variant["existing_sfcm"],
            labels_by_variant["normalized_pca_cosine_fcm"],
        )
    )
    configuration = {
        "input_json": str(args.input_json),
        "samples": int(len(embeddings)),
        "embedding_dimension": int(embeddings.shape[1]),
        "pca_components": int(raw_pca.shape[1]),
        "pre_pca_l2_normalized": True,
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "min_child_size": args.min_child_size,
        "seed": args.seed,
        "n_init": args.n_init,
        "max_attempts": args.max_attempts,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "fuzzifier": args.fuzzifier,
    }
    report = {
        "configuration": configuration,
        "runs": run_rows,
        "candidates": candidates,
        "existing_sfcm_vs_normalized_cosine_ari": equivalence_ari,
        "runtime_sec": float(time.perf_counter() - started_at),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(args.output_dir / "runs.csv", index=False)
    assignments.to_csv(args.output_dir / "assignments.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved comparison outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
