"""Benchmark online center adaptation after sampled automatic K selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from benchmark_cluster_selection_sample_ratios import (
    _project_to_centers,
    _quality,
    _select_k,
    choose_dataset_indices,
    choose_nested_sample_indices,
    online_refine_sample_centers,
)
from embedding_data import load_embeddings_from_json
from fcm_core import (
    DEFAULT_FCM_MIN_CENTER_SEPARATION,
    DEFAULT_FCM_N_INIT,
    spherical_fcm,
)
from pca_projection import fit_normalized_pca_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--total-size", type=int, default=3000)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=20260811)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--sample-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46]
    )
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
    return parser.parse_args()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    args = parse_args()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    if not 1 <= args.sample_size < args.total_size <= len(embeddings):
        raise ValueError("sample-size must be smaller than total-size within input")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    dataset_indices = choose_dataset_indices(
        len(embeddings), args.total_size, seed=args.dataset_seed + args.total_size
    )
    dataset_metadata = metadata.iloc[dataset_indices].reset_index(drop=True)
    fitted = fit_normalized_pca_projection(
        embeddings[dataset_indices],
        n_components=args.pca_components,
        seed=args.pca_seed,
    )
    features = fitted.normalized_prefix()

    full_results: dict[int, Any] = {}
    full_selection_seconds: list[float] = []
    for selection_seed in args.seeds:
        best, reason, elapsed = _select_k(
            features,
            args,
            seed=selection_seed,
            min_child_size=args.min_child_size,
        )
        if best is None:
            raise RuntimeError(f"full selection failed: {reason}")
        full_results[selection_seed] = best
        full_selection_seconds.append(elapsed)

    rows: list[dict[str, Any]] = []
    for sample_seed in args.sample_seeds:
        sample_indices = choose_nested_sample_indices(
            args.total_size, args.sample_size, seed=sample_seed
        )
        sampled_features = features[sample_indices]
        for selection_seed in args.seeds:
            selected, reason, selection_sec = _select_k(
                sampled_features,
                args,
                seed=selection_seed,
                min_child_size=max(
                    2,
                    int(np.ceil(args.min_child_size * args.sample_size / args.total_size)),
                ),
            )
            if selected is None:
                raise RuntimeError(f"sample selection failed: {reason}")
            baseline = full_results[selection_seed]

            started = time.perf_counter()
            fixed = _project_to_centers(features, selected.result)
            fixed_update_sec = time.perf_counter() - started

            started = time.perf_counter()
            online = online_refine_sample_centers(
                features,
                sample_indices,
                selected.result,
                batch_size=args.batch_size,
                order_seed=sample_seed + selection_seed * 1009,
            )
            online_update_sec = time.perf_counter() - started

            started = time.perf_counter()
            refit = spherical_fcm(
                features,
                n_clusters=selected.n_clusters,
                seed=selection_seed + selected.n_clusters * 1009,
                m=selected.m,
                n_init=args.n_init,
                max_attempts=args.max_attempts,
                min_cluster_size=args.min_child_size,
                min_center_separation=args.min_center_separation,
            )
            refit_sec = time.perf_counter() - started

            for strategy, result, update_sec in (
                ("fixed_sample_centers", fixed, fixed_update_sec),
                ("online_center_update", online, online_update_sec),
                ("full_refit", refit, refit_sec),
            ):
                quality = _quality(features, dataset_metadata, result, baseline.result.labels)
                rows.append(
                    {
                        "strategy": strategy,
                        "sample_seed": sample_seed,
                        "selection_seed": selection_seed,
                        "sample_selected_k": selected.n_clusters,
                        "full_selected_k": baseline.n_clusters,
                        "k_match": selected.n_clusters == baseline.n_clusters,
                        "selection_sec": selection_sec,
                        "update_sec": update_sec,
                        "algorithm_sec": selection_sec + update_sec,
                        "agreement_ari_vs_full_selection": float(
                            adjusted_rand_score(baseline.result.labels, result.labels)
                        ),
                        **quality,
                    }
                )
            print(
                f"sample_seed={sample_seed} selection_seed={selection_seed} "
                f"K={selected.n_clusters}/{baseline.n_clusters}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    summary = []
    baseline_seconds = float(np.mean(full_selection_seconds))
    for strategy, group in frame.groupby("strategy", sort=True):
        summary.append(
            {
                "strategy": strategy,
                "runs": int(len(group)),
                "mean_algorithm_sec": _mean(group.to_dict("records"), "algorithm_sec"),
                "speedup_vs_full_selection": baseline_seconds
                / _mean(group.to_dict("records"), "algorithm_sec"),
                "mean_agreement_ari": _mean(
                    group.to_dict("records"), "agreement_ari_vs_full_selection"
                ),
                "mean_silhouette": _mean(group.to_dict("records"), "silhouette"),
                "mean_xie_beni": _mean(group.to_dict("records"), "xie_beni"),
            }
        )
    report = {
        "configuration": vars(args),
        "full_selection_mean_sec": baseline_seconds,
        "summary": summary,
        "runs": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
