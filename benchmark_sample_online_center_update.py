"""Compare full refit with stale and refreshed incremental memberships.

Every method shares the exact same sampled K selection.  The incremental path
streams held-out rows through online center updates.  One result preserves the
memberships observed when each row was processed; the other recomputes every
membership once against the final online centers.
"""

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
    _quality,
    _select_k,
    choose_dataset_indices,
    choose_nested_sample_indices,
    online_refine_sample_centers_with_trace,
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

            trace = online_refine_sample_centers_with_trace(
                features,
                sample_indices,
                selected.result,
                batch_size=args.batch_size,
                order_seed=sample_seed + selection_seed * 1009,
            )

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

            refresh_changed_count = int(
                np.sum(
                    trace.stale_result.labels
                    != trace.refreshed_result.labels
                )
            )
            refresh_changed_rate = refresh_changed_count / len(features)
            stale_vs_refreshed_ari = float(
                adjusted_rand_score(
                    trace.stale_result.labels,
                    trace.refreshed_result.labels,
                )
            )
            polish_changed_count = int(
                np.sum(
                    trace.refreshed_result.labels
                    != trace.polished_result.labels
                )
            )
            polish_changed_rate = polish_changed_count / len(features)
            refreshed_vs_polished_ari = float(
                adjusted_rand_score(
                    trace.refreshed_result.labels,
                    trace.polished_result.labels,
                )
            )

            for strategy, result, update_sec in (
                (
                    "incremental_stale_memberships",
                    trace.stale_result,
                    trace.streaming_update_sec,
                ),
                (
                    "incremental_full_membership_refresh",
                    trace.refreshed_result,
                    trace.streaming_update_sec + trace.membership_refresh_sec,
                ),
                (
                    "incremental_one_pass_polish",
                    trace.polished_result,
                    trace.streaming_update_sec
                    + trace.membership_refresh_sec
                    + trace.polish_sec,
                ),
                ("full_refit", refit, refit_sec),
            ):
                quality = _quality(
                    features,
                    dataset_metadata,
                    result,
                    refit.labels,
                )
                agreement = quality.pop(
                    "agreement_ari_vs_full_selection"
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "sample_seed": sample_seed,
                        "selection_seed": selection_seed,
                        "sample_selected_k": selected.n_clusters,
                        "selection_sec": selection_sec,
                        "update_sec": update_sec,
                        "algorithm_sec": selection_sec + update_sec,
                        "agreement_ari_vs_same_k_full_refit": float(agreement),
                        "refresh_changed_count": refresh_changed_count,
                        "refresh_changed_rate": refresh_changed_rate,
                        "stale_vs_refreshed_ari": stale_vs_refreshed_ari,
                        "polish_changed_count": polish_changed_count,
                        "polish_changed_rate": polish_changed_rate,
                        "refreshed_vs_polished_ari": refreshed_vs_polished_ari,
                        **quality,
                    }
                )
            print(
                f"sample_seed={sample_seed} selection_seed={selection_seed} "
                f"K={selected.n_clusters} "
                f"refresh_changed={refresh_changed_count}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    summary = []
    full_refit_algorithm_sec = float(
        frame.loc[frame["strategy"] == "full_refit", "algorithm_sec"].mean()
    )
    full_refit_update_sec = float(
        frame.loc[frame["strategy"] == "full_refit", "update_sec"].mean()
    )
    for strategy, group in frame.groupby("strategy", sort=True):
        records = group.to_dict("records")
        mean_algorithm_sec = _mean(records, "algorithm_sec")
        mean_update_sec = _mean(records, "update_sec")
        summary.append(
            {
                "strategy": strategy,
                "runs": int(len(group)),
                "mean_algorithm_sec": mean_algorithm_sec,
                "mean_post_selection_sec": mean_update_sec,
                "speedup_vs_same_k_full_refit": (
                    full_refit_algorithm_sec / mean_algorithm_sec
                ),
                "post_selection_speedup_vs_same_k_full_refit": (
                    full_refit_update_sec / mean_update_sec
                ),
                "mean_agreement_ari_vs_same_k_full_refit": _mean(
                    records,
                    "agreement_ari_vs_same_k_full_refit",
                ),
                "mean_silhouette": _mean(records, "silhouette"),
                "mean_xie_beni": _mean(records, "xie_beni"),
                "mean_refresh_changed_count": _mean(
                    records,
                    "refresh_changed_count",
                ),
                "mean_refresh_changed_rate": _mean(
                    records,
                    "refresh_changed_rate",
                ),
                "mean_stale_vs_refreshed_ari": _mean(
                    records,
                    "stale_vs_refreshed_ari",
                ),
                "mean_polish_changed_count": _mean(
                    records,
                    "polish_changed_count",
                ),
                "mean_polish_changed_rate": _mean(
                    records,
                    "polish_changed_rate",
                ),
                "mean_refreshed_vs_polished_ari": _mean(
                    records,
                    "refreshed_vs_polished_ari",
                ),
            }
        )
    report = {
        "configuration": {
            **vars(args),
            "shared_sample_k_selection": True,
            "strategies": {
                "full_refit": "fit the sampled K on every row until convergence",
                "incremental_stale_memberships": (
                    "stream held-out rows and retain memberships from processing time"
                ),
                "incremental_full_membership_refresh": (
                    "stream held-out rows, then recompute all memberships once"
                ),
                "incremental_one_pass_polish": (
                    "refresh all memberships, update centers once, then reassign all rows"
                ),
            },
        },
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
