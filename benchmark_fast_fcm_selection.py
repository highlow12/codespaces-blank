"""Compare exact and fast FCM K selection across fixed data samples.

The benchmark treats the exhaustive selector as the reference for K selection.
It runs the fast selector with one or more ``refine_score_margin`` values and
records K agreement, label agreement, runtime, and the scout refinement
decision.  It is intentionally separate from the production pipeline so a
margin can be evaluated without changing the default configuration first.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from embedding_data import load_embeddings_from_json, sample_embedding_batch
from fast_fcm import FastFcmConfig, select_fast_fcm_cluster_count
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fcm_validity import select_fcm_cluster_count
from pca_projection import fit_normalized_pca_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure exact-vs-fast FCM K selection agreement."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--dataset-sample-sizes",
        type=int,
        nargs="+",
        default=[100, 300, 1000, 3000],
    )
    parser.add_argument("--dataset-sample-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--min-child-size", type=int, default=10)
    parser.add_argument("--exact-n-init", type=int, default=DEFAULT_FCM_N_INIT)
    parser.add_argument("--exact-max-attempts", type=int, default=None)
    parser.add_argument(
        "--min-center-separation",
        type=float,
        default=DEFAULT_FCM_MIN_CENTER_SEPARATION,
    )
    parser.add_argument("--scout-sample-size", type=int, default=1000)
    parser.add_argument(
        "--refine-score-margins",
        type=float,
        nargs="+",
        default=[0.15, 1.0],
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, row_count: int) -> None:
    if not args.dataset_sample_sizes or min(args.dataset_sample_sizes) < 2:
        raise ValueError("--dataset-sample-sizes must contain values >= 2")
    if max(args.dataset_sample_sizes) > row_count:
        raise ValueError(
            "--dataset-sample-sizes cannot exceed the number of input rows"
        )
    if not args.seeds:
        raise ValueError("--seeds must contain at least one value")
    if args.pca_components < 1:
        raise ValueError("--pca-components must be positive")
    if args.min_clusters < 2 or args.max_clusters < args.min_clusters:
        raise ValueError("invalid cluster-count bounds")
    if args.min_child_size < 1:
        raise ValueError("--min-child-size must be positive")
    if args.exact_n_init < 1:
        raise ValueError("--exact-n-init must be positive")
    if args.exact_max_attempts is not None and args.exact_max_attempts < args.exact_n_init:
        raise ValueError("--exact-max-attempts must cover --exact-n-init")
    if args.scout_sample_size < 2:
        raise ValueError("--scout-sample-size must be at least 2")
    if not args.refine_score_margins or any(
        margin < 0.0 or margin > 1.0 for margin in args.refine_score_margins
    ):
        raise ValueError("refine score margins must be between 0 and 1")


def _fast_refinement_summary(
    records: list[dict[str, Any]],
) -> tuple[list[int], str | None, float | None]:
    scout_records = [
        record
        for record in records
        if record.get("refine_decision") is not None
    ]
    selected_ks = sorted(
        {
            int(record["k"])
            for record in scout_records
            if record.get("refine_selected")
        }
    )
    decisions = {
        str(record["refine_decision"])
        for record in scout_records
        if record.get("refine_decision") is not None
    }
    gaps = {
        float(record["refine_score_gap"])
        for record in scout_records
        if record.get("refine_score_gap") is not None
    }
    decision = next(iter(decisions)) if len(decisions) == 1 else None
    score_gap = next(iter(gaps)) if len(gaps) == 1 else None
    return selected_ks, decision, score_gap


def _run_exact(
    features: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Any, list[dict[str, Any]], str, float]:
    started_at = time.perf_counter()
    best, records, reason = select_fcm_cluster_count(
        features,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_child_size=args.min_child_size,
        min_membership=0.40,
        selection_method="multi_metric",
        seed=seed,
        n_init=args.exact_n_init,
        max_attempts=args.exact_max_attempts,
        min_center_separation=args.min_center_separation,
    )
    return best, records, reason, time.perf_counter() - started_at


def _run_fast(
    features: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
    refine_score_margin: float,
) -> tuple[Any, list[dict[str, Any]], str, float]:
    config = FastFcmConfig(
        sample_size=min(args.scout_sample_size, features.shape[0]),
        scout_max_clusters=min(args.max_clusters, 8),
        refine_score_margin=refine_score_margin,
        min_center_separation=args.min_center_separation,
    )
    started_at = time.perf_counter()
    best, records, reason = select_fast_fcm_cluster_count(
        features,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_child_size=args.min_child_size,
        min_membership=0.40,
        selection_method="multi_metric",
        seed=seed,
        config=config,
    )
    return best, records, reason, time.perf_counter() - started_at


def _run_row(
    features: np.ndarray,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
    *,
    dataset_size: int,
    seed: int,
    exact: tuple[Any, list[dict[str, Any]], str, float],
    refine_score_margin: float,
) -> dict[str, Any]:
    exact_best, _exact_records, exact_reason, exact_seconds = exact
    fast_best, fast_records, fast_reason, fast_seconds = _run_fast(
        features,
        args,
        seed=seed,
        refine_score_margin=refine_score_margin,
    )
    refined_ks, refine_decision, score_gap = _fast_refinement_summary(
        fast_records
    )
    row: dict[str, Any] = {
        "dataset_size": dataset_size,
        "seed": seed,
        "refine_score_margin": refine_score_margin,
        "exact_selected_k": (
            int(exact_best.n_clusters) if exact_best is not None else None
        ),
        "fast_selected_k": (
            int(fast_best.n_clusters) if fast_best is not None else None
        ),
        "k_agreement": (
            exact_best is not None
            and fast_best is not None
            and exact_best.n_clusters == fast_best.n_clusters
        ),
        "exact_reason": exact_reason,
        "fast_reason": fast_reason,
        "exact_runtime_sec": exact_seconds,
        "fast_runtime_sec": fast_seconds,
        "speedup": (
            exact_seconds / fast_seconds if fast_seconds > 0.0 else None
        ),
        "refined_k_count": len(refined_ks),
        "refined_ks": ",".join(str(k) for k in refined_ks),
        "refine_decision": refine_decision,
        "refine_score_gap": score_gap,
    }
    if exact_best is not None and fast_best is not None:
        row["label_ari"] = float(
            adjusted_rand_score(exact_best.labels, fast_best.labels)
        )
        row["label_nmi"] = float(
            normalized_mutual_info_score(exact_best.labels, fast_best.labels)
        )
    if "tag" in metadata.columns and fast_best is not None:
        row["fast_tag_ari"] = float(
            adjusted_rand_score(metadata["tag"], fast_best.labels)
        )
        row["fast_tag_nmi"] = float(
            normalized_mutual_info_score(metadata["tag"], fast_best.labels)
        )
    return row


def _summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    group_columns = ["dataset_size", "refine_score_margin"]
    for (dataset_size, margin), group in frame.groupby(group_columns, sort=True):
        agreements = group["k_agreement"].astype(bool)
        decision_counts = Counter(
            str(value)
            for value in group["refine_decision"].dropna().tolist()
        )
        summaries.append(
            {
                "dataset_size": int(dataset_size),
                "refine_score_margin": float(margin),
                "runs": int(len(group)),
                "k_agreement_count": int(agreements.sum()),
                "k_agreement_rate": float(agreements.mean()),
                "mean_exact_runtime_sec": float(group["exact_runtime_sec"].mean()),
                "mean_fast_runtime_sec": float(group["fast_runtime_sec"].mean()),
                "mean_speedup": float(group["speedup"].mean()),
                "mean_refined_k_count": float(group["refined_k_count"].mean()),
                "mean_refine_score_gap": (
                    float(group["refine_score_gap"].dropna().mean())
                    if group["refine_score_gap"].notna().any()
                    else None
                ),
                "mean_label_ari": (
                    float(group["label_ari"].dropna().mean())
                    if "label_ari" in group and group["label_ari"].notna().any()
                    else None
                ),
                "mean_label_nmi": (
                    float(group["label_nmi"].dropna().mean())
                    if "label_nmi" in group and group["label_nmi"].notna().any()
                    else None
                ),
                "refine_decision_counts": dict(sorted(decision_counts.items())),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    _validate_args(args, len(embeddings))
    dataset_sizes = sorted(dict.fromkeys(args.dataset_sample_sizes))
    seeds = list(dict.fromkeys(args.seeds))
    rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    for dataset_size in dataset_sizes:
        sampled_embeddings, sampled_metadata = sample_embedding_batch(
            embeddings,
            metadata,
            sample_size=dataset_size,
            seed=args.dataset_sample_seed + dataset_size,
        )
        fitted = fit_normalized_pca_projection(
            sampled_embeddings,
            n_components=args.pca_components,
            seed=args.pca_seed,
        )
        features = fitted.normalized_prefix()
        for seed in seeds:
            exact = _run_exact(features, args, seed=seed)
            for margin in args.refine_score_margins:
                row = _run_row(
                    features,
                    sampled_metadata,
                    args,
                    dataset_size=dataset_size,
                    seed=seed,
                    exact=exact,
                    refine_score_margin=float(margin),
                )
                rows.append(row)
                print(
                    f"n={dataset_size} seed={seed} margin={margin:g}: "
                    f"exact K={row['exact_selected_k']} "
                    f"fast K={row['fast_selected_k']} "
                    f"agree={row['k_agreement']} "
                    f"refine={row['refined_ks'] or '-'}"
                )

    frame = pd.DataFrame(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    payload = {
        "configuration": {
            "input_json": str(args.input_json),
            "samples_in_input": int(len(embeddings)),
            "embedding_dimension": int(embeddings.shape[1]),
            "dataset_sample_sizes": dataset_sizes,
            "dataset_sample_seed": args.dataset_sample_seed,
            "seeds": seeds,
            "pca_components": args.pca_components,
            "pca_seed": args.pca_seed,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_child_size": args.min_child_size,
            "exact_n_init": args.exact_n_init,
            "exact_max_attempts": args.exact_max_attempts,
            "scout_sample_size": args.scout_sample_size,
            "refine_score_margins": [
                float(margin) for margin in args.refine_score_margins
            ],
        },
        "summary": _summary(frame),
        "runs": rows,
        "runtime_sec": time.perf_counter() - started_at,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
