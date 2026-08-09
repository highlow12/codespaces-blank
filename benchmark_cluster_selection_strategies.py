"""Benchmark full-data and sampled spherical-FCM cluster selection strategies.

All strategies use the same PCA representation fitted on the complete dataset.
This deliberately measures the cost/quality trade-off of K selection and FCM
fitting, not a confounded difference in dimensionality reduction.
"""

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
from sklearn.metrics.pairwise import euclidean_distances

from clustering_types import FCMResult
from embedding_data import load_embeddings_from_json
from fcm_core import (
    DEFAULT_FCM_MIN_CENTER_SEPARATION,
    DEFAULT_FCM_N_INIT,
    sfcm_memberships_from_centers,
    spherical_fcm,
)
from fcm_validity import select_fcm_cluster_count, xie_beni_index
from pca_projection import fit_normalized_pca_projection


STRATEGIES = (
    "full_selection",
    "sample_select_full_fit",
    "sample_select_project",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--pca-seed", type=int, default=42)
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


def _select_k(
    features: np.ndarray, args: argparse.Namespace, seed: int
) -> tuple[Any, str, float]:
    started = time.perf_counter()
    best, _candidates, reason = select_fcm_cluster_count(
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
    return best, reason, time.perf_counter() - started


def _project_to_centers(features: np.ndarray, selected: FCMResult) -> FCMResult:
    memberships, distances = sfcm_memberships_from_centers(
        features, selected.centers, m=selected.m
    )
    centers = np.asarray(selected.centers, dtype=np.float64)
    center_distances = euclidean_distances(centers, centers)
    np.fill_diagonal(center_distances, np.inf)
    return FCMResult(
        labels=memberships.argmax(axis=1),
        memberships=memberships,
        centers=centers,
        iterations=0,
        objective=float(np.sum((memberships**selected.m) * (distances**2)) / len(features)),
        m=selected.m,
        n_init=selected.n_init,
        attempts=selected.attempts,
        valid_restarts=selected.valid_restarts,
        restart_stability=selected.restart_stability,
        minimum_center_distance=float(np.min(center_distances)),
        squared_dissimilarities=distances**2,
    )


def _quality(
    features: np.ndarray,
    metadata: pd.DataFrame,
    result: FCMResult,
    baseline_labels: np.ndarray | None,
) -> dict[str, float | int | None]:
    labels = result.labels
    values: dict[str, float | int | None] = {
        "clusters": int(np.unique(labels).size),
        "silhouette": float(silhouette_score(features, labels, metric="euclidean")),
        "xie_beni": float(
            xie_beni_index(
                features,
                result,
                squared_dissimilarities=result.squared_dissimilarities,
            )
        ),
        "agreement_ari_vs_full_selection": (
            None
            if baseline_labels is None
            else float(adjusted_rand_score(baseline_labels, labels))
        ),
    }
    for column in ("tag", "class"):
        if column in metadata:
            values[f"{column}_nmi"] = float(
                normalized_mutual_info_score(metadata[column], labels)
            )
            values[f"{column}_ari"] = float(
                adjusted_rand_score(metadata[column], labels)
            )
    return values


def _strategy_row(
    *,
    strategy: str,
    seed: int,
    selected_k: int,
    reason: str,
    selection_sec: float,
    full_fit_sec: float,
    assignment_sec: float,
    quality_sec: float,
    quality: dict[str, float | int | None],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "seed": seed,
        "selected_k": selected_k,
        "selection_reason": reason,
        "selection_sec": selection_sec,
        "full_fit_sec": full_fit_sec,
        "assignment_sec": assignment_sec,
        "quality_sec": quality_sec,
        "algorithm_sec": selection_sec + full_fit_sec + assignment_sec,
        "end_to_end_sec": selection_sec + full_fit_sec + assignment_sec + quality_sec,
        **quality,
    }


def _summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    metric_columns = [
        column
        for column in frame.columns
        if column.endswith("_sec")
        or column in {
            "selected_k",
            "silhouette",
            "xie_beni",
            "agreement_ari_vs_full_selection",
            "tag_nmi",
            "tag_ari",
            "class_nmi",
            "class_ari",
        }
    ]
    rows: list[dict[str, Any]] = []
    baseline_time = float(
        frame.loc[frame["strategy"] == "full_selection", "algorithm_sec"].mean()
    )
    for strategy in STRATEGIES:
        group = frame[frame["strategy"] == strategy]
        row: dict[str, Any] = {"strategy": strategy, "runs": int(len(group))}
        for metric in metric_columns:
            if metric in group:
                row[f"mean_{metric}"] = float(group[metric].mean())
                row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        mean_algorithm_sec = float(group["algorithm_sec"].mean())
        row["speedup_vs_full_selection"] = baseline_time / mean_algorithm_sec
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.sample_size < args.min_child_size * args.min_clusters:
        raise ValueError("--sample-size is too small for the requested minimum clusters")
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    if args.sample_size > len(embeddings):
        raise ValueError("--sample-size cannot exceed the dataset size")

    pca_started = time.perf_counter()
    fitted = fit_normalized_pca_projection(
        embeddings, n_components=args.pca_components, seed=args.pca_seed
    )
    features = fitted.normalized_prefix()
    pca_sec = time.perf_counter() - pca_started
    rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        sample_indices = np.sort(rng.choice(len(features), args.sample_size, replace=False))
        sampled_features = features[sample_indices]

        full_best, full_reason, full_selection_sec = _select_k(features, args, seed)
        if full_best is None:
            raise RuntimeError(f"full selection failed for seed {seed}: {full_reason}")
        quality_started = time.perf_counter()
        full_quality = _quality(features, metadata, full_best.result, None)
        full_quality_sec = time.perf_counter() - quality_started
        rows.append(
            _strategy_row(
                strategy="full_selection",
                seed=seed,
                selected_k=full_best.n_clusters,
                reason=full_reason,
                selection_sec=full_selection_sec,
                full_fit_sec=0.0,
                assignment_sec=0.0,
                quality_sec=full_quality_sec,
                quality=full_quality,
            )
        )
        print(f"seed={seed} full_selection K={full_best.n_clusters} {full_selection_sec:.2f}s", flush=True)

        sampled_best, sample_reason, sample_selection_sec = _select_k(
            sampled_features, args, seed
        )
        if sampled_best is None:
            raise RuntimeError(f"sample selection failed for seed {seed}: {sample_reason}")

        fit_started = time.perf_counter()
        refit = spherical_fcm(
            features,
            n_clusters=sampled_best.n_clusters,
            seed=seed + sampled_best.n_clusters * 1009,
            n_init=args.n_init,
            max_attempts=args.max_attempts,
            min_cluster_size=args.min_child_size,
            min_center_separation=args.min_center_separation,
        )
        refit_sec = time.perf_counter() - fit_started
        quality_started = time.perf_counter()
        refit_quality = _quality(features, metadata, refit, full_best.result.labels)
        refit_quality_sec = time.perf_counter() - quality_started
        rows.append(
            _strategy_row(
                strategy="sample_select_full_fit",
                seed=seed,
                selected_k=sampled_best.n_clusters,
                reason=sample_reason,
                selection_sec=sample_selection_sec,
                full_fit_sec=refit_sec,
                assignment_sec=0.0,
                quality_sec=refit_quality_sec,
                quality=refit_quality,
            )
        )
        print(f"seed={seed} sample_select_full_fit K={sampled_best.n_clusters} {sample_selection_sec + refit_sec:.2f}s", flush=True)

        assignment_started = time.perf_counter()
        projected = _project_to_centers(features, sampled_best.result)
        assignment_sec = time.perf_counter() - assignment_started
        quality_started = time.perf_counter()
        projected_quality = _quality(features, metadata, projected, full_best.result.labels)
        projected_quality_sec = time.perf_counter() - quality_started
        rows.append(
            _strategy_row(
                strategy="sample_select_project",
                seed=seed,
                selected_k=sampled_best.n_clusters,
                reason=sample_reason,
                selection_sec=sample_selection_sec,
                full_fit_sec=0.0,
                assignment_sec=assignment_sec,
                quality_sec=projected_quality_sec,
                quality=projected_quality,
            )
        )
        print(f"seed={seed} sample_select_project K={sampled_best.n_clusters} {sample_selection_sec + assignment_sec:.2f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    report = {
        "configuration": {
            "dataset": str(args.input_json),
            "samples": int(len(embeddings)),
            "embedding_dimension": int(embeddings.shape[1]),
            "pca_components": int(fitted.pca.n_components_),
            "pca_fit_sec_shared_excluded_from_strategy_timing": pca_sec,
            "sample_size": args.sample_size,
            "seeds": args.seeds,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "min_child_size": args.min_child_size,
            "n_init": args.n_init,
            "max_attempts": args.max_attempts,
            "strategies": {
                "full_selection": "select K and fit FCM on all samples (current baseline)",
                "sample_select_full_fit": "select K on a random sample, then refit FCM on all samples",
                "sample_select_project": "select K and centers on a random sample, then assign all samples to those fixed centers",
            },
        },
        "summary": _summary(frame),
        "runs": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
