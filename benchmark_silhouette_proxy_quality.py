"""Validate the silhouette proxy against exact clustering on Gemini embeddings.

The benchmark has two complementary comparisons:

* ``selector`` compares exhaustive exact-silhouette and exhaustive proxy
  selection on the same candidate K values.  This isolates the metric's
  effect on the selected K and reports per-K silhouette error and rank
  agreement.
* ``hierarchy`` compares the production exact and fast recursive paths.  It
  reports selected K, natural/final noise ratios, assignment agreement, and
  external/internal clustering quality.

The default input is deliberately not sampled.  Use
``--dataset-sample-size`` for a smaller smoke run; the validation command for
the supplied Gemini data passes ``3000`` explicitly.
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
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from embedding_data import load_embeddings_from_json, sample_embedding_batch
from fast_fcm import FastFcmConfig, _deterministic_sample, _scout_m
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fcm_validity import select_fcm_cluster_count
from hierarchical_fcm import run_hierarchical_pca_fcm
from pca_projection import fit_normalized_pca_projection


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values and non-finite floats for JSON output."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mean(values: list[Any]) -> float | None:
    finite = [number for value in values if (number := _finite_float(value)) is not None]
    return float(np.mean(finite)) if finite else None


def _candidate_map(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(record["k"]): record
        for record in records
        if record.get("k") is not None
    }


def _rank_correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 2 or len(first) != len(second):
        return None
    if np.ptp(first) <= 1e-12 or np.ptp(second) <= 1e-12:
        return None
    correlation = spearmanr(first, second).statistic
    return _finite_float(correlation)


def _run_selector(
    features: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
    use_silhouette_proxy: bool,
) -> tuple[Any, list[dict[str, Any]], str, float]:
    started_at = time.perf_counter()
    best, records, reason = select_fcm_cluster_count(
        features,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_child_size=args.min_child_size,
        min_membership=args.min_membership,
        max_membership_gap=args.max_membership_gap,
        distance_z=args.distance_z,
        selection_method="multi_metric",
        seed=seed,
        n_init=args.selector_n_init,
        max_attempts=args.selector_max_attempts,
        min_center_separation=args.min_center_separation,
        m=args.m,
        max_iter=args.max_fcm_iter,
        tol=args.fcm_tol,
        xb_worsening_patience=args.xb_worsening_patience,
        use_silhouette_proxy=use_silhouette_proxy,
    )
    return best, records, reason, time.perf_counter() - started_at


def _selector_comparison(
    features: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_best, exact_records, exact_reason, exact_seconds = _run_selector(
        features,
        args,
        seed=seed,
        use_silhouette_proxy=False,
    )
    proxy_best, proxy_records, proxy_reason, proxy_seconds = _run_selector(
        features,
        args,
        seed=seed,
        use_silhouette_proxy=True,
    )

    exact_by_k = _candidate_map(exact_records)
    proxy_by_k = _candidate_map(proxy_records)
    common_ks = sorted(set(exact_by_k) & set(proxy_by_k))
    exact_silhouettes = [
        float(exact_by_k[k]["silhouette"])
        for k in common_ks
        if _finite_float(exact_by_k[k].get("silhouette")) is not None
        and _finite_float(proxy_by_k[k].get("silhouette")) is not None
    ]
    proxy_silhouettes = [
        float(proxy_by_k[k]["silhouette"])
        for k in common_ks
        if _finite_float(exact_by_k[k].get("silhouette")) is not None
        and _finite_float(proxy_by_k[k].get("silhouette")) is not None
    ]
    silhouette_errors = [
        abs(exact_value - proxy_value)
        for exact_value, proxy_value in zip(
            exact_silhouettes,
            proxy_silhouettes,
            strict=True,
        )
    ]
    exact_selected_k = (
        int(exact_best.n_clusters) if exact_best is not None else None
    )
    proxy_selected_k = (
        int(proxy_best.n_clusters) if proxy_best is not None else None
    )
    selected_k_delta = (
        proxy_selected_k - exact_selected_k
        if exact_selected_k is not None and proxy_selected_k is not None
        else None
    )
    run: dict[str, Any] = {
        "phase": "selector_run",
        "seed": int(seed),
        "exact_selected_k": exact_selected_k,
        "proxy_selected_k": proxy_selected_k,
        "selected_k_delta_proxy_minus_exact": selected_k_delta,
        "k_agreement": (
            exact_selected_k is not None
            and exact_selected_k == proxy_selected_k
        ),
        "exact_reason": exact_reason,
        "proxy_reason": proxy_reason,
        "exact_runtime_sec": exact_seconds,
        "proxy_runtime_sec": proxy_seconds,
        "metric_speedup": (
            exact_seconds / proxy_seconds if proxy_seconds > 0.0 else None
        ),
        "candidate_count_exact": len(exact_by_k),
        "candidate_count_proxy": len(proxy_by_k),
        "common_candidate_ks": common_ks,
        "mean_abs_silhouette_error": _mean(silhouette_errors),
        "max_abs_silhouette_error": max(silhouette_errors)
        if silhouette_errors
        else None,
        "silhouette_spearman": _rank_correlation(
            exact_silhouettes,
            proxy_silhouettes,
        ),
        "exact_silhouette_by_k": {
            str(k): exact_by_k[k].get("silhouette") for k in common_ks
        },
        "proxy_silhouette_by_k": {
            str(k): proxy_by_k[k].get("silhouette") for k in common_ks
        },
        "exact_selection_score_by_k": {
            str(k): exact_by_k[k].get("selection_score") for k in common_ks
        },
        "proxy_selection_score_by_k": {
            str(k): proxy_by_k[k].get("selection_score") for k in common_ks
        },
    }
    candidate_rows: list[dict[str, Any]] = []
    for k in common_ks:
        exact_silhouette = _finite_float(exact_by_k[k].get("silhouette"))
        proxy_silhouette = _finite_float(proxy_by_k[k].get("silhouette"))
        candidate_rows.append(
            {
                "phase": "selector_candidate",
                "seed": int(seed),
                "k": int(k),
                "exact_silhouette": exact_silhouette,
                "proxy_silhouette": proxy_silhouette,
                "silhouette_delta_proxy_minus_exact": (
                    proxy_silhouette - exact_silhouette
                    if exact_silhouette is not None
                    and proxy_silhouette is not None
                    else None
                ),
                "abs_silhouette_error": (
                    abs(proxy_silhouette - exact_silhouette)
                    if exact_silhouette is not None
                    and proxy_silhouette is not None
                    else None
                ),
                "exact_selection_score": exact_by_k[k].get(
                    "selection_score"
                ),
                "proxy_selection_score": proxy_by_k[k].get(
                    "selection_score"
                ),
                "exact_is_selected": int(k == exact_selected_k),
                "proxy_is_selected": int(k == proxy_selected_k),
            }
        )
    return run, candidate_rows


def _scout_comparison(
    features: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare exact and proxy metrics on the fast path's scout sample."""

    config = FastFcmConfig(
        sample_size=args.fast_sample_size,
        scout_n_init=args.fast_scout_n_init,
        scout_max_attempts=max(
            args.fast_scout_n_init + 1,
            args.fast_scout_n_init * 2,
        ),
        scout_max_iter=min(args.max_fcm_iter, 60),
        scout_tol=max(args.fcm_tol, 1e-4),
        min_center_separation=args.min_center_separation,
    )
    scout_features = _deterministic_sample(features, config.sample_size, seed)
    selected_m, _probe_records = _scout_m(
        scout_features,
        min_child_size=args.min_child_size,
        min_clusters=args.min_clusters,
        max_membership_gap=args.max_membership_gap,
        distance_z=args.distance_z,
        selection_method="multi_metric",
        seed=seed,
        config=config,
    )
    scout_max_clusters = min(args.max_clusters, config.scout_max_clusters)
    common = {
        "min_clusters": args.min_clusters,
        "max_clusters": scout_max_clusters,
        "min_child_size": args.min_child_size,
        "min_membership": args.min_membership,
        "max_membership_gap": args.max_membership_gap,
        "distance_z": args.distance_z,
        "selection_method": "multi_metric",
        "seed": seed + 97,
        "n_init": config.scout_n_init,
        "max_attempts": config.scout_max_attempts,
        "min_center_separation": config.min_center_separation,
        "m": selected_m,
        "max_iter": config.scout_max_iter,
        "tol": config.scout_tol,
        "xb_worsening_patience": scout_max_clusters,
        "collapse_center_separation": config.min_center_separation,
    }
    exact_best, exact_records, exact_reason = select_fcm_cluster_count(
        scout_features,
        **common,
        use_silhouette_proxy=False,
    )
    proxy_best, proxy_records, proxy_reason = select_fcm_cluster_count(
        scout_features,
        **common,
        use_silhouette_proxy=True,
    )
    exact_by_k = _candidate_map(exact_records)
    proxy_by_k = _candidate_map(proxy_records)
    common_ks = sorted(set(exact_by_k) & set(proxy_by_k))
    exact_selected_k = (
        int(exact_best.n_clusters) if exact_best is not None else None
    )
    proxy_selected_k = (
        int(proxy_best.n_clusters) if proxy_best is not None else None
    )
    return {
        "phase": "scout_run",
        "selection_seed": int(seed),
        "scout_sample_size": int(len(scout_features)),
        "selected_m": float(selected_m),
        "exact_selected_k": exact_selected_k,
        "proxy_selected_k": proxy_selected_k,
        "k_agreement": (
            exact_selected_k is not None
            and exact_selected_k == proxy_selected_k
        ),
        "exact_reason": exact_reason,
        "proxy_reason": proxy_reason,
        "exact_silhouette_by_k": {
            str(k): exact_by_k[k].get("silhouette") for k in common_ks
        },
        "proxy_silhouette_by_k": {
            str(k): proxy_by_k[k].get("silhouette") for k in common_ks
        },
        "exact_selection_score_by_k": {
            str(k): exact_by_k[k].get("selection_score") for k in common_ks
        },
        "proxy_selection_score_by_k": {
            str(k): proxy_by_k[k].get("selection_score") for k in common_ks
        },
    }


def _labels_from_assignments(result: Any) -> tuple[np.ndarray, np.ndarray]:
    assignments = result.assignments
    if "cluster_path" in assignments:
        labels = assignments["cluster_path"].fillna("unassigned").astype(str).to_numpy()
    elif "cluster" in assignments:
        labels = assignments["cluster"].to_numpy().copy()
    else:
        raise ValueError("hierarchy assignments do not contain cluster labels")
    if "is_noise" in assignments:
        noise = assignments["is_noise"].astype(bool).to_numpy()
    else:
        noise = labels == "noise"
    labels = labels.astype(object, copy=True)
    labels[noise] = "__noise__"
    return labels, noise


def _node_selected_ks(node: dict[str, Any]) -> dict[str, int]:
    selected: dict[str, int] = {}
    path = str(node.get("path") or "<root>")
    if node.get("selected_k") is not None:
        selected[path] = int(node["selected_k"])
    for child in node.get("children", []):
        selected.update(_node_selected_ks(child))
    return selected


def _hierarchy_quality(
    result: Any,
    metadata: pd.DataFrame,
    features: np.ndarray,
) -> dict[str, Any]:
    labels, noise = _labels_from_assignments(result)
    non_noise_labels = labels[~noise]
    non_noise_features = features[~noise]
    cluster_count = len(np.unique(non_noise_labels))
    final_silhouette: float | None = None
    if (
        cluster_count >= 2
        and len(non_noise_labels) > cluster_count
    ):
        try:
            final_silhouette = float(
                silhouette_score(non_noise_features, non_noise_labels)
            )
        except ValueError:
            final_silhouette = None

    quality: dict[str, Any] = {
        "samples": int(len(labels)),
        "noise_ratio": float(np.mean(noise)),
        "non_noise_count": int(np.sum(~noise)),
        "final_cluster_count": int(cluster_count),
        "final_silhouette": final_silhouette,
        "natural_noise_ratio": (
            float(result.summary.get("natural_noise_count", 0)) / len(labels)
        ),
        "forced_noise_ratio": (
            float(result.summary.get("forced_noise_count", 0)) / len(labels)
        ),
        "boundary_ratio": float(
            result.summary.get("boundary_count", 0)
        )
        / len(labels),
        "root_selected_k": result.tree["root"].get("selected_k"),
        "root_noise_ratio": (
            float(result.tree["root"].get("noise_count", 0))
            / max(int(result.tree["root"].get("size", len(labels))), 1)
        ),
        "selected_k_by_path": _node_selected_ks(result.tree["root"]),
        "node_count": int(result.summary.get("node_count", 0)),
        "leaf_count": int(result.summary.get("leaf_count", 0)),
    }
    for target_name in ("tag", "class"):
        if target_name not in metadata:
            continue
        target = metadata[target_name].astype(str).to_numpy()
        quality[f"{target_name}_ari"] = float(
            adjusted_rand_score(target, labels)
        )
        quality[f"{target_name}_nmi"] = float(
            normalized_mutual_info_score(target, labels)
        )
    return quality


def _run_hierarchy_comparison(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    features: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, Any]:
    common = {
        "metadata": metadata,
        "max_depth": args.max_depth,
        "min_node_size": args.min_node_size,
        "min_child_size": args.min_child_size,
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "min_membership": args.min_membership,
        "max_membership_gap": args.max_membership_gap,
        "distance_z": args.distance_z,
        "selection_method": "multi_metric",
        "xb_worsening_patience": args.xb_worsening_patience,
        "pca_components": args.pca_components,
        "seed": seed,
        "m": args.m,
        "max_fcm_iter": args.max_fcm_iter,
        "fcm_tol": args.fcm_tol,
    }
    exact_started_at = time.perf_counter()
    exact_result = run_hierarchical_pca_fcm(
        embeddings,
        **common,
        fast_mode=False,
    )
    exact_seconds = time.perf_counter() - exact_started_at
    fast_started_at = time.perf_counter()
    fast_result = run_hierarchical_pca_fcm(
        embeddings,
        **common,
        fast_mode=True,
        fast_sample_size=args.fast_sample_size,
        fast_scout_n_init=args.fast_scout_n_init,
        fast_refine_n_init=args.fast_refine_n_init,
        fast_refine_top_k=args.fast_refine_top_k,
        fast_stability_target=args.fast_stability_target,
    )
    fast_seconds = time.perf_counter() - fast_started_at

    exact_quality = _hierarchy_quality(exact_result, metadata, features)
    fast_quality = _hierarchy_quality(fast_result, metadata, features)
    exact_labels, _exact_noise = _labels_from_assignments(exact_result)
    fast_labels, _fast_noise = _labels_from_assignments(fast_result)
    exact_ks = exact_quality["selected_k_by_path"]
    fast_ks = fast_quality["selected_k_by_path"]
    common_paths = sorted(set(exact_ks) & set(fast_ks))
    node_k_agreement = (
        float(np.mean([exact_ks[path] == fast_ks[path] for path in common_paths]))
        if common_paths
        else None
    )
    row: dict[str, Any] = {
        "phase": "hierarchy_run",
        "seed": int(seed),
        "exact_runtime_sec": exact_seconds,
        "fast_runtime_sec": fast_seconds,
        "speedup": fast_seconds and exact_seconds / fast_seconds,
        "root_k_agreement": (
            exact_quality["root_selected_k"]
            == fast_quality["root_selected_k"]
        ),
        "root_k_delta_fast_minus_exact": (
            int(fast_quality["root_selected_k"])
            - int(exact_quality["root_selected_k"])
            if exact_quality["root_selected_k"] is not None
            and fast_quality["root_selected_k"] is not None
            else None
        ),
        "node_k_agreement": node_k_agreement,
        "common_selected_k_paths": common_paths,
        "assignment_ari_fast_vs_exact": float(
            adjusted_rand_score(exact_labels, fast_labels)
        ),
        "assignment_nmi_fast_vs_exact": float(
            normalized_mutual_info_score(exact_labels, fast_labels)
        ),
        "exact_quality": exact_quality,
        "fast_quality": fast_quality,
    }
    for metric in (
        "noise_ratio",
        "natural_noise_ratio",
        "forced_noise_ratio",
        "root_noise_ratio",
        "final_silhouette",
        "tag_ari",
        "tag_nmi",
        "class_ari",
        "class_nmi",
    ):
        exact_value = _finite_float(exact_quality.get(metric))
        fast_value = _finite_float(fast_quality.get(metric))
        row[f"{metric}_delta_fast_minus_exact"] = (
            fast_value - exact_value
            if exact_value is not None and fast_value is not None
            else None
        )
    return row


def _summary(
    selector_rows: list[dict[str, Any]],
    scout_rows: list[dict[str, Any]],
    hierarchy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selector_agreements = [row["k_agreement"] for row in selector_rows]
    hierarchy_agreements = [row["root_k_agreement"] for row in hierarchy_rows]
    selector_summary: dict[str, Any] = {
        "runs": len(selector_rows),
        "k_agreement_rate": (
            float(np.mean(selector_agreements)) if selector_agreements else None
        ),
        "mean_abs_selected_k_delta": _mean(
            [
                abs(row["selected_k_delta_proxy_minus_exact"])
                for row in selector_rows
                if row["selected_k_delta_proxy_minus_exact"] is not None
            ]
        ),
        "mean_abs_silhouette_error": _mean(
            [row["mean_abs_silhouette_error"] for row in selector_rows]
        ),
        "max_abs_silhouette_error": max(
            [
                value
                for row in selector_rows
                if (value := _finite_float(row["max_abs_silhouette_error"]))
                is not None
            ],
            default=None,
        ),
        "mean_silhouette_spearman": _mean(
            [row["silhouette_spearman"] for row in selector_rows]
        ),
        "mean_metric_speedup": _mean(
            [row["metric_speedup"] for row in selector_rows]
        ),
        "exact_selected_k_counts": dict(
            Counter(
                str(row["exact_selected_k"])
                for row in selector_rows
                if row["exact_selected_k"] is not None
            )
        ),
        "proxy_selected_k_counts": dict(
            Counter(
                str(row["proxy_selected_k"])
                for row in selector_rows
                if row["proxy_selected_k"] is not None
            )
        ),
    }
    hierarchy_summary: dict[str, Any] = {
        "runs": len(hierarchy_rows),
        "root_k_agreement_rate": (
            float(np.mean(hierarchy_agreements))
            if hierarchy_agreements
            else None
        ),
        "mean_abs_noise_ratio_delta": _mean(
            [
                abs(row["noise_ratio_delta_fast_minus_exact"])
                for row in hierarchy_rows
                if row["noise_ratio_delta_fast_minus_exact"] is not None
            ]
        ),
        "max_abs_noise_ratio_delta": max(
            [
                abs(value)
                for row in hierarchy_rows
                if (
                    value := _finite_float(
                        row["noise_ratio_delta_fast_minus_exact"]
                    )
                )
                is not None
            ],
            default=None,
        ),
        "mean_assignment_ari_fast_vs_exact": _mean(
            [row["assignment_ari_fast_vs_exact"] for row in hierarchy_rows]
        ),
        "mean_assignment_nmi_fast_vs_exact": _mean(
            [row["assignment_nmi_fast_vs_exact"] for row in hierarchy_rows]
        ),
        "mean_quality_deltas_fast_minus_exact": {
            metric: _mean(
                [
                    row[f"{metric}_delta_fast_minus_exact"]
                    for row in hierarchy_rows
                ]
            )
            for metric in (
                "noise_ratio",
                "natural_noise_ratio",
                "final_silhouette",
                "tag_ari",
                "tag_nmi",
                "class_ari",
                "class_nmi",
            )
        },
        "mean_speedup": _mean([row["speedup"] for row in hierarchy_rows]),
    }
    scout_agreements = [row["k_agreement"] for row in scout_rows]
    scout_summary = {
        "runs": len(scout_rows),
        "k_agreement_rate": (
            float(np.mean(scout_agreements)) if scout_agreements else None
        ),
        "exact_selected_k_counts": dict(
            Counter(
                str(row["exact_selected_k"])
                for row in scout_rows
                if row["exact_selected_k"] is not None
            )
        ),
        "proxy_selected_k_counts": dict(
            Counter(
                str(row["proxy_selected_k"])
                for row in scout_rows
                if row["proxy_selected_k"] is not None
            )
        ),
    }
    return {
        "selector": selector_summary,
        "scout": scout_summary,
        "hierarchy": hierarchy_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exact and proxy silhouette clustering quality."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dataset-sample-size", type=int, default=None)
    parser.add_argument("--dataset-sample-seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--min-child-size", type=int, default=50)
    parser.add_argument("--min-node-size", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-membership", type=float, default=0.40)
    parser.add_argument("--max-membership-gap", type=float, default=0.10)
    parser.add_argument("--distance-z", type=float, default=3.5)
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--selector-n-init", type=int, default=DEFAULT_FCM_N_INIT)
    parser.add_argument("--selector-max-attempts", type=int, default=None)
    parser.add_argument("--max-fcm-iter", type=int, default=200)
    parser.add_argument("--fcm-tol", type=float, default=1e-6)
    parser.add_argument("--xb-worsening-patience", type=int, default=2)
    parser.add_argument(
        "--min-center-separation",
        type=float,
        default=DEFAULT_FCM_MIN_CENTER_SEPARATION,
    )
    parser.add_argument("--fast-sample-size", type=int, default=1000)
    parser.add_argument("--fast-scout-n-init", type=int, default=2)
    parser.add_argument("--fast-refine-n-init", type=int, default=3)
    parser.add_argument("--fast-refine-top-k", type=int, default=2)
    parser.add_argument("--fast-stability-target", type=float, default=0.85)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, row_count: int) -> None:
    if args.dataset_sample_size is not None and not (
        2 <= args.dataset_sample_size <= row_count
    ):
        raise ValueError("--dataset-sample-size must be between 2 and dataset size")
    if args.pca_components < 1:
        raise ValueError("--pca-components must be positive")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if args.min_clusters < 2 or args.max_clusters < args.min_clusters:
        raise ValueError("invalid cluster-count bounds")
    if args.min_child_size < 2 or args.min_node_size < args.min_child_size:
        raise ValueError("invalid node and child sizes")
    if args.max_depth < 1:
        raise ValueError("--max-depth must be positive")
    if args.selector_n_init < 1:
        raise ValueError("--selector-n-init must be positive")
    if args.selector_max_attempts is not None and (
        args.selector_max_attempts < args.selector_n_init
    ):
        raise ValueError("--selector-max-attempts must cover --selector-n-init")
    if args.fast_sample_size < 2:
        raise ValueError("--fast-sample-size must be at least 2")


def main() -> None:
    args = parse_args()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    _validate_args(args, len(embeddings))
    if args.dataset_sample_size is not None:
        embeddings, metadata = sample_embedding_batch(
            embeddings,
            metadata,
            sample_size=args.dataset_sample_size,
            seed=args.dataset_sample_seed,
        )
    fitted = fit_normalized_pca_projection(
        embeddings,
        n_components=args.pca_components,
        seed=args.pca_seed,
    )
    features = fitted.normalized_prefix()

    scout_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    selector_candidate_rows: list[dict[str, Any]] = []
    for seed in dict.fromkeys(args.seeds):
        # The hierarchy derives its root selector seed from the experiment
        # seed and the root node size.  Reuse that convention here so the
        # exhaustive exact/proxy comparison is aligned with the production
        # root decision rather than comparing an unrelated FCM seed.
        selection_seed = int(seed) + len(features)
        scout_run = _scout_comparison(
            features,
            args,
            seed=selection_seed,
        )
        scout_run["experiment_seed"] = int(seed)
        scout_rows.append(scout_run)
        run, candidate_rows = _selector_comparison(
            features,
            args,
            seed=selection_seed,
        )
        run["experiment_seed"] = int(seed)
        selector_rows.append(run)
        selector_candidate_rows.extend(candidate_rows)
        print(
            f"scout seed={seed} (selection_seed={selection_seed}): "
            f"exact K={scout_run['exact_selected_k']} "
            f"proxy K={scout_run['proxy_selected_k']} "
            f"agree={scout_run['k_agreement']}"
        )
        print(
            f"selector seed={seed} (selection_seed={selection_seed}): "
            f"exact K={run['exact_selected_k']} "
            f"proxy K={run['proxy_selected_k']} "
            f"agree={run['k_agreement']} "
            f"mean_abs_silhouette_error={run['mean_abs_silhouette_error']}"
        )

    hierarchy_rows: list[dict[str, Any]] = []
    for seed in dict.fromkeys(args.seeds):
        row = _run_hierarchy_comparison(
            embeddings,
            metadata,
            features,
            args,
            seed=int(seed),
        )
        hierarchy_rows.append(row)
        exact_quality = row["exact_quality"]
        fast_quality = row["fast_quality"]
        print(
            f"hierarchy seed={seed}: exact root K={exact_quality['root_selected_k']} "
            f"fast root K={fast_quality['root_selected_k']} "
            f"noise exact={exact_quality['noise_ratio']:.4f} "
            f"fast={fast_quality['noise_ratio']:.4f} "
            f"assignment_ari={row['assignment_ari_fast_vs_exact']:.4f}"
        )

    summary = _summary(selector_rows, scout_rows, hierarchy_rows)
    all_rows: list[dict[str, Any]] = [
        *scout_rows,
        *selector_rows,
        *selector_candidate_rows,
    ]
    for row in hierarchy_rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"exact_quality", "fast_quality"}
        }
        for prefix, quality in (("exact", row["exact_quality"]), ("fast", row["fast_quality"])):
            for key, value in quality.items():
                flat[f"{prefix}_{key}"] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
        all_rows.append(flat)

    payload = _json_safe(
        {
            "configuration": {
                "input_json": str(args.input_json),
                "samples_in_input": int(len(embeddings)),
                "embedding_dimension": int(embeddings.shape[1]),
                "dataset_sample_size": args.dataset_sample_size,
                "dataset_sample_seed": args.dataset_sample_seed,
                "pca_components": int(features.shape[1]),
                "pca_seed": args.pca_seed,
                "seeds": list(dict.fromkeys(args.seeds)),
                "min_clusters": args.min_clusters,
                "max_clusters": args.max_clusters,
                "min_child_size": args.min_child_size,
                "min_node_size": args.min_node_size,
                "max_depth": args.max_depth,
                "selector_n_init": args.selector_n_init,
                "selector_max_attempts": args.selector_max_attempts,
                "fast_sample_size": args.fast_sample_size,
                "fast_scout_n_init": args.fast_scout_n_init,
                "fast_refine_n_init": args.fast_refine_n_init,
                "fast_refine_top_k": args.fast_refine_top_k,
                "fast_refine_score_margin": 0.15,
            },
            "summary": summary,
            "scout_runs": scout_rows,
            "selector_runs": selector_rows,
            "selector_candidates": selector_candidate_rows,
            "hierarchy_runs": hierarchy_rows,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)
    print(f"Saved benchmark JSON: {args.output_json}")
    print(f"Saved benchmark CSV: {args.output_csv}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
