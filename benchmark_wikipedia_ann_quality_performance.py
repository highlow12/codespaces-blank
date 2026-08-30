"""Separate ANN quality (original Wikipedia rows) from scaling timings.

The quality report never uses replicated rows.  ``--target-size`` is only
used for the optional scaling report, whose labels and clustering scores are
intentionally not used for an ANN acceptance decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pca_neighbor_search import build_pca_neighbor_index
from wikipedia_soft_benchmark.embeddings import l2_normalize
from wikipedia_soft_benchmark.benchmark_helpers import scale_splits, split_data
from wikipedia_soft_benchmark.hierarchy_benchmark import (
    DEFAULT_MIN_CLUSTER_SIZES,
    DEFAULT_MIN_SAMPLES,
    calibration_sweep,
    evaluate_split,
    fit_discovery,
    load_metadata,
    predict_memberships,
)

SEEDS = (42, 43, 44, 45, 46)
K_VALUES = (8, 15, 24)


def _neighbor_recall(exact: tuple[np.ndarray, np.ndarray], ann: tuple[np.ndarray, np.ndarray]) -> float:
    exact_indices, ann_indices = exact[1], ann[1]
    return float(np.mean([
        len(set(expected).intersection(observed)) / len(expected)
        for expected, observed in zip(exact_indices, ann_indices, strict=True)
    ]))


def _quality_seed(split, seed: int, pca_components: int | None, graph_neighbors: int, epsilon: float) -> dict[str, Any]:
    discovery, discovery_rows = split["discovery"]
    calibration, calibration_rows = split["calibration"]
    test, test_rows = split["test"]
    if pca_components is None:
        from pca_dimension_selection import select_pca_dimension_for_data
        pca_selection = select_pca_dimension_for_data(discovery, seed=seed)
        if pca_selection is None:
            raise ValueError("automatic PCA selection returned no result")
        pca_components = int(pca_selection.selected_dimension)
    sweep, selected = calibration_sweep(
        discovery, discovery_rows, calibration, calibration_rows,
        seeds=(seed,), min_cluster_sizes=DEFAULT_MIN_CLUSTER_SIZES,
        min_samples_values=DEFAULT_MIN_SAMPLES,
        neighbor_counts=K_VALUES, pca_components=pca_components,
        neighbor_backend="exact", jobs=1,
    )
    state = fit_discovery(
        discovery, discovery_rows, seed=seed,
        min_cluster_size=int(selected["min_cluster_size"]),
        min_samples=int(selected["min_samples"]), pca_components=pca_components,
        neighbor_backend="exact", neighbor_max_k=max(K_VALUES),
        neighbor_graph_neighbors=graph_neighbors, neighbor_query_epsilon=epsilon,
    )
    # Both methods see exactly the same PCA/HDBSCAN discovery state.  Only
    # the PCA neighbor index changes, isolating ANN approximation quality.
    exact_index = build_pca_neighbor_index(state.pca_discovery, max_neighbors=max(K_VALUES), backend="exact")
    ann_index = build_pca_neighbor_index(
        state.pca_discovery, max_neighbors=max(K_VALUES), backend="pynndescent",
        graph_neighbors=graph_neighbors, random_state=42, query_epsilon=epsilon,
    )
    test_pca = np.asarray(state.pca.transform(l2_normalize(np.asarray(test, dtype=np.float32))), dtype=np.float64)
    exact_neighbors = exact_index.query(test_pca, max(K_VALUES))
    ann_neighbors = ann_index.query(test_pca, max(K_VALUES))
    rows = []
    for k in K_VALUES:
        exact = evaluate_split(state, test, test_rows, neighbor_count=k, neighbor_backend="exact", neighbor_index=exact_index, neighbor_results=(exact_neighbors[0][:, :k], exact_neighbors[1][:, :k]), include_labels=True)["exact_knn"]
        ann = evaluate_split(state, test, test_rows, neighbor_count=k, neighbor_backend="pynndescent", neighbor_index=ann_index, neighbor_results=(ann_neighbors[0][:, :k], ann_neighbors[1][:, :k]), include_labels=True)["exact_knn"]
        rows.append({
            "k": k, "exact": exact, "pynndescent": ann,
            "recommended_leaf_agreement": float(np.mean(np.asarray(exact["mapped_labels"]) == np.asarray(ann["mapped_labels"]))),
            "neighbor_recall": _neighbor_recall((exact_neighbors[0][:, :k], exact_neighbors[1][:, :k]), (ann_neighbors[0][:, :k], ann_neighbors[1][:, :k])),
        })
        # Mapped labels are useful for agreement, while affinity error must
        # use the raw matrices and is filled below by the direct predictions.
        exact_prediction = predict_memberships(state, test, neighbor_count=k, neighbor_backend="exact", neighbor_index=exact_index, neighbor_results=(exact_neighbors[0][:, :k], exact_neighbors[1][:, :k]))
        ann_prediction = predict_memberships(state, test, neighbor_count=k, neighbor_backend="pynndescent", neighbor_index=ann_index, neighbor_results=(ann_neighbors[0][:, :k], ann_neighbors[1][:, :k]))
        rows[-1]["affinity_mae"] = float(np.mean(np.abs(exact_prediction.exact_knn - ann_prediction.exact_knn)))
    return {"seed": seed, "selected": dict(selected), "pca_components": int(pca_components), "k": rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.seeds:
        raise ValueError("at least one seed is required")
    embeddings = np.load(args.embedding_dir / "document_embeddings.npy")
    metadata = load_metadata(args.embedding_dir / "document_metadata.jsonl")
    original = split_data(embeddings, metadata)
    args.quality_output_dir.mkdir(parents=True, exist_ok=True)
    quality = [_quality_seed(original, seed, args.pca_components, args.graph_neighbors, args.query_epsilon) for seed in args.seeds]
    all_rows = [row for run_record in quality for row in run_record["k"]]
    quality_report = {
        "dataset": {"type": "original", "rows": len(embeddings), "quality_eligible": True},
        "protocol": {"seeds": list(args.seeds), "k_values": list(K_VALUES), "ann_backend": "pynndescent", "random_state": 42, "n_jobs": 1, "query_epsilon": args.query_epsilon, "graph_neighbors": args.graph_neighbors},
        "runs": quality,
        "aggregate": {
            "mean_exact_leaf_nmi": float(np.mean([row["exact"]["leaf_nmi"] for row in all_rows])),
            "mean_pynndescent_leaf_nmi": float(np.mean([row["pynndescent"]["leaf_nmi"] for row in all_rows])),
            "max_leaf_nmi_drop": float(max(row["exact"]["leaf_nmi"] - row["pynndescent"]["leaf_nmi"] for row in all_rows)),
            "min_recommended_leaf_agreement": float(min(row["recommended_leaf_agreement"] for row in all_rows)),
        },
        "acceptance": {"leaf_nmi_drop_max": 0.01, "recommended_leaf_agreement_min": 0.95},
    }
    quality_report["acceptance"]["passed"] = bool(
        quality_report["aggregate"]["max_leaf_nmi_drop"] <= 0.01
        and quality_report["aggregate"]["min_recommended_leaf_agreement"] >= 0.95
    )
    (args.quality_output_dir / "report.json").write_text(json.dumps(quality_report, indent=2, allow_nan=False), encoding="utf-8")
    with (args.quality_output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "k", "exact_leaf_nmi", "pynndescent_leaf_nmi", "recommended_leaf_agreement", "affinity_mae", "neighbor_recall"])
        writer.writeheader()
        for run_record in quality:
            for row in run_record["k"]:
                writer.writerow({"seed": run_record["seed"], "k": row["k"], "exact_leaf_nmi": row["exact"]["leaf_nmi"], "pynndescent_leaf_nmi": row["pynndescent"]["leaf_nmi"], **{key: row[key] for key in ("recommended_leaf_agreement", "affinity_mae", "neighbor_recall")}})
    scaling_report = None
    if args.target_size:
        scaled = scale_splits(original, args.target_size)
        discovery, discovery_rows = scaled["discovery"]
        calibration, calibration_rows = scaled["calibration"]
        test, test_rows = scaled["test"]
        started = time.perf_counter()
        scaling_pca = args.pca_components or 256
        sweep, selected = calibration_sweep(discovery, discovery_rows, calibration, calibration_rows, seeds=(42,), min_cluster_sizes=DEFAULT_MIN_CLUSTER_SIZES, min_samples_values=DEFAULT_MIN_SAMPLES, neighbor_counts=K_VALUES, pca_components=scaling_pca, neighbor_backend="pynndescent", neighbor_graph_neighbors=args.graph_neighbors, neighbor_query_epsilon=args.query_epsilon, jobs=1)
        state = fit_discovery(discovery, discovery_rows, seed=42, min_cluster_size=int(selected["min_cluster_size"]), min_samples=int(selected["min_samples"]), pca_components=scaling_pca, neighbor_backend="pynndescent", neighbor_max_k=max(K_VALUES), neighbor_graph_neighbors=args.graph_neighbors, neighbor_query_epsilon=args.query_epsilon)
        calibration_pca = state.pca.transform(l2_normalize(np.asarray(calibration, dtype=np.float32)))
        test_pca = state.pca.transform(l2_normalize(np.asarray(test, dtype=np.float32)))
        build_started = time.perf_counter(); index = build_pca_neighbor_index(state.pca_discovery, backend="pynndescent", max_neighbors=max(K_VALUES), graph_neighbors=args.graph_neighbors); index_build = time.perf_counter() - build_started
        query_started = time.perf_counter(); index.query(calibration_pca, max(K_VALUES)); calibration_query = time.perf_counter() - query_started
        query_started = time.perf_counter(); index.query(test_pca, max(K_VALUES)); test_query = time.perf_counter() - query_started
        scaling_report = {"dataset": {"type": "replicated", "rows": args.target_size, "quality_eligible": False}, "timing_sec": {"calibration_and_fit": time.perf_counter() - started, "index_build": index_build, "calibration_query": calibration_query, "test_query": test_query}, "selected": dict(selected), "quality_use_prohibited": True}
        args.scaling_output_dir.mkdir(parents=True, exist_ok=True)
        (args.scaling_output_dir / "report.json").write_text(json.dumps(scaling_report, indent=2, allow_nan=False), encoding="utf-8")
    return {"quality": quality_report, "scaling": scaling_report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--quality-output-dir", type=Path, required=True)
    parser.add_argument("--scaling-output-dir", type=Path, default=Path("ann-scaling"))
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--pca-components", type=int, default=None, help="PCA width; omit to select automatically for quality")
    parser.add_argument("--graph-neighbors", type=int, default=32)
    parser.add_argument("--query-epsilon", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
