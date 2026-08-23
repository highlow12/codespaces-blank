"""Split-aware Wikipedia comparison of HDBSCAN+kNN and hierarchical FCM.

The benchmark fits both pipelines on the discovery split, uses calibration
labels only to select the HDBSCAN density/kNN configuration, and evaluates the
test split once.  Hierarchical FCM selects PCA width and cluster count without
test labels.  Runs are deliberately sequential so the command can be pinned to
one CPU with ``taskset``.
"""

from __future__ import annotations

import argparse
import csv
import os
import resource
import threading
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)

from hierarchical_fcm import run_hierarchical_pca_fcm
from incremental_clustering import assign_to_hierarchy
from pca_dimension_selection import select_pca_dimension_for_data
from wikipedia_soft_benchmark.hierarchy_benchmark import (
    calibration_sweep,
    evaluate_split,
    fit_discovery,
    load_metadata,
    predict_memberships,
)
from wikipedia_soft_benchmark.embeddings import l2_normalize


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
LABEL_FIELDS = ("leaf", "parent", "top")


def _rss_kib() -> float:
    """Return current resident memory for stage-local peak measurements."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        # macOS and restricted runners may not expose procfs.  ru_maxrss is
        # still useful as a monotonic process-level fallback.
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / 1024.0 if value > 10_000 else value


class _MeasuredStage:
    """Measure wall time and peak RSS concurrently for one pipeline stage."""

    def __init__(self) -> None:
        self.elapsed_sec = 0.0
        self.peak_rss_kib = 0.0
        self.baseline_rss_kib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_MeasuredStage":
        self.baseline_rss_kib = _rss_kib()
        self.peak_rss_kib = self.baseline_rss_kib
        started = time.perf_counter()

        def sample() -> None:
            while not self._stop.wait(0.01):
                self.peak_rss_kib = max(self.peak_rss_kib, _rss_kib())

        self._started = started
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_kib = max(self.peak_rss_kib, _rss_kib())
        self.elapsed_sec = time.perf_counter() - self._started

    def as_dict(self) -> dict[str, float]:
        return {
            "sec": float(self.elapsed_sec),
            "baseline_rss_kib": float(self.baseline_rss_kib),
            "peak_rss_kib": float(self.peak_rss_kib),
            "peak_rss_delta_kib": float(max(0.0, self.peak_rss_kib - self.baseline_rss_kib)),
        }


def _split_data(
    embeddings: np.ndarray, metadata: list[dict[str, Any]]
) -> dict[str, tuple[np.ndarray, list[dict[str, Any]]]]:
    result = {}
    for split in ("discovery", "calibration", "test"):
        indices = [i for i, row in enumerate(metadata) if row.get("split") == split]
        if not indices:
            raise ValueError(f"Wikipedia metadata has no {split!r} rows")
        result[split] = (embeddings[indices], [metadata[i] for i in indices])
    return result


def _repeat_rows(
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    target_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Repeat one split to an exact size while assigning unique row IDs."""
    if target_size < 1:
        raise ValueError("target split size must be positive")
    indices = np.arange(target_size, dtype=np.int64) % len(rows)
    repeated_embeddings = np.asarray(embeddings[indices], dtype=embeddings.dtype)
    repeated_rows: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(indices):
        row = dict(rows[int(source_index)])
        original_id = row.get("id", row.get("source_id", int(source_index)))
        row["original_id"] = original_id
        row["id"] = f"{original_id}__repeat_{output_index}"
        row["source_id"] = row["id"]
        repeated_rows.append(row)
    return repeated_embeddings, repeated_rows


def _scale_splits(
    split: dict[str, tuple[np.ndarray, list[dict[str, Any]]]],
    target_size: int | None,
) -> dict[str, tuple[np.ndarray, list[dict[str, Any]]]]:
    if target_size is None:
        return split
    if target_size < 5:
        raise ValueError("--target-size must be at least 5")
    discovery_size = int(round(target_size * 0.60))
    calibration_size = int(round(target_size * 0.20))
    test_size = target_size - discovery_size - calibration_size
    sizes = {
        "discovery": discovery_size,
        "calibration": calibration_size,
        "test": test_size,
    }
    return {
        name: _repeat_rows(*split[name], sizes[name])
        for name in ("discovery", "calibration", "test")
    }


def _majority_mapping(
    paths: Sequence[str], rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, str]:
    grouped: dict[str, Counter[str]] = {}
    for path, row in zip(paths, rows, strict=True):
        if path == "__noise__":
            continue
        grouped.setdefault(path, Counter())[str(row.get(field, ""))] += 1
    return {
        path: sorted(counts, key=lambda value: (-counts[value], value))[0]
        for path, counts in grouped.items()
    }


def _evaluate_paths(
    train_assignments: pd.DataFrame,
    train_rows: Sequence[Mapping[str, Any]],
    test_assignments: pd.DataFrame,
    test_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_paths = np.where(
        train_assignments["is_noise"].to_numpy(dtype=bool),
        "__noise__",
        train_assignments["cluster_path"].fillna("").astype(str),
    )
    test_paths = np.where(
        test_assignments["is_noise"].to_numpy(dtype=bool),
        "__noise__",
        test_assignments["cluster_path"].fillna("").astype(str),
    )
    output: dict[str, Any] = {
        "non_noise_coverage": float(np.mean(test_paths != "__noise__")),
        "noise_count": int(np.sum(test_paths == "__noise__")),
        "cluster_count": int(len(set(test_paths) - {"__noise__"})),
    }
    mappings: dict[str, dict[str, str]] = {}
    for field in LABEL_FIELDS:
        truth = np.asarray([str(row.get(field, "")) for row in test_rows])
        mapping = _majority_mapping(train_paths, train_rows, field)
        mappings[field] = mapping
        mapped = np.asarray(
            [mapping.get(path, "__noise__") for path in test_paths], dtype=object
        )
        known = mapped != "__noise__"
        metrics = {
            "nmi": float(normalized_mutual_info_score(truth, test_paths)),
            "ari": float(adjusted_rand_score(truth, test_paths)),
            "mapped_macro_f1": (
                float(f1_score(truth[known], mapped[known], average="macro", zero_division=0))
                if np.any(known)
                else 0.0
            ),
            "mapped_balanced_accuracy": (
                float(balanced_accuracy_score(truth[known], mapped[known]))
                if np.any(known)
                else 0.0
            ),
        }
        output[field] = metrics

    distances = []
    for row, path in zip(test_rows, test_paths, strict=True):
        if path == "__noise__" or path not in mappings["leaf"]:
            distances.append(3.0)
        elif mappings["leaf"][path] == str(row.get("leaf", "")):
            distances.append(0.0)
        elif mappings["parent"].get(path) == str(row.get("parent", "")):
            distances.append(1.0)
        elif mappings["top"].get(path) == str(row.get("top", "")):
            distances.append(2.0)
        else:
            distances.append(3.0)
    output["hierarchy_distance"] = float(np.mean(distances))
    output["paths"] = test_paths.tolist()
    return output


def _plot_seed(
    output: Path,
    coordinates: np.ndarray,
    test_rows: Sequence[Mapping[str, Any]],
    hdbscan_labels: Sequence[int],
    fcm_paths: Sequence[str],
    seed: int,
) -> None:
    truth = [str(row.get("leaf", "")) for row in test_rows]
    panels = (
        (truth, "Ground-truth leaf"),
        ([str(value) for value in hdbscan_labels], "HDBSCAN → PCA exact-kNN"),
        (list(fcm_paths), "Hierarchical PCA + spherical FCM"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for axis, (values, title) in zip(axes, panels, strict=True):
        categories = {value: index for index, value in enumerate(sorted(set(values)))}
        colors = np.asarray([categories[value] for value in values])
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=25, cmap="tab20", alpha=0.85)
        axis.set_title(f"{title}\n{len(categories)} groups")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"Wikipedia held-out test split — seed {seed}")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.embedding_dir
    embeddings = np.load(root / "document_embeddings.npy")
    metadata = load_metadata(root / "document_metadata.jsonl")
    split = _scale_splits(_split_data(embeddings, metadata), args.target_size)
    discovery, discovery_rows = split["discovery"]
    calibration, calibration_rows = split["calibration"]
    test, test_rows = split["test"]
    discovery_frame = pd.DataFrame(discovery_rows)
    test_frame = pd.DataFrame(test_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, seed in enumerate(args.seeds, start=1):
        print(f"[{index}/{len(args.seeds)}] seed={seed}: automatic PCA", flush=True)
        run_started = time.perf_counter()
        pca_stage = _MeasuredStage()
        with pca_stage:
            pca_selection = select_pca_dimension_for_data(discovery, seed=seed)
        if pca_selection is None:
            raise RuntimeError("automatic PCA selection returned no result")
        pca_components = int(pca_selection.selected_dimension)
        hdb_auto_pca_sec = pca_stage.elapsed_sec

        print(f"[{index}/{len(args.seeds)}] seed={seed}: HDBSCAN calibration", flush=True)
        calibration_stage = _MeasuredStage()
        with calibration_stage:
            sweep, selected, calibration_artifacts = calibration_sweep(
                discovery,
                discovery_rows,
                calibration,
                calibration_rows,
                seeds=(seed,),
                pca_components=pca_components,
                jobs=1,
                return_prepared=True,
            )
        hdb_calibration_sec = calibration_stage.elapsed_sec
        # The selected candidate was already fitted during calibration.  Keep
        # its PCA, exact-kNN index, UMAP and HDBSCAN objects for the test.
        hdb_state = calibration_artifacts.selected_state
        neighbor_count = int(selected["neighbor_count"])
        test_matrix = l2_normalize(np.asarray(test, dtype=np.float32))
        test_pca_stage = _MeasuredStage()
        with test_pca_stage:
            test_pca = np.asarray(hdb_state.pca.transform(test_matrix), dtype=np.float64)
        test_umap_stage = _MeasuredStage()
        with test_umap_stage:
            test_umap = np.asarray(hdb_state.umap.transform(test_pca), dtype=np.float64)
        test_neighbor_stage = _MeasuredStage()
        with test_neighbor_stage:
            test_neighbor_results = hdb_state.neighbor_index.query(test_pca, neighbor_count, exclude_self=False)
        test_prediction_stage = _MeasuredStage()
        with test_prediction_stage:
            hdb_prediction = predict_memberships(
                hdb_state, test_matrix, neighbor_count=neighbor_count,
                neighbor_results=test_neighbor_results,
                pca_features=test_pca, umap_features=test_umap,
            )
            hdb_metrics = evaluate_split(
                hdb_state, test_matrix, test_rows, neighbor_count=neighbor_count,
                neighbor_results=test_neighbor_results,
                pca_features=test_pca, umap_features=test_umap,
            )["exact_knn"]

        print(f"[{index}/{len(args.seeds)}] seed={seed}: hierarchical FCM", flush=True)
        fcm_fit_stage = _MeasuredStage()
        with fcm_fit_stage:
            fcm = run_hierarchical_pca_fcm(
                discovery,
                discovery_frame,
                seed=seed,
                pca_components=None,
                max_depth=args.max_depth,
                min_node_size=args.min_node_size,
                min_child_size=args.min_child_size,
                fast_mode=False,
                consensus_k_selection=False,
                forced_noise_ratio=0.0,
            )
        if fcm.model is None:
            raise RuntimeError("hierarchical FCM did not return a fitted model")
        fcm_assignment_stage = _MeasuredStage()
        with fcm_assignment_stage:
            fcm_test, _ = assign_to_hierarchy(
                test,
                test_frame,
                fcm.model,
                min_membership=0.40,
                forced_noise_ratio=0.0,
            )
        fcm_metrics = _evaluate_paths(
            fcm.assignments, discovery_rows, fcm_test, test_rows
        )

        visualization_stage = _MeasuredStage()
        with visualization_stage:
            # Reuse the test UMAP transform measured above as well.
            test_coordinates = test_umap
            seed_dir = args.output_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            plot_path = seed_dir / "comparison.png"
            _plot_seed(
                plot_path,
                test_coordinates,
                test_rows,
                hdb_prediction.exact_labels,
                fcm_metrics["paths"],
                seed,
            )
        timing_details = {
            "hdbscan_automatic_pca": pca_stage.as_dict(),
            "hdbscan_calibration_total": {
                **calibration_stage.as_dict(),
                **calibration_artifacts.timing_sec,
            },
            "hdbscan_final_selected_state": {"sec": 0.0, "reused_from_calibration": True},
            "hdbscan_test_pca_transform": test_pca_stage.as_dict(),
            "hdbscan_test_umap_transform": test_umap_stage.as_dict(),
            "hdbscan_test_neighbor_query": test_neighbor_stage.as_dict(),
            "hdbscan_test_prediction": test_prediction_stage.as_dict(),
            "fcm_fit": fcm_fit_stage.as_dict(),
            "fcm_test_assignment": fcm_assignment_stage.as_dict(),
            "visualization": visualization_stage.as_dict(),
        }
        visualization_sec = visualization_stage.elapsed_sec
        hdbscan_final_fit_and_test_sec = (
            test_pca_stage.elapsed_sec + test_umap_stage.elapsed_sec
            + test_neighbor_stage.elapsed_sec + test_prediction_stage.elapsed_sec
        )
        timing = {
            "hdbscan_auto_pca_sec": float(hdb_auto_pca_sec),
            "hdbscan_calibration_sec": float(hdb_calibration_sec),
            "hdbscan_final_fit_and_test_sec": float(
                hdbscan_final_fit_and_test_sec
            ),
            "hdbscan_final_selected_state_reused": True,
            "hdbscan_total_sec": float(
                hdb_auto_pca_sec + hdb_calibration_sec + hdbscan_final_fit_and_test_sec
            ),
            "fcm_fit_sec": float(fcm_fit_stage.elapsed_sec),
            "fcm_test_assignment_sec": float(fcm_assignment_stage.elapsed_sec),
            "fcm_total_sec": float(fcm_fit_stage.elapsed_sec + fcm_assignment_stage.elapsed_sec),
            "visualization_sec": float(visualization_sec),
            "stages": timing_details,
        }
        run_record = {
            "seed": seed,
            "runtime_sec": float(time.perf_counter() - run_started),
            "timing": timing,
            "hdbscan_knn": {
                "automatic_pca_components": pca_components,
                "automatic_neighbor_count": neighbor_count,
                "selected_configuration": dict(selected),
                "cluster_count": hdb_state.cluster_count,
                "metrics": hdb_metrics,
                "calibration_runs": sweep,
            },
            "hierarchical_fcm": {
                "automatic_pca_components": int(fcm.summary["pca_components"]),
                "automatic_k_by_node": True,
                "fast_mode": False,
                "summary": fcm.summary,
                "metrics": fcm_metrics,
            },
            "artifacts": {"comparison_plot": str(plot_path)},
        }
        (seed_dir / "run.json").write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        runs.append(run_record)

    summary = {
        "benchmark": "wikipedia_hdbscan_pca_knn_vs_hierarchical_fcm",
        "protocol": {
            "discovery": len(discovery),
            "calibration": len(calibration),
            "test": len(test),
            "seeds": list(args.seeds),
            "sequential": True,
            "fast_mode": False,
            "incremental_updates": False,
            "hdbscan_auto_pca": True,
            "hdbscan_auto_neighbor_k": True,
            "fcm_auto_pca": True,
            "fcm_auto_cluster_k_per_node": True,
            "fcm_k_search": "exact_full_data",
            "target_size": args.target_size,
        },
        "aggregate": {
            "hdbscan_knn_leaf_nmi": _mean(runs, ("hdbscan_knn", "metrics", "leaf_nmi")),
            "hdbscan_knn_leaf_ari": _mean(runs, ("hdbscan_knn", "metrics", "leaf_ari")),
            "hdbscan_knn_hierarchy_distance": _mean(runs, ("hdbscan_knn", "metrics", "hierarchy_distance")),
            "fcm_leaf_nmi": _mean(runs, ("hierarchical_fcm", "metrics", "leaf", "nmi")),
            "fcm_leaf_ari": _mean(runs, ("hierarchical_fcm", "metrics", "leaf", "ari")),
            "fcm_hierarchy_distance": _mean(runs, ("hierarchical_fcm", "metrics", "hierarchy_distance")),
            "hdbscan_mean_runtime_sec": _mean(runs, ("timing", "hdbscan_total_sec")),
            "fcm_mean_runtime_sec": _mean(runs, ("timing", "fcm_total_sec")),
            "visualization_mean_runtime_sec": _mean(runs, ("timing", "visualization_sec")),
            "run_mean_runtime_sec": _mean(runs, ("runtime_sec",)),
        },
        "total_runtime_sec": float(time.perf_counter() - started),
        "runs": runs,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output_dir / "runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed", "hdbscan_pca", "hdbscan_neighbor_k", "hdbscan_leaf_nmi",
                "hdbscan_leaf_ari", "hdbscan_hierarchy_distance", "fcm_pca",
                "fcm_leaf_clusters", "fcm_leaf_nmi", "fcm_leaf_ari",
                "fcm_hierarchy_distance", "hdbscan_runtime_sec",
                "fcm_runtime_sec", "visualization_runtime_sec", "runtime_sec",
            ],
        )
        writer.writeheader()
        for row in runs:
            writer.writerow({
                "seed": row["seed"],
                "hdbscan_pca": row["hdbscan_knn"]["automatic_pca_components"],
                "hdbscan_neighbor_k": row["hdbscan_knn"]["automatic_neighbor_count"],
                "hdbscan_leaf_nmi": row["hdbscan_knn"]["metrics"]["leaf_nmi"],
                "hdbscan_leaf_ari": row["hdbscan_knn"]["metrics"]["leaf_ari"],
                "hdbscan_hierarchy_distance": row["hdbscan_knn"]["metrics"]["hierarchy_distance"],
                "fcm_pca": row["hierarchical_fcm"]["automatic_pca_components"],
                "fcm_leaf_clusters": row["hierarchical_fcm"]["metrics"]["cluster_count"],
                "fcm_leaf_nmi": row["hierarchical_fcm"]["metrics"]["leaf"]["nmi"],
                "fcm_leaf_ari": row["hierarchical_fcm"]["metrics"]["leaf"]["ari"],
                "fcm_hierarchy_distance": row["hierarchical_fcm"]["metrics"]["hierarchy_distance"],
                "hdbscan_runtime_sec": row["timing"]["hdbscan_total_sec"],
                "fcm_runtime_sec": row["timing"]["fcm_total_sec"],
                "visualization_runtime_sec": row["timing"]["visualization_sec"],
                "runtime_sec": row["runtime_sec"],
            })
    print(f"Saved report: {report_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-node-size", type=int, default=60)
    parser.add_argument("--min-child-size", type=int, default=20)
    parser.add_argument(
        "--target-size",
        type=int,
        help="Repeat the Wikipedia rows to this total size (60/20/20 splits).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    run(args)


if __name__ == "__main__":
    main()
