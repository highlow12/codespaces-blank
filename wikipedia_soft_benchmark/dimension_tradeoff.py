"""PCA/SVD dimension trade-off benchmark for the Wikipedia BGE hierarchy.

The projection is always fitted on discovery documents.  Geometry diagnostics
compare calibration and test neighbours with the original BGE cosine space;
the HDBSCAN sweep uses calibration labels for selection and evaluates the
selected configuration on test exactly once.  The module is deliberately
separate from the production benchmark so that a dimension sweep cannot
silently change the production defaults.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .hierarchy_benchmark import (
    DEFAULT_MIN_CLUSTER_SIZES,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_NEIGHBOR_COUNTS,
    DEFAULT_SEEDS,
    calibration_sweep,
    evaluate_split,
    fit_discovery,
    load_metadata,
)
from .projection_diagnostics import evaluate_projection, fit_projection


DEFAULT_DIMENSIONS = (32, 64, 128, 256)
DEFAULT_NEIGHBOR_COUNT = 24
SCHEMA_VERSION = 1


def effective_dimension(*, discovery_rows: int, input_dimension: int, requested: int, mode: str) -> int:
    """Return the maximum valid projection rank for a discovery fit.

    Centering removes one possible rank from an ``n``-row discovery matrix,
    so centered PCA has rank at most ``n - 1``.  Uncentered SVD has rank at
    most ``n`` (and both are bounded by the input dimension).
    """

    if discovery_rows < 2:
        raise ValueError("at least two discovery rows are required")
    if input_dimension < 1 or requested < 1:
        raise ValueError("dimensions must be positive")
    if mode not in ("centered-pca", "uncentered-svd"):
        raise ValueError("unknown projection mode")
    rank_cap = discovery_rows - 1 if mode == "centered-pca" else discovery_rows
    return min(int(requested), int(input_dimension), int(rank_cap))


def _summary(result: Mapping[str, Any]) -> dict[str, float]:
    projected = result["projected"]
    return {
        "cosine_pearson": float(projected["pairwise_cosine"]["pearson"]["mean"]),
        "cosine_spearman": float(projected["pairwise_cosine"]["spearman"]["mean"]),
        "cosine_knn_recall_at_24": float(projected["cosine_knn_recall_at_k"]),
        "euclidean_knn_recall_at_24": float(projected["euclidean_knn_recall_at_k"]),
        "leaf_purity_cosine": float(projected["neighbor_leaf_purity_at_k"]["cosine"]["mean"]),
        "leaf_purity_euclidean": float(projected["neighbor_leaf_purity_at_k"]["euclidean"]["mean"]),
        "leaf_cosine_margin": float(projected["same_vs_different_leaf_cosine"]["margin"]),
    }


def _mean_std(rows: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = sorted(rows[0])
    return {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "std": float(np.std([float(row[key]) for row in rows])),
        }
        for key in keys
    }


def _geometry_for_split(
    discovery: np.ndarray,
    discovery_rows: Sequence[Mapping[str, Any]],
    query: np.ndarray,
    query_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    dimension: int,
    seeds: Sequence[int],
    k: int,
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    summaries: list[dict[str, float]] = []
    for seed in seeds:
        fit = fit_projection(discovery, mode=mode, seed=int(seed), n_components=int(dimension))
        projected_query = np.asarray(fit.transformer.transform(query), dtype=np.float64)
        result = evaluate_projection(
            original_query=query,
            original_reference=discovery,
            projected_query=projected_query,
            projected_reference=fit.discovery_features,
            query_metadata=query_rows,
            reference_metadata=discovery_rows,
            k=k,
        )
        per_seed[str(int(seed))] = result
        summaries.append(_summary(result))
    return {"per_seed": per_seed, "summary": _mean_std(summaries)}


def _worker(arguments: tuple[Any, ...]) -> dict[str, Any]:
    (
        mode,
        dimension,
        embeddings,
        metadata,
        seeds,
        min_cluster_sizes,
        min_samples_values,
        neighbor_counts,
        k,
    ) = arguments
    matrix = np.asarray(embeddings, dtype=np.float32)
    rows = [dict(row) for row in metadata]
    split_indices = {
        split: [i for i, row in enumerate(rows) if row.get("split") == split]
        for split in ("discovery", "calibration", "test")
    }
    discovery = matrix[split_indices["discovery"]]
    calibration = matrix[split_indices["calibration"]]
    test = matrix[split_indices["test"]]
    discovery_rows = [rows[i] for i in split_indices["discovery"]]
    calibration_rows = [rows[i] for i in split_indices["calibration"]]
    test_rows = [rows[i] for i in split_indices["test"]]

    # Calibration is the only split used for sweep selection.  No test metric
    # is computed until this selection is complete below.
    sweep, selected = calibration_sweep(
        discovery,
        discovery_rows,
        calibration,
        calibration_rows,
        seeds=seeds,
        min_cluster_sizes=min_cluster_sizes,
        min_samples_values=min_samples_values,
        neighbor_counts=neighbor_counts,
        pca_components=int(dimension),
        projection_mode=str(mode),
        jobs=1,
    )
    state = fit_discovery(
        discovery,
        discovery_rows,
        seed=int(selected["seed"]),
        min_cluster_size=int(selected["min_cluster_size"]),
        min_samples=int(selected["min_samples"]),
        pca_components=int(dimension),
        projection_mode=str(mode),
    )
    effective_neighbors = min(int(selected["neighbor_count"]), len(discovery_rows))
    test_result = evaluate_split(state, test, test_rows, neighbor_count=effective_neighbors, include_labels=False)

    geometry = {
        split: _geometry_for_split(
            discovery,
            discovery_rows,
            calibration if split == "calibration" else test,
            calibration_rows if split == "calibration" else test_rows,
            mode=str(mode),
            dimension=int(dimension),
            seeds=seeds,
            k=k,
        )
        for split in ("calibration", "test")
    }
    return {
        "mode": str(mode),
        "dimension": int(dimension),
        "effective_dimension": int(effective_dimension(discovery_rows=len(discovery), input_dimension=matrix.shape[1], requested=int(dimension), mode=str(mode))),
        "geometry": geometry,
        "selected_configuration": dict(selected),
        "test": {
            "native": {key: value for key, value in test_result["native"].items() if key != "mapped_labels"},
            "exact_knn": {key: value for key, value in test_result["exact_knn"].items() if key != "mapped_labels"},
        },
        "calibration_run_count": len(sweep),
        "calibration": sweep,
    }


def run_dimension_tradeoff(
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    modes: Sequence[str] = ("centered-pca", "uncentered-svd"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
    min_samples_values: Sequence[int] = DEFAULT_MIN_SAMPLES,
    neighbor_counts: Sequence[int] = DEFAULT_NEIGHBOR_COUNTS,
    k: int = DEFAULT_NEIGHBOR_COUNT,
    jobs: int = 1,
) -> dict[str, Any]:
    """Run all mode/dimension points in deterministic order."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    rows = [dict(row) for row in metadata]
    if matrix.ndim != 2 or len(rows) != len(matrix):
        raise ValueError("embeddings and metadata must have equal 2D rows")
    if any(not any(row.get("split") == split for row in rows) for split in ("discovery", "calibration", "test")):
        raise ValueError("discovery, calibration, and test splits are required")
    dimensions_values = tuple(sorted({int(value) for value in dimensions}))
    modes_values = tuple(str(value) for value in modes)
    if any(value < 1 for value in dimensions_values):
        raise ValueError("dimensions must be positive")
    arguments = [
        (mode, dimension, matrix, rows, tuple(int(v) for v in seeds), tuple(int(v) for v in min_cluster_sizes), tuple(int(v) for v in min_samples_values), tuple(int(v) for v in neighbor_counts), int(k))
        for mode in modes_values
        for dimension in dimensions_values
    ]
    if jobs > 1 and len(arguments) > 1:
        with ProcessPoolExecutor(max_workers=int(jobs), mp_context=mp.get_context("fork")) as executor:
            results = list(executor.map(_worker, arguments))
    else:
        results = [_worker(argument) for argument in arguments]
    results.sort(key=lambda result: (result["mode"], int(result["dimension"])))
    return {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "dimensions": list(dimensions_values),
            "modes": list(modes_values),
            "seeds": [int(v) for v in seeds],
            "min_cluster_sizes": [int(v) for v in min_cluster_sizes],
            "min_samples": [int(v) for v in min_samples_values],
            "neighbor_counts": [int(v) for v in neighbor_counts],
            "geometry_neighbor_count": int(k),
            "fit_split": "discovery",
            "selection_split": "calibration",
            "test_evaluation_after_selection": True,
            "projection_input": "L2-normalized BGE document embeddings",
        },
        "dataset": {"documents": len(matrix), **{split: sum(row.get("split") == split for row in rows) for split in ("discovery", "calibration", "test")}},
        "rank_note": "Centered PCA on 432 discovery rows has maximum centered rank 431; 512D is therefore invalid (and a 512D centered curve point is meaningless). The common comparison uses 32, 64, 128, and 256 dimensions.",
        "results": results,
    }


def curve_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten geometry and selected test quality into deterministic CSV rows."""

    output: list[dict[str, Any]] = []
    for result in report["results"]:
        base = {"mode": result["mode"], "dimension": result["dimension"]}
        for split in ("calibration", "test"):
            row = dict(base)
            row["split"] = split
            for metric, values in result["geometry"][split]["summary"].items():
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_std"] = values["std"]
            output.append(row)
        for method in ("native", "exact_knn"):
            row = dict(base)
            row["split"] = "test_cluster_quality"
            row["method"] = method
            for key, value in result["test"][method].items():
                if isinstance(value, (float, int)):
                    row[key] = value
            row["selected_seed"] = result["selected_configuration"]["seed"]
            row["selected_min_cluster_size"] = result["selected_configuration"]["min_cluster_size"]
            row["selected_min_samples"] = result["selected_configuration"]["min_samples"]
            row["selected_neighbor_count"] = result["selected_configuration"]["neighbor_count"]
            output.append(row)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plot(report: Mapping[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    dimensions = sorted({int(result["dimension"]) for result in report["results"]})
    metrics = (
        ("test", "cosine_spearman", "Test cosine Spearman", (0.0, 1.02)),
        ("test", "cosine_knn_recall_at_24", "Test cosine kNN recall@24", (0.0, 1.02)),
        ("test", "leaf_cosine_margin", "Test same–different leaf cosine margin", None),
    )
    for axis, (split, metric, title, ylim) in zip(axes.flat[:3], metrics, strict=True):
        for mode in ("centered-pca", "uncentered-svd"):
            points = [r for r in report["results"] if r["mode"] == mode]
            points.sort(key=lambda r: r["dimension"])
            values = [r["geometry"][split]["summary"][metric]["mean"] for r in points]
            axis.plot([r["dimension"] for r in points], values, marker="o", label=mode)
        axis.set_xscale("log", base=2)
        axis.set_xticks(dimensions)
        axis.set_xticklabels([str(value) for value in dimensions])
        axis.set_title(title)
        axis.set_xlabel("Projection dimensions")
        axis.set_ylabel("Value")
        if ylim:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
    axis = axes.flat[3]
    for mode in ("centered-pca", "uncentered-svd"):
        points = [r for r in report["results"] if r["mode"] == mode]
        points.sort(key=lambda r: r["dimension"])
        values = [r["test"]["exact_knn"]["leaf_nmi"] for r in points]
        axis.plot([r["dimension"] for r in points], values, marker="o", label=f"{mode} exact-kNN")
    axis.set_xscale("log", base=2)
    axis.set_xticks(dimensions)
    axis.set_xticklabels([str(value) for value in dimensions])
    axis.set_title("Test exact-kNN leaf NMI (selected on calibration)")
    axis.set_xlabel("Projection dimensions")
    axis.set_ylabel("Leaf NMI")
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axes.flat[0].legend(loc="best")
    fig.suptitle("Wikipedia BGE projection dimension trade-off", fontsize=14)
    fig.savefig(path, dpi=150, metadata={"Software": "wikipedia_soft_benchmark.dimension_tradeoff"})
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run centered-PCA/uncentered-SVD dimension trade-off benchmark")
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.embedding_dir
    embeddings = np.load(root / "document_embeddings.npy")
    metadata = load_metadata(root / "document_metadata.jsonl")
    report = run_dimension_tradeoff(embeddings, metadata, dimensions=args.dimensions, jobs=args.jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "tradeoff_curve.csv"
    rows = curve_rows(report)
    fields = sorted({key for row in rows for key in row})
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_path = args.output_dir / "tradeoff_curve.png"
    _plot(report, plot_path)
    methodology_path = args.output_dir / "README.md"
    methodology_path.write_text(
        "# Wikipedia BGE projection dimension trade-off\n\n"
        "This benchmark compares centered PCA with uncentered TruncatedSVD on the 720-document BGE embedding set. "
        "Every projection is fit on the 432 discovery documents only. Calibration (144 documents) is used for the "
        "complete 3x3x3x3 HDBSCAN/membership sweep and configuration selection; the 144 test documents are evaluated "
        "only after that selection.\n\n"
        "The common curve uses 32, 64, 128, and 256 dimensions. Centered PCA on 432 rows has maximum centered rank "
        "431, so a 512-dimensional centered point is invalid (and cannot be compared fairly with the uncentered method).\n\n"
        "tradeoff_curve.csv reports, for calibration and test, Pearson/Spearman correlation to original BGE "
        "cross-cosines, cosine and projected-Euclidean kNN recall@24, leaf neighbour purity, and same-leaf minus "
        "different-leaf cosine margin. It also reports native and exact-kNN test cluster quality using the "
        "calibration-selected sweep setting. tradeoff_curve.png plots the test curves and exact-kNN leaf NMI.\n",
        encoding="utf-8",
    )
    report_path = args.output_dir / "report.json"
    report["inputs"] = {
        "document_embeddings": (root / "document_embeddings.npy").name,
        "document_metadata": (root / "document_metadata.jsonl").name,
        "document_embeddings_sha256": _sha256(root / "document_embeddings.npy"),
        "document_metadata_sha256": _sha256(root / "document_metadata.jsonl"),
    }
    report["artifacts"] = {
        "tradeoff_curve_csv": curve_path.name,
        "tradeoff_curve_png": plot_path.name,
        "methodology": methodology_path.name,
        "tradeoff_curve_csv_sha256": _sha256(curve_path),
        "tradeoff_curve_png_sha256": _sha256(plot_path),
        "methodology_sha256": _sha256(methodology_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report["artifacts"]["report_json_sha256"] = _sha256(report_path)
    print(json.dumps(report["artifacts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
