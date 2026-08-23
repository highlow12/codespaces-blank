"""Measure exact and PyNNDescent PCA-neighbor scaling in isolated processes.

The source is the 720-row Wikipedia document embedding set.  Larger points are
made by repeating rows *within their existing discovery/calibration/test
split*; this benchmark therefore measures computational scaling only and does
not make a clustering-quality claim about the repeated data.

Each backend/size pair runs in a fresh child process.  This makes the peak RSS
measurement independent of the previous backend and keeps PyNNDescent's numba
JIT cache from hiding its first-use cost.  The first ANN query is timed as a
separate warm-up, while the reported calibration/test query times exclude it.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pca_neighbor_search import build_pca_neighbor_index
from wikipedia_soft_benchmark.embeddings import l2_normalize


DEFAULT_SAMPLE_SIZES = (100, 178, 316, 562, 1000, 1778, 3162, 5623, 10000)
BACKENDS = ("exact", "pynndescent")


def _set_single_thread_environment() -> None:
    """Pin numerical libraries to one thread for comparable measurements."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"


def sample_sizes(start: int = 100, stop: int = 10000, count: int = 9) -> tuple[int, ...]:
    """Return deterministic, approximately logarithmic inclusive points."""
    if start < 4 or stop < start or count < 2:
        raise ValueError("sample range must satisfy 4 <= start <= stop and count >= 2")
    values = np.geomspace(start, stop, num=count)
    result = tuple(dict.fromkeys(int(round(value)) for value in values))
    if len(result) != count:
        raise ValueError("logarithmic sample points collapsed after rounding")
    return result


def _split_indices(metadata: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    result = {
        split: np.asarray([i for i, row in enumerate(metadata) if row.get("split") == split], dtype=np.int64)
        for split in ("discovery", "calibration", "test")
    }
    if any(len(indices) < 2 for indices in result.values()):
        raise ValueError("metadata must contain discovery, calibration, and test rows")
    return result


def repeat_rows(values: np.ndarray, size: int, *, seed: int = 42) -> np.ndarray:
    """Repeat a split deterministically, preserving its row order."""
    matrix = np.asarray(values)
    if matrix.ndim != 2 or len(matrix) == 0 or size < 1:
        raise ValueError("values must be a non-empty 2D matrix and size must be positive")
    # The seed is part of the public helper contract even though the default
    # prefix repetition is the most reproducible and cache-friendly method.
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    indices = np.arange(size, dtype=np.int64) % len(matrix)
    return np.asarray(matrix[indices], dtype=matrix.dtype)


def _rss_kib() -> int:
    """Current resident set size on Linux, with ru_maxrss fallback."""
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _max_rss_kib(started: int) -> int:
    return max(int(started), _rss_kib(), int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def _prepare_case(embedding_path: Path, metadata_path: Path, total_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.load(embedding_path, mmap_mode="r")
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(embeddings) != len(metadata):
        raise ValueError("embedding and metadata row counts differ")
    indexes = _split_indices(metadata)
    discovery_size = int(round(total_size * 0.60))
    calibration_size = int(round(total_size * 0.20))
    test_size = total_size - discovery_size - calibration_size
    sizes = (discovery_size, calibration_size, test_size)
    arrays = []
    for split, size in zip(("discovery", "calibration", "test"), sizes, strict=True):
        arrays.append(repeat_rows(np.asarray(embeddings[indexes[split]]), size))
    return tuple(arrays)  # type: ignore[return-value]


def _worker(
    connection: Any,
    embedding_path: str,
    metadata_path: str,
    total_size: int,
    backend: str,
    graph_neighbors: int,
    epsilon: float,
) -> None:
    _set_single_thread_environment()
    started_rss = _rss_kib()
    try:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        discovery, calibration, test = _prepare_case(Path(embedding_path), Path(metadata_path), total_size)
        # Keep the comparison in the same PCA space and use one feature width
        # wherever possible.  Tiny discovery sets necessarily cap PCA width.
        from sklearn.decomposition import PCA

        normalized_discovery = l2_normalize(discovery)
        normalized_calibration = l2_normalize(calibration)
        normalized_test = l2_normalize(test)
        pca_width = min(128, normalized_discovery.shape[0] - 1, normalized_discovery.shape[1])
        pca = PCA(n_components=pca_width, random_state=42)
        pca_discovery = np.asarray(pca.fit_transform(normalized_discovery), dtype=np.float64)
        pca_calibration = np.asarray(pca.transform(normalized_calibration), dtype=np.float64)
        pca_test = np.asarray(pca.transform(normalized_test), dtype=np.float64)

        max_k = 24
        build_started = time.perf_counter()
        index = build_pca_neighbor_index(
            pca_discovery,
            backend=backend,
            max_neighbors=max_k,
            graph_neighbors=graph_neighbors,
            random_state=42,
            query_epsilon=epsilon,
        )
        index_build = time.perf_counter() - build_started
        peak_rss = _max_rss_kib(started_rss)

        warmup_started = time.perf_counter()
        if backend == "pynndescent":
            index.query(pca_calibration[:1], max_k)
        warmup = time.perf_counter() - warmup_started
        peak_rss = _max_rss_kib(peak_rss)

        calibration_started = time.perf_counter()
        index.query(pca_calibration, max_k)
        calibration_query = time.perf_counter() - calibration_started
        peak_rss = _max_rss_kib(peak_rss)

        test_started = time.perf_counter()
        index.query(pca_test, max_k)
        test_query = time.perf_counter() - test_started
        peak_rss = _max_rss_kib(peak_rss)

        result = {
            "sample_size": int(total_size),
            "backend": backend,
            "discovery_size": int(len(discovery)),
            "calibration_size": int(len(calibration)),
            "test_size": int(len(test)),
            "pca_components": int(pca_width),
            "index_build_sec": float(index_build),
            "ann_jit_warmup_sec": float(warmup),
            "calibration_query_sec": float(calibration_query),
            "test_query_sec": float(test_query),
            "query_total_sec": float(calibration_query + test_query),
            "neighbor_search_total_sec": float(index_build + calibration_query + test_query),
            "neighbor_search_including_warmup_sec": float(index_build + warmup + calibration_query + test_query),
            "peak_rss_kib": int(peak_rss),
            "baseline_rss_kib": int(started_rss),
            "peak_rss_delta_kib": int(max(0, peak_rss - started_rss)),
        }
        connection.send({"ok": True, "result": result})
    except BaseException as exc:  # communicate worker failures to the parent
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def measure_case(
    embedding_path: Path,
    metadata_path: Path,
    total_size: int,
    backend: str,
    *,
    graph_neighbors: int = 32,
    epsilon: float = 0.1,
    poll_interval: float = 0.01,
) -> dict[str, Any]:
    """Run one isolated case and return its timings and peak RSS."""
    if total_size < 100:
        raise ValueError("total_size must be at least 100")
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(child, str(embedding_path), str(metadata_path), int(total_size), backend, int(graph_neighbors), float(epsilon)),
    )
    process.start()
    child.close()
    peak_parent_rss = 0
    while process.is_alive():
        try:
            status = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8")
            rss = next((int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:")), 0)
            peak_parent_rss = max(peak_parent_rss, rss)
        except (OSError, ValueError, IndexError):
            pass
        time.sleep(poll_interval)
    process.join()
    if not parent.poll():
        raise RuntimeError(f"worker exited without a result (exit code {process.exitcode})")
    try:
        message = parent.recv()
    except (EOFError, OSError) as exc:
        raise RuntimeError(f"worker exited without a result (exit code {process.exitcode})") from exc
    if not message.get("ok"):
        raise RuntimeError(str(message.get("error", "worker failed")))
    result = dict(message["result"])
    result["parent_observed_peak_rss_kib"] = int(peak_parent_rss)
    result["peak_rss_kib"] = max(int(result["peak_rss_kib"]), peak_parent_rss)
    result["peak_rss_delta_kib"] = max(0, result["peak_rss_kib"] - int(result["baseline_rss_kib"]))
    return result


def _plot(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for backend in BACKENDS:
        selected = [row for row in rows if row["backend"] == backend]
        selected = sorted(selected, key=lambda row: int(row["sample_size"]))
        x = [row["sample_size"] for row in selected]
        axes[0].plot(x, [row["neighbor_search_total_sec"] for row in selected], marker="o", label=backend)
        axes[1].plot(x, [row["peak_rss_delta_kib"] / 1024 for row in selected], marker="o", label=backend)
    for axis, ylabel in zip(axes, ("neighbor search time (s)", "peak RSS above worker baseline (MiB)"), strict=True):
        axis.set_xscale("log")
        axis.set_xlabel("total repeated samples (log scale)")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    # The fresh-process ANN compile/build cost and exact brute-force query
    # time differ by orders of magnitude at these sizes.  A logarithmic time
    # axis keeps both curves readable instead of flattening exact kNN at zero.
    axes[0].set_yscale("log")
    fig.suptitle("Exact kNN vs PyNNDescent scaling (Wikipedia split-internal repeats)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    _set_single_thread_environment()
    sizes = tuple(args.sample_sizes) if args.sample_sizes else sample_sizes(args.start, args.stop, args.points)
    if any(size < 100 for size in sizes):
        raise ValueError("sample sizes must be at least 100")
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for backend in BACKENDS:
            rows.append(measure_case(args.embedding_path, args.metadata_path, size, backend, graph_neighbors=args.graph_neighbors, epsilon=args.epsilon))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["sample_size", "backend"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "benchmark": "pca_neighbor_exact_vs_pynndescent_scaling",
        "dataset": {"embedding_path": str(args.embedding_path), "metadata_path": str(args.metadata_path), "source": "original Wikipedia 720 rows", "larger_points": "split-internal repetition", "quality_eligible": False},
        "protocol": {"sample_sizes": list(sizes), "split_fraction": {"discovery": 0.6, "calibration": 0.2, "test": 0.2}, "k": 24, "metric": "euclidean", "graph_neighbors": args.graph_neighbors, "epsilon": args.epsilon, "random_state": 42, "n_jobs": 1, "thread_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}},
        "measurements": rows,
        "artifacts": {"csv": csv_path.name, "plot": "scaling.png"},
        "notes": ["Each backend/sample pair runs in a fresh process.", "ANN JIT warm-up is recorded separately and excluded from neighbor_search_total_sec.", "Peak RSS is sampled by both worker ru_maxrss and parent /proc polling.", "Repeated-data timings are not clustering-quality evidence."],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    _plot(rows, args.output_dir / "scaling.png")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-path", type=Path, default=Path("wikipedia_embeddings/document_embeddings.npy"))
    parser.add_argument("--metadata-path", type=Path, default=Path("wikipedia_embeddings/document_metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/knn-ann-scaling"))
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--stop", type=int, default=10000)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--sample-sizes", type=int, nargs="*")
    parser.add_argument("--graph-neighbors", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    print(json.dumps(run_benchmark(build_parser().parse_args()), indent=2))
