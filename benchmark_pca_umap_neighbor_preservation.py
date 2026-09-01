"""PCA/UMAP neighbourhood-preservation benchmark.

The benchmark keeps the PCA selection and its input fixed while sweeping UMAP
settings.  It is deliberately separate from the clustering pipeline: this
module measures geometry only and does not fit HDBSCAN.

The default input follows the Wikipedia benchmark convention::

    wikipedia_embeddings/document_embeddings.npy

UMAP is imported only when a run is requested, so pure helper functions and
tests do not require a model download (or even an installed UMAP package).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from embedding_data import load_embeddings_from_json
from pca_dimension_search import (
    DEFAULT_K_VALUES as PCA_SELECTION_K_VALUES,
    mean_neighbor_preservation,
    neighbor_indices,
)
from pca_dimension_selection import PcaDimensionSelection, select_pca_dimension_for_data
from wikipedia_soft_benchmark.embeddings import l2_normalize


SCHEMA_VERSION = 1
DEFAULT_INPUT = Path("wikipedia_embeddings/document_embeddings.npy")
DEFAULT_OUTPUT = Path("benchmarks/pca-umap-neighbor-preservation")
DEFAULT_K_VALUES = (5, 10, 15, 30, 50)
DEFAULT_N_NEIGHBORS = (5, 10, 15, 30, 50, 100)
DEFAULT_N_COMPONENTS = (5, 10, 20, 50)
DEFAULT_MIN_DISTS = (0.0, 0.1, 0.25, 0.5)
DEFAULT_UMAP_SEEDS = (42, 43, 44, 45, 46)

# The active production PCA -> UMAP -> HDBSCAN path does not pass min_dist to
# UMAP.  UMAP's constructor default is 0.1.  Keep this explicit in reports so
# the baseline remains identifiable even when the grid override is used.
UMAP_DEFAULT_MIN_DIST = 0.1
PRODUCTION_BASELINE = {
    "n_neighbors": 15,
    "n_components": 20,
    "min_dist": UMAP_DEFAULT_MIN_DIST,
    "init": "random",
    "n_jobs": 1,
}
UMAPClass = Any


@dataclass(frozen=True)
class PreparedExperiment:
    """Fixed work shared by every UMAP run."""

    normalized_embeddings: np.ndarray
    raw_neighbors: np.ndarray
    pca_selection: PcaDimensionSelection
    pca_features: np.ndarray
    pca_neighbors: np.ndarray
    k_values: tuple[int, ...]
    timings_sec: dict[str, float]


def _as_finite_matrix(values: Any, *, name: str = "embeddings") -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a 2D matrix with at least two rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def resolve_input_path(input_path: Path | str | None) -> Path:
    """Resolve a file or a Wikipedia embedding directory to an input file."""

    path = Path(default_input_path() if input_path is None else input_path)
    if path.is_dir():
        path = path / "document_embeddings.npy"
    return path


def default_input_path() -> Path:
    """Return the Wikipedia artifact, or the repository's embedding fallback."""

    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT
    fallback = Path("dbpedia_gemini_embeddings.json.gz")
    if fallback.exists():
        return fallback
    return DEFAULT_INPUT


def load_benchmark_embeddings(
    input_path: Path | str | None = None,
    *,
    sample_size: int | None = None,
    sample_seed: int = 42,
) -> np.ndarray:
    """Load NPY or the repository's JSON embedding format.

    The directory form is accepted because the existing Wikipedia benchmark
    commands take ``--embedding-dir`` and read its document NPY file.
    """

    path = resolve_input_path(default_input_path() if input_path is None else input_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding input does not exist: {path}. Expected the Wikipedia "
            "benchmark artifact at wikipedia_embeddings/document_embeddings.npy, "
            "the repository fallback dbpedia_gemini_embeddings.json.gz, or pass "
            "--input with an explicit .npy/.json/.json.gz file."
        )

    try:
        if path.suffix == ".npy":
            matrix = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            try:
                key = "embeddings" if "embeddings" in archive.files else archive.files[0]
                matrix = archive[key]
            finally:
                archive.close()
        else:
            matrix, _ = load_embeddings_from_json(path)
    except (OSError, ValueError, KeyError, IndexError) as error:
        raise ValueError(f"Could not load embedding matrix from {path}: {error}") from error

    matrix = _as_finite_matrix(matrix)
    if sample_size is not None:
        sample_size = int(sample_size)
        if not 2 <= sample_size <= matrix.shape[0]:
            raise ValueError(
                f"sample_size must be between 2 and {matrix.shape[0]}, got {sample_size}"
            )
        rng = np.random.default_rng(int(sample_seed))
        indices = np.sort(rng.choice(matrix.shape[0], size=sample_size, replace=False))
        matrix = matrix[indices]
    return matrix


def valid_k_values(k_values: Iterable[int], n_samples: int) -> tuple[int, ...]:
    """Return unique, sorted exact-kNN values valid for ``n_samples``."""

    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("at least two samples are required for neighbourhoods")
    values = tuple(sorted({int(value) for value in k_values}))
    if any(value < 1 for value in values):
        raise ValueError("k values must be positive")
    return tuple(value for value in values if value < n_samples)


def _positive_unique(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or any(value < 1 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def effective_umap_grid(
    *,
    n_samples: int,
    pca_width: int,
    n_neighbors: Iterable[int] = DEFAULT_N_NEIGHBORS,
    n_components: Iterable[int] = DEFAULT_N_COMPONENTS,
    min_dists: Iterable[float] = DEFAULT_MIN_DISTS,
) -> tuple[dict[str, Any], ...]:
    """Build a safe grid and record requested versus effective parameters.

    UMAP accepts at most ``n_samples - 1`` useful neighbours.  Requested
    values above that limit are clamped, matching the existing benchmark
    convention.  Duplicate effective values are removed.  Output dimensions
    are bounded by both the selected PCA width and the existing small-dataset
    UMAP convention (at most ``n_samples - 2``).
    """

    n_samples = int(n_samples)
    pca_width = int(pca_width)
    if n_samples < 3:
        raise ValueError("UMAP requires at least three samples")
    if pca_width < 1:
        raise ValueError("pca_width must be positive")

    requested_neighbors = _positive_unique(n_neighbors, name="n_neighbors")
    requested_components = _positive_unique(n_components, name="n_components")
    effective_neighbors: dict[int, int] = {}
    for requested in requested_neighbors:
        effective = min(requested, n_samples - 1)
        if effective >= 2:
            effective_neighbors.setdefault(effective, requested)

    component_cap = min(pca_width, max(1, n_samples - 2))
    effective_components: dict[int, int] = {}
    for requested in requested_components:
        # The protocol excludes dimensions larger than the selected PCA
        # prefix. Silently clamping them would make a requested 50D run a
        # different experiment and could hide that PCA was too narrow.
        if requested <= component_cap:
            effective_components.setdefault(requested, requested)

    min_dist_values = tuple(sorted({float(value) for value in min_dists}))
    if not min_dist_values or any(
        not math.isfinite(value) or value < 0.0 for value in min_dist_values
    ):
        raise ValueError("min_dists must contain finite non-negative numbers")
    if not effective_neighbors:
        raise ValueError("no n_neighbors value is valid for this dataset")
    if not effective_components:
        raise ValueError(
            f"no requested n_components value is valid for PCA width {pca_width} "
            f"and {n_samples} samples"
        )

    configurations: list[dict[str, Any]] = []
    for effective_neighbor, requested_neighbor in effective_neighbors.items():
        for effective_component, requested_component in effective_components.items():
            for min_dist in min_dist_values:
                configurations.append(
                    {
                        "requested_n_neighbors": int(requested_neighbor),
                        "n_neighbors": int(effective_neighbor),
                        "requested_n_components": int(requested_component),
                        "n_components": int(effective_component),
                        "min_dist": float(min_dist),
                    }
                )
    return tuple(configurations)


def neighbors_by_k(
    values: np.ndarray,
    k_values: Sequence[int],
    *,
    metric: str,
) -> dict[int, np.ndarray]:
    """Compute one exact maximum-k search and expose its valid prefixes."""

    valid = valid_k_values(k_values, len(values))
    if not valid:
        return {}
    maximum = max(valid)
    all_neighbors = neighbor_indices(values, maximum, metric=metric)
    return {k: all_neighbors[:, :k] for k in valid}


def preservation_metrics(
    raw_neighbors: Mapping[int, np.ndarray],
    pca_neighbors: Mapping[int, np.ndarray],
    umap_neighbors: Mapping[int, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Calculate all required metrics for one UMAP result."""

    keys = tuple(sorted(set(raw_neighbors) & set(pca_neighbors) & set(umap_neighbors)))
    result: dict[str, dict[str, float]] = {}
    for k in keys:
        raw_pca = mean_neighbor_preservation(raw_neighbors[k], pca_neighbors[k])
        raw_umap = mean_neighbor_preservation(raw_neighbors[k], umap_neighbors[k])
        pca_umap = mean_neighbor_preservation(pca_neighbors[k], umap_neighbors[k])
        result[f"k{k}"] = {
            "raw_pca": float(raw_pca),
            "raw_umap": float(raw_umap),
            "pca_umap": float(pca_umap),
            "umap_additional_loss": float(raw_pca - raw_umap),
        }
    return result


def _load_umap() -> UMAPClass:
    try:
        from umap import UMAP
    except ImportError as error:
        raise RuntimeError(
            "UMAP is required to run this benchmark. Install the repository "
            "requirements or use the pure helper functions/tests."
        ) from error
    return UMAP


def _fit_umap(
    pca_features: np.ndarray,
    configuration: Mapping[str, Any],
    *,
    seed: int,
    umap_class: UMAPClass | None = None,
) -> tuple[Any, np.ndarray]:
    reducer_class = _load_umap() if umap_class is None else umap_class
    reducer = reducer_class(
        n_components=int(configuration["n_components"]),
        n_neighbors=int(configuration["n_neighbors"]),
        min_dist=float(configuration["min_dist"]),
        metric="euclidean",
        init="random",
        random_state=int(seed),
        n_jobs=1,
    )
    fitted = reducer.fit(pca_features)
    coordinates = np.asarray(getattr(fitted, "embedding_", None), dtype=np.float64)
    expected_shape = (pca_features.shape[0], int(configuration["n_components"]))
    if coordinates.shape != expected_shape:
        raise ValueError(
            f"UMAP output has shape {coordinates.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("UMAP output must contain only finite values")
    return fitted, coordinates


def prepare_experiment(
    embeddings: np.ndarray,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    pca_selection_k_values: Sequence[int] = PCA_SELECTION_K_VALUES,
    pca_max_components: int = 512,
    pca_min_components: int = 32,
    pca_component_step: int = 32,
    minimum_preservation_gain: float = 0.05,
    pca_seed: int = 42,
) -> PreparedExperiment:
    """Normalize, compute exact raw kNN, select PCA once, and cache its kNN."""

    matrix = _as_finite_matrix(embeddings)
    valid_k = valid_k_values(k_values, matrix.shape[0])
    if not valid_k:
        raise ValueError("no requested k is valid for this dataset")
    timings: dict[str, float] = {}

    started = time.perf_counter()
    normalized = l2_normalize(matrix)
    timings["l2_normalize_sec"] = float(time.perf_counter() - started)

    started = time.perf_counter()
    raw_neighbor_map = neighbors_by_k(normalized, valid_k, metric="cosine")
    timings["raw_knn_sec"] = float(time.perf_counter() - started)
    raw_neighbors = raw_neighbor_map[max(valid_k)]

    started = time.perf_counter()
    # The selector itself is the single PCA selection call.  It receives the
    # already-normalized matrix; its shared PCA projection is reused below.
    selection = select_pca_dimension_for_data(
        normalized,
        max_components=int(pca_max_components),
        min_components=int(pca_min_components),
        component_step=int(pca_component_step),
        # Keep automatic selection identical to the current production
        # selector (whose defaults are k=15 and k=30). The broader requested
        # k grid is evaluated independently after the selection is fixed.
        k_values=pca_selection_k_values,
        minimum_preservation_gain=float(minimum_preservation_gain),
        seed=int(pca_seed),
    )
    if selection is None:
        raise ValueError("PCA selection returned no result; at least two rows are required")
    timings["pca_selection_sec"] = float(time.perf_counter() - started)

    pca_features = np.asarray(selection.selected_features, dtype=np.float64)
    started = time.perf_counter()
    pca_neighbor_map = neighbors_by_k(pca_features, valid_k, metric="cosine")
    timings["pca_knn_sec"] = float(time.perf_counter() - started)
    pca_neighbors = pca_neighbor_map[max(valid_k)]
    return PreparedExperiment(
        normalized_embeddings=normalized,
        raw_neighbors=raw_neighbors,
        pca_selection=selection,
        pca_features=pca_features,
        pca_neighbors=pca_neighbors,
        k_values=valid_k,
        timings_sec=timings,
    )


def _prefix_neighbors(neighbors: np.ndarray, k_values: Sequence[int]) -> dict[int, np.ndarray]:
    return {int(k): neighbors[:, : int(k)] for k in k_values}


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _std(values: Sequence[float]) -> float | None:
    return None if not values else float(np.std(values))


def _reproducibility_metrics(
    neighbors_a: Mapping[int, np.ndarray],
    neighbors_b: Mapping[int, np.ndarray],
) -> dict[str, float]:
    return {
        f"k{k}": float(mean_neighbor_preservation(neighbors_a[k], neighbors_b[k]))
        for k in sorted(set(neighbors_a) & set(neighbors_b))
    }


def _is_baseline(configuration: Mapping[str, Any]) -> bool:
    return (
        int(configuration["n_neighbors"]) == PRODUCTION_BASELINE["n_neighbors"]
        and int(configuration["n_components"]) == PRODUCTION_BASELINE["n_components"]
        and math.isclose(float(configuration["min_dist"]), PRODUCTION_BASELINE["min_dist"])
    )


def run_experiment(
    embeddings: np.ndarray,
    *,
    n_neighbors: Sequence[int] = DEFAULT_N_NEIGHBORS,
    n_components: Sequence[int] = DEFAULT_N_COMPONENTS,
    min_dists: Sequence[float] = DEFAULT_MIN_DISTS,
    umap_seeds: Sequence[int] = DEFAULT_UMAP_SEEDS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    pca_selection_k_values: Sequence[int] = PCA_SELECTION_K_VALUES,
    pca_max_components: int = 512,
    pca_min_components: int = 32,
    pca_component_step: int = 32,
    minimum_preservation_gain: float = 0.05,
    pca_seed: int = 42,
    input_path: Path | str | None = None,
    umap_class: UMAPClass | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the fixed-PCA UMAP grid and return a JSON-serializable report."""

    seeds = tuple(dict.fromkeys(int(seed) for seed in umap_seeds))
    if not seeds:
        raise ValueError("umap_seeds must contain at least one seed")
    prepared = prepare_experiment(
        embeddings,
        k_values=k_values,
        pca_selection_k_values=pca_selection_k_values,
        pca_max_components=pca_max_components,
        pca_min_components=pca_min_components,
        pca_component_step=pca_component_step,
        minimum_preservation_gain=minimum_preservation_gain,
        pca_seed=pca_seed,
    )
    configurations = effective_umap_grid(
        n_samples=len(prepared.normalized_embeddings),
        pca_width=prepared.pca_features.shape[1],
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dists=min_dists,
    )

    raw_map = _prefix_neighbors(prepared.raw_neighbors, prepared.k_values)
    pca_map = _prefix_neighbors(prepared.pca_neighbors, prepared.k_values)
    runs: list[dict[str, Any]] = []
    reproducibility: list[dict[str, Any]] = []
    run_index = 0
    total_runs = len(configurations) * len(seeds)
    for configuration in configurations:
        prior_neighbors: dict[int, tuple[int, dict[int, np.ndarray]]] = {}
        for seed in seeds:
            if progress:
                print(
                    f"[{run_index + 1}/{total_runs}] "
                    f"n_neighbors={configuration['n_neighbors']} "
                    f"n_components={configuration['n_components']} "
                    f"min_dist={configuration['min_dist']} seed={seed}",
                    flush=True,
                )
            run_started = time.perf_counter()
            fit_started = time.perf_counter()
            _, coordinates = _fit_umap(
                prepared.pca_features,
                configuration,
                seed=seed,
                umap_class=umap_class,
            )
            umap_fit_sec = float(time.perf_counter() - fit_started)
            knn_started = time.perf_counter()
            umap_map = neighbors_by_k(coordinates, prepared.k_values, metric="euclidean")
            umap_neighbors = umap_map[max(prepared.k_values)]
            umap_knn_sec = float(time.perf_counter() - knn_started)
            metrics = preservation_metrics(raw_map, pca_map, umap_map)

            repro_values: dict[str, list[float]] = defaultdict(list)
            for previous_seed, previous_map in prior_neighbors.values():
                comparison = _reproducibility_metrics(previous_map, umap_map)
                comparison_record = {
                    "n_neighbors": int(configuration["n_neighbors"]),
                    "n_components": int(configuration["n_components"]),
                    "min_dist": float(configuration["min_dist"]),
                    "seed_a": int(previous_seed),
                    "seed_b": int(seed),
                    "preservation": comparison,
                }
                reproducibility.append(comparison_record)
                for key, value in comparison.items():
                    repro_values[key].append(value)

            row: dict[str, Any] = {
                "run_index": run_index,
                "seed": int(seed),
                "requested_n_neighbors": int(configuration["requested_n_neighbors"]),
                "n_neighbors": int(configuration["n_neighbors"]),
                "requested_n_components": int(configuration["requested_n_components"]),
                "n_components": int(configuration["n_components"]),
                "min_dist": float(configuration["min_dist"]),
                "is_baseline": _is_baseline(configuration),
                "umap_init": "random",
                "umap_n_jobs": 1,
                "umap_fit_sec": umap_fit_sec,
                "umap_knn_sec": umap_knn_sec,
                "run_total_sec": float(time.perf_counter() - run_started),
            }
            for key, values in metrics.items():
                for name, value in values.items():
                    row[f"{name}_{key}"] = float(value)
            for key, values in repro_values.items():
                row[f"reproducibility_{key}_mean"] = _mean(values)
                row[f"reproducibility_{key}_std"] = _std(values)
            runs.append(row)
            prior_neighbors[seed] = (seed, umap_map)
            run_index += 1

        # The local `prior_neighbors` maps hold only the current configuration;
        # this prevents reproducibility comparisons across different settings.

    summary = summarize_runs(runs, prepared.k_values)
    selection_dict = prepared.pca_selection.to_dict()
    selection_dict["selected_dimension_knn_preservation"] = {
        f"k{k}": float(
            mean_neighbor_preservation(
                raw_map[k],
                pca_map[k],
            )
        )
        for k in prepared.k_values
    }
    candidates = selection_dict.get("candidates", [])
    selected_candidate = next(
        (candidate for candidate in candidates if candidate["dimension"] == prepared.pca_selection.selected_dimension),
        None,
    )
    selection_dict["selected_cumulative_explained_variance"] = (
        None if selected_candidate is None else selected_candidate["cumulative_explained_variance"]
    )

    resolved_input = None if input_path is None else resolve_input_path(input_path)
    is_original_wikipedia = bool(
        resolved_input is not None
        and resolved_input.name == "document_embeddings.npy"
        and resolved_input.parent.name == "wikipedia_embeddings"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "rows": int(len(prepared.normalized_embeddings)),
            "embedding_dimension": int(prepared.normalized_embeddings.shape[1]),
            "input": None if resolved_input is None else str(resolved_input),
            "input_kind": (
                "wikipedia_bge_document_embeddings"
                if is_original_wikipedia
                else "explicit_or_repository_fallback_embeddings"
            ),
            "is_original_wikipedia_bge": is_original_wikipedia,
            "l2_normalized": True,
        },
        "protocol": {
            "k_values": list(prepared.k_values),
            "raw_metric": "cosine",
            "pca_metric": "cosine",
            "umap_metric": "euclidean",
            "knn": "exact sklearn NearestNeighbors algorithm=brute, non-self",
            "pca_seed": int(pca_seed),
            "pca_selection_k_values": [int(k) for k in pca_selection_k_values],
            "umap_seeds": list(seeds),
            "umap_init": "random",
            "umap_n_jobs": 1,
            "pca_fit_once": True,
            "pca_selection_once": True,
            "pca_selection_input": "L2-normalized embeddings",
            "additional_loss_note": (
                "raw_pca - raw_umap; auxiliary comparison, not a strict information-loss decomposition"
            ),
        },
        "production_baseline": {
            **PRODUCTION_BASELINE,
            "min_dist_source": (
                "UMAP constructor default used by the current PCA->UMAP->HDBSCAN path; effective value 0.1"
            ),
            "present_in_grid": any(row["is_baseline"] for row in runs),
        },
        "pca_selection": selection_dict,
        "raw_pca_preservation": {
            f"k{k}": float(mean_neighbor_preservation(raw_map[k], pca_map[k]))
            for k in prepared.k_values
        },
        "preprocessing_timings_sec": prepared.timings_sec,
        "grid": {
            "requested_n_neighbors": [int(value) for value in n_neighbors],
            "requested_n_components": [int(value) for value in n_components],
            "requested_min_dist": [float(value) for value in min_dists],
            "requested_umap_seeds": list(seeds),
            "effective_configuration_count": len(configurations),
        },
        "runs": runs,
        "summary": summary,
        "reproducibility_comparisons": reproducibility,
        "artifacts": {},
    }


def summarize_runs(
    runs: Sequence[Mapping[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> list[dict[str, Any]]:
    """Aggregate UMAP seeds by effective hyperparameter configuration."""

    grouped: dict[tuple[int, int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[
            (
                int(row["n_neighbors"]),
                int(row["n_components"]),
                float(row["min_dist"]),
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for (neighbors, components, min_dist), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "n_neighbors": neighbors,
            "n_components": components,
            "min_dist": min_dist,
            "seed_count": len(rows),
            "is_baseline": _is_baseline(
                {"n_neighbors": neighbors, "n_components": components, "min_dist": min_dist}
            ),
        }
        for k in k_values:
            key = f"k{int(k)}"
            for metric in ("raw_pca", "raw_umap", "pca_umap", "umap_additional_loss"):
                values = [float(row[f"{metric}_{key}"]) for row in rows if f"{metric}_{key}" in row]
                item[f"{metric}_{key}_mean"] = _mean(values)
                item[f"{metric}_{key}_std"] = _std(values)
            repro_values = [
                float(row[f"reproducibility_{key}_mean"])
                for row in rows
                if row.get(f"reproducibility_{key}_mean") is not None
            ]
            item[f"reproducibility_{key}_mean"] = _mean(repro_values)
            item[f"reproducibility_{key}_std"] = _std(repro_values)
        for metric in ("raw_pca", "raw_umap", "pca_umap", "umap_additional_loss"):
            means = [
                float(row[f"{metric}_k{int(k)}"])
                for row in rows
                for k in k_values
                if f"{metric}_k{int(k)}" in row
            ]
            item[f"mean_{metric}"] = _mean(means)
        reproducibility_means = [
            float(row[f"reproducibility_k{int(k)}_mean"])
            for row in rows
            for k in k_values
            if row.get(f"reproducibility_k{int(k)}_mean") is not None
        ]
        item["mean_reproducibility"] = _mean(reproducibility_means)
        summary.append(item)
    return summary


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _baseline_runs(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = [row for row in report["runs"] if row.get("is_baseline")]
    if rows:
        return rows
    # A small sampled dataset may not support 20 UMAP dimensions.  Use the
    # closest available configuration for a useful fallback plot and state it
    # in the plot title.
    rows = list(report["runs"])
    return sorted(
        rows,
        key=lambda row: (
            abs(int(row["n_neighbors"]) - 15),
            abs(int(row["n_components"]) - 20),
            abs(float(row["min_dist"]) - 0.1),
        ),
    )[: max(1, len(report["protocol"]["umap_seeds"]))]


def _group_axis_rows(
    report: Mapping[str, Any],
    axis: str,
) -> dict[Any, list[Mapping[str, Any]]]:
    all_rows = list(report["runs"])
    baseline = [
        row
        for row in all_rows
        if (
            axis == "n_neighbors"
            or int(row["n_neighbors"]) == PRODUCTION_BASELINE["n_neighbors"]
        )
        and (
            axis == "n_components"
            or int(row["n_components"]) == PRODUCTION_BASELINE["n_components"]
        )
        and (
            axis == "min_dist"
            or math.isclose(float(row["min_dist"]), PRODUCTION_BASELINE["min_dist"])
        )
    ]
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline:
        grouped[row[axis]].append(row)
    if len(grouped) < 2:
        grouped = defaultdict(list)
        for row in all_rows:
            grouped[row[axis]].append(row)
    return grouped


def _plot_save(path: Path, title: str, plotter: Callable[[Any], None]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate benchmark plots") from error
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    plotter(axis)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_plots(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    """Generate the five requested PNG plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(k) for k in report["protocol"]["k_values"]]
    reference_k = 15 if 15 in k_values else max(k_values)
    baseline = _baseline_runs(report)

    def baseline_plot(axis: Any) -> None:
        for metric, label in (
            ("raw_pca", "Raw ↔ PCA"),
            ("raw_umap", "Raw ↔ UMAP"),
            ("pca_umap", "PCA ↔ UMAP"),
        ):
            values = [
                float(np.mean([row[f"{metric}_k{k}"] for row in baseline]))
                for k in k_values
            ]
            axis.plot(k_values, values, marker="o", label=label)
        axis.set_xlabel("k")
        axis.set_ylabel("Neighbor preservation")
        axis.set_ylim(0.0, 1.02)
        axis.legend(loc="best")

    def sensitivity_plot(axis: Any, parameter: str, label: str) -> None:
        grouped = _group_axis_rows(report, parameter)
        x_values = sorted(grouped)
        for metric, metric_label in (("raw_umap", "Raw ↔ UMAP"), ("pca_umap", "PCA ↔ UMAP")):
            means: list[float] = []
            stds: list[float] = []
            for x_value in x_values:
                values = [
                    float(row[f"{metric}_k{reference_k}"])
                    for row in grouped[x_value]
                    if f"{metric}_k{reference_k}" in row
                ]
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
            axis.errorbar(x_values, means, yerr=stds, marker="o", capsize=3, label=metric_label)
        axis.set_xlabel(label)
        axis.set_ylabel(f"Mean preservation at k={reference_k}")
        axis.set_ylim(0.0, 1.02)
        axis.legend(loc="best")

    paths = {
        "baseline_k_preservation": output_dir / "baseline-k-preservation.png",
        "n_neighbors_sensitivity": output_dir / "n-neighbors-sensitivity.png",
        "n_components_sensitivity": output_dir / "n-components-sensitivity.png",
        "min_dist_sensitivity": output_dir / "min-dist-sensitivity.png",
        "seed_stability": output_dir / "seed-stability.png",
    }
    _plot_save(paths["baseline_k_preservation"], "Baseline UMAP neighborhood preservation", baseline_plot)
    _plot_save(
        paths["n_neighbors_sensitivity"],
        "UMAP n_neighbors sensitivity",
        lambda axis: sensitivity_plot(axis, "n_neighbors", "UMAP n_neighbors"),
    )
    _plot_save(
        paths["n_components_sensitivity"],
        "UMAP n_components sensitivity",
        lambda axis: sensitivity_plot(axis, "n_components", "UMAP n_components"),
    )
    _plot_save(
        paths["min_dist_sensitivity"],
        "UMAP min_dist sensitivity",
        lambda axis: sensitivity_plot(axis, "min_dist", "UMAP min_dist"),
    )

    def seed_plot(axis: Any) -> None:
        points = [
            row
            for row in report["summary"]
            if row.get(f"reproducibility_k{reference_k}_mean") is not None
        ]
        points.sort(key=lambda row: (int(row["n_neighbors"]), int(row["n_components"]), float(row["min_dist"])))
        x = np.arange(len(points))
        y = [float(row[f"reproducibility_k{reference_k}_mean"]) for row in points]
        error = [float(row[f"reproducibility_k{reference_k}_std"] or 0.0) for row in points]
        axis.errorbar(x, y, yerr=error, fmt="o", markersize=3, capsize=2)
        axis.set_xlabel("Effective configuration index")
        axis.set_ylabel(f"UMAP ↔ UMAP preservation at k={reference_k}")
        axis.set_ylim(0.0, 1.02)

    _plot_save(paths["seed_stability"], "UMAP seed stability", seed_plot)
    return {key: path.name for key, path in paths.items()}


def _format_float(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def write_report_artifacts(report: dict[str, Any], output_dir: Path) -> None:
    """Write JSON, CSV, plots, and the concise Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "runs.csv", report["runs"])
    _write_csv(output_dir / "summary.csv", report["summary"])
    (output_dir / "pca-selection.json").write_text(
        json.dumps(report["pca_selection"], ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plot_names = make_plots(report, output_dir)
    report["artifacts"] = {
        "report_json": "report.json",
        "runs_csv": "runs.csv",
        "summary_csv": "summary.csv",
        "pca_selection_json": "pca-selection.json",
        **plot_names,
        "methodology": "REPORT.md",
    }
    baseline_summary = next((row for row in report["summary"] if row.get("is_baseline")), None)
    top_rows = sorted(
        report["summary"],
        key=lambda row: (row.get("mean_raw_umap") is None, -(row.get("mean_raw_umap") or -1.0)),
    )[:5]
    top_pca_rows = sorted(
        report["summary"],
        key=lambda row: (row.get("mean_pca_umap") is None, -(row.get("mean_pca_umap") or -1.0)),
    )[:1]
    reference_k = 15 if 15 in report["protocol"]["k_values"] else report["protocol"]["k_values"][0]
    lines = [
        "# PCA·UMAP neighborhood-preservation experiment",
        "",
        "This report measures exact kNN overlap through the fixed `raw → PCA → UMAP` path. HDBSCAN is intentionally not fitted in this experiment.",
        "",
        "## Protocol",
        "",
        f"- Dataset: {report['dataset']['rows']} rows × {report['dataset']['embedding_dimension']} dimensions; input L2-normalized once before the fixed PCA selection.",
        f"- Metrics: raw/PCA cosine, UMAP Euclidean, exact non-self kNN; k = {', '.join(map(str, report['protocol']['k_values']))}.",
        f"- PCA selection: one fit/selection with seed {report['protocol']['pca_seed']}, selected dimension {report['pca_selection']['selected_dimension']} ({report['pca_selection']['selection_reason']}).",
        f"- UMAP: `init=random`, `n_jobs=1`, seeds = {report['protocol']['umap_seeds']}.",
        "- Additional loss is `Raw↔PCA - Raw↔UMAP`; it is a diagnostic difference, not a strict decomposition of information loss.",
        "",
        "## Production baseline",
        "",
        "The current PCA→UMAP→HDBSCAN code omits `min_dist`, so the effective UMAP constructor default is 0.1. The baseline is therefore `n_neighbors=15`, `n_components=20`, `min_dist=0.1`.",
        "",
    ]
    if not report["dataset"].get("is_original_wikipedia_bge", False):
        lines.extend(
            [
                "**Data note:** this run did not use the original `wikipedia_embeddings/document_embeddings.npy` artifact. It uses the explicitly supplied or repository fallback embedding input; its results must not be presented as the original 720-row Wikipedia BGE benchmark.",
                "",
            ]
        )
    if baseline_summary is None:
        lines.append("The production baseline was not available in the effective grid (usually because a small sample could not support 20 dimensions).")
    else:
        lines.append("| k | Raw↔PCA mean | Raw↔UMAP mean | PCA↔UMAP mean | additional loss | seed std (Raw↔UMAP) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for k in report["protocol"]["k_values"]:
            lines.append(
                f"| {k} | {_format_float(baseline_summary.get(f'raw_pca_k{k}_mean'))} | {_format_float(baseline_summary.get(f'raw_umap_k{k}_mean'))} | {_format_float(baseline_summary.get(f'pca_umap_k{k}_mean'))} | {_format_float(baseline_summary.get(f'umap_additional_loss_k{k}_mean'))} | {_format_float(baseline_summary.get(f'raw_umap_k{k}_std'))} |"
            )
    lines.extend(["", "## Highest Raw↔UMAP configurations", "", "| n_neighbors | n_components | min_dist | mean Raw↔UMAP |", "|---:|---:|---:|---:|"])
    for row in top_rows:
        lines.append(
            f"| {row['n_neighbors']} | {row['n_components']} | {row['min_dist']} | {_format_float(row.get('mean_raw_umap'))} |"
        )
    lines.extend(["", "## Interpretation and decision notes", ""])
    if baseline_summary is not None:
        lines.append(
            f"- Fixed PCA baseline: selected {report['pca_selection']['selected_dimension']}D; mean Raw↔PCA across k = {_format_float(baseline_summary.get('mean_raw_pca'))} (k={reference_k}: {_format_float(baseline_summary.get(f'raw_pca_k{reference_k}_mean'))})."
        )
        lines.append(
            f"- Production UMAP baseline at k={reference_k}: Raw↔UMAP = {_format_float(baseline_summary.get(f'raw_umap_k{reference_k}_mean'))}, PCA↔UMAP = {_format_float(baseline_summary.get(f'pca_umap_k{reference_k}_mean'))}, Raw↔UMAP seed std = {_format_float(baseline_summary.get(f'raw_umap_k{reference_k}_std'))}."
        )
        lines.append(
            f"- Production baseline seed reproducibility at k={reference_k}: UMAP↔UMAP = {_format_float(baseline_summary.get(f'reproducibility_k{reference_k}_mean'))} (std {_format_float(baseline_summary.get(f'reproducibility_k{reference_k}_std'))})."
        )
    if top_rows:
        best = top_rows[0]
        lines.append(
            f"- Highest mean Raw↔UMAP in the sweep: n_neighbors={best['n_neighbors']}, n_components={best['n_components']}, min_dist={best['min_dist']} (mean across k = {_format_float(best.get('mean_raw_umap'))})."
        )
    if top_pca_rows:
        best_pca = top_pca_rows[0]
        lines.append(
            f"- Highest mean PCA↔UMAP in the sweep: n_neighbors={best_pca['n_neighbors']}, n_components={best_pca['n_components']}, min_dist={best_pca['min_dist']} (mean across k = {_format_float(best_pca.get('mean_pca_umap'))})."
        )
    lines.append(
        "- These are relative geometry results only; HDBSCAN quality and the original Wikipedia BGE benchmark must be evaluated separately before changing production settings."
    )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `report.json`: complete raw run and reproducibility records.",
            "- `runs.csv`: one row per effective UMAP configuration and seed.",
            "- `summary.csv`: mean/std aggregation across UMAP seeds.",
            "- `pca-selection.json`: PCA candidates, selection reason, explained variance, and selected Raw↔PCA preservation.",
            "- `baseline-k-preservation.png`, `n-neighbors-sensitivity.png`, `n-components-sensitivity.png`, `min-dist-sensitivity.png`, `seed-stability.png`: requested plots.",
            "",
            "![Baseline k preservation](baseline-k-preservation.png)",
            "",
            "![n_neighbors sensitivity](n-neighbors-sensitivity.png)",
            "",
            "![n_components sensitivity](n-components-sensitivity.png)",
            "",
            "![min_dist sensitivity](min-dist-sensitivity.png)",
            "",
            "![Seed stability](seed-stability.png)",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_ints(values: Sequence[str]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        result.extend(int(part) for part in value.split(",") if part.strip())
    return tuple(result)


def _parse_floats(values: Sequence[str]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        result.extend(float(part) for part in value.split(",") if part.strip())
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure exact kNN preservation through the fixed PCA -> UMAP path"
    )
    parser.add_argument(
        "--input",
        "--input-json",
        dest="input",
        type=Path,
        default=None,
        help=(
            ".npy, .npz, .json, or .json.gz embedding input; a Wikipedia "
            "embedding directory is also accepted. Defaults to the available "
            "Wikipedia artifact, then the repository Gemini fallback."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-sample-size", type=int, default=None)
    parser.add_argument("--dataset-sample-seed", type=int, default=42)
    parser.add_argument("--n-neighbors", nargs="+", default=None, metavar="N")
    parser.add_argument("--n-components", nargs="+", default=None, metavar="D")
    parser.add_argument("--min-dist", nargs="+", default=None, metavar="DIST")
    parser.add_argument("--umap-seeds", nargs="+", default=None, metavar="SEED")
    parser.add_argument("--k", nargs="+", default=None, metavar="K")
    parser.add_argument("--pca-max-components", type=int, default=512)
    parser.add_argument("--pca-min-components", type=int, default=32)
    parser.add_argument("--pca-component-step", type=int, default=32)
    parser.add_argument("--minimum-preservation-gain", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true", help="small deterministic grid and sample for a smoke run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = resolve_input_path(args.input)
    default_fallback_sample = (
        args.input is None
        and input_path.name == "dbpedia_gemini_embeddings.json.gz"
    )
    raw_embeddings = load_benchmark_embeddings(
        input_path,
        sample_size=(
            720
            if args.dataset_sample_size is None and default_fallback_sample
            else args.dataset_sample_size
        ),
        sample_seed=args.dataset_sample_seed,
    )
    if args.quick:
        sample_size = args.dataset_sample_size
        if sample_size is None:
            sample_size = min(64, len(raw_embeddings))
            if sample_size < 2:
                raise ValueError("quick mode needs at least two input rows")
            raw_embeddings = load_benchmark_embeddings(
                input_path,
                sample_size=sample_size,
                sample_seed=args.dataset_sample_seed,
            )
        n_neighbors = (5, 10)
        n_components = (2, 5)
        min_dists = (0.1,)
        umap_seeds = (42,)
    else:
        n_neighbors = DEFAULT_N_NEIGHBORS
        n_components = DEFAULT_N_COMPONENTS
        min_dists = DEFAULT_MIN_DISTS
        umap_seeds = DEFAULT_UMAP_SEEDS
    if args.n_neighbors is not None:
        n_neighbors = _parse_ints(args.n_neighbors)
    if args.n_components is not None:
        n_components = _parse_ints(args.n_components)
    if args.min_dist is not None:
        min_dists = _parse_floats(args.min_dist)
    if args.umap_seeds is not None:
        umap_seeds = _parse_ints(args.umap_seeds)
    k_values = DEFAULT_K_VALUES if args.k is None else _parse_ints(args.k)

    report = run_experiment(
        raw_embeddings,
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dists=min_dists,
        umap_seeds=umap_seeds,
        k_values=k_values,
        pca_max_components=args.pca_max_components,
        pca_min_components=args.pca_min_components,
        pca_component_step=args.pca_component_step,
        minimum_preservation_gain=args.minimum_preservation_gain,
        pca_seed=42,
        input_path=input_path,
        progress=True,
    )
    write_report_artifacts(report, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "runs": len(report["runs"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
