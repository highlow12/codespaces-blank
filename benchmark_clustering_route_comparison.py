"""Compare the three offline UMAP/HDBSCAN clustering routes.

The benchmark is intentionally separate from the production clustering code.
It keeps one automatic PCA selection fixed for each sampled dataset and then
shares the seed-42 discovery fit between the native and guarded routes.  The
five-seed route reuses that same fit and adds only four more UMAP/HDBSCAN
fits.  This makes the cost of the stability route explicit instead of hiding
duplicate work behind separate route implementations.

The three routes are:

``umap_hdbscan_native``
    One baseline UMAP -> HDBSCAN fit with native HDBSCAN memberships.
``guarded_pca_hybrid``
    The same seed-42 fit, with a pre-registered confidence guard and PCA
    exact-kNN memberships.
``five_seed_stable``
    Five baseline fits, Hungarian alignment to an ARI medoid, a consensus
    guard, and PCA exact-kNN memberships.

This is an offline research benchmark.  The existing hierarchical PCA SFCM
production path remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import hdbscan
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from benchmark_pca_umap_neighbor_preservation import (
    DEFAULT_K_VALUES,
    mean_neighbor_preservation,
    neighbors_by_k,
)
from embedding_data import load_embeddings_from_json
from pca_dimension_selection import (
    DEFAULT_COMPONENT_STEP,
    DEFAULT_MAX_COMPONENTS,
    DEFAULT_MIN_COMPONENTS,
    PcaDimensionSelection,
    select_pca_dimension_for_data,
)
from wikipedia_soft_benchmark.embeddings import l2_normalize


SCHEMA_VERSION = 1
EXPERIMENT_NAME = "clustering_route_comparison"
DEFAULT_INPUT = Path("dbpedia_gemini_embeddings.json.gz")
DEFAULT_OUTPUT = Path("benchmarks/clustering-route-comparison")
DEFAULT_DATASET_SAMPLE_SIZE = 720
DEFAULT_DATASET_SAMPLE_SEED = 42
DEFAULT_SCALE_SAMPLE_SIZES = (1500, 3000)
DEFAULT_UMAP_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_K_VALUES = (5, 10, 15, 30, 50)
DEFAULT_PCA_SELECTION_K_VALUES = (15, 30)
DEFAULT_PCA_NEIGHBOR_COUNT = 15
DEFAULT_GUARD_THRESHOLD = 0.45
DEFAULT_CONSENSUS_THRESHOLD = 0.60
DEFAULT_GUARD_CURVE_THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
DEFAULT_UMAP_COMPONENTS = 20
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3

BASELINE_UMAP_CONFIGURATION: dict[str, Any] = {
    "n_neighbors": DEFAULT_UMAP_N_NEIGHBORS,
    "n_components": DEFAULT_UMAP_COMPONENTS,
    "min_dist": DEFAULT_UMAP_MIN_DIST,
    "metric": "euclidean",
    "init": "random",
    "n_jobs": 1,
}
HDBSCAN_CONFIGURATION: dict[str, Any] = {
    "min_cluster_size": DEFAULT_MIN_CLUSTER_SIZE,
    "min_samples": DEFAULT_MIN_SAMPLES,
    "metric": "euclidean",
    "cluster_selection_method": "leaf",
    "prediction_data": True,
}
GUARD_WEIGHTS: dict[str, float] = {
    "probability": 0.35,
    "one_minus_outlier": 0.20,
    "cluster_persistence": 0.20,
    "pca_local_support": 0.25,
}

ROUTE_NATIVE = "umap_hdbscan_native"
ROUTE_GUARDED = "guarded_pca_hybrid"
ROUTE_STABLE = "five_seed_stable"
ROUTE_NAMES = (ROUTE_NATIVE, ROUTE_GUARDED, ROUTE_STABLE)

UMAPClass = Any
HDBSCANClass = Any
MembershipFunction = Callable[[Any], np.ndarray]


@dataclass(frozen=True)
class PreparedDataset:
    """Work shared by all routes for one sampled dataset."""

    normalized_embeddings: np.ndarray
    raw_neighbors: dict[int, np.ndarray]
    pca_features: np.ndarray
    pca_neighbors: dict[int, np.ndarray]
    pca_selection: PcaDimensionSelection
    k_values: tuple[int, ...]
    source_indices: np.ndarray
    metadata: pd.DataFrame
    timings_sec: dict[str, float]


@dataclass(frozen=True)
class DiscoveryRun:
    """One baseline UMAP -> HDBSCAN fit and its compact diagnostics."""

    seed: int
    coordinates: np.ndarray
    umap_neighbors: dict[int, np.ndarray]
    labels: np.ndarray
    probabilities: np.ndarray
    outlier_scores: np.ndarray
    native_memberships: np.ndarray
    cluster_persistence: np.ndarray
    timings_sec: dict[str, float]


@dataclass(frozen=True)
class RouteOutput:
    """Final hard and soft outputs for one route."""

    name: str
    labels: np.ndarray
    memberships: np.ndarray
    coordinates: np.ndarray
    base_labels: np.ndarray
    probabilities: np.ndarray
    outlier_scores: np.ndarray
    persistence: np.ndarray
    pca_support: np.ndarray
    guard_scores: np.ndarray
    consensus_agreement: np.ndarray | None
    metrics: dict[str, Any]
    runtime_sec: dict[str, float]


def _json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats for strict JSON."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot write empty CSV without fieldnames: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames) if fieldnames is not None else sorted(
        {str(key) for row in rows for key in row}
    )
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


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def _std(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.std(finite))


def _as_finite_matrix(values: Any, *, name: str = "embeddings") -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a 2D matrix with at least two rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _valid_k_values(k_values: Iterable[int], n_samples: int) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in k_values}))
    if not values or any(value < 1 for value in values):
        raise ValueError("k_values must contain positive integers")
    return tuple(value for value in values if value < int(n_samples))


def _canonicalize_labels(labels: Any, n_samples: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (n_samples,):
        raise ValueError(
            f"labels must have shape ({n_samples},), got {values.shape}"
        )
    try:
        values = values.astype(np.int64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("labels must contain integers") from error
    if np.any(values < -1):
        raise ValueError("labels may only contain -1 or non-negative values")
    output = np.full(n_samples, -1, dtype=np.int64)
    clusters = sorted(int(value) for value in np.unique(values) if value >= 0)
    for new_label, old_label in enumerate(clusters):
        output[values == old_label] = new_label
    return output


def _validate_probability_vector(
    values: Any,
    n_samples: int,
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (n_samples,):
        raise ValueError(f"{name} must have shape ({n_samples},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return np.clip(result, 0.0, 1.0)


def _normalize_native_memberships(
    values: Any,
    *,
    n_samples: int,
    cluster_count: int,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if cluster_count == 0:
        if raw.ndim == 1 and raw.shape == (n_samples,) and np.allclose(raw, 0.0):
            return np.zeros((n_samples, 0), dtype=np.float64)
        if raw.shape == (n_samples, 0):
            return np.zeros((n_samples, 0), dtype=np.float64)
        raise ValueError("native memberships have an invalid no-cluster shape")
    if raw.shape != (n_samples, cluster_count):
        raise ValueError(
            "native memberships must have shape "
            f"({n_samples}, {cluster_count}), got {raw.shape}"
        )
    if not np.all(np.isfinite(raw)) or np.any(raw < -1e-12):
        raise ValueError("native memberships must be finite and non-negative")
    return np.clip(raw, 0.0, 1.0)


def load_gemini_dataset(
    input_path: Path,
    *,
    sample_size: int | None = DEFAULT_DATASET_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_DATASET_SAMPLE_SEED,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load the Gemini records and return an aligned deterministic sample."""

    if not input_path.exists():
        raise FileNotFoundError(f"Gemini embedding dataset does not exist: {input_path}")
    embeddings, metadata = load_embeddings_from_json(input_path)
    if "class" not in metadata.columns or "class_hierarchy" not in metadata.columns:
        raise ValueError(
            "Gemini clustering data must contain class and class_hierarchy metadata"
        )
    if sample_size is None:
        indices = np.arange(len(embeddings), dtype=np.int64)
    else:
        sample_size = int(sample_size)
        if not 2 <= sample_size <= len(embeddings):
            raise ValueError(
                f"sample_size must be between 2 and {len(embeddings)}, got {sample_size}"
            )
        rng = np.random.default_rng(int(sample_seed))
        indices = np.sort(
            rng.choice(len(embeddings), size=sample_size, replace=False)
        ).astype(np.int64)
    selected_embeddings = np.asarray(embeddings[indices], dtype=np.float32)
    selected_metadata = metadata.iloc[indices].reset_index(drop=True).copy()
    return selected_embeddings, selected_metadata, indices


def _metadata_hierarchies(metadata: pd.DataFrame) -> list[tuple[str, ...]]:
    if "class_hierarchy" not in metadata.columns:
        raise ValueError("metadata must contain class_hierarchy")
    result: list[tuple[str, ...]] = []
    for value in metadata["class_hierarchy"]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("every class_hierarchy value must contain at least two levels")
        path = tuple(str(item) for item in value)
        if any(not item for item in path):
            raise ValueError("class_hierarchy values must not contain empty labels")
        result.append(path)
    return result


def _dataset_fingerprint(
    normalized_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    source_indices: np.ndarray,
) -> dict[str, Any]:
    metadata_bytes = json.dumps(
        _json_safe(metadata.to_dict(orient="records")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    values = np.ascontiguousarray(normalized_embeddings, dtype=np.float32)
    return {
        "rows": int(values.shape[0]),
        "embedding_dimension": int(values.shape[1]),
        "sample_indices_sha256": hashlib.sha256(
            np.asarray(source_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "normalized_embeddings_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }


def _raw_selected_pca_prefix(selection: PcaDimensionSelection) -> np.ndarray:
    """Return the unnormalized PCA prefix used by the discovery path."""

    projected = selection.pca.transform(selection.normalized_input)
    features = np.asarray(projected[:, : selection.selected_dimension], dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("selected PCA features must be finite")
    if features.shape[1] < 1:
        raise ValueError("selected PCA prefix must contain at least one dimension")
    return features


def prepare_dataset(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    source_indices: np.ndarray | None = None,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    pca_selection_k_values: Sequence[int] = DEFAULT_PCA_SELECTION_K_VALUES,
    pca_max_components: int = DEFAULT_MAX_COMPONENTS,
    pca_min_components: int = DEFAULT_MIN_COMPONENTS,
    pca_component_step: int = DEFAULT_COMPONENT_STEP,
    minimum_preservation_gain: float = 0.05,
    pca_seed: int = 42,
) -> PreparedDataset:
    """Normalize, select PCA once, and cache every shared neighbor graph."""

    matrix = _as_finite_matrix(embeddings)
    if len(metadata) != len(matrix):
        raise ValueError("metadata must contain one row per embedding")
    valid_k = _valid_k_values(k_values, len(matrix))
    if not valid_k:
        raise ValueError("no requested k is valid for this dataset")
    indices = (
        np.arange(len(matrix), dtype=np.int64)
        if source_indices is None
        else np.asarray(source_indices, dtype=np.int64)
    )
    if indices.shape != (len(matrix),):
        raise ValueError("source_indices must align with embeddings")

    timings: dict[str, float] = {}
    started_total = time.perf_counter()

    started = time.perf_counter()
    normalized_embeddings = l2_normalize(matrix)
    timings["l2_normalize_sec"] = float(time.perf_counter() - started)

    started = time.perf_counter()
    raw_neighbors = neighbors_by_k(normalized_embeddings, valid_k, metric="cosine")
    timings["raw_knn_sec"] = float(time.perf_counter() - started)

    started = time.perf_counter()
    selection = select_pca_dimension_for_data(
        normalized_embeddings,
        max_components=int(pca_max_components),
        min_components=int(pca_min_components),
        component_step=int(pca_component_step),
        k_values=tuple(int(k) for k in pca_selection_k_values if int(k) < len(matrix)),
        minimum_preservation_gain=float(minimum_preservation_gain),
        seed=int(pca_seed),
    )
    if selection is None:
        raise ValueError("automatic PCA selection requires at least two samples")
    timings["pca_selection_sec"] = float(time.perf_counter() - started)

    started = time.perf_counter()
    pca_features = _raw_selected_pca_prefix(selection)
    timings["pca_projection_sec"] = float(time.perf_counter() - started)

    started = time.perf_counter()
    pca_neighbors = neighbors_by_k(pca_features, valid_k, metric="cosine")
    timings["pca_knn_sec"] = float(time.perf_counter() - started)
    timings["total"] = float(time.perf_counter() - started_total)
    return PreparedDataset(
        normalized_embeddings=normalized_embeddings,
        raw_neighbors=raw_neighbors,
        pca_features=pca_features,
        pca_neighbors=pca_neighbors,
        pca_selection=selection,
        k_values=valid_k,
        source_indices=indices,
        metadata=metadata.reset_index(drop=True).copy(),
        timings_sec=timings,
    )


def _load_umap() -> UMAPClass:
    try:
        from umap import UMAP
    except ImportError as error:
        raise RuntimeError("umap-learn is required for this benchmark") from error
    return UMAP


def _fit_umap_hdbscan(
    pca_features: np.ndarray,
    configuration: Mapping[str, Any],
    *,
    seed: int,
    hdbscan_configuration: Mapping[str, Any] = HDBSCAN_CONFIGURATION,
    umap_class: UMAPClass | None = None,
    hdbscan_class: HDBSCANClass | None = None,
    native_membership_function: MembershipFunction | None = None,
) -> DiscoveryRun:
    """Fit exactly one UMAP/HDBSCAN discovery run."""

    values = _as_finite_matrix(pca_features, name="pca_features")
    n_samples = len(values)
    reducer_class = _load_umap() if umap_class is None else umap_class
    effective_neighbors = min(int(configuration["n_neighbors"]), n_samples - 1)
    if effective_neighbors < 2:
        raise ValueError("UMAP requires at least two neighbors")

    started_total = time.perf_counter()
    started = time.perf_counter()
    reducer = reducer_class(
        n_components=int(configuration["n_components"]),
        n_neighbors=effective_neighbors,
        min_dist=float(configuration["min_dist"]),
        metric="euclidean",
        init="random",
        random_state=int(seed),
        n_jobs=1,
    )
    fitted_reducer = reducer.fit(values)
    coordinates = getattr(fitted_reducer, "embedding_", None)
    if coordinates is None:
        coordinates = getattr(reducer, "embedding_", None)
    if coordinates is None and hasattr(reducer, "fit_transform"):
        coordinates = reducer.fit_transform(values)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    expected_shape = (n_samples, int(configuration["n_components"]))
    if coordinates.shape != expected_shape:
        raise ValueError(
            f"UMAP output has shape {coordinates.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("UMAP output must contain only finite values")
    umap_sec = float(time.perf_counter() - started)

    started = time.perf_counter()
    clusterer_class = hdbscan.HDBSCAN if hdbscan_class is None else hdbscan_class
    clusterer = clusterer_class(**dict(hdbscan_configuration)).fit(coordinates)
    labels = _canonicalize_labels(getattr(clusterer, "labels_"), n_samples)
    probabilities = _validate_probability_vector(
        getattr(clusterer, "probabilities_"), n_samples, name="probabilities"
    )
    outlier_scores = _validate_probability_vector(
        getattr(clusterer, "outlier_scores_"), n_samples, name="outlier_scores"
    )
    hdbscan_sec = float(time.perf_counter() - started)

    started = time.perf_counter()
    cluster_count = int(np.unique(labels[labels >= 0]).size)
    if native_membership_function is None:
        raw_memberships = hdbscan.all_points_membership_vectors(clusterer)
    else:
        raw_memberships = native_membership_function(clusterer)
    native_memberships = _normalize_native_memberships(
        raw_memberships,
        n_samples=n_samples,
        cluster_count=cluster_count,
    )
    persistence = np.asarray(
        getattr(clusterer, "cluster_persistence_", np.zeros(cluster_count)),
        dtype=np.float64,
    ).reshape(-1)
    if persistence.size == 0 and cluster_count == 0:
        persistence = np.zeros(0, dtype=np.float64)
    if persistence.shape != (cluster_count,):
        raise ValueError(
            "cluster_persistence_ must align with discovered clusters; "
            f"got {persistence.shape} for {cluster_count} clusters"
        )
    if not np.all(np.isfinite(persistence)):
        raise ValueError("cluster persistence must be finite")
    persistence = np.clip(persistence, 0.0, 1.0)
    membership_sec = float(time.perf_counter() - started)
    return DiscoveryRun(
        seed=int(seed),
        coordinates=coordinates,
        umap_neighbors={},
        labels=labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
        native_memberships=native_memberships,
        cluster_persistence=persistence,
        timings_sec={
            "umap_sec": umap_sec,
            "hdbscan_sec": hdbscan_sec,
            "native_membership_sec": membership_sec,
            "total": float(time.perf_counter() - started_total),
        },
    )


def pca_local_cluster_support(
    pca_neighbors: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return each point's local support for its discovery label."""

    neighbors = np.asarray(pca_neighbors, dtype=np.int64)
    values = np.asarray(labels, dtype=np.int64)
    if neighbors.ndim != 2 or values.shape != (len(neighbors),):
        raise ValueError("pca_neighbors and labels have incompatible shapes")
    support = np.zeros(len(values), dtype=np.float64)
    for row_index, row in enumerate(neighbors):
        neighbor_labels = values[row]
        label = int(values[row_index])
        if label >= 0:
            support[row_index] = float(np.mean(neighbor_labels == label))
        else:
            non_noise = neighbor_labels[neighbor_labels >= 0]
            if non_noise.size:
                counts = np.bincount(non_noise)
                support[row_index] = float(np.max(counts) / len(neighbor_labels))
    return np.clip(support, 0.0, 1.0)


def compute_guard_scores(
    labels: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
    cluster_persistence: np.ndarray,
    pca_support: np.ndarray,
    *,
    weights: Mapping[str, float] = GUARD_WEIGHTS,
) -> np.ndarray:
    """Compute the pre-registered arithmetic guard score.

    The score is intentionally label-free with respect to the DBpedia target:
    it uses only HDBSCAN diagnostics and the PCA neighborhood graph.  Noise
    points receive score zero and are always rejected by a positive threshold.
    """

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = _validate_probability_vector(
        probabilities, len(labels), name="probabilities"
    )
    outlier_scores = _validate_probability_vector(
        outlier_scores, len(labels), name="outlier_scores"
    )
    pca_support = _validate_probability_vector(
        pca_support, len(labels), name="pca_support"
    )
    persistence = np.asarray(cluster_persistence, dtype=np.float64)
    if persistence.ndim != 1 or not np.all(np.isfinite(persistence)):
        raise ValueError("cluster_persistence must be a finite vector")
    if any(float(weights.get(key, 0.0)) < 0.0 for key in GUARD_WEIGHTS):
        raise ValueError("guard weights must be non-negative")
    weight_total = float(sum(float(weights.get(key, 0.0)) for key in GUARD_WEIGHTS))
    if not math.isclose(weight_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("guard weights must sum to one")
    cluster_values = np.zeros(len(labels), dtype=np.float64)
    valid = (labels >= 0) & (labels < len(persistence))
    cluster_values[valid] = np.clip(persistence[labels[valid]], 0.0, 1.0)
    score = (
        float(weights["probability"]) * probabilities
        + float(weights["one_minus_outlier"]) * (1.0 - outlier_scores)
        + float(weights["cluster_persistence"]) * cluster_values
        + float(weights["pca_local_support"]) * pca_support
    )
    score[labels < 0] = 0.0
    return np.clip(score, 0.0, 1.0)


def gate_labels(
    labels: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Set points below a fixed confidence threshold to HDBSCAN noise."""

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    values = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(score, dtype=np.float64)
    if values.shape != scores.shape:
        raise ValueError("labels and score must have the same shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("score must be finite")
    gated = values.copy()
    gated[scores < threshold] = -1
    return _canonicalize_labels(gated, len(gated))


def pca_exact_knn_memberships(
    pca_features: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    *,
    neighbor_indices_by_k: np.ndarray,
) -> dict[str, Any]:
    """Build soft memberships from one cached exact PCA kNN graph."""

    features = _as_finite_matrix(pca_features, name="pca_features")
    labels = _canonicalize_labels(labels, len(features))
    confidence = _validate_probability_vector(
        confidence, len(features), name="confidence"
    )
    neighbor_indices = np.asarray(neighbor_indices_by_k, dtype=np.int64)
    if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != len(features):
        raise ValueError("neighbor_indices_by_k must align with pca_features")
    cluster_count = int(np.unique(labels[labels >= 0]).size)
    affinities = np.zeros((len(features), cluster_count), dtype=np.float64)
    if cluster_count == 0:
        return {
            "affinities": affinities,
            "unexplained": np.ones(len(features), dtype=np.float64),
            "max_affinity": np.zeros(len(features), dtype=np.float64),
            "recommended_labels": np.full(len(features), -1, dtype=np.int64),
            "effective_k": int(neighbor_indices.shape[1]),
        }

    normalized_features = normalize(features, norm="l2")
    for row_index, row in enumerate(neighbor_indices):
        similarities = np.clip(normalized_features[row] @ normalized_features[row_index], -1.0, 1.0)
        distances = np.maximum(0.0, 1.0 - similarities)
        positive = distances[distances > 0.0]
        sigma = float(np.median(positive)) if positive.size else 1.0
        if not math.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        weights = np.exp(-np.square(distances / sigma))
        denominator = float(np.sum(weights))
        if denominator <= 0.0 or not math.isfinite(denominator):
            weights = np.ones(len(row), dtype=np.float64)
            denominator = float(len(row))
        votes = weights * confidence[row]
        for cluster in range(cluster_count):
            affinities[row_index, cluster] = float(
                np.sum(votes[labels[row] == cluster]) / denominator
            )
    affinities = np.clip(affinities, 0.0, 1.0)
    row_sums = np.sum(affinities, axis=1)
    max_affinity = (
        np.max(affinities, axis=1)
        if cluster_count
        else np.zeros(len(features), dtype=np.float64)
    )
    recommended = (
        np.argmax(affinities, axis=1).astype(np.int64)
        if cluster_count
        else np.full(len(features), -1, dtype=np.int64)
    )
    recommended[max_affinity <= 0.0] = -1
    return {
        "affinities": affinities,
        "unexplained": np.clip(1.0 - row_sums, 0.0, 1.0),
        "max_affinity": max_affinity,
        "recommended_labels": recommended,
        "effective_k": int(neighbor_indices.shape[1]),
    }


def choose_medoid_seed(labels_by_seed: Mapping[int, np.ndarray]) -> tuple[int, dict[int, float]]:
    """Choose the seed with the highest mean pairwise cluster ARI."""

    seeds = sorted(int(seed) for seed in labels_by_seed)
    if not seeds:
        raise ValueError("at least one seed is required")
    scores: dict[int, float] = {}
    for seed in seeds:
        pair_scores = [
            adjusted_rand_score(labels_by_seed[seed], labels_by_seed[other])
            for other in seeds
            if other != seed
        ]
        scores[seed] = float(np.mean(pair_scores)) if pair_scores else 1.0
    medoid = max(seeds, key=lambda seed: (scores[seed], -seed))
    return int(medoid), scores


def align_labels_to_medoid(
    labels: np.ndarray,
    medoid_labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Align source clusters to medoid IDs with Hungarian overlap matching."""

    source = _canonicalize_labels(labels, len(labels))
    target = _canonicalize_labels(medoid_labels, len(medoid_labels))
    if source.shape != target.shape:
        raise ValueError("labels and medoid_labels must have the same shape")
    source_clusters = sorted(int(value) for value in np.unique(source) if value >= 0)
    target_clusters = sorted(int(value) for value in np.unique(target) if value >= 0)
    mapping: dict[int, int] = {}
    if source_clusters and target_clusters:
        overlap = np.zeros((len(target_clusters), len(source_clusters)), dtype=np.int64)
        for target_index, target_cluster in enumerate(target_clusters):
            for source_index, source_cluster in enumerate(source_clusters):
                overlap[target_index, source_index] = int(
                    np.sum((target == target_cluster) & (source == source_cluster))
                )
        target_rows, source_columns = linear_sum_assignment(-overlap)
        for target_index, source_index in zip(target_rows, source_columns, strict=True):
            if overlap[target_index, source_index] > 0:
                mapping[source_clusters[int(source_index)]] = target_clusters[int(target_index)]
    aligned = np.full(len(source), -1, dtype=np.int64)
    for source_cluster, target_cluster in mapping.items():
        aligned[source == source_cluster] = target_cluster
    return aligned, mapping


def derive_consensus_labels(
    labels_by_seed: Mapping[int, np.ndarray],
    *,
    medoid_seed: int,
    threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[int, int]]]:
    """Align all seed labels, calculate agreement, and apply the fixed gate."""

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("consensus threshold must be between zero and one")
    if medoid_seed not in labels_by_seed:
        raise ValueError("medoid_seed must be present in labels_by_seed")
    medoid_labels = _canonicalize_labels(
        labels_by_seed[medoid_seed], len(labels_by_seed[medoid_seed])
    )
    aligned_by_seed: dict[int, np.ndarray] = {}
    mappings: dict[int, dict[int, int]] = {}
    for seed in sorted(labels_by_seed):
        if int(seed) == int(medoid_seed):
            aligned_by_seed[int(seed)] = medoid_labels.copy()
            mappings[int(seed)] = {
                int(cluster): int(cluster)
                for cluster in np.unique(medoid_labels)
                if int(cluster) >= 0
            }
        else:
            aligned, mapping = align_labels_to_medoid(
                labels_by_seed[seed], medoid_labels
            )
            aligned_by_seed[int(seed)] = aligned
            mappings[int(seed)] = mapping
    stacked = np.vstack([aligned_by_seed[seed] for seed in sorted(aligned_by_seed)])
    agreement = np.mean(stacked == medoid_labels[None, :], axis=0).astype(np.float64)
    consensus = medoid_labels.copy()
    consensus[agreement < threshold] = -1
    return _canonicalize_labels(consensus, len(consensus)), agreement, mappings


def pairwise_seed_agreement(
    runs_by_seed: Mapping[int, DiscoveryRun],
    *,
    reference_k: int,
    configuration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compute required pairwise stability diagnostics for all seed pairs."""

    rows: list[dict[str, Any]] = []
    seeds = sorted(runs_by_seed)
    for seed_a, seed_b in itertools.combinations(seeds, 2):
        labels_a = runs_by_seed[seed_a].labels
        labels_b = runs_by_seed[seed_b].labels
        noise_a = labels_a == -1
        noise_b = labels_b == -1
        union = int(np.sum(noise_a | noise_b))
        rows.append(
            {
                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "n_neighbors": int(configuration["n_neighbors"]),
                "n_components": int(configuration["n_components"]),
                "min_dist": float(configuration["min_dist"]),
                "cluster_ari": float(adjusted_rand_score(labels_a, labels_b)),
                "cluster_nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
                "noise_jaccard": float(
                    np.sum(noise_a & noise_b) / union if union else 1.0
                ),
                "cluster_count_a": int(np.unique(labels_a[labels_a >= 0]).size),
                "cluster_count_b": int(np.unique(labels_b[labels_b >= 0]).size),
                "cluster_count_abs_delta": int(
                    abs(
                        np.unique(labels_a[labels_a >= 0]).size
                        - np.unique(labels_b[labels_b >= 0]).size
                    )
                ),
                "umap_neighbor_overlap": float(
                    mean_neighbor_preservation(
                        runs_by_seed[seed_a].umap_neighbors[reference_k],
                        runs_by_seed[seed_b].umap_neighbors[reference_k],
                    )
                ),
            }
        )
    return rows


def _cluster_hierarchy_maps(
    labels: np.ndarray,
    hierarchies: Sequence[tuple[str, ...]],
) -> dict[str, dict[int, str]]:
    mappings = {"leaf": {}, "parent": {}, "top": {}}
    for cluster in sorted(int(value) for value in np.unique(labels) if value >= 0):
        selected = [
            path for path, label in zip(hierarchies, labels, strict=True)
            if int(label) == cluster
        ]
        mappings["leaf"][cluster] = _majority(path[-1] for path in selected)
        mappings["parent"][cluster] = _majority(path[-2] for path in selected)
        mappings["top"][cluster] = _majority(path[0] for path in selected)
    return mappings


def _majority(values: Iterable[str]) -> str | None:
    counts = Counter(str(value) for value in values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0] if counts else None


def _hierarchy_distance(
    labels: np.ndarray,
    hierarchies: Sequence[tuple[str, ...]],
    mappings: Mapping[str, Mapping[int, str]],
) -> float:
    distances: list[float] = []
    for path, label in zip(hierarchies, labels, strict=True):
        label = int(label)
        if label < 0 or label not in mappings["leaf"]:
            distances.append(3.0)
        elif mappings["leaf"][label] == path[-1]:
            distances.append(0.0)
        elif mappings["parent"].get(label) == path[-2]:
            distances.append(1.0)
        elif mappings["top"].get(label) == path[0]:
            distances.append(2.0)
        else:
            distances.append(3.0)
    return float(np.mean(distances)) if distances else 0.0


def _membership_diagnostics(
    memberships: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize soft memberships without serializing the membership matrix."""

    values = np.asarray(memberships, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError("memberships must align with labels")
    if values.shape[1] == 0:
        return {
            "membership_cluster_count": 0,
            "membership_coverage": 0.0,
            "mean_membership_row_sum": 0.0,
            "mean_membership_unexplained": 1.0,
            "mean_max_affinity": 0.0,
            "mean_normalized_entropy": None,
            "soft_recommended_leaf_nmi": 0.0,
            "soft_recommended_leaf_ari": 0.0,
        }
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError("memberships must be finite and non-negative")
    values = np.maximum(values, 0.0)
    row_sums = np.sum(values, axis=1)
    maximum = np.max(values, axis=1)
    recommended = np.argmax(values, axis=1).astype(np.int64)
    recommended[row_sums <= 1e-12] = -1
    safe_sums = np.maximum(row_sums, 1e-12)
    probabilities = values / safe_sums[:, None]
    safe_probabilities = np.maximum(probabilities, 1e-12)
    entropy = -np.sum(
        np.where(
            probabilities > 1e-12,
            probabilities * np.log(safe_probabilities),
            0.0,
        ),
        axis=1,
    )
    normalized_entropy = entropy / math.log(values.shape[1]) if values.shape[1] > 1 else np.zeros(len(values))
    hierarchies = _metadata_hierarchies(metadata)
    true_leaf = np.asarray([path[-1] for path in hierarchies], dtype=object)
    return {
        "membership_cluster_count": int(values.shape[1]),
        "membership_coverage": float(np.mean(row_sums > 1e-12)),
        "mean_membership_row_sum": float(np.mean(row_sums)),
        "mean_membership_unexplained": float(np.mean(np.clip(1.0 - row_sums, 0.0, 1.0))),
        "mean_max_affinity": float(np.mean(maximum)),
        "mean_normalized_entropy": float(np.mean(normalized_entropy)),
        "soft_recommended_leaf_nmi": float(normalized_mutual_info_score(true_leaf, recommended)),
        "soft_recommended_leaf_ari": float(adjusted_rand_score(true_leaf, recommended)),
    }


def route_quality_metrics(
    coordinates: np.ndarray,
    labels: np.ndarray,
    memberships: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
    metadata: pd.DataFrame,
    *,
    pca_support: np.ndarray | None = None,
    guard_scores: np.ndarray | None = None,
    consensus_agreement: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute hierarchy-aware hard metrics and compact soft diagnostics."""

    coordinates = _as_finite_matrix(coordinates, name="coordinates")
    labels = _canonicalize_labels(labels, len(coordinates))
    probabilities = _validate_probability_vector(
        probabilities, len(labels), name="probabilities"
    )
    outlier_scores = _validate_probability_vector(
        outlier_scores, len(labels), name="outlier_scores"
    )
    if len(metadata) != len(labels):
        raise ValueError("metadata must align with labels")
    hierarchies = _metadata_hierarchies(metadata)
    targets = {
        "leaf": np.asarray([path[-1] for path in hierarchies], dtype=object),
        "parent": np.asarray([path[-2] for path in hierarchies], dtype=object),
        "top": np.asarray([path[0] for path in hierarchies], dtype=object),
    }
    non_noise = labels >= 0
    cluster_count = int(np.unique(labels[non_noise]).size)
    silhouette: float | None = None
    if cluster_count >= 2 and int(np.sum(non_noise)) > cluster_count:
        try:
            silhouette = float(
                silhouette_score(
                    coordinates[non_noise],
                    labels[non_noise],
                    metric="euclidean",
                )
            )
        except ValueError:
            silhouette = None
    mappings = _cluster_hierarchy_maps(labels, hierarchies)
    result: dict[str, Any] = {
        "leaf_nmi": float(normalized_mutual_info_score(targets["leaf"], labels)),
        "leaf_ari": float(adjusted_rand_score(targets["leaf"], labels)),
        "parent_nmi": float(normalized_mutual_info_score(targets["parent"], labels)),
        "parent_ari": float(adjusted_rand_score(targets["parent"], labels)),
        "top_nmi": float(normalized_mutual_info_score(targets["top"], labels)),
        "top_ari": float(adjusted_rand_score(targets["top"], labels)),
        "hierarchy_distance": _hierarchy_distance(labels, hierarchies, mappings),
        "cluster_count": cluster_count,
        "clusters": cluster_count,
        "noise_ratio": float(np.mean(~non_noise)),
        "silhouette": silhouette,
        "mean_probability": float(np.mean(probabilities)),
        "mean_outlier_score": float(np.mean(outlier_scores)),
    }
    result.update(_membership_diagnostics(memberships, labels, metadata))
    if pca_support is not None:
        result["mean_pca_local_support"] = float(np.mean(pca_support))
        result["accepted_pca_support"] = float(
            np.mean(pca_support[labels >= 0]) if np.any(labels >= 0) else 0.0
        )
    if guard_scores is not None:
        result["mean_guard_score"] = float(np.mean(guard_scores))
        result["guard_acceptance"] = float(np.mean(guard_scores >= DEFAULT_GUARD_THRESHOLD))
    if consensus_agreement is not None:
        result["mean_consensus_agreement"] = float(np.mean(consensus_agreement))
        result["consensus_acceptance"] = float(
            np.mean(consensus_agreement >= DEFAULT_CONSENSUS_THRESHOLD)
        )
    return result


def _preservation_metrics(
    prepared: PreparedDataset,
    run: DiscoveryRun,
) -> dict[str, float]:
    """Flatten Raw↔PCA, Raw↔UMAP, and PCA↔UMAP preservation metrics."""

    result: dict[str, float] = {}
    for k in prepared.pca_neighbors:
        raw_pca = float(
            mean_neighbor_preservation(
                prepared.raw_neighbors[k], prepared.pca_neighbors[k]
            )
        )
        raw_umap = float(
            mean_neighbor_preservation(
                prepared.raw_neighbors[k], run.umap_neighbors[k]
            )
        )
        pca_umap = float(
            mean_neighbor_preservation(
                prepared.pca_neighbors[k], run.umap_neighbors[k]
            )
        )
        result[f"raw_pca_k{k}"] = raw_pca
        result[f"raw_umap_k{k}"] = raw_umap
        result[f"pca_umap_k{k}"] = pca_umap
        result[f"umap_additional_loss_k{k}"] = float(raw_pca - raw_umap)
    return result


def _mean_preservation(metrics: Mapping[str, Any], prefix: str, k_values: Sequence[int]) -> float | None:
    values = [metrics.get(f"{prefix}_k{k}") for k in k_values]
    return _mean(values)


def _mean_pairwise(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return _mean(row.get(field) for row in rows)


def _route_row(
    *,
    dataset_name: str,
    dataset_rows: int,
    route: RouteOutput,
    prepared: PreparedDataset,
    run_seed: int,
    fit_count: int,
    shared_fit_count: int,
    membership_source: str,
    medoid_seed: int,
    seed_agreement_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "dataset_rows": int(dataset_rows),
        "route": route.name,
        "run_seed": int(run_seed),
        "medoid_seed": int(medoid_seed),
        "fit_count": int(fit_count),
        "shared_unique_fit_count": int(shared_fit_count),
        "membership_source": membership_source,
        "pca_dimension": int(prepared.pca_selection.selected_dimension),
        "runtime_sec": float(route.runtime_sec["total"]),
        "pca_selection_sec": float(prepared.timings_sec["pca_selection_sec"]),
        "shared_preprocessing_sec": float(prepared.timings_sec["total"]),
        "umap_hdbscan_sec": float(route.runtime_sec.get("discovery_sec", 0.0)),
        "guard_sec": float(route.runtime_sec.get("guard_sec", 0.0)),
        "alignment_sec": float(route.runtime_sec.get("alignment_sec", 0.0)),
        "fit_reuse_seed42": bool(route.name in (ROUTE_NATIVE, ROUTE_GUARDED)),
        "seed_pair_cluster_ari_mean": _mean_pairwise(seed_agreement_rows, "cluster_ari"),
        "seed_pair_cluster_nmi_mean": _mean_pairwise(seed_agreement_rows, "cluster_nmi"),
        "seed_pair_noise_jaccard_mean": _mean_pairwise(seed_agreement_rows, "noise_jaccard"),
        "seed_pair_umap_neighbor_overlap_mean": _mean_pairwise(seed_agreement_rows, "umap_neighbor_overlap"),
    }
    row.update(route.metrics)
    row["mean_raw_umap"] = _mean_preservation(row, "raw_umap", prepared.k_values)
    row["mean_pca_umap"] = _mean_preservation(row, "pca_umap", prepared.k_values)
    row["mean_raw_pca"] = _mean_preservation(row, "raw_pca", prepared.k_values)
    return row


def _effective_baseline_configuration(n_samples: int) -> dict[str, Any]:
    """Clamp only the parameters that cannot be represented on tiny inputs."""

    return {
        **BASELINE_UMAP_CONFIGURATION,
        "requested_n_neighbors": int(DEFAULT_UMAP_N_NEIGHBORS),
        "requested_n_components": int(DEFAULT_UMAP_COMPONENTS),
        "n_neighbors": int(min(DEFAULT_UMAP_N_NEIGHBORS, n_samples - 1)),
        "n_components": int(min(DEFAULT_UMAP_COMPONENTS, max(1, n_samples - 2))),
    }


def _pca_neighbor_indices_for_membership(prepared: PreparedDataset) -> tuple[int, np.ndarray]:
    effective_k = min(DEFAULT_PCA_NEIGHBOR_COUNT, len(prepared.pca_features) - 1)
    if effective_k < 1:
        raise ValueError("PCA exact-kNN membership requires at least two samples")
    if effective_k not in prepared.pca_neighbors:
        neighbors = neighbors_by_k(
            prepared.pca_features,
            (effective_k,),
            metric="cosine",
        )[effective_k]
    else:
        neighbors = prepared.pca_neighbors[effective_k]
    return int(effective_k), neighbors


def _fit_row(
    dataset_name: str,
    run: DiscoveryRun,
    prepared: PreparedDataset,
    metadata: pd.DataFrame,
    k_values: Sequence[int],
) -> dict[str, Any]:
    metrics = route_quality_metrics(
        run.coordinates,
        run.labels,
        run.native_memberships,
        run.probabilities,
        run.outlier_scores,
        metadata,
    )
    metrics.update(_preservation_metrics(prepared, run))
    row: dict[str, Any] = {
        "row_type": "discovery_fit",
        "dataset": dataset_name,
        "dataset_rows": int(len(metadata)),
        "seed": int(run.seed),
        "cluster_count": int(np.unique(run.labels[run.labels >= 0]).size),
        "umap_hdbscan_sec": float(run.timings_sec["total"]),
        "umap_sec": float(run.timings_sec["umap_sec"]),
        "hdbscan_sec": float(run.timings_sec["hdbscan_sec"]),
        "native_membership_sec": float(run.timings_sec["native_membership_sec"]),
    }
    row.update(metrics)
    row["mean_raw_umap"] = _mean_preservation(row, "raw_umap", k_values)
    row["mean_pca_umap"] = _mean_preservation(row, "pca_umap", k_values)
    return row


def _threshold_diagnostic_rows(
    *,
    dataset_name: str,
    run: DiscoveryRun,
    prepared: PreparedDataset,
    pca_support: np.ndarray,
    guard_scores: np.ndarray,
    metadata: pd.DataFrame,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        gated = gate_labels(run.labels, guard_scores, threshold)
        hard_metrics = route_quality_metrics(
            run.coordinates,
            gated,
            np.zeros((len(gated), 0), dtype=np.float64),
            run.probabilities,
            run.outlier_scores,
            metadata,
            pca_support=pca_support,
            guard_scores=guard_scores,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "threshold": threshold,
                "is_default": math.isclose(threshold, DEFAULT_GUARD_THRESHOLD),
                "accepted_fraction": float(np.mean(gated >= 0)),
                "rejected_fraction": float(np.mean(gated < 0)),
                "base_non_noise_fraction": float(np.mean(run.labels >= 0)),
                "leaf_nmi": hard_metrics["leaf_nmi"],
                "leaf_ari": hard_metrics["leaf_ari"],
                "parent_nmi": hard_metrics["parent_nmi"],
                "top_nmi": hard_metrics["top_nmi"],
                "hierarchy_distance": hard_metrics["hierarchy_distance"],
                "cluster_count": hard_metrics["cluster_count"],
                "noise_ratio": hard_metrics["noise_ratio"],
                "mean_pca_local_support": hard_metrics["mean_pca_local_support"],
                "mean_guard_score": float(np.mean(guard_scores)),
            }
        )
    return rows


def _cluster_support_rows(
    *,
    dataset_name: str,
    route: RouteOutput,
    base_labels: np.ndarray,
    metadata: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build cluster-level support rows only; no point-level arrays are saved."""

    rows: list[dict[str, Any]] = []
    for cluster in sorted(int(value) for value in np.unique(base_labels) if value >= 0):
        base_mask = base_labels == cluster
        after_mask = base_mask & (route.labels >= 0)
        persistence = (
            float(route.persistence[cluster])
            if cluster < len(route.persistence)
            else None
        )
        rows.append(
            {
                "dataset": dataset_name,
                "dataset_rows": int(len(metadata)),
                "route": route.name,
                "cluster_id": int(cluster),
                "points_before_gate": int(np.sum(base_mask)),
                "points_after_gate": int(np.sum(after_mask)),
                "accepted_fraction": float(
                    np.mean(after_mask[base_mask]) if np.any(base_mask) else 0.0
                ),
                "persistence": persistence,
                "mean_probability": float(np.mean(route.probabilities[base_mask]))
                if np.any(base_mask)
                else None,
                "mean_pca_local_support": float(np.mean(route.pca_support[base_mask]))
                if np.any(base_mask)
                else None,
                "mean_guard_score": float(np.mean(route.guard_scores[base_mask]))
                if np.any(base_mask) and route.guard_scores.size
                else None,
                "mean_consensus_agreement": (
                    float(np.mean(route.consensus_agreement[base_mask]))
                    if route.consensus_agreement is not None and np.any(base_mask)
                    else None
                ),
            }
        )
    return rows


def run_route_comparison(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    source_indices: np.ndarray | None = None,
    *,
    dataset_name: str = "main_720",
    input_path: Path | str = DEFAULT_INPUT,
    sample_seed: int = DEFAULT_DATASET_SAMPLE_SEED,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    umap_seeds: Sequence[int] = DEFAULT_UMAP_SEEDS,
    pca_selection_k_values: Sequence[int] = DEFAULT_PCA_SELECTION_K_VALUES,
    pca_max_components: int = DEFAULT_MAX_COMPONENTS,
    pca_min_components: int = DEFAULT_MIN_COMPONENTS,
    pca_component_step: int = DEFAULT_COMPONENT_STEP,
    minimum_preservation_gain: float = 0.05,
    pca_seed: int = 42,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    guard_threshold: float = DEFAULT_GUARD_THRESHOLD,
    consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    guard_curve_thresholds: Sequence[float] = DEFAULT_GUARD_CURVE_THRESHOLDS,
    umap_class: UMAPClass | None = None,
    hdbscan_class: HDBSCANClass | None = None,
    native_membership_function: MembershipFunction | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run A/B/C with one shared PCA and exactly five default discovery fits."""

    seeds = tuple(dict.fromkeys(int(seed) for seed in umap_seeds))
    if seeds != DEFAULT_UMAP_SEEDS:
        raise ValueError(
            "five_seed_stable requires the fixed five seeds (42, 43, 44, 45, 46)"
        )
    if not 0.0 <= float(guard_threshold) <= 1.0:
        raise ValueError("guard_threshold must be between zero and one")
    if not 0.0 <= float(consensus_threshold) <= 1.0:
        raise ValueError("consensus_threshold must be between zero and one")
    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError("invalid HDBSCAN minimum cluster parameters")

    prepared = prepare_dataset(
        embeddings,
        metadata,
        source_indices,
        k_values=k_values,
        pca_selection_k_values=pca_selection_k_values,
        pca_max_components=pca_max_components,
        pca_min_components=pca_min_components,
        pca_component_step=pca_component_step,
        minimum_preservation_gain=minimum_preservation_gain,
        pca_seed=pca_seed,
    )
    configuration = _effective_baseline_configuration(len(prepared.pca_features))
    hdb_configuration = {
        **HDBSCAN_CONFIGURATION,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
    }
    pca_k, pca_membership_neighbors = _pca_neighbor_indices_for_membership(prepared)

    runs_by_seed: dict[int, DiscoveryRun] = {}
    fit_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        if progress:
            print(
                f"[{dataset_name}] discovery fit {index}/{len(seeds)} seed={seed}",
                flush=True,
            )
        run = _fit_umap_hdbscan(
            prepared.pca_features,
            configuration,
            seed=seed,
            hdbscan_configuration=hdb_configuration,
            umap_class=umap_class,
            hdbscan_class=hdbscan_class,
            native_membership_function=native_membership_function,
        )
        started = time.perf_counter()
        umap_neighbors = neighbors_by_k(
            run.coordinates,
            prepared.k_values,
            metric="euclidean",
        )
        umap_knn_sec = float(time.perf_counter() - started)
        run = replace(
            run,
            umap_neighbors=umap_neighbors,
            timings_sec={**run.timings_sec, "umap_knn_sec": umap_knn_sec},
        )
        runs_by_seed[int(seed)] = run
        fit_rows.append(
            _fit_row(
                dataset_name,
                run,
                prepared,
                prepared.metadata,
                prepared.k_values,
            )
        )

    reference_k = 15 if 15 in prepared.k_values else max(prepared.k_values)
    seed_agreement = pairwise_seed_agreement(
        runs_by_seed,
        reference_k=reference_k,
        configuration=configuration,
    )
    for row in seed_agreement:
        row.update(
            {
                "dataset": dataset_name,
                "dataset_rows": int(len(prepared.metadata)),
                "reference_k": int(reference_k),
            }
        )

    medoid_seed, medoid_scores = choose_medoid_seed(
        {seed: run.labels for seed, run in runs_by_seed.items()}
    )
    seed42 = runs_by_seed[42]
    medoid = runs_by_seed[medoid_seed]
    guard_started = time.perf_counter()
    seed42_pca_support = pca_local_cluster_support(
        prepared.pca_neighbors[pca_k] if pca_k in prepared.pca_neighbors else pca_membership_neighbors,
        seed42.labels,
    )
    seed42_guard_scores = compute_guard_scores(
        seed42.labels,
        seed42.probabilities,
        seed42.outlier_scores,
        seed42.cluster_persistence,
        seed42_pca_support,
    )
    b_labels = gate_labels(seed42.labels, seed42_guard_scores, guard_threshold)
    b_guard_sec = float(time.perf_counter() - guard_started)

    alignment_started = time.perf_counter()
    stable_labels, consensus_agreement, mappings = derive_consensus_labels(
        {seed: run.labels for seed, run in runs_by_seed.items()},
        medoid_seed=medoid_seed,
        threshold=consensus_threshold,
    )
    alignment_sec = float(time.perf_counter() - alignment_started)
    medoid_pca_support = pca_local_cluster_support(
        prepared.pca_neighbors[pca_k] if pca_k in prepared.pca_neighbors else pca_membership_neighbors,
        medoid.labels,
    )

    route_outputs: dict[str, RouteOutput] = {}
    common_discovery_sec = float(
        seed42.timings_sec["total"] + seed42.timings_sec.get("umap_knn_sec", 0.0)
    )
    all_discovery_sec = float(
        sum(
            run.timings_sec["total"] + run.timings_sec.get("umap_knn_sec", 0.0)
            for run in runs_by_seed.values()
        )
    )

    native_metrics = route_quality_metrics(
        seed42.coordinates,
        seed42.labels,
        seed42.native_memberships,
        seed42.probabilities,
        seed42.outlier_scores,
        prepared.metadata,
    )
    native_metrics.update(_preservation_metrics(prepared, seed42))
    route_outputs[ROUTE_NATIVE] = RouteOutput(
        name=ROUTE_NATIVE,
        labels=seed42.labels.copy(),
        memberships=seed42.native_memberships,
        coordinates=seed42.coordinates,
        base_labels=seed42.labels.copy(),
        probabilities=seed42.probabilities,
        outlier_scores=seed42.outlier_scores,
        persistence=seed42.cluster_persistence,
        pca_support=seed42_pca_support,
        guard_scores=np.zeros(len(seed42.labels), dtype=np.float64),
        consensus_agreement=None,
        metrics=native_metrics,
        runtime_sec={
            "shared_preprocessing_sec": prepared.timings_sec["total"],
            "discovery_sec": common_discovery_sec,
            "guard_sec": 0.0,
            "alignment_sec": 0.0,
            "pca_exact_knn_sec": 0.0,
            "total": float(prepared.timings_sec["total"] + common_discovery_sec),
        },
    )

    started = time.perf_counter()
    b_membership = pca_exact_knn_memberships(
        prepared.pca_features,
        b_labels,
        seed42.probabilities,
        neighbor_indices_by_k=pca_membership_neighbors,
    )
    b_soft_sec = float(time.perf_counter() - started)
    b_metrics = route_quality_metrics(
        seed42.coordinates,
        b_labels,
        b_membership["affinities"],
        seed42.probabilities,
        seed42.outlier_scores,
        prepared.metadata,
        pca_support=seed42_pca_support,
        guard_scores=seed42_guard_scores,
    )
    b_metrics.update(_preservation_metrics(prepared, seed42))
    route_outputs[ROUTE_GUARDED] = RouteOutput(
        name=ROUTE_GUARDED,
        labels=b_labels,
        memberships=b_membership["affinities"],
        coordinates=seed42.coordinates,
        base_labels=seed42.labels.copy(),
        probabilities=seed42.probabilities,
        outlier_scores=seed42.outlier_scores,
        persistence=seed42.cluster_persistence,
        pca_support=seed42_pca_support,
        guard_scores=seed42_guard_scores,
        consensus_agreement=None,
        metrics=b_metrics,
        runtime_sec={
            "shared_preprocessing_sec": prepared.timings_sec["total"],
            "discovery_sec": common_discovery_sec,
            "guard_sec": b_guard_sec,
            "alignment_sec": 0.0,
            "pca_exact_knn_sec": b_soft_sec,
            "total": float(
                prepared.timings_sec["total"]
                + common_discovery_sec
                + b_guard_sec
                + b_soft_sec
            ),
        },
    )
    started = time.perf_counter()
    c_membership = pca_exact_knn_memberships(
        prepared.pca_features,
        stable_labels,
        np.clip(medoid.probabilities * consensus_agreement, 0.0, 1.0),
        neighbor_indices_by_k=pca_membership_neighbors,
    )
    c_soft_sec = float(time.perf_counter() - started)
    c_metrics = route_quality_metrics(
        medoid.coordinates,
        stable_labels,
        c_membership["affinities"],
        medoid.probabilities,
        medoid.outlier_scores,
        prepared.metadata,
        pca_support=medoid_pca_support,
        consensus_agreement=consensus_agreement,
    )
    c_metrics.update(_preservation_metrics(prepared, medoid))
    # ``derive_consensus_labels`` performed all Hungarian alignments once;
    # this value is the measured route-level alignment/bookkeeping cost.
    _ = mappings
    route_outputs[ROUTE_STABLE] = RouteOutput(
        name=ROUTE_STABLE,
        labels=stable_labels,
        memberships=c_membership["affinities"],
        coordinates=medoid.coordinates,
        base_labels=medoid.labels.copy(),
        probabilities=medoid.probabilities,
        outlier_scores=medoid.outlier_scores,
        persistence=medoid.cluster_persistence,
        pca_support=medoid_pca_support,
        guard_scores=np.zeros(len(stable_labels), dtype=np.float64),
        consensus_agreement=consensus_agreement,
        metrics=c_metrics,
        runtime_sec={
            "shared_preprocessing_sec": prepared.timings_sec["total"],
            "discovery_sec": all_discovery_sec,
            "guard_sec": 0.0,
            "alignment_sec": alignment_sec,
            "pca_exact_knn_sec": c_soft_sec,
            "total": float(
                prepared.timings_sec["total"]
                + all_discovery_sec
                + alignment_sec
                + c_soft_sec
            ),
        },
    )

    threshold_rows = _threshold_diagnostic_rows(
        dataset_name=dataset_name,
        run=seed42,
        prepared=prepared,
        pca_support=seed42_pca_support,
        guard_scores=seed42_guard_scores,
        metadata=prepared.metadata,
        thresholds=guard_curve_thresholds,
    )

    route_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for route_name in ROUTE_NAMES:
        route = route_outputs[route_name]
        route_row = _route_row(
            dataset_name=dataset_name,
            dataset_rows=len(prepared.metadata),
            route=route,
            prepared=prepared,
            run_seed=42,
            fit_count=1 if route_name != ROUTE_STABLE else len(seeds),
            shared_fit_count=len(seeds),
            membership_source=(
                "hdbscan_native" if route_name == ROUTE_NATIVE else "pca_exact_knn"
            ),
            medoid_seed=medoid_seed,
            seed_agreement_rows=seed_agreement,
        )
        route_rows.append(route_row)
        for component, seconds in route.runtime_sec.items():
            timing_rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_rows": int(len(prepared.metadata)),
                    "route": route_name,
                    "component": component,
                    "seconds": float(seconds),
                    "fit_count": int(route_row["fit_count"]),
                }
            )
        support_rows.extend(
            _cluster_support_rows(
                dataset_name=dataset_name,
                route=route,
                base_labels=route.base_labels,
                metadata=prepared.metadata,
            )
        )

    selected_pca = prepared.pca_selection.to_dict()
    selected_pca["selected_dimension_knn_preservation"] = {
        f"k{k}": float(
            mean_neighbor_preservation(
                prepared.raw_neighbors[k], prepared.pca_neighbors[k]
            )
        )
        for k in prepared.k_values
    }
    selected_candidate = next(
        (
            candidate
            for candidate in selected_pca.get("candidates", [])
            if int(candidate["dimension"])
            == int(prepared.pca_selection.selected_dimension)
        ),
        None,
    )
    selected_pca["selected_cumulative_explained_variance"] = (
        None
        if selected_candidate is None
        else selected_candidate.get("cumulative_explained_variance")
    )
    return {
        "dataset": {
            "name": dataset_name,
            "rows": int(len(prepared.metadata)),
            "embedding_dimension": int(prepared.normalized_embeddings.shape[1]),
            "input": str(input_path),
            "sample_seed": int(sample_seed),
            "l2_normalized": True,
            "embedding_model": "gemini-embedding-001",
            "embedding_task": "CLUSTERING",
            "fingerprint": _dataset_fingerprint(
                prepared.normalized_embeddings,
                prepared.metadata,
                prepared.source_indices,
            ),
        },
        "selected_pca": selected_pca,
        "raw_pca_preservation": {
            f"k{k}": float(
                mean_neighbor_preservation(
                    prepared.raw_neighbors[k], prepared.pca_neighbors[k]
                )
            )
            for k in prepared.k_values
        },
        "preprocessing_timings_sec": prepared.timings_sec,
        "configuration": configuration,
        "fit_rows": fit_rows,
        "route_rows": route_rows,
        "timing_rows": timing_rows,
        "cluster_support_rows": support_rows,
        "seed_agreement_rows": seed_agreement,
        "threshold_rows": threshold_rows,
        "medoid": {
            "seed": int(medoid_seed),
            "mean_pairwise_ari_by_seed": medoid_scores,
            "consensus_threshold": float(consensus_threshold),
            "mean_consensus_agreement": float(np.mean(consensus_agreement)),
            "consensus_acceptance": float(
                np.mean(consensus_agreement >= float(consensus_threshold))
            ),
            "consensus_cluster_count": int(
                np.unique(stable_labels[stable_labels >= 0]).size
            ),
        },
        "guard": {
            "threshold": float(guard_threshold),
            "weights": dict(GUARD_WEIGHTS),
            "mean_score": float(np.mean(seed42_guard_scores)),
            "accepted_fraction": float(np.mean(b_labels >= 0)),
        },
    }


def _sample_loaded_dataset(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    sample_size: int,
    sample_seed: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if not 2 <= int(sample_size) <= len(embeddings):
        raise ValueError(
            f"sample_size must be between 2 and {len(embeddings)}, got {sample_size}"
        )
    if int(sample_size) == len(embeddings):
        indices = np.arange(len(embeddings), dtype=np.int64)
    else:
        rng = np.random.default_rng(int(sample_seed))
        indices = np.sort(
            rng.choice(len(embeddings), size=int(sample_size), replace=False)
        ).astype(np.int64)
    return (
        np.asarray(embeddings[indices], dtype=np.float32),
        metadata.iloc[indices].reset_index(drop=True).copy(),
        indices,
    )


def run_full_benchmark(
    input_path: Path = DEFAULT_INPUT,
    *,
    dataset_sample_size: int = DEFAULT_DATASET_SAMPLE_SIZE,
    dataset_sample_seed: int = DEFAULT_DATASET_SAMPLE_SEED,
    scale_sample_sizes: Sequence[int] = DEFAULT_SCALE_SAMPLE_SIZES,
    skip_scale: bool = False,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    umap_seeds: Sequence[int] = DEFAULT_UMAP_SEEDS,
    pca_max_components: int = DEFAULT_MAX_COMPONENTS,
    pca_min_components: int = DEFAULT_MIN_COMPONENTS,
    pca_component_step: int = DEFAULT_COMPONENT_STEP,
    minimum_preservation_gain: float = 0.05,
    umap_class: UMAPClass | None = None,
    hdbscan_class: HDBSCANClass | None = None,
    native_membership_function: MembershipFunction | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run the main 720-row benchmark and optional scale-size benchmarks."""

    if not input_path.exists():
        raise FileNotFoundError(f"Gemini embedding dataset does not exist: {input_path}")
    embeddings, metadata = load_embeddings_from_json(input_path)
    if "class" not in metadata.columns or "class_hierarchy" not in metadata.columns:
        raise ValueError("Gemini dataset must contain class and class_hierarchy metadata")

    requested_sizes = [int(dataset_sample_size)]
    if not skip_scale:
        requested_sizes.extend(int(size) for size in scale_sample_sizes)
    sizes: list[int] = []
    skipped_sizes: list[int] = []
    for size in requested_sizes:
        if size in sizes:
            continue
        if size > len(embeddings):
            skipped_sizes.append(size)
        else:
            sizes.append(size)
    if int(dataset_sample_size) not in sizes:
        raise ValueError(
            f"dataset_sample_size must be between 2 and {len(embeddings)}, got {dataset_sample_size}"
        )

    dataset_results: list[dict[str, Any]] = []
    for size in sizes:
        dataset_name = (
            f"main_{size}" if size == int(dataset_sample_size) else f"scale_{size}"
        )
        selected_embeddings, selected_metadata, source_indices = _sample_loaded_dataset(
            embeddings,
            metadata,
            sample_size=size,
            sample_seed=dataset_sample_seed,
        )
        dataset_results.append(
            run_route_comparison(
                selected_embeddings,
                selected_metadata,
                source_indices,
                dataset_name=dataset_name,
                input_path=input_path,
                sample_seed=dataset_sample_seed,
                k_values=k_values,
                umap_seeds=umap_seeds,
                pca_max_components=pca_max_components,
                pca_min_components=pca_min_components,
                pca_component_step=pca_component_step,
                minimum_preservation_gain=minimum_preservation_gain,
                umap_class=umap_class,
                hdbscan_class=hdbscan_class,
                native_membership_function=native_membership_function,
                progress=progress,
            )
        )

    route_rows = [
        row
        for result in dataset_results
        for row in result["route_rows"]
    ]
    fit_rows = [
        row
        for result in dataset_results
        for row in result["fit_rows"]
    ]
    timing_rows = [
        row
        for result in dataset_results
        for row in result["timing_rows"]
    ]
    support_rows = [
        row
        for result in dataset_results
        for row in result["cluster_support_rows"]
    ]
    seed_rows = [
        row
        for result in dataset_results
        for row in result["seed_agreement_rows"]
    ]
    threshold_rows = [
        row
        for result in dataset_results
        for row in result["threshold_rows"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "offline_research_benchmark": True,
        "production_status": (
            "The existing hierarchical PCA SFCM production path is unchanged; "
            "these three UMAP/HDBSCAN routes are research comparisons only."
        ),
        "dataset": {
            "input": str(input_path),
            "embedding_model": "gemini-embedding-001",
            "embedding_task": "CLUSTERING",
            "full_rows_loaded": int(len(embeddings)),
            "full_embedding_dimension": int(embeddings.shape[1]),
            "main_sample_size": int(dataset_sample_size),
            "sample_seed": int(dataset_sample_seed),
            "scale_sample_sizes_requested": [int(size) for size in scale_sample_sizes],
            "scale_sample_sizes_skipped": skipped_sizes,
        },
        "protocol": {
            "route_baseline": "existing single-run UMAP -> HDBSCAN discovery path",
            "pca_selection_once_per_dataset": True,
            "pca_selection_seed": 42,
            "pca_selection_input": "L2-normalized Gemini embeddings",
            "discovery_input": "unnormalized selected PCA prefix",
            "pca_feature_space_for_discovery": "raw_selected_pca_prefix",
            "raw_metric": "cosine",
            "pca_metric": "cosine",
            "umap_metric": "euclidean",
            "k_values": [int(k) for k in k_values],
            "umap": {
                "n_neighbors": DEFAULT_UMAP_N_NEIGHBORS,
                "n_components": DEFAULT_UMAP_COMPONENTS,
                "min_dist": DEFAULT_UMAP_MIN_DIST,
                "init": "random",
                "n_jobs": 1,
                "seeds": list(umap_seeds),
            },
            "hdbscan": HDBSCAN_CONFIGURATION,
            "pca_exact_knn_membership": {
                "k": DEFAULT_PCA_NEIGHBOR_COUNT,
                "metric": "cosine",
                "weight": "exp(-(cosine_distance / local_median_positive_distance)^2)",
            },
            "guard": {
                "threshold": DEFAULT_GUARD_THRESHOLD,
                "weights": GUARD_WEIGHTS,
                "label_free_tuning": True,
            },
            "consensus": {
                "medoid": "highest mean pairwise cluster ARI",
                "alignment": "Hungarian maximum-overlap matching",
                "threshold": DEFAULT_CONSENSUS_THRESHOLD,
            },
            "raw_pca_neighborhood": "reported baseline, not a fourth route",
        },
        "routes": {
            ROUTE_NATIVE: {
                "fit_count": 1,
                "seed": 42,
                "membership_source": "hdbscan_native",
                "description": "One baseline seed-42 UMAP/HDBSCAN fit with native memberships.",
            },
            ROUTE_GUARDED: {
                "fit_count": 1,
                "reuses": ROUTE_NATIVE,
                "guard_threshold": DEFAULT_GUARD_THRESHOLD,
                "membership_source": "pca_exact_knn",
                "description": "The same seed-42 fit, guarded by HDBSCAN and PCA local support.",
            },
            ROUTE_STABLE: {
                "fit_count": 5,
                "seeds": list(umap_seeds),
                "reuses_seed_42_fit": True,
                "consensus_threshold": DEFAULT_CONSENSUS_THRESHOLD,
                "membership_source": "pca_exact_knn",
                "description": "Five fixed fits, ARI medoid, Hungarian alignment, and consensus gate.",
            },
        },
        "datasets": [
            {
                "dataset": result["dataset"],
                "selected_pca": result["selected_pca"],
                "raw_pca_preservation": result["raw_pca_preservation"],
                "preprocessing_timings_sec": result["preprocessing_timings_sec"],
                "configuration": result["configuration"],
                "medoid": result["medoid"],
                "guard": result["guard"],
                "fit_rows": result["fit_rows"],
                "threshold_rows": result["threshold_rows"],
            }
            for result in dataset_results
        ],
        "runs": fit_rows + route_rows,
        "route_summary": route_rows,
        "timing": timing_rows,
        "cluster_support": support_rows,
        "seed_agreement": seed_rows,
        "threshold_diagnostics": threshold_rows,
        "decision_framework": {
            "prefer_guarded_over_native_if": (
                "guarded route adds no more than 25 percent overhead, rejects low-confidence "
                "points, and does not reduce quality by more than 0.02"
            ),
            "prefer_five_seed_if": (
                "offline quality or stability improves materially and a roughly five-fold "
                "discovery cost is acceptable"
            ),
            "runtime_rule": "five_seed_stable is offline-only; do not place it in the online production path",
        },
        "artifacts": {},
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def make_plots(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    """Create the five compact PNG summaries required by the benchmark."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    route_rows = list(report.get("route_summary", []))
    main_name = f"main_{report['dataset']['main_sample_size']}"
    main_routes = [row for row in route_rows if row.get("dataset") == main_name]
    thresholds = [
        row for row in report.get("threshold_diagnostics", []) if row.get("dataset") == main_name
    ]
    support = [
        row for row in report.get("cluster_support", [])
        if row.get("dataset") == main_name and row.get("route") == ROUTE_STABLE
    ]
    seed_rows = [
        row for row in report.get("seed_agreement", []) if row.get("dataset") == main_name
    ]

    def save(name: str, draw: Callable[[Any], None]) -> str:
        path = output_dir / name
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        draw(axis)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        return name

    colors = {ROUTE_NATIVE: "#4c78a8", ROUTE_GUARDED: "#f58518", ROUTE_STABLE: "#54a24b"}
    paths: dict[str, str] = {}

    def quality_runtime(axis: Any) -> None:
        for row in route_rows:
            route = str(row["route"])
            axis.scatter(
                row.get("runtime_sec", 0.0),
                row.get("leaf_nmi", 0.0),
                color=colors.get(route, "#777777"),
                label=route,
                alpha=0.85,
            )
            axis.annotate(str(row.get("dataset", "")), (row.get("runtime_sec", 0.0), row.get("leaf_nmi", 0.0)), fontsize=7)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=7)
        axis.set_xlabel("route runtime (seconds)")
        axis.set_ylabel("leaf NMI")
        axis.set_title("Quality / runtime comparison")
        axis.grid(alpha=0.25)

    paths["quality_runtime_pareto"] = save("quality-runtime-pareto.png", quality_runtime)

    def stability(axis: Any) -> None:
        if not seed_rows:
            axis.text(0.5, 0.5, "no seed rows", ha="center")
            return
        mean_ari = _mean(row.get("cluster_ari") for row in seed_rows) or 0.0
        mean_overlap = _mean(row.get("umap_neighbor_overlap") for row in seed_rows) or 0.0
        x = np.arange(len(ROUTE_NAMES))
        axis.bar(x - 0.18, [mean_ari] * len(ROUTE_NAMES), 0.36, label="pairwise cluster ARI")
        axis.bar(x + 0.18, [mean_overlap] * len(ROUTE_NAMES), 0.36, label="UMAP neighbor overlap")
        axis.set_xticks(x, [name.replace("_", "\n") for name in ROUTE_NAMES], fontsize=7)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("mean agreement")
        axis.set_title("Seed stability (shared baseline configuration)")
        axis.legend(fontsize=7)

    paths["stability_comparison"] = save("stability-comparison.png", stability)

    def support_agreement(axis: Any) -> None:
        points = [row for row in support if row.get("mean_consensus_agreement") is not None]
        if not points:
            axis.text(0.5, 0.5, "no consensus support rows", ha="center")
            return
        axis.scatter(
            [row.get("mean_pca_local_support", 0.0) for row in points],
            [row.get("mean_consensus_agreement", 0.0) for row in points],
            s=[max(12, 4 * int(row.get("points_before_gate", 1))) for row in points],
            alpha=0.55,
        )
        axis.set_xlabel("mean PCA local support")
        axis.set_ylabel("mean seed consensus agreement")
        axis.set_title("PCA support versus seed agreement (cluster level)")
        axis.grid(alpha=0.25)

    paths["pca_support_vs_seed_agreement"] = save(
        "pca-support-vs-seed-agreement.png", support_agreement
    )

    def rejection_curve(axis: Any) -> None:
        if not thresholds:
            axis.text(0.5, 0.5, "no guard rows", ha="center")
            return
        ordered = sorted(thresholds, key=lambda row: float(row["threshold"]))
        x = [row["threshold"] for row in ordered]
        axis.plot(x, [row["leaf_nmi"] for row in ordered], marker="o", label="leaf NMI")
        axis.plot(x, [row["leaf_ari"] for row in ordered], marker="o", label="leaf ARI")
        axis.plot(x, [row["noise_ratio"] for row in ordered], marker="o", label="noise ratio")
        axis.axvline(DEFAULT_GUARD_THRESHOLD, color="#777777", linestyle="--", linewidth=1)
        axis.set_xlabel("guard threshold")
        axis.set_ylabel("metric")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("Guard rejection-quality diagnostic")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)

    paths["rejection_quality_curve"] = save("rejection-quality-curve.png", rejection_curve)

    def scale_runtime(axis: Any) -> None:
        for route in ROUTE_NAMES:
            rows = sorted(
                [row for row in route_rows if row.get("route") == route],
                key=lambda row: int(row.get("dataset_rows", 0)),
            )
            axis.plot(
                [row.get("dataset_rows", 0) for row in rows],
                [row.get("runtime_sec", 0.0) for row in rows],
                marker="o",
                label=route,
                color=colors.get(route),
            )
        axis.set_xlabel("sample size")
        axis.set_ylabel("route runtime (seconds)")
        axis.set_title("Scale runtime")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)

    paths["scale_runtime"] = save("scale-runtime.png", scale_runtime)
    return paths


def _markdown_report(report: Mapping[str, Any]) -> str:
    route_rows = list(report.get("route_summary", []))
    main_size = int(report["dataset"]["main_sample_size"])
    main_rows = [row for row in route_rows if row.get("dataset") == f"main_{main_size}"]
    scale_rows = [row for row in route_rows if row.get("dataset") != f"main_{main_size}"]
    lines = [
        "# Three-route Gemini clustering benchmark",
        "",
        "This is an offline research benchmark using Gemini `gemini-embedding-001` CLUSTERING embeddings. The existing hierarchical PCA SFCM production path remains unchanged.",
        "",
        "## Protocol",
        "",
        "All routes share one automatic PCA selection per sampled dataset. PCA is fit on L2-normalized embeddings; discovery UMAP receives the unnormalized selected PCA prefix. The baseline is the existing single-run UMAP → HDBSCAN discovery path: UMAP `(n_neighbors=15, n_components=20, min_dist=0.1, metric=euclidean, init=random, n_jobs=1)` and HDBSCAN `(min_cluster_size=5, min_samples=3, metric=euclidean, leaf, prediction_data=True)`.",
        "",
        "- **A — `umap_hdbscan_native`**: one seed-42 fit and native HDBSCAN memberships.",
        "- **B — `guarded_pca_hybrid`**: reuses A's fit, applies the fixed 0.45 guard score, and uses PCA exact-kNN memberships.",
        "- **C — `five_seed_stable`**: five total fits for seeds 42–46, ARI medoid, Hungarian label alignment, fixed 0.60 consensus gate, and PCA exact-kNN memberships.",
        "",
        "Guard tuning is label-free and pre-registered; the threshold curve is diagnostic, not a label-based selector. Raw↔PCA preservation is a baseline, not a fourth route.",
        "",
        "## Main result",
        "",
        "| Route | Leaf NMI | Leaf ARI | Parent NMI | Top NMI | Hierarchy distance | Noise | Clusters | Silhouette | Raw↔UMAP | PCA↔UMAP | Runtime (s) | Seed ARI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['route']} | {_format(row.get('leaf_nmi'))} | {_format(row.get('leaf_ari'))} | {_format(row.get('parent_nmi'))} | {_format(row.get('top_nmi'))} | {_format(row.get('hierarchy_distance'))} | {_format(row.get('noise_ratio'))} | {_format(row.get('cluster_count'), 2)} | {_format(row.get('silhouette'))} | {_format(row.get('mean_raw_umap'))} | {_format(row.get('mean_pca_umap'))} | {_format(row.get('runtime_sec'), 2)} | {_format(row.get('seed_pair_cluster_ari_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Main-data result-based conclusion",
            "",
            "The documented quality-loss criterion is strict and data-driven: an alternative passes only when Leaf NMI loss, Leaf ARI loss, and hierarchy-distance increase versus the native baseline are each ≤ 0.02. For hierarchy distance, a positive delta is worse; runtime is reported separately and is not treated as a quality metric.",
            "",
            "| Route | Leaf NMI | Δ Leaf NMI loss | Leaf ARI | Δ Leaf ARI loss | Hierarchy distance | Δ distance increase | Runtime (s) | Δ runtime (s) | Max quality loss | Criterion |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    main_by_route = {str(row.get("route")): row for row in main_rows}
    native_row = main_by_route.get(ROUTE_NATIVE)
    comparison_rows: dict[str, dict[str, Any]] = {}
    for route_name in (ROUTE_NATIVE, ROUTE_GUARDED, ROUTE_STABLE):
        row = main_by_route.get(route_name)
        if row is None:
            continue
        if route_name == ROUTE_NATIVE or native_row is None:
            nmi_loss = None
            ari_loss = None
            distance_increase = None
            runtime_delta = None
            max_quality_loss = None
            criterion = "baseline"
        else:
            def numeric(source: Mapping[str, Any], key: str) -> float | None:
                try:
                    value = source.get(key)
                    return None if value is None or value == "" else float(value)
                except (TypeError, ValueError):
                    return None

            native_nmi = numeric(native_row, "leaf_nmi")
            native_ari = numeric(native_row, "leaf_ari")
            native_distance = numeric(native_row, "hierarchy_distance")
            native_runtime = numeric(native_row, "runtime_sec")
            nmi = numeric(row, "leaf_nmi")
            ari = numeric(row, "leaf_ari")
            distance = numeric(row, "hierarchy_distance")
            runtime = numeric(row, "runtime_sec")
            nmi_loss = None if native_nmi is None or nmi is None else max(0.0, native_nmi - nmi)
            ari_loss = None if native_ari is None or ari is None else max(0.0, native_ari - ari)
            distance_increase = (
                None
                if native_distance is None or distance is None
                else max(0.0, distance - native_distance)
            )
            runtime_delta = None if native_runtime is None or runtime is None else runtime - native_runtime
            quality_components = [
                value for value in (nmi_loss, ari_loss, distance_increase) if value is not None
            ]
            max_quality_loss = max(quality_components) if len(quality_components) == 3 else None
            criterion = (
                "WITHIN ≤0.02"
                if max_quality_loss is not None and max_quality_loss <= 0.02
                else "VIOLATES >0.02"
                if max_quality_loss is not None
                else "UNAVAILABLE"
            )
        comparison_rows[route_name] = {
            "nmi_loss": nmi_loss,
            "ari_loss": ari_loss,
            "distance_increase": distance_increase,
            "runtime_delta": runtime_delta,
            "max_quality_loss": max_quality_loss,
            "criterion": criterion,
        }
        lines.append(
            f"| {route_name} | {_format(row.get('leaf_nmi'))} | {_format(nmi_loss)} | {_format(row.get('leaf_ari'))} | {_format(ari_loss)} | {_format(row.get('hierarchy_distance'))} | {_format(distance_increase)} | {_format(row.get('runtime_sec'), 2)} | {_format(runtime_delta, 2)} | {_format(max_quality_loss)} | {criterion} |"
        )
    guarded_comparison = comparison_rows.get(ROUTE_GUARDED, {})
    stable_comparison = comparison_rows.get(ROUTE_STABLE, {})
    guarded_violates = guarded_comparison.get("criterion") == "VIOLATES >0.02"
    stable_violates = stable_comparison.get("criterion") == "VIOLATES >0.02"
    if native_row is None or ROUTE_GUARDED not in comparison_rows or ROUTE_STABLE not in comparison_rows:
        conclusion = "Recommendation: result-based recommendation is unavailable because the main dataset does not contain all three route rows."
    elif guarded_violates and stable_violates:
        conclusion = (
            "Recommendation: retain `umap_hdbscan_native` as the baseline. "
            "Both `guarded_pca_hybrid` and `five_seed_stable` violate the documented ≤0.02 quality-loss criterion on the main dataset; their runtime differences do not override that quality failure."
        )
    else:
        eligible = [
            route_name
            for route_name, comparison in comparison_rows.items()
            if route_name != ROUTE_NATIVE and comparison.get("criterion") == "WITHIN ≤0.02"
        ]
        eligible_text = ", ".join(f"`{route}`" for route in eligible) or "no alternative"
        conclusion = (
            "Recommendation: keep `umap_hdbscan_native` as the reference baseline; "
            f"the alternatives within the documented quality-loss criterion are {eligible_text}. "
            "Use runtime and the metric deltas above to decide whether an eligible alternative merits an offline follow-up."
        )
    lines.extend(["", conclusion])
    lines.extend(
        [
            "",
            "## Scale rows",
            "",
            "| Dataset | Route | Runtime (s) | Leaf NMI | Leaf ARI | Noise | Clusters |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scale_rows:
        lines.append(
            f"| {row['dataset']} | {row['route']} | {_format(row.get('runtime_sec'), 2)} | {_format(row.get('leaf_nmi'))} | {_format(row.get('leaf_ari'))} | {_format(row.get('noise_ratio'))} | {_format(row.get('cluster_count'), 2)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A is the direct current discovery baseline. B is the low-overhead guarded candidate when its rejection improves trustworthiness without materially reducing quality. C is an offline stability/audit route; it is not placed in an online path because it intentionally costs five discovery fits. The experiment does not automatically replace production hierarchical PCA SFCM.",
            "",
            "## Artifacts",
            "",
            "- `report.json`: compact machine-readable report; no embeddings, coordinates, or point arrays.",
            "- `runs.csv`: discovery-fit and route rows.",
            "- `route-summary.csv`, `timing.csv`, `cluster-support.csv`, `seed-agreement.csv`, `selected-pca.json`.",
            "- `quality-runtime-pareto.png`, `stability-comparison.png`, `pca-support-vs-seed-agreement.png`, `rejection-quality-curve.png`, `scale-runtime.png`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(report: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT) -> None:
    """Write deterministic JSON/CSV/PNG/Markdown artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "selected-pca.json",
        {
            "experiment": EXPERIMENT_NAME,
            "pca_selection_once_per_dataset": True,
            "datasets": [
                {
                    "dataset": dataset["dataset"],
                    "selected_pca": dataset["selected_pca"],
                    "raw_pca_preservation": dataset["raw_pca_preservation"],
                }
                for dataset in report.get("datasets", [])
            ],
        },
    )
    _write_csv(output_dir / "runs.csv", report.get("runs", []))
    _write_csv(output_dir / "route-summary.csv", report.get("route_summary", []))
    _write_csv(output_dir / "timing.csv", report.get("timing", []))
    _write_csv(output_dir / "cluster-support.csv", report.get("cluster_support", []))
    _write_csv(output_dir / "seed-agreement.csv", report.get("seed_agreement", []))
    plot_paths = make_plots(report, output_dir)
    report["artifacts"] = {
        "report": "report.json",
        "runs": "runs.csv",
        "route_summary": "route-summary.csv",
        "timing": "timing.csv",
        "cluster_support": "cluster-support.csv",
        "seed_agreement": "seed-agreement.csv",
        "selected_pca": "selected-pca.json",
        **plot_paths,
        "markdown": "REPORT.md",
    }
    _write_json(output_dir / "report.json", report)
    (output_dir / "REPORT.md").write_text(_markdown_report(report), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the three-route Gemini UMAP/HDBSCAN clustering benchmark"
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--dataset-sample-size", type=int, default=DEFAULT_DATASET_SAMPLE_SIZE
    )
    parser.add_argument(
        "--dataset-sample-seed", type=int, default=DEFAULT_DATASET_SAMPLE_SEED
    )
    parser.add_argument(
        "--scale-sample-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SCALE_SAMPLE_SIZES),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-scale", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip optional scale runs; the main run still uses all five fixed seeds",
    )
    parser.add_argument("--pca-max-components", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument("--pca-min-components", type=int, default=DEFAULT_MIN_COMPONENTS)
    parser.add_argument("--pca-component-step", type=int, default=DEFAULT_COMPONENT_STEP)
    parser.add_argument("--minimum-preservation-gain", type=float, default=0.05)
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_full_benchmark(
        args.input_json,
        dataset_sample_size=args.dataset_sample_size,
        dataset_sample_seed=args.dataset_sample_seed,
        scale_sample_sizes=args.scale_sample_sizes,
        skip_scale=bool(args.skip_scale or args.fast),
        pca_max_components=args.pca_max_components,
        pca_min_components=args.pca_min_components,
        pca_component_step=args.pca_component_step,
        minimum_preservation_gain=args.minimum_preservation_gain,
        progress=args.progress,
    )
    write_artifacts(report, args.output_dir)
    print(f"Wrote three-route benchmark artifacts to {args.output_dir}")
    for row in report["route_summary"]:
        print(
            f"{row['dataset']} {row['route']}: "
            f"leaf_nmi={_format(row.get('leaf_nmi'))} "
            f"leaf_ari={_format(row.get('leaf_ari'))} "
            f"runtime_sec={_format(row.get('runtime_sec'), 2)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
