"""Split-aware Wikipedia hierarchy benchmark.

Discovery is the only split used to fit PCA, UMAP, and HDBSCAN.  Calibration
and test rows are transformed with those fitted objects and receive
out-of-sample memberships.  The public functions are intentionally small so
that a report can be reproduced without loading the 1,925 chunk vectors.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
import tarfile
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import hdbscan
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score
from umap import UMAP

from pca_neighbor_search import PcaNeighborIndex, build_pca_neighbor_index

from .embeddings import l2_normalize
from .benchmark_helpers import write_gzip_csv

DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_MIN_CLUSTER_SIZES = (18, 24, 30)
DEFAULT_MIN_SAMPLES = (3, 5, 8)
DEFAULT_NEIGHBOR_COUNTS = (8, 15, 24)
PROJECTION_MODES = ("centered-pca", "uncentered-svd")


def _validate_rows(embeddings: np.ndarray, metadata: Sequence[Mapping[str, Any]], *, split: str | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be a finite 2D matrix")
    rows = [dict(row) for row in metadata]
    if len(rows) != len(matrix):
        raise ValueError("metadata and embeddings must have equal row counts")
    if split is not None:
        rows = [row for row in rows if row.get("split") == split]
        # Caller should pass a matching subset; silently dropping rows is a
        # common source of split leakage, so reject it here.
        if len(rows) != len(matrix):
            raise ValueError("metadata rows do not all belong to requested split")
    return l2_normalize(matrix), rows


def _majority(values: Iterable[str]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    if not counts:
        return None
    return sorted(counts, key=lambda key: (-counts[key], key))[0]


@dataclass
class DiscoveryState:
    # Kept under the historical ``pca`` name for API compatibility.  In
    # ``uncentered-svd`` mode this is a TruncatedSVD transformer.
    pca: PCA | TruncatedSVD
    umap: UMAP
    clusterer: Any
    pca_discovery: np.ndarray
    umap_discovery: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    cluster_to_leaf: dict[int, str]
    cluster_to_parent: dict[int, str]
    cluster_to_top: dict[int, str]
    discovery_metadata: list[dict[str, Any]]
    configuration: dict[str, Any]
    neighbor_index: PcaNeighborIndex | None = None

    @property
    def cluster_count(self) -> int:
        return int(max(self.labels, default=-1) + 1)


@dataclass
class _PreparedDiscoveryProjection:
    """Reusable discovery-side work shared by a calibration sweep.

    HDBSCAN's density parameters do not affect PCA, the PCA-space neighbor
    index, or UMAP.  Keeping these objects together makes that dependency
    explicit and prevents a calibration sweep from fitting them once per
    ``(min_cluster_size, min_samples)`` pair.
    """

    pca: PCA | TruncatedSVD
    umap: UMAP
    pca_discovery: np.ndarray
    umap_discovery: np.ndarray
    neighbor_index: PcaNeighborIndex
    discovery_metadata: list[dict[str, Any]]
    configuration: dict[str, Any]
    timing_sec: dict[str, float] | None = None


@dataclass
class CalibrationArtifacts:
    """Prepared objects retained from calibration for the selected candidate.

    Calibration normally returns the historical ``(rows, selected)`` pair.
    Callers that need to run the held-out test can opt into a third return
    value and reuse these objects, avoiding a second PCA/index/UMAP fit.
    """

    prepared_projection: _PreparedDiscoveryProjection
    selected_state: DiscoveryState
    timing_sec: dict[str, float]


@dataclass
class MembershipPrediction:
    native: np.ndarray
    exact_knn: np.ndarray
    native_unexplained: np.ndarray
    exact_unexplained: np.ndarray
    native_labels: np.ndarray
    exact_labels: np.ndarray
    pca_features: np.ndarray


@dataclass
class CalibrationResult:
    """Stable result object for :func:`calibration_sweep`.

    The object intentionally remains iterable (and subscriptable) so callers
    using the historical ``rows, selected = calibration_sweep(...)`` or
    ``rows, selected, artifacts = ...`` unpacking continue to work.
    """

    rows: list[dict[str, Any]]
    selected: Mapping[str, Any]
    artifacts: CalibrationArtifacts | None = None

    def __iter__(self):
        yield self.rows
        yield self.selected
        if self.artifacts is not None:
            yield self.artifacts

    def __len__(self) -> int:
        return 3 if self.artifacts is not None else 2

    def __getitem__(self, index: int):
        values = (self.rows, self.selected, self.artifacts)
        # Only the supported tuple positions are exposed.  In particular,
        # do not let Python's negative indexing reach the optional artifact
        # when this result has the historical two-value shape.
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return values[index]


def _normalize_memberships(raw: Any, n_rows: int, cluster_count: int) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if cluster_count == 0:
        return np.zeros((n_rows, 0), dtype=np.float64)
    if values.ndim == 1 and values.shape == (cluster_count,) and n_rows == 1:
        values = values[None, :]
    if values.shape != (n_rows, cluster_count):
        raise ValueError(f"membership shape {values.shape} != ({n_rows}, {cluster_count})")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-8) or np.any(values > 1 + 1e-8):
        raise ValueError("membership values must be finite and in [0,1]")
    return np.clip(values, 0.0, 1.0)


def _prepare_discovery_projection(
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    pca_components: int = 256,
    umap_components: int = 20,
    umap_n_neighbors: int = 15,
    projection_mode: str = "centered-pca",
    neighbor_backend: str = "exact",
    neighbor_max_k: int = max(DEFAULT_NEIGHBOR_COUNTS),
    neighbor_graph_neighbors: int = 32,
    neighbor_query_epsilon: float = 0.1,
) -> _PreparedDiscoveryProjection:
    matrix, rows = _validate_rows(embeddings, metadata)
    if any(row.get("split") not in (None, "discovery") for row in rows):
        raise ValueError("fit_discovery accepts discovery rows only")
    if len(rows) < 3:
        raise ValueError("discovery requires at least three rows")
    if projection_mode not in PROJECTION_MODES:
        raise ValueError(f"projection_mode must be one of {PROJECTION_MODES}")
    # PCA centers the discovery matrix before decomposition.  TruncatedSVD
    # deliberately does not center its input, preserving the raw normalized
    # embedding origin requested by the uncentered benchmark.
    sample_limit = matrix.shape[0] - 1 if projection_mode == "centered-pca" else matrix.shape[0]
    pca_dim = min(int(pca_components), sample_limit, matrix.shape[1])
    pca_dim = max(1, pca_dim)
    timing: dict[str, float] = {}
    pca_started = time.perf_counter()
    if projection_mode == "centered-pca":
        pca: PCA | TruncatedSVD = PCA(n_components=pca_dim, svd_solver="full", random_state=seed).fit(matrix)
    else:
        pca = TruncatedSVD(n_components=pca_dim, algorithm="randomized", n_iter=7, random_state=seed).fit(matrix)
    pca_features = np.asarray(pca.transform(matrix), dtype=np.float64)
    timing["pca_fit_transform_sec"] = float(time.perf_counter() - pca_started)
    index_started = time.perf_counter()
    neighbor_index = build_pca_neighbor_index(
        pca_features,
        backend=neighbor_backend,
        max_neighbors=min(int(neighbor_max_k), len(rows) - 1),
        graph_neighbors=int(neighbor_graph_neighbors),
        random_state=42,
        query_epsilon=float(neighbor_query_epsilon),
    )
    timing["neighbor_index_build_sec"] = float(time.perf_counter() - index_started)
    neighbors = min(max(2, int(umap_n_neighbors)), len(rows) - 1)
    u_dim = min(max(1, int(umap_components)), max(1, len(rows) - 2))
    umap_started = time.perf_counter()
    mapper = UMAP(n_components=u_dim, n_neighbors=neighbors, init="random", random_state=seed, n_jobs=1).fit(pca_features)
    umap_features = np.asarray(mapper.embedding_, dtype=np.float64)
    timing["umap_fit_sec"] = float(time.perf_counter() - umap_started)
    configuration = {
        "seed": int(seed),
        "projection_mode": projection_mode,
        "pca_components": int(pca_dim),
        "umap_components": int(u_dim),
        "umap_n_neighbors": int(neighbors),
        "cluster_selection_method": "leaf",
        "neighbor_backend": neighbor_backend,
        "neighbor_max_k": int(neighbor_index.max_neighbors),
        "neighbor_graph_neighbors": int(neighbor_index.graph_neighbors),
        "neighbor_random_state": 42,
        "neighbor_query_epsilon": float(neighbor_query_epsilon),
    }
    return _PreparedDiscoveryProjection(
        pca,
        mapper,
        pca_features,
        umap_features,
        neighbor_index,
        rows,
        configuration,
        timing,
    )


def _state_from_prepared_projection(
    prepared: _PreparedDiscoveryProjection,
    *,
    min_cluster_size: int,
    min_samples: int,
) -> DiscoveryState:
    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError("invalid HDBSCAN parameters")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean", cluster_selection_method="leaf", prediction_data=True).fit(prepared.umap_discovery)
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float64)
    count = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
    mappings: list[dict[int, str]] = [{}, {}, {}]
    fields = ("leaf", "parent", "top")
    for cluster in range(count):
        selected = [prepared.discovery_metadata[i] for i in range(len(prepared.discovery_metadata)) if labels[i] == cluster]
        for target, field in zip(mappings, fields):
            value = _majority(str(row.get(field, "")) for row in selected)
            if value is not None:
                target[cluster] = value
    configuration = dict(prepared.configuration)
    configuration.update({"min_cluster_size": int(min_cluster_size), "min_samples": int(min_samples)})
    return DiscoveryState(
        prepared.pca,
        prepared.umap,
        clusterer,
        prepared.pca_discovery,
        prepared.umap_discovery,
        labels,
        probabilities,
        mappings[0],
        mappings[1],
        mappings[2],
        prepared.discovery_metadata,
        configuration,
        prepared.neighbor_index,
    )


def fit_discovery(
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    min_cluster_size: int = 18,
    min_samples: int = 5,
    pca_components: int = 256,
    umap_components: int = 20,
    umap_n_neighbors: int = 15,
    projection_mode: str = "centered-pca",
    neighbor_backend: str = "exact",
    neighbor_max_k: int = max(DEFAULT_NEIGHBOR_COUNTS),
    neighbor_graph_neighbors: int = 32,
    neighbor_query_epsilon: float = 0.1,
) -> DiscoveryState:
    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError("invalid HDBSCAN parameters")
    prepared = _prepare_discovery_projection(
        embeddings,
        metadata,
        seed=seed,
        pca_components=pca_components,
        umap_components=umap_components,
        umap_n_neighbors=umap_n_neighbors,
        projection_mode=projection_mode,
        neighbor_backend=neighbor_backend,
        neighbor_max_k=neighbor_max_k,
        neighbor_graph_neighbors=neighbor_graph_neighbors,
        neighbor_query_epsilon=neighbor_query_epsilon,
    )
    return _state_from_prepared_projection(
        prepared,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )


def _neighbors_to_new(state: DiscoveryState, pca_features: np.ndarray, neighbor_count: int, *, neighbor_backend: str | None = None, neighbor_index: PcaNeighborIndex | None = None, neighbor_results: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    n = len(pca_features)
    if not 1 <= neighbor_count <= len(state.pca_discovery):
        raise ValueError("neighbor_count must be positive and no larger than discovery size")
    count = state.cluster_count
    if count == 0:
        return np.zeros((n, 0), dtype=np.float64)
    backend = neighbor_backend or getattr(state, "configuration", {}).get("neighbor_backend", "exact")
    index = neighbor_index or getattr(state, "neighbor_index", None)
    if index is None or index.backend != backend or neighbor_count > index.max_neighbors:
        index = build_pca_neighbor_index(state.pca_discovery, backend=backend, max_neighbors=max(neighbor_count, max(DEFAULT_NEIGHBOR_COUNTS)), random_state=42)
    if neighbor_results is None:
        distances, indices = index.query(pca_features, neighbor_count, exclude_self=False)
    else:
        distances, indices = neighbor_results
        distances = np.asarray(distances, dtype=np.float64)[:, :neighbor_count]
        indices = np.asarray(indices, dtype=np.int64)[:, :neighbor_count]
    positive = distances[distances > 0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    output = np.zeros((n, count), dtype=np.float64)
    for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        positive_row = row_distances[row_distances > 0]
        sigma = float(np.median(positive_row)) if positive_row.size else fallback
        sigma = sigma if np.isfinite(sigma) and sigma > 0 else fallback
        weights = np.exp(-np.square(row_distances / sigma))
        denominator = float(np.sum(weights))
        for weight, reference in zip(weights, row_indices, strict=True):
            label = int(state.labels[reference])
            if label >= 0:
                output[row_index, label] += float(weight * state.probabilities[reference])
        if denominator:
            output[row_index] /= denominator
    return np.clip(output, 0.0, 1.0)


def _labels_from_memberships(memberships: np.ndarray) -> np.ndarray:
    labels = np.argmax(memberships, axis=1).astype(np.int64) if memberships.shape[1] else np.full(len(memberships), -1, dtype=np.int64)
    labels[np.sum(memberships, axis=1) <= 0] = -1
    return labels


def predict_native_memberships(
    state: DiscoveryState, umap_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict native HDBSCAN memberships from already transformed rows."""
    transformed = np.asarray(umap_features, dtype=np.float64)
    if state.cluster_count:
        native = _normalize_memberships(
            hdbscan.prediction.membership_vector(state.clusterer, transformed),
            len(transformed), state.cluster_count,
        )
    else:
        native = np.zeros((len(transformed), 0), dtype=np.float64)
    unexplained = np.clip(1.0 - native.sum(axis=1), 0.0, 1.0)
    return native, unexplained, _labels_from_memberships(native)


def predict_knn_memberships(
    state: DiscoveryState,
    pca_features: np.ndarray,
    *,
    neighbor_count: int = 15,
    neighbor_backend: str | None = None,
    neighbor_index: PcaNeighborIndex | None = None,
    neighbor_results: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exact = _neighbors_to_new(
        state, np.asarray(pca_features, dtype=np.float64), neighbor_count,
        neighbor_backend=neighbor_backend, neighbor_index=neighbor_index,
        neighbor_results=neighbor_results,
    )
    unexplained = np.clip(1.0 - exact.sum(axis=1), 0.0, 1.0)
    return exact, unexplained, _labels_from_memberships(exact)


def predict_memberships(state: DiscoveryState, embeddings: np.ndarray, *, neighbor_count: int = 15, neighbor_backend: str | None = None, neighbor_index: PcaNeighborIndex | None = None, neighbor_results: tuple[np.ndarray, np.ndarray] | None = None, pca_features: np.ndarray | None = None, umap_features: np.ndarray | None = None) -> MembershipPrediction:
    matrix = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    if pca_features is None:
        pca_features = np.asarray(state.pca.transform(matrix), dtype=np.float64)
    else:
        pca_features = np.asarray(pca_features, dtype=np.float64)
        if pca_features.shape[0] != len(matrix):
            raise ValueError("pca_features and embeddings must have equal row counts")
    if umap_features is None:
        transformed = np.asarray(state.umap.transform(pca_features), dtype=np.float64)
    else:
        transformed = np.asarray(umap_features, dtype=np.float64)
        if transformed.shape[0] != len(matrix):
            raise ValueError("umap_features and embeddings must have equal row counts")
    native, native_unexplained, native_labels = predict_native_memberships(state, transformed)
    exact, exact_unexplained, exact_labels = predict_knn_memberships(
        state, pca_features, neighbor_count=neighbor_count,
        neighbor_backend=neighbor_backend, neighbor_index=neighbor_index,
        neighbor_results=neighbor_results,
    )
    return MembershipPrediction(native, exact, native_unexplained, exact_unexplained, native_labels, exact_labels, pca_features)


def _mapped(labels: np.ndarray, mapping: Mapping[int, str]) -> np.ndarray:
    return np.asarray([mapping.get(int(label), "__noise__") if label >= 0 else "__noise__" for label in labels], dtype=object)


def _metrics(true: Sequence[str], labels: np.ndarray, memberships: np.ndarray, mapping: Mapping[int, str], unexplained: np.ndarray, *, include_labels: bool = False) -> dict[str, Any]:
    truth = np.asarray([str(value) for value in true], dtype=object)
    mapped = _mapped(labels, mapping)
    coverage = float(np.mean(np.sum(memberships, axis=1) > 0)) if len(truth) else 0.0
    known = mapped != "__noise__"
    eval_pred = mapped[known]
    eval_true = truth[known]
    if len(eval_true):
        macro_f1 = float(f1_score(eval_true, eval_pred, average="macro", zero_division=0))
        balanced = float(balanced_accuracy_score(eval_true, eval_pred))
    else:
        macro_f1 = balanced = 0.0
    affinity: list[float] = []
    for row_index, actual in enumerate(truth):
        affinity.append(float(sum(memberships[row_index, cluster] for cluster, value in mapping.items() if value == actual)))
    result: dict[str, Any] = {"leaf_nmi": float(normalized_mutual_info_score(truth, mapped)), "leaf_ari": float(adjusted_rand_score(truth, mapped)), "non_noise_coverage": coverage, "mapped_macro_f1": macro_f1, "mapped_balanced_accuracy": balanced, "true_affinity": float(np.mean(affinity)) if affinity else 0.0, "unexplained_mass": float(np.mean(unexplained)) if len(unexplained) else 0.0}
    if include_labels:
        result["mapped_labels"] = mapped.tolist()
    return result


def hierarchy_distance(true_rows: Sequence[Mapping[str, Any]], labels: np.ndarray, mapping_leaf: Mapping[int, str], mapping_parent: Mapping[int, str], mapping_top: Mapping[int, str]) -> float:
    distances = []
    for row, label in zip(true_rows, labels, strict=True):
        if label < 0 or int(label) not in mapping_leaf:
            distances.append(3.0)
            continue
        if mapping_leaf[int(label)] == str(row.get("leaf")):
            distances.append(0.0)
        elif mapping_parent.get(int(label)) == str(row.get("parent")):
            distances.append(1.0)
        elif mapping_top.get(int(label)) == str(row.get("top")):
            distances.append(2.0)
        else:
            distances.append(3.0)
    return float(np.mean(distances)) if distances else 0.0


def evaluate_prediction(
    state: DiscoveryState,
    prediction: MembershipPrediction,
    metadata: Sequence[Mapping[str, Any]],
    *,
    include_labels: bool = False,
) -> dict[str, Any]:
    """Evaluate a prediction that has already been computed.

    Keeping prediction and metric calculation separate prevents callers that
    need labels for plotting from paying for a second HDBSCAN/kNN prediction.
    """
    rows = [dict(row) for row in metadata]
    row_count = len(rows)
    native = np.asarray(prediction.native)
    exact_knn = np.asarray(prediction.exact_knn)
    native_labels = np.asarray(prediction.native_labels)
    exact_labels = np.asarray(prediction.exact_labels)
    native_unexplained = np.asarray(prediction.native_unexplained)
    exact_unexplained = np.asarray(prediction.exact_unexplained)
    pca_features = np.asarray(prediction.pca_features)
    array_specs = (
        ("native", native, 2),
        ("exact_knn", exact_knn, 2),
        ("native_labels", native_labels, 1),
        ("exact_labels", exact_labels, 1),
        ("native_unexplained", native_unexplained, 1),
        ("exact_unexplained", exact_unexplained, 1),
        ("pca_features", pca_features, 2),
    )
    for name, value, ndim in array_specs:
        array = np.asarray(value)
        if array.ndim != ndim or array.shape[0] != row_count:
            raise ValueError(f"prediction.{name} must have {row_count} rows; got shape {array.shape}")
    if native.shape[1] != exact_knn.shape[1]:
        raise ValueError("native and exact_knn membership matrices must have equal cluster counts")
    if np.any(native_labels < -1) or np.any(native_labels >= native.shape[1]):
        raise ValueError("native labels contain a cluster outside membership columns")
    if np.any(exact_labels < -1) or np.any(exact_labels >= exact_knn.shape[1]):
        raise ValueError("exact_knn labels contain a cluster outside membership columns")
    prediction = MembershipPrediction(native, exact_knn, native_unexplained, exact_unexplained, native_labels, exact_labels, pca_features)
    true_leaf = [str(row.get("leaf", "")) for row in rows]
    true_parent = [str(row.get("parent", "")) for row in rows]
    true_top = [str(row.get("top", "")) for row in rows]
    result: dict[str, Any] = {
        "native": _metrics(true_leaf, prediction.native_labels, prediction.native, state.cluster_to_leaf, prediction.native_unexplained, include_labels=include_labels),
        "exact_knn": _metrics(true_leaf, prediction.exact_labels, prediction.exact_knn, state.cluster_to_leaf, prediction.exact_unexplained, include_labels=include_labels),
    }
    for name, labels, memberships, mapping, unexplained in (("native", prediction.native_labels, prediction.native, state.cluster_to_leaf, prediction.native_unexplained), ("exact_knn", prediction.exact_labels, prediction.exact_knn, state.cluster_to_leaf, prediction.exact_unexplained)):
        section = result[name]
        section["parent"] = _metrics(true_parent, labels, memberships, state.cluster_to_parent, unexplained, include_labels=include_labels)
        section["top"] = _metrics(true_top, labels, memberships, state.cluster_to_top, unexplained, include_labels=include_labels)
        section["hierarchy_distance"] = hierarchy_distance(rows, labels, state.cluster_to_leaf, state.cluster_to_parent, state.cluster_to_top)
    return result


def evaluate_split(state: DiscoveryState, embeddings: np.ndarray, metadata: Sequence[Mapping[str, Any]], *, neighbor_count: int, neighbor_backend: str | None = None, neighbor_index: PcaNeighborIndex | None = None, neighbor_results: tuple[np.ndarray, np.ndarray] | None = None, pca_features: np.ndarray | None = None, umap_features: np.ndarray | None = None, prediction: MembershipPrediction | None = None, include_labels: bool = True) -> dict[str, Any]:
    if prediction is not None:
        matrix = np.asarray(embeddings)
        if matrix.ndim != 2 or len(matrix) != len(metadata):
            raise ValueError("embeddings and metadata must have equal row counts")
        rows = [dict(row) for row in metadata]
    else:
        matrix, rows = _validate_rows(embeddings, metadata)
    if prediction is None:
        prediction = predict_memberships(state, matrix, neighbor_count=neighbor_count, neighbor_backend=neighbor_backend, neighbor_index=neighbor_index, neighbor_results=neighbor_results, pca_features=pca_features, umap_features=umap_features)
    return evaluate_prediction(state, prediction, rows, include_labels=include_labels)


def choose_calibration(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not results:
        raise ValueError("calibration results cannot be empty")
    # Rounded values give a meaningful deterministic tie group despite tiny
    # BLAS/UMAP floating point variation.
    return sorted(results, key=lambda item: (-round(float(item["mean_leaf_nmi"]), 12), float(item["mean_noise_rate"]), int(item["complexity"]), tuple(item["sort_key"])))[0]


def _calibration_group_core(
    arguments: tuple[
        int,
        tuple[int, ...],
        tuple[int, ...],
        np.ndarray,
        list[dict[str, Any]],
        np.ndarray,
        list[dict[str, Any]],
        tuple[int, ...],
        int,
        int,
        int,
        str,
        str,
        int,
        float,
    ],
    *,
    collect_artifacts: bool = False,
) -> Any:
    """Prepare one projection per seed, then fit every HDBSCAN combination.

    The projection and the maximum-k calibration query are independent of
    HDBSCAN's density parameters.  They are consequently computed once for
    this seed and shared by all ``min_cluster_size × min_samples`` states.
    """
    (
        seed,
        min_cluster_sizes,
        min_samples_values,
        discovery_embeddings,
        discovery_metadata,
        calibration_embeddings,
        calibration_metadata,
        neighbor_counts,
        pca_components,
        umap_components,
        umap_n_neighbors,
        projection_mode,
        neighbor_backend,
        neighbor_graph_neighbors,
        neighbor_query_epsilon,
    ) = arguments
    prepared = _prepare_discovery_projection(
        discovery_embeddings,
        discovery_metadata,
        seed=seed,
        pca_components=pca_components,
        umap_components=umap_components,
        umap_n_neighbors=umap_n_neighbors,
        projection_mode=projection_mode,
        neighbor_backend=neighbor_backend,
        neighbor_max_k=max(neighbor_counts),
        neighbor_graph_neighbors=neighbor_graph_neighbors,
        neighbor_query_epsilon=neighbor_query_epsilon,
    )
    timing = dict(getattr(prepared, "timing_sec", None) or {})
    calibration_matrix, _ = _validate_rows(calibration_embeddings, calibration_metadata)
    started = time.perf_counter()
    calibration_pca = np.asarray(prepared.pca.transform(calibration_matrix), dtype=np.float64)
    timing["calibration_pca_transform_sec"] = float(time.perf_counter() - started)
    started = time.perf_counter()
    calibration_umap = np.asarray(prepared.umap.transform(calibration_pca), dtype=np.float64)
    timing["calibration_umap_transform_sec"] = float(time.perf_counter() - started)
    started = time.perf_counter()
    index_max_neighbors = int(getattr(prepared.neighbor_index, "max_neighbors", len(discovery_metadata) - 1))
    calibration_neighbor_results = prepared.neighbor_index.query(
        calibration_pca,
        min(max(neighbor_counts), len(discovery_metadata), index_max_neighbors),
        exclude_self=False,
    )
    timing["calibration_neighbor_query_sec"] = float(time.perf_counter() - started)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: Any | None = None
    hdbscan_fit_sec = native_prediction_sec = knn_membership_sec = metric_evaluation_sec = 0.0
    for min_size, min_samples in itertools.product(min_cluster_sizes, min_samples_values):
        fit_started = time.perf_counter()
        state = _state_from_prepared_projection(
            prepared,
            min_cluster_size=min_size,
            min_samples=min_samples,
        )
        hdbscan_fit_sec += time.perf_counter() - fit_started
        labels = np.asarray(getattr(state, "labels", []))
        noise_rate = float(np.mean(labels < 0)) if len(labels) else 0.0
        native_prediction: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if hasattr(state, "cluster_count"):
            native_started = time.perf_counter()
            native_prediction = predict_native_memberships(state, calibration_umap)
            native_prediction_sec += time.perf_counter() - native_started
        # One maximum-k query is shared by all calibration candidates.  The
        # prefixes below avoid three ANN (or brute-force) queries for k=8/15/24.
        for neighbors in neighbor_counts:
            effective_count = min(neighbors, len(discovery_metadata), index_max_neighbors)
            prediction = None
            if native_prediction is not None:
                knn_started = time.perf_counter()
                exact, exact_unexplained, exact_labels = predict_knn_memberships(
                    state, calibration_pca, neighbor_count=effective_count,
                    neighbor_backend=neighbor_backend,
                    neighbor_results=calibration_neighbor_results,
                )
                knn_membership_sec += time.perf_counter() - knn_started
                prediction = MembershipPrediction(
                    native_prediction[0], exact, native_prediction[1], exact_unexplained,
                    native_prediction[2], exact_labels, calibration_pca,
                )
            metrics_started = time.perf_counter()
            evaluation = evaluate_split(
                state,
                calibration_embeddings,
                calibration_metadata,
                neighbor_count=effective_count,
                neighbor_backend=neighbor_backend,
                neighbor_results=calibration_neighbor_results,
                pca_features=calibration_pca,
                umap_features=calibration_umap,
                prediction=prediction,
                include_labels=False,
            )
            metric_evaluation_sec += time.perf_counter() - metrics_started
            native_nmi = float(evaluation["native"]["leaf_nmi"])
            knn_nmi = float(evaluation["exact_knn"]["leaf_nmi"])
            knn_key = f"{neighbor_backend}_leaf_nmi"
            row = {
                    "seed": int(seed),
                    "min_cluster_size": int(min_size),
                    "min_samples": int(min_samples),
                    "neighbor_count": int(neighbors),
                    "native_leaf_nmi": native_nmi,
                    "exact_knn_leaf_nmi": knn_nmi,
                    knn_key: knn_nmi,
                    "neighbor_backend": neighbor_backend,
                    "mean_leaf_nmi": (native_nmi + knn_nmi) / 2.0,
                    "mean_noise_rate": noise_rate,
                    "complexity": int(min_size + min_samples + neighbors),
                    "sort_key": [int(seed), int(min_size), int(min_samples), int(neighbors)],
                }
            rows.append(row)
            if best_row is None or choose_calibration((best_row, row)) is row:
                best_row = row
                best_state = state
    timing["hdbscan_fit_candidates_sec"] = float(hdbscan_fit_sec)
    timing["native_prediction_sec"] = float(native_prediction_sec)
    timing["knn_membership_sec"] = float(knn_membership_sec)
    timing["metric_evaluation_sec"] = float(metric_evaluation_sec)
    timing["hdbscan_calibration_sec"] = float(hdbscan_fit_sec + native_prediction_sec + knn_membership_sec + metric_evaluation_sec)
    if collect_artifacts:
        return rows, prepared, best_state, timing
    return rows


def _calibration_group(arguments: tuple[Any, ...]) -> list[dict[str, Any]]:
    return _calibration_group_core(arguments, collect_artifacts=False)


def _calibration_group_artifacts(arguments: tuple[Any, ...]) -> tuple[Any, ...]:
    return _calibration_group_core(arguments, collect_artifacts=True)


def calibration_sweep(
    discovery_embeddings: np.ndarray,
    discovery_metadata: Sequence[Mapping[str, Any]],
    calibration_embeddings: np.ndarray,
    calibration_metadata: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    min_cluster_sizes: Sequence[int] = DEFAULT_MIN_CLUSTER_SIZES,
    min_samples_values: Sequence[int] = DEFAULT_MIN_SAMPLES,
    neighbor_counts: Sequence[int] = DEFAULT_NEIGHBOR_COUNTS,
    pca_components: int = 256,
    umap_components: int = 20,
    umap_n_neighbors: int = 15,
    projection_mode: str = "centered-pca",
    neighbor_backend: str = "exact",
    neighbor_graph_neighbors: int = 32,
    neighbor_query_epsilon: float = 0.1,
    jobs: int = 1,
    return_prepared: bool = False,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]] | tuple[list[dict[str, Any]], Mapping[str, Any], CalibrationArtifacts]:
    """Run calibration with one projection per seed and one HDBSCAN fit per
    ``(seed, min_cluster_size, min_samples)`` configuration.

    Neighbor counts only affect out-of-sample membership evaluation.  They
    therefore share the discovery fit and one maximum-k query.  Process
    executor ``map`` preserves group order, keeping serial and parallel output
    deterministic.
    """
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if return_prepared and jobs != 1:
        raise ValueError("return_prepared requires jobs=1 so fitted objects stay in the caller")
    seed_values = tuple(int(value) for value in seeds)
    size_values = tuple(int(value) for value in min_cluster_sizes)
    sample_values = tuple(int(value) for value in min_samples_values)
    neighbor_values = tuple(int(value) for value in neighbor_counts)
    discovery_rows = [dict(row) for row in discovery_metadata]
    calibration_rows = [dict(row) for row in calibration_metadata]
    # A worker owns one seed/projection and sweeps all density combinations.
    # This is also the unit of parallelism: no worker repeats PCA, ANN index,
    # or UMAP merely because a different HDBSCAN configuration is evaluated.
    groups = [
        (
            seed,
            size_values,
            sample_values,
            np.asarray(discovery_embeddings),
            discovery_rows,
            np.asarray(calibration_embeddings),
            calibration_rows,
            neighbor_values,
            int(pca_components),
            int(umap_components),
            int(umap_n_neighbors),
            str(projection_mode),
            str(neighbor_backend),
            int(neighbor_graph_neighbors),
            float(neighbor_query_epsilon),
        )
        for seed in seed_values
    ]
    if return_prepared:
        # Process one seed at a time.  Only the current global winner's
        # projection/state is retained; previous groups become collectible as
        # soon as a later group wins.
        rows: list[dict[str, Any]] = []
        best_row: Mapping[str, Any] | None = None
        best_prepared: _PreparedDiscoveryProjection | None = None
        best_state: DiscoveryState | Any | None = None
        best_timing: dict[str, float] = {}
        for group in groups:
            group_rows, prepared, group_state, timing = _calibration_group_artifacts(group)
            rows.extend(group_rows)
            group_best = choose_calibration(group_rows)
            if best_row is None or choose_calibration((best_row, group_best)) is group_best:
                best_row = group_best
                best_prepared = prepared
                best_state = group_state
                best_timing = dict(timing)
        selected = choose_calibration(rows)
        if best_prepared is None or best_state is None or best_row is None:
            raise RuntimeError("calibration produced no retained artifact")
        prepared = best_prepared
        selected_state = best_state
        # The group retains only its best state.  If the global winner is in
        # that group, it is necessarily the retained state because the same
        # deterministic ordering is used by choose_calibration.
        state_configuration = getattr(selected_state, "configuration", {})
        if state_configuration and (int(state_configuration.get("min_cluster_size", -1)) != int(selected["min_cluster_size"]) or
                int(state_configuration.get("min_samples", -1)) != int(selected["min_samples"])):
            raise RuntimeError("calibration artifact state does not match selected configuration")
        artifacts = CalibrationArtifacts(prepared, selected_state, best_timing)
        return CalibrationResult(rows, selected, artifacts)
    if jobs == 1 or len(groups) <= 1:
        grouped_rows = [_calibration_group(group) for group in groups]
    else:
        # Python 3.14 defaults to ``forkserver`` on this Linux environment;
        # that start method requires a filesystem socket and is unavailable
        # in some restricted runners.  ``fork`` is safe here because workers
        # only receive immutable NumPy inputs and run independent fits, while
        # keeping the execution bounded by the requested worker count.
        with ProcessPoolExecutor(max_workers=int(jobs), mp_context=mp.get_context("fork")) as executor:
            grouped_rows = list(executor.map(_calibration_group, groups))
    rows = [row for group in grouped_rows for row in group]
    return rows, choose_calibration(rows)


def chunk_auxiliary_analysis(chunk_embeddings: np.ndarray, chunk_metadata: Sequence[Mapping[str, Any]], document_embeddings: np.ndarray, document_metadata: Sequence[Mapping[str, Any]], document_leaf_predictions: Mapping[str, str] | None = None, method_predictions: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, Any]:
    chunks = l2_normalize(np.asarray(chunk_embeddings, dtype=np.float32))
    docs = l2_normalize(np.asarray(document_embeddings, dtype=np.float32))
    by_source: dict[str, list[int]] = {}
    for i, row in enumerate(chunk_metadata):
        by_source.setdefault(str(row.get("source_id")), []).append(i)
    doc_index = {str(row.get("source_id", row.get("id"))): i for i, row in enumerate(document_metadata)}
    article_cosines: list[float] = []
    variances: list[float] = []
    label_matches: list[float] = []
    for source, indices in by_source.items():
        if source not in doc_index:
            continue
        center = docs[doc_index[source]]
        values = chunks[indices]
        article_cosines.extend((values @ center).tolist())
        variances.append(float(np.mean(np.square(values - np.mean(values, axis=0)))))
        # Dataset consistency: every chunk should retain its document's leaf.
        # An optional prediction map can override the target for callers that
        # explicitly want prediction agreement instead.
        target = str(document_metadata[doc_index[source]].get("leaf", ""))
        if document_leaf_predictions is not None:
            target = str(document_leaf_predictions.get(source, target))
        label_matches.extend(
            float(str(chunk_metadata[i].get("leaf", "")) == target)
            for i in indices
        )
    near: list[float] = []
    far: list[float] = []
    for left in range(len(docs)):
        for right in range(left + 1, len(docs)):
            similarity = float(np.dot(docs[left], docs[right]))
            if str(document_metadata[left].get("leaf")) == str(document_metadata[right].get("leaf")):
                near.append(similarity)
            else:
                far.append(similarity)
    boundary_cases: list[dict[str, Any]] = []
    if method_predictions and len(method_predictions) >= 2:
        methods = sorted(method_predictions)
        first, second = method_predictions[methods[0]], method_predictions[methods[1]]
        sources = sorted(set(first) & set(second))
        for source in sources:
            if first[source] != second[source]:
                boundary_cases.append({"source_id": source, methods[0]: first[source], methods[1]: second[source]})
    return {"chunk_count": int(len(chunks)), "document_count": int(len(docs)), "article_chunk_cosine": {"mean": float(np.mean(article_cosines)) if article_cosines else 0.0, "min": float(np.min(article_cosines)) if article_cosines else 0.0}, "within_document_embedding_variance": float(np.mean(variances)) if variances else 0.0, "document_label_chunk_match_rate": float(np.mean(label_matches)) if label_matches else 0.0, "near_leaf_similarity": {"count": len(near), "mean": float(np.mean(near)) if near else 0.0}, "far_leaf_similarity": {"count": len(far), "mean": float(np.mean(far)) if far else 0.0}, "boundary_case_count": len(boundary_cases), "boundary_cases": boundary_cases}


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write row assignments with a stable gzip header for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run split-aware Wikipedia BGE hierarchy benchmark")
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--umap-components", type=int, default=20)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--projection-mode", choices=PROJECTION_MODES, default="centered-pca", help="discovery projection: centered PCA (default) or uncentered TruncatedSVD")
    parser.add_argument("--neighbor-backend", choices=("exact", "pynndescent"), default="exact")
    parser.add_argument("--neighbor-graph-neighbors", type=int, default=32)
    parser.add_argument("--neighbor-query-epsilon", type=float, default=0.1)
    parser.add_argument("--jobs", type=int, default=1, help="maximum parallel discovery fits")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.embedding_dir
    embeddings = np.load(root / "document_embeddings.npy")
    metadata = load_metadata(root / "document_metadata.jsonl")
    split_indices = {split: [i for i, row in enumerate(metadata) if row.get("split") == split] for split in ("discovery", "calibration", "test")}
    if any(not value for value in split_indices.values()):
        raise ValueError("document metadata must contain discovery, calibration, and test rows")
    discovery_meta = [metadata[i] for i in split_indices["discovery"]]
    calibration_meta = [metadata[i] for i in split_indices["calibration"]]
    test_meta = [metadata[i] for i in split_indices["test"]]
    discovery = embeddings[split_indices["discovery"]]
    calibration = embeddings[split_indices["calibration"]]
    test = embeddings[split_indices["test"]]
    if args.jobs != 1:
        raise ValueError("--jobs > 1 is not supported by the CLI when reusing the selected calibration state; use calibration_sweep(..., return_prepared=False) for parallel calibration")
    # Keep the selected state and prepared projection from calibration.  This
    # avoids refitting PCA, the neighbor index, UMAP, and the selected HDBSCAN
    # state before the held-out test.  The historical two-value API remains
    # available to library callers.
    sweep, selected, artifacts = calibration_sweep(discovery, discovery_meta, calibration, calibration_meta, pca_components=args.pca_components, umap_components=args.umap_components, umap_n_neighbors=args.umap_n_neighbors, projection_mode=args.projection_mode, neighbor_backend=args.neighbor_backend, neighbor_graph_neighbors=args.neighbor_graph_neighbors, neighbor_query_epsilon=args.neighbor_query_epsilon, jobs=args.jobs, return_prepared=True)
    state = artifacts.selected_state
    effective_neighbors = min(int(selected["neighbor_count"]), len(discovery_meta))
    test_prediction = predict_memberships(state, test, neighbor_count=effective_neighbors, neighbor_backend=args.neighbor_backend)
    test_result = evaluate_prediction(state, test_prediction, test_meta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = []
    for index, row in enumerate(test_meta):
        assignments.append({"row_index": index, "source_id": row.get("source_id", row.get("id", index)), "leaf": row.get("leaf"), "parent": row.get("parent"), "top": row.get("top"), "native_recommended_cluster": int(test_prediction.native_labels[index]), "exact_knn_recommended_cluster": int(test_prediction.exact_labels[index]), "native_unexplained_mass": float(test_prediction.native_unexplained[index]), "exact_knn_unexplained_mass": float(test_prediction.exact_unexplained[index]), "native_max_affinity": float(np.max(test_prediction.native[index])) if test_prediction.native.shape[1] else 0.0, "exact_knn_max_affinity": float(np.max(test_prediction.exact_knn[index])) if test_prediction.exact_knn.shape[1] else 0.0})
    assignments_path = args.output_dir / "assignments.jsonl.gz"
    _write_gzip_jsonl(assignments_path, assignments)
    assignments_csv_path = args.output_dir / "assignments.csv.gz"
    csv_assignments = [
        {**row,
         "native_leaf": state.cluster_to_leaf.get(int(row["native_recommended_cluster"]), "__noise__") if int(row["native_recommended_cluster"]) >= 0 else "__noise__",
         "exact_knn_leaf": state.cluster_to_leaf.get(int(row["exact_knn_recommended_cluster"]), "__noise__") if int(row["exact_knn_recommended_cluster"]) >= 0 else "__noise__"}
        for row in assignments
    ]
    write_gzip_csv(assignments_csv_path, csv_assignments)
    calibration_runs_path = args.output_dir / "calibration_runs.jsonl"
    _write_jsonl(calibration_runs_path, sweep)
    runs_path = args.output_dir / "runs.csv"
    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "min_cluster_size", "min_samples", "neighbor_count", "native_leaf_nmi", "exact_knn_leaf_nmi", "mean_leaf_nmi", "mean_noise_rate", "complexity"])
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in sweep])
    auxiliary = None
    chunk_embeddings_path = root / "chunk_embeddings.npy"
    chunk_metadata_path = root / "chunk_metadata.jsonl"
    if chunk_embeddings_path.exists() and chunk_metadata_path.exists():
        method_predictions = {
            "native": {
                str(row.get("source_id", row.get("id", index))): state.cluster_to_leaf.get(
                    int(test_prediction.native_labels[index]), "__noise__"
                )
                for index, row in enumerate(test_meta)
            },
            "exact_knn": {
                str(row.get("source_id", row.get("id", index))): state.cluster_to_leaf.get(
                    int(test_prediction.exact_labels[index]), "__noise__"
                )
                for index, row in enumerate(test_meta)
            },
        }
        auxiliary = chunk_auxiliary_analysis(
            np.load(chunk_embeddings_path),
            load_metadata(chunk_metadata_path),
            embeddings,
            metadata,
            method_predictions=method_predictions,
        )
    file_artifacts = {
        "assignments": assignments_path.name,
        "assignments_csv": assignments_csv_path.name,
        "calibration_runs": calibration_runs_path.name,
        "runs": runs_path.name,
    }
    checksums = {
        name: _sha256_file(args.output_dir / filename)
        for name, filename in file_artifacts.items()
    }
    report = {"schema_version": 1, "dataset": {"documents": len(embeddings), "discovery": len(discovery), "calibration": len(calibration), "test": len(test)}, "selected_configuration": dict(selected), "calibration": sweep, "test": test_result, "chunk_auxiliary_analysis": auxiliary, "configuration": state.configuration, "artifacts": file_artifacts, "artifact_sha256": checksums, "timing_sec": dict(artifacts.timing_sec), "reuse": {"calibration_projection_reused_for_test": True, "selected_hdbscan_state_reused_for_test": True}}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
