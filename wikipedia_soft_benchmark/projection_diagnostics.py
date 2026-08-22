"""Diagnostics for centered PCA versus uncentered SVD geometry.

The hierarchy benchmark uses discovery rows to fit its projection and then
uses projected discovery rows as the reference set for out-of-sample
calibration/test matching.  This module measures two separate effects of the
projection:

* preservation of the original BGE cosine neighbourhood/ordering; and
* whether the resulting neighbourhood has useful Wikipedia leaf geometry.

No calibration/test row is used while fitting a transformer.  The CLI writes
one deterministic JSON report and is intentionally independent of UMAP and
HDBSCAN so that projection geometry can be inspected in isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

from .embeddings import l2_normalize
from .hierarchy_benchmark import PROJECTION_MODES, load_metadata


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_NEIGHBOR_COUNT = 24


@dataclass(frozen=True)
class ProjectionFit:
    """A projection fitted exclusively on discovery embeddings."""

    transformer: PCA | TruncatedSVD
    mode: str
    seed: int
    discovery_features: np.ndarray


def _finite_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be a finite 2D matrix")
    if not len(matrix):
        raise ValueError("embeddings cannot be empty")
    return l2_normalize(matrix)


def fit_projection(
    discovery_embeddings: np.ndarray,
    *,
    mode: str,
    seed: int,
    n_components: int = 256,
) -> ProjectionFit:
    """Fit centered PCA or uncentered randomized TruncatedSVD on discovery."""

    matrix = _finite_matrix(discovery_embeddings)
    if mode not in PROJECTION_MODES:
        raise ValueError(f"mode must be one of {PROJECTION_MODES}")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    # PCA cannot have more components than n_samples - 1 after centering;
    # TruncatedSVD can use the full sample rank.
    limit = len(matrix) - 1 if mode == "centered-pca" else len(matrix)
    components = min(int(n_components), limit, matrix.shape[1])
    if components < 1:
        raise ValueError("discovery must contain at least two rows for PCA")
    if mode == "centered-pca":
        transformer: PCA | TruncatedSVD = PCA(
            n_components=components, svd_solver="full", random_state=int(seed)
        ).fit(matrix)
    else:
        transformer = TruncatedSVD(
            n_components=components,
            algorithm="randomized",
            n_iter=7,
            random_state=int(seed),
        ).fit(matrix)
    features = np.asarray(transformer.transform(matrix), dtype=np.float64)
    return ProjectionFit(transformer, mode, int(seed), features)


def _ids(metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [str(row.get("source_id", row.get("id", index))) for index, row in enumerate(metadata)],
        dtype=object,
    )


def _leaves(metadata: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([str(row.get("leaf", "")) for row in metadata], dtype=object)


def _cosine_features(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-14) or not np.all(np.isfinite(norms)):
        raise ValueError("projected features must have non-zero finite norms")
    return matrix / norms


def _exclude_mask(
    query_ids: Sequence[Any] | None,
    reference_ids: Sequence[Any] | None,
    n_queries: int,
    n_references: int,
) -> np.ndarray:
    mask = np.zeros((n_queries, n_references), dtype=bool)
    if query_ids is None or reference_ids is None:
        return mask
    if len(query_ids) != n_queries or len(reference_ids) != n_references:
        raise ValueError("query/reference IDs have incompatible lengths")
    reference = np.asarray([str(value) for value in reference_ids], dtype=object)
    for row_index, value in enumerate(query_ids):
        mask[row_index] = reference == str(value)
    return mask


def neighbor_indices(
    scores: np.ndarray,
    *,
    k: int,
    largest: bool,
    query_ids: Sequence[Any] | None = None,
    reference_ids: Sequence[Any] | None = None,
) -> np.ndarray:
    """Return deterministic top-k indices, excluding equal IDs when given."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite 2D matrix")
    n_queries, n_references = values.shape
    excluded = _exclude_mask(query_ids, reference_ids, n_queries, n_references)
    available = n_references - excluded.sum(axis=1)
    if k < 1 or np.any(available < k):
        raise ValueError("k exceeds available non-self references")
    work = values.copy()
    if largest:
        work[excluded] = -np.inf
        order = np.argsort(-work, axis=1, kind="mergesort")
    else:
        work[excluded] = np.inf
        order = np.argsort(work, axis=1, kind="mergesort")
    return order[:, : int(k)]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis without a scipy dependency."""

    matrix = np.asarray(values, dtype=np.float64)
    ranks = np.empty_like(matrix)
    for row_index, row in enumerate(matrix):
        order = np.argsort(row, kind="mergesort")
        sorted_values = row[order]
        start = 0
        while start < len(row):
            stop = start + 1
            while stop < len(row) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[row_index, order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
    return ranks


def _row_correlations(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("correlation matrices must have equal 2D shape")
    pearson = np.zeros(len(left), dtype=np.float64)
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
        pearson[index] = float(np.dot(a_centered, b_centered) / denominator) if denominator > 1e-14 else (1.0 if np.allclose(a, b) else 0.0)
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    spearman = np.zeros(len(left), dtype=np.float64)
    for index, (a, b) in enumerate(zip(left_ranks, right_ranks, strict=True)):
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
        spearman[index] = float(np.dot(a_centered, b_centered) / denominator) if denominator > 1e-14 else (1.0 if np.allclose(a, b) else 0.0)
    return pearson, spearman


def _recall(projected: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return np.asarray([len(set(a.tolist()) & set(b.tolist())) / len(b) for a, b in zip(projected, baseline, strict=True)], dtype=np.float64)


def _purity(neighbors: np.ndarray, query_leaves: np.ndarray, reference_leaves: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(reference_leaves[index] == query_leaves[row]) for row, index in enumerate(neighbors)], dtype=np.float64)


def _similarity_groups(scores: np.ndarray, query_leaves: np.ndarray, reference_leaves: np.ndarray) -> dict[str, Any]:
    same_mask = query_leaves[:, None] == reference_leaves[None, :]
    same = np.asarray(scores, dtype=np.float64)[same_mask]
    different = np.asarray(scores, dtype=np.float64)[~same_mask]
    same_mean = float(np.mean(same)) if len(same) else 0.0
    different_mean = float(np.mean(different)) if len(different) else 0.0
    return {
        "same_leaf_count": int(len(same)),
        "different_leaf_count": int(len(different)),
        "same_leaf_mean": same_mean,
        "different_leaf_mean": different_mean,
        "margin": same_mean - different_mean,
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "std": float(np.std(values)) if len(values) else 0.0,
        "min": float(np.min(values)) if len(values) else 0.0,
        "max": float(np.max(values)) if len(values) else 0.0,
    }


def evaluate_projection(
    original_query: np.ndarray,
    original_reference: np.ndarray,
    projected_query: np.ndarray,
    projected_reference: np.ndarray,
    query_metadata: Sequence[Mapping[str, Any]],
    reference_metadata: Sequence[Mapping[str, Any]],
    *,
    k: int = DEFAULT_NEIGHBOR_COUNT,
) -> dict[str, Any]:
    """Compare one projected space against original BGE OOS neighbours."""

    query = _finite_matrix(original_query).astype(np.float64)
    reference = _finite_matrix(original_reference).astype(np.float64)
    if len(query_metadata) != len(query) or len(reference_metadata) != len(reference):
        raise ValueError("metadata and embeddings must have equal row counts")
    query_ids = _ids(query_metadata)
    reference_ids = _ids(reference_metadata)
    query_leaves = _leaves(query_metadata)
    reference_leaves = _leaves(reference_metadata)
    original_cosine = query @ reference.T
    projected_cosine = _cosine_features(projected_query) @ _cosine_features(projected_reference).T
    projected_distances = np.sum(
        np.square(np.asarray(projected_query, dtype=np.float64)[:, None, :] - np.asarray(projected_reference, dtype=np.float64)[None, :, :]),
        axis=2,
    )
    baseline_neighbors = neighbor_indices(original_cosine, k=k, largest=True, query_ids=query_ids, reference_ids=reference_ids)
    projected_cosine_neighbors = neighbor_indices(projected_cosine, k=k, largest=True, query_ids=query_ids, reference_ids=reference_ids)
    projected_euclidean_neighbors = neighbor_indices(projected_distances, k=k, largest=False, query_ids=query_ids, reference_ids=reference_ids)
    baseline_euclidean_neighbors = neighbor_indices(
        np.sum(np.square(query[:, None, :] - reference[None, :, :]), axis=2),
        k=k,
        largest=False,
        query_ids=query_ids,
        reference_ids=reference_ids,
    )
    pearson, spearman = _row_correlations(original_cosine, projected_cosine)
    baseline_purity = _purity(baseline_neighbors, query_leaves, reference_leaves)
    projected_cosine_purity = _purity(projected_cosine_neighbors, query_leaves, reference_leaves)
    projected_euclidean_purity = _purity(projected_euclidean_neighbors, query_leaves, reference_leaves)
    baseline_groups = _similarity_groups(original_cosine, query_leaves, reference_leaves)
    projected_groups = _similarity_groups(projected_cosine, query_leaves, reference_leaves)
    return {
        "query_count": int(len(query)),
        "reference_count": int(len(reference)),
        "neighbor_count": int(k),
        "original_bge": {
            "pairwise_cosine": {"pearson": {"mean": 1.0, "std": 0.0}, "spearman": {"mean": 1.0, "std": 0.0}},
            "cosine_knn_recall_at_k": 1.0,
            "euclidean_knn_recall_at_k": float(np.mean(_recall(baseline_euclidean_neighbors, baseline_neighbors))),
            "neighbor_leaf_purity_at_k": _summary(baseline_purity),
            "same_vs_different_leaf_cosine": baseline_groups,
        },
        "projected": {
            "pairwise_cosine": {"pearson": _summary(pearson), "spearman": _summary(spearman)},
            "cosine_knn_recall_at_k": float(np.mean(_recall(projected_cosine_neighbors, baseline_neighbors))),
            "euclidean_knn_recall_at_k": float(np.mean(_recall(projected_euclidean_neighbors, baseline_neighbors))),
            "neighbor_leaf_purity_at_k": {
                "cosine": _summary(projected_cosine_purity),
                "euclidean": _summary(projected_euclidean_purity),
            },
            "same_vs_different_leaf_cosine": projected_groups,
        },
    }


def _validate_dataset(embeddings: np.ndarray, metadata: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    matrix = _finite_matrix(embeddings)
    if len(metadata) != len(matrix):
        raise ValueError("metadata and embeddings must have equal row counts")
    indices = {split: [i for i, row in enumerate(metadata) if row.get("split") == split] for split in ("discovery", "calibration", "test")}
    if any(not values for values in indices.values()):
        raise ValueError("dataset must contain discovery, calibration, and test rows")
    return indices


def run_diagnostics(
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_components: int = 256,
    k: int = DEFAULT_NEIGHBOR_COUNT,
) -> dict[str, Any]:
    """Run all seed/split comparisons with discovery-only fitting."""

    matrix = _finite_matrix(embeddings).astype(np.float64)
    rows = [dict(row) for row in metadata]
    split_indices = _validate_dataset(matrix, rows)
    discovery_indices = split_indices["discovery"]
    discovery = matrix[discovery_indices]
    discovery_rows = [rows[index] for index in discovery_indices]
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {"documents": int(len(matrix)), **{split: int(len(indices)) for split, indices in split_indices.items()}},
        "configuration": {"seeds": [int(seed) for seed in seeds], "pca_components": int(n_components), "neighbor_count": int(k), "fit_split": "discovery", "reference_split": "discovery", "original_metric": "cosine", "projected_euclidean_metric": "squared_euclidean"},
        "splits": {},
    }
    fits = {
        (mode, int(seed)): fit_projection(
            discovery, mode=mode, seed=int(seed), n_components=n_components
        )
        for mode in PROJECTION_MODES
        for seed in seeds
    }
    for split in ("calibration", "test"):
        split_indices_values = split_indices[split]
        query = matrix[split_indices_values]
        query_rows = [rows[index] for index in split_indices_values]
        split_result: dict[str, Any] = {"per_seed": {mode: {} for mode in PROJECTION_MODES}}
        for mode in PROJECTION_MODES:
            for seed in seeds:
                fit = fits[(mode, int(seed))]
                projected_query = np.asarray(fit.transformer.transform(query), dtype=np.float64)
                result = evaluate_projection(projected_query=projected_query, projected_reference=fit.discovery_features, original_query=query, original_reference=discovery, query_metadata=query_rows, reference_metadata=discovery_rows, k=k)
                split_result["per_seed"][mode][str(int(seed))] = result
            split_result["mean_by_mode"] = split_result.get("mean_by_mode", {})
            # Means are kept for scalar headline metrics; full per-seed results
            # retain the geometry distributions needed to inspect randomness.
            mode_results = split_result["per_seed"][mode]
            split_result["mean_by_mode"][mode] = _mean_result(mode_results)
        report["splits"][split] = split_result
    return report


def _mean_result(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    first = next(iter(results.values()))
    scalar_paths = (
        ("projected", "cosine_knn_recall_at_k"),
        ("projected", "euclidean_knn_recall_at_k"),
        ("projected", "pairwise_cosine", "pearson", "mean"),
        ("projected", "pairwise_cosine", "spearman", "mean"),
        ("projected", "neighbor_leaf_purity_at_k", "euclidean", "mean"),
        ("projected", "same_vs_different_leaf_cosine", "margin"),
    )
    output: dict[str, Any] = {"seed_count": len(results), "neighbor_count": first["neighbor_count"]}
    for path in scalar_paths:
        values = []
        for result in results.values():
            value: Any = result
            for key in path:
                value = value[key]
            values.append(float(value))
        cursor = output
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = float(np.mean(values))
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose PCA/SVD preservation of Wikipedia BGE geometry")
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--neighbor-count", type=int, default=DEFAULT_NEIGHBOR_COUNT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.embedding_dir
    embedding_path = root / "document_embeddings.npy"
    metadata_path = root / "document_metadata.jsonl"
    embeddings = np.load(embedding_path)
    metadata = load_metadata(metadata_path)
    report = run_diagnostics(embeddings, metadata, seeds=args.seeds, n_components=args.pca_components, k=args.neighbor_count)
    report["inputs"] = {"document_embeddings": embedding_path.name, "document_metadata": metadata_path.name, "document_embeddings_sha256": _sha256(embedding_path), "document_metadata_sha256": _sha256(metadata_path)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report["artifacts"] = {"report": output_path.name, "report_sha256": _sha256(output_path)}
    # The report checksum is intentionally printed/available in the process
    # result; embedding it inside report.json would make it self-referential.
    print(json.dumps(report["artifacts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
