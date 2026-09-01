"""Evaluate HDBSCAN after the fixed PCA/UMAP neighborhood experiment.

This is the second-stage companion to
``benchmark_pca_umap_neighbor_preservation.py``.  It reuses one automatic PCA
selection and compares a PCA-only control with a small, auditable set of UMAP
configurations selected from the first-stage report.  HDBSCAN parameters stay
fixed so the experiment measures the effect of the discovery projection rather
than a joint UMAP/HDBSCAN search.

The repository's Gemini dataset is the default input.  The dataset contains
``gemini-embedding-001`` CLUSTERING embeddings with 3072 dimensions and a
three-level DBpedia hierarchy in its metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from embedding_data import load_embeddings_from_json
from benchmark_pca_umap_neighbor_preservation import (
    DEFAULT_K_VALUES,
    DEFAULT_UMAP_SEEDS as PHASE1_DEFAULT_UMAP_SEEDS,
    PRODUCTION_BASELINE,
    mean_neighbor_preservation,
    neighbors_by_k,
    prepare_experiment,
)


SCHEMA_VERSION = 1
DEFAULT_INPUT = Path("dbpedia_gemini_embeddings.json.gz")
DEFAULT_PHASE1_REPORT = Path(
    "benchmarks/pca-umap-neighbor-preservation/report.json"
)
DEFAULT_OUTPUT = Path("benchmarks/pca-umap-hdbscan-followup")
DEFAULT_DATASET_SAMPLE_SIZE = 720
DEFAULT_DATASET_SAMPLE_SEED = 42
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 3
DEFAULT_UMAP_SEEDS = PHASE1_DEFAULT_UMAP_SEEDS

HDBSCAN_CONFIGURATION: dict[str, Any] = {
    "min_cluster_size": DEFAULT_MIN_CLUSTER_SIZE,
    "min_samples": DEFAULT_MIN_SAMPLES,
    "metric": "euclidean",
    "cluster_selection_method": "leaf",
    "prediction_data": True,
}

HDBSCANClass = Any
UMAPClass = Any

SCALAR_CLUSTER_METRICS = (
    "leaf_nmi",
    "leaf_ari",
    "parent_nmi",
    "parent_ari",
    "top_nmi",
    "top_ari",
    "hierarchy_distance",
    "clusters",
    "cluster_count",
    "noise_ratio",
    "silhouette",
    "mean_probability",
    "mean_outlier_score",
)

STABILITY_CSV_FIELDS = (
    "n_neighbors",
    "n_components",
    "min_dist",
    "roles",
    "seed_a",
    "seed_b",
    "cluster_ari",
    "cluster_nmi",
    "noise_jaccard",
    "cluster_count_a",
    "cluster_count_b",
    "cluster_count_abs_delta",
    "umap_neighbor_reproducibility_k15",
)


def _json_safe(value: Any) -> Any:
    """Convert NumPy/path values and non-finite floats for strict JSON."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
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
        {key for row in rows for key in row}
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


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _std(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.std(values))


def _configuration_key(configuration: Mapping[str, Any]) -> tuple[int, int, float]:
    return (
        int(configuration["n_neighbors"]),
        int(configuration["n_components"]),
        round(float(configuration["min_dist"]), 12),
    )


def _same_configuration(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return _configuration_key(left) == _configuration_key(right)


def load_gemini_dataset(
    input_path: Path,
    *,
    sample_size: int | None = DEFAULT_DATASET_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_DATASET_SAMPLE_SEED,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load Gemini embeddings and return aligned rows plus source indices."""

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
    if len(selected_metadata) != len(selected_embeddings):
        raise ValueError("Gemini embeddings and metadata are not aligned")
    return selected_embeddings, selected_metadata, indices


def _fingerprints(
    normalized_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    source_indices: np.ndarray,
) -> dict[str, Any]:
    """Create reproducibility fingerprints for the exact sampled dataset."""

    normalized = np.ascontiguousarray(normalized_embeddings, dtype=np.float32)
    metadata_payload = metadata.to_dict(orient="records")
    metadata_bytes = json.dumps(
        _json_safe(metadata_payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "rows": int(normalized.shape[0]),
        "embedding_dimension": int(normalized.shape[1]),
        "sample_indices_sha256": hashlib.sha256(
            np.asarray(source_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "normalized_embeddings_sha256": hashlib.sha256(
            normalized.tobytes()
        ).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }


def _phase1_metric_mean(row: Mapping[str, Any], metric: str, k_values: Sequence[int]) -> float | None:
    direct = row.get(f"mean_{metric}")
    if direct is not None:
        return float(direct)
    values = [
        row.get(f"{metric}_k{int(k)}_mean")
        for k in k_values
        if row.get(f"{metric}_k{int(k)}_mean") is not None
    ]
    return _mean(values)


def _add_role(
    record: dict[str, Any],
    role: str,
) -> None:
    roles = list(record.get("roles", []))
    if role not in roles:
        roles.append(role)
    record["roles"] = roles


def select_configurations(
    phase1_report: Mapping[str, Any],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> list[dict[str, Any]]:
    """Select baseline, extrema, and one-factor ablations from phase 1."""

    phase1_summary = list(phase1_report.get("summary", []))
    if not phase1_summary:
        raise ValueError("phase 1 report has no summary configurations")

    def usable(row: Mapping[str, Any], metric: str) -> bool:
        return _phase1_metric_mean(row, metric, k_values) is not None

    baseline = next(
        (
            row
            for row in phase1_summary
            if _same_configuration(row, PRODUCTION_BASELINE)
        ),
        None,
    )
    if baseline is None:
        raise ValueError("phase 1 report does not contain the production baseline")

    raw_rows = [row for row in phase1_summary if usable(row, "raw_umap")]
    pca_rows = [row for row in phase1_summary if usable(row, "pca_umap")]
    if not raw_rows or not pca_rows:
        raise ValueError("phase 1 report lacks Raw↔UMAP or PCA↔UMAP summary values")
    best_raw = max(
        raw_rows,
        key=lambda row: _phase1_metric_mean(row, "raw_umap", k_values) or -1.0,
    )
    best_pca = max(
        pca_rows,
        key=lambda row: _phase1_metric_mean(row, "pca_umap", k_values) or -1.0,
    )
    negative = min(
        raw_rows,
        key=lambda row: _phase1_metric_mean(row, "raw_umap", k_values) or 2.0,
    )

    by_key = {_configuration_key(row): row for row in phase1_summary}
    requested_roles: list[tuple[str, Mapping[str, Any]]] = [
        ("production_baseline", baseline),
        ("best_raw_preservation", best_raw),
        ("best_pca_preservation", best_pca),
    ]
    for role, key in (
        ("n_neighbors_ablation", (100, 20, 0.1)),
        ("n_components_ablation", (15, 5, 0.1)),
        ("min_dist_ablation", (15, 20, 0.5)),
    ):
        candidate = by_key.get(key)
        if candidate is not None:
            requested_roles.append((role, candidate))
    requested_roles.append(("negative_control", negative))

    selected: dict[tuple[int, int, float], dict[str, Any]] = {}
    for role, row in requested_roles:
        key = _configuration_key(row)
        if key not in selected:
            selected[key] = {
                "n_neighbors": int(row["n_neighbors"]),
                "n_components": int(row["n_components"]),
                "min_dist": float(row["min_dist"]),
                "roles": [],
                "phase1_mean_raw_umap": _phase1_metric_mean(
                    row, "raw_umap", k_values
                ),
                "phase1_mean_pca_umap": _phase1_metric_mean(
                    row, "pca_umap", k_values
                ),
                "phase1_mean_umap_additional_loss": _phase1_metric_mean(
                    row, "umap_additional_loss", k_values
                ),
            }
        _add_role(selected[key], role)

    priority = {
        "production_baseline": 0,
        "best_raw_preservation": 1,
        "best_pca_preservation": 2,
        "n_neighbors_ablation": 3,
        "n_components_ablation": 4,
        "min_dist_ablation": 5,
        "negative_control": 6,
    }
    return sorted(
        selected.values(),
        key=lambda row: min(priority.get(role, 99) for role in row["roles"]),
    )


def _target_hierarchies(metadata: pd.DataFrame) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for value in metadata["class_hierarchy"]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("Every class_hierarchy value must contain at least two levels")
        path = tuple(str(item) for item in value)
        if any(not item for item in path):
            raise ValueError("class_hierarchy values must not contain empty labels")
        result.append(path)
    return result


def _majority(values: Iterable[str]) -> str | None:
    counts = Counter(str(value) for value in values)
    if not counts:
        return None
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _cluster_hierarchy_maps(
    labels: np.ndarray,
    hierarchies: Sequence[tuple[str, ...]],
) -> dict[str, dict[int, str]]:
    mappings = {"leaf": {}, "parent": {}, "top": {}}
    for cluster in sorted(int(value) for value in np.unique(labels) if value >= 0):
        selected = [
            hierarchy
            for hierarchy, label in zip(hierarchies, labels, strict=True)
            if int(label) == cluster
        ]
        mappings["leaf"][cluster] = _majority(path[-1] for path in selected)
        mappings["parent"][cluster] = _majority(path[-2] for path in selected)
        mappings["top"][cluster] = _majority(path[0] for path in selected)
    return mappings


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


def _fit_hdbscan(
    features: np.ndarray,
    *,
    configuration: Mapping[str, Any] | None = None,
    hdbscan_class: HDBSCANClass | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clusterer_class = hdbscan.HDBSCAN if hdbscan_class is None else hdbscan_class
    clusterer_configuration = (
        dict(HDBSCAN_CONFIGURATION)
        if configuration is None
        else dict(configuration)
    )
    clusterer = clusterer_class(**clusterer_configuration).fit(features)
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float64)
    outlier_scores = np.asarray(clusterer.outlier_scores_, dtype=np.float64)
    if labels.shape != (len(features),):
        raise ValueError(f"HDBSCAN labels have unexpected shape {labels.shape}")
    if probabilities.shape != (len(features),):
        raise ValueError(
            f"HDBSCAN probabilities have unexpected shape {probabilities.shape}"
        )
    if outlier_scores.shape != (len(features),):
        raise ValueError(
            f"HDBSCAN outlier scores have unexpected shape {outlier_scores.shape}"
        )
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(outlier_scores)):
        raise ValueError("HDBSCAN confidence outputs must be finite")
    return labels, probabilities, outlier_scores


def _cluster_quality_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    outlier_scores: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Compute existing hard-cluster metrics plus hierarchy-aware metrics."""

    labels = np.asarray(labels, dtype=np.int64)
    features = np.asarray(features, dtype=np.float64)
    if labels.shape != (len(metadata),):
        raise ValueError("HDBSCAN labels and metadata must have aligned rows")
    true_leaf = metadata["class"].astype(str).to_numpy()
    non_noise = labels >= 0
    cluster_count = int(np.unique(labels[non_noise]).size)
    normalized_features = normalize(features, norm="l2")
    silhouette: float | None = None
    if cluster_count >= 2 and int(np.sum(non_noise)) > cluster_count:
        try:
            silhouette = float(
                silhouette_score(normalized_features[non_noise], labels[non_noise])
            )
        except ValueError:
            silhouette = None
    hierarchies = _target_hierarchies(metadata)
    targets = {
        "leaf": np.asarray([path[-1] for path in hierarchies], dtype=object),
        "parent": np.asarray([path[-2] for path in hierarchies], dtype=object),
        "top": np.asarray([path[0] for path in hierarchies], dtype=object),
    }
    mappings = _cluster_hierarchy_maps(labels, hierarchies)
    non_noise = np.asarray(labels) >= 0
    cluster_count = int(np.unique(np.asarray(labels)[non_noise]).size)
    return {
        "leaf_nmi": float(normalized_mutual_info_score(true_leaf, labels)),
        "leaf_ari": float(adjusted_rand_score(true_leaf, labels)),
        "parent_nmi": float(normalized_mutual_info_score(targets["parent"], labels)),
        "parent_ari": float(adjusted_rand_score(targets["parent"], labels)),
        "top_nmi": float(normalized_mutual_info_score(targets["top"], labels)),
        "top_ari": float(adjusted_rand_score(targets["top"], labels)),
        "hierarchy_distance": _hierarchy_distance(labels, hierarchies, mappings),
        "clusters": cluster_count,
        "cluster_count": cluster_count,
        "noise_ratio": float(np.mean(~non_noise)),
        "silhouette": silhouette,
        "mean_probability": float(np.mean(probabilities)),
        "mean_outlier_score": float(np.mean(outlier_scores)),
    }


def _load_umap() -> UMAPClass:
    try:
        from umap import UMAP
    except ImportError as error:
        raise RuntimeError("umap-learn is required for the UMAP follow-up") from error
    return UMAP


def _fit_umap(
    pca_features: np.ndarray,
    configuration: Mapping[str, Any],
    *,
    seed: int,
    umap_class: UMAPClass | None = None,
) -> np.ndarray:
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
    expected_shape = (len(pca_features), int(configuration["n_components"]))
    if coordinates.shape != expected_shape:
        raise ValueError(
            f"UMAP output has shape {coordinates.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("UMAP output must contain only finite values")
    return coordinates


def _preservation_fields(
    raw_neighbors: Mapping[int, np.ndarray],
    pca_neighbors: Mapping[int, np.ndarray],
    umap_neighbors: Mapping[int, np.ndarray] | None,
    k_values: Sequence[int],
) -> dict[str, float | None]:
    fields: dict[str, float | None] = {}
    for k in k_values:
        fields[f"raw_pca_k{k}"] = float(
            mean_neighbor_preservation(raw_neighbors[k], pca_neighbors[k])
        )
        if umap_neighbors is None:
            fields[f"raw_umap_k{k}"] = None
            fields[f"pca_umap_k{k}"] = None
            fields[f"umap_additional_loss_k{k}"] = None
        else:
            raw_umap = float(
                mean_neighbor_preservation(raw_neighbors[k], umap_neighbors[k])
            )
            pca_umap = float(
                mean_neighbor_preservation(pca_neighbors[k], umap_neighbors[k])
            )
            fields[f"raw_umap_k{k}"] = raw_umap
            fields[f"pca_umap_k{k}"] = pca_umap
            fields[f"umap_additional_loss_k{k}"] = fields[f"raw_pca_k{k}"] - raw_umap
    return fields


def _pairwise_seed_stability(
    configuration: Mapping[str, Any],
    labels_by_seed: Mapping[int, np.ndarray],
    neighbors_by_seed: Mapping[int, Mapping[int, np.ndarray]],
    *,
    reference_k: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seeds = sorted(labels_by_seed)
    for seed_a, seed_b in itertools.combinations(seeds, 2):
        labels_a = labels_by_seed[seed_a]
        labels_b = labels_by_seed[seed_b]
        noise_a = labels_a == -1
        noise_b = labels_b == -1
        union = np.sum(noise_a | noise_b)
        noise_jaccard = (
            float(np.sum(noise_a & noise_b) / union) if union else 1.0
        )
        neighbor_overlap = mean_neighbor_preservation(
            neighbors_by_seed[seed_a][reference_k],
            neighbors_by_seed[seed_b][reference_k],
        )
        records.append(
            {
                "n_neighbors": int(configuration["n_neighbors"]),
                "n_components": int(configuration["n_components"]),
                "min_dist": float(configuration["min_dist"]),
                "roles": list(configuration.get("roles", [])),
                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "cluster_ari": float(adjusted_rand_score(labels_a, labels_b)),
                "cluster_nmi": float(
                    normalized_mutual_info_score(labels_a, labels_b)
                ),
                "noise_jaccard": noise_jaccard,
                "cluster_count_a": int(np.unique(labels_a[labels_a >= 0]).size),
                "cluster_count_b": int(np.unique(labels_b[labels_b >= 0]).size),
                "cluster_count_abs_delta": int(
                    abs(
                        np.unique(labels_a[labels_a >= 0]).size
                        - np.unique(labels_b[labels_b >= 0]).size
                    )
                ),
                f"umap_neighbor_reproducibility_k{reference_k}": float(
                    neighbor_overlap
                ),
            }
        )
    return records


def _stability_summary(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[
            (
                int(row["n_neighbors"]),
                int(row["n_components"]),
                round(float(row["min_dist"]), 12),
            )
        ].append(row)
    metrics = (
        "cluster_ari",
        "cluster_nmi",
        "noise_jaccard",
        "cluster_count_abs_delta",
    )
    repro_metrics = [
        key
        for row in records
        for key in row
        if key.startswith("umap_neighbor_reproducibility_k")
    ]
    metrics = tuple(dict.fromkeys((*metrics, *repro_metrics)))
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "n_neighbors": key[0],
            "n_components": key[1],
            "min_dist": key[2],
            "pair_count": len(rows),
            "roles": list(rows[0].get("roles", [])),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            item[f"mean_{metric}"] = _mean(values)
            item[f"std_{metric}"] = _std(values)
        output.append(item)
    return output


def _summary_key(row: Mapping[str, Any]) -> tuple[str, int, int, float]:
    return (
        str(row["condition"]),
        int(row.get("n_neighbors") or -1),
        int(row.get("n_components") or -1),
        round(float(row.get("min_dist") or 0.0), 12),
    )


def summarize_runs(
    runs: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> list[dict[str, Any]]:
    """Aggregate run-level HDBSCAN and preservation metrics."""

    grouped: dict[tuple[str, int, int, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[_summary_key(row)].append(row)
    stability_lookup = {
        (
            int(row["n_neighbors"]),
            int(row["n_components"]),
            round(float(row["min_dist"]), 12),
        ): row
        for row in stability
    }

    output: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        first = rows[0]
        item: dict[str, Any] = {
            "condition": key[0],
            "configuration_name": first["configuration_name"],
            "roles": first.get("roles", []),
            "n_neighbors": None if key[0] == "pca_only" else key[1],
            "n_components": None if key[0] == "pca_only" else key[2],
            "min_dist": None if key[0] == "pca_only" else key[3],
            "seed_count": len(rows),
            "is_baseline": bool("production_baseline" in first.get("roles", [])),
        }
        for metric in SCALAR_CLUSTER_METRICS:
            values = [row.get(metric) for row in rows if row.get(metric) is not None]
            item[f"mean_{metric}"] = _mean(values)
            item[f"std_{metric}"] = _std(values)
        for metric in ("raw_pca", "raw_umap", "pca_umap", "umap_additional_loss"):
            values = [
                row.get(f"{metric}_k{k}")
                for row in rows
                for k in k_values
                if row.get(f"{metric}_k{k}") is not None
            ]
            item[f"mean_{metric}"] = _mean(values)
            item[f"std_{metric}"] = _std(values)
            for k in k_values:
                per_k = [
                    row.get(f"{metric}_k{k}")
                    for row in rows
                    if row.get(f"{metric}_k{k}") is not None
                ]
                item[f"{metric}_k{k}_mean"] = _mean(per_k)
                item[f"{metric}_k{k}_std"] = _std(per_k)
        if key[0] == "umap":
            stable = stability_lookup.get((key[1], key[2], key[3]))
            if stable is not None:
                for stable_key, value in stable.items():
                    if stable_key.startswith(("mean_", "std_")) or stable_key == "pair_count":
                        item[stable_key] = value
        output.append(item)

    output.sort(
        key=lambda row: (
            0 if row["condition"] == "pca_only" else 1,
            min(
                {
                    "production_baseline": 0,
                    "best_raw_preservation": 1,
                    "best_pca_preservation": 2,
                    "n_neighbors_ablation": 3,
                    "n_components_ablation": 4,
                    "min_dist_ablation": 5,
                    "negative_control": 6,
                }.get(role, 99)
                for role in row.get("roles", [])
            ),
        )
    )
    baseline = next(
        (row for row in output if "production_baseline" in row.get("roles", [])),
        None,
    )
    if baseline is not None:
        for row in output:
            for metric in (
                "leaf_nmi",
                "leaf_ari",
                "hierarchy_distance",
                "noise_ratio",
                "raw_umap",
                "pca_umap",
            ):
                base_value = baseline.get(f"mean_{metric}")
                current_value = row.get(f"mean_{metric}")
                row[f"delta_{metric}_vs_baseline"] = (
                    None
                    if base_value is None or current_value is None
                    else float(current_value) - float(base_value)
                )
    return output


def _display_label(row: Mapping[str, Any]) -> str:
    if row.get("condition") == "pca_only":
        return "PCA-only"
    roles = set(row.get("roles", []))
    if "production_baseline" in roles:
        return "baseline"
    labels = []
    for role, label in (
        ("best_raw_preservation", "best raw"),
        ("best_pca_preservation", "best PCA"),
        ("n_neighbors_ablation", "n_neighbors ablation"),
        ("n_components_ablation", "n_components ablation"),
        ("min_dist_ablation", "min_dist ablation"),
        ("negative_control", "negative"),
    ):
        if role in roles:
            labels.append(label)
    return "/".join(labels) or str(row.get("configuration_name", "UMAP"))


def _plot_save(path: Path, title: str, plotter: Callable[[Any], None]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for follow-up plots") from error
    figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    plotter(axis)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def make_plots(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = list(report["summary"])
    labels = [_display_label(row) for row in summaries]
    x = np.arange(len(summaries))

    paths = {
        "configuration_comparison": output_dir / "configuration-comparison.png",
        "preservation_vs_leaf_nmi": output_dir / "preservation-vs-leaf-nmi.png",
        "preservation_vs_leaf_ari": output_dir / "preservation-vs-leaf-ari.png",
        "preservation_vs_hierarchy_distance": output_dir / "preservation-vs-hierarchy-distance.png",
        "noise_ratio_comparison": output_dir / "noise-ratio-comparison.png",
        "seed_cluster_stability": output_dir / "seed-cluster-stability.png",
    }

    def configuration_plot(axis: Any) -> None:
        width = 0.36
        nmi = [float(row.get("mean_leaf_nmi") or 0.0) for row in summaries]
        ari = [float(row.get("mean_leaf_ari") or 0.0) for row in summaries]
        axis.bar(x - width / 2, nmi, width, label="Leaf NMI")
        axis.bar(x + width / 2, ari, width, label="Leaf ARI")
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.02)
        axis.legend(loc="best")

    def preservation_x(row: Mapping[str, Any]) -> float | None:
        value = row.get("mean_raw_umap")
        if value is None:
            value = row.get("mean_raw_pca")
        return None if value is None else float(value)

    def scatter_plot(axis: Any, y_key: str, y_label: str, invert: bool = False) -> None:
        for row, label in zip(summaries, labels, strict=True):
            px = preservation_x(row)
            py = row.get(y_key)
            if px is None or py is None:
                continue
            axis.scatter(px, float(py), s=55)
            axis.annotate(label, (px, float(py)), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel("Mean Raw↔UMAP (PCA-only uses Raw↔PCA)")
        axis.set_ylabel(y_label)
        if invert:
            axis.invert_yaxis()

    def noise_plot(axis: Any) -> None:
        values = [float(row.get("mean_noise_ratio") or 0.0) for row in summaries]
        axis.bar(x, values, color="#d95f02")
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_ylabel("Noise ratio")
        axis.set_ylim(0.0, 1.0)

    stable_rows = [row for row in summaries if row.get("mean_cluster_ari") is not None]

    def stability_plot(axis: Any) -> None:
        stable_labels = [_display_label(row) for row in stable_rows]
        stable_x = np.arange(len(stable_rows))
        ari = [float(row["mean_cluster_ari"]) for row in stable_rows]
        ari_err = [float(row.get("std_cluster_ari") or 0.0) for row in stable_rows]
        jaccard = [float(row["mean_noise_jaccard"]) for row in stable_rows]
        jaccard_err = [float(row.get("std_noise_jaccard") or 0.0) for row in stable_rows]
        axis.errorbar(stable_x, ari, yerr=ari_err, marker="o", capsize=3, label="Cluster ARI")
        axis.errorbar(stable_x, jaccard, yerr=jaccard_err, marker="o", capsize=3, label="Noise Jaccard")
        axis.set_xticks(stable_x, stable_labels, rotation=35, ha="right")
        axis.set_ylabel("Seed-to-seed stability")
        axis.set_ylim(0.0, 1.02)
        axis.legend(loc="best")

    _plot_save(paths["configuration_comparison"], "PCA-only and UMAP/HDBSCAN quality", configuration_plot)
    _plot_save(
        paths["preservation_vs_leaf_nmi"],
        "Neighborhood preservation vs Leaf NMI",
        lambda axis: scatter_plot(axis, "mean_leaf_nmi", "Mean Leaf NMI"),
    )
    _plot_save(
        paths["preservation_vs_leaf_ari"],
        "Neighborhood preservation vs Leaf ARI",
        lambda axis: scatter_plot(axis, "mean_leaf_ari", "Mean Leaf ARI"),
    )
    _plot_save(
        paths["preservation_vs_hierarchy_distance"],
        "Neighborhood preservation vs hierarchy distance",
        lambda axis: scatter_plot(axis, "mean_hierarchy_distance", "Mean hierarchy distance", invert=True),
    )
    _plot_save(paths["noise_ratio_comparison"], "Noise ratio by discovery space", noise_plot)
    _plot_save(paths["seed_cluster_stability"], "Seed-to-seed cluster stability", stability_plot)
    return {key: path.name for key, path in paths.items()}


def _format(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def write_artifacts(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "dataset-fingerprint.json", report["dataset_fingerprint"])
    _write_json(output_dir / "pca-selection.json", report["fixed_pca"])
    _write_json(output_dir / "selected-configs.json", report["selected_configurations"])
    _write_csv(output_dir / "runs.csv", report["runs"])
    _write_csv(output_dir / "summary.csv", report["summary"])
    _write_csv(
        output_dir / "seed-cluster-stability.csv",
        report["cluster_stability"],
        fieldnames=STABILITY_CSV_FIELDS,
    )

    plot_names = make_plots(report, output_dir)
    report["artifacts"] = {
        "report_json": "report.json",
        "runs_csv": "runs.csv",
        "summary_csv": "summary.csv",
        "seed_cluster_stability_csv": "seed-cluster-stability.csv",
        "dataset_fingerprint_json": "dataset-fingerprint.json",
        "pca_selection_json": "pca-selection.json",
        "selected_configs_json": "selected-configs.json",
        **plot_names,
        "methodology": "REPORT.md",
    }

    summaries = report["summary"]
    baseline = next(
        (row for row in summaries if "production_baseline" in row.get("roles", [])),
        None,
    )
    pca_only = next((row for row in summaries if row["condition"] == "pca_only"), None)
    best_raw = next(
        (row for row in summaries if "best_raw_preservation" in row.get("roles", [])),
        None,
    )
    lines = [
        "# PCA·UMAP·HDBSCAN follow-up benchmark",
        "",
        "This report evaluates HDBSCAN after the fixed PCA/UMAP neighborhood experiment. HDBSCAN parameters are fixed; the PCA-only row is the no-UMAP control.",
        "",
        "## Dataset and protocol",
        "",
        f"- Input: `{report['dataset']['input']}`; Gemini model = `{report['dataset']['embedding_model']}`, task = `{report['dataset']['embedding_task']}`, dimensionality = {report['dataset']['embedding_dimension']}.",
        f"- Rows: {report['dataset']['rows']} sampled with seed {report['dataset']['sample_seed']}; fingerprints are stored in `dataset-fingerprint.json`.",
        f"- Fixed PCA: {report['fixed_pca']['selected_dimension']}D, seed {report['protocol']['pca_seed']}, selection reason `{report['fixed_pca']['selection_reason']}`.",
        "- HDBSCAN input uses the normalized selected PCA prefix from phase 1 so this follow-up is directly aligned with the neighborhood-preservation experiment; the repository production comparison path currently keeps an unnormalized PCA prefix for its auxiliary membership calculation.",
        f"- UMAP seeds: {report['protocol']['umap_seeds']}; metric = Euclidean, init = random, n_jobs = 1.",
        f"- HDBSCAN: min_cluster_size = {report['protocol']['hdbscan']['min_cluster_size']}, min_samples = {report['protocol']['hdbscan']['min_samples']}, metric = Euclidean, cluster_selection_method = leaf.",
        "- Evaluated metrics: Leaf/parent/top NMI and ARI, DBpedia ground-truth hierarchy distance, noise ratio, cluster count, silhouette, and seed-to-seed cluster stability.",
        "",
        "## Data scope note",
        "",
        "The committed run uses the repository's Gemini DBpedia dataset, not the unavailable Wikipedia BGE artifact. These results are valid for this Gemini dataset and should not be relabeled as the Wikipedia benchmark.",
        "Hierarchy distance is measured against the DBpedia class hierarchy in metadata; it is not a claim that HDBSCAN's unsupervised density tree is a semantic taxonomy.",
        "",
        "## Configuration results",
        "",
        "| condition | roles | UMAP | mean Raw↔UMAP | mean PCA↔UMAP | Leaf NMI | Leaf ARI | hierarchy distance | noise ratio | clusters | seed cluster ARI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        umap_text = (
            "PCA-only"
            if row["condition"] == "pca_only"
            else f"{row['n_neighbors']} / {row['n_components']} / {row['min_dist']}"
        )
        lines.append(
            f"| {row['condition']} | {', '.join(row.get('roles', [])) or '—'} | {umap_text} | {_format(row.get('mean_raw_umap'))} | {_format(row.get('mean_pca_umap'))} | {_format(row.get('mean_leaf_nmi'))} | {_format(row.get('mean_leaf_ari'))} | {_format(row.get('mean_hierarchy_distance'))} | {_format(row.get('mean_noise_ratio'))} | {_format(row.get('mean_cluster_count'), 2)} | {_format(row.get('mean_cluster_ari'))} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if pca_only is not None and baseline is not None:
        lines.append(
            f"- PCA-only vs production baseline: Leaf NMI {_format(pca_only.get('mean_leaf_nmi'))} vs {_format(baseline.get('mean_leaf_nmi'))}; Leaf ARI {_format(pca_only.get('mean_leaf_ari'))} vs {_format(baseline.get('mean_leaf_ari'))}."
        )
    if baseline is not None:
        lines.append(
            f"- Production baseline at k=15: Raw↔UMAP {_format(baseline.get('raw_umap_k15_mean'))}, PCA↔UMAP {_format(baseline.get('pca_umap_k15_mean'))}, seed cluster ARI {_format(baseline.get('mean_cluster_ari'))}."
        )
    if best_raw is not None:
        lines.append(
            f"- The configuration selected for highest phase-1 Raw↔UMAP preservation is {best_raw['n_neighbors']} / {best_raw['n_components']} / {best_raw['min_dist']}; its Leaf NMI is {_format(best_raw.get('mean_leaf_nmi'))} and Leaf ARI is {_format(best_raw.get('mean_leaf_ari'))}."
        )
    lines.append(
        "- A production change should require joint evidence from neighborhood preservation, Leaf NMI/ARI, hierarchy distance, noise behavior, and seed stability. This benchmark does not change production settings automatically."
    )

    lines.extend(["", "## Artifacts", ""])
    for filename, description in (
        ("report.json", "complete run-level and aggregate report"),
        ("runs.csv", "one row per PCA-only or UMAP seed run"),
        ("summary.csv", "configuration-level means and standard deviations"),
        ("seed-cluster-stability.csv", "pairwise UMAP seed cluster comparisons"),
        ("pca-selection.json", "fixed PCA selection and Raw↔PCA baseline"),
        ("selected-configs.json", "phase-1-derived comparison roles"),
    ):
        lines.append(f"- `{filename}`: {description}.")
    for key, filename in plot_names.items():
        lines.append(f"- `{filename}`: `{key}` plot.")
    lines.extend(
        [
            "",
            "![Configuration comparison](configuration-comparison.png)",
            "",
            "![Preservation versus Leaf NMI](preservation-vs-leaf-nmi.png)",
            "",
            "![Preservation versus Leaf ARI](preservation-vs-leaf-ari.png)",
            "",
            "![Preservation versus hierarchy distance](preservation-vs-hierarchy-distance.png)",
            "",
            "![Noise ratio comparison](noise-ratio-comparison.png)",
            "",
            "![Seed cluster stability](seed-cluster-stability.png)",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_dir / "report.json", report)


def run_followup(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    source_indices: np.ndarray,
    phase1_report: Mapping[str, Any],
    *,
    input_path: Path,
    umap_seeds: Sequence[int] = DEFAULT_UMAP_SEEDS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    pca_max_components: int = 512,
    pca_min_components: int = 32,
    pca_component_step: int = 32,
    minimum_preservation_gain: float = 0.05,
    progress: bool = False,
    selected_configurations: Sequence[Mapping[str, Any]] | None = None,
    umap_class: UMAPClass | None = None,
    hdbscan_class: HDBSCANClass | None = None,
) -> dict[str, Any]:
    """Run PCA-only and selected UMAP/HDBSCAN configurations."""

    if min_cluster_size < 2 or min_samples < 1:
        raise ValueError("invalid HDBSCAN parameters")
    seeds = tuple(dict.fromkeys(int(seed) for seed in umap_seeds))
    if not seeds:
        raise ValueError("umap_seeds must contain at least one seed")

    phase1_protocol = phase1_report.get("protocol", {})
    pca_selection_k_values = tuple(
        int(value)
        for value in phase1_protocol.get("pca_selection_k_values", (15, 30))
    )
    prepared = prepare_experiment(
        embeddings,
        k_values=k_values,
        pca_selection_k_values=pca_selection_k_values,
        pca_max_components=pca_max_components,
        pca_min_components=pca_min_components,
        pca_component_step=pca_component_step,
        minimum_preservation_gain=minimum_preservation_gain,
        pca_seed=int(phase1_protocol.get("pca_seed", 42)),
    )
    expected_dimension = phase1_report.get("pca_selection", {}).get("selected_dimension")
    if expected_dimension is not None and int(expected_dimension) != prepared.pca_selection.selected_dimension:
        raise ValueError(
            "phase 1 and follow-up selected PCA dimensions differ: "
            f"{expected_dimension} vs {prepared.pca_selection.selected_dimension}"
        )

    raw_neighbor_map = neighbors_by_k(
        prepared.normalized_embeddings, k_values, metric="cosine"
    )
    pca_neighbor_map = neighbors_by_k(
        prepared.pca_features, k_values, metric="cosine"
    )
    phase1_raw_pca = phase1_report.get("raw_pca_preservation", {})
    for k in k_values:
        expected = phase1_raw_pca.get(f"k{k}")
        actual = mean_neighbor_preservation(raw_neighbor_map[k], pca_neighbor_map[k])
        if expected is not None and not math.isclose(float(expected), actual, abs_tol=1e-8):
            raise ValueError(
                f"phase 1 Raw↔PCA mismatch at k={k}: {expected} vs {actual}"
            )

    if selected_configurations is None:
        selected_configurations = select_configurations(phase1_report, k_values=k_values)
    selected = [dict(configuration) for configuration in selected_configurations]
    hdbscan_configuration = dict(HDBSCAN_CONFIGURATION)
    hdbscan_configuration.update(
        {"min_cluster_size": int(min_cluster_size), "min_samples": int(min_samples)}
    )

    runs: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []

    pca_started = time.perf_counter()
    pca_labels, pca_probabilities, pca_outliers = _fit_hdbscan(
        prepared.pca_features,
        configuration=hdbscan_configuration,
        hdbscan_class=hdbscan_class,
    )
    pca_hdbscan_seconds = float(time.perf_counter() - pca_started)
    pca_quality = _cluster_quality_metrics(
        prepared.pca_features,
        pca_labels,
        pca_probabilities,
        pca_outliers,
        metadata,
    )
    pca_row: dict[str, Any] = {
        "condition": "pca_only",
        "configuration_name": "pca_only",
        "roles": ["pca_only_control"],
        "seed": None,
        "n_neighbors": None,
        "n_components": None,
        "min_dist": None,
        "feature_space": "normalized_selected_pca_prefix",
        "umap_metric": None,
        "umap_init": None,
        "umap_n_jobs": None,
        "hdbscan_fit_sec": pca_hdbscan_seconds,
        "umap_fit_sec": None,
        "run_total_sec": pca_hdbscan_seconds,
    }
    pca_row.update(pca_quality)
    pca_row.update(
        _preservation_fields(raw_neighbor_map, pca_neighbor_map, None, k_values)
    )
    runs.append(pca_row)

    total = len(selected) * len(seeds)
    completed = 0
    for configuration_index, configuration in enumerate(selected):
        labels_by_seed: dict[int, np.ndarray] = {}
        neighbors_by_seed: dict[int, dict[int, np.ndarray]] = {}
        for seed in seeds:
            completed += 1
            if progress:
                print(
                    f"[{completed}/{total}] "
                    f"n_neighbors={configuration['n_neighbors']} "
                    f"n_components={configuration['n_components']} "
                    f"min_dist={configuration['min_dist']} seed={seed}",
                    flush=True,
                )
            started = time.perf_counter()
            umap_started = time.perf_counter()
            coordinates = _fit_umap(
                prepared.pca_features,
                configuration,
                seed=seed,
                umap_class=umap_class,
            )
            umap_seconds = float(time.perf_counter() - umap_started)
            umap_neighbors = neighbors_by_k(coordinates, k_values, metric="euclidean")
            hdb_started = time.perf_counter()
            labels, probabilities, outliers = _fit_hdbscan(
                coordinates,
                configuration=hdbscan_configuration,
                hdbscan_class=hdbscan_class,
            )
            hdbscan_seconds = float(time.perf_counter() - hdb_started)
            quality = _cluster_quality_metrics(
                coordinates,
                labels,
                probabilities,
                outliers,
                metadata,
            )
            row: dict[str, Any] = {
                "condition": "umap",
                "configuration_name": f"umap_config_{configuration_index + 1}",
                "roles": list(configuration.get("roles", [])),
                "seed": int(seed),
                "n_neighbors": int(configuration["n_neighbors"]),
                "n_components": int(configuration["n_components"]),
                "min_dist": float(configuration["min_dist"]),
                "feature_space": "umap_coordinates_from_normalized_selected_pca_prefix",
                "umap_metric": "euclidean",
                "umap_init": "random",
                "umap_n_jobs": 1,
                "umap_fit_sec": umap_seconds,
                "hdbscan_fit_sec": hdbscan_seconds,
                "run_total_sec": float(time.perf_counter() - started),
            }
            row.update(quality)
            row.update(
                _preservation_fields(
                    raw_neighbor_map,
                    pca_neighbor_map,
                    umap_neighbors,
                    k_values,
                )
            )
            runs.append(row)
            labels_by_seed[int(seed)] = labels
            neighbors_by_seed[int(seed)] = umap_neighbors
        stability_records.extend(
            _pairwise_seed_stability(
                configuration,
                labels_by_seed,
                neighbors_by_seed,
                reference_k=15 if 15 in k_values else int(k_values[0]),
            )
        )

    stable_summary = _stability_summary(stability_records)
    summary = summarize_runs(runs, stable_summary, k_values=k_values)
    fixed_pca = prepared.pca_selection.to_dict()
    fixed_pca["selected_dimension_knn_preservation"] = {
        f"k{k}": float(
            mean_neighbor_preservation(raw_neighbor_map[k], pca_neighbor_map[k])
        )
        for k in k_values
    }
    selected_candidate = next(
        (
            candidate
            for candidate in fixed_pca.get("candidates", [])
            if int(candidate["dimension"]) == prepared.pca_selection.selected_dimension
        ),
        None,
    )
    fixed_pca["selected_cumulative_explained_variance"] = (
        None
        if selected_candidate is None
        else selected_candidate["cumulative_explained_variance"]
    )
    source_name = input_path.name.lower()
    is_gemini = "gemini" in source_name
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "pca_umap_hdbscan_followup",
        "dataset": {
            "input": str(input_path),
            "rows": int(len(prepared.normalized_embeddings)),
            "embedding_dimension": int(prepared.normalized_embeddings.shape[1]),
            "embedding_model": "gemini-embedding-001" if is_gemini else "unspecified",
            "embedding_task": "CLUSTERING" if is_gemini else "unspecified",
            "sample_seed": int(DEFAULT_DATASET_SAMPLE_SEED),
            "l2_normalized": True,
            "label_columns": ["class", "class_hierarchy"],
        },
        "dataset_fingerprint": _fingerprints(
            prepared.normalized_embeddings,
            metadata,
            source_indices,
        ),
        "phase1": {
            "report": str(DEFAULT_PHASE1_REPORT),
            "selected_dimension": phase1_report.get("pca_selection", {}).get(
                "selected_dimension"
            ),
            "raw_pca_preservation": phase1_report.get("raw_pca_preservation", {}),
        },
        "fixed_pca": fixed_pca,
        "protocol": {
            "k_values": [int(k) for k in k_values],
            "raw_metric": "cosine",
            "pca_metric": "cosine",
            "umap_metric": "euclidean",
            "pca_feature_space": "normalized_selected_pca_prefix",
            "pca_selection_once": True,
            "pca_seed": int(phase1_protocol.get("pca_seed", 42)),
            "umap_seeds": list(seeds),
            "umap_init": "random",
            "umap_n_jobs": 1,
            "hdbscan": hdbscan_configuration,
            "pca_only_control": True,
            "phase1_configurations_used": len(selected),
        },
        "selected_configurations": selected,
        "runs": runs,
        "summary": summary,
        "cluster_stability": stability_records,
        "cluster_stability_summary": stable_summary,
        "artifacts": {},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate HDBSCAN after fixed PCA/UMAP neighborhood preservation"
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--phase1-report", type=Path, default=DEFAULT_PHASE1_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-sample-size", type=int, default=DEFAULT_DATASET_SAMPLE_SIZE)
    parser.add_argument("--dataset-sample-seed", type=int, default=DEFAULT_DATASET_SAMPLE_SEED)
    parser.add_argument("--umap-seeds", nargs="+", type=int, default=list(DEFAULT_UMAP_SEEDS))
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--pca-max-components", type=int, default=512)
    parser.add_argument("--pca-min-components", type=int, default=32)
    parser.add_argument("--pca-component-step", type=int, default=32)
    parser.add_argument("--minimum-preservation-gain", type=float, default=0.05)
    parser.add_argument("--quick", "--fast", action="store_true", help="run only baseline and the phase-1 best configuration with one seed")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phase1_report = json.loads(args.phase1_report.read_text(encoding="utf-8"))
    embeddings, metadata, source_indices = load_gemini_dataset(
        args.input_json,
        sample_size=args.dataset_sample_size,
        sample_seed=args.dataset_sample_seed,
    )
    configurations = select_configurations(
        phase1_report,
        k_values=tuple(int(k) for k in phase1_report["protocol"]["k_values"]),
    )
    seeds = tuple(args.umap_seeds)
    if args.quick:
        configurations = [
            configuration
            for configuration in configurations
            if "production_baseline" in configuration["roles"]
            or "best_raw_preservation" in configuration["roles"]
            or "best_pca_preservation" in configuration["roles"]
        ]
        seeds = (int(seeds[0]),)
    report = run_followup(
        embeddings,
        metadata,
        source_indices,
        phase1_report,
        input_path=args.input_json,
        umap_seeds=seeds,
        k_values=tuple(int(k) for k in phase1_report["protocol"]["k_values"]),
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        pca_max_components=args.pca_max_components,
        pca_min_components=args.pca_min_components,
        pca_component_step=args.pca_component_step,
        minimum_preservation_gain=args.minimum_preservation_gain,
        progress=args.progress,
        selected_configurations=configurations,
    )
    report["dataset"]["sample_seed"] = int(args.dataset_sample_seed)
    report["dataset"]["sample_size_requested"] = args.dataset_sample_size
    report["phase1"]["report"] = str(args.phase1_report)
    write_artifacts(report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "runs": len(report["runs"]),
                "umap_configurations": report["protocol"]["phase1_configurations_used"],
                "pca_only": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
