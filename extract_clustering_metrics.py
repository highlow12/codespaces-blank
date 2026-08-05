"""Extract clustering quality metrics without fitting a clustering model.

The tool consumes saved assignments and, optionally, the feature matrix used
to create those assignments. It never calls PCA, FCM, or any other fitting
routine. Ground-truth columns are used only for external evaluation metrics.

For saved FCM memberships it reports:

* external metrics: NMI, ARI, homogeneity, completeness, and V-measure;
* internal metrics: silhouette, Xie-Beni (XB), partition coefficient (PC),
  partition entropy (PE), and normalized PE;
* assignment diagnostics: cluster count, noise ratio, and target fragmentation.

The XB calculation is performed in the feature space supplied to this tool.
To reproduce a pipeline's exact post-PCA XB/silhouette, pass the same saved
post-PCA feature matrix. Passing the original embedding JSON instead measures
the raw embedding space and is still useful for comparing assignments.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import normalize

from embedding_data import load_embeddings_from_json


MEMBERSHIP_PATTERN = re.compile(r"^membership_(\d+)$")
EXTERNAL_METRICS = (
    "nmi",
    "ari",
    "homogeneity",
    "completeness",
    "v_measure",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_hierarchy(value: Any) -> tuple[str, ...] | None:
    """Parse JSON/Python-list hierarchy values written by pandas to CSV."""

    if isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return None
        values = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple)):
                values = list(parsed)
                break
        if values is None:
            return None
    else:
        return None

    if len(values) < 1 or not all(str(item).strip() for item in values):
        return None
    return tuple(str(item) for item in values)


def _membership_columns(columns: Iterable[str]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for column in columns:
        match = MEMBERSHIP_PATTERN.fullmatch(str(column))
        if match:
            indexed.append((int(match.group(1)), str(column)))
    indexed.sort()
    return [column for _, column in indexed]


def _load_features(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "features" in archive:
                values = archive["features"]
            elif "arr_0" in archive:
                values = archive["arr_0"]
            else:
                raise ValueError(
                    f"Expected 'features' or 'arr_0' in NumPy archive {path}"
                )
    elif suffix == ".json":
        values, _metadata = load_embeddings_from_json(path)
    elif suffix == ".csv":
        values = pd.read_csv(path).to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported feature format {path}; use .npy, .npz, .json, or .csv"
        )

    return _validate_features(values, source=path)


def _validate_features(values: Any, *, source: str | Path) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Features at {source} must be a non-empty 2D matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Features at {source} must contain only finite values")
    if np.any(np.linalg.norm(values, axis=1) <= 1e-12):
        raise ValueError(f"Features at {source} contain a zero-length row")
    return values


def _resolve_paths(
    paths: list[Path],
    count: int,
    *,
    option_name: str,
) -> list[Path | None]:
    if not paths:
        return [None] * count
    if len(paths) == 1:
        return paths * count
    if len(paths) != count:
        raise ValueError(
            f"{option_name} accepts one path or exactly one path per assignment "
            f"({count} assignments, {len(paths)} paths)"
        )
    return list(paths)


def _align_metadata(
    assignments: pd.DataFrame,
    metadata_path: Path,
) -> pd.DataFrame:
    _metadata_embeddings, metadata = load_embeddings_from_json(metadata_path)
    if len(assignments) == len(metadata) and "id" not in assignments.columns:
        return metadata.reset_index(drop=True)
    if "id" not in assignments.columns or "id" not in metadata.columns:
        if len(assignments) != len(metadata):
            raise ValueError(
                f"Cannot align {metadata_path}: row counts differ and no shared id"
            )
        return metadata.reset_index(drop=True)

    assignment_ids = assignments["id"].astype(str)
    metadata_ids = metadata["id"].astype(str)
    if metadata_ids.duplicated().any():
        raise ValueError(f"Metadata IDs must be unique in {metadata_path}")
    lookup = metadata.copy()
    lookup["_metric_id"] = metadata_ids.to_numpy()
    aligned = (
        lookup.set_index("_metric_id")
        .reindex(assignment_ids.to_numpy())
        .reset_index(drop=True)
    )
    if aligned["id"].isna().any():
        missing = assignment_ids[aligned["id"].isna()].tolist()[:5]
        raise ValueError(f"Metadata is missing assignment IDs, examples: {missing}")
    return aligned


def _ensure_targets(
    assignments: pd.DataFrame,
    metadata_path: Path | None,
) -> pd.DataFrame:
    output = assignments.copy()
    if metadata_path is not None:
        metadata = _align_metadata(output, metadata_path)
        for column in ("class", "class_hierarchy"):
            if column not in output.columns and column in metadata.columns:
                output[column] = metadata[column].to_numpy()
    return output


def _target_levels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    if "class" in frame.columns:
        classes = frame["class"].astype(str).to_numpy()
        if np.any(pd.isna(frame["class"])):
            raise ValueError("Assignment class labels contain missing values")
    else:
        classes = None

    hierarchy_values: list[tuple[str, ...] | None] | None = None
    if "class_hierarchy" in frame.columns:
        hierarchy_values = [
            _parse_hierarchy(value) for value in frame["class_hierarchy"]
        ]
        if any(value is None for value in hierarchy_values):
            hierarchy_values = None

    if classes is None and hierarchy_values is None:
        return {}
    if classes is None:
        classes = np.asarray(
            [hierarchy[-1] for hierarchy in hierarchy_values if hierarchy]
        )
    targets: dict[str, np.ndarray] = {"leaf": classes}
    if hierarchy_values is not None:
        targets["top"] = np.asarray(
            [hierarchy[0] for hierarchy in hierarchy_values if hierarchy]
        )
        if all(len(hierarchy) >= 2 for hierarchy in hierarchy_values if hierarchy):
            targets["subtopic"] = np.asarray(
                [hierarchy[1] for hierarchy in hierarchy_values if hierarchy]
            )
    return targets


def _external_metrics(
    target: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    return {
        "nmi": float(normalized_mutual_info_score(target, labels)),
        "ari": float(adjusted_rand_score(target, labels)),
        "homogeneity": float(homogeneity_score(target, labels)),
        "completeness": float(completeness_score(target, labels)),
        "v_measure": float(v_measure_score(target, labels)),
    }


def _fuzzy_partition_metrics(
    memberships: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(memberships, dtype=np.float64)
    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 1e-12):
        raise ValueError("Membership rows must have a positive sum")
    if np.any(values < -1e-12) or not np.all(np.isfinite(values)):
        raise ValueError("Memberships must be finite and non-negative")
    values = np.maximum(values, 0.0)
    values /= values.sum(axis=1, keepdims=True)
    cluster_count = values.shape[1]
    pc = float(np.mean(np.sum(values**2, axis=1)))
    safe = np.maximum(values, 1e-12)
    pe = float(-np.mean(np.sum(values * np.log(safe), axis=1)))
    normalized_pe = float(pe / np.log(cluster_count)) if cluster_count > 1 else 0.0
    baseline = 1.0 / cluster_count
    modified_pc = (
        float((pc - baseline) / (1.0 - baseline))
        if cluster_count > 1
        else 1.0
    )
    return {
        "partition_coefficient": pc,
        "modified_partition_coefficient": modified_pc,
        "partition_entropy": pe,
        "normalized_partition_entropy": normalized_pe,
        "pc": pc,
        "pe": pe,
        "normalized_pe": normalized_pe,
        "membership_clusters": int(cluster_count),
    }


def _centers_from_memberships(
    features: np.ndarray,
    memberships: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    normalized_features = normalize(features, norm="l2")
    values = np.asarray(memberships, dtype=np.float64)
    values = np.maximum(values, 0.0)
    values /= values.sum(axis=1, keepdims=True)
    weights = values**2
    centers = weights.T @ normalized_features
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if np.any(center_norms <= 1e-12):
        return None
    centers /= center_norms
    return normalized_features, centers


def _prepare_centers(
    features: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    normalized_features = normalize(features, norm="l2")
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != features.shape[1]:
        raise ValueError("Centers must be a 2D matrix matching feature width")
    if not np.all(np.isfinite(values)):
        raise ValueError("Centers must contain only finite values")
    center_norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(center_norms <= 1e-12):
        return None
    return normalized_features, normalize(values, norm="l2")


def _internal_metrics(
    labels: np.ndarray,
    features: np.ndarray,
    memberships: np.ndarray | None,
    centers: np.ndarray | None,
) -> dict[str, float | None]:
    normalized_features = normalize(features, norm="l2")
    non_noise = labels != -1
    cluster_count = np.unique(labels[non_noise]).size
    metrics: dict[str, float | None] = {
        "silhouette": None,
        "xie_beni": None,
        "xb": None,
        "fuzzy_silhouette": None,
    }
    if cluster_count >= 2 and int(np.sum(non_noise)) >= 3:
        try:
            metrics["silhouette"] = float(
                silhouette_score(
                    normalized_features[non_noise],
                    labels[non_noise],
                )
            )
        except ValueError:
            pass

    if memberships is None:
        unique_labels = sorted(int(label) for label in np.unique(labels) if label != -1)
        if len(unique_labels) < 2:
            return metrics
        hard_memberships = np.zeros((len(labels), len(unique_labels)))
        label_to_column = {label: index for index, label in enumerate(unique_labels)}
        for row_index, label in enumerate(labels):
            if int(label) in label_to_column:
                hard_memberships[row_index, label_to_column[int(label)]] = 1.0
        memberships = hard_memberships

    prepared = (
        _centers_from_memberships(features, memberships)
        if centers is None
        else _prepare_centers(features, centers)
    )
    if prepared is None or memberships.shape[1] < 2:
        return metrics
    normalized_features, centers = prepared
    squared_distances = np.maximum(
        2.0 - 2.0 * (normalized_features @ centers.T),
        0.0,
    )
    weights = memberships**2
    numerator = float(np.sum(weights * squared_distances))
    center_distances = np.maximum(2.0 - 2.0 * (centers @ centers.T), 0.0)
    np.fill_diagonal(center_distances, np.inf)
    minimum_separation = float(np.min(center_distances))
    if minimum_separation > 1e-12:
        xb = numerator / (features.shape[0] * minimum_separation)
        metrics["xie_beni"] = xb
        metrics["xb"] = xb

    distances = np.sqrt(squared_distances)
    membership_values = np.asarray(memberships, dtype=np.float64)
    membership_values = np.maximum(membership_values, 0.0)
    membership_values /= membership_values.sum(axis=1, keepdims=True)
    fuzzy_weights = membership_values**2
    a = np.sum(fuzzy_weights * distances, axis=1) / np.sum(
        fuzzy_weights,
        axis=1,
    )
    b = np.partition(distances, 1, axis=1)[:, 1]
    fuzzy_scores = (b - a) / np.maximum(a, b)
    metrics["fuzzy_silhouette"] = float(np.mean(fuzzy_scores))
    return metrics


def extract_metrics_from_frame(
    frame: pd.DataFrame,
    *,
    source: str = "<memory>",
    features: np.ndarray | None = None,
    feature_source: str | None = None,
    centers: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract metrics from an in-memory assignment frame without fitting.

    ``centers`` is optional. When supplied by an in-memory pipeline caller,
    XB and fuzzy silhouette use those fitted centers exactly. The file-based
    CLI derives centers from saved memberships because assignment CSVs do not
    contain model centers.
    """

    frame = frame.copy()
    if "cluster" not in frame.columns:
        raise ValueError(f"Assignments at {source} need a 'cluster' column")
    labels_numeric = pd.to_numeric(frame["cluster"], errors="raise")
    if not np.all(np.isfinite(labels_numeric)):
        raise ValueError(f"Cluster labels at {source} must be finite")
    labels = labels_numeric.to_numpy(dtype=np.int64)
    non_noise = labels != -1
    target_levels = _target_levels(frame)

    metrics: dict[str, Any] = {
        "source": str(source),
        "samples": int(len(labels)),
        "clusters": int(np.unique(labels[non_noise]).size),
        "noise_ratio": float(np.mean(~non_noise)),
        "feature_source": feature_source,
        "features_available": features is not None,
        "memberships_available": False,
    }
    for level in ("leaf", "top", "subtopic"):
        for metric_name in EXTERNAL_METRICS:
            metrics[f"{metric_name}_{level}"] = None
    metrics["nmi"] = None
    metrics["ari"] = None
    metrics["tag_fragmentation"] = None

    for level, target in target_levels.items():
        external = _external_metrics(target, labels)
        for metric_name, value in external.items():
            metrics[f"{metric_name}_{level}"] = value
        if level == "leaf":
            metrics["nmi"] = external["nmi"]
            metrics["ari"] = external["ari"]
            fragmentation: list[float] = []
            for target_label in np.unique(target):
                assigned = labels[target == target_label]
                assigned = assigned[assigned != -1]
                if assigned.size:
                    fragmentation.append(float(np.unique(assigned).size))
            if fragmentation:
                metrics["tag_fragmentation"] = float(np.mean(fragmentation))

    membership_columns = _membership_columns(frame.columns)
    memberships: np.ndarray | None = None
    if membership_columns:
        memberships = frame[membership_columns].to_numpy(dtype=np.float64)
        metrics.update(_fuzzy_partition_metrics(memberships))
        metrics["memberships_available"] = True
    else:
        for metric_name in (
            "partition_coefficient",
            "modified_partition_coefficient",
            "partition_entropy",
            "normalized_partition_entropy",
            "pc",
            "pe",
            "normalized_pe",
        ):
            metrics[metric_name] = None
        metrics["membership_clusters"] = None

    if features is not None:
        features = _validate_features(features, source=feature_source or source)
        if features.shape[0] != len(frame):
            raise ValueError(
                f"Feature rows ({features.shape[0]}) do not match assignment rows "
                f"({len(frame)}) for {source}"
            )
        metrics.update(_internal_metrics(labels, features, memberships, centers))
    else:
        metrics.update(
            {
                "silhouette": None,
                "xie_beni": None,
                "xb": None,
                "fuzzy_silhouette": None,
            }
        )
    return metrics


def extract_metrics(
    assignment_path: Path,
    *,
    features_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Extract all available metrics from one saved assignment file."""

    frame = pd.read_csv(assignment_path)
    frame = _ensure_targets(frame, metadata_path)
    features = None if features_path is None else _load_features(features_path)
    return extract_metrics_from_frame(
        frame,
        source=str(assignment_path),
        features=features,
        feature_source=None if features_path is None else str(features_path),
    )


def _write_outputs(
    rows: list[dict[str, Any]],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    payload = {
        "tool": {
            "name": "extract_clustering_metrics",
            "fits_clustering_model": False,
            "ground_truth_used_for_fit": False,
            "notes": (
                "XB and silhouette are measured in the supplied feature space; "
                "PC and PE use saved membership_* columns."
            ),
        },
        "results": rows,
    }
    output_json.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clustering metrics without rerunning clustering."
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        nargs="+",
        required=True,
        help="One or more assignment CSV files containing cluster labels.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        nargs="*",
        default=[],
        help="Optional feature path(s), one per assignment or one broadcast path.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        nargs="*",
        default=[],
        help="Optional embedding JSON path(s) supplying class/hierarchy targets.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("clustering_metrics.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment_paths = args.assignments
    feature_paths = _resolve_paths(
        args.features,
        len(assignment_paths),
        option_name="--features",
    )
    metadata_paths = _resolve_paths(
        args.metadata_json,
        len(assignment_paths),
        option_name="--metadata-json",
    )
    rows = [
        extract_metrics(
            assignment_path,
            features_path=feature_path,
            metadata_path=metadata_path,
        )
        for assignment_path, feature_path, metadata_path in zip(
            assignment_paths,
            feature_paths,
            metadata_paths,
            strict=True,
        )
    ]
    output_json = args.output_json or args.output_csv.with_suffix(".json")
    _write_outputs(rows, args.output_csv, output_json)

    columns = [
        "source",
        "samples",
        "clusters",
        "nmi",
        "ari",
        "silhouette",
        "xb",
        "pc",
        "pe",
        "normalized_pe",
    ]
    print(pd.DataFrame(rows).reindex(columns=columns).to_string(index=False))
    print(f"\nMetrics CSV: {args.output_csv}")
    print(f"Metrics JSON: {output_json}")


if __name__ == "__main__":
    main()
