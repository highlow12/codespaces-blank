"""Measure sampled-K selection across dataset sizes and sample ratios.

The experiment creates four fixed-size datasets from one source embedding file
and, within each dataset, evaluates nested random samples at several ratios.
PCA is fitted once per dataset size and shared by every strategy in that
dataset.  The primary strategy selects K on the sample and refits spherical
FCM on the complete dataset; the project strategy is retained as a faster but
less faithful comparison.  Progress is checkpointed after each completed
selection/fit combination and can be resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances

from clustering_types import FCMResult
from embedding_data import load_embeddings_from_json
from fcm_core import (
    DEFAULT_FCM_MIN_CENTER_SEPARATION,
    DEFAULT_FCM_N_INIT,
    sfcm_memberships_from_centers,
    spherical_fcm,
)
from fcm_validity import select_fcm_cluster_count, xie_beni_index
from incremental_core import atomic_pickle_dump
from pca_projection import fit_normalized_pca_projection


PRIMARY_STRATEGY = "sample_select_full_fit"
STRATEGIES = ("full_selection", PRIMARY_STRATEGY, "sample_select_project")
DEFAULT_TOTAL_SIZES = (100, 300, 1000, 3000)
DEFAULT_SAMPLE_RATIOS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0)
DEFAULT_SELECTION_SEEDS = (42, 43, 44)
DEFAULT_SAMPLE_SEEDS = (42, 43, 44, 45, 46)
CHECKPOINT_FORMAT = "cluster_selection_sample_ratios_checkpoint_v1"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class _CachedResult:
    """Small part of a full-selection result needed after resuming."""

    labels: np.ndarray


@dataclass(frozen=True)
class _CachedSelection:
    """Compact full-data K selection context restored from a checkpoint."""

    n_clusters: int
    result: _CachedResult


def _source_fingerprint(embeddings: np.ndarray, metadata: pd.DataFrame) -> str:
    """Fingerprint loaded input so resume cannot silently mix datasets."""

    values = np.ascontiguousarray(np.asarray(embeddings))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes(order="C"))
    digest.update(
        metadata.reset_index(drop=True)
        .to_json(
            orient="split",
            date_format="iso",
            force_ascii=False,
            default_handler=str,
        )
        .encode("utf-8")
    )
    return digest.hexdigest()


def _checkpoint_configuration(
    args: argparse.Namespace,
    *,
    source_size: int,
    source_fingerprint: str,
) -> dict[str, Any]:
    """Return only the settings that affect benchmark results."""

    return {
        "input_json": str(args.input_json.resolve()),
        "source_size": int(source_size),
        "source_fingerprint": source_fingerprint,
        "total_sizes": list(args.total_sizes),
        "sample_ratios": list(args.sample_ratios),
        "selection_seeds": list(args.seeds),
        "sample_seeds": list(args.sample_seeds),
        "dataset_seed": int(args.dataset_seed),
        "pca_components": int(args.pca_components),
        "pca_seed": int(args.pca_seed),
        "min_clusters": int(args.min_clusters),
        "max_clusters": int(args.max_clusters),
        "min_child_size": int(args.min_child_size),
        "sample_min_child_floor": int(args.sample_min_child_floor),
        "n_init": int(args.n_init),
        "max_attempts": int(args.max_attempts),
        "min_center_separation": float(args.min_center_separation),
    }


def _serialize_full_results(
    full_results_by_dataset: dict[int, dict[int, tuple[Any, str, float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Keep only labels/K for full selections; FCM centers are not needed on resume."""

    serialized: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_size, results in full_results_by_dataset.items():
        dataset_payload: dict[str, dict[str, Any]] = {}
        for selection_seed, (selected, reason, selection_sec) in results.items():
            dataset_payload[str(selection_seed)] = {
                "n_clusters": int(selected.n_clusters),
                "labels": np.asarray(
                    selected.result.labels,
                    dtype=np.int32,
                ).copy(),
                "reason": str(reason),
                "selection_sec": float(selection_sec),
            }
        serialized[str(dataset_size)] = dataset_payload
    return serialized


def _deserialize_full_results(
    payload: Any,
) -> dict[int, dict[int, tuple[_CachedSelection, str, float]]]:
    """Restore compact full-selection contexts from a checkpoint payload."""

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("checkpoint full_results must be a mapping")

    restored: dict[int, dict[int, tuple[_CachedSelection, str, float]]] = {}
    for dataset_key, dataset_payload in payload.items():
        if not isinstance(dataset_payload, dict):
            raise ValueError("checkpoint full_results entries must be mappings")
        dataset_size = int(dataset_key)
        restored_dataset: dict[int, tuple[_CachedSelection, str, float]] = {}
        for seed_key, record in dataset_payload.items():
            if not isinstance(record, dict):
                raise ValueError("checkpoint full selection entries must be mappings")
            labels = np.asarray(record.get("labels"), dtype=np.int32)
            if labels.ndim != 1 or labels.size == 0:
                raise ValueError("checkpoint full selection labels are invalid")
            selected = _CachedSelection(
                n_clusters=int(record["n_clusters"]),
                result=_CachedResult(labels=labels.copy()),
            )
            restored_dataset[int(seed_key)] = (
                selected,
                str(record["reason"]),
                float(record["selection_sec"]),
            )
        restored[dataset_size] = restored_dataset
    return restored


def _checkpoint_payload(
    *,
    configuration: dict[str, Any],
    rows: list[dict[str, Any]],
    pca_timings: dict[str, float],
    dataset_records: list[dict[str, Any]],
    full_results_by_dataset: dict[int, dict[int, tuple[Any, str, float]]],
    completed: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "configuration": configuration,
        "rows": [dict(row) for row in rows],
        "pca_timings": dict(pca_timings),
        "dataset_records": [dict(record) for record in dataset_records],
        "full_results": _serialize_full_results(full_results_by_dataset),
        "completed": bool(completed),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a checksummed checkpoint without exposing a half-written file."""

    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "format": CHECKPOINT_FORMAT,
        "checksum_algorithm": "sha256",
        "checksum": hashlib.sha256(body).hexdigest(),
        "payload_bytes": body,
    }
    atomic_pickle_dump(envelope, path)


def _load_checkpoint(
    path: Path,
    *,
    expected_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Load and validate a checkpoint created by this benchmark."""

    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"checkpoint not found: {path}; run without --resume to start a new run"
        ) from None
    except (OSError, pickle.PickleError, EOFError) as error:
        raise ValueError(f"could not read checkpoint: {path}") from error

    if not isinstance(envelope, dict) or envelope.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid checkpoint format: {path}")
    expected_checksum = envelope.get("checksum")
    body = envelope.get("payload_bytes")
    if not isinstance(expected_checksum, str) or not isinstance(body, bytes):
        raise ValueError(f"invalid checkpoint checksum envelope: {path}")
    actual_checksum = hashlib.sha256(body).hexdigest()
    if not hmac.compare_digest(actual_checksum, expected_checksum):
        raise ValueError(f"checkpoint checksum mismatch: {path}")
    try:
        payload = pickle.loads(body)
    except Exception as error:  # pragma: no cover - malformed pickle detail varies.
        raise ValueError(f"invalid checkpoint payload: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {path}")
    if payload.get("configuration") != expected_configuration:
        raise ValueError(
            "checkpoint configuration does not match this run; "
            "use the original arguments or a new checkpoint path"
        )
    if not isinstance(payload.get("rows"), list):
        raise ValueError("checkpoint rows must be a list")
    if not isinstance(payload.get("dataset_records"), list):
        raise ValueError("checkpoint dataset_records must be a list")
    return payload


def _row_key(
    *,
    dataset_size: int,
    sample_ratio: float,
    sample_seed: int,
    selection_seed: int,
    strategy: str,
) -> tuple[int, float, int, int, str]:
    return (
        int(dataset_size),
        float(sample_ratio),
        int(sample_seed),
        int(selection_seed),
        str(strategy),
    )


def _completed_row_keys(rows: list[dict[str, Any]]) -> set[tuple[int, float, int, int, str]]:
    return {
        _row_key(
            dataset_size=row["dataset_size"],
            sample_ratio=row["sample_ratio"],
            sample_seed=row["sample_seed"],
            selection_seed=row["selection_seed"],
            strategy=row["strategy"],
        )
        for row in rows
    }


def _expected_dataset_row_keys(
    total_size: int,
    args: argparse.Namespace,
) -> set[tuple[int, float, int, int, str]]:
    expected = {
        _row_key(
            dataset_size=total_size,
            sample_ratio=1.0,
            sample_seed=-1,
            selection_seed=selection_seed,
            strategy="full_selection",
        )
        for selection_seed in args.seeds
    }
    for sample_ratio in args.sample_ratios:
        for sample_seed in args.sample_seeds:
            for selection_seed in args.seeds:
                for strategy in (PRIMARY_STRATEGY, "sample_select_project"):
                    expected.add(
                        _row_key(
                            dataset_size=total_size,
                            sample_ratio=sample_ratio,
                            sample_seed=sample_seed,
                            selection_seed=selection_seed,
                            strategy=strategy,
                        )
                    )
    return expected


def _dataset_is_complete(
    total_size: int,
    args: argparse.Namespace,
    completed_keys: set[tuple[int, float, int, int, str]],
) -> bool:
    return _expected_dataset_row_keys(total_size, args).issubset(completed_keys)


def _float_list(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("at least one ratio is required")
    if any(value <= 0.0 or value > 1.0 for value in result):
        raise ValueError("sample ratios must be in (0, 1]")
    return result


def _int_list(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"at least one {name} is required")
    if any(value < 1 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def sample_size_for_ratio(total_size: int, ratio: float) -> int:
    """Round a ratio to a usable, non-empty sample size."""

    if total_size < 1:
        raise ValueError("total_size must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if ratio >= 1.0:
        return total_size
    return max(1, min(total_size, int(round(total_size * ratio))))


def scaled_sample_min_child_size(
    full_min_child_size: int,
    *,
    total_size: int,
    sample_size: int,
    floor: int = 2,
) -> int:
    """Scale the population support constraint into sample-row units.

    A fixed population minimum of 20 would make 5% samples unable to test
    most K values by construction.  The floor keeps tiny samples from
    accepting one-row children while preserving the population proportion.
    """

    if full_min_child_size < 2:
        raise ValueError("full_min_child_size must be at least 2")
    if not 1 <= sample_size <= total_size:
        raise ValueError("sample_size must be within total_size")
    if floor < 2:
        raise ValueError("floor must be at least 2")
    return max(floor, int(np.ceil(full_min_child_size * sample_size / total_size)))


def choose_dataset_indices(
    source_size: int,
    total_size: int,
    *,
    seed: int,
) -> np.ndarray:
    """Choose one reproducible dataset subset from the source rows."""

    if not 1 <= total_size <= source_size:
        raise ValueError("total_size must be within source_size")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(source_size, total_size, replace=False))


def choose_nested_sample_indices(
    total_size: int,
    sample_size: int,
    *,
    seed: int,
) -> np.ndarray:
    """Choose a nested sample; larger ratios retain smaller-ratio rows."""

    if not 1 <= sample_size <= total_size:
        raise ValueError("sample_size must be within total_size")
    if sample_size == total_size:
        return np.arange(total_size, dtype=int)
    permutation = np.random.default_rng(seed).permutation(total_size)
    return np.sort(permutation[:sample_size])


def _select_k(
    features: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int,
    min_child_size: int,
) -> tuple[Any, str, float]:
    started = time.perf_counter()
    best, _candidates, reason = select_fcm_cluster_count(
        features,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_child_size=min_child_size,
        min_membership=0.40,
        selection_method="multi_metric",
        seed=seed,
        n_init=args.n_init,
        max_attempts=args.max_attempts,
        min_center_separation=args.min_center_separation,
    )
    return best, reason, time.perf_counter() - started


def _project_to_centers(features: np.ndarray, selected: FCMResult) -> FCMResult:
    memberships, distances = sfcm_memberships_from_centers(
        features,
        selected.centers,
        m=selected.m,
    )
    centers = np.asarray(selected.centers, dtype=np.float64)
    center_distances = euclidean_distances(centers, centers)
    np.fill_diagonal(center_distances, np.inf)
    return FCMResult(
        labels=memberships.argmax(axis=1),
        memberships=memberships,
        centers=centers,
        iterations=0,
        objective=float(
            np.sum((memberships**selected.m) * (distances**2))
            / len(features)
        ),
        m=selected.m,
        n_init=selected.n_init,
        attempts=selected.attempts,
        valid_restarts=selected.valid_restarts,
        restart_stability=selected.restart_stability,
        minimum_center_distance=float(np.min(center_distances)),
        squared_dissimilarities=distances**2,
    )


def online_refine_sample_centers(
    features: np.ndarray,
    sample_indices: np.ndarray,
    selected: FCMResult,
    *,
    batch_size: int,
    order_seed: int,
) -> FCMResult:
    """Adapt sampled SFCM centers in one streaming pass over held-out rows.

    The sample's fitted memberships initialize cumulative fuzzy sufficient
    statistics. Each subsequent batch is assigned to the current centers,
    added to those statistics, and then updates the centers once. This avoids
    full-data FCM iterations while letting the prototypes represent every row.
    """

    values = np.asarray(features, dtype=np.float64)
    indices = np.asarray(sample_indices, dtype=int)
    centers = np.asarray(selected.centers, dtype=np.float64).copy()
    memberships = np.asarray(selected.memberships, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    if indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("sample_indices must be a non-empty unique 1D array")
    if np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("sample_indices must be within features")
    if memberships.shape != (len(indices), centers.shape[0]):
        raise ValueError("selected memberships must align with sample indices")
    if centers.ndim != 2 or centers.shape[1] != values.shape[1]:
        raise ValueError("selected centers must align with feature dimensions")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    weights = memberships**selected.m
    weighted_sum = weights.T @ values[indices]
    total_weight = weights.sum(axis=0)
    held_out_mask = np.ones(len(values), dtype=bool)
    held_out_mask[indices] = False
    held_out = np.flatnonzero(held_out_mask)
    ordered_held_out = np.random.default_rng(order_seed).permutation(held_out)

    for start in range(0, len(ordered_held_out), batch_size):
        batch_indices = ordered_held_out[start : start + batch_size]
        batch_memberships, _ = sfcm_memberships_from_centers(
            values[batch_indices],
            centers,
            m=selected.m,
        )
        batch_weights = batch_memberships**selected.m
        weighted_sum += batch_weights.T @ values[batch_indices]
        total_weight += batch_weights.sum(axis=0)
        raw_centers = weighted_sum / np.maximum(total_weight[:, None], 1e-12)
        centers = raw_centers / np.maximum(
            np.linalg.norm(raw_centers, axis=1, keepdims=True),
            1e-12,
        )

    all_memberships, distances = sfcm_memberships_from_centers(
        values,
        centers,
        m=selected.m,
    )
    center_distances = euclidean_distances(centers, centers)
    np.fill_diagonal(center_distances, np.inf)
    return FCMResult(
        labels=all_memberships.argmax(axis=1),
        memberships=all_memberships,
        centers=centers,
        iterations=int(np.ceil(len(ordered_held_out) / batch_size)),
        objective=float(
            np.sum((all_memberships**selected.m) * (distances**2)) / len(values)
        ),
        m=selected.m,
        n_init=selected.n_init,
        attempts=selected.attempts,
        valid_restarts=selected.valid_restarts,
        restart_stability=selected.restart_stability,
        minimum_center_distance=float(np.min(center_distances)),
        squared_dissimilarities=distances**2,
    )


def _quality(
    features: np.ndarray,
    metadata: pd.DataFrame,
    result: FCMResult,
    baseline_labels: np.ndarray | None,
) -> dict[str, float | int | None]:
    labels = result.labels
    values: dict[str, float | int | None] = {
        "clusters": int(np.unique(labels).size),
        "silhouette": float(silhouette_score(features, labels, metric="euclidean")),
        "xie_beni": float(
            xie_beni_index(
                features,
                result,
                squared_dissimilarities=result.squared_dissimilarities,
            )
        ),
        "agreement_ari_vs_full_selection": (
            None
            if baseline_labels is None
            else float(adjusted_rand_score(baseline_labels, labels))
        ),
    }
    for column in ("tag", "class"):
        if column in metadata:
            values[f"{column}_nmi"] = float(
                normalized_mutual_info_score(metadata[column], labels)
            )
            values[f"{column}_ari"] = float(
                adjusted_rand_score(metadata[column], labels)
            )
    return values


def _failed_row(
    *,
    dataset_size: int,
    sample_ratio: float,
    sample_size: int,
    sample_seed: int,
    selection_seed: int,
    sample_min_child_size: int,
    full_k: int,
    strategy: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset_size": dataset_size,
        "sample_ratio": sample_ratio,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "selection_seed": selection_seed,
        "sample_min_child_size": sample_min_child_size,
        "full_k": full_k,
        "selected_k": None,
        "k_match": False,
        "status": "selection_failed",
        "selection_reason": reason,
        "strategy": strategy,
        "selection_sec": None,
        "full_fit_sec": None,
        "assignment_sec": None,
        "algorithm_sec": None,
        "end_to_end_sec": None,
    }


def _strategy_row(
    *,
    dataset_size: int,
    sample_ratio: float,
    sample_size: int,
    sample_seed: int,
    selection_seed: int,
    sample_min_child_size: int,
    full_k: int,
    strategy: str,
    selected_k: int,
    reason: str,
    selection_sec: float,
    full_fit_sec: float,
    assignment_sec: float,
    quality_sec: float,
    quality: dict[str, float | int | None],
) -> dict[str, Any]:
    return {
        "dataset_size": dataset_size,
        "sample_ratio": sample_ratio,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "selection_seed": selection_seed,
        "sample_min_child_size": sample_min_child_size,
        "full_k": full_k,
        "selected_k": selected_k,
        "k_match": bool(selected_k == full_k),
        "status": "ok",
        "selection_reason": reason,
        "strategy": strategy,
        "selection_sec": selection_sec,
        "full_fit_sec": full_fit_sec,
        "assignment_sec": assignment_sec,
        "quality_sec": quality_sec,
        "algorithm_sec": selection_sec + full_fit_sec + assignment_sec,
        "end_to_end_sec": selection_sec + full_fit_sec + assignment_sec + quality_sec,
        **quality,
    }


def _summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    group_columns = ["dataset_size", "sample_ratio", "strategy"]
    for group_key, group in frame.groupby(group_columns, sort=True, dropna=False):
        dataset_size, sample_ratio, strategy = group_key
        successful = group[group["status"] == "ok"]
        row: dict[str, Any] = {
            "dataset_size": int(dataset_size),
            "sample_ratio": float(sample_ratio),
            "sample_size": int(group["sample_size"].iloc[0]),
            "strategy": strategy,
            "runs": int(len(group)),
            "successful_runs": int(len(successful)),
            "success_rate": float(len(successful) / len(group)),
        }
        if not successful.empty:
            agreement = successful["agreement_ari_vs_full_selection"].dropna()
            row.update(
                {
                    "mean_selected_k": float(successful["selected_k"].mean()),
                    "std_selected_k": float(successful["selected_k"].std(ddof=0)),
                    "k_match_rate": float(successful["k_match"].mean()),
                    "mean_algorithm_sec": float(successful["algorithm_sec"].mean()),
                    "median_algorithm_sec": float(successful["algorithm_sec"].median()),
                    "mean_silhouette": float(successful["silhouette"].mean()),
                    "mean_xie_beni": float(successful["xie_beni"].mean()),
                }
            )
            if not agreement.empty:
                row["mean_agreement_ari"] = float(agreement.mean())
                row["median_agreement_ari"] = float(agreement.median())
            for column in ("tag_nmi", "tag_ari", "class_nmi", "class_ari"):
                if column in successful:
                    row[f"mean_{column}"] = float(successful[column].mean())
        summary.append(row)
    return summary


def _add_speedups(summary: list[dict[str, Any]]) -> None:
    baselines = {
        int(row["dataset_size"]): row
        for row in summary
        if row["strategy"] == "full_selection"
        and row["successful_runs"] > 0
    }
    for row in summary:
        baseline = baselines.get(int(row["dataset_size"]))
        if baseline is None or "mean_algorithm_sec" not in row:
            continue
        row["speedup_vs_full_selection"] = float(
            baseline["mean_algorithm_sec"] / row["mean_algorithm_sec"]
        )


def _write_outputs(
    args: argparse.Namespace,
    *,
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    pca_timings: dict[str, float],
    dataset_records: list[dict[str, Any]],
    source_fingerprint: str,
    checkpoint_path: Path,
    resumed: bool,
    completed: bool,
) -> None:
    """Write the current partial or final tabular/JSON outputs."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "runs.csv", index=False)
    summary = [] if frame.empty else _summary(frame)
    _add_speedups(summary)
    report = {
        "configuration": {
            "dataset": str(args.input_json),
            "source_samples": int(len(embeddings)),
            "source_fingerprint": source_fingerprint,
            "total_sizes": list(args.total_sizes),
            "sample_ratios": list(args.sample_ratios),
            "selection_seeds": list(args.seeds),
            "sample_seeds": list(args.sample_seeds),
            "dataset_seed": args.dataset_seed,
            "pca_components_requested": args.pca_components,
            "pca_fit_sec_excluded_from_strategy_timing": pca_timings,
            "min_clusters": args.min_clusters,
            "max_clusters": args.max_clusters,
            "full_min_child_size": args.min_child_size,
            "sample_min_child_floor": args.sample_min_child_floor,
            "n_init": args.n_init,
            "max_attempts": args.max_attempts,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_completed": bool(completed),
            "resumed": bool(resumed),
            "strategies": {
                "full_selection": "select K and fit FCM on all rows",
                PRIMARY_STRATEGY: "select K on a sample, then refit FCM on all rows",
                "sample_select_project": "select K and centers on a sample, then assign all rows",
            },
        },
        "datasets": dataset_records,
        "summary": summary,
        "runs": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Progress checkpoint path (default: <output-dir>/checkpoint.pkl).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed work from --checkpoint-path.",
    )
    parser.add_argument("--total-sizes", type=int, nargs="+", default=DEFAULT_TOTAL_SIZES)
    parser.add_argument(
        "--sample-ratios",
        type=float,
        nargs="+",
        default=DEFAULT_SAMPLE_RATIOS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SELECTION_SEEDS)
    parser.add_argument(
        "--sample-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SEEDS,
    )
    parser.add_argument("--dataset-seed", type=int, default=20260811)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--pca-seed", type=int, default=42)
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=20)
    parser.add_argument("--sample-min-child-floor", type=int, default=2)
    parser.add_argument("--n-init", type=int, default=DEFAULT_FCM_N_INIT)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument(
        "--min-center-separation",
        type=float,
        default=DEFAULT_FCM_MIN_CENTER_SEPARATION,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, source_size: int) -> None:
    args.total_sizes = _int_list(args.total_sizes, name="total sizes")
    args.sample_ratios = _float_list(args.sample_ratios)
    args.seeds = _int_list(args.seeds, name="selection seeds")
    args.sample_seeds = _int_list(args.sample_seeds, name="sample seeds")
    if any(size > source_size for size in args.total_sizes):
        raise ValueError("every total size must be within the input dataset")
    if args.min_child_size < 2:
        raise ValueError("--min-child-size must be at least 2")
    if args.sample_min_child_floor < 2:
        raise ValueError("--sample-min-child-floor must be at least 2")


def main() -> None:
    args = parse_args()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    _validate_args(args, len(embeddings))

    args.checkpoint_path = args.checkpoint_path or args.output_dir / "checkpoint.pkl"
    source_fingerprint = _source_fingerprint(embeddings, metadata)
    configuration = _checkpoint_configuration(
        args,
        source_size=len(embeddings),
        source_fingerprint=source_fingerprint,
    )

    rows: list[dict[str, Any]]
    pca_timings: dict[str, float]
    dataset_records: list[dict[str, Any]]
    full_results_by_dataset: dict[int, dict[int, tuple[Any, str, float]]]
    resumed = bool(args.resume)
    if args.resume:
        checkpoint = _load_checkpoint(
            args.checkpoint_path,
            expected_configuration=configuration,
        )
        rows = [dict(row) for row in checkpoint["rows"]]
        pca_timings = {
            str(dataset_size): float(seconds)
            for dataset_size, seconds in checkpoint.get("pca_timings", {}).items()
        }
        dataset_records = [dict(record) for record in checkpoint["dataset_records"]]
        full_results_by_dataset = _deserialize_full_results(
            checkpoint.get("full_results")
        )
        if checkpoint.get("completed", False):
            _write_outputs(
                args,
                embeddings=embeddings,
                rows=rows,
                pca_timings=pca_timings,
                dataset_records=dataset_records,
                source_fingerprint=source_fingerprint,
                checkpoint_path=args.checkpoint_path,
                resumed=True,
                completed=True,
            )
            print(f"Checkpoint already complete: {args.checkpoint_path}", flush=True)
            return
    else:
        rows = []
        pca_timings = {}
        dataset_records = []
        full_results_by_dataset = {}

    completed_keys = _completed_row_keys(rows)

    def persist(*, completed: bool = False) -> None:
        _save_checkpoint(
            args.checkpoint_path,
            _checkpoint_payload(
                configuration=configuration,
                rows=rows,
                pca_timings=pca_timings,
                dataset_records=dataset_records,
                full_results_by_dataset=full_results_by_dataset,
                completed=completed,
            ),
        )
        _write_outputs(
            args,
            embeddings=embeddings,
            rows=rows,
            pca_timings=pca_timings,
            dataset_records=dataset_records,
            source_fingerprint=source_fingerprint,
            checkpoint_path=args.checkpoint_path,
            resumed=resumed,
            completed=completed,
        )

    if not resumed:
        # Create a valid initial checkpoint before the first expensive fit.
        persist()

    for total_size in args.total_sizes:
        if _dataset_is_complete(total_size, args, completed_keys):
            if resumed:
                print(f"N={total_size} already complete; skipping", flush=True)
            continue

        dataset_indices = choose_dataset_indices(
            len(embeddings),
            total_size,
            seed=args.dataset_seed + total_size,
        )
        dataset_embeddings = embeddings[dataset_indices]
        dataset_metadata = metadata.iloc[dataset_indices].reset_index(drop=True)
        pca_started = time.perf_counter()
        fitted = fit_normalized_pca_projection(
            dataset_embeddings,
            n_components=args.pca_components,
            seed=args.pca_seed,
        )
        features = fitted.normalized_prefix()
        pca_timings[str(total_size)] = time.perf_counter() - pca_started
        dataset_record = {
            "dataset_size": total_size,
            "source_indices_sha256": hashlib.sha256(
                dataset_indices.tobytes()
            ).hexdigest(),
            "pca_components": int(fitted.pca.n_components_),
        }
        existing_record = next(
            (
                index
                for index, record in enumerate(dataset_records)
                if int(record.get("dataset_size", -1)) == total_size
            ),
            None,
        )
        if existing_record is None:
            dataset_records.append(dataset_record)
        else:
            dataset_records[existing_record] = dataset_record

        full_results = full_results_by_dataset.setdefault(total_size, {})
        for selection_seed in args.seeds:
            full_key = _row_key(
                dataset_size=total_size,
                sample_ratio=1.0,
                sample_seed=-1,
                selection_seed=selection_seed,
                strategy="full_selection",
            )
            cached = full_results.get(selection_seed)
            cache_is_usable = (
                cached is not None
                and len(np.asarray(cached[0].result.labels)) == len(features)
            )
            if full_key in completed_keys and cache_is_usable:
                full_best, full_reason, full_selection_sec = cached
            else:
                full_best, full_reason, full_selection_sec = _select_k(
                    features,
                    args,
                    seed=selection_seed,
                    min_child_size=args.min_child_size,
                )
                if full_best is None:
                    raise RuntimeError(
                        f"full selection failed for dataset_size={total_size}, "
                        f"seed={selection_seed}: {full_reason}"
                    )
                full_results[selection_seed] = (
                    full_best,
                    full_reason,
                    full_selection_sec,
                )
                if full_key not in completed_keys:
                    quality_started = time.perf_counter()
                    full_quality = _quality(
                        features,
                        dataset_metadata,
                        full_best.result,
                        None,
                    )
                    quality_sec = time.perf_counter() - quality_started
                    rows.append(
                        _strategy_row(
                            dataset_size=total_size,
                            sample_ratio=1.0,
                            sample_size=total_size,
                            sample_seed=-1,
                            selection_seed=selection_seed,
                            sample_min_child_size=args.min_child_size,
                            full_k=full_best.n_clusters,
                            strategy="full_selection",
                            selected_k=full_best.n_clusters,
                            reason=full_reason,
                            selection_sec=full_selection_sec,
                            full_fit_sec=0.0,
                            assignment_sec=0.0,
                            quality_sec=quality_sec,
                            quality=full_quality,
                        )
                    )
                    completed_keys.add(full_key)
                # Persist the compact full-selection context even if only the
                # row was missing from an older/hand-edited checkpoint.
                persist()

        for sample_ratio in args.sample_ratios:
            sample_size = sample_size_for_ratio(total_size, sample_ratio)
            sample_min_child_size = scaled_sample_min_child_size(
                args.min_child_size,
                total_size=total_size,
                sample_size=sample_size,
                floor=args.sample_min_child_floor,
            )
            for sample_seed in args.sample_seeds:
                sample_indices = choose_nested_sample_indices(
                    total_size,
                    sample_size,
                    seed=sample_seed,
                )
                sampled_features = features[sample_indices]
                for selection_seed in args.seeds:
                    primary_key = _row_key(
                        dataset_size=total_size,
                        sample_ratio=sample_ratio,
                        sample_seed=sample_seed,
                        selection_seed=selection_seed,
                        strategy=PRIMARY_STRATEGY,
                    )
                    project_key = _row_key(
                        dataset_size=total_size,
                        sample_ratio=sample_ratio,
                        sample_seed=sample_seed,
                        selection_seed=selection_seed,
                        strategy="sample_select_project",
                    )
                    if primary_key in completed_keys and project_key in completed_keys:
                        continue

                    full_best, _full_reason, _full_selection_sec = full_results[
                        selection_seed
                    ]
                    sampled_best, sample_reason, sample_selection_sec = _select_k(
                        sampled_features,
                        args,
                        seed=selection_seed,
                        min_child_size=sample_min_child_size,
                    )
                    if sampled_best is None:
                        for strategy in (PRIMARY_STRATEGY, "sample_select_project"):
                            key = primary_key if strategy == PRIMARY_STRATEGY else project_key
                            if key not in completed_keys:
                                rows.append(
                                    _failed_row(
                                        dataset_size=total_size,
                                        sample_ratio=sample_ratio,
                                        sample_size=sample_size,
                                        sample_seed=sample_seed,
                                        selection_seed=selection_seed,
                                        sample_min_child_size=sample_min_child_size,
                                        full_k=full_best.n_clusters,
                                        strategy=strategy,
                                        reason=sample_reason,
                                    )
                                )
                                completed_keys.add(key)
                        persist()
                        continue

                    if primary_key not in completed_keys:
                        fit_started = time.perf_counter()
                        refit = spherical_fcm(
                            features,
                            n_clusters=sampled_best.n_clusters,
                            seed=selection_seed + sampled_best.n_clusters * 1009,
                            m=sampled_best.m,
                            n_init=args.n_init,
                            max_attempts=args.max_attempts,
                            min_cluster_size=args.min_child_size,
                            min_center_separation=args.min_center_separation,
                        )
                        refit_sec = time.perf_counter() - fit_started
                        quality_started = time.perf_counter()
                        refit_quality = _quality(
                            features,
                            dataset_metadata,
                            refit,
                            full_best.result.labels,
                        )
                        refit_quality_sec = time.perf_counter() - quality_started
                        rows.append(
                            _strategy_row(
                                dataset_size=total_size,
                                sample_ratio=sample_ratio,
                                sample_size=sample_size,
                                sample_seed=sample_seed,
                                selection_seed=selection_seed,
                                sample_min_child_size=sample_min_child_size,
                                full_k=full_best.n_clusters,
                                strategy=PRIMARY_STRATEGY,
                                selected_k=sampled_best.n_clusters,
                                reason=sample_reason,
                                selection_sec=sample_selection_sec,
                                full_fit_sec=refit_sec,
                                assignment_sec=0.0,
                                quality_sec=refit_quality_sec,
                                quality=refit_quality,
                            )
                        )
                        completed_keys.add(primary_key)

                    if project_key not in completed_keys:
                        assignment_started = time.perf_counter()
                        projected = _project_to_centers(features, sampled_best.result)
                        assignment_sec = time.perf_counter() - assignment_started
                        quality_started = time.perf_counter()
                        projected_quality = _quality(
                            features,
                            dataset_metadata,
                            projected,
                            full_best.result.labels,
                        )
                        projected_quality_sec = time.perf_counter() - quality_started
                        rows.append(
                            _strategy_row(
                                dataset_size=total_size,
                                sample_ratio=sample_ratio,
                                sample_size=sample_size,
                                sample_seed=sample_seed,
                                selection_seed=selection_seed,
                                sample_min_child_size=sample_min_child_size,
                                full_k=full_best.n_clusters,
                                strategy="sample_select_project",
                                selected_k=sampled_best.n_clusters,
                                reason=sample_reason,
                                selection_sec=sample_selection_sec,
                                full_fit_sec=0.0,
                                assignment_sec=assignment_sec,
                                quality_sec=projected_quality_sec,
                                quality=projected_quality,
                            )
                        )
                        completed_keys.add(project_key)
                    persist()
                    print(
                        f"N={total_size} ratio={sample_ratio:.2f} "
                        f"sample={sample_size} sample_seed={sample_seed} "
                        f"selection_seed={selection_seed} "
                        f"K={sampled_best.n_clusters}/{full_best.n_clusters}",
                        flush=True,
                    )

    persist(completed=True)
    print(f"Saved {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
