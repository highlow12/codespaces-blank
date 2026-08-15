"""Benchmark production hierarchical clustering under several fuzzifier policies.

The benchmark intentionally calls the public incremental fit path with
``fit_visualization=False``.  This keeps the hierarchy, PCA, consensus, and
Fast settings equal to an operational ``incremental_clustering fit`` while
leaving out UMAP and all incremental updates.  The full Gemini dataset is
required by default; use ``--output-dir`` to select a dated result directory.

The twelve runs (four policies x three seeds) are checkpointed after every
completed fit.  A run can therefore be resumed after an interruption without
re-fitting completed condition/seed pairs.  Compact assignment files are
written so seed-to-seed and base-to-Fast partition ARIs remain auditable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from clustering_defaults import DEFAULT_FAST_FUZZIFIER_VALUES
from embedding_data import load_embeddings_from_json
from incremental_clustering import fit_incremental_state


DEFAULT_INPUT = Path("dbpedia_gemini_embeddings.json.gz")
DEFAULT_SEEDS = (42, 43, 44)
EXPECTED_ROWS = 3_000

# These values are the current public incremental fit defaults.  They are
# passed explicitly so a future change to a lower-level helper cannot silently
# change the historical comparison.  The resulting state.config is also
# copied into each run's record for an exact audit trail.
COMMON_FIT_KWARGS: dict[str, Any] = {
    "max_depth": 4,
    "min_node_size": 60,
    "min_child_size": 20,
    "min_clusters": 2,
    "max_clusters": 4,
    "min_membership": 0.20,
    "max_membership_gap": 0.10,
    "forced_noise_ratio": 0.0,
    "distance_z": 3.5,
    "selection_method": "multi_metric",
    "min_xb_relative_improvement": 0.05,
    "xb_worsening_patience": 2,
    "min_split_silhouette": 0.05,
    "pca_components": None,
    "pca_max_components": 512,
    "pca_min_components": 32,
    "pca_component_step": 32,
    "pca_k_values": (15, 30),
    "pca_minimum_preservation_gain": 0.05,
    "noise_threshold": 0.05,
    "noise_release_threshold": None,
    "drift_min_samples": 20,
    "drift_ewma_alpha": 0.30,
    "recluster_cooldown_updates": 3,
    "visual_pca_components": None,
    "visual_cluster_target_weight": 0.01,
    "visual_n_neighbors": 24,
    "visual_min_dist": 1.0,
    "visual_metric": "euclidean",
    "visual_spread": 1.8,
    "visual_densmap": False,
    "center_updates_before_membership_refresh": 10,
    "selective_membership_refresh": True,
    "membership_refresh_min_center_movement": 0.01,
    "membership_refresh_min_influence": 0.05,
    "max_xb_relative_degradation": 0.05,
    "max_fcm_iter": 200,
    "fcm_tol": 1e-6,
    "embedding_storage_dtype": "float32",
    "include_conditional_memberships": False,
    "fast_sample_size": 1000,
    "fast_scout_n_init": 2,
    "fast_refine_n_init": 3,
    "fast_refine_top_k": 2,
    "fast_stability_target": 0.85,
    "fast_m_values": tuple(DEFAULT_FAST_FUZZIFIER_VALUES),
    "fast_reuse_scout_m": True,
    "consensus_k_selection": True,
    "consensus_min_rows": 500,
    "consensus_sample_ratio": 0.20,
    "consensus_max_scouts": 5,
    "consensus_vote_threshold": 3,
    "consensus_scout_n_init": 3,
    "fit_visualization": False,
}


CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "consensus_fixed_m_2_0",
        "label": "basic/consensus fixed m=2.0",
        "path": "basic_consensus",
        "fast_mode": False,
        "fuzzifier": 2.0,
        "fuzzifier_request": 2.0,
    },
    {
        "name": "consensus_fixed_m_1_2",
        "label": "basic/consensus fixed m=1.2",
        "path": "basic_consensus",
        "fast_mode": False,
        "fuzzifier": 1.2,
        "fuzzifier_request": 1.2,
    },
    {
        "name": "consensus_auto_m",
        "label": "basic/consensus automatic m",
        "path": "basic_consensus",
        "fast_mode": False,
        "fuzzifier": None,
        "fuzzifier_request": "auto",
    },
    {
        "name": "fast_auto_m",
        "label": "Fast automatic m",
        "path": "fast",
        "fast_mode": True,
        "fuzzifier": None,
        "fuzzifier_request": "auto",
    },
)


def _json_safe(value: Any) -> Any:
    """Convert numpy/path values and non-finite floats for JSON output."""

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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json_write(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temporary file so progress is resumable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_by_name(name: str) -> dict[str, Any]:
    for condition in CONDITIONS:
        if condition["name"] == name:
            return condition
    raise KeyError(name)


def _run_id(condition_name: str, seed: int) -> str:
    return f"{condition_name}__seed_{int(seed)}"


def _assignment_columns(
    assignments: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return top labels, leaf paths, noise flags, and boundary flags."""

    if "level_1_cluster" not in assignments.columns:
        top = np.full(len(assignments), -1, dtype=int)
    else:
        top = (
            pd.to_numeric(assignments["level_1_cluster"], errors="coerce")
            .fillna(-1)
            .to_numpy(dtype=int)
        )
    if "cluster_path" not in assignments.columns:
        leaf = np.full(len(assignments), "noise", dtype=object)
    else:
        leaf = assignments["cluster_path"].fillna("noise").astype(str).to_numpy()
    noise = (
        assignments["is_noise"].astype(bool).to_numpy()
        if "is_noise" in assignments.columns
        else np.zeros(len(assignments), dtype=bool)
    )
    boundary = (
        assignments["is_boundary"].astype(bool).to_numpy()
        if "is_boundary" in assignments.columns
        else np.zeros(len(assignments), dtype=bool)
    )
    return top, leaf, noise, boundary


def _external_metrics(
    metadata: pd.DataFrame,
    top_labels: np.ndarray,
    leaf_labels: np.ndarray,
) -> dict[str, float]:
    """Measure top (tag) and leaf (class) external quality."""

    if "tag" not in metadata.columns or "class" not in metadata.columns:
        raise ValueError("Gemini metadata must contain tag and class columns")
    top_target = metadata["tag"].astype(str).to_numpy()
    leaf_target = metadata["class"].astype(str).to_numpy()
    return {
        "top_nmi": float(normalized_mutual_info_score(top_target, top_labels)),
        "top_ari": float(adjusted_rand_score(top_target, top_labels)),
        "leaf_nmi": float(normalized_mutual_info_score(leaf_target, leaf_labels)),
        "leaf_ari": float(adjusted_rand_score(leaf_target, leaf_labels)),
    }


def _root_selected_record(root: dict[str, Any]) -> dict[str, Any] | None:
    """Pick restart metrics for the candidate actually used at the root."""

    selected_k = root.get("selected_k")
    selected_m = root.get("selected_m")
    if selected_k is None:
        return None
    records = root.get("candidate_metrics", [])
    if not isinstance(records, list):
        return None
    phase_priority = {
        "consensus_full_fit": 50,
        "consensus_full_fallback": 45,
        "consensus_full_data_direct": 45,
        "refine": 40,
        "full": 35,
        "m_probe": 10,
    }
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        try:
            if int(record.get("k")) != int(selected_k):
                continue
        except (TypeError, ValueError):
            continue
        record_m = record.get("m")
        if selected_m is not None and record_m is not None:
            try:
                if not np.isclose(float(record_m), float(selected_m)):
                    continue
            except (TypeError, ValueError):
                continue
        phase = str(record.get("phase", ""))
        matches.append((phase_priority.get(phase, 0), index, record))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[-1][2]


def _state_pickle_size(state: Any) -> int | None:
    """Return the serializable state size, if the state can be pickled."""

    try:
        return int(len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)))
    except (pickle.PickleError, TypeError, ValueError):
        return None


def _compact_assignment_frame(
    assignments: pd.DataFrame,
    top_labels: np.ndarray,
    leaf_labels: np.ndarray,
    noise: np.ndarray,
    boundary: np.ndarray,
) -> pd.DataFrame:
    identifier = (
        assignments["id"].to_numpy()
        if "id" in assignments.columns
        else np.arange(len(assignments))
    )
    return pd.DataFrame(
        {
            "id": identifier,
            "top_cluster": top_labels,
            "leaf_cluster_path": leaf_labels,
            "is_noise": noise,
            "is_boundary": boundary,
        }
    )


def _save_compact_assignments(
    path: Path,
    assignments: pd.DataFrame,
    top_labels: np.ndarray,
    leaf_labels: np.ndarray,
    noise: np.ndarray,
    boundary: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _compact_assignment_frame(
        assignments,
        top_labels,
        leaf_labels,
        noise,
        boundary,
    )
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _load_compact_assignments(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, compression="gzip")
    top = frame["top_cluster"].to_numpy(dtype=int)
    leaf = frame["leaf_cluster_path"].astype(str).to_numpy()
    return top, leaf


def _fit_run(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    condition: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    fit_kwargs = dict(COMMON_FIT_KWARGS)
    fit_kwargs.update(
        {
            "seed": int(seed),
            "fast_mode": bool(condition["fast_mode"]),
            "fuzzifier": condition["fuzzifier"],
        }
    )
    state = fit_incremental_state(embeddings, metadata, **fit_kwargs)
    elapsed = float(time.perf_counter() - started)
    assignments = state.assignments
    top_labels, leaf_labels, noise, boundary = _assignment_columns(assignments)
    metrics = _external_metrics(metadata, top_labels, leaf_labels)
    tree = state.tree
    root = tree.get("root", {}) if isinstance(tree, dict) else {}
    summary = tree.get("summary", {}) if isinstance(tree, dict) else {}
    root_record = _root_selected_record(root)
    root_k = root.get("selected_k")
    root_m = root.get("selected_m")
    selected_m = state.config.get("m")
    if root_m is not None:
        selected_m = root_m
    n_rows = len(assignments)
    assignment_file = (
        output_dir
        / "assignments"
        / f"{condition['name']}__seed_{int(seed)}.csv.gz"
    )
    _save_compact_assignments(
        assignment_file,
        assignments,
        top_labels,
        leaf_labels,
        noise,
        boundary,
    )
    record: dict[str, Any] = {
        "run_id": _run_id(condition["name"], seed),
        "condition": condition["name"],
        "condition_label": condition["label"],
        "path": condition["path"],
        "seed": int(seed),
        "samples": int(n_rows),
        "fuzzifier_requested": condition["fuzzifier_request"],
        "selected_m": None if selected_m is None else float(selected_m),
        "state_config_m": state.config.get("m"),
        "root_k": None if root_k is None else int(root_k),
        "root_selection_reason": root.get("selection_reason"),
        "root_restart_stability": (
            None
            if root_record is None or root_record.get("restart_stability") is None
            else float(root_record["restart_stability"])
        ),
        "root_valid_restarts": (
            None
            if root_record is None or root_record.get("valid_restarts") is None
            else int(root_record["valid_restarts"])
        ),
        "root_candidate_phase": (
            None if root_record is None else root_record.get("phase")
        ),
        "hierarchy_leaf_count": int(summary.get("leaf_cluster_count", 0)),
        "structural_leaf_count": int(summary.get("leaf_count", 0)),
        "hierarchy_depth": int(summary.get("levels_reached", 0)),
        "noise_count": int(noise.sum()),
        "boundary_count": int(boundary.sum()),
        "noise_ratio": float(noise.mean()),
        "boundary_ratio": float(boundary.mean()),
        "natural_noise_ratio": float(
            summary.get("natural_noise_count", noise.sum()) / max(n_rows, 1)
        ),
        "forced_noise_ratio_observed": float(
            summary.get("forced_noise_count", 0) / max(n_rows, 1)
        ),
        "runtime_sec": elapsed,
        "state_pickle_bytes": _state_pickle_size(state),
        "assignment_file": str(assignment_file.relative_to(output_dir)),
        **metrics,
    }
    record["quality_mean"] = float(
        np.mean([metrics["top_nmi"], metrics["top_ari"], metrics["leaf_nmi"], metrics["leaf_ari"]])
    )
    # Drop the large PCA/state objects before the next condition while keeping
    # only the compact labels needed for pairwise comparisons.
    del state
    gc.collect()
    return record, top_labels, leaf_labels


def _pairwise_seed_aris(
    records: Iterable[dict[str, Any]],
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_condition.setdefault(str(record["condition"]), []).append(record)
    output: dict[str, Any] = {}
    for condition, condition_records in by_condition.items():
        condition_records.sort(key=lambda item: int(item["seed"]))
        pairs: list[dict[str, Any]] = []
        top_scores: list[float] = []
        leaf_scores: list[float] = []
        for left_index in range(len(condition_records)):
            for right_index in range(left_index + 1, len(condition_records)):
                left = condition_records[left_index]
                right = condition_records[right_index]
                left_top, left_leaf = partitions[left["run_id"]]
                right_top, right_leaf = partitions[right["run_id"]]
                top_ari = float(adjusted_rand_score(left_top, right_top))
                leaf_ari = float(adjusted_rand_score(left_leaf, right_leaf))
                top_scores.append(top_ari)
                leaf_scores.append(leaf_ari)
                pairs.append(
                    {
                        "seed_left": int(left["seed"]),
                        "seed_right": int(right["seed"]),
                        "top_ari": top_ari,
                        "leaf_ari": leaf_ari,
                    }
                )
        output[condition] = {
            "pairs": pairs,
            "mean_top_ari": None if not top_scores else float(np.mean(top_scores)),
            "mean_leaf_ari": None if not leaf_scores else float(np.mean(leaf_scores)),
        }
    return output


def _cross_condition_aris(
    records: Iterable[dict[str, Any]],
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    by_condition_seed = {
        (str(record["condition"]), int(record["seed"])): record
        for record in records
    }
    pairs: list[dict[str, Any]] = []
    for seed in DEFAULT_SEEDS:
        baseline = by_condition_seed.get(("consensus_auto_m", seed))
        fast = by_condition_seed.get(("fast_auto_m", seed))
        if baseline is None or fast is None:
            continue
        baseline_top, baseline_leaf = partitions[baseline["run_id"]]
        fast_top, fast_leaf = partitions[fast["run_id"]]
        pairs.append(
            {
                "seed": int(seed),
                "top_ari": float(adjusted_rand_score(baseline_top, fast_top)),
                "leaf_ari": float(adjusted_rand_score(baseline_leaf, fast_leaf)),
            }
        )
    return {
        "baseline_condition": "consensus_auto_m",
        "fast_condition": "fast_auto_m",
        "pairs": pairs,
        "mean_top_ari": (
            None if not pairs else float(np.mean([pair["top_ari"] for pair in pairs]))
        ),
        "mean_leaf_ari": (
            None if not pairs else float(np.mean([pair["leaf_ari"] for pair in pairs]))
        ),
    }


def _summary_by_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "selected_m",
        "root_k",
        "root_restart_stability",
        "hierarchy_leaf_count",
        "structural_leaf_count",
        "hierarchy_depth",
        "noise_ratio",
        "boundary_ratio",
        "top_nmi",
        "top_ari",
        "leaf_nmi",
        "leaf_ari",
        "quality_mean",
        "runtime_sec",
        "state_pickle_bytes",
    )
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row["condition"] == condition["name"]]
        summary: dict[str, Any] = {
            "label": condition["label"],
            "run_count": len(rows),
            "selected_m_values": sorted(
                {
                    float(row["selected_m"])
                    for row in rows
                    if row.get("selected_m") is not None
                }
            ),
        }
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in rows
                if row.get(field) is not None and np.isfinite(float(row[field]))
            ]
            summary[f"mean_{field}"] = None if not values else float(np.mean(values))
            summary[f"median_{field}"] = None if not values else float(np.median(values))
        output[condition["name"]] = summary
    return output


def _analysis(summary: dict[str, Any], cross_ari: dict[str, Any]) -> dict[str, Any]:
    """Generate a small machine-readable conclusion from the completed runs."""

    quality = {
        condition: values.get("mean_quality_mean")
        for condition, values in summary.items()
        if values.get("mean_quality_mean") is not None
    }
    best_quality = max(quality, key=quality.get) if quality else None
    base = summary.get("consensus_auto_m", {})
    fast = summary.get("fast_auto_m", {})
    base_time = base.get("mean_runtime_sec")
    fast_time = fast.get("mean_runtime_sec")
    speedup = (
        None
        if not base_time or not fast_time
        else float(base_time / fast_time)
    )
    quality_delta = (
        None
        if base.get("mean_quality_mean") is None or fast.get("mean_quality_mean") is None
        else float(fast["mean_quality_mean"] - base["mean_quality_mean"])
    )
    return {
        "best_mean_quality_condition": best_quality,
        "best_mean_quality": None if best_quality is None else quality[best_quality],
        "fast_vs_consensus_auto_runtime_speedup": speedup,
        "fast_vs_consensus_auto_quality_mean_delta": quality_delta,
        "fast_vs_consensus_auto_mean_top_ari": cross_ari.get("mean_top_ari"),
        "fast_vs_consensus_auto_mean_leaf_ari": cross_ari.get("mean_leaf_ari"),
        "interpretation": (
            "Quality mean is the unweighted mean of top/leaf NMI and ARI; "
            "cross-condition ARI compares the leaf/top partitions on each same seed."
        ),
    }


def _configuration(
    input_path: Path,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    input_sha256: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "benchmark": "production_m_fuzzifier_comparison",
        "input_json": str(input_path),
        "input_sha256": input_sha256,
        "records": int(len(embeddings)),
        "embedding_dimension": int(embeddings.shape[1]),
        "metadata_columns": [str(column) for column in metadata.columns],
        "top_ground_truth": "metadata.tag (the first class hierarchy level)",
        "leaf_ground_truth": "metadata.class (the DBpedia leaf class)",
        "predicted_top_partition": "assignments.level_1_cluster",
        "predicted_leaf_partition": "assignments.cluster_path (noise is a separate path)",
        "seeds": [int(seed) for seed in seeds],
        "visualization": {
            "fit_visualization": False,
            "reason": "clustering-only benchmark; UMAP/plot outputs are excluded",
        },
        "incremental_updates": {
            "performed": False,
            "reason": "this compares initial clustering only; no update batches are applied",
        },
        "public_incremental_fit_defaults": {
            "max_clusters": 4,
            "max_depth": 4,
            "min_node_size": 60,
            "min_child_size": 20,
        },
        "lower_level_hierarchical_function_default": {
            "max_clusters": 8,
            "note": (
                "run_hierarchical_pca_fcm itself defaults to max_clusters=8; "
                "this benchmark uses the public incremental fit setting max_clusters=4"
            ),
        },
        "hierarchy": {
            "max_depth": COMMON_FIT_KWARGS["max_depth"],
            "min_node_size": COMMON_FIT_KWARGS["min_node_size"],
            "min_child_size": COMMON_FIT_KWARGS["min_child_size"],
            "min_clusters": COMMON_FIT_KWARGS["min_clusters"],
            "max_clusters": COMMON_FIT_KWARGS["max_clusters"],
            "selection_method": COMMON_FIT_KWARGS["selection_method"],
            "pca_components": "auto",
            "pca_max_components": COMMON_FIT_KWARGS["pca_max_components"],
            "embedding_storage_dtype": COMMON_FIT_KWARGS["embedding_storage_dtype"],
        },
        "consensus": {
            "enabled_in_all_conditions": True,
            "min_rows": COMMON_FIT_KWARGS["consensus_min_rows"],
            "sample_ratio": COMMON_FIT_KWARGS["consensus_sample_ratio"],
            "max_scouts": COMMON_FIT_KWARGS["consensus_max_scouts"],
            "vote_threshold": COMMON_FIT_KWARGS["consensus_vote_threshold"],
            "scout_n_init": COMMON_FIT_KWARGS["consensus_scout_n_init"],
            "description": (
                "Nodes with at least 500 rows use sampled consensus K selection; "
                "smaller nodes use the regular selector"
            ),
        },
        "fast": {
            "enabled_only_in": "fast_auto_m",
            "sample_size": COMMON_FIT_KWARGS["fast_sample_size"],
            "scout_n_init": COMMON_FIT_KWARGS["fast_scout_n_init"],
            "refine_n_init": COMMON_FIT_KWARGS["fast_refine_n_init"],
            "refine_top_k": COMMON_FIT_KWARGS["fast_refine_top_k"],
            "stability_target": COMMON_FIT_KWARGS["fast_stability_target"],
            "m_values": list(COMMON_FIT_KWARGS["fast_m_values"]),
            "reuse_scout_m": COMMON_FIT_KWARGS["fast_reuse_scout_m"],
        },
        "common_fcm": {
            "max_iter": COMMON_FIT_KWARGS["max_fcm_iter"],
            "tol": COMMON_FIT_KWARGS["fcm_tol"],
            "min_membership": COMMON_FIT_KWARGS["min_membership"],
            "max_membership_gap": COMMON_FIT_KWARGS["max_membership_gap"],
            "forced_noise_ratio": COMMON_FIT_KWARGS["forced_noise_ratio"],
        },
        "conditions": [
            {
                "name": condition["name"],
                "label": condition["label"],
                "path": condition["path"],
                "fast_mode": condition["fast_mode"],
                "fuzzifier_request": condition["fuzzifier_request"],
                "difference_from_basic_consensus": (
                    "Fast bounded scout/refine K selection; same public max_k, "
                    "hierarchy, and consensus settings"
                    if condition["fast_mode"]
                    else "none"
                ),
            }
            for condition in CONDITIONS
        ],
        "state_size": {
            "recorded": True,
            "field": "runs[].state_pickle_bytes",
            "meaning": (
                "HIGHEST_PROTOCOL pickle size of the in-memory clustering-only "
                "state; it still contains full float32 embeddings, assignments, "
                "PCA, and hierarchy models, so it is not an output/checkpoint size"
            ),
            "checkpoint_written": False,
        },
    }


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return {
        str(record["run_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("run_id")
    }


def _write_progress(
    path: Path,
    configuration: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> None:
    _atomic_json_write(
        path,
        {
            "version": 1,
            "configuration": configuration,
            "records": list(records.values()),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Model seeds; the production comparison uses 42 43 44.",
    )
    parser.add_argument(
        "--condition",
        dest="conditions",
        action="append",
        choices=[condition["name"] for condition in CONDITIONS],
        help="Restrict a run to selected conditions; repeat the option as needed.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore progress.json and re-fit all selected condition/seed pairs.",
    )
    parser.add_argument(
        "--allow-row-count",
        action="store_true",
        help="Allow a dataset other than the required 3,000 records (testing only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    benchmark_started = time.perf_counter()
    input_path = args.input_json.expanduser()
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else Path("benchmarks")
        / f"production-m-fuzzifier-{datetime.now(timezone.utc).date().isoformat()}"
    )
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    seeds = tuple(int(seed) for seed in args.seed)
    if not seeds:
        raise ValueError("at least one seed is required")
    selected_names = args.conditions or [condition["name"] for condition in CONDITIONS]
    selected_conditions = [_condition_by_name(name) for name in selected_names]

    print(f"Loading full Gemini dataset: {input_path}", flush=True)
    embeddings, metadata = load_embeddings_from_json(input_path)
    if len(embeddings) != EXPECTED_ROWS and not args.allow_row_count:
        raise ValueError(
            f"Expected the full {EXPECTED_ROWS}-record Gemini dataset, got {len(embeddings)}"
        )
    if embeddings.shape[1] != 3072:
        raise ValueError(
            f"Expected 3,072-dimensional Gemini embeddings, got {embeddings.shape[1]}"
        )
    input_sha256 = _sha256(input_path)
    configuration = _configuration(
        input_path,
        embeddings,
        metadata,
        input_sha256,
        seeds,
    )
    configuration["selected_conditions"] = [
        condition["name"] for condition in selected_conditions
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    completed = {} if args.force_rerun else _load_progress(progress_path)
    partitions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Reuse compact assignment artifacts from a previous partial run so the
    # final pairwise ARI report remains complete after a resumed invocation.
    for record in list(completed.values()):
        assignment_file = output_dir / str(record.get("assignment_file", ""))
        if assignment_file.exists():
            partitions[str(record["run_id"])] = _load_compact_assignments(assignment_file)

    total = len(selected_conditions) * len(seeds)
    run_index = 0
    for condition in selected_conditions:
        for seed in seeds:
            run_index += 1
            run_id = _run_id(condition["name"], seed)
            existing = completed.get(run_id)
            assignment_file = (
                output_dir
                / "assignments"
                / f"{condition['name']}__seed_{int(seed)}.csv.gz"
            )
            if existing is not None and assignment_file.exists():
                print(f"[{run_index}/{total}] resume {run_id}", flush=True)
                continue
            print(
                f"[{run_index}/{total}] fitting {condition['label']} seed={seed}",
                flush=True,
            )
            record, top_labels, leaf_labels = _fit_run(
                embeddings,
                metadata,
                condition,
                seed,
                output_dir,
            )
            completed[run_id] = record
            partitions[run_id] = (top_labels, leaf_labels)
            _write_progress(progress_path, configuration, completed)
            print(
                f"  m={record['selected_m']} k={record['root_k']} "
                f"leaf_ari={record['leaf_ari']:.4f} runtime={record['runtime_sec']:.2f}s",
                flush=True,
            )

    # Include only the selected set in the final report.  A partial invocation
    # can share a directory with a full invocation, so stale records are kept
    # in progress.json but do not leak into this invocation's report.
    records = [
        completed[_run_id(condition["name"], seed)]
        for condition in selected_conditions
        for seed in seeds
        if _run_id(condition["name"], seed) in completed
    ]
    records.sort(key=lambda row: (str(row["condition"]), int(row["seed"])))
    for record in records:
        run_id = str(record["run_id"])
        if run_id not in partitions:
            assignment_file = output_dir / str(record["assignment_file"])
            if assignment_file.exists():
                partitions[run_id] = _load_compact_assignments(assignment_file)

    seed_aris = _pairwise_seed_aris(records, partitions)
    cross_aris = _cross_condition_aris(records, partitions)
    summaries = _summary_by_condition(records)
    analysis = _analysis(summaries, cross_aris)
    for record in records:
        condition_seed = seed_aris.get(str(record["condition"]), {})
        record["seed_partition_top_ari_mean"] = condition_seed.get("mean_top_ari")
        record["seed_partition_leaf_ari_mean"] = condition_seed.get("mean_leaf_ari")
        if record["condition"] == "consensus_auto_m":
            pairs = {
                int(pair["seed"]): pair
                for pair in cross_aris.get("pairs", [])
            }
            pair = pairs.get(int(record["seed"]))
            record["baseline_vs_fast_top_ari"] = None if pair is None else pair["top_ari"]
            record["baseline_vs_fast_leaf_ari"] = None if pair is None else pair["leaf_ari"]
        else:
            record["baseline_vs_fast_top_ari"] = None
            record["baseline_vs_fast_leaf_ari"] = None

    # A full invocation is expected for the production result.  Still write a
    # partial report when interrupted so users can inspect what completed.
    report = {
        "configuration": configuration,
        "runs": records,
        "seed_partition_ari": seed_aris,
        "basic_auto_vs_fast_ari": cross_aris,
        "summary_by_condition": summaries,
        "analysis": analysis,
        "completed_runs": len(records),
        "expected_runs": len(selected_conditions) * len(seeds),
        # ``benchmark_runtime_sec`` is the sum of fit wall times, which remains
        # meaningful when this invocation resumes an earlier checkpoint.  The
        # process wall time is retained separately for operational diagnostics.
        "benchmark_runtime_sec": float(
            sum(float(record["runtime_sec"]) for record in records)
        ),
        "report_generation_runtime_sec": float(
            time.perf_counter() - benchmark_started
        ),
    }
    _atomic_json_write(output_dir / "report.json", report)
    pd.DataFrame(records).to_csv(output_dir / "runs.csv", index=False)
    _write_progress(progress_path, configuration, completed)
    print(
        f"Saved {len(records)}/{len(selected_conditions) * len(seeds)} runs to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
