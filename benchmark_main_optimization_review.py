"""Audit main-branch optimizations against a pre-optimization checkout.

The Gemini dataset remains the primary benchmark. Deterministic synthetic
stress datasets supplement it to expose assumptions that may be hidden by one
embedding distribution: severe imbalance, nearly coincident centers, exact
duplicates and equal-distance bridges, rank deficiency, and tiny high-K
clusters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

from incremental_clustering import load_state


THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "2",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "PYTHONDONTWRITEBYTECODE": "1",
}


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    embeddings: np.ndarray
    truth_labels: np.ndarray
    truth_noise: np.ndarray
    max_clusters: int = 6
    min_child_size: int = 8
    description: str = ""


@dataclass(frozen=True)
class Version:
    name: str
    checkout: Path
    extra_args: tuple[str, ...] = ()
    input_cache: Path | None = None


@dataclass(frozen=True)
class RunSpec:
    name: str
    pair_key: str
    benchmark_group: str
    version: Version
    input_json: Path | None
    seed: int
    fast: bool
    sample_size: int | None = None
    sample_seed: int | None = None
    pca_components: int | None = None
    max_depth: int = 4
    min_node_size: int = 60
    min_child_size: int = 20
    max_clusters: int = 4
    repeat: int = 1
    synthetic_case: str | None = None


@dataclass
class CompletedRun:
    public: dict[str, Any]
    assignments: pd.DataFrame = field(repr=False)
    metadata: pd.DataFrame = field(repr=False)
    hierarchy_model: Any = field(repr=False)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _random_unit_centers(
    rng: np.random.Generator,
    count: int,
    dimensions: int,
) -> np.ndarray:
    return _normalize_rows(rng.normal(size=(count, dimensions)))


def _sample_clusters(
    rng: np.random.Generator,
    centers: np.ndarray,
    sizes: list[int],
    scales: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    batches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for cluster_id, (center, size, scale) in enumerate(
        zip(centers, sizes, scales)
    ):
        batch = center + rng.normal(scale=scale, size=(size, centers.shape[1]))
        batches.append(_normalize_rows(batch))
        labels.append(np.full(size, cluster_id, dtype=np.int32))
    return np.vstack(batches), np.concatenate(labels)


def generate_synthetic_cases(seed: int = 20260809) -> list[SyntheticCase]:
    """Build deterministic adversarial datasets for generalization checks."""

    rng = np.random.default_rng(seed)
    dimensions = 64
    cases: list[SyntheticCase] = []

    centers = _random_unit_centers(rng, 4, dimensions)
    centers[3] = _normalize_rows(
        (0.97 * centers[2] + 0.03 * centers[3])[None, :]
    )[0]
    embeddings, labels = _sample_clusters(
        rng,
        centers,
        [260, 100, 35, 5],
        [0.08, 0.10, 0.12, 0.05],
    )
    cases.append(
        SyntheticCase(
            name="imbalanced_overlap",
            embeddings=embeddings,
            truth_labels=labels,
            truth_noise=np.zeros(len(labels), dtype=bool),
            description=(
                "Four clusters with a 52:20:7:1 size ratio and two nearly "
                "coincident centers."
            ),
        )
    )

    centers = _random_unit_centers(rng, 3, dimensions)
    regular, regular_labels = _sample_clusters(
        rng,
        centers,
        [90, 90, 90],
        [0.04, 0.04, 0.04],
    )
    duplicates = np.repeat(centers, repeats=10, axis=0)
    duplicate_labels = np.repeat(np.arange(3, dtype=np.int32), repeats=10)
    bridge_pairs = ((0, 1), (1, 2), (0, 2))
    bridge_rows = []
    for first, second in bridge_pairs:
        midpoint = _normalize_rows(
            (centers[first] + centers[second])[None, :]
        )[0]
        bridge_rows.append(np.repeat(midpoint[None, :], repeats=20, axis=0))
    bridges = np.vstack(bridge_rows)
    embeddings = np.vstack([regular, duplicates, bridges])
    labels = np.concatenate(
        [regular_labels, duplicate_labels, np.full(len(bridges), -1)]
    )
    cases.append(
        SyntheticCase(
            name="duplicate_ties",
            embeddings=embeddings,
            truth_labels=labels,
            truth_noise=labels == -1,
            description=(
                "Exact duplicate centers and repeated equal-distance bridge "
                "points that force deterministic tie handling."
            ),
        )
    )

    common = _random_unit_centers(rng, 1, dimensions)[0]
    offsets = _random_unit_centers(rng, 3, dimensions)
    close_centers = _normalize_rows(common + 0.32 * offsets)
    regular, regular_labels = _sample_clusters(
        rng,
        close_centers,
        [100, 100, 100],
        [0.12, 0.12, 0.12],
    )
    bridge_rows = []
    for index in range(60):
        first = index % 3
        second = (first + 1) % 3
        alpha = 0.35 + 0.30 * ((index % 11) / 10.0)
        row = (
            alpha * close_centers[first]
            + (1.0 - alpha) * close_centers[second]
            + rng.normal(scale=0.04, size=dimensions)
        )
        bridge_rows.append(row)
    bridges = _normalize_rows(np.asarray(bridge_rows))
    uniform_noise = _normalize_rows(rng.normal(size=(40, dimensions)))
    embeddings = np.vstack([regular, bridges, uniform_noise])
    labels = np.concatenate(
        [regular_labels, np.full(100, -1, dtype=np.int32)]
    )
    cases.append(
        SyntheticCase(
            name="boundary_and_noise",
            embeddings=embeddings,
            truth_labels=labels,
            truth_noise=labels == -1,
            description=(
                "Three overlapping spherical clusters, interpolated boundary "
                "points, and uniform high-dimensional noise."
            ),
        )
    )

    intrinsic_dimensions = 5
    projection = rng.normal(size=(intrinsic_dimensions, 96))
    intrinsic_centers = _random_unit_centers(rng, 4, intrinsic_dimensions)
    intrinsic, labels = _sample_clusters(
        rng,
        intrinsic_centers,
        [90, 90, 90, 90],
        [0.10, 0.10, 0.10, 0.10],
    )
    rank_deficient = _normalize_rows(intrinsic @ projection)
    duplicate_indices = rng.choice(len(rank_deficient), size=72, replace=False)
    rank_deficient[duplicate_indices] = rank_deficient[
        duplicate_indices // 2
    ]
    cases.append(
        SyntheticCase(
            name="rank_deficient_duplicates",
            embeddings=rank_deficient,
            truth_labels=labels,
            truth_noise=np.zeros(len(labels), dtype=bool),
            description=(
                "Ninety-six observed dimensions with intrinsic rank five and "
                "twenty percent duplicated rows."
            ),
        )
    )

    centers = _random_unit_centers(rng, 8, dimensions)
    embeddings, labels = _sample_clusters(
        rng,
        centers,
        [100, 90, 70, 55, 40, 25, 15, 5],
        [0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.05],
    )
    cases.append(
        SyntheticCase(
            name="tiny_clusters_high_k",
            embeddings=embeddings,
            truth_labels=labels,
            truth_noise=np.zeros(len(labels), dtype=bool),
            max_clusters=8,
            min_child_size=5,
            description=(
                "Eight clusters with progressively smaller support down to "
                "five rows, stressing K selection and hierarchy stopping."
            ),
        )
    )
    return cases


def _write_synthetic_case(case: SyntheticCase, path: Path) -> None:
    records = []
    for index, (embedding, label, truth_noise) in enumerate(
        zip(case.embeddings, case.truth_labels, case.truth_noise)
    ):
        records.append(
            {
                "id": f"{case.name}-{index}",
                "embedding": embedding.tolist(),
                "tag": f"truth_{int(label)}",
                "class": f"truth_{int(label)}",
                "truth_label": int(label),
                "truth_noise": bool(truth_noise),
            }
        )
    path.write_text(
        json.dumps({"records": records}, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_peak_rss_kb(process_id: int) -> int:
    try:
        status = Path(f"/proc/{process_id}/status").read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    return 0 if match is None else int(match.group(1))


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        text=True,
    ).strip()


def _quality_metrics(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    assignment_by_id = assignments.set_index("id")
    metadata_by_id = metadata.set_index("id")
    shared_ids = metadata_by_id.index.intersection(assignment_by_id.index)
    predicted = (
        assignment_by_id.loc[shared_ids, "cluster_path"]
        .fillna("noise")
        .astype(str)
        .to_numpy()
    )
    metrics: dict[str, Any] = {}
    for column in ("truth_label", "tag", "class"):
        if column not in metadata_by_id:
            continue
        truth = metadata_by_id.loc[shared_ids, column].astype(str).to_numpy()
        metrics[f"{column}_ari"] = float(adjusted_rand_score(truth, predicted))
        metrics[f"{column}_nmi"] = float(
            normalized_mutual_info_score(truth, predicted)
        )
    if "truth_noise" in metadata_by_id:
        truth_noise = (
            metadata_by_id.loc[shared_ids, "truth_noise"].astype(bool).to_numpy()
        )
        predicted_noise = (
            assignment_by_id.loc[shared_ids, "is_noise"].astype(bool).to_numpy()
        )
        precision, recall, f1, _support = precision_recall_fscore_support(
            truth_noise,
            predicted_noise,
            average="binary",
            zero_division=0,
        )
        metrics.update(
            {
                "truth_noise_precision": float(precision),
                "truth_noise_recall": float(recall),
                "truth_noise_f1": float(f1),
            }
        )
    return metrics


def _selected_pca_dimension(state: Any) -> int | None:
    selected = state.config.get("pca_components_selected")
    if selected is not None:
        return int(selected)
    pca = state.hierarchy_model.pca
    dimension = getattr(pca, "dimension", None)
    if dimension is not None:
        return int(dimension)
    components = getattr(pca, "n_components_", None)
    return None if components is None else int(components)


def _run_fit(spec: RunSpec, python_bin: Path, directory: Path) -> CompletedRun:
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec.name)
    state_path = directory / f"{run_name}.state.pkl"
    assignments_path = directory / f"{run_name}.assignments.csv"
    coordinates_path = directory / f"{run_name}.coordinates.csv"
    tree_path = directory / f"{run_name}.tree.json"
    log_path = directory / f"{run_name}.log"
    command = [
        str(python_bin),
        str(spec.version.checkout / "incremental_clustering.py"),
        "fit",
    ]
    if spec.version.input_cache is not None:
        command.extend(["--input-cache", str(spec.version.input_cache)])
    elif spec.input_json is not None:
        command.extend(["--input-json", str(spec.input_json)])
    else:
        raise ValueError("A JSON input or cache is required")
    command.extend(
        [
            "--state-output",
            str(state_path),
            "--assignments-output",
            str(assignments_path),
            "--coordinates-output",
            str(coordinates_path),
            "--tree-output",
            str(tree_path),
            "--skip-visualization",
            "--seed",
            str(spec.seed),
            "--max-depth",
            str(spec.max_depth),
            "--min-node-size",
            str(spec.min_node_size),
            "--min-child-size",
            str(spec.min_child_size),
            "--max-clusters",
            str(spec.max_clusters),
        ]
    )
    if spec.fast:
        command.append("--fast")
    if spec.sample_size is not None:
        command.extend(["--dataset-sample-size", str(spec.sample_size)])
        command.extend(
            [
                "--dataset-sample-seed",
                str(spec.sample_seed if spec.sample_seed is not None else spec.seed),
            ]
        )
    if spec.pca_components is not None:
        command.extend(["--pca-components", str(spec.pca_components)])
    command.extend(spec.version.extra_args)

    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    peak_rss_kb = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=spec.version.checkout,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            peak_rss_kb = max(peak_rss_kb, _read_peak_rss_kb(process.pid))
            time.sleep(0.05)
        return_code = process.wait()
    elapsed_sec = time.perf_counter() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if return_code:
        raise RuntimeError(
            f"Benchmark failed ({return_code}): {' '.join(command)}\n"
            f"{log_path.read_text(encoding='utf-8')[-8000:]}"
        )

    state = load_state(state_path)
    assignments = state.assignments.copy()
    metadata = state.metadata.copy()
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    root = tree.get("root", {})
    summary = tree.get("summary", {})
    public = {
        "name": spec.name,
        "pair_key": spec.pair_key,
        "benchmark_group": spec.benchmark_group,
        "version": spec.version.name,
        "seed": spec.seed,
        "repeat": spec.repeat,
        "fast": spec.fast,
        "sample_size": spec.sample_size,
        "synthetic_case": spec.synthetic_case,
        "elapsed_sec": elapsed_sec,
        "user_sec": usage_after.ru_utime - usage_before.ru_utime,
        "system_sec": usage_after.ru_stime - usage_before.ru_stime,
        "peak_rss_kb": peak_rss_kb,
        "state_bytes": state_path.stat().st_size,
        "samples": int(len(assignments)),
        "pca_dimension": _selected_pca_dimension(state),
        "root_k": root.get("selected_k"),
        "leaf_clusters": summary.get("leaf_clusters"),
        "noise_count": int(assignments["is_noise"].sum()),
        "quality": _quality_metrics(assignments, metadata),
    }
    return CompletedRun(
        public=public,
        assignments=assignments,
        metadata=metadata,
        hierarchy_model=state.hierarchy_model,
    )


def _compare_runs(
    baseline: CompletedRun,
    candidate: CompletedRun,
) -> dict[str, Any]:
    baseline_frame = baseline.assignments.set_index("id")
    candidate_frame = candidate.assignments.set_index("id")
    shared_ids = baseline_frame.index.intersection(candidate_frame.index)
    baseline_paths = (
        baseline_frame.loc[shared_ids, "cluster_path"]
        .fillna("noise")
        .astype(str)
        .to_numpy()
    )
    candidate_paths = (
        candidate_frame.loc[shared_ids, "cluster_path"]
        .fillna("noise")
        .astype(str)
        .to_numpy()
    )
    baseline_noise = (
        baseline_frame.loc[shared_ids, "is_noise"].astype(bool).to_numpy()
    )
    candidate_noise = (
        candidate_frame.loc[shared_ids, "is_noise"].astype(bool).to_numpy()
    )

    baseline_nodes = baseline.hierarchy_model.nodes
    candidate_nodes = candidate.hierarchy_model.nodes
    node_paths_equal = set(baseline_nodes) == set(candidate_nodes)
    center_shapes_equal = node_paths_equal and all(
        baseline_nodes[path].centers.shape == candidate_nodes[path].centers.shape
        for path in baseline_nodes
    )
    center_max_abs = None
    if center_shapes_equal:
        center_max_abs = max(
            (
                float(
                    np.max(
                        np.abs(
                            baseline_nodes[path].centers
                            - candidate_nodes[path].centers
                        )
                    )
                )
                for path in baseline_nodes
                if baseline_nodes[path].centers.size
            ),
            default=0.0,
        )
    return {
        "pair_key": baseline.public["pair_key"],
        "baseline_version": baseline.public["version"],
        "candidate_version": candidate.public["version"],
        "samples": int(len(shared_ids)),
        "path_exact_fraction": float(np.mean(baseline_paths == candidate_paths)),
        "path_ari": float(adjusted_rand_score(baseline_paths, candidate_paths)),
        "path_nmi": float(
            normalized_mutual_info_score(baseline_paths, candidate_paths)
        ),
        "noise_disagreement_fraction": float(
            np.mean(baseline_noise != candidate_noise)
        ),
        "pca_dimension_equal": (
            baseline.public["pca_dimension"] == candidate.public["pca_dimension"]
        ),
        "root_k_equal": baseline.public["root_k"] == candidate.public["root_k"],
        "node_paths_equal": node_paths_equal,
        "center_shapes_equal": center_shapes_equal,
        "center_max_abs": center_max_abs,
    }


def _version_specs(args: argparse.Namespace) -> dict[str, Version]:
    current = Version("current", args.current_checkout)
    versions = {
        "baseline": Version("baseline", args.baseline_checkout),
        "current": current,
        "current_reference": Version(
            "current_reference",
            args.current_checkout,
            extra_args=(
                "--embedding-storage-dtype",
                "float64",
                "--no-fast-m-reuse",
                "--include-conditional-memberships",
            ),
        ),
    }
    if args.current_cache is not None:
        versions["current_cache"] = Version(
            "current_cache",
            args.current_checkout,
            input_cache=args.current_cache,
        )
    return versions


def _build_specs(
    args: argparse.Namespace,
    synthetic_paths: dict[str, Path],
    cases: list[SyntheticCase],
) -> list[RunSpec]:
    versions = _version_specs(args)
    specs: list[RunSpec] = []
    for case in cases:
        for version_name in ("baseline", "current", "current_reference"):
            specs.append(
                RunSpec(
                    name=f"synthetic-{case.name}-{version_name}",
                    pair_key=f"synthetic-{case.name}",
                    benchmark_group=f"synthetic-{case.name}",
                    version=versions[version_name],
                    input_json=synthetic_paths[case.name],
                    seed=17,
                    fast=True,
                    pca_components=min(32, case.embeddings.shape[1]),
                    max_depth=3,
                    min_node_size=24,
                    min_child_size=case.min_child_size,
                    max_clusters=case.max_clusters,
                    synthetic_case=case.name,
                )
            )

    for sample_size in (300, 1000):
        for seed in (42, 43, 44):
            for version_name in ("baseline", "current"):
                specs.append(
                    RunSpec(
                        name=(
                            f"gemini-fast-{sample_size}-seed-{seed}-{version_name}"
                        ),
                        pair_key=f"gemini-fast-{sample_size}-seed-{seed}",
                        benchmark_group=f"gemini-fast-{sample_size}-quality",
                        version=versions[version_name],
                        input_json=args.gemini_input,
                        seed=seed,
                        fast=True,
                        sample_size=sample_size,
                        sample_seed=seed,
                    )
                )
    for seed in (42, 43):
        for version_name in ("baseline", "current"):
            specs.append(
                RunSpec(
                    name=f"gemini-exact-300-seed-{seed}-{version_name}",
                    pair_key=f"gemini-exact-300-seed-{seed}",
                    benchmark_group="gemini-exact-300-quality",
                    version=versions[version_name],
                    input_json=args.gemini_input,
                    seed=seed,
                    fast=False,
                    sample_size=300,
                    sample_seed=seed,
                    pca_components=64,
                    max_depth=3,
                    min_node_size=30,
                    min_child_size=10,
                )
            )

    for sample_size in (1000, 3000):
        performance_versions = ["baseline", "current"]
        if "current_cache" in versions:
            performance_versions.append("current_cache")
        for repeat in range(1, args.performance_repeats + 1):
            for version_name in performance_versions:
                specs.append(
                    RunSpec(
                        name=(
                            f"gemini-perf-{sample_size}-repeat-{repeat}-"
                            f"{version_name}"
                        ),
                        pair_key=f"gemini-perf-{sample_size}-repeat-{repeat}",
                        benchmark_group=f"gemini-perf-{sample_size}",
                        version=versions[version_name],
                        input_json=args.gemini_input,
                        seed=42,
                        fast=True,
                        sample_size=sample_size,
                        sample_seed=42,
                        repeat=repeat,
                    )
                )

    for seed in (42, 43, 44):
        specs.append(
            RunSpec(
                name=f"gemini-ablation-1000-seed-{seed}-current-reference",
                pair_key=f"gemini-fast-1000-seed-{seed}",
                benchmark_group="gemini-fast-1000-ablation",
                version=versions["current_reference"],
                input_json=args.gemini_input,
                seed=seed,
                fast=True,
                sample_size=1000,
                sample_seed=seed,
            )
        )
    return specs


def _performance_summary(runs: list[CompletedRun]) -> list[dict[str, Any]]:
    frame = pd.DataFrame([run.public for run in runs])
    rows: list[dict[str, Any]] = []
    for (group, version), subset in frame.groupby(
        ["benchmark_group", "version"], sort=True
    ):
        if not str(group).startswith("gemini-perf-"):
            continue
        elapsed = subset["elapsed_sec"].astype(float).tolist()
        rss = subset["peak_rss_kb"].astype(float).tolist()
        state_bytes = subset["state_bytes"].astype(float).tolist()
        rows.append(
            {
                "benchmark_group": group,
                "version": version,
                "runs": len(subset),
                "wall_p50_sec": median(elapsed),
                "wall_p90_sec": _percentile(elapsed, 90),
                "wall_min_sec": min(elapsed),
                "wall_max_sec": max(elapsed),
                "peak_rss_p50_kb": median(rss),
                "state_p50_bytes": median(state_bytes),
            }
        )
    return rows


def _render_runs_csv(runs: list[CompletedRun], path: Path) -> None:
    rows = []
    for completed in runs:
        row = dict(completed.public)
        quality = row.pop("quality")
        row.update(quality)
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkout", type=Path, required=True)
    parser.add_argument("--current-checkout", type=Path, required=True)
    parser.add_argument("--gemini-input", type=Path, required=True)
    parser.add_argument("--current-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--performance-repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.performance_repeats < 3:
        raise ValueError("--performance-repeats must be at least 3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = generate_synthetic_cases()
    with tempfile.TemporaryDirectory(prefix="main-optimization-review-") as temp:
        temporary_directory = Path(temp)
        synthetic_paths: dict[str, Path] = {}
        for case in cases:
            path = temporary_directory / f"{case.name}.json"
            _write_synthetic_case(case, path)
            synthetic_paths[case.name] = path
        specs = _build_specs(args, synthetic_paths, cases)
        runs: list[CompletedRun] = []
        for index, spec in enumerate(specs, start=1):
            print(
                f"[{index}/{len(specs)}] {spec.name}",
                flush=True,
            )
            completed = _run_fit(spec, args.python_bin, temporary_directory)
            runs.append(completed)
            print(
                f"  {completed.public['elapsed_sec']:.3f}s "
                f"rss={completed.public['peak_rss_kb']}KiB "
                f"state={completed.public['state_bytes']}",
                flush=True,
            )

        by_pair: dict[str, dict[str, CompletedRun]] = {}
        for run in runs:
            by_pair.setdefault(run.public["pair_key"], {})[
                run.public["version"]
            ] = run
        comparisons: list[dict[str, Any]] = []
        for pair_key, versions in sorted(by_pair.items()):
            baseline = versions.get("baseline")
            current = versions.get("current")
            if baseline is not None and current is not None:
                comparisons.append(_compare_runs(baseline, current))
            reference = versions.get("current_reference")
            if current is not None and reference is not None:
                comparisons.append(_compare_runs(reference, current))
            cached = versions.get("current_cache")
            if current is not None and cached is not None:
                comparisons.append(_compare_runs(current, cached))

    report = {
        "configuration": {
            "baseline_checkout": str(args.baseline_checkout),
            "baseline_revision": _git_revision(args.baseline_checkout),
            "current_checkout": str(args.current_checkout),
            "current_revision": _git_revision(args.current_checkout),
            "gemini_input": str(args.gemini_input),
            "gemini_sha256": _sha256(args.gemini_input),
            "current_cache": (
                None if args.current_cache is None else str(args.current_cache)
            ),
            "performance_repeats": args.performance_repeats,
            "thread_environment": THREAD_ENVIRONMENT,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        "synthetic_cases": [
            {
                "name": case.name,
                "samples": len(case.embeddings),
                "dimensions": case.embeddings.shape[1],
                "truth_clusters": int(len(set(case.truth_labels) - {-1})),
                "truth_noise": int(case.truth_noise.sum()),
                "description": case.description,
            }
            for case in cases
        ],
        "performance_summary": _performance_summary(runs),
        "comparisons": comparisons,
        "runs": [run.public for run in runs],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _render_runs_csv(runs, args.output_dir / "runs.csv")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
