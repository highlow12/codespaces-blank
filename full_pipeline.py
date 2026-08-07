"""Run the project's default Auto-PCA SFCM workflow end to end.

The default workflow is intentionally narrow and reproducible:

1. fit Auto-PCA SFCM;
2. choose the visualization PCA dimension automatically and render UMAP; and
3. when requested, run a mixed new-document/modified-document incremental
   update and render the updated visualization again.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cluster_visualization import (
    DEFAULT_CLUSTER_TARGET_WEIGHT,
    build_cluster_supervision,
    make_selected_coordinate_plot,
)
from clustering_pipelines import build_soft_assignments
from embedding_data import load_embeddings_from_json
from fast_fcm import FastFcmConfig, select_fast_fcm_cluster_count
from fcm_core import (
    fit_clustering_pca,
    sfcm_memberships_from_centers,
    spherical_fcm,
    transform_pca_normalized_features,
)
from fcm_validity import select_fcm_cluster_count
from fuzzy_cmeans import SphericalGeometry
from incremental_core import (
    batch_fingerprint,
    find_processed_batch,
    merge_rows_by_id,
    remember_processed_batch,
    replay_summary,
    resolve_batch_id,
)
from visualization_pca_dimension_selection import (
    VisualizationPcaDimensionSelection,
    select_visualization_pca_dimension_for_data,
)


DEFAULT_INCREMENTAL_RATIO = 0.20
DEFAULT_MODIFICATION_NOISE = 0.05


@dataclass
class AutoPcaSfcmState:
    """Reusable flat Auto-PCA SFCM state for an incremental test."""

    embeddings: np.ndarray
    metadata: pd.DataFrame
    pca: Any
    centers: np.ndarray
    assignments: pd.DataFrame
    selected_clusters: int
    cluster_selection_reason: str
    cluster_selection_metrics: list[dict[str, Any]]
    m: float = 2.0
    fast_mode: bool = False
    generation: int = 0
    processed_batches: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class IncrementalTestSplit:
    """A deterministic split with equal new and modified update rows."""

    initial_embeddings: np.ndarray
    initial_metadata: pd.DataFrame
    update_embeddings: np.ndarray
    update_metadata: pd.DataFrame
    new_count: int
    modified_count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "initial_samples": int(len(self.initial_embeddings)),
            "update_samples": int(len(self.update_embeddings)),
            "new_samples": int(self.new_count),
            "modified_samples": int(self.modified_count),
            "update_ratio_of_input": float(
                len(self.update_embeddings)
                / (len(self.initial_embeddings) + self.new_count)
            ),
        }


def _validate_data(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings must contain only finite values")
    if len(metadata) != len(values) or "id" not in metadata:
        raise ValueError("metadata must align with embeddings and contain an id column")
    frame = metadata.copy().reset_index(drop=True)
    identifiers = frame["id"].tolist()
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("metadata ids must be unique")
    return values, frame


def fit_auto_pca_sfcm(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 2,
    seed: int = 42,
    m: float = 2.0,
    fast_mode: bool = False,
    fast_config: FastFcmConfig | None = None,
) -> AutoPcaSfcmState:
    """Fit Auto-PCA SFCM and select its cluster count automatically."""

    values, frame = _validate_data(embeddings, metadata)
    projected, pca, _selection = fit_clustering_pca(values, seed=seed)
    if fast_mode:
        best, selection_metrics, selection_reason = select_fast_fcm_cluster_count(
            projected,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_child_size=min_child_size,
            selection_method="multi_metric",
            seed=seed,
            config=fast_config,
        )
    else:
        best, selection_metrics, selection_reason = select_fcm_cluster_count(
            projected,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_child_size=min_child_size,
            selection_method="multi_metric",
            seed=seed,
            m=m,
        )
    if best is None:
        # A one-cluster result is the only valid automatic answer when the
        # data cannot support two children under the requested constraints.
        result = spherical_fcm(projected, n_clusters=1, m=m, seed=seed)
        selected_clusters = 1
        selection_reason = f"single_cluster_fallback:{selection_reason}"
    else:
        result = best.result
        selected_clusters = int(best.n_clusters)
    assignments = build_soft_assignments(frame, result.labels, result.memberships)
    return AutoPcaSfcmState(
        embeddings=values,
        metadata=frame,
        pca=pca,
        centers=result.centers,
        assignments=assignments,
        selected_clusters=selected_clusters,
        cluster_selection_reason=selection_reason,
        cluster_selection_metrics=selection_metrics,
        m=float(result.m),
        fast_mode=fast_mode,
        generation=0,
        processed_batches={},
    )


def _merge_updated_rows(
    state: AutoPcaSfcmState,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, int, int]:
    merged = merge_rows_by_id(
        state.embeddings,
        state.metadata,
        embeddings,
        metadata,
    )
    return (
        merged.embeddings,
        merged.metadata,
        len(merged.replaced_ids),
        len(merged.appended_ids),
    )


def update_auto_pca_sfcm(
    state: AutoPcaSfcmState,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    batch_id: str | None = None,
) -> tuple[AutoPcaSfcmState, dict[str, Any]]:
    """Update SFCM centers without refitting the selected PCA projection."""

    incoming_embeddings, incoming_metadata = _validate_data(embeddings, metadata)
    if incoming_embeddings.shape[1] != state.embeddings.shape[1]:
        raise ValueError("incremental embeddings have a different dimensionality")
    fingerprint = batch_fingerprint(incoming_embeddings, incoming_metadata)
    resolved_batch_id = resolve_batch_id(batch_id, fingerprint)
    replay = find_processed_batch(
        getattr(state, "processed_batches", {}),
        resolved_batch_id,
        fingerprint,
    )
    if replay is not None:
        return state, replay_summary(replay, batch_id=resolved_batch_id)
    (
        combined_embeddings,
        combined_metadata,
        replaced_count,
        appended_count,
    ) = _merge_updated_rows(state, incoming_embeddings, incoming_metadata)

    projected = transform_pca_normalized_features(combined_embeddings, state.pca)
    memberships, _ = sfcm_memberships_from_centers(
        projected,
        state.centers,
        m=state.m,
    )
    geometry = SphericalGeometry()
    normalized_projected = geometry.prepare_samples(projected)
    updated_centers = geometry.update_centers(
        normalized_projected,
        memberships,
        m=state.m,
        rng=np.random.default_rng(0),
    )
    updated_memberships, _ = sfcm_memberships_from_centers(
        projected,
        updated_centers,
        m=state.m,
    )
    labels = updated_memberships.argmax(axis=1)
    assignments = build_soft_assignments(
        combined_metadata,
        labels,
        updated_memberships,
    )
    generation = int(getattr(state, "generation", 0)) + 1
    summary: dict[str, Any] = {
        "replaced_samples": replaced_count,
        "appended_samples": appended_count,
        "total_samples": len(combined_embeddings),
        "batch_id": resolved_batch_id,
        "idempotent_replay": False,
        "generation": generation,
    }
    processed_batches = remember_processed_batch(
        getattr(state, "processed_batches", {}),
        batch_id=resolved_batch_id,
        fingerprint=fingerprint,
        summary=summary,
        generation=generation,
    )
    return (
        AutoPcaSfcmState(
            embeddings=combined_embeddings,
            metadata=combined_metadata,
            pca=state.pca,
            centers=updated_centers,
            assignments=assignments,
            selected_clusters=state.selected_clusters,
            cluster_selection_reason=state.cluster_selection_reason,
            cluster_selection_metrics=state.cluster_selection_metrics,
            m=state.m,
            fast_mode=state.fast_mode,
            generation=generation,
            processed_batches=processed_batches,
        ),
        summary,
    )


def make_incremental_test_split(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    ratio: float = DEFAULT_INCREMENTAL_RATIO,
    modification_noise: float = DEFAULT_MODIFICATION_NOISE,
    seed: int = 42,
) -> IncrementalTestSplit:
    """Make a 20% mixed update: unseen rows plus changed existing rows.

    Half of the update rows are withheld from the initial fit, therefore they
    are genuinely new IDs at update time.  The other half remains in the
    initial fit and is replaced with a deterministic perturbation of its
    embedding, modeling a changed document.
    """

    values, frame = _validate_data(embeddings, metadata)
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between 0 and 1")
    if modification_noise < 0.0:
        raise ValueError("modification_noise must be non-negative")
    update_count = 2 * max(1, int(round(len(values) * ratio / 2.0)))
    if update_count >= len(values):
        raise ValueError("input must leave samples for the initial fit")

    rng = np.random.default_rng(seed)
    sampled = rng.permutation(len(values))[:update_count]
    half = update_count // 2
    new_indices = sampled[:half]
    modified_indices = sampled[half:]
    initial_mask = np.ones(len(values), dtype=bool)
    initial_mask[new_indices] = False

    initial_embeddings = values[initial_mask]
    initial_metadata = frame.iloc[np.flatnonzero(initial_mask)].copy()
    initial_metadata["incremental_operation"] = "initial"

    new_embeddings = values[new_indices].copy()
    new_metadata = frame.iloc[new_indices].copy()
    new_metadata["incremental_operation"] = "new"

    modified_embeddings = values[modified_indices].copy()
    if modification_noise > 0.0:
        perturbation = rng.normal(size=modified_embeddings.shape)
        perturbation /= np.maximum(
            np.linalg.norm(perturbation, axis=1, keepdims=True),
            1e-12,
        )
        scale = np.linalg.norm(modified_embeddings, axis=1, keepdims=True)
        modified_embeddings += modification_noise * scale * perturbation
    modified_metadata = frame.iloc[modified_indices].copy()
    modified_metadata["incremental_operation"] = "modified"

    update_embeddings = np.vstack([new_embeddings, modified_embeddings])
    update_metadata = pd.concat([new_metadata, modified_metadata], ignore_index=True)
    return IncrementalTestSplit(
        initial_embeddings=initial_embeddings,
        initial_metadata=initial_metadata,
        update_embeddings=update_embeddings,
        update_metadata=update_metadata,
        new_count=half,
        modified_count=half,
    )


def save_auto_pca_visualization(
    state: AutoPcaSfcmState,
    *,
    output_path: Path,
    report_path: Path,
    title: str,
    seed: int,
    n_neighbors: int = 15,
    min_dist: float = 0.02,
    metric: str = "cosine",
    spread: float = 0.85,
    cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> VisualizationPcaDimensionSelection:
    """Select visualization PCA automatically, then save one UMAP plot/report."""

    target, target_metric, target_description = build_cluster_supervision(
        state.assignments
    )
    selection = select_visualization_pca_dimension_for_data(
        state.embeddings,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=False,
        cluster_target=target,
        cluster_target_metric=target_metric,
        cluster_target_weight=cluster_target_weight,
        seed=seed,
    )
    make_selected_coordinate_plot(
        selection.selected_coordinates,
        state.assignments,
        output_path,
        title=title,
        color_by="cluster",
        pca_components=selection.selected_dimension,
        cluster_target_weight=cluster_target_weight,
    )
    report = selection.to_dict()
    report["configuration"]["cluster_target"] = target_description
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selection


def run_full_pipeline(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    output_dir: Path,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 2,
    seed: int = 42,
    incremental_test: bool = False,
    incremental_ratio: float = DEFAULT_INCREMENTAL_RATIO,
    modification_noise: float = DEFAULT_MODIFICATION_NOISE,
    incremental_batch_id: str | None = None,
    fast_mode: bool = False,
    fast_config: FastFcmConfig | None = None,
) -> dict[str, Any]:
    """Run the configured full workflow and return its saved artifact summary."""

    values, frame = _validate_data(embeddings, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = (
        make_incremental_test_split(
            values,
            frame,
            ratio=incremental_ratio,
            modification_noise=modification_noise,
            seed=seed,
        )
        if incremental_test
        else None
    )
    initial_embeddings = values if split is None else split.initial_embeddings
    initial_metadata = frame if split is None else split.initial_metadata
    initial_state = fit_auto_pca_sfcm(
        initial_embeddings,
        initial_metadata,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_child_size=min_child_size,
        seed=seed,
        fast_mode=fast_mode,
        fast_config=fast_config,
    )
    initial_assignments = output_dir / "auto_pca_sfcm_assignments.csv"
    initial_plot = output_dir / "auto_pca_sfcm_visualization.png"
    initial_report = output_dir / "auto_pca_sfcm_visualization_pca.json"
    initial_state.assignments.to_csv(initial_assignments, index=False)
    initial_selection = save_auto_pca_visualization(
        initial_state,
        output_path=initial_plot,
        report_path=initial_report,
        title="Auto-PCA SFCM clustering",
        seed=seed,
    )
    summary: dict[str, Any] = {
        "pipeline": "auto_pca_sfcm",
        "initial_samples": len(initial_state.embeddings),
        "initial_selected_clusters": initial_state.selected_clusters,
        "selected_fuzzifier": initial_state.m,
        "fast_mode": initial_state.fast_mode,
        "cluster_selection_reason": initial_state.cluster_selection_reason,
        "cluster_selection_metrics": initial_state.cluster_selection_metrics,
        "initial_visualization_pca_components": initial_selection.selected_dimension,
        "artifacts": {
            "initial_assignments": str(initial_assignments),
            "initial_visualization": str(initial_plot),
            "initial_visualization_pca": str(initial_report),
        },
    }
    if split is None:
        return summary

    update_batch_path = output_dir / "incremental_test_batch.csv"
    updated_assignments = output_dir / "incremental_test_assignments.csv"
    updated_plot = output_dir / "incremental_test_visualization.png"
    updated_report = output_dir / "incremental_test_visualization_pca.json"
    split.update_metadata.to_csv(update_batch_path, index=False)
    updated_state, update_summary = update_auto_pca_sfcm(
        initial_state,
        split.update_embeddings,
        split.update_metadata,
        batch_id=incremental_batch_id,
    )
    updated_state.assignments.to_csv(updated_assignments, index=False)
    updated_selection = save_auto_pca_visualization(
        updated_state,
        output_path=updated_plot,
        report_path=updated_report,
        title="Auto-PCA SFCM after incremental test",
        seed=seed,
    )
    summary["incremental_test"] = {
        **split.to_dict(),
        **update_summary,
        "updated_visualization_pca_components": updated_selection.selected_dimension,
        "artifacts": {
            "update_batch": str(update_batch_path),
            "updated_assignments": str(updated_assignments),
            "updated_visualization": str(updated_plot),
            "updated_visualization_pca": str(updated_report),
        },
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Auto-PCA SFCM clustering and automatic-PCA visualization."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/full_pipeline"))
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--incremental-test", action="store_true")
    parser.add_argument("--incremental-ratio", type=float, default=DEFAULT_INCREMENTAL_RATIO)
    parser.add_argument("--modification-noise", type=float, default=DEFAULT_MODIFICATION_NOISE)
    parser.add_argument(
        "--incremental-batch-id",
        type=str,
        default=None,
        help="Stable ID for the optional incremental test batch.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use sample-based K scouting, automatic fuzzifier search, dynamic "
            "restarts, and bounded FCM iterations."
        ),
    )
    parser.add_argument("--fast-sample-size", type=int, default=1000)
    parser.add_argument("--fast-scout-n-init", type=int, default=2)
    parser.add_argument("--fast-refine-n-init", type=int, default=3)
    parser.add_argument("--fast-refine-top-k", type=int, default=2)
    parser.add_argument("--fast-stability-target", type=float, default=0.85)
    parser.add_argument(
        "--fast-m",
        type=float,
        nargs="+",
        default=[2.0, 1.8, 1.6, 1.4],
        help="Fuzzifier search schedule used by --fast.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    fast_config = FastFcmConfig(
        sample_size=args.fast_sample_size,
        scout_n_init=args.fast_scout_n_init,
        scout_max_attempts=max(args.fast_scout_n_init + 1, args.fast_scout_n_init * 2),
        refine_top_k=args.fast_refine_top_k,
        refine_n_init=args.fast_refine_n_init,
        refine_max_attempts=max(
            args.fast_refine_n_init + 2,
            args.fast_refine_n_init * 2,
        ),
        stability_target=args.fast_stability_target,
        m_values=tuple(args.fast_m),
    )
    summary = run_full_pipeline(
        embeddings,
        metadata,
        output_dir=args.output_dir,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_child_size=args.min_child_size,
        seed=args.seed,
        incremental_test=args.incremental_test,
        incremental_ratio=args.incremental_ratio,
        modification_noise=args.modification_noise,
        incremental_batch_id=args.incremental_batch_id,
        fast_mode=args.fast,
        fast_config=fast_config,
    )
    summary_path = args.output_dir / "full_pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
