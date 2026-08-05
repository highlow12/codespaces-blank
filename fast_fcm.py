"""Fast coarse-to-fine FCM cluster-count selection.

The regular selector is intentionally exhaustive and is useful for final
experiments. This module adds a bounded search for repeated development runs:
it probes the fuzzifier and K on a deterministic sample, then refines only the
best K values on the complete node. The returned candidate is still a regular
FCM candidate with full-data labels and centers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from clustering_types import FCMKCandidate
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION
from fcm_validity import (
    _candidate_to_record,
    _score_multi_metric_candidates,
    select_fcm_cluster_count,
)


@dataclass(frozen=True)
class FastFcmConfig:
    """Bounds for the fast scout/refine path."""

    sample_size: int = 1000
    scout_n_init: int = 2
    scout_max_attempts: int = 3
    scout_max_iter: int = 60
    scout_tol: float = 1e-4
    scout_max_clusters: int = 8
    refine_top_k: int = 2
    refine_n_init: int = 3
    refine_max_attempts: int = 5
    refine_max_iter: int = 100
    refine_tol: float = 1e-5
    max_refine_n_init: int = 10
    stability_target: float = 0.85
    m_values: tuple[float, ...] = (2.0, 1.8, 1.6, 1.4)
    minimum_probe_stability: float = 0.80
    min_center_separation: float = DEFAULT_FCM_MIN_CENTER_SEPARATION

    def validate(self) -> None:
        if self.sample_size < 2:
            raise ValueError("sample_size must be at least 2")
        if self.scout_n_init < 1 or self.refine_n_init < 1:
            raise ValueError("restart counts must be at least 1")
        if self.scout_max_attempts < self.scout_n_init:
            raise ValueError("scout_max_attempts must cover scout_n_init")
        if self.refine_max_attempts < self.refine_n_init:
            raise ValueError("refine_max_attempts must cover refine_n_init")
        if self.max_refine_n_init < self.refine_n_init:
            raise ValueError("max_refine_n_init must cover refine_n_init")
        if self.refine_top_k < 1:
            raise ValueError("refine_top_k must be at least 1")
        if self.scout_max_iter < 1 or self.refine_max_iter < 1:
            raise ValueError("iteration limits must be at least 1")
        if self.scout_max_clusters < 2:
            raise ValueError("scout_max_clusters must be at least 2")
        if self.scout_tol <= 0.0 or self.refine_tol <= 0.0:
            raise ValueError("tolerances must be positive")
        if not 0.0 <= self.stability_target <= 1.0:
            raise ValueError("stability_target must be between 0 and 1")
        if not 0.0 <= self.minimum_probe_stability <= 1.0:
            raise ValueError("minimum_probe_stability must be between 0 and 1")
        if not self.m_values or any(value <= 1.0 for value in self.m_values):
            raise ValueError("m_values must contain values greater than 1")


def _deterministic_sample(X: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if X.shape[0] <= sample_size:
        return X
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(X.shape[0], size=sample_size, replace=False))
    return X[indices]


def _scout_m(
    X: np.ndarray,
    *,
    min_child_size: int,
    min_clusters: int,
    max_membership_gap: float,
    distance_z: float,
    selection_method: str,
    seed: int,
    config: FastFcmConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the largest fuzzifier that produces a stable two-way probe."""

    records: list[dict[str, Any]] = []
    for m_index, m in enumerate(config.m_values):
        probe, probe_records, reason = select_fcm_cluster_count(
            X,
            min_clusters=2,
            max_clusters=2,
            min_child_size=min_child_size,
            min_membership=0.0,
            max_membership_gap=max_membership_gap,
            distance_z=distance_z,
            selection_method="multi_metric",
            seed=seed + m_index * 10_007,
            n_init=config.scout_n_init,
            max_attempts=config.scout_max_attempts,
            m=m,
            max_iter=config.scout_max_iter,
            tol=config.scout_tol,
            collapse_center_separation=config.min_center_separation,
        )
        for record in probe_records:
            records.append(
                {
                    **record,
                    "phase": "m_probe",
                    "m_probe": float(m),
                    "m_probe_reason": reason,
                }
            )
        if probe is not None and (
            probe.valid_restarts >= config.scout_n_init
            and probe.restart_stability >= config.minimum_probe_stability
        ):
            return float(m), records
    return float(config.m_values[-1]), records


def _scout_candidate_ks(
    records: list[dict[str, Any]],
    *,
    refine_top_k: int,
    min_clusters: int,
) -> list[int]:
    usable = [
        record
        for record in records
        if record.get("phase") != "m_probe"
        and record.get("valid_clusters", 0) >= 2
        and record.get("selection_score") is not None
    ]
    usable.sort(
        key=lambda record: (
            float(record["selection_score"]),
            float(record.get("restart_stability") or 0.0),
            -float(record.get("xie_beni") or np.inf),
        ),
        reverse=True,
    )
    ks: list[int] = []
    for record in usable:
        k = int(record["k"])
        if k >= min_clusters and k not in ks:
            ks.append(k)
        if len(ks) >= refine_top_k:
            break
    return ks


def _choose_full_candidate(
    candidates: list[FCMKCandidate],
    selection_method: str,
) -> FCMKCandidate:
    if selection_method == "multi_metric":
        _score_multi_metric_candidates(candidates)
        return max(
            candidates,
            key=lambda candidate: (
                candidate.selection_score,
                -candidate.xie_beni,
                candidate.restart_stability,
                candidate.silhouette,
                -candidate.n_clusters,
            ),
        )
    if selection_method == "silhouette":
        return max(candidates, key=lambda candidate: candidate.silhouette)
    return min(candidates, key=lambda candidate: candidate.xie_beni)


def select_fast_fcm_cluster_count(
    X: np.ndarray,
    *,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 20,
    min_membership: float = 0.40,
    max_membership_gap: float = 0.10,
    distance_z: float = 3.5,
    selection_method: str = "multi_metric",
    seed: int = 42,
    config: FastFcmConfig | None = None,
) -> tuple[FCMKCandidate | None, list[dict[str, Any]], str]:
    """Scout a node cheaply and refine only its best full-data K values."""

    fast = config or FastFcmConfig()
    fast.validate()
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")

    scout_X = _deterministic_sample(X, fast.sample_size, seed)
    selected_m, probe_records = _scout_m(
        scout_X,
        min_child_size=min_child_size,
        min_clusters=min_clusters,
        max_membership_gap=max_membership_gap,
        distance_z=distance_z,
        selection_method=selection_method,
        seed=seed,
        config=fast,
    )
    scout_max_clusters = min(max_clusters, fast.scout_max_clusters)
    scout_best, scout_records, scout_reason = select_fcm_cluster_count(
        scout_X,
        min_clusters=min_clusters,
        max_clusters=scout_max_clusters,
        min_child_size=min_child_size,
        min_membership=min_membership,
        max_membership_gap=max_membership_gap,
        distance_z=distance_z,
        selection_method=selection_method,
        seed=seed + 97,
        n_init=fast.scout_n_init,
        max_attempts=fast.scout_max_attempts,
        m=selected_m,
        max_iter=fast.scout_max_iter,
        tol=fast.scout_tol,
        collapse_center_separation=fast.min_center_separation,
        # The scout is already bounded by sample size, restarts, iterations,
        # and scout_max_clusters. Evaluate its complete K range so one noisy
        # XB worsening cannot hide a later, stronger split.
        xb_worsening_patience=scout_max_clusters,
    )
    if scout_best is None:
        return None, [*probe_records, *scout_records], f"fast_scout:{scout_reason}"

    candidate_ks = _scout_candidate_ks(
        scout_records,
        refine_top_k=fast.refine_top_k,
        min_clusters=min_clusters,
    )
    if not candidate_ks:
        candidate_ks = [int(scout_best.n_clusters)]

    refined: list[FCMKCandidate] = []
    refined_records: list[dict[str, Any]] = []
    selected_m_index = min(
        range(len(fast.m_values)),
        key=lambda index: abs(float(fast.m_values[index]) - selected_m),
    )
    m_schedule = [float(value) for value in fast.m_values[selected_m_index:]]
    used_m = selected_m
    for m_index, refine_m in enumerate(m_schedule):
        trial_refined: list[FCMKCandidate] = []
        trial_records: list[dict[str, Any]] = []
        for index, candidate_k in enumerate(candidate_ks):
            refine_n_init = fast.refine_n_init
            while True:
                full_best, full_records, _reason = select_fcm_cluster_count(
                    X,
                    min_clusters=candidate_k,
                    max_clusters=candidate_k,
                    min_child_size=min_child_size,
                    min_membership=min_membership,
                    max_membership_gap=max_membership_gap,
                    distance_z=distance_z,
                    selection_method=selection_method,
                    seed=seed + 1_003 + m_index * 101_003 + index * 11_009,
                    n_init=refine_n_init,
                    max_attempts=max(
                        fast.refine_max_attempts,
                        refine_n_init,
                    ),
                    m=refine_m,
                    max_iter=fast.refine_max_iter,
                    tol=fast.refine_tol,
                    collapse_center_separation=fast.min_center_separation,
                )
                if full_best is None:
                    break
                if (
                    full_best.restart_stability >= fast.stability_target
                    or refine_n_init >= fast.max_refine_n_init
                ):
                    trial_refined.append(full_best)
                    trial_records.extend(
                        {
                            **record,
                            "phase": "refine",
                            "selected_m": refine_m,
                        }
                        for record in full_records
                    )
                    break
                refine_n_init = min(fast.max_refine_n_init, refine_n_init * 2)
        if trial_refined:
            refined = trial_refined
            refined_records = trial_records
            used_m = refine_m
            break

    if not refined:
        return None, [*probe_records, *scout_records], "fast_refine:no_valid_split"

    best = _choose_full_candidate(refined, selection_method)
    return (
        best,
        [*probe_records, *scout_records, *refined_records],
        "selected_fast_scout_refine",
    )
