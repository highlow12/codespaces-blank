"""Consensus-based sampled FCM cluster-count selection.

The exhaustive selector fits every candidate K on all rows.  This module
instead asks several inexpensive, independent samples to vote on K.  Once a
strict majority agrees, only that K is fitted on the complete dataset.  If the
samples do not agree, the regular exhaustive selector is used as a safe
fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from clustering_types import FCMKCandidate
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fcm_validity import select_fcm_cluster_count


DEFAULT_CONSENSUS_MIN_ROWS = 500


@dataclass(frozen=True)
class ConsensusFcmConfig:
    """Compute bounds for sampled K voting and the final full-data fit."""

    sample_ratio: float = 0.20
    sample_size: int | None = None
    max_scouts: int = 5
    vote_threshold: int = 3
    sample_seed: int = 42
    sample_seed_stride: int = 1
    sample_min_child_floor: int = 2
    scout_n_init: int = 3
    scout_max_attempts: int = 5
    scout_max_iter: int = 100
    scout_tol: float = 1e-5
    full_n_init: int = DEFAULT_FCM_N_INIT
    full_max_attempts: int | None = None
    full_max_iter: int = 200
    full_tol: float = 1e-6
    min_center_separation: float = DEFAULT_FCM_MIN_CENTER_SEPARATION

    def validate(self) -> None:
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be within (0, 1]")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be positive when provided")
        if self.max_scouts < 1:
            raise ValueError("max_scouts must be positive")
        if not 1 <= self.vote_threshold <= self.max_scouts:
            raise ValueError("vote_threshold must be within max_scouts")
        if self.vote_threshold <= self.max_scouts // 2:
            raise ValueError("vote_threshold must be a strict majority")
        if self.sample_seed_stride < 1:
            raise ValueError("sample_seed_stride must be positive")
        if self.sample_min_child_floor < 2:
            raise ValueError("sample_min_child_floor must be at least 2")
        if self.scout_n_init < 1 or self.full_n_init < 1:
            raise ValueError("restart counts must be positive")
        if self.scout_max_attempts < self.scout_n_init:
            raise ValueError("scout_max_attempts must cover scout_n_init")
        if (
            self.full_max_attempts is not None
            and self.full_max_attempts < self.full_n_init
        ):
            raise ValueError("full_max_attempts must cover full_n_init")
        if self.scout_max_iter < 1 or self.full_max_iter < 1:
            raise ValueError("iteration limits must be positive")
        if self.scout_tol <= 0.0 or self.full_tol <= 0.0:
            raise ValueError("tolerances must be positive")
        if self.min_center_separation < 0.0:
            raise ValueError("min_center_separation must be non-negative")


def _consensus_sample_size(
    row_count: int,
    *,
    min_clusters: int,
    config: ConsensusFcmConfig,
) -> int:
    requested = (
        config.sample_size
        if config.sample_size is not None
        else int(np.ceil(row_count * config.sample_ratio))
    )
    minimum_feasible = min_clusters * config.sample_min_child_floor
    return min(row_count, max(minimum_feasible, requested))


def _scaled_sample_min_child_size(
    min_child_size: int,
    *,
    row_count: int,
    sample_size: int,
    floor: int,
) -> int:
    return max(
        floor,
        int(np.ceil(min_child_size * sample_size / row_count)),
    )


def _sample_rows(X: np.ndarray, sample_size: int, *, seed: int) -> np.ndarray:
    # Match the benchmark's reproducible nested-sample construction.  Taking
    # a permutation prefix also lets callers compare larger sample sizes while
    # retaining every row from the smaller sample for a given seed.
    indices = np.sort(
        np.random.default_rng(seed).permutation(X.shape[0])[:sample_size]
    )
    return X[indices]


def _annotated_records(
    records: list[dict[str, Any]],
    **annotations: Any,
) -> list[dict[str, Any]]:
    return [{**record, **annotations} for record in records]


def select_consensus_fcm_cluster_count(
    X: np.ndarray,
    *,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 20,
    min_membership: float = 0.40,
    max_membership_gap: float = 0.10,
    distance_z: float = 3.5,
    selection_method: str = "multi_metric",
    min_xb_relative_improvement: float = 0.05,
    xb_worsening_patience: int = 2,
    seed: int = 42,
    m: float = 2.0,
    collapse_center_separation: float | None = None,
    config: ConsensusFcmConfig | None = None,
) -> tuple[FCMKCandidate | None, list[dict[str, Any]], str]:
    """Select K by strict-majority sample voting, then fit that K on all rows.

    Each scout uses a different reproducible row sample but the same FCM seed,
    isolating sampling uncertainty from optimizer uncertainty.  A missing
    strict majority triggers the regular exhaustive full-data selection.
    """

    values = np.asarray(X)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")

    consensus = config or ConsensusFcmConfig()
    consensus.validate()
    sample_size = _consensus_sample_size(
        values.shape[0],
        min_clusters=min_clusters,
        config=consensus,
    )

    common = {
        "min_child_size": min_child_size,
        "min_membership": min_membership,
        "max_membership_gap": max_membership_gap,
        "distance_z": distance_z,
        "selection_method": selection_method,
        "min_xb_relative_improvement": min_xb_relative_improvement,
        "xb_worsening_patience": xb_worsening_patience,
        "seed": seed,
        "min_center_separation": consensus.min_center_separation,
        "m": m,
        "collapse_center_separation": collapse_center_separation,
    }

    if sample_size >= values.shape[0]:
        best, records, reason = select_fcm_cluster_count(
            values,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            n_init=consensus.full_n_init,
            max_attempts=consensus.full_max_attempts,
            max_iter=consensus.full_max_iter,
            tol=consensus.full_tol,
            **common,
        )
        return (
            best,
            _annotated_records(records, phase="consensus_full_data_direct"),
            f"consensus_full_data_direct:{reason}",
        )

    sample_min_child_size = _scaled_sample_min_child_size(
        min_child_size,
        row_count=values.shape[0],
        sample_size=sample_size,
        floor=consensus.sample_min_child_floor,
    )
    records: list[dict[str, Any]] = []
    votes: Counter[int] = Counter()
    winner: int | None = None
    for scout_index in range(consensus.max_scouts):
        sample_seed = (
            consensus.sample_seed
            + scout_index * consensus.sample_seed_stride
        )
        sample = _sample_rows(values, sample_size, seed=sample_seed)
        scout_common = {
            **common,
            "min_child_size": sample_min_child_size,
        }
        scout_best, scout_records, scout_reason = select_fcm_cluster_count(
            sample,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            n_init=consensus.scout_n_init,
            max_attempts=consensus.scout_max_attempts,
            max_iter=consensus.scout_max_iter,
            tol=consensus.scout_tol,
            **scout_common,
        )
        vote = None if scout_best is None else int(scout_best.n_clusters)
        records.extend(
            _annotated_records(
                scout_records,
                phase="consensus_scout",
                scout_index=scout_index,
                sample_seed=sample_seed,
                sample_size=sample_size,
                vote_k=vote,
                scout_reason=scout_reason,
            )
        )
        if vote is None:
            continue
        votes[vote] += 1
        if votes[vote] >= consensus.vote_threshold:
            winner = vote
            break
        remaining_scouts = consensus.max_scouts - scout_index - 1
        maximum_possible_votes = (
            max(votes.values(), default=0) + remaining_scouts
        )
        if maximum_possible_votes < consensus.vote_threshold:
            break

    vote_summary = {str(k): int(count) for k, count in sorted(votes.items())}
    if winner is not None:
        best, refine_records, reason = select_fcm_cluster_count(
            values,
            min_clusters=winner,
            max_clusters=winner,
            n_init=consensus.full_n_init,
            max_attempts=consensus.full_max_attempts,
            max_iter=consensus.full_max_iter,
            tol=consensus.full_tol,
            **common,
        )
        records.extend(
            _annotated_records(
                refine_records,
                phase="consensus_full_fit",
                consensus_k=winner,
                consensus_votes=vote_summary,
                consensus_reason=reason,
            )
        )
        if best is not None:
            return best, records, "selected_consensus_sample_vote"

    best, fallback_records, reason = select_fcm_cluster_count(
        values,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        n_init=consensus.full_n_init,
        max_attempts=consensus.full_max_attempts,
        max_iter=consensus.full_max_iter,
        tol=consensus.full_tol,
        **common,
    )
    records.extend(
        _annotated_records(
            fallback_records,
            phase="consensus_full_fallback",
            consensus_votes=vote_summary,
            consensus_reason=reason,
        )
    )
    return best, records, f"consensus_fallback:{reason}"
