"""Automatic k selection for explicit cosine and Euclidean FCM geometries."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from clustering_types import FCMKCandidate
from fcm_core import DEFAULT_FCM_MIN_CENTER_SEPARATION, DEFAULT_FCM_N_INIT
from fcm_geometry import ExperimentalFcmGeometry, GeometryName, geometry_fcm
from fcm_validity import (
    _candidate_is_valid,
    _candidate_to_record,
    _score_multi_metric_candidates,
    modified_partition_coefficient,
    normalized_partition_entropy,
    partition_coefficient,
    partition_entropy,
)


def _xie_beni(
    samples: np.ndarray,
    candidate_result: Any,
    geometry: ExperimentalFcmGeometry,
) -> float:
    squared = np.asarray(candidate_result.squared_dissimilarities)
    numerator = np.sum((candidate_result.memberships**candidate_result.m) * squared)
    center_squared = geometry.squared_dissimilarities(
        candidate_result.centers,
        candidate_result.centers,
    )
    np.fill_diagonal(center_squared, np.inf)
    return float(numerator / max(len(samples) * np.min(center_squared), 1e-12))


def select_geometry_fcm_cluster_count(
    X: np.ndarray,
    *,
    geometry_name: GeometryName,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 20,
    seed: int = 42,
    n_init: int = DEFAULT_FCM_N_INIT,
    max_attempts: int | None = None,
    min_center_separation: float = DEFAULT_FCM_MIN_CENTER_SEPARATION,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    xb_worsening_patience: int = 2,
) -> tuple[FCMKCandidate | None, list[dict[str, Any]], str]:
    """Select k with the production multi-metric policy for one geometry."""

    if min_clusters < 2 or max_clusters < min_clusters:
        raise ValueError("invalid cluster-count range")
    if min_child_size < 2:
        raise ValueError("min_child_size must be at least 2")
    if xb_worsening_patience < 0:
        raise ValueError("xb_worsening_patience must be non-negative")

    geometry = ExperimentalFcmGeometry(geometry_name)
    samples = geometry.prepare_samples(X)
    maximum_k = min(max_clusters, len(samples) // min_child_size)
    if maximum_k < min_clusters:
        return None, [], "too_few_samples_for_two_valid_children"

    candidates: list[FCMKCandidate] = []
    stop_k: int | None = None
    for candidate_k in range(min_clusters, maximum_k + 1):
        result = geometry_fcm(
            samples,
            candidate_k,
            geometry_name=geometry_name,
            seed=seed + candidate_k * 1009,
            n_init=n_init,
            max_attempts=max_attempts,
            min_cluster_size=min_child_size,
            min_center_separation=min_center_separation,
            m=m,
            max_iter=max_iter,
            tol=tol,
        )
        labels = result.memberships.argmax(axis=1)
        cluster_sizes = [
            int(np.sum(labels == cluster_id)) for cluster_id in range(candidate_k)
        ]
        try:
            silhouette = float(
                silhouette_score(samples, labels, metric=geometry.silhouette_metric())
            )
        except Exception:
            silhouette = float("nan")
        xie_beni = _xie_beni(samples, result, geometry)
        previous_xb = candidates[-1].xie_beni if candidates else None
        relative_improvement = (
            None
            if previous_xb is None
            else float((previous_xb - xie_beni) / max(abs(previous_xb), 1e-12))
        )
        candidate = FCMKCandidate(
            n_clusters=candidate_k,
            result=result,
            labels=labels,
            silhouette=silhouette,
            xie_beni=xie_beni,
            xb_relative_improvement=relative_improvement,
            partition_coefficient=partition_coefficient(result),
            modified_partition_coefficient=modified_partition_coefficient(result),
            partition_entropy=partition_entropy(result),
            normalized_partition_entropy=normalized_partition_entropy(result),
            selection_score=None,
            objective=float(result.objective),
            noise_count=0,
            cluster_sizes=cluster_sizes,
            m=result.m,
            restart_stability=result.restart_stability,
            valid_restarts=result.valid_restarts,
            attempts=result.attempts,
            minimum_center_distance=result.minimum_center_distance,
        )
        candidates.append(candidate)

        if (
            len(candidates) >= 2
            and relative_improvement is not None
            and relative_improvement < 0.0
            and _candidate_is_valid(
                candidates[-2],
                min_child_size=min_child_size,
                min_center_separation=min_center_separation,
                selection_method="multi_metric",
            )
            and _candidate_is_valid(
                candidate,
                min_child_size=min_child_size,
                min_center_separation=min_center_separation,
                selection_method="multi_metric",
            )
            and stop_k is None
        ):
            stop_k = min(maximum_k, candidate_k + xb_worsening_patience)
        if stop_k is not None and candidate_k >= stop_k:
            break

    valid = [
        candidate
        for candidate in candidates
        if _candidate_is_valid(
            candidate,
            min_child_size=min_child_size,
            min_center_separation=min_center_separation,
            selection_method="multi_metric",
        )
    ]
    if not valid:
        return None, [_candidate_to_record(item) for item in candidates], "no_valid_xie_beni_split"

    _score_multi_metric_candidates(valid)
    best = max(
        valid,
        key=lambda item: (
            item.selection_score,
            -item.xie_beni,
            item.restart_stability,
            item.silhouette,
            item.modified_partition_coefficient,
            -item.n_clusters,
        ),
    )
    reason = (
        "selected_multi_metric_xb_worsening_patience"
        if stop_k is not None
        else "selected_multi_metric_max_k"
    )
    return best, [_candidate_to_record(item) for item in candidates], reason
