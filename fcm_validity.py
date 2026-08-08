"""FCM validity metrics and automatic cluster-count selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from clustering_types import FCMKCandidate, FCMResult
from fcm_core import (
    DEFAULT_FCM_MIN_CENTER_SEPARATION,
    DEFAULT_FCM_N_INIT,
    spherical_fcm,
)
from fcm_document_classification import (
    DEFAULT_MAX_MEMBERSHIP_GAP,
    fcm_noise_mask,
)
from fuzzy_cmeans import SphericalGeometry


FCM_SELECTION_METHODS = frozenset(
    {"silhouette", "knee", "xie_beni", "multi_metric"}
)




def _sfcm_metric_inputs(
    X: np.ndarray,
    result: FCMResult,
    *,
    squared_dissimilarities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = SphericalGeometry()
    normalized = geometry.prepare_samples(X)
    memberships = _validated_memberships(result)
    centers = geometry.prepare_samples(result.centers)
    if squared_dissimilarities is None:
        values = geometry.squared_dissimilarities(normalized, centers)
    else:
        values = np.asarray(squared_dissimilarities, dtype=np.float64)
        expected_shape = (normalized.shape[0], centers.shape[0])
        if values.shape != expected_shape:
            raise ValueError(
                "squared_dissimilarities must align with samples and centers"
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "squared_dissimilarities must be finite and non-negative"
            )
    return memberships, centers, values


def xie_beni_index(
    X: np.ndarray,
    result: FCMResult,
    *,
    squared_dissimilarities: np.ndarray | None = None,
) -> float:
    memberships, centers, squared_dissimilarities = _sfcm_metric_inputs(
        X,
        result,
        squared_dissimilarities=squared_dissimilarities,
    )
    numerator = np.sum(
        (memberships ** float(result.m)) * squared_dissimilarities
    )
    center_dissimilarities = SphericalGeometry().squared_dissimilarities(
        centers,
        centers,
    )
    np.fill_diagonal(center_dissimilarities, np.inf)
    denominator = X.shape[0] * np.min(center_dissimilarities)
    return float(numerator / max(denominator, 1e-12))


def _validated_memberships(result: FCMResult) -> np.ndarray:
    memberships = np.asarray(result.memberships, dtype=np.float64)
    if memberships.ndim != 2 or memberships.shape[0] == 0:
        raise ValueError("memberships must be a non-empty 2D array")
    return memberships


def partition_coefficient(result: FCMResult) -> float:
    """Return the FCM partition coefficient (higher is crisper)."""

    memberships = _validated_memberships(result)
    return float(np.mean(np.sum(memberships**2, axis=1)))


def modified_partition_coefficient(result: FCMResult) -> float:
    """Remove the raw partition coefficient's 1/k baseline."""

    cluster_count = result.memberships.shape[1]
    if cluster_count < 2:
        return 1.0
    coefficient = partition_coefficient(result)
    baseline = 1.0 / cluster_count
    return float((coefficient - baseline) / (1.0 - baseline))


def partition_entropy(result: FCMResult) -> float:
    """Return fuzzy partition entropy (lower is crisper)."""

    memberships = _validated_memberships(result)
    safe_memberships = np.maximum(memberships, 1e-12)
    return float(-np.mean(np.sum(memberships * np.log(safe_memberships), axis=1)))


def normalized_partition_entropy(result: FCMResult) -> float:
    """Normalize partition entropy to [0, 1] using log(k)."""

    cluster_count = result.memberships.shape[1]
    if cluster_count < 2:
        return 0.0
    return float(partition_entropy(result) / np.log(cluster_count))


def fuzzy_silhouette_proxy(
    X: np.ndarray,
    result: FCMResult,
    *,
    m: float = 2.0,
) -> float:
    memberships, _centers, squared_dissimilarities = _sfcm_metric_inputs(X, result)
    distances = np.sqrt(squared_dissimilarities)
    weights = memberships**m
    a = np.sum(weights * distances, axis=1) / np.sum(weights, axis=1)
    b = np.partition(distances, 1, axis=1)[:, 1]
    scores = (b - a) / np.maximum(a, b)


def spherical_fcm_objective(
    X: np.ndarray,
    result: FCMResult,
    *,
    m: float | None = None,
    squared_dissimilarities: np.ndarray | None = None,
) -> float:
    """Return the cosine-equivalent fuzzy compactness on the unit sphere."""

    memberships, _centers, squared_dissimilarities = _sfcm_metric_inputs(
        X,
        result,
        squared_dissimilarities=squared_dissimilarities,
    )
    exponent = float(result.m if m is None else m)
    return float(
        np.sum((memberships**exponent) * squared_dissimilarities)
        / squared_dissimilarities.shape[0]
    )


def _filter_fcm_labels(
    X: np.ndarray,
    result: FCMResult,
    *,
    min_child_size: int,
    min_membership: float,
    max_membership_gap: float,
    distance_z: float,
    assigned_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Apply noise rules and remap surviving FCM labels to contiguous IDs."""

    labels = result.labels.copy()
    labels[
        fcm_noise_mask(
            X,
            result,
            min_membership=min_membership,
            max_membership_gap=max_membership_gap,
            distance_z=distance_z,
            assigned_distances=assigned_distances,
        )
    ] = -1

    surviving_old_labels = [
        cluster_id
        for cluster_id in range(result.memberships.shape[1])
        if int(np.sum(labels == cluster_id)) >= min_child_size
    ]
    filtered = np.full(labels.shape, -1, dtype=int)
    cluster_sizes: list[int] = []
    for new_label, old_label in enumerate(surviving_old_labels):
        mask = labels == old_label
        filtered[mask] = new_label
        cluster_sizes.append(int(np.sum(mask)))
    return filtered, cluster_sizes


def _candidate_to_record(candidate: FCMKCandidate) -> dict[str, Any]:
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "k": int(candidate.n_clusters),
        "silhouette": finite_or_none(candidate.silhouette),
        "xie_beni": finite_or_none(candidate.xie_beni),
        "xb_relative_improvement": (
            finite_or_none(candidate.xb_relative_improvement)
            if candidate.xb_relative_improvement is not None
            else None
        ),
        "partition_coefficient": finite_or_none(
            candidate.partition_coefficient
        ),
        "modified_partition_coefficient": finite_or_none(
            candidate.modified_partition_coefficient
        ),
        "partition_entropy": finite_or_none(candidate.partition_entropy),
        "normalized_partition_entropy": finite_or_none(
            candidate.normalized_partition_entropy
        ),
        "selection_score": (
            finite_or_none(candidate.selection_score)
            if candidate.selection_score is not None
            else None
        ),
        "restart_stability": finite_or_none(candidate.restart_stability),
        "valid_restarts": int(candidate.valid_restarts),
        "attempts": int(candidate.attempts),
        "minimum_center_distance": (
            finite_or_none(candidate.minimum_center_distance)
            if candidate.minimum_center_distance is not None
            else None
        ),
        "objective": finite_or_none(candidate.objective),
        "m": float(candidate.m),
        "valid_clusters": int(len(candidate.cluster_sizes)),
        "noise_count": int(candidate.noise_count),
        "cluster_sizes": [int(size) for size in candidate.cluster_sizes],
    }


def _rank_desirability(
    values: np.ndarray,
    *,
    higher_is_better: bool,
) -> np.ndarray:
    """Convert metric ranks to [0, 1] desirability with averaged ties."""

    if not np.all(np.isfinite(values)):
        raise ValueError("rank desirability requires finite values")
    if values.size == 1:
        return np.ones(values.shape, dtype=np.float64)

    sort_values = -values if higher_is_better else values
    order = np.argsort(sort_values, kind="stable")
    ordered_values = sort_values[order]
    ranks = np.empty(values.shape, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and np.isclose(
            ordered_values[end],
            ordered_values[start],
            rtol=1e-12,
            atol=1e-12,
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return 1.0 - (ranks - 1.0) / (values.size - 1.0)


def _score_multi_metric_candidates(candidates: list[FCMKCandidate]) -> None:
    """Score compactness, separation, restart stability, and fuzziness."""

    if not candidates:
        return
    metric_specs = (
        ("xie_beni", False, 0.25),
        ("silhouette", True, 0.45),
        ("restart_stability", True, 0.20),
        ("modified_partition_coefficient", True, 0.10),
    )
    weighted_desirabilities = []
    for attribute, higher_is_better, weight in metric_specs:
        values = np.asarray(
            [getattr(candidate, attribute) for candidate in candidates],
            dtype=np.float64,
        )
        weighted_desirabilities.append(
            weight
            * _rank_desirability(
                values,
                higher_is_better=higher_is_better,
            )
        )
    scores = np.sum(np.vstack(weighted_desirabilities), axis=0)
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.selection_score = float(score)


def _choose_knee_candidate(candidates: list[FCMKCandidate]) -> FCMKCandidate:
    """Choose the largest curvature point of the FCM objective curve."""

    if len(candidates) <= 2:
        return candidates[0]

    objectives = np.asarray([candidate.objective for candidate in candidates], dtype=float)
    if not np.all(np.isfinite(objectives)) or np.ptp(objectives) <= 1e-12:
        return candidates[0]

    x = np.linspace(0.0, 1.0, len(candidates))
    y = (objectives - objectives.min()) / np.ptp(objectives)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 1e-12:
        return candidates[0]

    points = np.column_stack([x, y])
    distances = (
        np.abs(
            line[0] * (start[1] - points[:, 1])
            - line[1] * (start[0] - points[:, 0])
        )
        / line_norm
    )
    best_distance = float(np.max(distances))
    best_indices = np.flatnonzero(np.isclose(distances, best_distance))
    return candidates[int(best_indices[0])]


def _validate_fcm_selection_parameters(
    *,
    min_clusters: int,
    max_clusters: int,
    min_child_size: int,
    min_membership: float,
    max_membership_gap: float,
    selection_method: str,
    min_xb_relative_improvement: float,
    xb_worsening_patience: int,
    n_init: int = DEFAULT_FCM_N_INIT,
    max_attempts: int | None = None,
    min_center_separation: float = DEFAULT_FCM_MIN_CENTER_SEPARATION,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> None:
    if not 0.0 <= min_membership <= 1.0:
        raise ValueError("min_membership must be between 0 and 1")
    if not 0.0 <= max_membership_gap <= 1.0:
        raise ValueError("max_membership_gap must be between 0 and 1")
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be at least min_clusters")
    if min_child_size < 2:
        raise ValueError("min_child_size must be at least 2")
    if selection_method not in FCM_SELECTION_METHODS:
        raise ValueError(
            "selection_method must be 'silhouette', 'knee', 'xie_beni', "
            "or 'multi_metric'"
        )
    if not 0.0 <= min_xb_relative_improvement <= 1.0:
        raise ValueError("min_xb_relative_improvement must be between 0 and 1")
    if xb_worsening_patience < 0:
        raise ValueError("xb_worsening_patience must be non-negative")
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    if max_attempts is not None and max_attempts < n_init:
        raise ValueError("max_attempts must be at least n_init")
    if min_center_separation < 0.0:
        raise ValueError("min_center_separation must be non-negative")
    if m <= 1.0:
        raise ValueError("m must be greater than 1")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0.0:
        raise ValueError("tol must be positive")


def _candidate_is_valid(
    candidate: FCMKCandidate,
    *,
    min_child_size: int,
    min_center_separation: float,
    selection_method: str,
) -> bool:
    if (
        len(candidate.cluster_sizes) < 2
        or min(candidate.cluster_sizes) < min_child_size
        or candidate.valid_restarts < 1
        or not np.isfinite(candidate.xie_beni)
        or not np.isfinite(candidate.silhouette)
    ):
        return False
    if (
        candidate.minimum_center_distance is not None
        and candidate.minimum_center_distance < min_center_separation
    ):
        return False
    if selection_method == "multi_metric":
        return bool(
            np.isfinite(candidate.modified_partition_coefficient)
            and np.isfinite(candidate.restart_stability)
        )
    return True


def select_fcm_cluster_count(
    X: np.ndarray,
    *,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 20,
    min_membership: float = 0.40,
    max_membership_gap: float = DEFAULT_MAX_MEMBERSHIP_GAP,
    distance_z: float = 3.5,
    selection_method: str = "multi_metric",
    min_xb_relative_improvement: float = 0.05,
    xb_worsening_patience: int = 2,
    seed: int = 42,
    n_init: int = DEFAULT_FCM_N_INIT,
    max_attempts: int | None = None,
    min_center_separation: float = DEFAULT_FCM_MIN_CENTER_SEPARATION,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    collapse_center_separation: float | None = None,
) -> tuple[FCMKCandidate | None, list[dict[str, Any]], str]:
    """Evaluate increasing k values and return the best FCM split.

    With ``selection_method="multi_metric"``, candidates are evaluated from
    the configured minimum k upward. After XB first worsens, two additional k
    values are evaluated by default. XB, silhouette, restart stability, and
    modified partition coefficient are converted to rank desirabilities and
    combined with weights 0.25, 0.45, 0.20, and 0.10. Partition entropy is
    retained for diagnostics but is not scored.
    """

    _validate_fcm_selection_parameters(
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_child_size=min_child_size,
        min_membership=min_membership,
        max_membership_gap=max_membership_gap,
        selection_method=selection_method,
        min_xb_relative_improvement=min_xb_relative_improvement,
        xb_worsening_patience=xb_worsening_patience,
        n_init=n_init,
        max_attempts=max_attempts,
        min_center_separation=min_center_separation,
        m=m,
        max_iter=max_iter,
        tol=tol,
    )

    Xn = normalize(X, norm="l2")
    node_size = Xn.shape[0]
    maximum_k = min(max_clusters, node_size // min_child_size)
    if maximum_k < min_clusters:
        return None, [], "too_few_samples_for_two_valid_children"

    candidates: list[FCMKCandidate] = []
    xb_stop_candidate: FCMKCandidate | None = None
    multi_metric_stop_k: int | None = None
    for candidate_k in range(min_clusters, maximum_k + 1):
        result = spherical_fcm(
            Xn,
            n_clusters=candidate_k,
            seed=seed + candidate_k * 1009,
            n_init=n_init,
            max_attempts=max_attempts,
            min_cluster_size=min_child_size,
            min_center_separation=min_center_separation,
            m=m,
            max_iter=max_iter,
            tol=tol,
            collapse_center_separation=collapse_center_separation,
        )
        cached_squared_dissimilarities = result.squared_dissimilarities
        if (
            cached_squared_dissimilarities is not None
            and np.asarray(cached_squared_dissimilarities).shape
            != result.memberships.shape
        ):
            cached_squared_dissimilarities = None
        assigned_distances = None
        if cached_squared_dissimilarities is not None:
            row_indices = np.arange(result.labels.shape[0])
            assigned_distances = np.sqrt(
                cached_squared_dissimilarities[row_indices, result.labels]
            )
        if selection_method == "multi_metric":
            labels = result.memberships.argmax(axis=1)
            cluster_sizes = [
                int(np.sum(labels == cluster_id))
                for cluster_id in range(candidate_k)
            ]
        else:
            labels, cluster_sizes = _filter_fcm_labels(
                Xn,
                result,
                min_child_size=min_child_size,
                min_membership=min_membership,
                max_membership_gap=max_membership_gap,
                distance_z=distance_z,
                assigned_distances=assigned_distances,
            )
        non_noise = labels != -1
        valid_cluster_count = len(cluster_sizes)
        silhouette = float("nan")
        if valid_cluster_count >= 2 and int(np.sum(non_noise)) >= 2:
            try:
                silhouette = float(
                    silhouette_score(
                        Xn[non_noise],
                        labels[non_noise],
                        metric="euclidean",
                    )
                )
            except Exception:
                silhouette = float("nan")

        xie_beni = xie_beni_index(
            Xn,
            result,
            squared_dissimilarities=cached_squared_dissimilarities,
        )
        xb_relative_improvement: float | None = None
        if (
            candidates
            and np.isfinite(candidates[-1].xie_beni)
            and np.isfinite(xie_beni)
        ):
            previous_xb = candidates[-1].xie_beni
            xb_relative_improvement = float(
                (previous_xb - xie_beni) / max(abs(previous_xb), 1e-12)
            )

        candidate = FCMKCandidate(
            n_clusters=candidate_k,
            result=result,
            labels=labels,
            silhouette=silhouette,
            xie_beni=xie_beni,
            xb_relative_improvement=xb_relative_improvement,
            partition_coefficient=partition_coefficient(result),
            modified_partition_coefficient=modified_partition_coefficient(
                result
            ),
            partition_entropy=partition_entropy(result),
            normalized_partition_entropy=normalized_partition_entropy(result),
            selection_score=None,
            objective=(
                result.objective
                if result.objective is not None
                else spherical_fcm_objective(Xn, result, m=m)
            ),
            noise_count=int(np.sum(~non_noise)),
            cluster_sizes=cluster_sizes,
            m=result.m,
            restart_stability=result.restart_stability,
            valid_restarts=result.valid_restarts,
            attempts=result.attempts,
            minimum_center_distance=result.minimum_center_distance,
        )
        candidates.append(candidate)

        previous_candidate = candidates[-2] if len(candidates) >= 2 else None
        current_is_valid = _candidate_is_valid(
            candidate,
            min_child_size=min_child_size,
            min_center_separation=min_center_separation,
            selection_method=selection_method,
        )
        previous_is_valid = (
            previous_candidate is not None
            and _candidate_is_valid(
                previous_candidate,
                min_child_size=min_child_size,
                min_center_separation=min_center_separation,
                selection_method=selection_method,
            )
        )
        if (
            selection_method in {"xie_beni", "multi_metric"}
            and current_is_valid
            and previous_is_valid
            and xb_relative_improvement is not None
            and (
                (
                    selection_method == "xie_beni"
                    and xb_relative_improvement < min_xb_relative_improvement
                )
                or (
                    selection_method == "multi_metric"
                    and xb_relative_improvement < 0.0
                )
            )
        ):
            if selection_method == "xie_beni":
                eligible = [
                    evaluated
                    for evaluated in candidates[:-1]
                    if _candidate_is_valid(
                        evaluated,
                        min_child_size=min_child_size,
                        min_center_separation=min_center_separation,
                        selection_method=selection_method,
                    )
                ]
                xb_stop_candidate = min(
                    eligible,
                    key=lambda evaluated: (
                        evaluated.xie_beni,
                        evaluated.n_clusters,
                    ),
                )
                break
            if multi_metric_stop_k is None:
                multi_metric_stop_k = min(
                    maximum_k,
                    candidate_k + xb_worsening_patience,
                )

        if (
            selection_method == "multi_metric"
            and multi_metric_stop_k is not None
            and candidate_k >= multi_metric_stop_k
        ):
            break

    valid_candidates = [
        candidate
        for candidate in candidates
        if _candidate_is_valid(
            candidate,
            min_child_size=min_child_size,
            min_center_separation=min_center_separation,
            selection_method=selection_method,
        )
    ]
    if not valid_candidates:
        invalid_reason = (
            "no_valid_xie_beni_split"
            if selection_method in {"xie_beni", "multi_metric"}
            else "no_valid_silhouette_split"
        )
        return (
            None,
            [_candidate_to_record(candidate) for candidate in candidates],
            invalid_reason,
        )

    if selection_method == "silhouette":
        best = max(
            valid_candidates,
            key=lambda candidate: (
                candidate.silhouette,
                -candidate.xie_beni
                if np.isfinite(candidate.xie_beni)
                else float("-inf"),
                -candidate.n_clusters,
            ),
        )
    elif selection_method == "knee":
        best = _choose_knee_candidate(valid_candidates)
    elif selection_method == "multi_metric":
        _score_multi_metric_candidates(valid_candidates)
        best = max(
            valid_candidates,
            key=lambda candidate: (
                candidate.selection_score,
                -candidate.xie_beni,
                candidate.restart_stability,
                candidate.silhouette,
                candidate.modified_partition_coefficient,
                -candidate.n_clusters,
            ),
        )
    elif xb_stop_candidate is not None:
        best = xb_stop_candidate
    else:
        best = min(
            valid_candidates,
            key=lambda candidate: (candidate.xie_beni, candidate.n_clusters),
        )

    if selection_method == "multi_metric":
        selection_reason = (
            "selected_multi_metric_xb_worsening_patience"
            if multi_metric_stop_k is not None
            else "selected_multi_metric_max_k"
        )
    elif selection_method == "xie_beni":
        selection_reason = (
            "selected_xb_relative_improvement"
            if xb_stop_candidate is not None
            else "selected_xb_minimum"
        )
    else:
        selection_reason = "selected"
    return (
        best,
        [_candidate_to_record(candidate) for candidate in candidates],
        selection_reason,
    )
