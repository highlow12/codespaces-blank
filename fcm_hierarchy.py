from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from clustering_types import (
    FCMKCandidate,
    FCMResult,
    HierarchicalModel,
    HierarchicalResult,
    HierarchyNodeModel,
)


def spherical_fcm(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> FCMResult:
    """Run FCM on the unit sphere using Euclidean distances.

    Every invocation re-normalizes its input. The weighted FCM centers are
    projected back to unit length after each update, so both samples and
    centers remain on the same sphere while the distance calculation stays
    ordinary Euclidean distance.
    """

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if m <= 1.0:
        raise ValueError("m must be greater than 1")

    X = normalize(X, norm="l2")
    rng = np.random.default_rng(seed)
    memberships = rng.random((X.shape[0], n_clusters))
    memberships /= memberships.sum(axis=1, keepdims=True)

    epsilon = 1e-12
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    for iteration in range(1, max_iter + 1):
        previous = memberships.copy()
        weights = memberships**m
        centers = weights.T @ X
        centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        zero_center_mask = centers_norm[:, 0] < epsilon
        if np.any(zero_center_mask):
            replacement_indices = rng.integers(
                0,
                X.shape[0],
                size=int(np.sum(zero_center_mask)),
            )
            centers[zero_center_mask] = X[replacement_indices]
            centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(centers_norm, epsilon)

        distances = euclidean_distances(X, centers)
        distances = np.maximum(distances, epsilon)

        exponent = 2.0 / (m - 1.0)
        ratios = (distances[:, :, None] / distances[:, None, :]) ** exponent
        memberships = 1.0 / ratios.sum(axis=2)

        change = np.max(np.abs(memberships - previous))
        if change < tol:
            break

    labels = memberships.argmax(axis=1)
    return FCMResult(
        labels=labels,
        memberships=memberships,
        centers=centers,
        iterations=iteration,
    )


def xie_beni_index(X: np.ndarray, result: FCMResult) -> float:
    X = normalize(X, norm="l2")
    memberships = result.memberships
    centers = result.centers
    distances = euclidean_distances(X, centers)
    numerator = np.sum((memberships**2) * (distances**2))
    center_distances = euclidean_distances(centers, centers)
    np.fill_diagonal(center_distances, np.inf)
    denominator = X.shape[0] * np.min(center_distances) ** 2
    return float(numerator / max(denominator, 1e-12))


def fuzzy_silhouette_proxy(
    X: np.ndarray,
    result: FCMResult,
    *,
    m: float = 2.0,
) -> float:
    X = normalize(X, norm="l2")
    memberships = result.memberships
    centers = result.centers
    distances = euclidean_distances(X, centers)
    weights = memberships**m
    a = np.sum(weights * distances, axis=1) / np.sum(weights, axis=1)
    b = np.partition(distances, 1, axis=1)[:, 1]
    scores = (b - a) / np.maximum(a, b)
    return float(np.mean(scores))


def pca_normalized_features(
    X: np.ndarray,
    *,
    n_components: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Create the same normalized PCA representation used by PCA+FCM."""

    projected, _ = fit_pca_normalized_features(
        X,
        n_components=n_components,
        seed=seed,
    )
    return projected


def fit_pca_normalized_features(
    X: np.ndarray,
    *,
    n_components: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, PCA]:
    """Fit the PCA representation and return its transformer for later batches."""

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if n_components < 1:
        raise ValueError("n_components must be at least 1")

    component_count = min(n_components, X.shape[0], X.shape[1])
    normalized_input = normalize(X, norm="l2")
    pca = PCA(n_components=component_count, random_state=seed).fit(
        normalized_input
    )
    projected = pca.transform(normalized_input)
    return normalize(projected, norm="l2"), pca


def transform_pca_normalized_features(X: np.ndarray, pca: PCA) -> np.ndarray:
    """Transform a future batch with a previously fitted PCA representation."""

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    normalized_input = normalize(X, norm="l2")
    projected = pca.transform(normalized_input)
    return normalize(projected, norm="l2")


def fcm_memberships_from_centers(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate FCM memberships for new points using fixed centers."""

    if m <= 1.0:
        raise ValueError("m must be greater than 1")
    if X.ndim != 2 or centers.ndim != 2:
        raise ValueError("X and centers must be 2D arrays")
    if centers.shape[0] < 1 or X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers have incompatible shapes")

    normalized_X = normalize(X, norm="l2")
    normalized_centers = normalize(centers, norm="l2")
    distances = euclidean_distances(normalized_X, normalized_centers)
    distances = np.maximum(distances, 1e-12)
    exponent = 2.0 / (m - 1.0)
    ratios = (distances[:, :, None] / distances[:, None, :]) ** exponent
    memberships = 1.0 / ratios.sum(axis=2)
    return memberships, distances


def fcm_noise_mask(
    X: np.ndarray,
    result: FCMResult,
    *,
    min_membership: float = 0.40,
    distance_z: float = 3.5,
) -> np.ndarray:
    """Identify ambiguous or outlying points in a spherical FCM result."""

    if not 0.0 <= min_membership <= 1.0:
        raise ValueError("min_membership must be between 0 and 1")
    if distance_z < 0.0:
        raise ValueError("distance_z must be non-negative")

    Xn = normalize(X, norm="l2")
    labels = result.labels
    memberships = result.memberships
    distances = euclidean_distances(Xn, result.centers)
    row_indices = np.arange(Xn.shape[0])
    assigned_distances = distances[row_indices, labels]

    noise = memberships.max(axis=1) < min_membership
    for cluster_id in range(result.memberships.shape[1]):
        cluster_mask = labels == cluster_id
        cluster_distances = assigned_distances[cluster_mask]
        if cluster_distances.size < 4:
            continue

        median = float(np.median(cluster_distances))
        mad = float(np.median(np.abs(cluster_distances - median)))
        if mad <= 1e-12:
            continue
        robust_scale = 1.4826 * mad
        threshold = median + distance_z * robust_scale
        noise |= cluster_mask & (assigned_distances > threshold)

    return noise


def spherical_fcm_objective(
    X: np.ndarray,
    result: FCMResult,
    *,
    m: float = 2.0,
) -> float:
    """Return the Euclidean fuzzy compactness objective on the unit sphere."""

    Xn = normalize(X, norm="l2")
    distances = euclidean_distances(Xn, result.centers)
    return float(np.sum((result.memberships**m) * (distances**2)) / Xn.shape[0])


def _filter_fcm_labels(
    X: np.ndarray,
    result: FCMResult,
    *,
    min_child_size: int,
    min_membership: float,
    distance_z: float,
) -> tuple[np.ndarray, list[int]]:
    """Apply noise rules and remap surviving FCM labels to contiguous IDs."""

    labels = result.labels.copy()
    labels[
        fcm_noise_mask(
            X,
            result,
            min_membership=min_membership,
            distance_z=distance_z,
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
        "objective": finite_or_none(candidate.objective),
        "valid_clusters": int(len(candidate.cluster_sizes)),
        "noise_count": int(candidate.noise_count),
        "cluster_sizes": [int(size) for size in candidate.cluster_sizes],
    }


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


def select_fcm_cluster_count(
    X: np.ndarray,
    *,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_child_size: int = 20,
    min_membership: float = 0.40,
    distance_z: float = 3.5,
    selection_method: str = "silhouette",
    seed: int = 42,
) -> tuple[FCMKCandidate | None, list[dict[str, Any]], str]:
    """Evaluate a node's variable k and return the best FCM split."""

    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be at least min_clusters")
    if min_child_size < 2:
        raise ValueError("min_child_size must be at least 2")
    if selection_method not in {"silhouette", "knee"}:
        raise ValueError("selection_method must be 'silhouette' or 'knee'")

    Xn = normalize(X, norm="l2")
    node_size = Xn.shape[0]
    maximum_k = min(max_clusters, node_size // min_child_size)
    if maximum_k < min_clusters:
        return None, [], "too_few_samples_for_two_valid_children"

    candidates: list[FCMKCandidate] = []
    for candidate_k in range(min_clusters, maximum_k + 1):
        result = spherical_fcm(
            Xn,
            n_clusters=candidate_k,
            seed=seed + candidate_k * 1009,
        )
        labels, cluster_sizes = _filter_fcm_labels(
            Xn,
            result,
            min_child_size=min_child_size,
            min_membership=min_membership,
            distance_z=distance_z,
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

        candidates.append(
            FCMKCandidate(
                n_clusters=candidate_k,
                result=result,
                labels=labels,
                silhouette=silhouette,
                xie_beni=xie_beni_index(Xn, result),
                objective=spherical_fcm_objective(Xn, result),
                noise_count=int(np.sum(~non_noise)),
                cluster_sizes=cluster_sizes,
            )
        )

    valid_candidates = [
        candidate
        for candidate in candidates
        if len(candidate.cluster_sizes) >= 2 and np.isfinite(candidate.silhouette)
    ]
    if not valid_candidates:
        return (
            None,
            [_candidate_to_record(candidate) for candidate in candidates],
            "no_valid_silhouette_split",
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
    else:
        best = _choose_knee_candidate(valid_candidates)

    return best, [_candidate_to_record(candidate) for candidate in candidates], "selected"


def run_hierarchical_pca_fcm(
    X: np.ndarray,
    metadata: pd.DataFrame | None = None,
    *,
    max_depth: int = 4,
    min_node_size: int = 60,
    min_child_size: int = 20,
    min_clusters: int = 2,
    max_clusters: int = 8,
    min_membership: float = 0.40,
    distance_z: float = 3.5,
    selection_method: str = "silhouette",
    min_split_silhouette: float = 0.05,
    pca_components: int = 64,
    seed: int = 42,
) -> HierarchicalResult:
    """Recursively split a dataset with spherical PCA+FCM."""

    started_at = time.perf_counter()
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if metadata is not None and len(metadata) != X.shape[0]:
        raise ValueError("metadata must contain exactly one row per embedding")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if min_node_size < min_child_size:
        raise ValueError("min_node_size must be at least min_child_size")
    if min_child_size < 2:
        raise ValueError("min_child_size must be at least 2")
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be at least min_clusters")
    if selection_method not in {"silhouette", "knee"}:
        raise ValueError("selection_method must be 'silhouette' or 'knee'")
    if min_split_silhouette < -1.0 or min_split_silhouette > 1.0:
        raise ValueError("min_split_silhouette must be between -1 and 1")

    if metadata is None:
        metadata = pd.DataFrame({"id": np.arange(X.shape[0])})
    else:
        metadata = metadata.copy()

    Xp, pca = fit_pca_normalized_features(
        X,
        n_components=pca_components,
        seed=seed,
    )
    labels_by_level = np.full((X.shape[0], max_depth), -1, dtype=int)
    soft_memberships_by_level = [
        np.full((X.shape[0], max_clusters), np.nan, dtype=np.float64)
        for _ in range(max_depth)
    ]
    is_noise = np.zeros(X.shape[0], dtype=bool)
    noise_level = np.full(X.shape[0], -1, dtype=int)
    node_models: dict[str, HierarchyNodeModel] = {}

    def node_template(
        *,
        node_id: str,
        parent_id: str | None,
        path: str,
        depth: int,
        size: int,
    ) -> dict[str, Any]:
        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "path": path,
            "depth": depth,
            "size": int(size),
            "selected_k": None,
            "selected_silhouette": None,
            "selected_valid_clusters": 0,
            "noise_count": 0,
            "candidate_metrics": [],
            "stop_reason": None,
            "children": [],
        }

    root = node_template(
        node_id="root",
        parent_id=None,
        path="",
        depth=0,
        size=X.shape[0],
    )

    def make_root_fallback(reason: str) -> None:
        labels_by_level[:, 0] = 0
        soft_memberships_by_level[0][:, 0] = 1.0
        root["stop_reason"] = reason
        root["fallback_single_cluster"] = True
        child = node_template(
            node_id="0",
            parent_id="root",
            path="0",
            depth=1,
            size=X.shape[0],
        )
        child["stop_reason"] = f"root_not_split:{reason}"
        root["children"].append(child)

    def recurse(indices: np.ndarray, node: dict[str, Any], depth: int) -> None:
        if depth >= max_depth:
            node["stop_reason"] = "max_depth_reached"
            return
        if indices.size < min_node_size:
            node["stop_reason"] = "node_too_small"
            return

        best, candidate_metrics, reason = select_fcm_cluster_count(
            Xp[indices],
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_child_size=min_child_size,
            min_membership=min_membership,
            distance_z=distance_z,
            selection_method=selection_method,
            seed=seed + depth * 100_003 + indices.size,
        )
        node["candidate_metrics"] = candidate_metrics
        if best is None:
            node["stop_reason"] = reason
            if depth == 0:
                make_root_fallback(reason)
            return
        if best.silhouette < min_split_silhouette:
            node["stop_reason"] = "silhouette_below_threshold"
            node["selected_silhouette"] = float(best.silhouette)
            if depth == 0:
                make_root_fallback("silhouette_below_threshold")
            return

        node["selected_k"] = int(best.n_clusters)
        node["selected_silhouette"] = float(best.silhouette)
        node["selected_valid_clusters"] = int(len(best.cluster_sizes))
        node["noise_count"] = int(best.noise_count)

        current_level = depth
        local_labels = best.labels

        model_centers: list[np.ndarray] = []
        distance_thresholds: list[float] = []
        surviving_source_labels: list[int] = []
        node_features = Xp[indices]
        for cluster_id in range(len(best.cluster_sizes)):
            cluster_mask = local_labels == cluster_id
            source_labels = best.result.labels[cluster_mask]
            if source_labels.size == 0:
                raise RuntimeError("Selected cluster has no surviving samples")
            source_label = int(np.bincount(source_labels).argmax())
            surviving_source_labels.append(source_label)
            center = best.result.centers[source_label]
            model_centers.append(center.copy())

            cluster_distances = euclidean_distances(
                node_features[cluster_mask],
                center.reshape(1, -1),
            ).ravel()
            if cluster_distances.size < 4:
                distance_thresholds.append(float("inf"))
            else:
                median = float(np.median(cluster_distances))
                mad = float(np.median(np.abs(cluster_distances - median)))
                if mad <= 1e-12:
                    distance_thresholds.append(float("inf"))
                else:
                    distance_thresholds.append(
                        median + distance_z * 1.4826 * mad
                    )

        selected_memberships = best.result.memberships[:, surviving_source_labels]
        membership_sums = selected_memberships.sum(axis=1, keepdims=True)
        normalized_memberships = np.divide(
            selected_memberships,
            membership_sums,
            out=np.zeros_like(selected_memberships),
            where=membership_sums > 1e-12,
        )
        valid_membership_rows = local_labels >= 0
        soft_memberships_by_level[current_level][
            indices[valid_membership_rows], : len(surviving_source_labels)
        ] = normalized_memberships[valid_membership_rows]

        node_models[str(node["path"])] = HierarchyNodeModel(
            path=str(node["path"]),
            depth=current_level,
            centers=np.vstack(model_centers),
            distance_thresholds=np.asarray(distance_thresholds, dtype=np.float64),
        )

        noise_indices = indices[local_labels == -1]
        if noise_indices.size:
            is_noise[noise_indices] = True
            noise_level[noise_indices] = current_level + 1

        for cluster_id, cluster_size in enumerate(best.cluster_sizes):
            child_indices = indices[local_labels == cluster_id]
            if child_indices.size != cluster_size:
                raise RuntimeError("FCM cluster size bookkeeping is inconsistent")
            labels_by_level[child_indices, current_level] = cluster_id
            child_path = (
                f"{node['path']}/{cluster_id}"
                if node["path"]
                else str(cluster_id)
            )
            child = node_template(
                node_id=child_path,
                parent_id=str(node["node_id"]),
                path=child_path,
                depth=current_level + 1,
                size=child_indices.size,
            )
            node["children"].append(child)
            recurse(child_indices, child, depth + 1)

    recurse(np.arange(X.shape[0], dtype=int), root, 0)

    assigned_depth = np.sum(labels_by_level >= 0, axis=1)
    leaf_level = np.where(assigned_depth > 0, assigned_depth, -1).astype(int)
    leaf_cluster = np.full(X.shape[0], -1, dtype=int)
    has_leaf = assigned_depth > 0
    row_indices = np.arange(X.shape[0])
    leaf_cluster[has_leaf] = labels_by_level[
        row_indices[has_leaf], assigned_depth[has_leaf] - 1
    ]
    leaf_cluster[is_noise] = -1

    assignments = metadata.copy()
    for level in range(max_depth):
        assignments[f"level_{level + 1}_cluster"] = labels_by_level[:, level]
        for cluster_id in range(max_clusters):
            assignments[f"level_{level + 1}_membership_{cluster_id}"] = (
                soft_memberships_by_level[level][:, cluster_id]
            )
    assignments["cluster"] = leaf_cluster

    cluster_paths: list[str] = []
    for row in range(X.shape[0]):
        path_parts = [
            str(int(label)) for label in labels_by_level[row] if label >= 0
        ]
        if is_noise[row]:
            cluster_paths.append(
                "/".join(path_parts + ["noise"]) if path_parts else "noise"
            )
        else:
            cluster_paths.append("/".join(path_parts) if path_parts else "root")
    assignments["cluster_path"] = cluster_paths
    assignments["is_noise"] = is_noise
    assignments["noise_level"] = noise_level
    assignments["leaf_level"] = leaf_level

    def collect_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = [node]
        for child in node["children"]:
            nodes.extend(collect_nodes(child))
        return nodes

    nodes = collect_nodes(root)
    non_root_nodes = [node for node in nodes if node["depth"] > 0]
    levels_reached = int(np.max(assigned_depth)) if np.any(assigned_depth) else 0
    summary: dict[str, Any] = {
        "method": f"recursive_pca{Xp.shape[1]}_spherical_fcm",
        "samples": int(X.shape[0]),
        "pca_components": int(Xp.shape[1]),
        "levels_requested": int(max_depth),
        "levels_reached": levels_reached,
        "node_count": int(len(non_root_nodes)),
        "leaf_count": int(sum(not node["children"] for node in nodes)),
        "noise_count": int(np.sum(is_noise)),
        "noise_by_level": {
            str(level): int(np.sum(noise_level == level))
            for level in range(1, max_depth + 1)
        },
        "leaf_cluster_count": int(
            assignments.loc[~assignments["is_noise"], "cluster_path"].nunique()
        ),
        "runtime_sec": float(time.perf_counter() - started_at),
    }
    config = {
        "max_depth": int(max_depth),
        "min_node_size": int(min_node_size),
        "min_child_size": int(min_child_size),
        "min_clusters": int(min_clusters),
        "max_clusters": int(max_clusters),
        "selection_method": selection_method,
        "min_split_silhouette": float(min_split_silhouette),
        "min_membership": float(min_membership),
        "distance_z": float(distance_z),
        "pca_components_requested": int(pca_components),
        "seed": int(seed),
    }
    tree = {"config": config, "summary": summary, "root": root}
    model = HierarchicalModel(
        pca=pca,
        nodes=node_models,
        max_depth=max_depth,
        fallback_single_cluster=bool(root.get("fallback_single_cluster", False)),
    )
    return HierarchicalResult(
        assignments=assignments,
        tree=tree,
        summary=summary,
        memberships={
            level + 1: level_memberships
            for level, level_memberships in enumerate(soft_memberships_by_level)
        },
        model=model,
    )
