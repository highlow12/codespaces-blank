"""Pandas-free bottom-up hierarchy over HDBSCAN leaves."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import normalize


def _validate_labels(labels: Any, n_samples: int) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int64)
    if result.shape != (n_samples,):
        raise ValueError("leaf_labels must contain one value per sample")
    if np.any(result < -1):
        raise ValueError("leaf_labels may only contain -1 or non-negative labels")
    non_noise = result[result >= 0]
    if non_noise.size and not np.array_equal(np.unique(non_noise), np.arange(non_noise.max() + 1)):
        raise ValueError("non-noise leaf_labels must be contiguous from zero")
    return result


def _merge_tree(centers: np.ndarray, masses: np.ndarray) -> list[dict[str, Any]]:
    count = len(centers)
    if count < 2:
        return []
    centers = normalize(np.asarray(centers, dtype=float), norm="l2")
    base = np.clip(1.0 - centers @ centers.T, 0.0, 2.0)
    distances = {(left, right): float(base[left, right]) for left in range(count) for right in range(left + 1, count)}
    active = set(range(count))
    node_mass = {leaf: float(masses[leaf]) for leaf in active}
    node_leaves = {leaf: (leaf,) for leaf in active}
    merges: list[dict[str, Any]] = []
    for index in range(count - 1):
        distance, left, right = min(
            (distance, left, right)
            for (left, right), distance in distances.items()
            if left in active and right in active
        )
        node = count + index
        mass = node_mass[left] + node_mass[right]
        leaves = tuple(sorted((*node_leaves[left], *node_leaves[right])))
        for other in sorted(active - {left, right}):
            left_distance = distances[tuple(sorted((left, other)))]
            right_distance = distances[tuple(sorted((right, other)))]
            distances[tuple(sorted((node, other)))] = float(
                (node_mass[left] * left_distance + node_mass[right] * right_distance) / mass
            )
        active -= {left, right}
        active.add(node)
        node_mass[node] = mass
        node_leaves[node] = leaves
        merges.append({"node": node, "left": left, "right": right, "distance": float(distance), "mass": float(mass), "leaves": list(leaves)})
    return merges


def _cut(leaf_count: int, merges: list[dict[str, Any]], cluster_count: int) -> np.ndarray:
    active = {leaf: (leaf,) for leaf in range(leaf_count)}
    for merge in merges[: leaf_count - cluster_count]:
        active[merge["node"]] = tuple(sorted((*active.pop(merge["left"]), *active.pop(merge["right"]))))
    mapping = np.empty(leaf_count, dtype=np.int64)
    for group, leaves in enumerate(sorted(active.values(), key=lambda item: (min(item), item))):
        mapping[np.asarray(leaves)] = group
    return mapping


def build_hierarchy(
    pca_features: Any,
    leaf_labels: Any,
    memberships: Any,
    *,
    probabilities: Any | None = None,
    outlier_scores: Any | None = None,
) -> dict[str, Any]:
    """Build plain-Python hierarchy artifacts from discovery arrays."""

    features = np.asarray(pca_features, dtype=np.float64)
    if features.ndim != 2 or not len(features):
        raise ValueError("pca_features must be a non-empty 2D array")
    labels = _validate_labels(leaf_labels, len(features))
    weights = np.asarray(memberships, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != len(features):
        raise ValueError("memberships must be a 2D array aligned with pca_features")
    leaf_count = weights.shape[1]
    if labels.size and leaf_count and int(labels.max(initial=-1)) >= leaf_count:
        raise ValueError("leaf_labels exceed membership columns")
    assignments: dict[str, Any] = {"hdbscan_leaf": labels.tolist()}
    for name, values in (("hdbscan_probability", probabilities), ("hdbscan_outlier_score", outlier_scores)):
        if values is not None:
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (len(features),):
                raise ValueError(f"{name} must contain one value per sample")
            assignments[name] = array.tolist()

    masses = np.zeros(leaf_count, dtype=np.float64)
    merges: list[dict[str, Any]] = []
    if leaf_count:
        masses = weights.sum(axis=0)
        if np.any(masses <= np.finfo(float).eps):
            raise ValueError("every discovered leaf must have positive membership mass")
        centers = normalize((weights.T @ features) / masses[:, None], norm="l2")
        merges = _merge_tree(centers, masses)
        for cluster_count in range(leaf_count, 0, -1):
            mapping = _cut(leaf_count, merges, cluster_count)
            lifted = np.full(len(labels), -1, dtype=np.int64)
            mask = labels >= 0
            lifted[mask] = mapping[labels[mask]]
            assignments[f"bottom_up_k{cluster_count}"] = lifted.tolist()

    tree = {
        "schema_version": 1,
        "method": "hdbscan_leaf_bottom_up",
        "merge_space": "membership-weighted normalized PCA space",
        "merge_distance": "cosine distance between soft leaf centers",
        "merge_linkage": "membership-mass-weighted average linkage",
        "leaf_count": int(leaf_count),
        "noise_count": int(np.sum(labels == -1)),
        "leaves": [{"leaf": i, "mass": float(masses[i]), "sample_count": int(np.sum(labels == i))} for i in range(leaf_count)],
        "merges": merges,
        "cuts": [{"cluster_count": k, "assignment_column": f"bottom_up_k{k}"} for k in range(leaf_count, 0, -1)],
    }
    summary = {
        "hierarchy": "hdbscan_leaf_bottom_up",
        "leaf_cluster_count": int(leaf_count),
        "merge_count": len(merges),
        "levels_reached": len(merges) + (1 if leaf_count else 0),
        "noise_count": int(np.sum(labels == -1)),
        "noise_ratio": float(np.mean(labels == -1)),
    }
    return {"assignments": assignments, "tree": tree, "summary": summary}
