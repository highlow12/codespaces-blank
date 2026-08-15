from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
from hdbscan import all_points_membership_vectors
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from umap import UMAP

from embedding_data import load_embeddings_from_json


@dataclass(frozen=True)
class MergeStep:
    node: int
    left: int
    right: int
    distance: float
    mass: float
    leaves: tuple[int, ...]


def soft_leaf_centers(
    pca_features: np.ndarray,
    memberships: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return L2-normalized membership-weighted leaf centers and masses."""

    features = np.asarray(pca_features, dtype=np.float64)
    weights = np.asarray(memberships, dtype=np.float64)
    if features.ndim != 2 or weights.ndim != 2:
        raise ValueError("Features and memberships must both be 2D")
    if features.shape[0] != weights.shape[0]:
        raise ValueError("Features and memberships must have aligned rows")
    masses = weights.sum(axis=0)
    if np.any(masses <= np.finfo(np.float64).eps):
        raise ValueError("Every leaf must have positive membership mass")
    centers = weights.T @ features
    centers /= masses[:, None]
    return normalize(centers, norm="l2"), masses


def weighted_average_linkage(
    centers: np.ndarray,
    masses: np.ndarray,
) -> list[MergeStep]:
    """Build a deterministic weighted average-linkage tree over leaf centers."""

    centers = normalize(np.asarray(centers, dtype=np.float64), norm="l2")
    masses = np.asarray(masses, dtype=np.float64)
    leaf_count = centers.shape[0]
    if centers.ndim != 2 or leaf_count < 2:
        raise ValueError("At least two 2D leaf centers are required")
    if masses.shape != (leaf_count,) or np.any(masses <= 0):
        raise ValueError("Masses must be positive and aligned with leaf centers")

    base = np.clip(1.0 - centers @ centers.T, 0.0, 2.0)
    distances: dict[tuple[int, int], float] = {}
    for left in range(leaf_count):
        for right in range(left + 1, leaf_count):
            distances[(left, right)] = float(base[left, right])

    active = set(range(leaf_count))
    node_masses = {leaf: float(masses[leaf]) for leaf in active}
    node_leaves = {leaf: (leaf,) for leaf in active}
    merges: list[MergeStep] = []

    for merge_index in range(leaf_count - 1):
        candidates = (
            (distance, left, right)
            for (left, right), distance in distances.items()
            if left in active and right in active
        )
        distance, left, right = min(candidates)
        node = leaf_count + merge_index
        left_mass = node_masses[left]
        right_mass = node_masses[right]
        mass = left_mass + right_mass
        leaves = tuple(sorted((*node_leaves[left], *node_leaves[right])))

        for other in sorted(active - {left, right}):
            left_key = tuple(sorted((left, other)))
            right_key = tuple(sorted((right, other)))
            merged_distance = (
                left_mass * distances[left_key]
                + right_mass * distances[right_key]
            ) / mass
            distances[tuple(sorted((node, other)))] = float(merged_distance)

        active.remove(left)
        active.remove(right)
        active.add(node)
        node_masses[node] = mass
        node_leaves[node] = leaves
        merges.append(
            MergeStep(
                node=node,
                left=left,
                right=right,
                distance=float(distance),
                mass=float(mass),
                leaves=leaves,
            )
        )

    return merges


def cut_tree(
    leaf_count: int,
    merges: list[MergeStep],
    cluster_count: int,
) -> np.ndarray:
    """Map each leaf to one of exactly ``cluster_count`` dendrogram groups."""

    if not 1 <= cluster_count <= leaf_count:
        raise ValueError("cluster_count must be between 1 and leaf_count")
    active: dict[int, tuple[int, ...]] = {
        leaf: (leaf,) for leaf in range(leaf_count)
    }
    for merge in merges[: leaf_count - cluster_count]:
        left = active.pop(merge.left)
        right = active.pop(merge.right)
        active[merge.node] = tuple(sorted((*left, *right)))

    mapping = np.empty(leaf_count, dtype=np.int64)
    ordered_groups = sorted(active.values(), key=lambda leaves: (min(leaves), leaves))
    for group, leaves in enumerate(ordered_groups):
        mapping[np.asarray(leaves, dtype=np.int64)] = group
    return mapping


def lift_leaf_labels(labels: np.ndarray, leaf_mapping: np.ndarray) -> np.ndarray:
    lifted = np.full(len(labels), -1, dtype=np.int64)
    non_noise = labels >= 0
    lifted[non_noise] = leaf_mapping[labels[non_noise]]
    return lifted


def partition_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    non_noise = predicted != -1
    return {
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ari": float(adjusted_rand_score(truth, predicted)),
        "predicted_clusters": int(np.unique(predicted[non_noise]).size),
        "noise_count": int((~non_noise).sum()),
        "noise_ratio": float((~non_noise).mean()),
    }


def merge_gap_candidates(
    leaf_count: int,
    merges: list[MergeStep],
    *,
    limit: int = 5,
) -> list[dict[str, float | int]]:
    candidates: list[dict[str, float | int]] = []
    for index in range(len(merges) - 1):
        current = merges[index].distance
        following = merges[index + 1].distance
        candidates.append(
            {
                "clusters": leaf_count - index - 1,
                "distance_before": current,
                "distance_after": following,
                "distance_gap": following - current,
            }
        )
    return sorted(
        candidates,
        key=lambda item: (item["distance_gap"], item["clusters"]),
        reverse=True,
    )[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a semantic bottom-up hierarchy over HDBSCAN leaves."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("dbpedia_gemini_embeddings.json.gz"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=96)
    parser.add_argument("--umap-components", type=int, default=20)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--min-cluster-size", type=int, default=40)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    if "class_hierarchy" not in metadata or "class" not in metadata:
        raise ValueError("Input metadata must contain class_hierarchy and class")
    hierarchies = metadata["class_hierarchy"].tolist()
    if not hierarchies or any(
        not isinstance(path, list) or len(path) != 3 for path in hierarchies
    ):
        raise ValueError("Every class_hierarchy must contain exactly three levels")

    normalized_embeddings = normalize(embeddings, norm="l2")
    pca = PCA(n_components=args.pca_components, random_state=args.seed)
    pca_features = normalize(
        pca.fit_transform(normalized_embeddings),
        norm="l2",
    )
    umap_model = UMAP(
        n_components=args.umap_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
        n_jobs=1,
    )
    umap_features = umap_model.fit_transform(pca_features)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    ).fit(umap_features)
    leaf_labels = np.asarray(clusterer.labels_, dtype=np.int64)
    leaf_memberships = np.asarray(
        all_points_membership_vectors(clusterer),
        dtype=np.float64,
    )
    if leaf_memberships.ndim != 2:
        raise ValueError("HDBSCAN must discover at least two leaves")
    leaf_count = leaf_memberships.shape[1]
    centers, masses = soft_leaf_centers(pca_features, leaf_memberships)
    merges = weighted_average_linkage(centers, masses)

    truths = {
        "top": np.asarray([path[0] for path in hierarchies]),
        "middle": np.asarray([path[1] for path in hierarchies]),
        "leaf": metadata["class"].astype(str).to_numpy(),
    }
    truth_cluster_counts = {
        name: int(np.unique(labels).size) for name, labels in truths.items()
    }
    requested_cuts = sorted({*truth_cluster_counts.values(), leaf_count})
    cut_labels: dict[int, np.ndarray] = {}
    cut_metrics: dict[str, dict[str, Any]] = {}
    for cluster_count in requested_cuts:
        mapping = cut_tree(leaf_count, merges, cluster_count)
        predicted = lift_leaf_labels(leaf_labels, mapping)
        cut_labels[cluster_count] = predicted
        cut_metrics[str(cluster_count)] = {
            truth_name: partition_metrics(truth, predicted)
            for truth_name, truth in truths.items()
        }

    best_external_cuts: dict[str, dict[str, Any]] = {}
    for truth_name, truth in truths.items():
        candidates: list[dict[str, Any]] = []
        for cluster_count in range(2, leaf_count + 1):
            predicted = lift_leaf_labels(
                leaf_labels,
                cut_tree(leaf_count, merges, cluster_count),
            )
            metrics = partition_metrics(truth, predicted)
            metrics["quality_mean"] = float((metrics["nmi"] + metrics["ari"]) / 2)
            metrics["cut_clusters"] = cluster_count
            candidates.append(metrics)
        best_external_cuts[truth_name] = max(
            candidates,
            key=lambda item: (item["quality_mean"], -item["noise_ratio"]),
        )

    assignments = metadata.copy()
    assignments["hdbscan_leaf"] = leaf_labels
    assignments["hdbscan_probability"] = clusterer.probabilities_
    assignments["hdbscan_outlier_score"] = clusterer.outlier_scores_
    for cluster_count, predicted in cut_labels.items():
        assignments[f"bottom_up_k{cluster_count}"] = predicted

    report = {
        "configuration": {
            "input_json": str(args.input_json),
            "samples": int(len(embeddings)),
            "embedding_dimensions": int(embeddings.shape[1]),
            "pca_components": args.pca_components,
            "post_pca_l2": True,
            "umap_components": args.umap_components,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_metric": "euclidean",
            "hdbscan_min_cluster_size": args.min_cluster_size,
            "hdbscan_min_samples": args.min_samples,
            "hdbscan_cluster_selection_method": "eom",
            "seed": args.seed,
            "noise_policy": "retain label -1 at every hierarchy cut",
            "merge_space": "L2-normalized PCA space",
            "merge_distance": "cosine distance between soft leaf centers",
            "merge_linkage": "membership-mass-weighted average linkage",
        },
        "runtime_sec": float(time.perf_counter() - started),
        "pca_explained_variance_ratio": float(
            pca.explained_variance_ratio_.sum()
        ),
        "hdbscan": {
            "leaf_count": leaf_count,
            "noise_count": int(np.sum(leaf_labels == -1)),
            "noise_ratio": float(np.mean(leaf_labels == -1)),
            "membership_mass": masses.tolist(),
        },
        "truth_cluster_counts": truth_cluster_counts,
        "matched_cut_metrics": cut_metrics,
        "best_label_informed_cuts": best_external_cuts,
        "largest_merge_gaps": merge_gap_candidates(leaf_count, merges),
        "merges": [asdict(merge) for merge in merges],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output_dir / "assignments.csv.gz", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
