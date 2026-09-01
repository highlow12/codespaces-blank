"""Focused tests for the PCA/UMAP/HDBSCAN follow-up benchmark."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import benchmark_pca_umap_hdbscan_followup as benchmark


class FakeUMAP:
    instances: list["FakeUMAP"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.embedding_: np.ndarray | None = None
        FakeUMAP.instances.append(self)

    def fit(self, values: np.ndarray) -> "FakeUMAP":
        components = int(self.kwargs["n_components"])
        self.embedding_ = np.asarray(values[:, :components], dtype=np.float64)
        return self


def phase1_report() -> dict[str, object]:
    rows = [
        {"n_neighbors": 15, "n_components": 20, "min_dist": 0.1, "mean_raw_umap": 0.50, "mean_pca_umap": 0.52},
        {"n_neighbors": 100, "n_components": 20, "min_dist": 0.1, "mean_raw_umap": 0.61, "mean_pca_umap": 0.62},
        {"n_neighbors": 15, "n_components": 5, "min_dist": 0.1, "mean_raw_umap": 0.45, "mean_pca_umap": 0.47},
        {"n_neighbors": 15, "n_components": 20, "min_dist": 0.5, "mean_raw_umap": 0.58, "mean_pca_umap": 0.59},
        {"n_neighbors": 5, "n_components": 5, "min_dist": 0.0, "mean_raw_umap": 0.30, "mean_pca_umap": 0.31},
    ]
    return {"protocol": {"k_values": [5, 15]}, "summary": rows}


class FollowupBenchmarkTest(unittest.TestCase):
    def test_select_configurations_keeps_ablation_roles(self) -> None:
        selected = benchmark.select_configurations(phase1_report(), k_values=(5, 15))
        by_key = {
            (row["n_neighbors"], row["n_components"], row["min_dist"]): row
            for row in selected
        }
        self.assertIn((15, 20, 0.1), by_key)
        self.assertIn((100, 20, 0.1), by_key)
        self.assertIn((15, 5, 0.1), by_key)
        self.assertIn((15, 20, 0.5), by_key)
        self.assertIn((5, 5, 0.0), by_key)
        self.assertIn("production_baseline", by_key[(15, 20, 0.1)]["roles"])
        self.assertIn("n_neighbors_ablation", by_key[(100, 20, 0.1)]["roles"])
        self.assertIn("n_components_ablation", by_key[(15, 5, 0.1)]["roles"])
        self.assertIn("min_dist_ablation", by_key[(15, 20, 0.5)]["roles"])
        self.assertIn("negative_control", by_key[(5, 5, 0.0)]["roles"])

    def test_fit_umap_uses_production_geometry_settings(self) -> None:
        FakeUMAP.instances.clear()
        coordinates = benchmark._fit_umap(
            np.ones((6, 3)),
            {"n_components": 2, "n_neighbors": 3, "min_dist": 0.1},
            seed=43,
            umap_class=FakeUMAP,
        )
        self.assertEqual(coordinates.shape, (6, 2))
        self.assertEqual(FakeUMAP.instances[0].kwargs["metric"], "euclidean")
        self.assertEqual(FakeUMAP.instances[0].kwargs["init"], "random")
        self.assertEqual(FakeUMAP.instances[0].kwargs["n_jobs"], 1)

    def test_hierarchy_metrics_cover_leaf_parent_and_top(self) -> None:
        metadata = pd.DataFrame(
            {
                "class": ["A", "A", "B", "B", "C", "C"],
                "class_hierarchy": [
                    ["T1", "P1", "A"],
                    ["T1", "P1", "A"],
                    ["T1", "P2", "B"],
                    ["T1", "P2", "B"],
                    ["T2", "P3", "C"],
                    ["T2", "P3", "C"],
                ],
            }
        )
        metrics = benchmark._cluster_quality_metrics(
            np.arange(1, 13, dtype=np.float64).reshape(6, 2),
            np.array([0, 0, 1, 1, 2, -1]),
            np.ones(6),
            np.zeros(6),
            metadata,
        )
        self.assertIn("leaf_nmi", metrics)
        self.assertIn("parent_nmi", metrics)
        self.assertIn("top_nmi", metrics)
        self.assertGreaterEqual(metrics["hierarchy_distance"], 0.0)
        self.assertLessEqual(metrics["hierarchy_distance"], 3.0)

    def test_pairwise_stability_reports_cluster_and_neighbor_overlap(self) -> None:
        configuration = {
            "n_neighbors": 15,
            "n_components": 20,
            "min_dist": 0.1,
            "roles": ["production_baseline"],
        }
        labels = {
            42: np.array([0, 0, 1, 1]),
            43: np.array([0, 0, 1, 1]),
            44: np.array([0, 1, 1, 1]),
        }
        neighbors = {
            seed: {2: np.array([[1, 2], [0, 2], [1, 3], [2, 1]])}
            for seed in labels
        }
        records = benchmark._pairwise_seed_stability(
            configuration,
            labels,
            neighbors,
            reference_k=2,
        )
        self.assertEqual(len(records), 3)
        self.assertIn("cluster_ari", records[0])
        self.assertIn("umap_neighbor_reproducibility_k2", records[0])


if __name__ == "__main__":
    unittest.main()
