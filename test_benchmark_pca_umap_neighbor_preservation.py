"""Focused tests for the PCA/UMAP neighbourhood benchmark."""

from __future__ import annotations

import unittest

import numpy as np

import benchmark_pca_umap_neighbor_preservation as benchmark


class FakeUMAP:
    instances: list["FakeUMAP"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.embedding_: np.ndarray | None = None
        FakeUMAP.instances.append(self)

    def fit(self, values: np.ndarray) -> "FakeUMAP":
        components = int(self.kwargs["n_components"])
        # A deterministic projection double is sufficient to exercise the
        # benchmark orchestration without importing umap-learn.
        self.embedding_ = np.asarray(values[:, :components], dtype=np.float64)
        return self


class BenchmarkHelpersTest(unittest.TestCase):
    def test_valid_k_values_filters_only_values_too_large_for_n(self) -> None:
        self.assertEqual(benchmark.valid_k_values((5, 2, 5, 20), 10), (2, 5))

    def test_effective_grid_clamps_neighbors_and_filters_pca_width(self) -> None:
        grid = benchmark.effective_umap_grid(
            n_samples=8,
            pca_width=5,
            n_neighbors=(5, 100),
            n_components=(2, 8),
            min_dists=(0.1,),
        )
        self.assertEqual(len(grid), 2)
        self.assertEqual({row["n_neighbors"] for row in grid}, {5, 7})
        self.assertEqual({row["n_components"] for row in grid}, {2})
        clamped = next(row for row in grid if row["requested_n_neighbors"] == 100)
        self.assertEqual(clamped["n_neighbors"], 7)
        self.assertNotIn(8, {row["requested_n_components"] for row in grid})

    def test_preservation_metrics_include_both_paths_and_loss(self) -> None:
        reference = np.array([[1, 2], [0, 2], [0, 1]])
        candidate = np.array([[1, 2], [0, 2], [0, 1]])
        changed = np.array([[3, 0], [3, 1], [3, 2]])
        metrics = benchmark.preservation_metrics(
            {2: reference}, {2: candidate}, {2: changed}
        )
        self.assertEqual(metrics["k2"]["raw_pca"], 1.0)
        self.assertEqual(metrics["k2"]["pca_umap"], 0.0)
        self.assertEqual(metrics["k2"]["raw_umap"], 0.0)
        self.assertEqual(metrics["k2"]["umap_additional_loss"], 1.0)

    def test_quick_library_smoke_reuses_fixed_pca_and_runs_seed_comparison(self) -> None:
        FakeUMAP.instances.clear()
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(12, 6)).astype(np.float32)
        report = benchmark.run_experiment(
            embeddings,
            n_neighbors=(3,),
            n_components=(2,),
            min_dists=(0.1,),
            umap_seeds=(42, 43),
            k_values=(2, 3, 5),
            pca_max_components=4,
            pca_min_components=2,
            pca_component_step=2,
            umap_class=FakeUMAP,
        )
        self.assertEqual(len(report["runs"]), 2)
        self.assertEqual(len(report["reproducibility_comparisons"]), 1)
        self.assertEqual(report["pca_selection"]["configuration"]["input_normalized"], True)
        self.assertEqual(
            report["protocol"]["pca_selection_k_values"],
            [15, 30],
        )
        self.assertTrue(all(row.kwargs["n_jobs"] == 1 for row in FakeUMAP.instances))
        self.assertTrue(all(row.kwargs["metric"] == "euclidean" for row in FakeUMAP.instances))
        self.assertTrue(all("raw_umap_k5" in row for row in report["runs"]))
        self.assertIn("reproducibility_k5_mean", report["runs"][1])
        self.assertIn("mean_raw_umap", report["summary"][0])
        self.assertIn("mean_pca_umap", report["summary"][0])


if __name__ == "__main__":
    unittest.main()
