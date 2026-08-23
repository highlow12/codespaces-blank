import unittest
import inspect
from unittest.mock import patch

import numpy as np

from wikipedia_soft_benchmark.hierarchy_benchmark import (
    calibration_sweep,
    choose_calibration,
    build_parser,
    fit_discovery,
    predict_memberships,
)


class WikipediaHierarchyBenchmarkTests(unittest.TestCase):
    def test_exact_neighbor_backend_is_the_default(self):
        self.assertEqual(inspect.signature(fit_discovery).parameters["neighbor_backend"].default, "exact")
        self.assertEqual(inspect.signature(calibration_sweep).parameters["neighbor_backend"].default, "exact")

    def test_cli_parses_jobs(self):
        args = build_parser().parse_args(
            ["--embedding-dir", "embeddings", "--output-dir", "output", "--jobs", "3"]
        )
        self.assertEqual(args.jobs, 3)

    def test_cli_parses_uncentered_projection_mode(self):
        args = build_parser().parse_args(
            ["--embedding-dir", "embeddings", "--output-dir", "output", "--projection-mode", "uncentered-svd"]
        )
        self.assertEqual(args.projection_mode, "uncentered-svd")

    def test_projection_modes_have_expected_centering_semantics(self):
        rng = np.random.default_rng(81)
        embeddings = rng.random((30, 12), dtype=np.float32)
        rows = [{"split": "discovery", "leaf": str(i % 3), "parent": str(i % 2), "top": "t"} for i in range(30)]
        centered = fit_discovery(
            embeddings, rows, seed=4, min_cluster_size=3, min_samples=2,
            pca_components=6, umap_components=2, umap_n_neighbors=5,
            projection_mode="centered-pca",
        )
        uncentered = fit_discovery(
            embeddings, rows, seed=4, min_cluster_size=3, min_samples=2,
            pca_components=6, umap_components=2, umap_n_neighbors=5,
            projection_mode="uncentered-svd",
        )
        self.assertTrue(np.max(np.abs(np.mean(centered.pca_discovery, axis=0))) < 1e-6)
        self.assertGreater(float(np.max(np.abs(np.mean(uncentered.pca_discovery, axis=0)))), 1e-3)
        self.assertEqual(centered.configuration["projection_mode"], "centered-pca")
        self.assertEqual(uncentered.configuration["projection_mode"], "uncentered-svd")
        self.assertEqual(uncentered.pca_discovery.shape, centered.pca_discovery.shape)

    def test_calibration_reuses_fit_for_neighbor_counts(self):
        class FakeState:
            labels = np.array([0, 0, 1, 1])

        class FakeTransformer:
            def transform(self, values):
                return np.zeros((len(values), 2), dtype=np.float64)

        class FakeNeighborIndex:
            def query(self, values, count, *, exclude_self=False):
                return (
                    np.zeros((len(values), count), dtype=np.float64),
                    np.zeros((len(values), count), dtype=np.int64),
                )

        class FakePrepared:
            pca = FakeTransformer()
            umap = FakeTransformer()
            neighbor_index = FakeNeighborIndex()

        prepared = FakePrepared()

        discovery = np.ones((4, 3), dtype=np.float32)
        calibration = np.ones((2, 3), dtype=np.float32)
        metadata = [{"split": "discovery", "leaf": "a", "parent": "p", "top": "t"}] * 4
        calibration_metadata = [{"split": "calibration", "leaf": "a", "parent": "p", "top": "t"}] * 2

        def fake_evaluate(*args, **kwargs):
            return {"native": {"leaf_nmi": 0.5}, "exact_knn": {"leaf_nmi": 0.25}}

        with patch("wikipedia_soft_benchmark.hierarchy_benchmark._prepare_discovery_projection", return_value=prepared) as prepare, patch(
            "wikipedia_soft_benchmark.hierarchy_benchmark._state_from_prepared_projection", return_value=FakeState()
        ) as fit, patch(
            "wikipedia_soft_benchmark.hierarchy_benchmark.evaluate_split", side_effect=fake_evaluate
        ):
            rows, selected = calibration_sweep(
                discovery,
                metadata,
                calibration,
                calibration_metadata,
                seeds=(42,),
                min_cluster_sizes=(2,),
                min_samples_values=(1,),
                neighbor_counts=(1, 2, 3),
            )
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual([row["neighbor_count"] for row in rows], [1, 2, 3])
        self.assertEqual(selected["neighbor_count"], 1)

    def test_calibration_prepares_projection_once_for_all_hdbscan_combinations(self):
        class FakeState:
            labels = np.array([0, 0, 1, 1])

        class FakeTransformer:
            def transform(self, values):
                return np.zeros((len(values), 2), dtype=np.float64)

        class FakeNeighborIndex:
            def query(self, values, count, *, exclude_self=False):
                return (
                    np.zeros((len(values), count), dtype=np.float64),
                    np.zeros((len(values), count), dtype=np.int64),
                )

        class FakePrepared:
            pca = FakeTransformer()
            umap = FakeTransformer()
            neighbor_index = FakeNeighborIndex()

        discovery = np.ones((4, 3), dtype=np.float32)
        calibration = np.ones((2, 3), dtype=np.float32)
        metadata = [{"split": "discovery", "leaf": "a", "parent": "p", "top": "t"}] * 4
        calibration_metadata = [{"split": "calibration", "leaf": "a", "parent": "p", "top": "t"}] * 2

        def fake_evaluate(*args, **kwargs):
            return {"native": {"leaf_nmi": 0.5}, "exact_knn": {"leaf_nmi": 0.25}}

        with patch("wikipedia_soft_benchmark.hierarchy_benchmark._prepare_discovery_projection", return_value=FakePrepared()) as prepare, patch(
            "wikipedia_soft_benchmark.hierarchy_benchmark._state_from_prepared_projection", return_value=FakeState()
        ) as fit, patch(
            "wikipedia_soft_benchmark.hierarchy_benchmark.evaluate_split", side_effect=fake_evaluate
        ):
            rows, _ = calibration_sweep(
                discovery,
                metadata,
                calibration,
                calibration_metadata,
                seeds=(42,),
                min_cluster_sizes=(2, 3, 4),
                min_samples_values=(1, 2, 3),
                neighbor_counts=(1,),
            )

        self.assertEqual(len(rows), 9)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(fit.call_count, 9)

    def test_calibration_can_return_and_reuse_selected_state_and_projection(self):
        class FakeState:
            labels = np.array([0, 0, 1, 1])

        class FakeTransformer:
            def transform(self, values):
                return np.zeros((len(values), 2), dtype=np.float64)

        class FakeNeighborIndex:
            def query(self, values, count, *, exclude_self=False):
                return np.zeros((len(values), count)), np.zeros((len(values), count), dtype=np.int64)

        class FakePrepared:
            pca = FakeTransformer()
            umap = FakeTransformer()
            neighbor_index = FakeNeighborIndex()
            timing_sec = {"pca_fit_transform_sec": 1.0}

        prepared = FakePrepared()
        discovery = np.ones((4, 3), dtype=np.float32)
        calibration = np.ones((2, 3), dtype=np.float32)
        metadata = [{"split": "discovery", "leaf": "a", "parent": "p", "top": "t"}] * 4
        calibration_metadata = [{"split": "calibration", "leaf": "a", "parent": "p", "top": "t"}] * 2

        def fake_evaluate(*args, **kwargs):
            return {"native": {"leaf_nmi": 0.5}, "exact_knn": {"leaf_nmi": 0.25}}

        with patch("wikipedia_soft_benchmark.hierarchy_benchmark._prepare_discovery_projection", return_value=prepared), patch(
            "wikipedia_soft_benchmark.hierarchy_benchmark._state_from_prepared_projection", return_value=FakeState()
        ) as fit, patch("wikipedia_soft_benchmark.hierarchy_benchmark.evaluate_split", side_effect=fake_evaluate):
            result = calibration_sweep(
                discovery, metadata, calibration, calibration_metadata,
                seeds=(42,), min_cluster_sizes=(2,), min_samples_values=(1,),
                neighbor_counts=(1,), return_prepared=True,
            )
        rows, selected, artifacts = result
        self.assertIs(artifacts.prepared_projection, prepared)
        self.assertIs(artifacts.selected_state, fit.return_value)
        self.assertTrue(artifacts.timing_sec["pca_fit_transform_sec"] == 1.0)
        self.assertEqual(selected["seed"], 42)

    def test_discovery_fit_rejects_calibration_or_test_rows(self):
        rng = np.random.default_rng(8)
        rows = [{"split": "discovery", "leaf": "a", "parent": "p", "top": "t"} for _ in range(8)]
        rows[-1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "discovery rows only"):
            fit_discovery(rng.normal(size=(8, 4)), rows, min_cluster_size=2, min_samples=1, pca_components=2, umap_components=2, umap_n_neighbors=3)

    def test_out_of_sample_memberships_have_expected_shapes_and_bounds(self):
        rng = np.random.default_rng(3)
        rows = [{"split": "discovery", "leaf": str(i % 3), "parent": str(i % 2), "top": "t"} for i in range(24)]
        state = fit_discovery(rng.normal(size=(24, 8)), rows, min_cluster_size=3, min_samples=2, pca_components=4, umap_components=2, umap_n_neighbors=5)
        prediction = predict_memberships(state, rng.normal(size=(5, 8)), neighbor_count=4)
        self.assertEqual(prediction.native.shape, (5, state.cluster_count))
        self.assertEqual(prediction.exact_knn.shape, (5, state.cluster_count))
        self.assertTrue(np.all((prediction.native >= 0) & (prediction.native <= 1)))
        self.assertTrue(np.all((prediction.exact_knn >= 0) & (prediction.exact_knn <= 1)))
        self.assertTrue(np.all((prediction.native_unexplained >= 0) & (prediction.native_unexplained <= 1)))
        self.assertTrue(np.all((prediction.exact_unexplained >= 0) & (prediction.exact_unexplained <= 1)))

    def test_calibration_tie_break_is_deterministic(self):
        options = [
            {"mean_leaf_nmi": 0.5, "mean_noise_rate": 0.1, "complexity": 20, "sort_key": [43, 18, 3, 8]},
            {"mean_leaf_nmi": 0.5, "mean_noise_rate": 0.1, "complexity": 20, "sort_key": [42, 18, 3, 8]},
        ]
        self.assertEqual(choose_calibration(options)["sort_key"], [42, 18, 3, 8])


if __name__ == "__main__":
    unittest.main()
