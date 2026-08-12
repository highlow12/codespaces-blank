from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark_cluster_selection_sample_ratios import (
    PRIMARY_STRATEGY,
    _checkpoint_payload,
    _dataset_is_complete,
    _deserialize_full_results,
    _load_checkpoint,
    _row_key,
    _save_checkpoint,
    choose_dataset_indices,
    choose_nested_sample_indices,
    online_refine_sample_centers,
    sample_size_for_ratio,
    scaled_sample_min_child_size,
)


class ClusterSelectionSampleRatioTests(unittest.TestCase):
    def test_sample_size_rounds_and_keeps_full_dataset_exact(self) -> None:
        self.assertEqual(sample_size_for_ratio(100, 0.05), 5)
        self.assertEqual(sample_size_for_ratio(300, 0.15), 45)
        self.assertEqual(sample_size_for_ratio(1000, 0.333), 333)
        self.assertEqual(sample_size_for_ratio(3000, 1.0), 3000)

    def test_sample_size_rejects_invalid_ratio(self) -> None:
        with self.assertRaises(ValueError):
            sample_size_for_ratio(100, 0.0)
        with self.assertRaises(ValueError):
            sample_size_for_ratio(100, 1.1)

    def test_scaled_min_child_preserves_population_proportion(self) -> None:
        self.assertEqual(
            scaled_sample_min_child_size(
                20,
                total_size=3000,
                sample_size=300,
            ),
            2,
        )
        self.assertEqual(
            scaled_sample_min_child_size(
                20,
                total_size=3000,
                sample_size=1500,
            ),
            10,
        )
        self.assertEqual(
            scaled_sample_min_child_size(
                20,
                total_size=3000,
                sample_size=3000,
            ),
            20,
        )

    def test_dataset_indices_are_reproducible_and_unique(self) -> None:
        first = choose_dataset_indices(3000, 300, seed=42)
        second = choose_dataset_indices(3000, 300, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 300)
        self.assertTrue(np.all(np.diff(first) > 0))

    def test_nested_samples_retain_smaller_sample(self) -> None:
        small = choose_nested_sample_indices(1000, 100, seed=42)
        large = choose_nested_sample_indices(1000, 300, seed=42)
        np.testing.assert_array_equal(small, np.intersect1d(small, large))
        np.testing.assert_array_equal(
            choose_nested_sample_indices(1000, 1000, seed=42),
            np.arange(1000),
        )

    def test_checkpoint_round_trip_keeps_rows_and_compact_full_context(self) -> None:
        selected = SimpleNamespace(
            n_clusters=3,
            result=SimpleNamespace(labels=np.array([0, 1, 1, 2], dtype=int)),
        )
        payload = _checkpoint_payload(
            configuration={"experiment": "test"},
            rows=[{"dataset_size": 10, "status": "ok"}],
            pca_timings={"10": 0.5},
            dataset_records=[{"dataset_size": 10}],
            full_results_by_dataset={
                10: {42: (selected, "selected", 1.25)},
            },
            completed=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            _save_checkpoint(path, payload)
            restored = _load_checkpoint(
                path,
                expected_configuration={"experiment": "test"},
            )

        self.assertEqual(restored["rows"], payload["rows"])
        contexts = _deserialize_full_results(restored["full_results"])
        self.assertEqual(contexts[10][42][0].n_clusters, 3)
        np.testing.assert_array_equal(
            contexts[10][42][0].result.labels,
            np.array([0, 1, 1, 2], dtype=np.int32),
        )

    def test_checkpoint_rejects_different_configuration(self) -> None:
        payload = _checkpoint_payload(
            configuration={"seed": 42},
            rows=[],
            pca_timings={},
            dataset_records=[],
            full_results_by_dataset={},
            completed=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pkl"
            _save_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, "configuration"):
                _load_checkpoint(path, expected_configuration={"seed": 43})

    def test_dataset_is_incomplete_until_both_sample_strategies_are_saved(self) -> None:
        args = SimpleNamespace(
            seeds=(42,),
            sample_seeds=(7,),
            sample_ratios=(0.5,),
        )
        full_key = _row_key(
            dataset_size=10,
            sample_ratio=1.0,
            sample_seed=-1,
            selection_seed=42,
            strategy="full_selection",
        )
        primary_key = _row_key(
            dataset_size=10,
            sample_ratio=0.5,
            sample_seed=7,
            selection_seed=42,
            strategy=PRIMARY_STRATEGY,
        )
        project_key = _row_key(
            dataset_size=10,
            sample_ratio=0.5,
            sample_seed=7,
            selection_seed=42,
            strategy="sample_select_project",
        )
        self.assertFalse(_dataset_is_complete(10, args, {full_key, primary_key}))
        self.assertTrue(
            _dataset_is_complete(
                10,
                args,
                {full_key, primary_key, project_key},
            )
        )

    def test_online_refinement_updates_sample_centers_with_held_out_rows(self) -> None:
        features = np.array(
            [
                [1.0, 0.0],
                [0.98, 0.20],
                [0.0, 1.0],
                [0.20, 0.98],
            ]
        )
        selected = SimpleNamespace(
            centers=np.array([[1.0, 0.0], [0.0, 1.0]]),
            memberships=np.array([[1.0, 0.0], [0.0, 1.0]]),
            m=2.0,
            n_init=1,
            attempts=1,
            valid_restarts=1,
            restart_stability=1.0,
        )
        result = online_refine_sample_centers(
            features,
            np.array([0, 2]),
            selected,
            batch_size=1,
            order_seed=42,
        )
        self.assertEqual(result.labels.shape, (4,))
        np.testing.assert_allclose(
            np.linalg.norm(result.centers, axis=1),
            np.ones(2),
        )
        self.assertGreater(result.centers[0, 1], 0.0)
        self.assertGreater(result.centers[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
