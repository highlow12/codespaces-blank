from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from clustering_types import FCMResult
from clustering_pipelines import run_pipeline_by_name
from fcm_core import spherical_fcm
from fcm_hierarchy import (
    modified_partition_coefficient,
    normalized_partition_entropy,
    partition_coefficient,
    partition_entropy,
    run_hierarchical_pca_fcm,
    select_fcm_cluster_count,
)
from hierarchical_fcm import _sqrt_selected_squared_distances
from fast_fcm import (
    FastFcmConfig,
    _scout_candidate_ks,
    select_fast_fcm_cluster_count,
)


class XieBeniClusterSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.tile(np.eye(3, dtype=np.float64), (8, 1))

    @staticmethod
    def _fcm_result(
        X: np.ndarray,
        n_clusters: int,
        **_: object,
    ) -> FCMResult:
        labels = np.arange(X.shape[0]) % n_clusters
        memberships = np.zeros((X.shape[0], n_clusters), dtype=np.float64)
        memberships[np.arange(X.shape[0]), labels] = 1.0
        centers = np.ones((n_clusters, X.shape[1]), dtype=np.float64)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        return FCMResult(labels, memberships, centers, iterations=1)

    @staticmethod
    def _filtered_labels(
        _X: np.ndarray,
        result: FCMResult,
        **_: object,
    ) -> tuple[np.ndarray, list[int]]:
        labels = result.labels.copy()
        sizes = [int(np.sum(labels == k)) for k in range(result.centers.shape[0])]
        return labels, sizes

    def _select(
        self,
        xb_values: list[float],
        *,
        threshold: float = 0.05,
        selection_method: str = "xie_beni",
        patience: int = 2,
        silhouette_values: list[float] | None = None,
    ):
        with (
            patch("fcm_validity.spherical_fcm", side_effect=self._fcm_result),
            patch("fcm_validity._filter_fcm_labels", side_effect=self._filtered_labels),
            patch(
                "fcm_validity.silhouette_score",
                side_effect=silhouette_values,
                return_value=0.5,
            ),
            patch("fcm_validity.xie_beni_index", side_effect=xb_values),
            patch("fcm_validity.spherical_fcm_objective", return_value=1.0),
        ):
            return select_fcm_cluster_count(
                self.features,
                min_clusters=2,
                max_clusters=len(xb_values) + 1,
                min_child_size=4,
                selection_method=selection_method,
                min_xb_relative_improvement=threshold,
                xb_worsening_patience=patience,
            )

    def test_stops_at_first_small_relative_improvement(self) -> None:
        best, metrics, reason = self._select([0.50, 0.30, 0.29, 0.10])

        self.assertEqual(reason, "selected_xb_relative_improvement")
        self.assertIsNotNone(best)
        self.assertEqual(best.n_clusters, 3)
        self.assertEqual([metric["k"] for metric in metrics], [2, 3, 4])
        self.assertIsNone(metrics[0]["xb_relative_improvement"])
        self.assertAlmostEqual(metrics[1]["xb_relative_improvement"], 0.40)
        self.assertAlmostEqual(
            metrics[2]["xb_relative_improvement"],
            (0.30 - 0.29) / 0.30,
        )

    def test_uses_global_xb_minimum_when_improvement_stays_large(self) -> None:
        best, metrics, reason = self._select([0.50, 0.40, 0.30, 0.20])

        self.assertEqual(reason, "selected_xb_minimum")
        self.assertIsNotNone(best)
        self.assertEqual(best.n_clusters, 5)
        self.assertEqual(len(metrics), 4)

    def test_rejects_invalid_relative_improvement_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be between 0 and 1"):
            select_fcm_cluster_count(
                self.features,
                min_child_size=4,
                selection_method="xie_beni",
                min_xb_relative_improvement=1.01,
            )

    def test_partition_validity_metrics(self) -> None:
        memberships = np.asarray([[1.0, 0.0], [0.5, 0.5]])
        result = FCMResult(
            labels=np.asarray([0, 0]),
            memberships=memberships,
            centers=np.eye(2),
            iterations=1,
        )

        self.assertAlmostEqual(partition_coefficient(result), 0.75)
        self.assertAlmostEqual(modified_partition_coefficient(result), 0.50)
        self.assertAlmostEqual(partition_entropy(result), np.log(2.0) / 2.0)
        self.assertAlmostEqual(normalized_partition_entropy(result), 0.50)

    def test_multistart_retries_collapse_and_selects_best_valid_objective(
        self,
    ) -> None:
        X = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        )
        collapsed = FCMResult(
            labels=np.zeros(6, dtype=int),
            memberships=np.tile([0.5, 0.5], (6, 1)),
            centers=np.tile([1.0, 0.0], (2, 1)),
            iterations=2,
            objective=0.01,
            minimum_center_distance=0.0,
        )
        valid_labels = np.asarray([0, 0, 0, 1, 1, 1])
        valid_memberships = np.eye(2)[valid_labels]
        valid_slow = FCMResult(
            labels=valid_labels,
            memberships=valid_memberships,
            centers=np.eye(2),
            iterations=4,
            objective=0.50,
            minimum_center_distance=np.sqrt(2.0),
        )
        valid_best = FCMResult(
            labels=1 - valid_labels,
            memberships=np.eye(2)[1 - valid_labels],
            centers=np.flipud(np.eye(2)),
            iterations=3,
            objective=0.30,
            minimum_center_distance=np.sqrt(2.0),
        )

        with patch(
            "fcm_core._spherical_fcm_once",
            side_effect=[collapsed, valid_slow, valid_best],
        ):
            result = spherical_fcm(
                X,
                n_clusters=2,
                n_init=2,
                max_attempts=3,
                min_cluster_size=2,
                min_center_separation=0.01,
            )

        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.valid_restarts, 2)
        self.assertAlmostEqual(result.objective, 0.30)
        self.assertAlmostEqual(result.restart_stability, 1.0)
        np.testing.assert_array_equal(result.labels, valid_best.labels)

    def test_multi_metric_checks_two_more_k_values_after_xb_worsens(self) -> None:
        best, metrics, reason = self._select(
            [0.50, 0.30, 0.31, 0.25, 0.20, 0.10],
            selection_method="multi_metric",
            silhouette_values=[0.99, 0.50, 0.20, 0.00, -0.90],
        )

        self.assertEqual(
            reason,
            "selected_multi_metric_xb_worsening_patience",
        )
        self.assertIsNotNone(best)
        self.assertEqual(best.n_clusters, 3)
        self.assertEqual(
            [metric["k"] for metric in metrics],
            [2, 3, 4, 5, 6],
        )
        for metric in metrics:
            self.assertIsNotNone(metric["partition_coefficient"])
            self.assertIsNotNone(metric["partition_entropy"])
            self.assertIsNotNone(metric["selection_score"])

    def test_rejects_negative_xb_worsening_patience(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            select_fcm_cluster_count(
                self.features,
                min_child_size=4,
                selection_method="multi_metric",
                xb_worsening_patience=-1,
            )

    def test_rank_scoring_is_not_compressed_by_extreme_xb(self) -> None:
        best, metrics, reason = self._select(
            [0.237, 0.221, 1.11, 1.5e15, 1.75e14],
            selection_method="multi_metric",
        )

        self.assertEqual(
            reason,
            "selected_multi_metric_xb_worsening_patience",
        )
        self.assertIsNotNone(best)
        self.assertEqual(best.n_clusters, 3)
        scores = {metric["k"]: metric["selection_score"] for metric in metrics}
        self.assertGreater(scores[3], scores[2])

    def test_hierarchical_multi_metric_path_runs(self) -> None:
        result = run_hierarchical_pca_fcm(
            self.features,
            max_depth=1,
            min_node_size=8,
            min_child_size=4,
            min_clusters=2,
            max_clusters=2,
            min_membership=0.0,
            max_membership_gap=0.0,
            forced_noise_ratio=0.0,
            selection_method="multi_metric",
            min_split_silhouette=-1.0,
            pca_components=3,
            seed=42,
        )

        self.assertIsNotNone(result.model)
        self.assertEqual(len(result.assignments), len(self.features))

    def test_hierarchy_reuses_selected_distance_artifact(self) -> None:
        with patch(
            "hierarchical_fcm.sfcm_memberships_from_centers",
            side_effect=AssertionError("selected result distances should be reused"),
        ):
            result = run_hierarchical_pca_fcm(
                self.features,
                max_depth=1,
                min_node_size=8,
                min_child_size=4,
                min_clusters=2,
                max_clusters=2,
                min_membership=0.0,
                max_membership_gap=0.0,
                forced_noise_ratio=0.0,
                selection_method="multi_metric",
                min_split_silhouette=-1.0,
                pca_components=3,
                seed=42,
            )

        self.assertEqual(len(result.assignments), len(self.features))

    def test_hierarchy_extracts_only_selected_squared_distances(self) -> None:
        squared = np.arange(20, dtype=np.float64).reshape(5, 4)
        rows = np.arange(5)
        labels = np.asarray([3, 1, 0, 2, 1])
        cluster_mask = np.asarray([True, False, True, False, True])
        original_sqrt = np.sqrt
        sqrt_input_shapes: list[tuple[int, ...]] = []

        def capture_sqrt(values: np.ndarray) -> np.ndarray:
            sqrt_input_shapes.append(np.asarray(values).shape)
            return original_sqrt(values)

        with patch("hierarchical_fcm.np.sqrt", side_effect=capture_sqrt):
            assigned = _sqrt_selected_squared_distances(squared, rows, labels)
            source_center = _sqrt_selected_squared_distances(
                squared,
                cluster_mask,
                2,
            )

        np.testing.assert_allclose(assigned, np.sqrt(squared[rows, labels]))
        np.testing.assert_allclose(
            source_center,
            np.sqrt(squared[cluster_mask, 2]),
        )
        self.assertEqual(sqrt_input_shapes, [(5,), (3,)])

    def test_selector_passes_distance_artifact_to_xie_beni(self) -> None:
        observed: list[np.ndarray | None] = []

        def capture_artifact(
            _X: np.ndarray,
            _result: FCMResult,
            *,
            squared_dissimilarities: np.ndarray | None = None,
        ) -> float:
            observed.append(squared_dissimilarities)
            return 0.1

        with patch(
            "fcm_validity.xie_beni_index",
            side_effect=capture_artifact,
        ):
            select_fcm_cluster_count(
                self.features,
                min_clusters=2,
                max_clusters=2,
                min_child_size=4,
                selection_method="multi_metric",
                seed=7,
            )

        self.assertEqual(len(observed), 1)
        self.assertIsNotNone(observed[0])

    def test_fast_selector_returns_full_data_candidate_and_adaptive_m(self) -> None:
        angles = np.repeat(
            np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]),
            24,
        )
        X = np.column_stack([np.cos(angles), np.sin(angles)])
        X += np.random.default_rng(7).normal(0.0, 0.02, size=X.shape)

        best, records, reason = select_fast_fcm_cluster_count(
            X,
            min_clusters=2,
            max_clusters=4,
            min_child_size=10,
            config=FastFcmConfig(
                sample_size=36,
                scout_n_init=2,
                scout_max_attempts=3,
                scout_max_iter=40,
                refine_n_init=2,
                refine_max_attempts=3,
                refine_max_iter=50,
                refine_top_k=1,
            ),
            seed=7,
        )

        self.assertEqual(reason, "selected_fast_scout_refine")
        self.assertIsNotNone(best)
        self.assertEqual(len(best.labels), len(X))
        self.assertEqual(best.labels.shape[0], best.result.memberships.shape[0])
        self.assertGreater(best.m, 1.0)
        self.assertTrue(any(record.get("phase") == "refine" for record in records))

    def test_fast_selector_scouts_full_k_range_and_refines_top_two(self) -> None:
        angles = np.repeat(np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False), 20)
        X = np.column_stack([np.cos(angles), np.sin(angles)])
        X += np.random.default_rng(11).normal(0.0, 0.01, size=X.shape)

        best, records, reason = select_fast_fcm_cluster_count(
            X,
            min_clusters=2,
            max_clusters=6,
            min_child_size=8,
            config=FastFcmConfig(
                sample_size=120,
                scout_n_init=2,
                scout_max_attempts=3,
                scout_max_iter=40,
                scout_max_clusters=6,
                refine_n_init=2,
                refine_max_attempts=3,
                refine_max_iter=50,
                refine_top_k=2,
                refine_score_margin=1.0,
            ),
            seed=11,
        )

        scout_ks = {
            int(record["k"])
            for record in records
            if record.get("phase") not in {"m_probe", "refine"}
        }
        refined_ks = {
            int(record["k"])
            for record in records
            if record.get("phase") == "refine"
        }
        self.assertEqual(reason, "selected_fast_scout_refine")
        self.assertIsNotNone(best)
        self.assertEqual(scout_ks, set(range(2, 7)))
        self.assertEqual(len(refined_ks), 2)
        self.assertTrue(
            all(
                record.get("silhouette_kind") == "center_distance_proxy"
                for record in records
                if record.get("phase") not in {"m_probe", "refine"}
            )
        )

    def test_fast_selector_refines_only_a_clear_scout_winner(self) -> None:
        ks, decision, score_gap = _scout_candidate_ks(
            [
                {"k": 2, "selection_score": 0.25, "valid_clusters": 2},
                {"k": 3, "selection_score": 0.80, "valid_clusters": 3},
                {"k": 4, "selection_score": 0.40, "valid_clusters": 4},
            ],
            refine_top_k=2,
            min_clusters=2,
            refine_score_margin=0.15,
        )

        self.assertEqual(ks, [3])
        self.assertEqual(decision, "single_clear_scout_winner")
        self.assertAlmostEqual(score_gap, 0.40)

    def test_center_distance_silhouette_proxy_avoids_pairwise_metric(self) -> None:
        with patch(
            "fcm_validity.silhouette_score",
            side_effect=AssertionError("scout proxy must not call silhouette_score"),
        ):
            best, _records, reason = select_fcm_cluster_count(
                self.features,
                min_clusters=2,
                max_clusters=2,
                min_child_size=4,
                selection_method="multi_metric",
                use_silhouette_proxy=True,
            )

        self.assertIsNotNone(best)
        self.assertTrue(np.isfinite(best.silhouette))
        self.assertIn(
            reason,
            {
                "selected_multi_metric",
                "selected_multi_metric_max_k",
                "selected_multi_metric_xb_worsening_patience",
            },
        )

    def test_default_hierarchical_path_records_automatic_pca_selection(self) -> None:
        result = run_hierarchical_pca_fcm(
            self.features,
            max_depth=1,
            min_node_size=8,
            min_child_size=4,
            min_clusters=2,
            max_clusters=2,
            min_membership=0.0,
            max_membership_gap=0.0,
            forced_noise_ratio=0.0,
            selection_method="multi_metric",
            min_split_silhouette=-1.0,
            seed=42,
        )

        self.assertIsNotNone(result.model)
        self.assertEqual(result.summary["pca_components"], 3)
        self.assertEqual(
            result.tree["config"]["pca_components_requested"],
            "auto",
        )
        self.assertEqual(
            result.tree["config"]["pca_components_selected"],
            3,
        )
        self.assertIsNotNone(result.tree["config"]["pca_selection"])

    def test_default_flat_pipeline_uses_automatic_pca_selection(self) -> None:
        result = run_pipeline_by_name(
            "2_auto_pca_fcm",
            self.features,
            np.arange(len(self.features)) % 3,
            3,
        )

        self.assertEqual(result.metrics["pca_components_requested"], "auto")
        self.assertEqual(result.metrics["pca_components"], 3)
        self.assertIsNotNone(result.metrics["pca_selection"])


if __name__ == "__main__":
    unittest.main()
