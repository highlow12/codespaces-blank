from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from clustering_types import FCMResult
from fcm_hierarchy import (
    modified_partition_coefficient,
    normalized_partition_entropy,
    partition_coefficient,
    partition_entropy,
    run_hierarchical_pca_fcm,
    select_fcm_cluster_count,
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
        self.assertEqual(best.n_clusters, 6)
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


if __name__ == "__main__":
    unittest.main()
