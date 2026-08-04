from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from clustering_types import FCMResult
from fcm_hierarchy import select_fcm_cluster_count


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
    ):
        with (
            patch("fcm_hierarchy.spherical_fcm", side_effect=self._fcm_result),
            patch("fcm_hierarchy._filter_fcm_labels", side_effect=self._filtered_labels),
            patch("fcm_hierarchy.silhouette_score", return_value=0.5),
            patch("fcm_hierarchy.xie_beni_index", side_effect=xb_values),
            patch("fcm_hierarchy.spherical_fcm_objective", return_value=1.0),
        ):
            return select_fcm_cluster_count(
                self.features,
                min_clusters=2,
                max_clusters=len(xb_values) + 1,
                min_child_size=4,
                selection_method="xie_beni",
                min_xb_relative_improvement=threshold,
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


if __name__ == "__main__":
    unittest.main()
