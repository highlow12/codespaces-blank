from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from fcm_core import spherical_fcm
from fcm_geometry import ExperimentalFcmGeometry, geometry_fcm
from geometry_fcm_selection import select_geometry_fcm_cluster_count


class ExperimentalFcmGeometryTests(unittest.TestCase):
    def test_raw_cosine_distance_ignores_row_scale_but_center_update_does_not(self) -> None:
        geometry = ExperimentalFcmGeometry("cosine_raw")
        samples = np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        centers = np.asarray([[1.0, 0.5], [0.5, 1.0]])
        scaled = samples * np.asarray([[4.0], [0.5], [3.0]])
        np.testing.assert_allclose(
            geometry.squared_dissimilarities(samples, centers),
            geometry.squared_dissimilarities(scaled, centers),
        )
        memberships = np.asarray([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
        first = geometry.update_centers(samples, memberships, m=2.0)
        second = geometry.update_centers(scaled, memberships, m=2.0)
        self.assertFalse(
            np.allclose(normalize(first), normalize(second), atol=1e-8)
        )

    def test_euclidean_center_is_fuzzy_weighted_mean(self) -> None:
        geometry = ExperimentalFcmGeometry("euclidean")
        samples = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        memberships = np.asarray([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
        weights = memberships**2
        expected = (weights.T @ samples) / weights.sum(axis=0)[:, None]
        np.testing.assert_allclose(
            geometry.update_centers(samples, memberships, m=2.0),
            expected,
        )

    def test_normalized_cosine_matches_existing_sfcm_partition(self) -> None:
        rng = np.random.default_rng(7)
        samples = normalize(
            np.vstack(
                [
                    rng.normal(loc=(3.0, 0.0, 0.0), scale=0.25, size=(30, 3)),
                    rng.normal(loc=(0.0, 3.0, 0.0), scale=0.25, size=(30, 3)),
                    rng.normal(loc=(0.0, 0.0, 3.0), scale=0.25, size=(30, 3)),
                ]
            )
        )
        existing = spherical_fcm(samples, 3, seed=11, n_init=3, max_attempts=6)
        comparison = geometry_fcm(
            samples,
            3,
            geometry_name="cosine_normalized",
            seed=11,
            n_init=3,
            max_attempts=6,
        )
        self.assertAlmostEqual(
            adjusted_rand_score(existing.labels, comparison.labels), 1.0
        )
        self.assertAlmostEqual(existing.objective, comparison.objective, places=9)

    def test_geometry_auto_k_smoke(self) -> None:
        rng = np.random.default_rng(19)
        samples = np.vstack(
            [
                rng.normal(loc=(-2.0, 0.0), scale=0.15, size=(25, 2)),
                rng.normal(loc=(2.0, 0.0), scale=0.15, size=(25, 2)),
            ]
        )
        best, records, reason = select_geometry_fcm_cluster_count(
            samples,
            geometry_name="euclidean",
            min_clusters=2,
            max_clusters=3,
            min_child_size=5,
            seed=3,
            n_init=2,
            max_attempts=4,
        )
        self.assertIsNotNone(best)
        self.assertGreaterEqual(len(records), 2)
        self.assertTrue(reason.startswith("selected_multi_metric"))


if __name__ == "__main__":
    unittest.main()
