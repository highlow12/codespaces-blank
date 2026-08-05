from __future__ import annotations

import unittest

import numpy as np

from fcm_core import sfcm_memberships_from_centers, spherical_fcm
from fuzzy_cmeans import FuzzyCMeans, SphericalGeometry


class SphericalFuzzyCMeansTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.02, 0.0],
                [0.0, 1.0, 0.0],
                [0.02, 0.98, 0.0],
            ]
        )

    def test_spherical_fit_keeps_centers_on_the_unit_sphere(self) -> None:
        result = spherical_fcm(self.features, n_clusters=2, seed=7)

        np.testing.assert_allclose(
            np.linalg.norm(result.centers, axis=1),
            np.ones(2),
        )
        np.testing.assert_allclose(
            result.memberships.sum(axis=1),
            np.ones(len(self.features)),
        )
        self.assertEqual(set(result.labels), {0, 1})

    def test_fixed_center_memberships_use_exact_spherical_matches(self) -> None:
        memberships, distances = sfcm_memberships_from_centers(
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        )

        np.testing.assert_allclose(memberships, np.eye(2))
        np.testing.assert_allclose(distances.diagonal(), np.zeros(2))

    def test_tied_exact_centers_share_membership_equally(self) -> None:
        memberships, _ = sfcm_memberships_from_centers(
            np.asarray([[1.0, 0.0]]),
            np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        )

        np.testing.assert_allclose(memberships, np.asarray([[0.5, 0.5]]))

    def test_zero_length_sample_is_not_a_point_on_the_sphere(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-length"):
            spherical_fcm(np.asarray([[0.0, 0.0], [1.0, 0.0]]), 2)

    def test_generic_optimizer_accepts_the_spherical_geometry_strategy(self) -> None:
        result = FuzzyCMeans(
            geometry=SphericalGeometry(),
            seed=7,
        ).fit(self.features, n_clusters=2)

        np.testing.assert_allclose(
            np.linalg.norm(result.centers, axis=1),
            np.ones(2),
        )


if __name__ == "__main__":
    unittest.main()
