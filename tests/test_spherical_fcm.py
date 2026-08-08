from __future__ import annotations

import unittest

import numpy as np

from fcm_core import (
    _memberships_from_distances,
    sfcm_memberships_from_centers,
    spherical_fcm,
)
from fuzzy_cmeans import (
    FuzzyCMeans,
    SphericalGeometry,
    memberships_from_squared_dissimilarities,
)


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

    def test_spherical_fit_retains_final_squared_dissimilarities(self) -> None:
        result = spherical_fcm(self.features, n_clusters=2, seed=7)

        self.assertIsNotNone(result.squared_dissimilarities)
        geometry = SphericalGeometry()
        expected = geometry.squared_dissimilarities(
            geometry.prepare_samples(self.features),
            geometry.prepare_samples(result.centers),
        )
        np.testing.assert_allclose(result.squared_dissimilarities, expected)

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

    def test_linear_membership_formula_matches_pairwise_distance_ratios(self) -> None:
        rng = np.random.default_rng(2026)
        distances = rng.uniform(0.01, 2.0, size=(64, 5))

        for fuzzifier in (1.4, 2.0, 2.5):
            exponent = 2.0 / (fuzzifier - 1.0)
            ratios = (
                distances[:, :, None] / distances[:, None, :]
            ) ** exponent
            expected = 1.0 / ratios.sum(axis=2)

            actual = _memberships_from_distances(
                distances,
                m=fuzzifier,
            )

            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_linear_squared_formula_matches_pairwise_ratios(self) -> None:
        rng = np.random.default_rng(2027)
        squared_dissimilarities = rng.uniform(0.001, 4.0, size=(64, 5))

        for fuzzifier in (1.4, 2.0, 2.5):
            exponent = 1.0 / (fuzzifier - 1.0)
            ratios = (
                squared_dissimilarities[:, :, None]
                / squared_dissimilarities[:, None, :]
            ) ** exponent
            expected = 1.0 / ratios.sum(axis=2)

            actual = memberships_from_squared_dissimilarities(
                squared_dissimilarities,
                m=fuzzifier,
            )

            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

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
