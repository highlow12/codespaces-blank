from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import normalize

from pca_dimension_selection import (
    select_pca_dimension,
    transform_with_selected_dimension,
)
from pca_dimension_search import GLOBAL_KNEE_REASON


class PcaDimensionSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        latent = rng.normal(size=(80, 6))
        projection = rng.normal(size=(6, 48))
        self.X = latent @ projection + 0.02 * rng.normal(size=(80, 48))

    def test_fits_once_and_evaluates_prefix_dimensions(self) -> None:
        selection = select_pca_dimension(
            self.X,
            max_components=32,
            min_components=8,
            component_step=8,
            k_values=(3, 5),
            minimum_preservation_gain=0.0,
        )

        self.assertEqual(selection.fitted_dimension, 32)
        self.assertEqual(
            [candidate.dimension for candidate in selection.candidates],
            [8, 16, 24, 32],
        )
        self.assertEqual(selection.selected_dimension, 32)
        self.assertEqual(
            selection.selection_reason,
            "all_gains_meet_minimum_use_maximum_dimension",
        )
        explained_variances = [
            candidate.cumulative_explained_variance
            for candidate in selection.candidates
        ]
        self.assertTrue(
            all(
                current >= previous
                for previous, current in zip(
                    explained_variances,
                    explained_variances[1:],
                )
            )
        )
        for candidate in selection.candidates:
            self.assertGreaterEqual(candidate.mean_knn_preservation, 0.0)
            self.assertLessEqual(candidate.mean_knn_preservation, 1.0)

        maximum_projection = selection.pca.transform(normalize(self.X, norm="l2"))
        expected = normalize(maximum_projection[:, :32], norm="l2")
        np.testing.assert_allclose(selection.selected_features, expected)

    def test_selects_previous_dimension_at_first_below_minimum_gain(self) -> None:
        selection = select_pca_dimension(
            self.X,
            max_components=32,
            min_components=8,
            component_step=8,
            k_values=(3,),
            minimum_preservation_gain=1.0,
        )
        self.assertEqual(selection.selected_dimension, 8)
        self.assertEqual(
            selection.selection_reason,
            "first_below_minimum_gain_use_previous_dimension",
        )

    def test_transform_uses_selected_prefix(self) -> None:
        selection = select_pca_dimension(
            self.X,
            max_components=24,
            min_components=8,
            component_step=8,
            k_values=(3,),
            minimum_preservation_gain=0.0,
        )
        transformed = transform_with_selected_dimension(self.X[:5], selection)
        np.testing.assert_allclose(
            transformed,
            selection.selected_features[:5],
            atol=1e-12,
        )

    def test_global_knee_recovers_from_noisy_early_plateau(self) -> None:
        with patch(
            "pca_dimension_search.mean_neighbor_preservation",
            side_effect=[0.50, 0.54, 0.70, 0.78],
        ):
            selection = select_pca_dimension(
                self.X,
                max_components=32,
                min_components=8,
                component_step=8,
                k_values=(3,),
                minimum_preservation_gain=0.05,
            )

        self.assertEqual(selection.selected_dimension, 24)
        self.assertEqual(selection.selection_reason, GLOBAL_KNEE_REASON)

    def test_rejects_inputs_that_cannot_reach_minimum_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "minimum PCA dimension"):
            select_pca_dimension(
                np.ones((10, 5)),
                max_components=32,
                min_components=8,
                component_step=8,
                k_values=(2,),
            )


if __name__ == "__main__":
    unittest.main()
