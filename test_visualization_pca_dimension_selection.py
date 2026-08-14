from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pca_projection import fit_normalized_pca_projection
from umap_projection import fit_projection_model, transform_projection
from visualization_pca_dimension_selection import (
    select_visualization_pca_dimension,
    transform_with_selected_visualization,
)


class _FakeUmap:
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        del y
        self.offset = np.mean(X[:, :2], axis=0)
        return X[:, :2] - self.offset

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, :2] - self.offset


def _fake_umap_factory(**kwargs: object) -> _FakeUmap:
    del kwargs
    return _FakeUmap()


class VisualizationPcaDimensionSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(19)
        latent = rng.normal(size=(80, 6))
        projection = rng.normal(size=(6, 48))
        self.X = latent @ projection + 0.02 * rng.normal(size=(80, 48))

    def test_defaults_start_at_16_and_increase_by_16(self) -> None:
        selection = select_visualization_pca_dimension(
            self.X,
            k_values=(3,),
            minimum_preservation_gain=1.0,
            n_neighbors=5,
            cluster_target_weight=0.0,
            umap_factory=_fake_umap_factory,
        )
        self.assertEqual(selection.selected_dimension, 16)
        self.assertEqual(
            [candidate.dimension for candidate in selection.candidates],
            [16, 32],
        )

    def test_selects_previous_dimension_when_umap_gain_plateaus(self) -> None:
        selection = select_visualization_pca_dimension(
            self.X,
            max_components=32,
            min_components=8,
            component_step=8,
            k_values=(3, 5),
            minimum_preservation_gain=0.05,
            n_neighbors=5,
            cluster_target_weight=0.0,
            umap_factory=_fake_umap_factory,
        )

        self.assertEqual(selection.selected_dimension, 8)
        self.assertEqual(
            selection.selection_reason,
            "first_below_minimum_gain_use_previous_dimension",
        )
        self.assertEqual(
            [candidate.dimension for candidate in selection.candidates],
            [8, 16],
        )
        self.assertEqual(selection.selected_coordinates.shape, (80, 2))

    def test_uses_maximum_when_every_gain_meets_minimum(self) -> None:
        selection = select_visualization_pca_dimension(
            self.X,
            max_components=24,
            min_components=8,
            component_step=8,
            k_values=(3,),
            minimum_preservation_gain=0.0,
            n_neighbors=5,
            cluster_target_weight=0.0,
            umap_factory=_fake_umap_factory,
        )
        self.assertEqual(selection.selected_dimension, 24)
        self.assertEqual(
            selection.selection_reason,
            "all_gains_meet_minimum_use_maximum_dimension",
        )

    def test_transform_reuses_selected_models(self) -> None:
        selection = select_visualization_pca_dimension(
            self.X,
            max_components=24,
            min_components=8,
            component_step=8,
            k_values=(3,),
            minimum_preservation_gain=1.0,
            n_neighbors=5,
            cluster_target_weight=0.0,
            umap_factory=_fake_umap_factory,
        )
        transformed = transform_with_selected_visualization(self.X, selection)
        np.testing.assert_allclose(
            transformed,
            selection.selected_coordinates,
            atol=1e-12,
        )

    def test_projection_model_uses_selected_prefix_for_fit_and_transform(self) -> None:
        fitted = fit_normalized_pca_projection(
            self.X,
            n_components=24,
            seed=42,
        )
        fake_umap = _FakeUmap()
        expected_coordinates = fake_umap.fit_transform(fitted.projected[:, :8])
        fake_selection = SimpleNamespace(
            pca=fitted.pca,
            selected_dimension=8,
            umap=fake_umap,
            selected_coordinates=expected_coordinates,
        )

        with patch(
            "visualization_pca_dimension_selection."
            "select_visualization_pca_dimension_for_data",
            return_value=fake_selection,
        ):
            pca, reducer, coordinates = fit_projection_model(
                self.X,
                seed=42,
                pca_components=None,
                n_neighbors=5,
                min_dist=0.02,
                metric="cosine",
                spread=0.85,
                densmap=False,
                cluster_target_weight=0.0,
            )

        self.assertEqual(pca.n_components_, 8)
        self.assertIs(reducer, fake_umap)
        np.testing.assert_allclose(coordinates, expected_coordinates)
        transformed = transform_projection(
            self.X[:5],
            pca=pca,
            reducer=reducer,
        )
        np.testing.assert_allclose(
            transformed,
            fake_umap.transform(fitted.projected[:5, :8]),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
