import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from hdbscan_membership_comparison import propagate_exact_knn_memberships
from pca_neighbor_search import build_pca_neighbor_index


class _FakeNNDescent:
    instances = []

    def __init__(self, data, **kwargs):
        self.data = np.asarray(data, dtype=float)
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def query(self, values, k=10, epsilon=0.1):
        values = np.asarray(values, dtype=float)
        distances = np.linalg.norm(values[:, None, :] - self.data[None, :, :], axis=2)
        # Deliberately return reverse order: the production wrapper must
        # provide sorted distances for both exact and ANN backends.
        indices = np.argsort(distances, axis=1)[:, ::-1][:, :k]
        selected = np.take_along_axis(distances, indices, axis=1)
        return indices, selected


class PcaNeighborSearchTests(unittest.TestCase):
    def test_real_ann_returns_indices_and_euclidean_distances_in_correct_order(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(40, 3))
        index = build_pca_neighbor_index(
            values, backend="pynndescent", max_neighbors=3
        )
        queries = values[:4] + 0.01
        distances, indices = index.query(queries, 3)
        self.assertTrue(np.issubdtype(indices.dtype, np.integer))
        self.assertTrue(np.all((indices >= 0) & (indices < len(values))))
        expected = np.linalg.norm(queries[:, None, :] - values[indices], axis=2)
        np.testing.assert_allclose(distances, expected, rtol=1e-5, atol=1e-6)

    def test_ann_shape_sorted_deterministic_and_self_excluded(self):
        fake_module = types.SimpleNamespace(NNDescent=_FakeNNDescent)
        values = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]])
        with patch.dict(sys.modules, {"pynndescent": fake_module}):
            first = build_pca_neighbor_index(values, backend="pynndescent", max_neighbors=2)
            second = build_pca_neighbor_index(values, backend="pynndescent", max_neighbors=2)
            distances, indices = first.query(values, 2, exclude_self=True)
            distances_again, indices_again = second.query(values, 2, exclude_self=True)
        self.assertEqual(distances.shape, (5, 2))
        self.assertEqual(indices.shape, (5, 2))
        self.assertTrue(np.all(np.diff(distances, axis=1) >= 0.0))
        for row, neighbors in enumerate(indices):
            self.assertNotIn(row, neighbors.tolist())
        np.testing.assert_allclose(distances, distances_again)
        np.testing.assert_array_equal(indices, indices_again)
        self.assertEqual(_FakeNNDescent.instances[-1].kwargs["n_jobs"], 1)
        self.assertEqual(_FakeNNDescent.instances[-1].kwargs["random_state"], 42)
        self.assertGreaterEqual(_FakeNNDescent.instances[-1].kwargs["n_neighbors"], min(32, len(values)))

    def test_ann_membership_preserves_noise_mass(self):
        fake_module = types.SimpleNamespace(NNDescent=_FakeNNDescent)
        values = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        labels = np.asarray([0, -1, 0, 0])
        probabilities = np.ones(4)
        with patch.dict(sys.modules, {"pynndescent": fake_module}):
            result = propagate_exact_knn_memberships(
                values, labels, probabilities, neighbor_count=2,
                neighbor_backend="pynndescent",
            )
        self.assertEqual(result.affinities.shape, (4, 1))
        self.assertTrue(np.all(result.unexplained >= 0.0))
        self.assertTrue(np.any(result.unexplained > 0.0))


if __name__ == "__main__":
    unittest.main()
