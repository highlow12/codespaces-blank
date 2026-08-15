import unittest

import numpy as np

from hdbscan_bottom_up import (
    cut_tree,
    lift_leaf_labels,
    soft_leaf_centers,
    weighted_average_linkage,
)


class BottomUpHierarchyTest(unittest.TestCase):
    def test_closest_centers_merge_first_and_cut_is_exact(self) -> None:
        centers = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ]
        )
        merges = weighted_average_linkage(centers, np.ones(4))

        self.assertEqual(set((merges[0].left, merges[0].right)), {0, 1})
        mapping = cut_tree(4, merges, 2)
        self.assertEqual(mapping[0], mapping[1])
        self.assertEqual(mapping[2], mapping[3])
        self.assertNotEqual(mapping[0], mapping[2])

    def test_soft_centers_and_noise_preserving_lift(self) -> None:
        features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        memberships = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        centers, masses = soft_leaf_centers(features, memberships)

        np.testing.assert_allclose(masses, [1.5, 1.5])
        np.testing.assert_allclose(np.linalg.norm(centers, axis=1), [1.0, 1.0])
        lifted = lift_leaf_labels(
            np.asarray([0, 1, -1]),
            np.asarray([1, 0]),
        )
        np.testing.assert_array_equal(lifted, [1, 0, -1])


if __name__ == "__main__":
    unittest.main()
