import unittest

import numpy as np

from hdbscan_bottom_up import (
    build_hdbscan_hierarchy,
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

    def test_hierarchy_artifact_handles_zero_discovered_leaves(self) -> None:
        result = build_hdbscan_hierarchy(
            np.eye(3),
            np.full(3, -1),
            np.zeros((3, 0)),
        )

        self.assertEqual(result.summary["leaf_cluster_count"], 0)
        self.assertEqual(result.summary["merge_count"], 0)
        self.assertEqual(result.tree["merges"], [])
        self.assertNotIn("bottom_up_k1", result.assignments)

    def test_hierarchy_artifact_handles_single_leaf_without_merge(self) -> None:
        result = build_hdbscan_hierarchy(
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            np.asarray([0, 0, -1]),
            np.asarray([[1.0], [0.8], [0.2]]),
        )

        self.assertEqual(result.summary["leaf_cluster_count"], 1)
        self.assertEqual(result.summary["merge_count"], 0)
        np.testing.assert_array_equal(
            result.assignments["bottom_up_k1"].to_numpy(),
            [0, 0, -1],
        )

    def test_linkage_accepts_degenerate_leaf_sets(self) -> None:
        self.assertEqual(weighted_average_linkage(np.empty((0, 2)), np.empty(0)), [])
        self.assertEqual(weighted_average_linkage(np.asarray([[1.0, 0.0]]), np.ones(1)), [])


if __name__ == "__main__":
    unittest.main()
