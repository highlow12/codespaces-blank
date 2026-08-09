from __future__ import annotations

import unittest

import pandas as pd

from visual_assignments import build_cluster_supervision, prepare_visual_assignments


def _assignments_with_path_memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "level_1_cluster": [0, 1],
            "level_2_cluster": [1, 0],
            "level_1_membership_0": [0.9, 0.1],
            "level_1_membership_1": [0.1, 0.9],
            "level_2_membership_0": [0.2, 0.7],
            "level_2_membership_1": [0.8, 0.3],
            "level_1_path_membership_0": [0.9, 0.1],
            "level_1_path_membership_1": [0.1, 0.9],
            "level_2_path_membership_0_0": [0.2, 0.0],
            "level_2_path_membership_0_1": [0.8, 0.0],
            "level_2_path_membership_1_0": [0.0, 0.7],
            "level_2_path_membership_1_1": [0.0, 0.3],
            "cluster": [1, 0],
            "cluster_path": ["0/1", "1/0"],
            "is_noise": [False, False],
        }
    )


class VisualAssignmentTests(unittest.TestCase):
    def test_path_memberships_drive_hierarchical_display_labels(self) -> None:
        prepared = prepare_visual_assignments(
            _assignments_with_path_memberships()
        )

        self.assertEqual(prepared["display_label"].tolist(), ["1-2", "2-1"])
        self.assertTrue(prepared["is_hierarchical"].all())

    def test_path_memberships_are_used_for_cluster_supervision(self) -> None:
        target, metric, description = build_cluster_supervision(
            _assignments_with_path_memberships()
        )

        self.assertEqual(metric, "euclidean")
        self.assertEqual(description, "soft cluster membership (6 dims)")
        self.assertIsNotNone(target)
        self.assertEqual(target.shape, (2, 6))

    def test_assignments_without_path_memberships_use_label_fallback(self) -> None:
        assignments = _assignments_with_path_memberships()
        assignments = assignments.drop(
            columns=[
                column
                for column in assignments
                if "path_membership" in column
            ]
        )

        target, metric, description = build_cluster_supervision(assignments)

        self.assertEqual(metric, "categorical")
        self.assertEqual(description, "cluster labels (2 groups)")
        self.assertEqual(target.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
