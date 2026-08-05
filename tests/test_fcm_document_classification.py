import unittest

import numpy as np

from fcm_hierarchy import (
    DEFAULT_FORCED_NOISE_RATIO,
    classify_fcm_documents,
    fcm_noise_scores,
    forced_noise_mask,
)


class FcmDocumentClassificationTest(unittest.TestCase):
    def test_forced_noise_is_disabled_by_default(self) -> None:
        self.assertEqual(DEFAULT_FORCED_NOISE_RATIO, 0.0)

    def test_classifies_noise_boundary_and_core_from_three_signals(self) -> None:
        memberships = np.array(
            [
                [0.36, 0.34, 0.30],  # ambiguous and far
                [0.36, 0.34, 0.30],  # ambiguous but close
                [0.39, 0.31, 0.30],  # low maximum, but gap is too large
                [0.45, 0.44, 0.11],  # small gap, but maximum is not low
                [0.70, 0.20, 0.10],  # confident and close
            ]
        )

        result = classify_fcm_documents(
            memberships,
            assigned_distances=np.array([1.2, 0.8, 1.2, 1.2, 0.4]),
            distance_thresholds=np.ones(5),
            min_membership=0.40,
            max_membership_gap=0.05,
        )

        np.testing.assert_array_equal(
            result,
            ["noise", "boundary", "core", "core", "core"],
        )

    def test_membership_thresholds_are_strict(self) -> None:
        memberships = np.array(
            [
                [0.50, 0.375, 0.125],
                [0.375, 0.3125, 0.3125],
            ]
        )

        result = classify_fcm_documents(
            memberships,
            assigned_distances=np.array([2.0, 2.0]),
            distance_thresholds=np.ones(2),
            min_membership=0.50,
            max_membership_gap=0.0625,
        )

        np.testing.assert_array_equal(result, ["core", "core"])

    def test_distance_threshold_is_strict(self) -> None:
        result = classify_fcm_documents(
            np.array([[0.36, 0.34, 0.30]]),
            assigned_distances=np.array([1.0]),
            distance_thresholds=np.array([1.0]),
            min_membership=0.40,
            max_membership_gap=0.05,
        )

        np.testing.assert_array_equal(result, ["boundary"])

    def test_single_cluster_is_core(self) -> None:
        result = classify_fcm_documents(
            np.array([[0.30], [1.00]]),
            assigned_distances=np.array([2.0, 2.0]),
            distance_thresholds=np.ones(2),
            min_membership=0.40,
            max_membership_gap=0.10,
        )

        np.testing.assert_array_equal(result, ["core", "core"])

    def test_rejects_misaligned_distances(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            classify_fcm_documents(
                np.array([[0.5, 0.5]]),
                assigned_distances=np.array([]),
                distance_thresholds=np.array([1.0]),
            )

    def test_noise_score_combines_all_three_ranked_signals(self) -> None:
        scores = fcm_noise_scores(
            np.array(
                [
                    [0.36, 0.34, 0.30],
                    [0.70, 0.20, 0.10],
                    [0.45, 0.44, 0.11],
                ]
            ),
            assigned_distances=np.array([1.2, 0.4, 0.8]),
            assigned_labels=np.zeros(3, dtype=int),
        )

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_forces_exact_top_one_percent_with_stable_id_tie_break(self) -> None:
        scores = np.zeros(200)
        ids = np.array([f"doc-{index:03d}" for index in range(199, -1, -1)])

        result = forced_noise_mask(
            scores,
            ids,
            forced_noise_ratio=0.01,
        )

        self.assertEqual(int(result.sum()), 2)
        self.assertEqual(set(ids[result]), {"doc-000", "doc-001"})


if __name__ == "__main__":
    unittest.main()
