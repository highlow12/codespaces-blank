import unittest

import numpy as np

from fcm_hierarchy import fcm_membership_noise_mask


class FcmMembershipNoiseMaskTest(unittest.TestCase):
    def test_noise_requires_low_maximum_and_small_top_two_gap(self) -> None:
        memberships = np.array(
            [
                [0.36, 0.34, 0.30],  # both conditions
                [0.45, 0.44, 0.11],  # small gap, maximum is not low
                [0.39, 0.31, 0.30],  # low maximum, gap is not small
                [0.70, 0.20, 0.10],  # neither condition
            ]
        )

        result = fcm_membership_noise_mask(
            memberships,
            min_membership=0.40,
            max_membership_gap=0.05,
        )

        np.testing.assert_array_equal(result, [True, False, False, False])

    def test_thresholds_are_strict(self) -> None:
        memberships = np.array(
            [
                [0.50, 0.375, 0.125],
                [0.375, 0.3125, 0.3125],
            ]
        )

        result = fcm_membership_noise_mask(
            memberships,
            min_membership=0.50,
            max_membership_gap=0.0625,
        )

        np.testing.assert_array_equal(result, [False, False])

    def test_single_cluster_has_no_boundary(self) -> None:
        result = fcm_membership_noise_mask(
            np.array([[0.30], [1.00]]),
            min_membership=0.40,
            max_membership_gap=0.10,
        )

        np.testing.assert_array_equal(result, [False, False])

    def test_rejects_invalid_gap_threshold(self) -> None:
        for threshold in (-0.01, 1.01):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "max_membership_gap"):
                    fcm_membership_noise_mask(
                        np.array([[0.5, 0.5]]),
                        max_membership_gap=threshold,
                    )


if __name__ == "__main__":
    unittest.main()
