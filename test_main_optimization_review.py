import unittest

import numpy as np

from benchmark_main_optimization_review import generate_synthetic_cases


class SyntheticReviewDatasetTests(unittest.TestCase):
    def test_cases_are_deterministic_finite_and_normalized(self) -> None:
        first = generate_synthetic_cases()
        second = generate_synthetic_cases()

        self.assertEqual(len(first), 5)
        self.assertEqual(len({case.name for case in first}), len(first))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.embeddings, right.embeddings)
            np.testing.assert_array_equal(left.truth_labels, right.truth_labels)
            np.testing.assert_array_equal(left.truth_noise, right.truth_noise)
            self.assertEqual(len(left.embeddings), len(left.truth_labels))
            self.assertTrue(np.isfinite(left.embeddings).all())
            np.testing.assert_allclose(
                np.linalg.norm(left.embeddings, axis=1),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_cases_include_expected_adversarial_properties(self) -> None:
        cases = {case.name: case for case in generate_synthetic_cases()}

        imbalanced_counts = np.bincount(
            cases["imbalanced_overlap"].truth_labels
        )
        self.assertGreaterEqual(
            imbalanced_counts.max() / imbalanced_counts.min(), 50
        )

        tied = cases["duplicate_ties"]
        self.assertEqual(int(tied.truth_noise.sum()), 60)
        self.assertLess(len(np.unique(tied.embeddings, axis=0)), len(tied.embeddings))

        noisy = cases["boundary_and_noise"]
        self.assertEqual(int(noisy.truth_noise.sum()), 100)

        rank_deficient = cases["rank_deficient_duplicates"]
        self.assertLessEqual(np.linalg.matrix_rank(rank_deficient.embeddings), 5)
        self.assertLess(
            len(np.unique(rank_deficient.embeddings, axis=0)),
            len(rank_deficient.embeddings),
        )

        tiny = cases["tiny_clusters_high_k"]
        self.assertEqual(np.bincount(tiny.truth_labels).min(), 5)
        self.assertEqual(tiny.max_clusters, 8)
        self.assertEqual(tiny.min_child_size, 5)


if __name__ == "__main__":
    unittest.main()
