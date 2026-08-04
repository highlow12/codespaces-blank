from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hierarchical_assignments import build_hierarchical_assignments
from pca_projection import (
    fit_normalized_pca_projection,
    transform_normalized_pca_projection,
)


class SharedPcaProjectionTests(unittest.TestCase):
    def test_fit_and_transform_use_the_same_normalized_prefix(self) -> None:
        rng = np.random.default_rng(23)
        embeddings = rng.normal(size=(20, 8))

        fitted = fit_normalized_pca_projection(
            embeddings,
            n_components=6,
            seed=7,
        )
        transformed = transform_normalized_pca_projection(
            embeddings,
            fitted.pca,
            dimension=4,
        )

        np.testing.assert_allclose(
            transformed,
            fitted.normalized_prefix(4),
            atol=1e-12,
        )

    def test_transform_rejects_a_different_embedding_width(self) -> None:
        fitted = fit_normalized_pca_projection(
            np.eye(6),
            n_components=4,
            seed=7,
        )
        with self.assertRaisesRegex(ValueError, "expected 6"):
            transform_normalized_pca_projection(
                np.ones((2, 5)),
                fitted.pca,
            )


class SharedHierarchicalAssignmentTests(unittest.TestCase):
    def test_builds_leaf_and_path_fields_for_fit_and_incremental_callers(self) -> None:
        metadata = pd.DataFrame({"id": ["a", "b", "c"]})
        labels = np.asarray([[0, 1], [1, -1], [0, -1]])
        is_natural_noise = np.asarray([False, True, False])
        is_forced_noise = np.asarray([False, False, True])
        is_noise = is_natural_noise | is_forced_noise
        document_types = np.asarray(["core", "noise", "noise"], dtype=object)

        assignments = build_hierarchical_assignments(
            metadata,
            labels,
            is_noise,
            is_natural_noise,
            is_forced_noise,
            document_types,
            np.asarray([0.1, 0.8, 0.9]),
            np.asarray([-1, -1, -1]),
            np.asarray([-1, 1, 0]),
            conditional_memberships={"0": np.asarray([0.9, 0.1, 0.8])},
        )

        self.assertEqual(assignments["cluster"].tolist(), [1, -1, -1])
        self.assertEqual(
            assignments["cluster_path"].tolist(),
            ["0/1", "1/noise", "0/noise"],
        )
        self.assertEqual(assignments["leaf_level"].tolist(), [2, 1, 1])
        self.assertIn("level_1_path_membership_0", assignments)


if __name__ == "__main__":
    unittest.main()
