from __future__ import annotations

import unittest

import numpy as np

from tag_embedding_features import build_tag_augmented_features


class TagEmbeddingFeaturesTests(unittest.TestCase):
    def test_builds_normalized_content_and_weighted_tag_blocks(self) -> None:
        content = np.array([[3.0, 4.0], [0.0, 2.0]])
        tags = np.array([[1.0, 0.0], [1.0, 1.0]])

        actual = build_tag_augmented_features(content, tags, tag_weight=2.0)

        np.testing.assert_allclose(actual[0], [0.6, 0.8, 2.0, 0.0])
        np.testing.assert_allclose(
            actual[1], [0.0, 1.0, np.sqrt(2.0), np.sqrt(2.0)]
        )

    def test_rejects_non_positive_tag_weight(self) -> None:
        with self.assertRaises(ValueError):
            build_tag_augmented_features(
                np.ones((1, 2)),
                np.ones((1, 2)),
                tag_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
