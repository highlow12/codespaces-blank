import unittest

import numpy as np
import pandas as pd

from embedding_data import sample_embedding_batch


class EmbeddingDataSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.column_stack(
            [
                np.arange(10, dtype=np.float64),
                np.arange(10, dtype=np.float64) + 100.0,
                np.arange(10, dtype=np.float64) + 200.0,
            ]
        )
        self.metadata = pd.DataFrame(
            {
                "id": np.arange(10),
                "tag": [f"tag-{index}" for index in range(10)],
            }
        )

    def test_sampling_is_reproducible_and_keeps_rows_aligned(self) -> None:
        sampled_a, metadata_a = sample_embedding_batch(
            self.embeddings,
            self.metadata,
            sample_size=4,
            seed=123,
        )
        sampled_b, metadata_b = sample_embedding_batch(
            self.embeddings,
            self.metadata,
            sample_size=4,
            seed=123,
        )

        np.testing.assert_array_equal(sampled_a, sampled_b)
        pd.testing.assert_frame_equal(metadata_a, metadata_b)
        np.testing.assert_array_equal(
            sampled_a[:, 0].astype(int),
            metadata_a["id"].to_numpy(),
        )
        self.assertEqual(len(sampled_a), 4)
        self.assertEqual(metadata_a.index.tolist(), list(range(4)))

    def test_sampling_rejects_invalid_size_and_metadata_length(self) -> None:
        with self.assertRaises(ValueError):
            sample_embedding_batch(
                self.embeddings,
                self.metadata,
                sample_size=0,
            )
        with self.assertRaises(ValueError):
            sample_embedding_batch(
                self.embeddings,
                self.metadata,
                sample_size=len(self.embeddings) + 1,
            )
        with self.assertRaises(ValueError):
            sample_embedding_batch(
                self.embeddings,
                self.metadata.iloc[:-1],
                sample_size=4,
            )


if __name__ == "__main__":
    unittest.main()
