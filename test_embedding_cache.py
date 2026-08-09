import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from embedding_cache import (
    build_embedding_cache,
    load_embeddings_from_cache,
    read_cache_manifest,
)
from embedding_data import load_embeddings_from_json


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": f"doc-{index}",
                "tag": f"tag-{index}",
                "class_hierarchy": ["top", f"class-{index}"],
                "embedding": [
                    float(index),
                    float(index) + 0.25,
                    float(index) + 0.5,
                ],
            }
            for index in range(6)
        ]

    def _write_gzip_json(self, directory: str) -> Path:
        path = Path(directory) / "embeddings.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(self.records, handle)
        return path

    def test_cache_manifest_and_partial_load_match_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = self._write_gzip_json(directory)
            cache_dir = Path(directory) / "cache"
            manifest_path = build_embedding_cache(input_path, cache_dir)
            manifest_path_from_read, manifest = read_cache_manifest(cache_dir)

            self.assertEqual(manifest_path, manifest_path_from_read)
            self.assertEqual(manifest["record_count"], len(self.records))
            self.assertEqual(manifest["embedding"]["dtype"], "<f4")
            self.assertEqual(manifest["embedding"]["shape"], [6, 3])
            self.assertEqual(manifest["metadata"]["schema_version"], 1)

            expected_embeddings, expected_metadata = load_embeddings_from_json(
                input_path
            )
            cached_embeddings, cached_metadata = load_embeddings_from_cache(
                cache_dir,
                start=1,
                limit=3,
            )
            np.testing.assert_array_equal(
                cached_embeddings,
                expected_embeddings[1:4].astype(np.float32),
            )
            pd.testing.assert_frame_equal(
                cached_metadata,
                expected_metadata.iloc[1:4].reset_index(drop=True),
            )

    def test_sampled_cache_load_uses_same_seed_and_does_not_decode_source_json(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = self._write_gzip_json(directory)
            cache_dir = Path(directory) / "cache"
            build_embedding_cache(input_path, cache_dir)

            with patch(
                "embedding_cache.json.load",
                side_effect=AssertionError("cache load decoded source JSON"),
            ):
                cached_embeddings, cached_metadata = load_embeddings_from_cache(
                    cache_dir,
                    sample_size=2,
                    sample_seed=17,
                )

            selected_indices = np.sort(
                np.random.default_rng(17).choice(6, size=2, replace=False)
            )
            expected_embeddings = np.asarray(
                [self.records[index]["embedding"] for index in selected_indices],
                dtype=np.float32,
            )
            expected_ids = [
                self.records[index]["id"] for index in selected_indices
            ]
            np.testing.assert_array_equal(cached_embeddings, expected_embeddings)
            self.assertEqual(cached_metadata["id"].tolist(), expected_ids)

    def test_cache_preserves_json_id_and_tag_fallbacks(self) -> None:
        records = [
            {"class": "Artist", "embedding": [1.0, 2.0]},
            {"resource": "resource-2", "embedding": [3.0, 4.0]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "fallbacks.json"
            input_path.write_text(json.dumps(records), encoding="utf-8")
            cache_dir = Path(directory) / "cache"
            build_embedding_cache(input_path, cache_dir, id_offset=10)
            _embeddings, metadata = load_embeddings_from_cache(cache_dir)

        self.assertEqual(metadata["id"].tolist(), [10, "resource-2"])
        self.assertEqual(metadata["tag"].tolist(), ["Artist", "Document_1"])


if __name__ == "__main__":
    unittest.main()
