import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark_knn_ann_scaling import (
    BACKENDS,
    DEFAULT_SAMPLE_SIZES,
    _split_indices,
    repeat_rows,
    run_benchmark,
    sample_sizes,
)


class ScalingBenchmarkHelpersTest(unittest.TestCase):
    def test_default_sizes_are_logarithmic_and_include_endpoints(self):
        self.assertEqual(sample_sizes(), DEFAULT_SAMPLE_SIZES)
        self.assertEqual(sample_sizes(100, 10000, 9)[0], 100)
        self.assertEqual(sample_sizes(100, 10000, 9)[-1], 10000)

    def test_repeat_rows_preserves_dtype_and_prefix_order(self):
        values = np.arange(12, dtype=np.float32).reshape(4, 3)
        repeated = repeat_rows(values, 7)
        np.testing.assert_array_equal(repeated, values[[0, 1, 2, 3, 0, 1, 2]])
        self.assertEqual(repeated.dtype, values.dtype)

    def test_split_indices_requires_all_three_splits(self):
        rows = [
            {"split": "discovery"}, {"split": "discovery"},
            {"split": "calibration"}, {"split": "calibration"},
            {"split": "test"}, {"split": "test"},
        ]
        result = _split_indices(rows)
        self.assertEqual({key: value.tolist() for key, value in result.items()}, {"discovery": [0, 1], "calibration": [2, 3], "test": [4, 5]})
        with self.assertRaises(ValueError):
            _split_indices([{"split": "discovery"}, {"split": "calibration"}])

    def test_run_writes_csv_and_json_with_backend_rows(self):
        fake_row = {
            "sample_size": 100,
            "backend": "exact",
            "discovery_size": 60,
            "calibration_size": 20,
            "test_size": 20,
            "pca_components": 10,
            "index_build_sec": 0.1,
            "ann_jit_warmup_sec": 0.0,
            "calibration_query_sec": 0.01,
            "test_query_sec": 0.01,
            "query_total_sec": 0.02,
            "neighbor_search_total_sec": 0.12,
            "neighbor_search_including_warmup_sec": 0.12,
            "peak_rss_kib": 1000,
            "baseline_rss_kib": 500,
            "peak_rss_delta_kib": 500,
            "parent_observed_peak_rss_kib": 1000,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = Namespace(
                sample_sizes=[100], start=100, stop=10000, points=9,
                embedding_path=Path("embeddings.npy"), metadata_path=Path("metadata.jsonl"),
                output_dir=output, graph_neighbors=32, epsilon=0.1,
            )
            with patch("benchmark_knn_ann_scaling.measure_case", side_effect=[fake_row, {**fake_row, "backend": "pynndescent"}]), patch("benchmark_knn_ann_scaling._plot"):
                report = run_benchmark(args)
            self.assertEqual({row["backend"] for row in report["measurements"]}, set(BACKENDS))
            self.assertTrue((output / "scaling.csv").exists())
            loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["protocol"]["k"], 24)
            self.assertFalse(loaded["dataset"]["quality_eligible"])


if __name__ == "__main__":
    unittest.main()
