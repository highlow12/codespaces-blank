from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from extract_clustering_metrics import extract_metrics, extract_metrics_from_frame


class ExtractClusteringMetricsTests(unittest.TestCase):
    def _write_inputs(self, directory: Path) -> tuple[Path, Path]:
        assignments = directory / "assignments.csv"
        features = directory / "features.npy"
        frame = pd.DataFrame(
            {
                "class": ["A", "A", "B", "B"],
                "class_hierarchy": [
                    ["Top1", "Sub1", "A"],
                    ["Top1", "Sub1", "A"],
                    ["Top2", "Sub2", "B"],
                    ["Top2", "Sub2", "B"],
                ],
                "cluster": [1, 1, 0, 0],
                "membership_0": [0.1, 0.2, 0.8, 0.9],
                "membership_1": [0.9, 0.8, 0.2, 0.1],
            }
        )
        frame.to_csv(assignments, index=False)
        np.save(
            features,
            np.asarray(
                [
                    [3.0, 3.0],
                    [3.2, 3.1],
                    [-3.0, -3.0],
                    [-3.1, -3.2],
                ]
            ),
        )
        return assignments, features

    def test_extracts_external_fuzzy_and_feature_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assignments, features = self._write_inputs(Path(temporary))
            metrics = extract_metrics(
                assignments,
                features_path=features,
            )

        self.assertAlmostEqual(metrics["nmi"], 1.0)
        self.assertAlmostEqual(metrics["ari"], 1.0)
        self.assertAlmostEqual(metrics["nmi_top"], 1.0)
        self.assertAlmostEqual(metrics["tag_fragmentation"], 1.0)
        self.assertGreater(metrics["pc"], 0.5)
        self.assertLess(metrics["pe"], np.log(2.0))
        self.assertIsNotNone(metrics["silhouette"])
        self.assertIsNotNone(metrics["xb"])

    def test_external_metrics_remain_available_without_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assignments, _features = self._write_inputs(Path(temporary))
            metrics = extract_metrics(assignments)

        self.assertAlmostEqual(metrics["nmi"], 1.0)
        self.assertIsNone(metrics["silhouette"])
        self.assertIsNone(metrics["xb"])
        self.assertIsNotNone(metrics["pc"])

    def test_assignment_without_memberships_reports_missing_fuzzy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assignments = Path(temporary) / "hard.csv"
            pd.DataFrame(
                {
                    "class": ["A", "B"],
                    "cluster": [0, 1],
                }
            ).to_csv(assignments, index=False)
            metrics = extract_metrics(assignments)

        self.assertIsNone(metrics["pc"])
        self.assertIsNone(metrics["pe"])
        self.assertEqual(metrics["clusters"], 2)

    def test_in_memory_api_uses_the_same_metric_core(self) -> None:
        frame = pd.DataFrame(
            {
                "class": ["A", "A", "B", "B"],
                "cluster": [0, 0, 1, 1],
                "membership_0": [0.9, 0.8, 0.1, 0.2],
                "membership_1": [0.1, 0.2, 0.9, 0.8],
            }
        )
        metrics = extract_metrics_from_frame(
            frame,
            features=np.asarray(
                [[3.0, 3.0], [3.2, 3.1], [-3.0, -3.0], [-3.1, -3.2]]
            ),
            source="unit-test",
            centers=np.asarray([[3.1, 3.05], [-3.05, -3.1]]),
        )

        self.assertEqual(metrics["source"], "unit-test")
        self.assertAlmostEqual(metrics["nmi"], 1.0)
        self.assertIsNotNone(metrics["xb"])


if __name__ == "__main__":
    unittest.main()
