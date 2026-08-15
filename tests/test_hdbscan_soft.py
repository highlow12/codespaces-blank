from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hdbscan_soft import _build_distance_soft_result
from hdbscan_soft_pipeline import build_assignments, build_parser, run_pipeline


def _unit(angle: float) -> np.ndarray:
    return np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)


class HdbscanSoftMembershipTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two compact hard clusters, followed by a close and an ambiguous noise.
        angles = [-0.08, -0.04, -0.02, 0.02, 0.04, 0.08]
        self.features = np.vstack(
            [_unit(angle) for angle in angles]
            + [_unit(np.pi / 2 + angle) for angle in angles]
            # The compact-cluster medoids are at -0.02 and pi/2 - 0.02, so
            # their exact angular midpoint is intentionally ambiguous.
            + [_unit(0.12), _unit(np.pi / 4 - 0.02)]
        )
        self.labels = np.asarray([0] * 6 + [1] * 6 + [-1, -1])
        self.result = _build_distance_soft_result(
            self.features, self.labels, reassignment_threshold=0.60,
            neighbor_count=5, medoid_candidate_budget=256,
            medoid_evaluation_budget=1024,
        )

    def test_memberships_are_finite_one_hot_for_members_and_sum_to_one(self) -> None:
        for memberships in (self.result.medoid_memberships, self.result.neighbor_memberships):
            self.assertTrue(np.all(np.isfinite(memberships)))
            np.testing.assert_allclose(memberships.sum(axis=1), np.ones(len(memberships)))
        np.testing.assert_array_equal(self.result.medoid_memberships[:6, 0], np.ones(6))
        np.testing.assert_array_equal(self.result.neighbor_memberships[6:12, 1], np.ones(6))

    def test_close_noise_is_reassigned_but_ambiguous_noise_remains_noise(self) -> None:
        self.assertEqual(self.result.labels[12], 0)
        self.assertGreaterEqual(self.result.medoid_confidences[12], 0.60)
        self.assertEqual(self.result.labels[13], -1)
        self.assertLess(self.result.medoid_confidences[13], 0.60)

    def test_neighbor_method_uses_up_to_exactly_requested_non_noise_members(self) -> None:
        np.testing.assert_array_equal(self.result.neighbor_member_counts, [5, 5])
        small = _build_distance_soft_result(
            self.features[:8], np.asarray([0] * 6 + [1] * 2),
            reassignment_threshold=0.60, neighbor_count=5,
            medoid_candidate_budget=256, medoid_evaluation_budget=1024,
        )
        np.testing.assert_array_equal(small.neighbor_member_counts, [5, 2])

    def test_assignments_and_summary_outputs_are_deterministic(self) -> None:
        metadata = pd.DataFrame({"id": [f"doc-{i}" for i in range(len(self.features))]})
        first = build_assignments(metadata, self.result)
        second = build_assignments(metadata, self.result)
        pd.testing.assert_frame_equal(first, second)
        self.assertIn("membership_medoid_0", first)
        self.assertIn("membership_neighbor_1", first)
        self.assertTrue(first["recommended_labels_agree"].dtype == bool)


class HdbscanSoftCliTests(unittest.TestCase):
    def test_gemini_small_sample_cli_smoke(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = root / "dbpedia_gemini_embeddings.json.gz"
        self.assertTrue(source.exists())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hdbscan"
            args = build_parser().parse_args([
                "--input-json", str(source), "--output-dir", str(output),
                "--dataset-sample-size", "100", "--dataset-sample-seed", "42",
                "--pca-components", "8", "--min-cluster-size", "5", "--min-samples", "3",
            ])
            summary = run_pipeline(args)
            assignments = pd.read_csv(output / "assignments.csv")
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(assignments), 100)
        self.assertEqual(summary["noise_count"], saved["noise_count"])
        self.assertIn("cluster", assignments)
        self.assertIn("medoid_max_membership", assignments)


if __name__ == "__main__":
    unittest.main()
