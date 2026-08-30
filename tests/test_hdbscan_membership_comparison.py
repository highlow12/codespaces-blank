from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hdbscan_membership_comparison import (
    fit_hdbscan_membership_comparison,
    normalize_native_membership_vectors,
    propagate_exact_knn_memberships,
)
from hdbscan_membership_comparison_pipeline import (
    build_assignments,
    build_boundary_cases,
    build_parser,
    run_pipeline,
)


class ExactKnnMembershipTests(unittest.TestCase):
    def test_self_is_excluded_and_noise_stays_in_denominator(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
        labels = np.asarray([0, -1, 0], dtype=np.int64)
        probabilities = np.ones(3, dtype=np.float64)
        result = propagate_exact_knn_memberships(
            features,
            labels,
            probabilities,
            neighbor_count=2,
        )

        for row_index, neighbor_row in enumerate(result.neighbor_indices):
            self.assertNotIn(row_index, neighbor_row.tolist())
        self.assertIn(1, result.neighbor_indices[0].tolist())
        # Row 0 sees a noise point at d=1 and a leaf point at d=2.  The
        # noise weight remains in the denominator, so the leaf affinity is
        # strictly below one.
        distances = result.neighbor_distances[0]
        sigma = result.local_sigmas[0]
        weights = np.exp(-np.square(distances / sigma))
        expected = weights[result.neighbor_indices[0] == 2].sum() / weights.sum()
        self.assertAlmostEqual(result.affinities[0, 0], float(expected))
        self.assertLess(result.affinities[0, 0], 1.0)
        self.assertGreater(result.unexplained[0], 0.0)

    def test_two_leaf_overlap_is_independent_and_not_softmaxed(self) -> None:
        features = np.asarray([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        labels = np.asarray([-1, 0, 1], dtype=np.int64)
        probabilities = np.asarray([0.0, 0.9, 0.9], dtype=np.float64)
        result = propagate_exact_knn_memberships(
            features,
            labels,
            probabilities,
            neighbor_count=2,
        )

        np.testing.assert_allclose(result.affinities[0], [0.45, 0.45], atol=1e-12)
        self.assertGreater(result.affinities[0, 0], 0.4)
        self.assertGreater(result.affinities[0, 1], 0.4)
        self.assertAlmostEqual(result.affinities[0].sum(), 0.9)
        self.assertAlmostEqual(result.unexplained[0], 0.1)

    def test_low_confidence_and_noise_neighborhoods_retain_unexplained_mass(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
        low_confidence = propagate_exact_knn_memberships(
            features,
            np.zeros(4, dtype=np.int64),
            np.full(4, 0.1, dtype=np.float64),
            neighbor_count=2,
        )
        self.assertTrue(np.all(low_confidence.unexplained >= 0.89))

        all_noise = propagate_exact_knn_memberships(
            features,
            np.full(4, -1, dtype=np.int64),
            np.zeros(4, dtype=np.float64),
            neighbor_count=2,
        )
        self.assertEqual(all_noise.affinities.shape, (4, 0))
        np.testing.assert_allclose(all_noise.unexplained, np.ones(4))
        np.testing.assert_array_equal(all_noise.recommended_labels, -np.ones(4, dtype=np.int64))

    def test_determinism_and_shape_validation(self) -> None:
        features = np.asarray([[0.0], [1.0], [3.0], [7.0]], dtype=np.float64)
        labels = np.asarray([0, 0, 1, -1], dtype=np.int64)
        probabilities = np.asarray([1.0, 0.8, 0.7, 0.0], dtype=np.float64)
        first = propagate_exact_knn_memberships(
            features, labels, probabilities, neighbor_count=2
        )
        second = propagate_exact_knn_memberships(
            features, labels, probabilities, neighbor_count=2
        )
        np.testing.assert_allclose(first.affinities, second.affinities)
        np.testing.assert_array_equal(first.neighbor_indices, second.neighbor_indices)
        with self.assertRaisesRegex(ValueError, "neighbor_count"):
            propagate_exact_knn_memberships(
                features, labels, probabilities, neighbor_count=4
            )
        with self.assertRaisesRegex(ValueError, "probabilities"):
            propagate_exact_knn_memberships(
                features, labels, np.ones(3), neighbor_count=2
            )

    def test_no_cluster_native_membership_shape_is_explicit(self) -> None:
        normalized = normalize_native_membership_vectors(
            np.zeros(4), n_samples=4, cluster_count=0
        )
        self.assertEqual(normalized.shape, (4, 0))
        with self.assertRaisesRegex(ValueError, "non-zero one-dimensional"):
            normalize_native_membership_vectors(
                np.ones(4), n_samples=4, cluster_count=0
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            normalize_native_membership_vectors(
                np.zeros(4), n_samples=4, cluster_count=1
            )


class HdbscanMembershipComparisonPipelineTests(unittest.TestCase):
    def test_raw_pca_prefix_is_not_post_normalized(self) -> None:
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(24, 6))
        result = fit_hdbscan_membership_comparison(
            embeddings,
            pca_components=3,
            umap_n_neighbors=5,
            min_cluster_size=3,
            min_samples=2,
            neighbor_count=4,
            seed=19,
        )
        expected = result.pca_selection.pca.transform(
            result.pca_selection.normalized_input
        )[:, : result.pca_selection.selected_dimension]
        np.testing.assert_allclose(result.pca_features, expected)
        self.assertFalse(np.allclose(np.linalg.norm(result.pca_features, axis=1), 1.0))

    def test_gemini_100_row_cli_smoke_and_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = root / "dbpedia_gemini_embeddings.json.gz"
        self.assertTrue(source.exists())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison"
            args = build_parser().parse_args(
                [
                    "--input-json",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--dataset-sample-size",
                    "100",
                    "--dataset-sample-seed",
                    "42",
                    "--pca-components",
                    "8",
                    "--umap-n-neighbors",
                    "8",
                    "--min-cluster-size",
                    "5",
                    "--min-samples",
                    "3",
                    "--neighbor-count",
                    "8",
                    "--seed",
                    "42",
                ]
            )
            summary = run_pipeline(args)
            assignments = pd.read_csv(output / "assignments.csv")
            boundary = pd.read_csv(output / "boundary_cases.csv")
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(len(assignments), 100)
        self.assertEqual(summary["samples"], 100)
        self.assertEqual(summary["cluster_count"], saved["cluster_count"])
        self.assertIn("hdbscan_leaf_label", assignments)
        self.assertIn("native_unexplained", assignments)
        self.assertIn("pca_exact_knn_unexplained", assignments)
        self.assertIn("pca_exact_knn_max_affinity", assignments)
        self.assertIn("pca_exact_knn_recommended_leaf", assignments)
        self.assertGreaterEqual(len(boundary), 0)

    def test_no_cluster_fit_still_builds_zero_width_native_artifact(self) -> None:
        rng = np.random.default_rng(8)
        embeddings = rng.normal(size=(12, 5))
        metadata = pd.DataFrame({"id": [f"n-{i}" for i in range(12)]})
        result = fit_hdbscan_membership_comparison(
            embeddings,
            pca_components=2,
            umap_n_neighbors=4,
            min_cluster_size=20,
            min_samples=2,
            neighbor_count=3,
            seed=2,
        )
        self.assertEqual(result.cluster_count, 0)
        assignments = build_assignments(metadata, result)
        self.assertEqual(result.native_memberships.shape, (12, 0))
        self.assertTrue(np.all(assignments["native_unexplained"] == 1.0))
        boundary = build_boundary_cases(metadata, result)
        self.assertEqual(len(boundary), 0)


if __name__ == "__main__":
    unittest.main()
