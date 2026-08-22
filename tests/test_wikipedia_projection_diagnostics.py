import unittest
from unittest.mock import patch

import numpy as np

from wikipedia_soft_benchmark.projection_diagnostics import (
    build_parser,
    evaluate_projection,
    fit_projection,
    neighbor_indices,
    run_diagnostics,
)
from wikipedia_soft_benchmark.embeddings import l2_normalize


class WikipediaProjectionDiagnosticsTests(unittest.TestCase):
    def test_cli_defaults_to_requested_seeds_and_k(self):
        args = build_parser().parse_args(["--embedding-dir", "emb", "--output-dir", "out"])
        self.assertEqual(args.seeds, [42, 43, 44])
        self.assertEqual(args.neighbor_count, 24)

    def test_neighbor_indices_excludes_equal_ids(self):
        scores = np.asarray([[1.0, 0.9, 0.8]], dtype=np.float64)
        indices = neighbor_indices(
            scores,
            k=2,
            largest=True,
            query_ids=["same"],
            reference_ids=["same", "other", "third"],
        )
        self.assertEqual(indices.tolist(), [[1, 2]])

    def test_evaluate_projection_reports_unit_baseline_recall(self):
        reference = np.eye(4, dtype=np.float32)
        query = reference[[0, 1]]
        metadata_reference = [
            {"id": f"r{i}", "source_id": f"r{i}", "leaf": "a" if i < 2 else "b"}
            for i in range(4)
        ]
        metadata_query = [
            {"id": f"q{i}", "source_id": f"q{i}", "leaf": "a" if i < 2 else "b"}
            for i in range(2)
        ]
        result = evaluate_projection(
            query,
            reference,
            query,
            reference,
            metadata_query,
            metadata_reference,
            k=2,
        )
        self.assertEqual(result["original_bge"]["cosine_knn_recall_at_k"], 1.0)
        self.assertEqual(result["projected"]["cosine_knn_recall_at_k"], 1.0)
        self.assertEqual(result["projected"]["euclidean_knn_recall_at_k"], 1.0)
        self.assertAlmostEqual(result["projected"]["pairwise_cosine"]["pearson"]["mean"], 1.0)

    def test_fit_is_discovery_only(self):
        rng = np.random.default_rng(91)
        discovery = rng.normal(size=(12, 8)).astype(np.float32)
        calibration = rng.normal(size=(4, 8)).astype(np.float32)
        fit_a = fit_projection(discovery, mode="uncentered-svd", seed=42, n_components=4)
        # Changing rows outside discovery cannot change the fitted transformer.
        _ = fit_projection(discovery, mode="uncentered-svd", seed=42, n_components=4)
        transformed_a = fit_a.transformer.transform(calibration)
        transformed_b = fit_a.transformer.transform(calibration * 100.0)
        self.assertFalse(np.allclose(transformed_a, transformed_b))
        self.assertTrue(np.allclose(fit_a.discovery_features, fit_a.transformer.transform(l2_normalize(discovery)), atol=1e-6))

    def test_run_diagnostics_fits_only_discovery_rows(self):
        rng = np.random.default_rng(93)
        embeddings = rng.normal(size=(9, 6)).astype(np.float32)
        metadata = [
            {"id": str(index), "source_id": str(index), "split": split, "leaf": str(index % 2)}
            for index, split in enumerate(("discovery",) * 5 + ("calibration",) * 2 + ("test",) * 2)
        ]
        original_fit = fit_projection
        seen_shapes = []

        def recording_fit(discovery_embeddings, **kwargs):
            seen_shapes.append(tuple(np.asarray(discovery_embeddings).shape))
            return original_fit(discovery_embeddings, **kwargs)

        with patch("wikipedia_soft_benchmark.projection_diagnostics.fit_projection", side_effect=recording_fit):
            run_diagnostics(embeddings, metadata, seeds=(42,), n_components=2, k=2)
        self.assertEqual(seen_shapes, [(5, 6), (5, 6)])

    def test_run_diagnostics_has_both_splits_and_modes(self):
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(12, 10)).astype(np.float32)
        metadata = []
        for index, split in enumerate(("discovery",) * 6 + ("calibration",) * 3 + ("test",) * 3):
            metadata.append({"id": str(index), "source_id": str(index), "split": split, "leaf": str(index % 2)})
        report = run_diagnostics(embeddings, metadata, seeds=(42,), n_components=3, k=2)
        self.assertEqual(set(report["splits"]), {"calibration", "test"})
        for split in report["splits"].values():
            self.assertIn("centered-pca", split["per_seed"])
            self.assertIn("uncentered-svd", split["per_seed"])
            self.assertEqual(split["per_seed"]["centered-pca"]["42"]["reference_count"], 6)


if __name__ == "__main__":
    unittest.main()
