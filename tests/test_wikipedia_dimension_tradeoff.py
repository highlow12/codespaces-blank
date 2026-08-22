import unittest

import numpy as np

from wikipedia_soft_benchmark.dimension_tradeoff import (
    curve_rows,
    effective_dimension,
)


class WikipediaDimensionTradeoffTests(unittest.TestCase):
    def test_centered_rank_cap_excludes_512_for_discovery(self):
        self.assertEqual(
            effective_dimension(discovery_rows=432, input_dimension=768, requested=512, mode="centered-pca"),
            431,
        )
        self.assertEqual(
            effective_dimension(discovery_rows=432, input_dimension=768, requested=512, mode="uncentered-svd"),
            432,
        )

    def test_rank_cap_respects_input_dimension_and_rejects_invalid(self):
        self.assertEqual(
            effective_dimension(discovery_rows=12, input_dimension=8, requested=99, mode="centered-pca"),
            8,
        )
        with self.assertRaises(ValueError):
            effective_dimension(discovery_rows=1, input_dimension=8, requested=2, mode="centered-pca")

    def test_curve_rows_have_deterministic_split_order(self):
        geometry = {
            "cosine_pearson": {"mean": 0.9, "std": 0.01},
            "cosine_spearman": {"mean": 0.8, "std": 0.02},
            "cosine_knn_recall_at_24": {"mean": 0.7, "std": 0.03},
            "euclidean_knn_recall_at_24": {"mean": 0.6, "std": 0.04},
            "leaf_purity_cosine": {"mean": 0.5, "std": 0.05},
            "leaf_purity_euclidean": {"mean": 0.4, "std": 0.06},
            "leaf_cosine_margin": {"mean": 0.3, "std": 0.07},
        }
        test_metrics = {"leaf_nmi": 0.5, "leaf_ari": 0.4, "unexplained_mass": 0.2}
        report = {
            "results": [
                {
                    "mode": "centered-pca",
                    "dimension": 32,
                    "geometry": {"calibration": {"summary": geometry}, "test": {"summary": geometry}},
                    "test": {"native": test_metrics, "exact_knn": test_metrics},
                    "selected_configuration": {"seed": 42, "min_cluster_size": 18, "min_samples": 3, "neighbor_count": 8},
                }
            ]
        }
        rows = curve_rows(report)
        self.assertEqual([row["split"] for row in rows], ["calibration", "test", "test_cluster_quality", "test_cluster_quality"])
        self.assertEqual(rows[0]["cosine_spearman_mean"], 0.8)
        self.assertEqual(rows[-1]["method"], "exact_knn")


if __name__ == "__main__":
    unittest.main()
