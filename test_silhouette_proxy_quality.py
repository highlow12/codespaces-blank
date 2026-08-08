"""Unit tests for the exact-vs-proxy quality benchmark helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmark_silhouette_proxy_quality import (
    _labels_from_assignments,
    _rank_correlation,
)


class SilhouetteProxyQualityHelperTest(unittest.TestCase):
    def test_rank_correlation_is_perfect_for_matching_candidate_order(self) -> None:
        self.assertAlmostEqual(
            _rank_correlation([0.1, 0.4, 0.2], [0.3, 0.8, 0.5]),
            1.0,
        )

    def test_rank_correlation_is_not_faked_for_constant_scores(self) -> None:
        self.assertIsNone(_rank_correlation([0.1, 0.1], [0.2, 0.3]))

    def test_labels_replace_noise_with_a_shared_comparison_label(self) -> None:
        result = SimpleNamespace(
            assignments=pd.DataFrame(
                {
                    "cluster_path": ["0", "1", "1"],
                    "is_noise": [False, True, False],
                }
            )
        )

        labels, noise = _labels_from_assignments(result)

        np.testing.assert_array_equal(labels, ["0", "__noise__", "1"])
        np.testing.assert_array_equal(noise, [False, True, False])


if __name__ == "__main__":
    unittest.main()
