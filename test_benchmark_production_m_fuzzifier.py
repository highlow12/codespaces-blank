from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmark_production_m_fuzzifier import (
    _external_metrics,
    _pairwise_seed_aris,
    _root_selected_record,
)


class ProductionMFuzzifierBenchmarkHelpersTest(unittest.TestCase):
    def test_external_metrics_use_top_tag_and_leaf_class(self) -> None:
        metadata = pd.DataFrame(
            {
                "tag": ["People", "People", "Arts", "Arts"],
                "class": ["Artist", "Artist", "Painter", "Painter"],
            }
        )
        metrics = _external_metrics(
            metadata,
            np.array([0, 0, 1, 1]),
            np.array(["0", "0", "1", "1"]),
        )
        self.assertEqual(metrics["top_ari"], 1.0)
        self.assertEqual(metrics["leaf_nmi"], 1.0)

    def test_root_record_prefers_final_fit_over_probe(self) -> None:
        record = _root_selected_record(
            {
                "selected_k": 2,
                "selected_m": 1.2,
                "candidate_metrics": [
                    {"k": 2, "m": 1.2, "phase": "m_probe", "restart_stability": 0.8},
                    {
                        "k": 2,
                        "m": 1.2,
                        "phase": "consensus_full_fit",
                        "restart_stability": 1.0,
                    },
                ],
            }
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["phase"], "consensus_full_fit")
        self.assertEqual(record["restart_stability"], 1.0)

    def test_seed_and_basic_fast_partition_aris_are_computed(self) -> None:
        records = [
            {"run_id": "consensus_auto_m__seed_42", "condition": "consensus_auto_m", "seed": 42},
            {"run_id": "consensus_auto_m__seed_43", "condition": "consensus_auto_m", "seed": 43},
            {"run_id": "consensus_auto_m__seed_44", "condition": "consensus_auto_m", "seed": 44},
            {"run_id": "fast_auto_m__seed_42", "condition": "fast_auto_m", "seed": 42},
            {"run_id": "fast_auto_m__seed_43", "condition": "fast_auto_m", "seed": 43},
            {"run_id": "fast_auto_m__seed_44", "condition": "fast_auto_m", "seed": 44},
        ]
        partitions = {
            row["run_id"]: (np.array([0, 0, 1, 1]), np.array(["0", "0", "1", "1"]))
            for row in records
        }
        seed_aris = _pairwise_seed_aris(records, partitions)
        self.assertEqual(seed_aris["consensus_auto_m"]["mean_leaf_ari"], 1.0)


if __name__ == "__main__":
    unittest.main()
