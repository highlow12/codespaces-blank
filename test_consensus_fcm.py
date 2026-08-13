from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from consensus_fcm import ConsensusFcmConfig, select_consensus_fcm_cluster_count


class ConsensusFcmSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.random.default_rng(7).normal(size=(100, 4))

    def test_stops_at_majority_and_fits_only_the_winning_k(self) -> None:
        selected = SimpleNamespace(n_clusters=5)
        side_effect = [
            (SimpleNamespace(n_clusters=5), [{"k": 5}], "scout"),
            (SimpleNamespace(n_clusters=7), [{"k": 7}], "scout"),
            (SimpleNamespace(n_clusters=5), [{"k": 5}], "scout"),
            (SimpleNamespace(n_clusters=5), [{"k": 5}], "scout"),
            (selected, [{"k": 5}], "refined"),
        ]
        config = ConsensusFcmConfig(sample_ratio=0.20)

        with patch(
            "consensus_fcm.select_fcm_cluster_count",
            side_effect=side_effect,
        ) as mocked:
            best, records, reason = select_consensus_fcm_cluster_count(
                self.features,
                min_clusters=2,
                max_clusters=8,
                min_child_size=20,
                config=config,
            )

        self.assertIs(best, selected)
        self.assertEqual(reason, "selected_consensus_sample_vote")
        self.assertEqual(mocked.call_count, 5)
        for call in mocked.call_args_list[:4]:
            self.assertEqual(call.args[0].shape[0], 20)
            self.assertEqual(call.kwargs["min_child_size"], 4)
            self.assertEqual(call.kwargs["n_init"], 3)
        final_call = mocked.call_args_list[-1]
        self.assertEqual(final_call.args[0].shape[0], 100)
        self.assertEqual(final_call.kwargs["min_clusters"], 5)
        self.assertEqual(final_call.kwargs["max_clusters"], 5)
        self.assertEqual(final_call.kwargs["n_init"], 10)
        self.assertEqual(
            [record["phase"] for record in records],
            [
                "consensus_scout",
                "consensus_scout",
                "consensus_scout",
                "consensus_scout",
                "consensus_full_fit",
            ],
        )

    def test_falls_back_to_exhaustive_selection_without_a_majority(self) -> None:
        selected = SimpleNamespace(n_clusters=5)
        side_effect = [
            (SimpleNamespace(n_clusters=k), [{"k": k}], "scout")
            for k in (2, 3, 4, 5)
        ] + [(selected, [{"k": 5}], "selected_multi_metric_max_k")]

        with patch(
            "consensus_fcm.select_fcm_cluster_count",
            side_effect=side_effect,
        ) as mocked:
            best, records, reason = select_consensus_fcm_cluster_count(
                self.features,
                min_clusters=2,
                max_clusters=8,
                config=ConsensusFcmConfig(sample_ratio=0.20),
            )

        self.assertIs(best, selected)
        self.assertEqual(
            reason,
            "consensus_fallback:selected_multi_metric_max_k",
        )
        self.assertEqual(mocked.call_count, 5)
        fallback_call = mocked.call_args_list[-1]
        self.assertEqual(fallback_call.args[0].shape[0], 100)
        self.assertEqual(fallback_call.kwargs["min_clusters"], 2)
        self.assertEqual(fallback_call.kwargs["max_clusters"], 8)
        self.assertEqual(records[-1]["phase"], "consensus_full_fallback")

    def test_small_dataset_uses_direct_full_selection(self) -> None:
        selected = SimpleNamespace(n_clusters=3)
        config = ConsensusFcmConfig(sample_size=100)

        with patch(
            "consensus_fcm.select_fcm_cluster_count",
            return_value=(selected, [{"k": 3}], "selected"),
        ) as mocked:
            best, records, reason = select_consensus_fcm_cluster_count(
                self.features,
                config=config,
            )

        self.assertIs(best, selected)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(records[0]["phase"], "consensus_full_data_direct")
        self.assertEqual(reason, "consensus_full_data_direct:selected")

    def test_vote_threshold_must_be_a_strict_majority(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict majority"):
            ConsensusFcmConfig(max_scouts=5, vote_threshold=2).validate()


if __name__ == "__main__":
    unittest.main()
