from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from consensus_fcm import select_consensus_fcm_cluster_count
from full_pipeline import (
    build_parser,
    fit_auto_pca_sfcm,
    make_incremental_test_split,
    run_full_pipeline,
    update_auto_pca_sfcm,
)
from fast_fcm import FastFcmConfig


class FullPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(9)
        left = rng.normal(loc=[-3.0, 0.0, 0.0, 0.0], scale=0.15, size=(10, 4))
        right = rng.normal(loc=[3.0, 0.0, 0.0, 0.0], scale=0.15, size=(10, 4))
        self.embeddings = np.vstack([left, right])
        self.metadata = pd.DataFrame({"id": [f"doc-{index}" for index in range(20)]})

    def test_incremental_split_has_equal_new_and_modified_rows(self) -> None:
        split = make_incremental_test_split(
            self.embeddings,
            self.metadata,
            ratio=0.20,
            modification_noise=0.10,
            seed=12,
        )

        self.assertEqual(len(split.initial_embeddings), 18)
        self.assertEqual(len(split.update_embeddings), 4)
        self.assertEqual(split.new_count, 2)
        self.assertEqual(split.modified_count, 2)
        new_rows = split.update_metadata["incremental_operation"] == "new"
        modified_rows = ~new_rows
        initial_ids = set(split.initial_metadata["id"])
        self.assertFalse(set(split.update_metadata.loc[new_rows, "id"]) & initial_ids)
        self.assertTrue(set(split.update_metadata.loc[modified_rows, "id"]) <= initial_ids)
        original_by_id = {
            identifier: embedding
            for identifier, embedding in zip(
                self.metadata["id"], self.embeddings, strict=True
            )
        }
        for identifier, embedding in zip(
            split.update_metadata.loc[modified_rows, "id"],
            split.update_embeddings[modified_rows.to_numpy()],
            strict=True,
        ):
            self.assertFalse(np.allclose(embedding, original_by_id[identifier]))

    def test_incremental_update_replaces_and_appends_without_refitting_pca(self) -> None:
        split = make_incremental_test_split(
            self.embeddings,
            self.metadata,
            seed=12,
        )
        state = fit_auto_pca_sfcm(
            split.initial_embeddings,
            split.initial_metadata,
            min_clusters=2,
            max_clusters=2,
            min_child_size=2,
            seed=12,
        )

        updated, summary = update_auto_pca_sfcm(
            state,
            split.update_embeddings,
            split.update_metadata,
        )

        self.assertIs(updated.pca, state.pca)
        self.assertEqual(summary["replaced_samples"], 2)
        self.assertEqual(summary["appended_samples"], 2)
        self.assertEqual(summary["total_samples"], 20)
        self.assertEqual(len(updated.assignments), 20)
        np.testing.assert_allclose(
            updated.assignments.filter(like="membership_").sum(axis=1),
            np.ones(20),
        )

    def test_default_large_fit_uses_consensus_k_selection(self) -> None:
        with patch(
            "full_pipeline.select_consensus_fcm_cluster_count",
            wraps=select_consensus_fcm_cluster_count,
        ) as consensus_selector:
            state = fit_auto_pca_sfcm(
                self.embeddings,
                self.metadata,
                min_clusters=2,
                max_clusters=2,
                min_child_size=2,
                seed=12,
                consensus_min_rows=20,
            )

        consensus_selector.assert_called_once()
        self.assertTrue(
            state.cluster_selection_reason.startswith("consensus_")
            or state.cluster_selection_reason == "selected_consensus_sample_vote"
        )

    def test_small_fit_keeps_exact_k_selection(self) -> None:
        with patch(
            "full_pipeline.select_consensus_fcm_cluster_count",
            side_effect=AssertionError("small fits must stay exact"),
        ):
            state = fit_auto_pca_sfcm(
                self.embeddings,
                self.metadata,
                min_clusters=2,
                max_clusters=2,
                min_child_size=2,
                seed=12,
            )

        self.assertTrue(state.cluster_selection_reason.startswith("selected_"))

    def test_incremental_update_accepts_new_metadata_columns(self) -> None:
        state = fit_auto_pca_sfcm(
            self.embeddings,
            self.metadata,
            min_clusters=2,
            max_clusters=2,
            min_child_size=2,
            seed=12,
        )
        update_metadata = self.metadata.iloc[:2].copy()
        update_metadata["incremental_operation"] = "modified"

        updated, summary = update_auto_pca_sfcm(
            state,
            self.embeddings[:2],
            update_metadata,
        )

        self.assertEqual(summary["replaced_samples"], 2)
        self.assertEqual(
            updated.metadata.loc[:1, "incremental_operation"].tolist(),
            ["modified", "modified"],
        )
        self.assertTrue(
            updated.metadata.loc[2:, "incremental_operation"].isna().all()
        )

    def test_incremental_update_is_idempotent_by_batch_id(self) -> None:
        split = make_incremental_test_split(
            self.embeddings,
            self.metadata,
            seed=12,
        )
        state = fit_auto_pca_sfcm(
            split.initial_embeddings,
            split.initial_metadata,
            min_clusters=2,
            max_clusters=2,
            min_child_size=2,
            seed=12,
        )

        updated, first_summary = update_auto_pca_sfcm(
            state,
            split.update_embeddings,
            split.update_metadata,
            batch_id="flat-batch-1",
        )
        replayed, replay_summary = update_auto_pca_sfcm(
            updated,
            split.update_embeddings,
            split.update_metadata,
            batch_id="flat-batch-1",
        )

        self.assertFalse(first_summary["idempotent_replay"])
        self.assertTrue(replay_summary["idempotent_replay"])
        self.assertIs(replayed, updated)
        self.assertEqual(updated.generation, 1)
        self.assertEqual(replay_summary["generation"], 1)

        changed_embeddings = split.update_embeddings.copy()
        changed_embeddings[0, 0] += 0.25
        with self.assertRaisesRegex(ValueError, "different content"):
            update_auto_pca_sfcm(
                updated,
                changed_embeddings,
                split.update_metadata,
                batch_id="flat-batch-1",
            )

    def test_fast_fit_searches_fuzzifier_and_preserves_selected_value(self) -> None:
        fast_config = FastFcmConfig(
            sample_size=20,
            scout_n_init=1,
            scout_max_attempts=1,
            scout_max_iter=30,
            refine_n_init=1,
            refine_max_attempts=1,
            refine_max_iter=30,
            max_refine_n_init=1,
            stability_target=0.0,
            minimum_probe_stability=0.0,
            refine_top_k=1,
            m_values=(1.6,),
        )

        initial_metadata = self.metadata.copy()
        initial_metadata["incremental_operation"] = "initial"
        state = fit_auto_pca_sfcm(
            self.embeddings,
            initial_metadata,
            min_clusters=2,
            max_clusters=2,
            min_child_size=2,
            seed=12,
            fast_mode=True,
            fast_config=fast_config,
        )

        self.assertTrue(state.fast_mode)
        self.assertEqual(state.cluster_selection_reason, "selected_fast_scout_refine")
        self.assertAlmostEqual(state.m, 1.6)
        self.assertTrue(
            any(
                metric.get("phase") == "m_probe"
                for metric in state.cluster_selection_metrics
            )
        )
        update_metadata = initial_metadata.iloc[:2].copy()
        update_metadata["incremental_operation"] = "modified"
        updated, _summary = update_auto_pca_sfcm(
            state,
            self.embeddings[:2],
            update_metadata,
        )
        self.assertTrue(updated.fast_mode)
        self.assertAlmostEqual(updated.m, 1.6)

    def test_parser_exposes_fast_fuzzifier_search_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--input-json",
                "embeddings.json",
                "--fast",
                "--fast-m",
                "1.9",
                "1.5",
            ]
        )

        self.assertTrue(args.fast)
        self.assertEqual(args.fast_m, [1.9, 1.5])

    def test_parser_can_disable_default_consensus_k_selection(self) -> None:
        args = build_parser().parse_args(
            ["--input-json", "embeddings.json", "--exact-k-selection"]
        )

        self.assertFalse(args.consensus_k_selection)

    def test_full_pipeline_defaults_to_hierarchy_with_automatic_k(self) -> None:
        initial_state = SimpleNamespace(
            embeddings=self.embeddings[:18],
            tree={
                "summary": {"leaf_cluster_count": 2, "levels_reached": 1},
                "config": {"consensus_k_selection": True},
            },
        )
        updated_state = SimpleNamespace(embeddings=self.embeddings)
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "full_pipeline.fit_incremental_state",
            return_value=initial_state,
        ) as fit_hierarchy, patch(
            "full_pipeline.update_incremental_state",
            return_value=(
                updated_state,
                {"new_samples": 4, "total_samples": 20},
            ),
        ) as update_hierarchy, patch(
            "full_pipeline.write_incremental_outputs",
        ) as write_outputs:
            summary = run_full_pipeline(
                self.embeddings,
                self.metadata,
                output_dir=Path(temporary_directory),
                seed=12,
                incremental_test=True,
            )

            fit_kwargs = fit_hierarchy.call_args.kwargs
            self.assertEqual(fit_kwargs["max_depth"], 4)
            self.assertEqual(fit_kwargs["min_node_size"], 60)
            self.assertEqual(fit_kwargs["min_child_size"], 20)
            self.assertTrue(fit_kwargs["consensus_k_selection"])
            self.assertEqual(write_outputs.call_count, 2)
            update_hierarchy.assert_called_once()
            self.assertEqual(summary["initial_samples"], 18)
            self.assertEqual(summary["pipeline"], "hierarchical_pca_sfcm")
            self.assertEqual(summary["hierarchy"]["leaf_cluster_count"], 2)
            self.assertEqual(summary["incremental_test"]["new_samples"], 2)
            self.assertEqual(summary["incremental_test"]["modified_samples"], 2)
            self.assertEqual(summary["incremental_test"]["total_samples"], 20)


if __name__ == "__main__":
    unittest.main()
