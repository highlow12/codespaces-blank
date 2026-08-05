from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from full_pipeline import (
    fit_auto_pca_sfcm,
    make_incremental_test_split,
    run_full_pipeline,
    update_auto_pca_sfcm,
)


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

    def test_full_pipeline_runs_the_updated_visualization_after_incremental_test(self) -> None:
        selections = [
            SimpleNamespace(selected_dimension=3),
            SimpleNamespace(selected_dimension=4),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "full_pipeline.save_auto_pca_visualization",
            side_effect=selections,
        ) as save_visualization:
            summary = run_full_pipeline(
                self.embeddings,
                self.metadata,
                output_dir=Path(temporary_directory),
                min_clusters=2,
                max_clusters=2,
                min_child_size=2,
                seed=12,
                incremental_test=True,
            )

            self.assertEqual(save_visualization.call_count, 2)
            self.assertEqual(summary["initial_samples"], 18)
            self.assertEqual(summary["initial_selected_clusters"], 2)
            self.assertEqual(summary["incremental_test"]["new_samples"], 2)
            self.assertEqual(summary["incremental_test"]["modified_samples"], 2)
            self.assertEqual(summary["incremental_test"]["total_samples"], 20)
            self.assertTrue(
                Path(summary["artifacts"]["initial_assignments"]).is_file()
            )
            self.assertTrue(
                Path(
                    summary["incremental_test"]["artifacts"]["updated_assignments"]
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
