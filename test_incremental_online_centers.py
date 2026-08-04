from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from clustering_types import HierarchicalModel, HierarchyNodeModel
from fcm_hierarchy import fit_pca_normalized_features, spherical_fcm
from incremental_clustering import (
    DEFAULT_NOISE_THRESHOLD,
    IncrementalClusterState,
    _aggregate_center_contributions,
    _build_center_statistics,
    _center_contributions_for_batch,
    _rebuild_tree_counts,
    _update_hierarchy_centers_from_statistics,
    assign_to_hierarchy,
    hierarchy_xie_beni_index,
    load_state,
    save_state,
    update_incremental_state,
)


class _IdentityReducer:
    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X[:, :2], dtype=np.float64)


def _tree_template() -> dict[str, object]:
    def child(cluster_id: int) -> dict[str, object]:
        return {
            "node_id": str(cluster_id),
            "parent_id": "root",
            "path": str(cluster_id),
            "depth": 1,
            "size": 0,
            "noise_count": 0,
            "children": [],
        }

    return {
        "root": {
            "node_id": "root",
            "parent_id": None,
            "path": "",
            "depth": 0,
            "size": 0,
            "noise_count": 0,
            "children": [child(0), child(1)],
        },
        "summary": {},
        "config": {},
    }


class IncrementalOnlineCenterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.asarray(
            [
                [3.0, 0.2, 0.1, 0.0],
                [2.8, 0.3, 0.0, 0.1],
                [3.2, 0.1, 0.2, 0.0],
                [0.1, 3.0, 0.0, 0.2],
                [0.2, 2.9, 0.1, 0.0],
                [0.0, 3.1, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        self.metadata = pd.DataFrame({"id": np.arange(len(self.X))})

    def _make_state(
        self,
        *,
        center_refresh_interval: int = 10,
        max_xb_relative_degradation: float = 0.05,
    ) -> IncrementalClusterState:
        projected, pca = fit_pca_normalized_features(
            self.X,
            n_components=3,
            seed=11,
        )
        fcm = spherical_fcm(projected, n_clusters=2, seed=11)
        model = HierarchicalModel(
            pca=pca,
            nodes={
                "": HierarchyNodeModel(
                    path="",
                    depth=0,
                    centers=fcm.centers.copy(),
                    distance_thresholds=np.full(2, np.inf),
                )
            },
            max_depth=1,
        )
        assignments, _ = assign_to_hierarchy(
            self.X,
            self.metadata,
            model,
            min_membership=0.0,
        )
        config = {
            "max_depth": 1,
            "min_node_size": 4,
            "min_child_size": 2,
            "min_clusters": 2,
            "max_clusters": 2,
            "min_membership": 0.0,
            "distance_z": 3.5,
            "selection_method": "silhouette",
            "min_split_silhouette": -1.0,
            "pca_components": 3,
            "seed": 11,
            "m": 2.0,
            "noise_threshold": 1.0,
            "visual_pca_components": 3,
            "visual_cluster_target_weight": 0.0,
            "visual_n_neighbors": 3,
            "visual_min_dist": 0.02,
            "visual_metric": "cosine",
            "visual_spread": 0.85,
            "visual_densmap": False,
            "update_count": 0,
            "center_updates_before_membership_refresh": center_refresh_interval,
            "max_xb_relative_degradation": max_xb_relative_degradation,
            "center_updates_since_membership_refresh": 0,
            "membership_refreshes_since_recluster": 0,
            "total_center_updates": 0,
            "total_membership_refreshes": 0,
            "total_reclusters": 0,
        }
        baseline_xie_beni = hierarchy_xie_beni_index(
            self.X,
            model,
            min_membership=0.0,
            m=2.0,
        )
        config["baseline_xie_beni"] = baseline_xie_beni
        config["current_xie_beni"] = baseline_xie_beni
        tree = _rebuild_tree_counts(_tree_template(), assignments, 0)
        reducer = _IdentityReducer()
        coordinates = reducer.transform(projected)
        return IncrementalClusterState(
            embeddings=self.X.copy(),
            metadata=self.metadata.copy(),
            assignments=assignments,
            coordinates=coordinates,
            hierarchy_model=model,
            tree=tree,
            config=config,
            visual_pca=pca,
            visual_reducer=reducer,
            center_statistics=_build_center_statistics(
                self.X,
                model,
                min_membership=0.0,
                m=2.0,
            ),
        )

    def test_each_batch_updates_fcm_centers_without_full_refresh(self) -> None:
        state = self._make_state(center_refresh_interval=5)
        previous_centers = state.hierarchy_model.nodes[""].centers.copy()
        batch = np.asarray(
            [[3.0, 0.8, 0.4, 0.0], [2.7, 0.9, 0.2, 0.1]],
            dtype=np.float64,
        )
        metadata = pd.DataFrame({"id": [100, 101]})

        updated, summary = update_incremental_state(state, batch, metadata)

        self.assertTrue(summary["center_updated"])
        self.assertFalse(summary["membership_refreshed"])
        self.assertFalse(summary["reclustered"])
        self.assertGreater(
            np.max(
                np.abs(
                    updated.hierarchy_model.nodes[""].centers
                    - previous_centers
                )
            ),
            1e-8,
        )
        np.testing.assert_allclose(
            state.hierarchy_model.nodes[""].centers,
            previous_centers,
        )

    def test_changed_note_replaces_embedding_and_center_contribution(self) -> None:
        state = self._make_state(center_refresh_interval=10)
        changed_embedding = np.asarray(
            [[2.7, 0.9, 0.2, 0.1]],
            dtype=np.float64,
        )
        changed_metadata = pd.DataFrame(
            {
                "id": [1],
                "text": ["updated note"],
            }
        )

        old_contributions = _center_contributions_for_batch(
            state.embeddings,
            state.metadata,
            state.hierarchy_model,
            min_membership=0.0,
            m=2.0,
        )
        replacement_contributions = _center_contributions_for_batch(
            changed_embedding,
            changed_metadata,
            state.hierarchy_model,
            min_membership=0.0,
            m=2.0,
        )
        expected_contributions = dict(old_contributions)
        expected_contributions[1] = replacement_contributions[1]
        expected_statistics = _aggregate_center_contributions(
            expected_contributions
        )
        expected_model, _ = _update_hierarchy_centers_from_statistics(
            state.hierarchy_model,
            expected_statistics,
        )

        updated, summary = update_incremental_state(
            state,
            changed_embedding,
            changed_metadata,
        )

        self.assertFalse(summary["reclustered"])
        self.assertEqual(summary["replaced_samples"], 1)
        self.assertEqual(summary["appended_samples"], 0)
        self.assertEqual(len(updated.embeddings), len(state.embeddings))
        self.assertEqual(
            updated.metadata["id"].tolist(),
            state.metadata["id"].tolist(),
        )
        np.testing.assert_allclose(updated.embeddings[1], changed_embedding[0])
        np.testing.assert_allclose(
            updated.hierarchy_model.nodes[""].centers,
            expected_model.nodes[""].centers,
        )
        np.testing.assert_allclose(
            updated.center_statistics[""]["weighted_sum"],
            expected_statistics[""]["weighted_sum"],
        )
        self.assertEqual(
            set(updated.center_contributions),
            set(state.metadata["id"].tolist()),
        )
        self.assertEqual(
            updated.assignments["id"].tolist(),
            state.metadata["id"].tolist(),
        )
        self.assertEqual(updated.tree["summary"]["samples"], len(state.embeddings))
        self.assertEqual(updated.metadata.loc[1, "text"], "updated note")

    def test_xb_degradation_reclusters_without_revisualizing(self) -> None:
        state = self._make_state(
            center_refresh_interval=1,
            max_xb_relative_degradation=100.0,
        )
        first_batch = np.asarray(
            [[3.1, 0.4, 0.2, 0.0], [0.2, 3.2, 0.1, 0.0]],
            dtype=np.float64,
        )
        first_metadata = pd.DataFrame({"id": [100, 101]})
        first, first_summary = update_incremental_state(
            state,
            first_batch,
            first_metadata,
        )

        self.assertTrue(first_summary["membership_refreshed"])
        self.assertFalse(first_summary["reclustered"])
        self.assertEqual(first.config["membership_refreshes_since_recluster"], 1)
        self.assertEqual(len(first.assignments), len(first.embeddings))
        first.config["baseline_xie_beni"] = first.config["current_xie_beni"] * 0.1
        first.config["max_xb_relative_degradation"] = 0.05

        second_batch = np.asarray(
            [[2.9, 0.5, 0.2, 0.1], [0.3, 3.0, 0.0, 0.2]],
            dtype=np.float64,
        )
        second_metadata = pd.DataFrame({"id": [102, 103]})
        with patch(
            "incremental_clustering._fit_visualization",
        ) as fit_visualization:
            second, second_summary = update_incremental_state(
                first,
                second_batch,
                second_metadata,
            )

        self.assertTrue(second_summary["membership_refreshed"])
        self.assertTrue(second_summary["xb_degradation_recluster"])
        self.assertTrue(second_summary["reclustered"])
        self.assertFalse(second_summary["visualization_refitted"])
        fit_visualization.assert_not_called()
        np.testing.assert_array_equal(
            second.coordinates[: len(first.coordinates)],
            first.coordinates,
        )
        self.assertIs(second.visual_reducer, first.visual_reducer)
        self.assertEqual(second.config["membership_refreshes_since_recluster"], 0)
        self.assertEqual(second.config["total_reclusters"], 1)

    def test_default_noise_threshold_is_five_percent(self) -> None:
        self.assertEqual(DEFAULT_NOISE_THRESHOLD, 0.05)

    def test_noise_above_five_percent_reclusters_without_revisualizing(self) -> None:
        state = self._make_state(center_refresh_interval=10)
        state.config["noise_threshold"] = DEFAULT_NOISE_THRESHOLD
        previous_coordinates = state.coordinates.copy()
        batch = np.asarray(
            [[3.0, 0.8, 0.4, 0.0], [0.4, 3.0, 0.2, 0.1]],
            dtype=np.float64,
        )
        metadata = pd.DataFrame({"id": [200, 201]})
        forced_assignments, _ = assign_to_hierarchy(
            batch,
            metadata,
            state.hierarchy_model,
            min_membership=0.0,
            forced_noise_ratio=0.0,
        )
        forced_assignments["cluster"] = -1
        forced_assignments["is_noise"] = True
        forced_assignments["is_natural_noise"] = True
        forced_assignments["document_type"] = "noise"

        with (
            patch(
                "incremental_clustering.assign_to_hierarchy",
                return_value=(forced_assignments, 1.0),
            ),
            patch("incremental_clustering._fit_visualization") as fit_visualization,
        ):
            updated, summary = update_incremental_state(
                state,
                batch,
                metadata,
            )

        self.assertGreater(summary["new_noise_ratio"], 0.05)
        self.assertTrue(summary["emergency_recluster"])
        self.assertTrue(summary["reclustered"])
        self.assertFalse(summary["visualization_refitted"])
        fit_visualization.assert_not_called()
        np.testing.assert_array_equal(
            updated.coordinates[: len(previous_coordinates)],
            previous_coordinates,
        )

    def test_version_two_state_preserves_center_statistics(self) -> None:
        state = self._make_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pkl"
            save_state(state, path)
            loaded = load_state(path)
        self.assertEqual(loaded.center_statistics.keys(), {""})
        np.testing.assert_allclose(
            loaded.center_statistics[""]["weighted_sum"],
            state.center_statistics[""]["weighted_sum"],
        )

    def test_version_one_state_loads_without_center_statistics(self) -> None:
        state = self._make_state()
        payload = {
            "version": 1,
            "embeddings": state.embeddings,
            "metadata": state.metadata,
            "assignments": state.assignments,
            "coordinates": state.coordinates,
            "hierarchy_model": state.hierarchy_model,
            "tree": state.tree,
            "config": state.config,
            "visual_pca": state.visual_pca,
            "visual_reducer": state.visual_reducer,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version-one.pkl"
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            loaded = load_state(path)
        self.assertEqual(loaded.center_statistics, {})
        self.assertEqual(loaded.config["noise_threshold"], 0.05)
        self.assertEqual(
            loaded.config["recluster_trigger_policy"],
            "xb_and_noise_v2",
        )


if __name__ == "__main__":
    unittest.main()
