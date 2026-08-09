from __future__ import annotations

import copy
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
    CENTER_CONTRIBUTION_FORMAT,
    DEFAULT_NOISE_THRESHOLD,
    RECLUSTER_TRIGGER_POLICY,
    STATE_VERSION,
    IncrementalClusterState,
    _aggregate_center_contributions,
    _center_contributions_for_batch,
    _evaluate_noise_drift,
    _hierarchy_xb_from_contributions,
    _initialize_update_config,
    _rebuild_tree_counts,
    _select_center_affected_ids,
    _snapshot_hierarchy_centers,
    _update_hierarchy_centers_from_statistics,
    assign_to_hierarchy,
    fit_incremental_state,
    hierarchy_xie_beni_index,
    load_state,
    save_state,
    update_incremental_state,
)
from pca_projection import calibrate_pca_projection_support_threshold


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
        center_contributions = _center_contributions_for_batch(
            self.X,
            self.metadata,
            model,
            min_membership=0.0,
            m=2.0,
        )
        config["center_contribution_format"] = CENTER_CONTRIBUTION_FORMAT
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
            center_statistics=_aggregate_center_contributions(
                center_contributions
            ),
            center_contributions=center_contributions,
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
        self.assertGreater(summary["center_movement_max"], 0.0)
        self.assertGreaterEqual(summary["cluster_occupancy_change"], 0.0)
        self.assertGreaterEqual(summary["assignment_change_rate"], 0.0)
        self.assertEqual(summary["compared_assignment_count"], len(self.X))
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

    def test_float32_state_storage_survives_update_and_reload(self) -> None:
        state = self._make_state(center_refresh_interval=5)
        state.embeddings = state.embeddings.astype(np.float32)
        state.config["embedding_storage_dtype"] = "float32"
        batch = np.asarray(
            [[3.0, 0.8, 0.4, 0.0], [2.7, 0.9, 0.2, 0.1]],
            dtype=np.float64,
        )
        metadata = pd.DataFrame({"id": [100, 101]})

        updated, summary = update_incremental_state(
            state,
            batch,
            metadata,
        )

        self.assertEqual(updated.embeddings.dtype, np.dtype(np.float32))
        self.assertEqual(updated.config["embedding_storage_dtype"], "float32")
        self.assertTrue(summary["center_updated"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float32.state.pkl"
            save_state(updated, path)
            loaded = load_state(path)
        self.assertEqual(loaded.embeddings.dtype, np.dtype(np.float32))

    def test_fit_defaults_to_float32_embedding_storage(self) -> None:
        state = fit_incremental_state(
            self.X,
            self.metadata,
            max_depth=1,
            min_node_size=4,
            min_child_size=2,
            min_clusters=2,
            max_clusters=2,
            min_membership=0.0,
            selection_method="silhouette",
            min_split_silhouette=-1.0,
            pca_components=3,
            fit_visualization=False,
        )

        self.assertEqual(state.embeddings.dtype, np.dtype(np.float32))
        self.assertEqual(state.config["embedding_storage_dtype"], "float32")

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

    def test_compact_contributions_apply_only_batch_deltas(self) -> None:
        state = self._make_state(center_refresh_interval=50)
        for contribution in state.center_contributions.values():
            self.assertEqual(
                set(contribution),
                {"projected", "weights_by_path"},
            )
            self.assertNotIn("weighted_sum", contribution)

        batch = np.asarray(
            [[3.0, 0.6, 0.2, 0.0], [0.2, 3.1, 0.1, 0.1]],
            dtype=np.float64,
        )
        metadata = pd.DataFrame({"id": [100, 101]})
        with patch(
            "incremental_clustering._aggregate_center_contributions",
        ) as aggregate:
            updated, summary = update_incremental_state(state, batch, metadata)

        aggregate.assert_not_called()
        self.assertFalse(summary["membership_refreshed"])
        self.assertIs(
            updated.center_contributions[0],
            state.center_contributions[0],
        )
        expected = _aggregate_center_contributions(updated.center_contributions)
        np.testing.assert_allclose(
            updated.center_statistics[""]["weighted_sum"],
            expected[""]["weighted_sum"],
        )
        np.testing.assert_allclose(
            updated.center_statistics[""]["weight"],
            expected[""]["weight"],
        )

    def test_compact_delta_statistics_remain_consistent_across_many_batches(self) -> None:
        state = self._make_state(center_refresh_interval=100)
        for index in range(20):
            if index % 2 == 0:
                embedding = np.asarray([[3.0, 0.4, 0.1, 0.0]])
            else:
                embedding = np.asarray([[0.1, 3.0, 0.2, 0.0]])
            state, summary = update_incremental_state(
                state,
                embedding,
                pd.DataFrame({"id": [100 + index]}),
            )
            self.assertFalse(summary["membership_refreshed"])

        expected = _aggregate_center_contributions(state.center_contributions)
        np.testing.assert_allclose(
            state.center_statistics[""]["weighted_sum"],
            expected[""]["weighted_sum"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            state.center_statistics[""]["weight"],
            expected[""]["weight"],
            atol=1e-12,
        )
        self.assertEqual(len(state.center_contributions), len(self.X) + 20)

    def test_legacy_outer_product_contributions_migrate_on_update(self) -> None:
        state = self._make_state(center_refresh_interval=50)
        legacy: dict[object, dict[str, dict[str, np.ndarray]]] = {}
        for identifier, contribution in state.center_contributions.items():
            projected = contribution["projected"]
            legacy[identifier] = {
                path: {
                    "weighted_sum": np.outer(weights, projected),
                    "weight": weights.copy(),
                }
                for path, weights in contribution["weights_by_path"].items()
            }
        state.center_contributions = legacy
        state.config.pop("center_contribution_format")

        updated, _summary = update_incremental_state(
            state,
            np.asarray([[3.1, 0.4, 0.2, 0.0]]),
            pd.DataFrame({"id": [100]}),
        )

        self.assertEqual(
            updated.config["center_contribution_format"],
            CENTER_CONTRIBUTION_FORMAT,
        )
        self.assertTrue(
            all(
                "projected" in contribution
                and "weights_by_path" in contribution
                for contribution in updated.center_contributions.values()
            )
        )

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
        self.assertEqual(
            first_summary["membership_refresh_scope"],
            "full_legacy",
        )
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

    def test_center_influence_selects_only_nearby_weighted_notes(self) -> None:
        state = self._make_state()
        current_model = copy.deepcopy(state.hierarchy_model)
        reference_centers = _snapshot_hierarchy_centers(current_model)
        moved = current_model.nodes[""].centers[0].copy()
        moved[0] += 0.25
        moved /= np.linalg.norm(moved)
        current_model.nodes[""].centers[0] = moved
        movement = float(
            np.linalg.norm(moved - reference_centers[""][0])
        )
        contributions = {
            "near": {
                "projected": np.zeros(3),
                "weights_by_path": {"": np.asarray([0.81, 0.01])},
            },
            "far": {
                "projected": np.zeros(3),
                "weights_by_path": {"": np.asarray([0.01, 0.81])},
            },
        }

        selected, affected_paths, diagnostics = _select_center_affected_ids(
            ["near", "far", "incoming"],
            contributions,
            reference_centers,
            current_model,
            always_include={"incoming"},
            min_center_movement=movement * 0.5,
            min_influence=movement * 0.5,
        )

        self.assertEqual(selected, ["near", "incoming"])
        self.assertEqual(affected_paths, {""})
        self.assertEqual(diagnostics["affected_center_cluster_count"], 1)

    def test_selective_refresh_never_recomputes_all_note_memberships(self) -> None:
        state = self._make_state(
            center_refresh_interval=1,
            max_xb_relative_degradation=1e12,
        )
        state.config.update(
            {
                "selective_membership_refresh": True,
                "membership_refresh_min_center_movement": 1.0,
                "membership_refresh_min_influence": 1.0,
            }
        )
        state.membership_reference_centers = _snapshot_hierarchy_centers(
            state.hierarchy_model
        )
        batch = np.asarray(
            [[3.1, 0.4, 0.2, 0.0], [0.2, 3.2, 0.1, 0.0]],
            dtype=np.float64,
        )
        metadata = pd.DataFrame({"id": [100, 101]})

        with (
            patch(
                "incremental_clustering.assign_to_hierarchy",
                wraps=assign_to_hierarchy,
            ) as assign,
            patch(
                "incremental_clustering._refresh_distance_thresholds",
            ) as full_threshold_refresh,
        ):
            updated, summary = update_incremental_state(
                state,
                batch,
                metadata,
            )

        membership_batch_sizes = [len(call.args[0]) for call in assign.call_args_list]
        self.assertEqual(membership_batch_sizes, [2, 2])
        full_threshold_refresh.assert_not_called()
        self.assertEqual(summary["membership_refresh_scope"], "selective")
        self.assertEqual(summary["membership_refresh_sample_count"], 2)
        self.assertEqual(summary["membership_refresh_skipped_count"], len(self.X))
        self.assertIs(
            updated.center_contributions[0],
            state.center_contributions[0],
        )

    def test_contribution_xb_matches_exact_xb_for_fresh_weights(self) -> None:
        state = self._make_state()
        approximate = _hierarchy_xb_from_contributions(
            state.hierarchy_model,
            state.center_contributions,
        )
        exact = hierarchy_xie_beni_index(
            state.embeddings,
            state.hierarchy_model,
            min_membership=0.0,
            m=2.0,
        )
        self.assertAlmostEqual(approximate, exact, places=12)

    def test_update_is_idempotent_by_batch_id(self) -> None:
        state = self._make_state(center_refresh_interval=50)
        batch = np.asarray(
            [[3.1, 0.4, 0.2, 0.0], [0.2, 3.2, 0.1, 0.0]],
            dtype=np.float64,
        )
        metadata = pd.DataFrame({"id": [100, 101]})

        updated, first_summary = update_incremental_state(
            state,
            batch,
            metadata,
            batch_id="hierarchical-batch-1",
        )
        replayed, replay_summary = update_incremental_state(
            updated,
            batch,
            metadata,
            batch_id="hierarchical-batch-1",
        )

        self.assertFalse(first_summary["idempotent_replay"])
        self.assertTrue(replay_summary["idempotent_replay"])
        self.assertIs(replayed, updated)
        self.assertEqual(updated.config["state_generation"], 1)
        self.assertEqual(replay_summary["generation"], 1)
        self.assertEqual(len(updated.embeddings), len(state.embeddings) + 2)

        changed_batch = batch.copy()
        changed_batch[0, 0] += 0.2
        with self.assertRaisesRegex(ValueError, "different content"):
            update_incremental_state(
                updated,
                changed_batch,
                metadata,
                batch_id="hierarchical-batch-1",
            )

    def test_state_checksum_rejects_tampering(self) -> None:
        state = self._make_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pkl"
            save_state(state, path)
            with path.open("rb") as handle:
                envelope = pickle.load(handle)
            envelope["checksum"] = "0" * 64
            with path.open("wb") as handle:
                pickle.dump(envelope, handle)

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_state(path)

    def test_default_noise_threshold_is_five_percent(self) -> None:
        self.assertEqual(DEFAULT_NOISE_THRESHOLD, 0.05)

    def test_projection_outlier_is_natural_noise_without_forced_quota(self) -> None:
        rng = np.random.default_rng(37)
        training = np.zeros((40, 8), dtype=np.float64)
        training[:, :2] = rng.normal(size=(40, 2))
        projected, pca = fit_pca_normalized_features(
            training,
            n_components=2,
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
            projection_support_threshold=(
                calibrate_pca_projection_support_threshold(training, pca)
            ),
        )
        orthogonal = np.zeros(8, dtype=np.float64)
        orthogonal[-1] = 1.0
        batch = np.vstack([training[0], orthogonal])

        assignments, noise_ratio = assign_to_hierarchy(
            batch,
            pd.DataFrame({"id": ["known", "unknown"]}),
            model,
            min_membership=0.0,
        )

        known = assignments.loc[assignments["id"] == "known"].iloc[0]
        unknown = assignments.loc[assignments["id"] == "unknown"].iloc[0]
        self.assertFalse(bool(known["is_noise"]))
        self.assertTrue(bool(unknown["is_projection_outlier"]))
        self.assertTrue(bool(unknown["is_natural_noise"]))
        self.assertFalse(bool(unknown["is_forced_noise"]))
        self.assertEqual(unknown["cluster_path"], "noise")
        self.assertEqual(noise_ratio, 0.5)

        all_outliers, all_outlier_ratio = assign_to_hierarchy(
            np.vstack([orthogonal, orthogonal]),
            pd.DataFrame({"id": ["unknown-1", "unknown-2"]}),
            model,
            min_membership=0.0,
        )
        self.assertTrue(all_outliers["is_natural_noise"].all())
        self.assertFalse(all_outliers["is_forced_noise"].any())
        self.assertEqual(all_outlier_ratio, 1.0)

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

        self.assertGreater(summary["new_natural_noise_ratio"], 0.05)
        self.assertTrue(summary["emergency_recluster"])
        self.assertTrue(summary["reclustered"])
        self.assertFalse(summary["visualization_refitted"])
        fit_visualization.assert_not_called()
        np.testing.assert_array_equal(
            updated.coordinates[: len(previous_coordinates)],
            previous_coordinates,
        )

    def test_small_abrupt_batches_accumulate_before_drift_evaluation(self) -> None:
        config = _initialize_update_config(
            {
                "noise_threshold": 0.50,
                "noise_release_threshold": 0.20,
                "drift_min_samples": 10,
                "drift_ewma_alpha": 0.50,
                "recluster_cooldown_updates": 2,
                "recluster_trigger_policy": RECLUSTER_TRIGGER_POLICY,
            }
        )

        first = _evaluate_noise_drift(
            config,
            natural_noise_count=4,
            sample_count=5,
        )
        second = _evaluate_noise_drift(
            config,
            natural_noise_count=4,
            sample_count=5,
        )

        self.assertFalse(first["evaluated"])
        self.assertEqual(first["pending_samples"], 5)
        self.assertTrue(second["evaluated"])
        self.assertEqual(second["evaluation_samples"], 10)
        self.assertAlmostEqual(second["observed_ratio"], 0.8)
        self.assertTrue(second["alarm_active"])

        still_active = _evaluate_noise_drift(
            config,
            natural_noise_count=0,
            sample_count=10,
        )
        released = _evaluate_noise_drift(
            config,
            natural_noise_count=0,
            sample_count=10,
        )
        self.assertTrue(still_active["alarm_active"])
        self.assertFalse(released["alarm_active"])

    def test_gradual_drift_crosses_ewma_threshold_over_time(self) -> None:
        config = _initialize_update_config(
            {
                "noise_threshold": 0.50,
                "noise_release_threshold": 0.20,
                "drift_min_samples": 10,
                "drift_ewma_alpha": 0.50,
                "recluster_cooldown_updates": 2,
                "recluster_trigger_policy": RECLUSTER_TRIGGER_POLICY,
            }
        )

        results = [
            _evaluate_noise_drift(
                config,
                natural_noise_count=noise_count,
                sample_count=10,
            )
            for noise_count in (1, 3, 5, 7)
        ]

        self.assertEqual(
            [round(result["smoothed_ratio"], 3) for result in results],
            [0.1, 0.2, 0.35, 0.525],
        )
        self.assertEqual(
            [result["alarm_active"] for result in results],
            [False, False, False, True],
        )

    def test_recluster_cooldown_suppresses_repeated_noise_trigger(self) -> None:
        state = self._make_state(center_refresh_interval=10)
        state.config.update(
            {
                "noise_threshold": DEFAULT_NOISE_THRESHOLD,
                "noise_release_threshold": 0.02,
                "drift_min_samples": 2,
                "drift_ewma_alpha": 1.0,
                "recluster_cooldown_updates": 2,
                "recluster_trigger_policy": RECLUSTER_TRIGGER_POLICY,
            }
        )
        summaries = []
        for update_index in range(4):
            batch = np.asarray(
                [[3.0, 0.8, 0.4, 0.0], [0.4, 3.0, 0.2, 0.1]],
                dtype=np.float64,
            )
            metadata = pd.DataFrame(
                {"id": [300 + update_index * 2, 301 + update_index * 2]}
            )
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
            with patch(
                "incremental_clustering.assign_to_hierarchy",
                return_value=(forced_assignments, 1.0),
            ):
                state, summary = update_incremental_state(
                    state,
                    batch,
                    metadata,
                )
            summaries.append(summary)

        self.assertTrue(summaries[0]["reclustered"])
        self.assertTrue(summaries[1]["recluster_suppressed_by_cooldown"])
        self.assertTrue(summaries[2]["recluster_suppressed_by_cooldown"])
        self.assertFalse(summaries[1]["reclustered"])
        self.assertFalse(summaries[2]["reclustered"])
        self.assertTrue(summaries[3]["reclustered"])

    def test_current_state_version_preserves_compact_center_statistics(self) -> None:
        state = self._make_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pkl"
            save_state(state, path)
            with path.open("rb") as handle:
                envelope = pickle.load(handle)
            payload = pickle.loads(envelope["payload_bytes"])
            loaded = load_state(path)
        self.assertEqual(envelope["format"], "incremental_state_envelope_v1")
        self.assertEqual(payload["version"], STATE_VERSION)
        self.assertEqual(loaded.center_statistics.keys(), {""})
        self.assertEqual(
            loaded.config["center_contribution_format"],
            CENTER_CONTRIBUTION_FORMAT,
        )
        np.testing.assert_allclose(
            loaded.center_statistics[""]["weighted_sum"],
            state.center_statistics[""]["weighted_sum"],
        )
        self.assertEqual(loaded.membership_reference_centers.keys(), {""})

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
        self.assertEqual(loaded.config["noise_threshold"], 1.0)
        self.assertEqual(loaded.config["drift_min_samples"], 1)
        self.assertEqual(loaded.config["drift_ewma_alpha"], 1.0)
        self.assertEqual(loaded.config["recluster_cooldown_updates"], 0)
        self.assertFalse(loaded.config["selective_membership_refresh"])
        self.assertEqual(loaded.membership_reference_centers.keys(), {""})
        self.assertEqual(
            loaded.config["recluster_trigger_policy"],
            RECLUSTER_TRIGGER_POLICY,
        )

    def test_version_five_state_keeps_full_membership_refresh(self) -> None:
        state = self._make_state()
        legacy_config = dict(state.config)
        legacy_config["recluster_trigger_policy"] = RECLUSTER_TRIGGER_POLICY
        legacy_config.pop("selective_membership_refresh", None)
        payload = {
            "version": 5,
            "embeddings": state.embeddings,
            "metadata": state.metadata,
            "assignments": state.assignments,
            "coordinates": state.coordinates,
            "hierarchy_model": state.hierarchy_model,
            "tree": state.tree,
            "config": legacy_config,
            "visual_pca": state.visual_pca,
            "visual_reducer": state.visual_reducer,
            "center_statistics": state.center_statistics,
            "center_contributions": state.center_contributions,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version-five.pkl"
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            loaded = load_state(path)

        self.assertFalse(loaded.config["selective_membership_refresh"])
        self.assertEqual(loaded.membership_reference_centers.keys(), {""})


if __name__ == "__main__":
    unittest.main()
