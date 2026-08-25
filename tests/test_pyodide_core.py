from __future__ import annotations

import unittest

import numpy as np

from pyodide_core.atomic_clustering import cluster_documents, dependency_status
from pyodide_core.atomic_clustering.hierarchy import build_hierarchy
from pyodide_core.atomic_clustering.pca import fit_pca


class PyodideCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = np.asarray(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.9, 0.2, 0.0, 0.0],
                [0.8, 0.1, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.9, 0.2],
                [0.1, 0.0, 0.8, 0.1],
            ],
            dtype=float,
        )

    def test_pca_auto_selection_is_serializable(self) -> None:
        selection = fit_pca(
            self.embeddings,
            min_components=2,
            component_step=1,
            k_values=(2,),
        )
        self.assertGreaterEqual(selection.selected_dimension, 2)
        self.assertEqual(selection.features.shape[0], len(self.embeddings))
        self.assertIn("candidates", selection.to_dict())

    def test_hierarchy_is_pandas_free_and_handles_noise(self) -> None:
        result = build_hierarchy(
            self.embeddings[:, :2],
            np.asarray([0, 0, -1, 1, 1, -1]),
            np.asarray(
                [
                    [0.8, 0.0],
                    [0.7, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.9],
                    [0.0, 0.8],
                    [0.0, 0.0],
                ]
            ),
        )
        self.assertEqual(result["tree"]["leaf_count"], 2)
        self.assertEqual(len(result["tree"]["merges"]), 1)
        self.assertEqual(result["assignments"]["bottom_up_k1"][2], -1)
        self.assertIsInstance(result["tree"]["merges"][0]["leaves"], list)

    def test_api_accepts_injected_discovery_and_returns_plain_values(self) -> None:
        def runner(features: np.ndarray, config: dict[str, object]) -> dict[str, object]:
            del config
            return {
                "umap_features": features[:, :2],
                "leaf_labels": np.asarray([0, 0, 0, 1, 1, 1]),
                "memberships": np.asarray(
                    [
                        [0.9, 0.0],
                        [0.8, 0.0],
                        [0.7, 0.0],
                        [0.0, 0.9],
                        [0.0, 0.8],
                        [0.0, 0.7],
                    ]
                ),
                "probabilities": np.ones(6),
                "outlier_scores": np.zeros(6),
                "configuration": {"runtime": "injected"},
            }

        result = cluster_documents(
            self.embeddings,
            ids=["a", "b", "c", "d", "e", "f"],
            config={"pca": {"min_components": 2, "component_step": 1, "k_values": (2,)}},
            discovery_runner=runner,
        )
        self.assertEqual(result["pipeline"], "pca_umap_hdbscan_bottom_up")
        self.assertEqual(result["ids"], ["a", "b", "c", "d", "e", "f"])
        self.assertEqual(result["discovery"]["cluster_count"], 2)
        self.assertEqual(result["hierarchy"]["tree"]["leaf_count"], 2)
        self.assertIsInstance(result["pca"]["features"], list)
        self.assertIsInstance(result["hierarchy"]["assignments"]["bottom_up_k1"], list)

    def test_dependency_probe_has_explicit_contract(self) -> None:
        status = dependency_status()
        self.assertIn("available", status)
        self.assertIn("runtime_note", status)
        self.assertIn("errors", status)


if __name__ == "__main__":
    unittest.main()
