from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pyodide_core.atomic_clustering import cluster_documents


FIXTURE = Path(__file__).parent.parent / "atomic-clusters" / "tests" / "fixtures" / "python-wasm-golden.json"


class PyodideGoldenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_python_reference_matches_contract(self) -> None:
        fixture = self.fixture
        expected = fixture["expected"]
        embeddings = np.asarray(fixture["embeddings"], dtype=float)
        discovery = expected["discovery"]

        def runner(_features: np.ndarray, _config: dict[str, object]) -> dict[str, object]:
            return discovery

        result = cluster_documents(embeddings, ids=fixture["ids"], config=fixture["config"], discovery_runner=runner)
        self.assertEqual(result["ids"], fixture["ids"])
        self.assertEqual(result["pca"]["selected_dimension"], expected["pca"]["selected_dimension"])
        self.assertEqual(result["pca"]["fitted_dimension"], expected["pca"]["fitted_dimension"])
        self.assertEqual(result["pca"]["selection_reason"], expected["pca"]["selection_reason"])
        self.assertAlmostEqual(
            result["pca"]["candidates"][0]["cumulative_explained_variance"],
            expected["pca"]["cumulative_explained_variance"],
            delta=fixture["tolerances"]["pca_abs"],
        )
        self.assertEqual(result["discovery"]["leaf_labels"], discovery["leaf_labels"])
        self.assertEqual(result["discovery"]["cluster_count"], discovery["cluster_count"])
        np.testing.assert_allclose(result["discovery"]["umap_features"], discovery["umap_features"], atol=fixture["tolerances"]["umap_abs"])
        np.testing.assert_allclose(result["discovery"]["probabilities"], discovery["probabilities"], atol=fixture["tolerances"]["hdbscan_abs"])
        self.assertEqual(result["hierarchy"]["tree"]["leaf_count"], expected["hierarchy"]["leaf_count"])
        self.assertEqual(result["hierarchy"]["summary"]["merge_count"], expected["hierarchy"]["merge_count"])
        self.assertEqual(result["hierarchy"]["assignments"]["bottom_up_k1"], expected["hierarchy"]["bottom_up_k1"])

    def test_contract_rejects_bad_discovery_shape(self) -> None:
        fixture = self.fixture
        embeddings = np.asarray(fixture["embeddings"], dtype=float)

        def runner(_features: np.ndarray, _config: dict[str, object]) -> dict[str, object]:
            output = dict(fixture["expected"]["discovery"])
            output["leaf_labels"] = output["leaf_labels"][:-1]
            return output

        with self.assertRaises(ValueError):
            cluster_documents(embeddings, config=fixture["config"], discovery_runner=runner)


if __name__ == "__main__":
    unittest.main()
