"""Focused tests for the three-route clustering benchmark."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_clustering_route_comparison as benchmark


class FakeUMAP:
    instances: list["FakeUMAP"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.embedding_: np.ndarray | None = None
        FakeUMAP.instances.append(self)

    def fit(self, values: np.ndarray) -> "FakeUMAP":
        components = int(self.kwargs["n_components"])
        base = np.asarray(values, dtype=np.float64)
        if base.shape[1] < components:
            base = np.pad(base, ((0, 0), (0, components - base.shape[1])))
        self.embedding_ = base[:, :components].copy()
        self.embedding_[:, 0] += float(self.kwargs["random_state"]) * 1e-5
        return self


class FakeHDBSCAN:
    instances: list["FakeHDBSCAN"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.labels_: np.ndarray | None = None
        self.probabilities_: np.ndarray | None = None
        self.outlier_scores_: np.ndarray | None = None
        self.cluster_persistence_: np.ndarray | None = None
        FakeHDBSCAN.instances.append(self)

    def fit(self, values: np.ndarray) -> "FakeHDBSCAN":
        n_samples = len(values)
        self.labels_ = np.arange(n_samples, dtype=np.int64) % 3
        self.probabilities_ = np.full(n_samples, 0.8, dtype=np.float64)
        self.outlier_scores_ = np.full(n_samples, 0.1, dtype=np.float64)
        self.cluster_persistence_ = np.asarray([0.8, 0.7, 0.6], dtype=np.float64)
        return self


def fake_native_memberships(clusterer: FakeHDBSCAN) -> np.ndarray:
    labels = np.asarray(clusterer.labels_, dtype=np.int64)
    memberships = np.zeros((len(labels), 3), dtype=np.float64)
    memberships[np.arange(len(labels)), labels] = 1.0
    return memberships


def make_metadata(n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class": [f"leaf-{index % 3}" for index in range(n_samples)],
            "class_hierarchy": [
                ["root", f"parent-{index % 2}", f"leaf-{index % 3}"]
                for index in range(n_samples)
            ],
        }
    )


class RouteHelperTests(unittest.TestCase):
    def test_medoid_and_hungarian_alignment(self) -> None:
        medoid_labels = np.asarray([0, 0, 1, 1, 2, 2])
        labels_by_seed = {
            42: medoid_labels,
            43: np.asarray([1, 1, 0, 0, 2, 2]),
            44: np.asarray([0, 0, 1, 2, 2, 2]),
        }
        medoid, scores = benchmark.choose_medoid_seed(labels_by_seed)
        self.assertEqual(medoid, 42)
        self.assertGreater(scores[42], scores[44])
        aligned, mapping = benchmark.align_labels_to_medoid(
            labels_by_seed[43], medoid_labels
        )
        np.testing.assert_array_equal(aligned, medoid_labels)
        self.assertEqual(mapping, {0: 1, 1: 0, 2: 2})

    def test_guard_score_and_gate_are_label_free_and_deterministic(self) -> None:
        labels = np.asarray([0, 0, -1])
        scores = benchmark.compute_guard_scores(
            labels,
            np.asarray([0.9, 0.2, 0.0]),
            np.asarray([0.1, 0.9, 1.0]),
            np.asarray([0.8]),
            np.asarray([0.9, 0.2, 0.7]),
        )
        self.assertAlmostEqual(scores[0], 0.88)
        self.assertLess(scores[1], benchmark.DEFAULT_GUARD_THRESHOLD)
        self.assertEqual(scores[2], 0.0)
        np.testing.assert_array_equal(
            benchmark.gate_labels(labels, scores, benchmark.DEFAULT_GUARD_THRESHOLD),
            np.asarray([0, -1, -1]),
        )

    def test_native_and_guarded_routes_share_one_seed42_fit(self) -> None:
        FakeUMAP.instances.clear()
        FakeHDBSCAN.instances.clear()
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(24, 8)).astype(np.float32)
        report = benchmark.run_route_comparison(
            embeddings,
            make_metadata(len(embeddings)),
            dataset_name="test_24",
            k_values=(2, 5, 10),
            pca_max_components=4,
            pca_min_components=2,
            pca_component_step=2,
            umap_class=FakeUMAP,
            hdbscan_class=FakeHDBSCAN,
            native_membership_function=fake_native_memberships,
        )
        self.assertEqual(len(FakeUMAP.instances), 5)
        self.assertEqual(len(FakeHDBSCAN.instances), 5)
        routes = {row["route"]: row for row in report["route_rows"]}
        self.assertEqual(routes[benchmark.ROUTE_NATIVE]["fit_count"], 1)
        self.assertEqual(routes[benchmark.ROUTE_GUARDED]["fit_count"], 1)
        self.assertEqual(routes[benchmark.ROUTE_STABLE]["fit_count"], 5)
        self.assertTrue(routes[benchmark.ROUTE_GUARDED]["fit_reuse_seed42"])
        self.assertEqual(len(report["seed_agreement_rows"]), 10)
        self.assertEqual(report["medoid"]["seed"], 42)

    def test_artifact_smoke_is_compact_and_writes_required_outputs(self) -> None:
        rng = np.random.default_rng(7)
        report = benchmark.run_route_comparison(
            rng.normal(size=(18, 6)).astype(np.float32),
            make_metadata(18),
            dataset_name="test_18",
            k_values=(2, 5, 10),
            pca_max_components=4,
            pca_min_components=2,
            pca_component_step=2,
            umap_class=FakeUMAP,
            hdbscan_class=FakeHDBSCAN,
            native_membership_function=fake_native_memberships,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            benchmark.write_artifacts(
                {
                    **{
                        "schema_version": 1,
                        "experiment": benchmark.EXPERIMENT_NAME,
                        "offline_research_benchmark": True,
                        "dataset": {
                            "main_sample_size": 18,
                        },
                        "datasets": [report],
                        "route_summary": report["route_rows"],
                        "runs": report["fit_rows"] + report["route_rows"],
                        "timing": report["timing_rows"],
                        "cluster_support": report["cluster_support_rows"],
                        "seed_agreement": report["seed_agreement_rows"],
                        "threshold_diagnostics": report["threshold_rows"],
                    }
                },
                output_dir,
            )
            required = (
                "report.json",
                "runs.csv",
                "route-summary.csv",
                "timing.csv",
                "cluster-support.csv",
                "seed-agreement.csv",
                "selected-pca.json",
                "quality-runtime-pareto.png",
                "stability-comparison.png",
                "pca-support-vs-seed-agreement.png",
                "rejection-quality-curve.png",
                "scale-runtime.png",
                "REPORT.md",
            )
            for filename in required:
                self.assertTrue((output_dir / filename).exists(), filename)
            saved = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
            serialized = json.dumps(saved)
            self.assertNotIn("coordinates", serialized)
            self.assertNotIn("embedding_vectors", serialized)
            self.assertNotIn("point_arrays", serialized)


if __name__ == "__main__":
    unittest.main()
