from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from synthetic_tag_fusion import (
    SyntheticTagConfig,
    build_additive_fusion,
    build_fusion_features,
    cluster_fusion_dataset,
    evaluate_soft_memberships,
    generate_synthetic_tag_dataset,
    shuffle_tag_embeddings,
)


class SyntheticTagFusionTests(unittest.TestCase):
    def test_generation_is_deterministic_and_memberships_are_valid(self) -> None:
        config = SyntheticTagConfig(
            n_samples=80,
            n_roots=6,
            embedding_dim=24,
            factor_dim=3,
            seed=7,
            tag_corruption=1.0,
        )
        first = generate_synthetic_tag_dataset(config)
        second = generate_synthetic_tag_dataset(config)

        np.testing.assert_array_equal(
            first.content_embeddings,
            second.content_embeddings,
        )
        np.testing.assert_array_equal(
            first.observed_tag_embeddings,
            second.observed_tag_embeddings,
        )
        np.testing.assert_array_equal(
            first.true_memberships,
            second.true_memberships,
        )
        pd.testing.assert_frame_equal(first.corruption_flags, second.corruption_flags)
        np.testing.assert_allclose(first.true_memberships.sum(axis=1), 1.0)
        np.testing.assert_allclose(
            np.linalg.norm(first.content_embeddings, axis=1),
            1.0,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertTrue(np.isfinite(first.observed_tag_embeddings).all())
        self.assertGreater(int(first.metadata["is_boundary"].sum()), 0)

    def test_root_vectors_have_correlated_groups(self) -> None:
        dataset = generate_synthetic_tag_dataset(
            SyntheticTagConfig(
                n_samples=40,
                n_roots=10,
                embedding_dim=64,
                factor_dim=5,
                seed=11,
            )
        )
        cosine = dataset.root_embeddings @ dataset.root_embeddings.T
        related = []
        unrelated = []
        for left in range(len(dataset.root_groups)):
            for right in range(left + 1, len(dataset.root_groups)):
                if dataset.root_groups[left] == dataset.root_groups[right]:
                    related.append(cosine[left, right])
                else:
                    unrelated.append(cosine[left, right])
        self.assertGreater(float(np.mean(related)), float(np.mean(unrelated)) + 0.15)
        self.assertGreater(float(np.mean(related)), 0.15)
        self.assertLess(float(np.mean(related)), 0.75)
        self.assertLess(float(np.mean(unrelated)), 0.25)

    def test_corruption_multiplier_zero_removes_corruption(self) -> None:
        base = dict(
            n_samples=160,
            n_roots=8,
            embedding_dim=24,
            factor_dim=4,
            seed=19,
        )
        clean = generate_synthetic_tag_dataset(
            SyntheticTagConfig(**base, tag_corruption=0.0)
        )
        corrupted = generate_synthetic_tag_dataset(
            SyntheticTagConfig(**base, tag_corruption=2.0)
        )
        self.assertEqual(int(clean.corruption_flags["is_corrupted"].sum()), 0)
        self.assertGreater(
            int(corrupted.corruption_flags["is_corrupted"].sum()),
            0,
        )
        self.assertGreaterEqual(
            int(corrupted.corruption_flags["missing_tag_count"].sum()),
            int(clean.corruption_flags["missing_tag_count"].sum()),
        )

    def test_fusion_variants_preserve_alignment_and_finite_values(self) -> None:
        dataset = generate_synthetic_tag_dataset(
            SyntheticTagConfig(
                n_samples=32,
                n_roots=4,
                embedding_dim=12,
                factor_dim=2,
                seed=23,
            )
        )
        additive = build_additive_fusion(
            dataset.content_embeddings,
            dataset.observed_tag_embeddings,
            tag_weight=0.5,
        )
        concatenated = build_fusion_features(
            dataset.content_embeddings,
            dataset.observed_tag_embeddings,
            variant="concat",
            tag_weight=0.5,
        )
        np.testing.assert_allclose(np.linalg.norm(additive, axis=1), 1.0)
        self.assertEqual(concatenated.shape, (32, 24))
        self.assertTrue(np.isfinite(concatenated).all())
        zero_tag = np.zeros_like(dataset.observed_tag_embeddings)
        np.testing.assert_allclose(
            build_additive_fusion(
                dataset.content_embeddings,
                zero_tag,
                tag_weight=2.0,
            ),
            dataset.content_embeddings,
        )

    def test_shuffled_tags_keep_marginal_rows(self) -> None:
        dataset = generate_synthetic_tag_dataset(
            SyntheticTagConfig(
                n_samples=20,
                n_roots=4,
                embedding_dim=10,
                factor_dim=2,
                seed=29,
            )
        )
        shuffled = shuffle_tag_embeddings(
            dataset.observed_tag_embeddings,
            seed=3,
        )
        self.assertEqual(shuffled.shape, dataset.observed_tag_embeddings.shape)
        np.testing.assert_array_equal(
            np.sort(shuffled, axis=0),
            np.sort(dataset.observed_tag_embeddings, axis=0),
        )

    def test_membership_metrics_match_permuted_clusters(self) -> None:
        rng = np.random.default_rng(31)
        truth = rng.dirichlet(np.ones(4), size=50)
        permutation = [2, 0, 3, 1]
        metrics, mapping = evaluate_soft_memberships(
            truth,
            truth[:, permutation],
        )
        self.assertEqual(set(mapping), set(range(4)))
        self.assertAlmostEqual(metrics["membership_cosine"], 1.0, places=12)
        self.assertAlmostEqual(metrics["membership_js_divergence"], 0.0, places=12)
        self.assertAlmostEqual(metrics["membership_mae"], 0.0, places=12)
        self.assertAlmostEqual(metrics["boundary_membership_mae"], 0.0, places=12)

    def test_fixed_k_spherical_fcm_smoke(self) -> None:
        dataset = generate_synthetic_tag_dataset(
            SyntheticTagConfig(
                n_samples=96,
                n_roots=4,
                embedding_dim=16,
                factor_dim=2,
                seed=37,
            )
        )
        result = cluster_fusion_dataset(
            dataset,
            variant="same_pca_additive",
            tag_weight=0.5,
            pca_components=8,
            seed=41,
            n_init=1,
            max_iter=60,
        )
        self.assertEqual(result.memberships.shape, (96, 4))
        np.testing.assert_allclose(result.memberships.sum(axis=1), 1.0)
        self.assertEqual(result.pca_components, 8)
        self.assertTrue(np.isfinite(result.projected).all())
        self.assertIn("boundary_membership_cosine", result.metrics)


if __name__ == "__main__":
    unittest.main()
