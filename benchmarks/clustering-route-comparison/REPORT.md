# Three-route Gemini clustering benchmark

This is an offline research benchmark using Gemini `gemini-embedding-001` CLUSTERING embeddings. The existing hierarchical PCA SFCM production path remains unchanged.

## Protocol

All routes share one automatic PCA selection per sampled dataset. PCA is fit on L2-normalized embeddings; discovery UMAP receives the unnormalized selected PCA prefix. The baseline is the existing single-run UMAP → HDBSCAN discovery path: UMAP `(n_neighbors=15, n_components=20, min_dist=0.1, metric=euclidean, init=random, n_jobs=1)` and HDBSCAN `(min_cluster_size=5, min_samples=3, metric=euclidean, leaf, prediction_data=True)`.

- **A — `umap_hdbscan_native`**: one seed-42 fit and native HDBSCAN memberships.
- **B — `guarded_pca_hybrid`**: reuses A's fit, applies the fixed 0.45 guard score, and uses PCA exact-kNN memberships.
- **C — `five_seed_stable`**: five total fits for seeds 42–46, ARI medoid, Hungarian label alignment, fixed 0.60 consensus gate, and PCA exact-kNN memberships.

Guard tuning is label-free and pre-registered; the threshold curve is diagnostic, not a label-based selector. Raw↔PCA preservation is a baseline, not a fourth route.

## Main result

| Route | Leaf NMI | Leaf ARI | Parent NMI | Top NMI | Hierarchy distance | Noise | Clusters | Silhouette | Raw↔UMAP | PCA↔UMAP | Runtime (s) | Seed ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| umap_hdbscan_native | 0.6595 | 0.1978 | 0.6419 | 0.4821 | 0.8653 | 0.2444 | 54.00 | 0.5601 | 0.5719 | 0.5878 | 10.34 | 0.4199 |
| guarded_pca_hybrid | 0.6211 | 0.1202 | 0.6093 | 0.4559 | 1.0458 | 0.3125 | 54.00 | 0.6119 | 0.5719 | 0.5878 | 10.48 | 0.4199 |
| five_seed_stable | 0.6221 | 0.1265 | 0.6107 | 0.4627 | 1.0917 | 0.3292 | 46.00 | 0.6118 | 0.5723 | 0.5888 | 15.18 | 0.4199 |

## Main-data result-based conclusion

The documented quality-loss criterion is strict and data-driven: an alternative passes only when Leaf NMI loss, Leaf ARI loss, and hierarchy-distance increase versus the native baseline are each ≤ 0.02. For hierarchy distance, a positive delta is worse; runtime is reported separately and is not treated as a quality metric.

| Route | Leaf NMI | Δ Leaf NMI loss | Leaf ARI | Δ Leaf ARI loss | Hierarchy distance | Δ distance increase | Runtime (s) | Δ runtime (s) | Max quality loss | Criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| umap_hdbscan_native | 0.6595 | — | 0.1978 | — | 0.8653 | — | 10.34 | — | — | baseline |
| guarded_pca_hybrid | 0.6211 | 0.0384 | 0.1202 | 0.0776 | 1.0458 | 0.1806 | 10.48 | 0.14 | 0.1806 | VIOLATES >0.02 |
| five_seed_stable | 0.6221 | 0.0374 | 0.1265 | 0.0713 | 1.0917 | 0.2264 | 15.18 | 4.84 | 0.2264 | VIOLATES >0.02 |

Recommendation: retain `umap_hdbscan_native` as the baseline. Both `guarded_pca_hybrid` and `five_seed_stable` violate the documented ≤0.02 quality-loss criterion on the main dataset; their runtime differences do not override that quality failure.

## Scale rows

| Dataset | Route | Runtime (s) | Leaf NMI | Leaf ARI | Noise | Clusters |
|---|---|---:|---:|---:|---:|---:|
| scale_1500 | umap_hdbscan_native | 5.57 | 0.5903 | 0.0909 | 0.2947 | 95.00 |
| scale_1500 | guarded_pca_hybrid | 6.05 | 0.5597 | 0.0602 | 0.3607 | 95.00 |
| scale_1500 | five_seed_stable | 19.45 | 0.5343 | 0.0482 | 0.4227 | 88.00 |
| scale_3000 | umap_hdbscan_native | 16.91 | 0.5401 | 0.0435 | 0.3363 | 177.00 |
| scale_3000 | guarded_pca_hybrid | 18.59 | 0.4928 | 0.0189 | 0.4337 | 177.00 |
| scale_3000 | five_seed_stable | 59.24 | 0.4733 | 0.0185 | 0.4827 | 162.00 |

## Interpretation rule

A is the direct current discovery baseline. B is the low-overhead guarded candidate when its rejection improves trustworthiness without materially reducing quality. C is an offline stability/audit route; it is not placed in an online path because it intentionally costs five discovery fits. The experiment does not automatically replace production hierarchical PCA SFCM.

## Artifacts

- `report.json`: compact machine-readable report; no embeddings, coordinates, or point arrays.
- `runs.csv`: discovery-fit and route rows.
- `route-summary.csv`, `timing.csv`, `cluster-support.csv`, `seed-agreement.csv`, `selected-pca.json`.
- `quality-runtime-pareto.png`, `stability-comparison.png`, `pca-support-vs-seed-agreement.png`, `rejection-quality-curve.png`, `scale-runtime.png`.
