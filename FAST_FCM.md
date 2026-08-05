# Fast FCM pipeline

The bounded coarse-to-fine path is available from both the incremental fit
command and the default end-to-end pipeline. It is enabled with `--fast` and
keeps the regular exhaustive selector as the default.

```bash
python full_pipeline.py \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json \
  --output-dir results/full_pipeline_fast \
  --fast
```

The default fuzzifier search schedule is `m=[2.0, 1.8, 1.6, 1.4]`. Override
it with, for example, `--fast-m 2.2 2.0 1.8 1.6`. The selected value is saved
as `selected_fuzzifier` in `full_pipeline_summary.json` and is reused by the
optional incremental test.

```bash
python incremental_clustering.py fit \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json \
  --state-output results/fast.state.pkl \
  --assignments-output results/fast_assignments.csv \
  --coordinates-output results/fast_coordinates.csv \
  --tree-output results/fast_tree.json \
  --plot-output results/fast_scatter.png \
  --pca-components 192 \
  --max-depth 4 \
  --max-clusters 8 \
  --fast
```

For repeated clustering-only experiments, skip UMAP and fit it once for the
final candidate:

```bash
python incremental_clustering.py fit \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json \
  --state-output results/fast_scout.state.pkl \
  --assignments-output results/fast_scout_assignments.csv \
  --tree-output results/fast_scout_tree.json \
  --pca-components 192 \
  --fast \
  --skip-visualization
```

`--fast` samples each node for K scouting, probes `m` in descending order,
refines only the best K, and increases restart count only when stability is
below target. `--skip-visualization` creates a clustering-only state; it must
be refit without that flag before using incremental updates.

To run the fit on a reproducible random subset of the loaded dataset, add
`--dataset-sample-size` and optionally `--dataset-sample-seed`:

```bash
python incremental_clustering.py fit \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json \
  --dataset-sample-size 500 \
  --dataset-sample-seed 2026 \
  --state-output results/sample500.state.pkl \
  --pca-components 192 \
  --fast
```

This samples without replacement and preserves document IDs and metadata. The
sampling seed defaults to `--seed`. This is a dataset-level sample; it is
separate from `--fast-sample-size`, which controls per-node K scouting inside
the fast clustering algorithm.
