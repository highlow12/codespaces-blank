# Fast FCM pipeline

The `fast-fcm-pipeline` branch adds a bounded coarse-to-fine path to the
incremental fit command. It is enabled with `--fast` and keeps the regular
exhaustive selector as the default.

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
