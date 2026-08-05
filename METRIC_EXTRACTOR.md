# Metric-only clustering evaluator

`extract_clustering_metrics.py` reads saved assignment CSVs and does not fit
PCA, FCM, or any other clustering model. It can evaluate one condition or
several conditions in one call:

```bash
python extract_clustering_metrics.py \
  --assignments \
    /path/to/assignments_content_only.csv \
    /path/to/assignments_content_plus_tag.csv \
  --features \
    /path/to/content_features.npy \
    /path/to/content_plus_tag_features.npy \
  --output-csv clustering_metrics.csv
```

`--features` is optional. Without it, external metrics and saved-membership
metrics are still available. With it, the tool calculates silhouette and XB
in that supplied feature space. To reproduce post-PCA metrics exactly, pass
the saved post-PCA feature matrix. An embedding JSON is accepted as a feature
source and is evaluated in its raw embedding space.

The existing in-memory pipeline passes its fitted centers to the same metric
core, so its XB and fuzzy-silhouette values use exact model centers. The
standalone CSV tool derives centers from saved memberships unless centers are
made available separately.

The assignment CSV may include `class` and `class_hierarchy` columns for
external evaluation, and `membership_0`, `membership_1`, ... columns for PC,
modified PC, PE, and normalized PE. If target columns are absent, use
`--metadata-json` to align them by `id` (or by row order when no IDs exist).

Metric direction:

- NMI, ARI, silhouette, PC: higher is better.
- XB, PE, normalized PE: lower is better.

The tool reports `fits_clustering_model: false` and does not use ground-truth
labels for fitting.
