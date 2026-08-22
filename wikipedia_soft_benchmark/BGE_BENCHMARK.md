# BGE hierarchy benchmark

The embedding stage reads `chunks.jsonl` (or a package directory/archive) and
writes row-aligned `chunk_embeddings.npy`/`chunk_metadata.jsonl` plus
`document_embeddings.npy`/`document_metadata.jsonl`.  It uses
`BAAI/bge-base-en-v1.5` at revision
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, CLS pooling, float32 vectors, and
L2 normalization.  A document vector is the L2-normalized mean of its chunks.

```bash
./.venv/bin/python wikipedia_bge_embeddings.py \
  --input wikipedia_artifacts/chunks.jsonl \
  --output-dir wikipedia_embeddings \
  --batch-size 16 --device cpu
```

The checkpoint records the canonical input SHA-256, model revision, pooling,
dimension, and completed rows.  An interrupted run can be resumed with
`--resume`; a changed input is rejected.

The hierarchy command fits PCA, UMAP, and leaf-HDBSCAN on discovery documents
only.  Calibration sweeps seeds `42,43,44`, cluster sizes `18,24,30`, sample
counts `3,5,8`, and exact-kNN widths `8,15,24`.  The selection score is the
mean of native and exact-kNN leaf NMI; ties use discovery noise, complexity,
and numeric configuration order.  Test rows are transformed only after the
selection is fixed.

```bash
./.venv/bin/python wikipedia_hierarchy_benchmark.py \
  --embedding-dir wikipedia_embeddings \
  --output-dir benchmarks/wikipedia-soft-bge-2026-08-22
```

The report contains leaf/parent/top NMI, ARI, coverage, mapped macro-F1 and
balanced accuracy, hierarchy distance, true affinity, and unexplained mass
for both membership methods.  `assignments.jsonl.gz` and `runs.csv` are
deterministic companion artifacts.  Generated embeddings and benchmark
outputs are ignored by Git.
