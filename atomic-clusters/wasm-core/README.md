# Atomic Clusters WASM core

This crate contains the CPU-heavy operations for the desktop-only Obsidian
plugin. It has no filesystem or network access. Matrix inputs and outputs are
flat row-major `f32` buffers, which wasm-bindgen exposes as typed-array
compatible values.

## Exports

- `normalize` and `matmul`
- `randomized_pca` / `pca`: deterministic randomized PCA/SVD. The result has
  flat `projected`, `basis`, `mean`, and `explained` fields.
- `cosine_distances_tiled`: dense diagnostic path; `exact_knn_cosine_tiled`:
  tiled O(nk)-output exact cosine kNN production path.
- `HnswIndex(points, count, dimension, m, seed)`: deterministic multi-layer
  HNSW with `search` and `search_with_ef`.
- `mst` for small dense fixtures and `mutual_reachability_mst` for the sparse
  HDBSCAN kNN graph.

The API rejects non-finite values, invalid dimensions, invalid k, and invalid
graph indices with JavaScript errors before allocating output. Zero vectors
remain zero on normalization.

## Build and test

Rust is not assumed to be installed in this repository. Run these on the
release/build machine:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.100
cd atomic-clusters/wasm-core
cargo test
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir ../src/engine/generated \
  target/wasm32-unknown-unknown/release/atomic_clusters_wasm_core.wasm
```

Bundle the generated JS/WASM from the plugin build; never download it at
runtime. Calls are synchronous. The worker supplies cancellation by terminating
between calls; the `tile` argument is its available progress boundary.

`cosine_distances_tiled` and dense `mst` materialize O(n²) data and are only
for small diagnostics. Normal clustering uses exact/HNSW kNN followed by the
sparse mutual-reachability MST.
