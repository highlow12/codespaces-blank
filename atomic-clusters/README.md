# Atomic Clusters

Atomic Clusters is an Obsidian desktop-only plugin that builds an offline,
hierarchical map of Markdown notes. It provides the commands **Build note
clusters**, **Open cluster explorer**, and **Cancel clustering**.

한국어 개발 현황, 아키텍처, 현재 기본값, PCA 보정 이력, Python/WASM 정렬 기준,
검증 수치와 다음 단계는 [개발 현황 문서](docs/development-status.md)에 정리되어
있다.

## Runtime design

The plugin reads Markdown through the Obsidian Vault API, fingerprints content,
and stores provider/model-scoped embedding cache records under the plugin data
directory. Clustering is lazy and runs in a Node `worker_thread`:

```text
normalized embeddings → sampled automatic PCA → umap-js → HDBSCAN provider → bottom-up hierarchy
```

The plugin has no runtime package installer and does not use Pyodide. Gemini is
an explicit network provider using Obsidian `requestUrl`; the API key is
resolved by a SecretStorage reference, never persisted in plugin settings.
The local `multilingual-e5-small` provider is an opt-in boundary and download
UI; a future release supplies the bundled ONNX runner/model asset.

The numerical boundary is `wasm-core/`. Its Rust/wasm-bindgen kernels own
normalization, matrix multiplication, PCA/power iteration, tiled cosine
distances, exact kNN, an HNSW-compatible index, and MST processing for the
HDBSCAN graph. `HdbscanProvider` is the swap point for a hdbscan-rs
wasm-bindgen provider when that external crate is vendored and audited. The
current checked-in MVP uses the deterministic TypeScript
kernel when the generated WASM asset is not present, so development and tests
remain runnable. This fallback is not a claim of HDBSCAN parity; production
builds should generate and load the WASM module before enabling large vaults.

Build the numerical asset from a Rust toolchain with:

```bash
wasm-pack build wasm-core --target web --release --out-dir ../src/wasm
```

Then expose the generated exports as `globalThis.__ATOMIC_CLUSTERS_WASM__`
before the worker is initialized (the `src/wasm-loader.ts` hook). No network is
required by the packaged plugin after those assets are bundled. When present,
the worker routes normalization, PCA, tiled distances, kNN, and MST/HDBSCAN
graph work through this module.

## Development

```bash
npm install
npm run build
npm test
```

Run an offline end-to-end check against the checked-in 3,000-record Gemini
embedding dataset. This exercises the same `clusterEmbeddings` orchestration
as the worker and uses the packaged Rust/WASM kernels when `wasm-core/pkg` is
present; it never calls Gemini or performs local embedding inference:

```bash
npm run validate:offline -- --dataset-sample-size 100 --dataset-sample-seed 42 --fast
```

The report is written outside the source tree under `/tmp` by default. Supply
`--output PATH` to choose a destination. The report includes load and
orchestration timings, WASM status, PCA selection, cluster/noise counts,
probability/outlier summaries, hierarchy counts, progress events, and row
assignments. The runner accepts either the plain `.json` or gzip `.json.gz`
Gemini dataset and deliberately refuses `dbpedia_label_embeddings.json`.

Copy `dist/main.js`, `dist/manifest.json`, and `dist/styles.css` to
`.obsidian/plugins/atomic-clusters/` for a local install. The worker source is
embedded in `main.js`, so Community releases contain only the three required
plugin files.
The repository intentionally keeps the model weights out of the plugin; local
model download must be explicit and is disclosed in Settings.

## Scope and known gaps

This MVP implements the complete vault → embedding → worker → hierarchy →
explorer path and cache invalidation by content hash. HDBSCAN's production
WASM binding and local ONNX inference asset are integration points, not hidden
runtime downloads. Until those assets are built and loaded, the deterministic
density-graph fallback is suitable for fixtures and small development vaults,
but should not be used as a scientific parity benchmark.
