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

The default plugin runtime has no runtime package installer and uses the bundled
WASM worker. An optional Pyodide worker runs the Python reference
`cluster_documents` API and loads the pinned Pyodide runtime only when selected
in Settings. Gemini is
an explicit network provider using Obsidian `requestUrl`; the API key is
resolved by a SecretStorage reference, never persisted in plugin settings.
The local `multilingual-e5-small` provider is opt-in. Settings provides an
explicit consent dialog to download the ONNX model and tokenizer, stores them
under the plugin model directory with a versioned SHA-256 manifest, and offers
deletion. After installation, embedding uses only the local model and the
bundled `onnxruntime-web/wasm` runtime plus tokenizer; no network request is
made by `embed()`.

The Pyodide worker embeds the `pyodide_core` source package in the release
bundle, loads NumPy and scikit-learn, fits the authoritative Python PCA, and
injects the browser UMAP/WASM-HDBSCAN discovery result through the documented
`discovery_runner` seam. This preserves the Python membership-weighted
hierarchy while keeping UMAP/HDBSCAN native-extension packages out of Pyodide.

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

Copy `dist/main.js`, `dist/manifest.json`, `dist/styles.css`, and the two
`dist/ort-wasm-simd-threaded.*` assets to `.obsidian/plugins/atomic-clusters/`
for a local install. The worker source and Python core are embedded in
`main.js`; the ORT assets stay adjacent so the bundled local provider can load
them without document-relative URL assumptions.
The repository intentionally keeps the model weights out of the plugin; local
model download must be explicit and is disclosed in Settings.

## Scope and known gaps

This MVP implements the complete vault → embedding → worker → hierarchy →
explorer path and cache invalidation by content hash. The local ONNX runtime is
bundled and configured beside the CommonJS plugin bundle; an explicit injection
override remains available for alternate execution providers. Model download
and integrity management have no hidden runtime downloads. The deterministic
density-graph fallback is suitable for fixtures and small development vaults,
but should not be used as a scientific parity benchmark.
