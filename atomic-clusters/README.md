# Atomic Clusters

Atomic Clusters is an Obsidian desktop-only plugin that builds an offline,
hierarchical map of Markdown notes. It provides the commands **Build note
clusters**, **Refresh changed notes**, **Rebuild all clusters**, **Pause automatic
refresh**, **Open cluster explorer**, and **Cancel clustering**.

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

The plugin runtime has no runtime package installer and uses the bundled WASM
worker (with the existing Node/Chromium/in-process fallbacks). The Python
reference remains in the sibling `pyodide_core/` package for algorithm parity
and audits; it is not loaded or bundled into the Obsidian plugin. Gemini is
an explicit network provider using Obsidian `requestUrl`; the API key is
resolved by a SecretStorage reference, never persisted in plugin settings.
The local `multilingual-e5-small` provider is opt-in. Settings provides an
explicit consent dialog to download the ONNX model and tokenizer, stores them
under the plugin model directory with a versioned SHA-256 manifest, and offers
deletion. After installation, embedding uses only the local model, bundled
onnxruntime-web WASM/WebGPU (JSEP) assets, and tokenizer; no network request is
made by `embed()`. The local backend setting defaults to Auto: it tries WebGPU
first and falls back to the WASM CPU backend when WebGPU is unavailable or
session creation fails. The selected backend or fallback reason appears in
preflight progress and the run log.

The numerical boundary is `wasm-core/`. Its Rust/wasm-bindgen kernels own
normalization, matrix multiplication, PCA/power iteration, tiled cosine
distances, exact kNN, an HNSW-compatible index, and MST processing for the
HDBSCAN graph. `ExternalHdbscanProviderAdapter` is the explicit integration
point for a separately maintained hdbscan-rs/native provider. Every provider
must return the shared label/probability/outlier contract and may optionally
return a complete soft membership matrix. No external crate is vendored yet;
the checked-in Rust implementation remains the production provider. The
current development build uses the deterministic TypeScript
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
npm run audit:hdbscan -- --dataset-sample-size 100 --dataset-sample-seed 42 --fast
```

The HDBSCAN audit first fits the authoritative Python `hdbscan` result, then
feeds its exact UMAP coordinates to the WASM extractor. Its report compares
cluster labels after the best deterministic label permutation, noise agreement,
probability MAE/RMSE, outlier MAE, and soft-membership error. It is an
informational feature-level audit, not an end-to-end parity claim: Python's
full cross-cluster membership matrix is richer than the WASM assigned-membership
contract, and `umap-learn` versus `umap-js` remains a separate difference. The
command explicitly rejects `dbpedia_label_embeddings.json` and uses the
3,000-record Gemini dataset by default.

Build note clusters uses one persistent Obsidian Notice for cache scan,
embedding, clustering, completion, error, and cancellation progress; it does
not create one Notice per embedding callback. Local model installation has a
Settings progress bar with consent, model/tokenizer download, SHA-256 verify,
and install phases. Embedding runs persist redacted per-note diagnostics at
`.obsidian/plugins/atomic-clusters/embedding-log.json` (path, timestamp,
provider/model, duration, success/failure/cached status, and bounded error
message). Note content, vectors, API keys, and other secrets are never logged.
Batch provider failures are split when the provider supports safe retries;
local inference falls back to per-note attempts so healthy notes can still be
clustered while failed notes remain visible in the log.

The Settings tab also provides **Build clusters** and **Cancel** controls for
the same command-palette pipeline; while running, the build control is
disabled and progress remains in the persistent Notice.

When **Automatic refresh** is enabled, Markdown create/modify/delete/rename
events are debounced (5 seconds by default, with a 60-second cap). Only notes
whose content hash changed are embedded. Same-content renames move the cached
path and PCA row, while small edits reuse the saved PCA and hierarchy as
provisional placements. The Explorer labels those placements and the refresh
policy schedules a full rebuild when its change, deletion, cumulative, or
provisional thresholds are exceeded. **Refresh changed notes** runs the same
path manually; **Rebuild all clusters** always uses the complete WASM pipeline.

When the local provider is selected, **Test local runtime** verifies the
installed model, bundled renderer-safe ORT assets, ONNX session initialization,
and one safe probe before a bulk embedding run. The preflight result and any
sanitized cause are written as a run-level entry in the embedding log.

Use the **Open embedding log** button in Settings or the command-palette
command after the Notice disappears. It opens the persisted JSON in the
operating system's default text editor (for example, Notepad on a Windows
vault). Provider setup failures such as missing credentials, missing model, or
cancelled consent are recorded as run-level failures/cancellations; they are
not incorrectly assigned to every note.

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

Copy `dist/main.js`, `dist/manifest.json`, `dist/styles.css`, and the bundled
`dist/ort-wasm-simd-threaded.*` assets to `.obsidian/plugins/atomic-clusters/`
for a local install. The worker source is embedded in `main.js`; the Python
reference stays outside the plugin bundle. The ORT assets stay adjacent so the bundled local provider can load
them without document-relative URL assumptions.
The repository intentionally keeps the model weights out of the plugin; local
model download must be explicit and is disclosed in Settings.

### Current visualization issues

The following explorer behaviors are recorded for a future follow-up and are
not addressed by the current release:

- During viewport panning, notes can move farther than the viewport/cloud
  before snapping back into place.
- Content at the viewport edges can still be clipped.
- Hover feedback is ambiguous and can remain active while the pointer is over
  empty space instead of clearing.

## Scope and known gaps

This MVP implements the complete vault → embedding → worker → hierarchy →
explorer path and cache invalidation by content hash. The local ONNX runtime is
bundled and configured beside the CommonJS plugin bundle; an explicit injection
override remains available for alternate execution providers. Model download
and integrity management have no hidden runtime downloads. The deterministic
density-graph fallback is suitable for fixtures and small development vaults,
but should not be used as a scientific parity benchmark.
