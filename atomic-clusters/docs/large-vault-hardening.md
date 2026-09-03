# Large Vault hardening workflow

This workflow covers `plan.md` §5.1 and §5.2 without treating a development
fallback as a production large-vault result.

## Reproduce the measurement

From `atomic-clusters/`:

```bash
npm run benchmark:large-vault -- \
  --input-json ../dbpedia_gemini_embeddings.json.gz \
  --sizes 1000,3000 \
  --output-dir /tmp/atomic-clusters-large-vault-hardening-<run>
```

The default configuration is the normal `offline-e2e` configuration with seed
`42`; omit `--fast` for a release-scale measurement. `--fast` is intended for
checking the runner and report schema only. The runner executes the two primary
sizes sequentially and never runs the 3,000-row job more than once in one
invocation. It samples deterministically with `sampleSeed` and records a sample
fingerprint.

Before starting a primary job, the runner checks the source row count, a
conservative working-set estimate, available memory, and verified release WASM.
The checked-in TypeScript fallback rejects 512 or more rows by design. If
`wasm-core/pkg` is absent or fails `verifyWasmAsset`, 1,000 and 3,000 are
reported as `unavailable` and are not forced through the fallback.

## Report contents

The output directory contains `hardening-report.json`, a human-readable
`hardening-report.md`, and one compact `run-<size>.json` per selected size.
Successful runs record:

- dataset path, compressed-input SHA-256, row count, dimension, seed, and
  deterministic sample fingerprint;
- worker initialization, transfer-plus-clustering wall time, the
  `clusterEmbeddings` implementation time, metadata-only title-generation time,
  and progress-boundary phase timings;
- peak RSS, parent/worker heap samples, process `resourceUsage().maxRSS`,
  heartbeat gaps, `monitorEventLoopDelay()` percentiles, and any observed
  interval of at least 250 ms;
- progress event count, phase order, monotonicity, maximum gap, and liveness
  status;
- runtime backend and whether the worker observed the bundled WASM module;
- result summary (PCA width, leaf/noise counts, probabilities, hierarchy, and
  visualization row count).

The checked-in embeddings contain no Markdown body. Therefore title generation
is measured separately against a clearly labelled metadata-only fixture made
from each row's class and class hierarchy. It is not included in clustering
wall time and must not be read as a note-body benchmark. Embedding generation
and local ORT preflight are likewise excluded because the input is already
materialized Gemini output.

Cancellation probes use a deterministic subset of the same Gemini file, cap
the probe dimension at 64, and send `CANCEL` at the first progress callback of
each PCA, UMAP, HDBSCAN, hierarchy, and visualization boundary. The report
records request/observation timestamps and latency. This is a bounded worker
protocol probe, not a large-vault quality run.

## 5,000 and 10,000 rows

The available checked-in Gemini dataset has 3,000 records. The report always
includes explicit `unavailable` entries for 5,000 and 10,000 with
`duplicatedRows: false` and a reason stating that no synthetic duplication or
extrapolation was performed. A future real dataset can be supplied through
`--input-json`; the source-count and report validator will then prevent an
unsupported scale from being marked measured.

## Release validation

`releaseWasm` invokes the same `verifyWasmAsset` contract used by
`build:release` and lists the required exports. The release build remains
strict: missing `wasm-core/pkg` is an error for `npm run build:release`.
Pass `--require-release-wasm` to make the benchmark command fail after writing
its report when assets are absent or invalid. Without that flag, unavailable
release tooling is preserved as an explicit measurement limitation so a
development machine can still run report and cancellation checks.

The runner does not edit `plan.md` and does not use the tag-only embedding
fixture. It is intentionally offline after reading the checked-in Gemini
file.

## Renderer product response

The production build now performs a renderer-safe memory preflight immediately
before the full worker clustering request. It estimates the actual vector
row-count × dimension matrix plus worker clone, normalization, PCA covariance,
UMAP/hierarchy working copies, and an explicit safety margin. A hard stop is
used only when `performance.memory` provides finite JS-heap headroom and the
estimate reaches the dangerous threshold. `navigator.deviceMemory` or no
available-memory signal produces a visible warning/estimate and continues;
those signals are not treated as available RAM.

The persistent clustering Notice also emits a 10-second heartbeat. After 30
seconds without an accepted phase progress update it retains the current phase
and percentage while appending elapsed time and `Still working…`. The heartbeat
is stopped on completion, failure, cancellation, and plugin unload; it does not
alter worker clustering or cancellation semantics.
