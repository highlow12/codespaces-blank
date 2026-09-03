import test from "node:test";
import assert from "node:assert/strict";
import {
  BENCHMARK_SIZES,
  parseArgs,
  preflightLargeVault,
  estimateMemory,
  summarizeProgress,
  summarizeCancellation,
  validateHardeningReport,
  runInstrumentedWorkerJob
} from "../scripts/large-vault-hardening.mjs";

test("large-vault CLI parser keeps benchmark sizes and seeds explicit", () => {
  const options = parseArgs(["--sizes", "1000,3000", "--seed", "7", "--dataset-sample-seed", "11", "--fast", "--skip-cancellation"]);
  assert.deepEqual(options.sizes, BENCHMARK_SIZES);
  assert.equal(options.seed, 7);
  assert.equal(options.sampleSeed, 11);
  assert.equal(options.fast, true);
  assert.equal(options.skipCancellation, true);
  assert.equal(parseArgs(["--seed", "13"]).sampleSeed, 13);
  assert.throws(() => parseArgs(["--sizes", "5000"]), /only 1000 and 3000/);
});

test("large-vault preflight blocks large TypeScript fallback and marks missing source rows unavailable", () => {
  const blocked = preflightLargeVault({
    rowCount: 1000,
    dimension: 3072,
    availableRecords: 3000,
    wasmLoaded: false,
    memory: { heapLimitBytes: 4e9, freeSystemBytes: 16e9, totalSystemBytes: 32e9, currentRssBytes: 100e6 }
  });
  assert.equal(blocked.status, "blocked");
  assert.equal(blocked.canRun, false);
  assert.match(blocked.reasons[0], /release WASM/);

  const unavailable = preflightLargeVault({
    rowCount: 5000,
    dimension: 3072,
    availableRecords: 3000,
    wasmLoaded: true,
    memory: { heapLimitBytes: 4e9, freeSystemBytes: 16e9, totalSystemBytes: 32e9, currentRssBytes: 100e6 }
  });
  assert.equal(unavailable.status, "unavailable");
  assert.equal(unavailable.canRun, false);
  assert.match(unavailable.reasons[0], /3000 records/);
});

test("preflight exposes a warning when the conservative estimate is material", () => {
  const estimate = estimateMemory({ rowCount: 100, dimension: 64 });
  const warning = preflightLargeVault({
    rowCount: 100,
    dimension: 64,
    availableRecords: 100,
    wasmLoaded: true,
    memory: {
      heapLimitBytes: estimate.bytes.workingSetEstimate / 0.4,
      freeSystemBytes: estimate.bytes.workingSetEstimate / 0.4,
      totalSystemBytes: 4e9,
      currentRssBytes: 10e6
    }
  });
  assert.equal(warning.status, "warning");
  assert.equal(warning.canRun, true);
  assert.match(warning.reasons[0], /material/);
});

test("progress summary reports phase boundary spans, gaps, monotonicity, and liveness", () => {
  const summary = summarizeProgress([
    { phase: "pca", progress: 0.05, atMs: 10 },
    { phase: "umap", progress: 0.2, atMs: 25 },
    { phase: "umap", progress: 0.4, atMs: 40 },
    { phase: "complete", progress: 1, atMs: 100 }
  ], 100, 50);
  assert.deepEqual(summary.phases, ["pca", "umap", "complete"]);
  assert.equal(summary.progressMonotonic, true);
  assert.equal(summary.maxGapMs, 60);
  assert.equal(summary.maxGapPhase, "umap→complete");
  assert.equal(summary.phaseTimings.pca.untilNextPhaseMs, 15);
  assert.equal(summary.liveness.stalled, true);
  assert.equal(summary.liveness.status, "live");

  const cancelled = summarizeProgress([{ phase: "umap", progress: 0.2, atMs: 1 }], 20, 50, false);
  assert.equal(cancelled.liveness.status, "live");
  assert.equal(cancelled.liveness.completeEventObserved, false);
});

test("report validator rejects fabricated scales and the tag-only dataset", () => {
  const base = {
    schemaVersion: 1,
    runner: "atomic-clusters-large-vault-hardening",
    dataset: { path: "/data/dbpedia_gemini_embeddings.json.gz", sourceRecords: 3000, dimension: 3072 },
    scalePlan: [
      { size: 1000, status: "measured", dataset: { duplicatedRows: false } },
      { size: 3000, status: "unavailable" },
      { size: 5000, status: "unavailable" },
      { size: 10000, status: "unavailable" }
    ]
  };
  assert.equal(validateHardeningReport(base), true);
  assert.throws(() => validateHardeningReport({ ...base, dataset: { ...base.dataset, path: "/data/dbpedia_label_embeddings.json" } }), /tag-only/);
  assert.throws(() => validateHardeningReport({ ...base, dataset: { ...base.dataset, path: "/data/dbpedia_label_embeddings.json.gz" } }), /tag-only/);
  assert.throws(() => validateHardeningReport({ ...base, scalePlan: base.scalePlan.map((entry) => entry.size === 5000 ? { ...entry, status: "measured" } : entry) }), /5000/);
  assert.throws(() => validateHardeningReport({ ...base, scalePlan: base.scalePlan.map((entry) => entry.size === 10000 ? { ...entry, status: "measured" } : entry) }), /sourceRecords/);
});

test("worker instrumentation measures cancellation latency at a progress boundary", async () => {
  const workerSource = `
    const { parentPort } = require("worker_threads");
    let cancelled = false;
    parentPort.on("message", async (request) => {
      if (request.type === "INIT") { parentPort.postMessage({ type: "READY", version: 1 }); return; }
      if (request.type === "CANCEL") { cancelled = true; return; }
      parentPort.postMessage({ type: "PROGRESS", jobId: request.jobId, phase: "umap", progress: 0.2 });
      await new Promise((resolve) => setTimeout(resolve, 20));
      if (cancelled) { parentPort.postMessage({ type: "ERROR", jobId: request.jobId, code: "CANCELLED", message: "Clustering cancelled" }); return; }
      parentPort.postMessage({ type: "PROGRESS", jobId: request.jobId, phase: "complete", progress: 1 });
      parentPort.postMessage({ type: "RESULT", jobId: request.jobId, result: { timings: { totalMs: 1 } } });
    });
  `;
  const outcome = await runInstrumentedWorkerJob(workerSource, [{ id: "a", embedding: [1] }, { id: "b", embedding: [2] }, { id: "c", embedding: [3] }], {}, { cancelPhase: "umap", cancelTimeoutMs: 1000, timeoutMs: 1000 });
  const cancellation = summarizeCancellation(outcome);
  assert.equal(outcome.status, "cancelled");
  assert.equal(cancellation.requestObserved, true);
  assert.equal(cancellation.cancellationObserved, true);
  assert.ok(cancellation.latencyMs >= 0);
  assert.equal(outcome.progressSummary.liveness.status, "live");
  assert.ok(outcome.memory.peakRssBytes > 0);
  assert.ok(outcome.responsiveness);
});

test("worker instrumentation retains successful phase, RSS, and responsiveness measurements", async () => {
  const workerSource = `
    const { parentPort } = require("worker_threads");
    parentPort.on("message", (request) => {
      if (request.type === "INIT") { parentPort.postMessage({ type: "READY", version: 1 }); return; }
      if (request.type !== "CLUSTER") return;
      setTimeout(() => {
        parentPort.postMessage({ type: "METRIC", wasmLoaded: true, memory: { rssBytes: 123456, heapUsedBytes: 45678 } });
        parentPort.postMessage({ type: "PROGRESS", jobId: request.jobId, phase: "pca", progress: 0.05 });
        parentPort.postMessage({ type: "PROGRESS", jobId: request.jobId, phase: "complete", progress: 1 });
        parentPort.postMessage({ type: "RESULT", jobId: request.jobId, result: { timings: { totalMs: 2 }, leafLabels: [-1, -1, -1], probabilities: [0, 0, 0], outlierProxy: [1, 1, 1], hierarchy: { leaves: [], merges: [], root: null }, pca: { selected: 1, sampleSize: 3, candidates: [1] } } });
      }, 120);
    });
  `;
  const outcome = await runInstrumentedWorkerJob(workerSource, [{ id: "a", embedding: [1] }, { id: "b", embedding: [2] }, { id: "c", embedding: [3] }], {}, { timeoutMs: 1000 });
  assert.equal(outcome.status, "completed");
  assert.equal(outcome.progressSummary.liveness.status, "live");
  assert.equal(outcome.progressSummary.liveness.completeEventObserved, true);
  assert.equal(outcome.workerRuntimeObserved, true);
  assert.equal(outcome.memory.workerSampleCount, 1);
  assert.ok(outcome.memory.peakRssBytes >= 123456);
  assert.equal(outcome.responsiveness.status, "observed");
});
