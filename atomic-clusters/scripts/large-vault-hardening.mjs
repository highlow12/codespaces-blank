#!/usr/bin/env node

/**
 * Reproducible large-vault hardening measurements for the real plugin worker.
 *
 * The checked-in Gemini file contains 3,000 rows.  This runner measures the
 * two supported dataset sizes (1,000 and 3,000) and records 5,000/10,000 as
 * unavailable unless a future checked-in dataset actually contains those
 * rows.  It never duplicates rows to manufacture a large-vault result.
 *
 * Large runs require a verified release WASM asset.  The TypeScript fallback
 * intentionally rejects 512+ rows, so silently benchmarking it would either
 * fail or give a misleading product result.  Small cancellation probes can
 * still run without WASM because they exercise the worker protocol and
 * cancellation boundaries on a bounded fixture.
 */

import { build } from "esbuild";
import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { monitorEventLoopDelay, performance } from "node:perf_hooks";
import { Worker } from "node:worker_threads";
import { cpus, freemem, platform, release, totalmem, arch } from "node:os";
import { getHeapStatistics } from "node:v8";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { clusteringConfig, readDataset, sampleRecords } from "./offline-e2e.mjs";
import { REQUIRED_EXPORTS, verifyWasmAsset } from "./verify-wasm.mjs";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pluginDir = resolve(scriptDir, "..");
const repositoryDir = resolve(pluginDir, "..");
const defaultInput = join(repositoryDir, "dbpedia_gemini_embeddings.json.gz");

export const LARGE_VAULT_SIZES = [1000, 3000, 5000, 10000];
export const BENCHMARK_SIZES = [1000, 3000];
export const CANCELLATION_PHASES = ["pca", "umap", "hdbscan", "hierarchy", "visualization"];
export const DEFAULT_PROGRESS_STALL_WARNING_MS = 30_000;
export const DEFAULT_CANCEL_PROBE_SIZE = 100;
export const DEFAULT_CANCEL_TIMEOUT_MS = 60_000;
export const DEFAULT_JOB_TIMEOUT_MS = 30 * 60_000;

function safeInteger(value, name) {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) throw new Error(`${name} must be a safe integer`);
  return number;
}

function positiveInteger(value, name) {
  const number = safeInteger(value, name);
  if (number < 1) throw new Error(`${name} must be positive`);
  return number;
}

function nonNegativeInteger(value, name) {
  const number = safeInteger(value, name);
  if (number < 0) throw new Error(`${name} must be non-negative`);
  return number;
}

function optionValue(arg, argv, index) {
  const inline = arg.indexOf("=");
  if (inline >= 0) return [arg.slice(inline + 1), index];
  if (index + 1 >= argv.length || argv[index + 1].startsWith("--")) throw new Error(`${arg} requires a value`);
  return [argv[index + 1], index + 1];
}

export function parseSizes(value) {
  const sizes = String(value).split(",").map((item) => positiveInteger(item.trim(), "benchmark size"));
  if (!sizes.length || sizes.some((size) => !BENCHMARK_SIZES.includes(size))) {
    throw new Error(`--sizes accepts only ${BENCHMARK_SIZES.join(" and ")} (comma-separated)`);
  }
  return [...new Set(sizes)];
}

export function usage() {
  return `Usage: node scripts/large-vault-hardening.mjs [options]

Options:
  --input-json PATH             Gemini embedding JSON or .json.gz
  --sizes LIST                  Primary benchmark sizes, e.g. 1000,3000
  --seed N                      Clustering seed (default: 42)
  --dataset-sample-seed N      Deterministic sample seed (default: seed)
  --fast                       Use offline-e2e reduced settings (not release-scale)
  --output-dir PATH            Directory for JSON/Markdown reports
  --skip-cancellation          Do not run small worker cancellation probes
  --cancel-probe-size N        Probe rows (default: 100)
  --cancel-timeout-ms N        Cancellation probe timeout (default: 60000)
  --job-timeout-ms N           Successful job timeout (default: 1800000)
  --progress-stall-ms N        Progress liveness warning threshold (default: 30000)
  --strict-preflight           Treat memory warnings as a run blocker
  --require-release-wasm       Fail after writing the report if release WASM is unavailable/invalid
  --help                       Show this help

The report always includes an explicit unavailable entry for 5,000 and 10,000
when the source dataset has only 3,000 rows. No row duplication is performed.
`;
}

export function parseArgs(argv) {
  const options = {
    input: defaultInput,
    sizes: BENCHMARK_SIZES.slice(),
    seed: 42,
    sampleSeed: undefined,
    fast: false,
    outputDir: undefined,
    skipCancellation: false,
    cancelProbeSize: DEFAULT_CANCEL_PROBE_SIZE,
    cancelTimeoutMs: DEFAULT_CANCEL_TIMEOUT_MS,
    jobTimeoutMs: DEFAULT_JOB_TIMEOUT_MS,
    progressStallWarningMs: DEFAULT_PROGRESS_STALL_WARNING_MS,
    strictPreflight: false,
    requireReleaseWasm: false
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") return { help: true };
    if (arg === "--fast") { options.fast = true; continue; }
    if (arg === "--skip-cancellation") { options.skipCancellation = true; continue; }
    if (arg === "--strict-preflight") { options.strictPreflight = true; continue; }
    if (arg === "--require-release-wasm") { options.requireReleaseWasm = true; continue; }
    const names = [
      ["--input-json", "input", (value) => value],
      ["--sizes", "sizes", parseSizes],
      ["--seed", "seed", (value) => safeInteger(value, "seed")],
      ["--dataset-sample-seed", "sampleSeed", (value) => safeInteger(value, "dataset sample seed")],
      ["--output-dir", "outputDir", (value) => value],
      ["--output", "outputDir", (value) => value],
      ["--cancel-probe-size", "cancelProbeSize", (value) => positiveInteger(value, "cancel probe size")],
      ["--cancel-timeout-ms", "cancelTimeoutMs", (value) => positiveInteger(value, "cancel timeout")],
      ["--job-timeout-ms", "jobTimeoutMs", (value) => positiveInteger(value, "job timeout")],
      ["--progress-stall-ms", "progressStallWarningMs", (value) => positiveInteger(value, "progress stall threshold")]
    ];
    const match = names.find(([name]) => arg === name || arg.startsWith(`${name}=`));
    if (!match) throw new Error(`Unknown option: ${arg}`);
    const [value, next] = optionValue(arg, argv, index);
    options[match[1]] = match[2](value);
    index = next;
  }
  if (options.sampleSeed === undefined) options.sampleSeed = options.seed;
  return options;
}

function memorySnapshot() {
  const memory = process.memoryUsage();
  return {
    rssBytes: memory.rss,
    heapTotalBytes: memory.heapTotal,
    heapUsedBytes: memory.heapUsed,
    externalBytes: memory.external,
    arrayBuffersBytes: memory.arrayBuffers
  };
}

function maxValue(values, fallback = 0) {
  return values.length ? Math.max(...values) : fallback;
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function quantile(values, fraction) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((left, right) => left - right);
  const position = Math.max(0, Math.min(1, fraction)) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function bytes(value) {
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0;
}

function boundedBytes(value) {
  return bytes(Math.min(Number.MAX_SAFE_INTEGER, value));
}

/**
 * Estimate memory before a large job. These are deliberately labelled
 * estimates: JavaScript array/object overhead and allocator behavior vary by
 * Node/Electron build. The covariance term is a conservative upper bound for
 * the development fallback; a release WASM build may use less.
 */
export function estimateMemory({ rowCount, dimension, pcaSampleSize = Math.min(2000, rowCount), pcaComponents = Math.min(512, dimension, rowCount), umapComponents = 20, worker = true }) {
  if (!Number.isSafeInteger(rowCount) || rowCount < 1) throw new Error("rowCount must be positive");
  if (!Number.isSafeInteger(dimension) || dimension < 1) throw new Error("dimension must be positive");
  const jsMatrixBytes = rowCount * dimension * 8;
  const workerCloneBytes = worker ? jsMatrixBytes : 0;
  const normalizedBytes = jsMatrixBytes;
  const pilotBytes = pcaSampleSize * pcaComponents * 8;
  const pcaProjectionBytes = rowCount * pcaComponents * 8;
  const covarianceBytes = dimension * dimension * 8;
  const umapBytes = rowCount * umapComponents * 8;
  const hierarchyBytes = rowCount * 256;
  const lowerBoundBytes = jsMatrixBytes + normalizedBytes + pcaProjectionBytes + umapBytes;
  const workingSetEstimateBytes = boundedBytes(
    jsMatrixBytes * 2.5 + workerCloneBytes + normalizedBytes + pilotBytes * 2 +
    pcaProjectionBytes * 2 + covarianceBytes + umapBytes * 3 + hierarchyBytes
  );
  return {
    rowCount,
    dimension,
    assumptions: {
      scalarBytes: 8,
      workerStructuredCloneIncluded: worker,
      pcaCovarianceUpperBoundIncluded: true,
      estimateIsNotAnAllocatorGuarantee: true
    },
    components: { pcaSampleSize, pcaComponents, umapComponents },
    bytes: {
      jsEmbeddingMatrix: boundedBytes(jsMatrixBytes),
      workerClone: boundedBytes(workerCloneBytes),
      normalizedMatrix: boundedBytes(normalizedBytes),
      pcaPilotProjection: boundedBytes(pilotBytes),
      pcaProjection: boundedBytes(pcaProjectionBytes),
      pcaCovarianceUpperBound: boundedBytes(covarianceBytes),
      umapFeaturesAndWorkingCopies: boundedBytes(umapBytes * 3),
      hierarchyEstimate: boundedBytes(hierarchyBytes),
      lowerBound: boundedBytes(lowerBoundBytes),
      workingSetEstimate: workingSetEstimateBytes
    }
  };
}

/**
 * Preflight gate used by the runner. `blocked` means the requested measurement
 * must not start; `warning` remains runnable unless strict mode is selected.
 */
export function preflightLargeVault({ rowCount, dimension, availableRecords, wasmLoaded, memory = {}, worker = true }) {
  const estimate = estimateMemory({ rowCount, dimension, worker });
  const heapLimitBytes = bytes(memory.heapLimitBytes ?? getHeapStatistics().heap_size_limit);
  const freeSystemBytes = bytes(memory.freeSystemBytes ?? freemem());
  const currentRssBytes = bytes(memory.currentRssBytes ?? process.memoryUsage().rss);
  const reasons = [];
  let status = "pass";
  if (availableRecords !== undefined && rowCount > availableRecords) {
    status = "unavailable";
    reasons.push(`source dataset contains ${availableRecords} records; ${rowCount} records are not available`);
  } else if (rowCount >= 512 && !wasmLoaded) {
    status = "blocked";
    reasons.push("verified release WASM is required for 512+ rows; the TypeScript fallback is intentionally limited");
  }
  if (status === "pass" || status === "warning") {
    const heapRatio = heapLimitBytes ? estimate.bytes.workingSetEstimate / heapLimitBytes : 0;
    const freeRatio = freeSystemBytes ? estimate.bytes.workingSetEstimate / freeSystemBytes : 0;
    if (heapRatio >= 0.8 || freeRatio >= 0.8) {
      status = "blocked";
      reasons.push(`estimated working set is too close to available memory (heap ratio ${heapRatio.toFixed(3)}, free-system ratio ${freeRatio.toFixed(3)})`);
    } else if (heapRatio >= 0.35 || freeRatio >= 0.35) {
      status = "warning";
      reasons.push(`estimated working set is material (heap ratio ${heapRatio.toFixed(3)}, free-system ratio ${freeRatio.toFixed(3)})`);
    }
  }
  return {
    status,
    canRun: status === "pass" || status === "warning",
    reasons,
    requestedRecords: rowCount,
    availableRecords: availableRecords ?? null,
    wasmLoaded: Boolean(wasmLoaded),
    memory: {
      currentRssBytes,
      heapLimitBytes,
      freeSystemBytes,
      totalSystemBytes: bytes(memory.totalSystemBytes ?? totalmem())
    },
    estimate
  };
}

async function sha256File(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

function relativePath(path) {
  return relative(repositoryDir, path) || path;
}

export async function inspectReleaseWasm() {
  const generatedDir = resolve(pluginDir, "wasm-core/pkg");
  const gluePath = resolve(generatedDir, "atomic_clusters_wasm_core.js");
  const wasmPath = resolve(generatedDir, "atomic_clusters_wasm_core_bg.wasm");
  const assetsPresent = existsSync(gluePath) && existsSync(wasmPath);
  const base = {
    buildCommand: "npm run build:release",
    buildPolicy: "release build requires wasm-core/pkg and runs verifyWasmAsset",
    gluePath: relativePath(gluePath),
    wasmPath: relativePath(wasmPath),
    requiredExports: REQUIRED_EXPORTS.slice(),
    assetsPresent
  };
  if (!assetsPresent) {
    return { ...base, status: "unavailable", reason: "generated release WASM assets are absent; wasm-pack/Rust build is required" };
  }
  try {
    const result = await verifyWasmAsset(gluePath, wasmPath);
    return { ...base, status: "passed", bytes: result.bytes, exports: result.exports };
  } catch (error) {
    return { ...base, status: "failed", reason: error instanceof Error ? error.message : String(error) };
  }
}

function wasmBootstrapPlugin(gluePath, wasmPath) {
  const generatedDir = dirname(gluePath);
  return {
    name: "atomic-clusters-wasm-bootstrap",
    setup(plugin) {
      plugin.onResolve({ filter: /^atomic-clusters-wasm-bootstrap$/ }, () => ({ path: "bootstrap", namespace: "atomic-wasm" }));
      plugin.onLoad({ filter: /.*/, namespace: "atomic-wasm" }, async () => {
        if (!existsSync(gluePath) || !existsSync(wasmPath)) return { contents: "// Large-vault measurement fallback: no release WASM asset." };
        const encoded = (await readFile(wasmPath)).toString("base64");
        return { resolveDir: generatedDir, contents: `
          import { initSync, normalize, matmul, pca, randomized_pca,
            cosine_distances, exact_knn, exact_knn_cosine_tiled,
            euclidean_mutual_reachability_mst, mst,
            mutual_reachability_mst, hdbscan_extract,
            hdbscan_extract_with_rows, HnswIndex } from ${JSON.stringify(gluePath)};
          const bytes = Uint8Array.from(Buffer.from(${JSON.stringify(encoded)}, "base64"));
          initSync({ module: new WebAssembly.Module(bytes) });
          globalThis.__ATOMIC_CLUSTERS_WASM__ = { normalize, matmul, pca, randomized_pca,
            cosine_distances, exact_knn, exact_knn_cosine_tiled,
            euclidean_mutual_reachability_mst, mst, mutual_reachability_mst,
            hdbscan_extract, hdbscan_extract_with_rows, HnswIndex };
        ` };
      });
    }
  };
}

let workerSourcePromise;

export async function buildInstrumentedWorkerSource() {
  if (workerSourcePromise) return workerSourcePromise;
  workerSourcePromise = (async () => {
    const generatedDir = resolve(pluginDir, "wasm-core/pkg");
    const gluePath = resolve(generatedDir, "atomic_clusters_wasm_core.js");
    const wasmPath = resolve(generatedDir, "atomic_clusters_wasm_core_bg.wasm");
    const result = await build({
      bundle: true,
      entryPoints: [resolve(pluginDir, "src/worker.ts")],
      format: "cjs",
      platform: "node",
      target: "node20",
      external: ["worker_threads", "node:*"] ,
      write: false,
      logLevel: "silent",
      plugins: [wasmBootstrapPlugin(gluePath, wasmPath)]
    });
    const bundle = new TextDecoder().decode(result.outputFiles[0].contents);
    // This prelude is outside the application worker source and only emits
    // redacted process metrics. It does not alter clustering inputs/results.
    const prelude = `
      const __atomicClustersMetricsPort = require("worker_threads").parentPort;
      const __atomicClustersStartedAt = globalThis.performance?.now?.() ?? Date.now();
      function __atomicClustersSendMetrics() {
        try {
          const memory = process.memoryUsage();
          __atomicClustersMetricsPort.postMessage({
            type: "METRIC",
            atMs: (globalThis.performance?.now?.() ?? Date.now()) - __atomicClustersStartedAt,
            wasmLoaded: Boolean(globalThis.__ATOMIC_CLUSTERS_WASM__),
            memory: {
              rssBytes: memory.rss,
              heapTotalBytes: memory.heapTotal,
              heapUsedBytes: memory.heapUsed,
              externalBytes: memory.external,
              arrayBuffersBytes: memory.arrayBuffers
            }
          });
        } catch {}
      }
      __atomicClustersSendMetrics();
      const __atomicClustersMetricsTimer = setInterval(__atomicClustersSendMetrics, 100);
      __atomicClustersMetricsTimer.unref?.();
    `;
    return { source: `${prelude}\n${bundle}`, assetPresent: existsSync(gluePath) && existsSync(wasmPath) };
  })();
  return workerSourcePromise;
}

function heartbeatSummary(samples, expectedIntervalMs = 50) {
  const gaps = [];
  for (let index = 1; index < samples.length; index++) gaps.push(samples[index].atMs - samples[index - 1].atMs);
  const over250 = gaps.filter((gap) => gap >= 250).length;
  return {
    expectedIntervalMs,
    sampleCount: samples.length,
    meanGapMs: mean(gaps),
    p95GapMs: quantile(gaps, 0.95),
    maxGapMs: maxValue(gaps),
    gapsOver250ms: over250,
    status: samples.length >= 2 ? "observed" : "insufficient_samples"
  };
}

function histogramSummary(histogram) {
  const toMs = (value) => Number.isFinite(value) ? value / 1e6 : 0;
  return {
    resolutionMs: toMs(histogram.resolution),
    count: Number(histogram.count || 0),
    minMs: toMs(histogram.min),
    meanMs: toMs(histogram.mean),
    p50Ms: toMs(histogram.percentile(50)),
    p95Ms: toMs(histogram.percentile(95)),
    p99Ms: toMs(histogram.percentile(99)),
    maxMs: toMs(histogram.max),
    exceeds250ms: toMs(histogram.max) >= 250
  };
}

/** Summarize progress callbacks and phase-boundary timing without inventing internal spans. */
export function summarizeProgress(events, durationMs, stallWarningMs = DEFAULT_PROGRESS_STALL_WARNING_MS, expectedComplete = true) {
  const sorted = events.slice().sort((left, right) => left.atMs - right.atMs);
  const gaps = sorted.slice(1).map((event, index) => event.atMs - sorted[index].atMs);
  const phaseNames = [];
  const byPhase = new Map();
  for (const event of sorted) {
    if (!byPhase.has(event.phase)) { byPhase.set(event.phase, []); phaseNames.push(event.phase); }
    byPhase.get(event.phase).push(event);
  }
  const firstByPhase = new Map(phaseNames.map((phase) => [phase, byPhase.get(phase)[0]]));
  const phaseTimings = {};
  for (let index = 0; index < phaseNames.length; index++) {
    const phase = phaseNames[index];
    const phaseEvents = byPhase.get(phase);
    const first = phaseEvents[0];
    const last = phaseEvents[phaseEvents.length - 1];
    const next = phaseNames[index + 1] ? firstByPhase.get(phaseNames[index + 1]) : undefined;
    phaseTimings[phase] = {
      eventCount: phaseEvents.length,
      firstProgress: first.progress,
      lastProgress: last.progress,
      firstAtMs: first.atMs,
      lastAtMs: last.atMs,
      observedSpanMs: Math.max(0, last.atMs - first.atMs),
      untilNextPhaseMs: next ? Math.max(0, next.atMs - first.atMs) : Math.max(0, durationMs - first.atMs)
    };
  }
  let monotonic = true;
  for (let index = 1; index < sorted.length; index++) if (sorted[index].progress + 1e-9 < sorted[index - 1].progress) monotonic = false;
  const completeEvent = [...sorted].reverse().find((event) => event.phase === "complete" && event.progress >= 1);
  const maxGapMs = maxValue(gaps);
  return {
    eventCount: sorted.length,
    phases: phaseNames,
    firstProgressAtMs: sorted[0]?.atMs ?? null,
    lastProgressAtMs: sorted.at(-1)?.atMs ?? null,
    maxGapMs,
    maxGapPhase: maxGapMs ? `${sorted[gaps.indexOf(maxGapMs)].phase}→${sorted[gaps.indexOf(maxGapMs) + 1].phase}` : null,
    progressMonotonic: monotonic,
    phaseTimings,
    liveness: {
      expectedComplete,
      completeEventObserved: Boolean(completeEvent),
      noProgressEvents: sorted.length === 0,
      stalled: maxGapMs > stallWarningMs,
      stallWarningMs,
      status: sorted.length > 0 && (!expectedComplete || completeEvent) ? "live" : "incomplete"
    }
  };
}

function resultSummary(result, records) {
  const sizes = new Map();
  for (const label of result.leafLabels) if (label >= 0) sizes.set(label, (sizes.get(label) || 0) + 1);
  const clusterSizes = [...sizes.values()].sort((left, right) => right - left);
  const noiseCount = result.leafLabels.filter((label) => label < 0).length;
  return {
    recordCount: records.length,
    dimension: records[0]?.embedding?.length || 0,
    clusterCount: sizes.size,
    clusteredCount: records.length - noiseCount,
    noiseCount,
    noiseRate: records.length ? noiseCount / records.length : 0,
    largestClusterSize: clusterSizes[0] || 0,
    medianClusterSize: clusterSizes.length ? clusterSizes[Math.floor((clusterSizes.length - 1) / 2)] : 0,
    probabilityMean: mean(result.probabilities || []),
    outlierProxyMean: mean(result.outlierProxy || []),
    pca: result.pca ? { selected: result.pca.selected, sampleSize: result.pca.sampleSize, candidateCount: result.pca.candidates?.length || 0 } : null,
    hierarchy: { leafCount: result.hierarchy?.leaves?.length || 0, mergeCount: result.hierarchy?.merges?.length || 0, root: result.hierarchy?.root ?? null },
    visualization: result.visualization ? { coordinateCount: result.visualization.coordinates?.length || 0, runtime: result.visualization.configuration?.runtime } : { status: "deferred_or_unavailable" }
  };
}

function metadataNotes(records) {
  return records.map((record, index) => {
    const hierarchy = Array.isArray(record.classHierarchy) ? record.classHierarchy.filter((value) => typeof value === "string") : [];
    const title = record.class || `DBpedia record ${index}`;
    const content = [record.class, ...hierarchy].filter(Boolean).join(" ");
    let hash = 2166136261;
    for (const character of `${record.id}:${title}:${content}`) { hash ^= character.charCodeAt(0); hash = Math.imul(hash, 16777619); }
    return { path: record.id, title, content, hash: (hash >>> 0).toString(16), mtime: 0 };
  });
}

let titleModulePromise;

async function loadTitleModule() {
  if (titleModulePromise) return titleModulePromise;
  titleModulePromise = (async () => {
    const source = `import { generateKeywordTitles } from ${JSON.stringify(resolve(pluginDir, "src/title.ts"))}; export { generateKeywordTitles };`;
    const result = await build({
      stdin: { contents: source, resolveDir: pluginDir, sourcefile: "large-vault-title-entry.ts", loader: "ts" },
      bundle: true,
      format: "esm",
      platform: "node",
      target: "node20",
      write: false,
      logLevel: "silent"
    });
    return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString("base64")}`);
  })();
  return titleModulePromise;
}

async function measureMetadataTitleGeneration(result, records) {
  const { generateKeywordTitles } = await loadTitleModule();
  const notes = metadataNotes(records);
  const progress = [];
  const started = performance.now();
  const titled = generateKeywordTitles(result, notes, { onProgress: (done, total) => progress.push({ done, total, atMs: performance.now() - started }) });
  const durationMs = performance.now() - started;
  return {
    status: "measured",
    measurementKind: "metadata-only",
    contentSource: "class and class_hierarchy fields from the embedding dataset; no Markdown body is present",
    includedInClusteringWallTime: false,
    recordCount: records.length,
    nodeCount: titled.titleGeneration?.nodeCount || 0,
    durationMs,
    progress: {
      eventCount: progress.length,
      finalDone: progress.at(-1)?.done || 0,
      finalTotal: progress.at(-1)?.total || 0,
      maxGapMs: maxValue(progress.slice(1).map((event, index) => event.atMs - progress[index].atMs))
    }
  };
}

async function waitForWorkerReady(worker, timeoutMs) {
  return new Promise((resolveReady, rejectReady) => {
    const timer = setTimeout(() => rejectReady(new Error("worker initialization timed out")), timeoutMs);
    const onMessage = (message) => {
      if (message?.type !== "READY") return;
      clearTimeout(timer);
      worker.off("message", onMessage);
      resolveReady();
    };
    worker.on("message", onMessage);
    worker.postMessage({ type: "INIT", version: 1 });
  });
}

/**
 * Run one job in the same Node worker protocol as the plugin. The optional
 * cancelPhase sends CANCEL at the first progress event for that stage.
 */
export async function runInstrumentedWorkerJob(workerSource, records, config, { cancelPhase, timeoutMs = DEFAULT_JOB_TIMEOUT_MS, cancelTimeoutMs = DEFAULT_CANCEL_TIMEOUT_MS, progressStallWarningMs = DEFAULT_PROGRESS_STALL_WARNING_MS } = {}) {
  const worker = new Worker(workerSource, { eval: true });
  const startedAt = performance.now();
  const progress = [];
  const parentMemory = [memorySnapshot()];
  const workerMemory = [];
  const heartbeat = [];
  const eventLoopDelay = monitorEventLoopDelay({ resolution: 10 });
  eventLoopDelay.enable();
  const resourceUsageStarted = process.resourceUsage?.().maxRSS || 0;
  let cancelRequestedAtMs = null;
  let cancelObservedAtMs = null;
  let cancelSent = false;
  let settled = false;
  let terminating = false;
  let jobStartedAt = null;
  let initAt = null;
  let resolveJob;
  let rejectJob;
  let timeout;
  const outcomePromise = new Promise((resolveOutcome, rejectOutcome) => { resolveJob = resolveOutcome; rejectJob = rejectOutcome; });
  const heartbeatTimer = setInterval(() => {
    const atMs = performance.now() - startedAt;
    heartbeat.push({ atMs });
  }, 50);
  const memoryTimer = setInterval(() => parentMemory.push({ atMs: performance.now() - startedAt, ...memorySnapshot() }), 100);
  heartbeatTimer.unref?.();
  memoryTimer.unref?.();

  const finish = (outcome) => {
    if (settled) return;
    settled = true;
    if (timeout) clearTimeout(timeout);
    resolveJob(outcome);
  };
  worker.on("message", (message) => {
    const receivedAtMs = performance.now() - startedAt;
    if (message?.type === "METRIC") {
      workerMemory.push({ atMs: receivedAtMs, wasmLoaded: Boolean(message.wasmLoaded), ...(message.memory || {}) });
      return;
    }
    if (message?.type === "PROGRESS") {
      const event = { phase: String(message.phase), progress: Number(message.progress), atMs: receivedAtMs };
      progress.push(event);
      if (cancelPhase && !cancelSent && event.phase === cancelPhase) {
        cancelSent = true;
        cancelRequestedAtMs = performance.now() - startedAt;
        worker.postMessage({ type: "CANCEL" });
      }
      return;
    }
    if (message?.type === "RESULT") {
      finish({ status: cancelSent ? "completed_after_cancel_request" : "completed", result: message.result, finishedAtMs: receivedAtMs });
      return;
    }
    if (message?.type === "ERROR") {
      const cancelled = message.code === "CANCELLED" || /cancel/i.test(String(message.message));
      if (cancelled) cancelObservedAtMs = receivedAtMs;
      finish({ status: cancelled ? "cancelled" : "failed", error: String(message.message || "worker failed"), code: message.code, finishedAtMs: receivedAtMs });
    }
  });
  worker.on("error", (error) => {
    if (!settled) { rejectJob(error); settled = true; }
  });
  worker.on("exit", (code) => {
    if (!settled && !terminating) {
      const error = new Error(`worker exited before producing a result (code ${code})`);
      rejectJob(error);
      settled = true;
    }
  });

  let result;
  let diagnostics;
  try {
    initAt = performance.now();
    await waitForWorkerReady(worker, Math.min(timeoutMs, 10_000));
    const initFinishedAt = performance.now();
    jobStartedAt = initFinishedAt;
    worker.postMessage({ type: "CLUSTER", jobId: `large-vault-${Date.now()}-${Math.random().toString(16).slice(2)}`, ids: records.map((record) => record.id), embeddings: records.map((record) => record.embedding), config });
    const effectiveTimeout = cancelPhase ? cancelTimeoutMs : timeoutMs;
    timeout = setTimeout(() => {
      if (cancelSent) finish({ status: "cancel-timeout", error: `worker did not observe cancellation within ${cancelTimeoutMs}ms`, finishedAtMs: performance.now() - startedAt });
      else finish({ status: "timeout", error: `worker did not finish within ${timeoutMs}ms`, finishedAtMs: performance.now() - startedAt });
    }, effectiveTimeout);
    const outcome = await outcomePromise;
    const finishedAtMs = outcome.finishedAtMs ?? performance.now() - startedAt;
    result = {
      ...outcome,
      startedAtMs: 0,
      workerInitMs: initFinishedAt - initAt,
      jobStartedAtMs: initFinishedAt - startedAt,
      jobWallTimeMs: Math.max(0, finishedAtMs - (initFinishedAt - startedAt)),
      finishedAtMs,
      cancelPhase: cancelPhase || null,
      cancelRequestedAtMs,
      cancelObservedAtMs,
      progress,
      progressSummary: summarizeProgress(progress, Math.max(0, finishedAtMs - (initFinishedAt - startedAt)), progressStallWarningMs, !cancelPhase),
      responsiveness: null,
      memory: null,
      workerRuntimeObserved: workerMemory.some((sample) => sample.wasmLoaded),
      workerMetrics: workerMemory
    };
  } catch (error) {
    result = {
      status: "failed",
      error: error instanceof Error ? error.message : String(error),
      startedAtMs: 0,
      workerInitMs: initAt ? performance.now() - initAt : 0,
      jobStartedAtMs: jobStartedAt ? jobStartedAt - startedAt : null,
      jobWallTimeMs: 0,
      finishedAtMs: performance.now() - startedAt,
      cancelPhase: cancelPhase || null,
      cancelRequestedAtMs,
      cancelObservedAtMs,
      progress,
      progressSummary: summarizeProgress(progress, performance.now() - startedAt, progressStallWarningMs, !cancelPhase),
      responsiveness: null,
      memory: null,
      workerRuntimeObserved: workerMemory.some((sample) => sample.wasmLoaded),
      workerMetrics: workerMemory
    };
  } finally {
    clearInterval(heartbeatTimer);
    clearInterval(memoryTimer);
    parentMemory.push({ atMs: performance.now() - startedAt, ...memorySnapshot() });
    const histogram = histogramSummary(eventLoopDelay);
    eventLoopDelay.disable();
    const heartbeatData = heartbeatSummary(heartbeat);
    const maxParentRss = maxValue(parentMemory.map((sample) => sample.rssBytes));
    const maxWorkerRss = maxValue(workerMemory.map((sample) => sample.rssBytes));
    const maxResourceRss = bytes((process.resourceUsage?.().maxRSS || 0) * 1024);
    const resourceUsageStartedBytes = bytes(resourceUsageStarted * 1024);
    const peakRssBytes = maxValue([maxParentRss, maxWorkerRss, maxResourceRss]);
    const peakWorkerHeapUsedBytes = maxValue(workerMemory.map((sample) => sample.heapUsedBytes));
    const peakParentHeapUsedBytes = maxValue(parentMemory.map((sample) => sample.heapUsedBytes));
    const responsiveness = {
      status: heartbeatData.status === "observed" ? "observed" : "insufficient_samples",
      eventLoopDelay: histogram,
      heartbeat: heartbeatData,
      mainThreadStallOver250ms: histogram.exceeds250ms || heartbeatData.gapsOver250ms > 0
    };
    const memory = {
      rssBeforeBytes: parentMemory[0]?.rssBytes || 0,
      rssAfterBytes: parentMemory.at(-1)?.rssBytes || 0,
      peakRssBytes,
      peakRssDeltaBytes: Math.max(0, peakRssBytes - (parentMemory[0]?.rssBytes || 0)),
      peakParentHeapUsedBytes,
      peakWorkerHeapUsedBytes,
      parentSampleCount: parentMemory.length,
      workerSampleCount: workerMemory.length,
      resourceUsageMaxRssBytes: maxResourceRss,
      resourceUsageMaxRssBeforeBytes: resourceUsageStartedBytes,
      resourceUsageMaxRssDeltaBytes: Math.max(0, maxResourceRss - resourceUsageStartedBytes),
      rssSemantics: "Node RSS is process-wide; parent/worker samples are reported separately and peak is the maximum observed, not their sum"
    };
    if (resolveJob && !settled) rejectJob(new Error("worker measurement ended without an outcome"));
    try { terminating = true; await worker.terminate(); } catch {}
    diagnostics = { responsiveness, memory };
  }
  return { ...result, responsiveness: diagnostics?.responsiveness || result.responsiveness, memory: diagnostics?.memory || result.memory };
}

export function summarizeCancellation(outcome) {
  const requested = outcome.cancelRequestedAtMs;
  const observed = outcome.cancelObservedAtMs;
  const latencyMs = requested !== null && observed !== null ? Math.max(0, observed - requested) : null;
  return {
    phase: outcome.cancelPhase,
    status: outcome.status,
    requestObserved: requested !== null,
    cancellationObserved: observed !== null,
    requestedAtMs: requested,
    observedAtMs: observed,
    latencyMs,
    interpretation: outcome.status === "cancelled" ? "worker reached a cancellation check after the requested phase boundary" : outcome.status === "completed_after_cancel_request" ? "job completed before the worker observed CANCEL" : outcome.status === "cancel-timeout" ? "cancellation was not observed before the probe timeout" : "cancellation request was not completed"
  };
}

async function runCancellationProbes(workerSource, records, options) {
  const probes = [];
  // Cancellation is a protocol probe, not a large-vault quality run. Keep
  // the sample drawn from the Gemini dataset but cap its dimensions so the
  // TypeScript fallback can exercise every boundary without an enormous
  // covariance calculation when release WASM is unavailable.
  const inputDimension = Math.min(64, records[0].embedding.length);
  const probeRecords = records.map((record) => ({ ...record, embedding: record.embedding.slice(0, inputDimension) }));
  const config = clusteringConfig({ fast: true, seed: options.seed }, probeRecords.length, inputDimension);
  for (const phase of CANCELLATION_PHASES) {
    const outcome = await runInstrumentedWorkerJob(workerSource, probeRecords, config, {
      cancelPhase: phase,
      cancelTimeoutMs: options.cancelTimeoutMs,
      timeoutMs: options.cancelTimeoutMs,
      progressStallWarningMs: options.progressStallWarningMs
    });
    probes.push(summarizeCancellation(outcome));
  }
  return {
    status: probes.every((probe) => probe.status === "cancelled") ? "measured" : "partial",
    mechanism: "postMessage(CANCEL) at the first progress callback for each phase; cancellation is observed at the next cooperative check",
    measurementKind: "bounded-protocol-probe",
    probeRecordCount: probeRecords.length,
    sourceEmbeddingDimension: records[0].embedding.length,
    probeEmbeddingDimension: inputDimension,
    config,
    boundaries: probes
  };
}

function sampleFingerprint(records) {
  let hash = 2166136261;
  for (const record of records) for (const character of record.id) { hash ^= character.charCodeAt(0); hash = Math.imul(hash, 16777619); }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

async function runBenchmark(size, dataset, options, workerSource, releaseWasm) {
  const recordsStartedAt = performance.now();
  const records = sampleRecords(dataset.records, size, options.sampleSeed);
  const sampleSelectionMs = performance.now() - recordsStartedAt;
  const config = clusteringConfig({ fast: options.fast, seed: options.seed }, records.length, dataset.dimension);
  const preflight = preflightLargeVault({
    rowCount: size,
    dimension: dataset.dimension,
    availableRecords: dataset.records.length,
    wasmLoaded: releaseWasm.status === "passed"
  });
  const common = {
    size,
    status: "unavailable",
    dataset: { sourceRecords: dataset.records.length, sampledRecords: records.length, dimension: dataset.dimension, sampleSeed: options.sampleSeed, sampleFingerprint: sampleFingerprint(records) },
    config,
    preflight,
    runtime: { backend: releaseWasm.status === "passed" ? "rust-wasm" : "typescript-fallback", wasmLoaded: releaseWasm.status === "passed", wasmStatus: releaseWasm.status, umap: "umap-js" },
    embedding: { status: "excluded", provider: "precomputed Gemini embeddings", includedInWallTime: false },
    timings: { sampleSelectionMs },
    limitation: null
  };
  if (!preflight.canRun || (options.strictPreflight && preflight.status === "warning")) {
    return { ...common, limitation: preflight.reasons.join("; ") || "preflight blocked this measurement" };
  }
  if (releaseWasm.status !== "passed") {
    return { ...common, limitation: "large-vault measurement was not started because release WASM verification did not pass" };
  }
  const startedAt = performance.now();
  const outcome = await runInstrumentedWorkerJob(workerSource.source, records, config, { timeoutMs: options.jobTimeoutMs, progressStallWarningMs: options.progressStallWarningMs });
  if (outcome.status !== "completed") {
    return {
      ...common,
      status: "failed",
      limitation: outcome.error || `worker ended with ${outcome.status}`,
      timings: { sampleSelectionMs, workerInitMs: outcome.workerInitMs, clusteringWallMs: outcome.jobWallTimeMs, wallTimeMs: performance.now() - startedAt },
      progress: outcome.progressSummary,
      responsiveness: outcome.responsiveness,
      memory: outcome.memory
    };
  }
  const titleGeneration = await measureMetadataTitleGeneration(outcome.result, records);
  const wallTimeMs = performance.now() - startedAt;
  return {
    ...common,
    status: "measured",
    runtime: { ...common.runtime, workerRuntimeObserved: outcome.workerRuntimeObserved },
    timings: {
      sampleSelectionMs,
      workerInitMs: outcome.workerInitMs,
      transferAndClusterWallMs: outcome.jobWallTimeMs,
      clusterImplementationMs: outcome.result.timings?.totalMs ?? null,
      titleGenerationMs: titleGeneration.durationMs ?? null,
      wallTimeMs
    },
    progress: outcome.progressSummary,
    responsiveness: outcome.responsiveness,
    memory: outcome.memory,
    titleGeneration,
    summary: resultSummary(outcome.result, records),
    limitation: options.fast ? "fast configuration; not comparable to a default full release run" : "title generation uses metadata-only fields because the embedding dataset has no Markdown bodies"
  };
}

function unavailableScaleEntry(size, sourceRecords, dimension, reason) {
  return {
    size,
    status: "unavailable",
    dataset: { sourceRecords, dimension, duplicatedRows: false },
    reason
  };
}

export function validateHardeningReport(report) {
  if (!report || report.schemaVersion !== 1 || report.runner !== "atomic-clusters-large-vault-hardening") throw new Error("invalid large-vault hardening report identity");
  const sourceRecords = report.dataset?.sourceRecords;
  if (!Number.isSafeInteger(sourceRecords) || sourceRecords < 1) throw new Error("report dataset sourceRecords is invalid");
  if (["dbpedia_label_embeddings.json", "dbpedia_label_embeddings.json.gz"].includes(basename(report.dataset.path || ""))) throw new Error("report must not use the tag-only dataset");
  for (const entry of report.scalePlan || []) {
    if (entry.size > sourceRecords && entry.status !== "unavailable") throw new Error(`scale ${entry.size} cannot be measured beyond sourceRecords ${sourceRecords}`);
    if ((entry.size === 5000 || entry.size === 10000) && entry.status !== "unavailable") throw new Error(`${entry.size} must remain unavailable without source rows`);
    if (entry.status === "measured" && entry.dataset?.duplicatedRows) throw new Error(`scale ${entry.size} falsely reports duplicated rows`);
  }
  return true;
}

function renderMarkdown(report) {
  const lines = [
    "# Large Vault hardening report",
    "",
    `Generated: ${report.generatedAt}`,
    `Status: **${report.status}**`,
    "",
    "## Dataset and policy",
    "",
    `- Dataset: \`${basename(report.dataset.path)}\` (SHA-256 of compressed input: \`${report.dataset.sha256}\`)`,
    `- Source rows: ${report.dataset.sourceRecords}; embedding dimension: ${report.dataset.dimension}`,
    `- Sampling seed: ${report.options.sampleSeed}; clustering seed: ${report.options.seed}`,
    "- Rows are never duplicated to claim a 5,000- or 10,000-row clustering result.",
    "- Large runs require verified release WASM; the TypeScript fallback is not treated as a large-vault backend.",
    "",
    "## Measurements",
    "",
    "| Rows | Status | Backend | Cluster wall time (ms) | Peak RSS (MB) | Progress max gap (ms) |",
    "| ---: | --- | --- | ---: | ---: | ---: |"
  ];
  for (const run of report.runs) {
    const wall = run.timings?.transferAndClusterWallMs ?? "—";
    const rss = run.memory?.peakRssBytes ? (run.memory.peakRssBytes / 1024 / 1024).toFixed(1) : "—";
    const gap = run.progress?.maxGapMs === undefined ? "—" : run.progress.maxGapMs.toFixed(1);
    lines.push(`| ${run.size} | ${run.status} | ${run.runtime?.backend || "—"} | ${typeof wall === "number" ? wall.toFixed(1) : wall} | ${rss} | ${gap} |`);
    if (run.limitation) lines.push(`|  | limitation: ${run.limitation.replaceAll("|", "\\|")} |  |  |  |  |`);
  }
  lines.push("", "## 5,000 / 10,000 policy", "", "| Rows | Status | Reason |", "| ---: | --- | --- |", ...report.scalePlan.filter((entry) => entry.size >= 5000).map((entry) => `| ${entry.size} | ${entry.status} | ${entry.reason || "—"} |`));
  lines.push("", "## Cancellation boundaries", "", `Status: **${report.cancellation.status}**`, "", "| Phase | Status | Latency (ms) | Interpretation |", "| --- | --- | ---: | --- |", ...report.cancellation.boundaries.map((boundary) => `| ${boundary.phase} | ${boundary.status} | ${boundary.latencyMs === null ? "—" : boundary.latencyMs.toFixed(1)} | ${boundary.interpretation} |`));
  lines.push("", "## WASM release validation", "", `- Status: **${report.releaseWasm.status}**`, `- Build policy: \`${report.releaseWasm.buildCommand}\``, `- Assets: \`${report.releaseWasm.gluePath}\`, \`${report.releaseWasm.wasmPath}\``, `- Detail: ${report.releaseWasm.reason || `${report.releaseWasm.bytes} bytes; required exports verified`}`, "", "## Limitations", "", ...report.limitations.map((limitation) => `- ${limitation}`), "");
  return lines.join("\n");
}

export async function run(options = parseArgs(process.argv.slice(2))) {
  if (options.help) return { help: true };
  const outputDir = resolve(options.outputDir || `/tmp/atomic-clusters-large-vault-hardening-${Date.now()}`);
  await mkdir(outputDir, { recursive: true });
  const dataset = await readDataset(options.input);
  const inputSha256 = await sha256File(dataset.absolute);
  const releaseWasm = await inspectReleaseWasm();
  const host = {
    node: process.version,
    platform: platform(),
    osRelease: release(),
    arch: arch(),
    cpuCount: cpus().length,
    cpuModel: cpus()[0]?.model || "unknown",
    totalMemoryBytes: totalmem(),
    freeMemoryAtStartBytes: freemem(),
  };
  const report = {
    schemaVersion: 1,
    runner: "atomic-clusters-large-vault-hardening",
    generatedAt: new Date().toISOString(),
    options: { seed: options.seed, sampleSeed: options.sampleSeed, sizes: options.sizes, fast: options.fast, skipCancellation: options.skipCancellation, strictPreflight: options.strictPreflight },
    host,
    dataset: { path: dataset.absolute, sha256: inputSha256, sourceRecords: dataset.records.length, dimension: dataset.dimension, loadMs: dataset.loadMs },
    releaseWasm,
    scalePlan: [],
    runs: [],
    cancellation: { status: "unavailable", boundaries: [], reason: null },
    limitations: [
      "Embedding generation is excluded: the checked-in file contains precomputed Gemini vectors, so no Gemini/local provider or ORT preflight is exercised.",
      "Phase timings are progress-boundary timings emitted by clusterEmbeddings; synchronous sub-operations inside one boundary are not separately observable without source instrumentation.",
      "RSS is reported by Node process samples/resourceUsage; allocator sharing and Electron renderer memory are not separately measured.",
      "5,000 and 10,000 are unavailable until a real dataset with at least those rows is supplied; no extrapolation is made."
    ]
  };
  for (const size of LARGE_VAULT_SIZES) {
    if (size > dataset.records.length) {
      report.scalePlan.push(unavailableScaleEntry(size, dataset.records.length, dataset.dimension, `checked-in source has ${dataset.records.length} rows; no synthetic duplication or extrapolation was performed`));
      continue;
    }
    report.scalePlan.push({ size, status: options.sizes.includes(size) ? "scheduled" : "not_requested", dataset: { sourceRecords: dataset.records.length, dimension: dataset.dimension, duplicatedRows: false }, reason: options.sizes.includes(size) ? null : "not selected by --sizes" });
  }
  let workerSource = null;
  // An absent asset is a valid development fallback for the bounded
  // cancellation probe. A present-but-invalid asset is not safe to bundle.
  if (releaseWasm.status !== "failed") workerSource = await buildInstrumentedWorkerSource();
  for (const size of BENCHMARK_SIZES) {
    if (!options.sizes.includes(size)) continue;
    const result = size > dataset.records.length
      ? unavailableScaleEntry(size, dataset.records.length, dataset.dimension, "requested size exceeds source dataset")
      : await runBenchmark(size, dataset, options, workerSource, releaseWasm);
    report.runs.push(result);
    await writeFile(join(outputDir, `run-${size}.json`), `${JSON.stringify(result, null, 2)}\n`, "utf8");
  }
  if (!options.skipCancellation && workerSource) {
    const probeSize = Math.min(options.cancelProbeSize, dataset.records.length);
    if (probeSize >= 3) {
      const probeRecords = sampleRecords(dataset.records, probeSize, options.sampleSeed);
      report.cancellation = await runCancellationProbes(workerSource.source, probeRecords, options);
    } else report.cancellation = { status: "unavailable", boundaries: [], reason: "fewer than three source records" };
  } else report.cancellation = { status: "unavailable", boundaries: [], reason: options.skipCancellation ? "disabled by --skip-cancellation" : "worker source unavailable because release WASM validation failed" };
  report.scalePlan = report.scalePlan.map((entry) => {
    const measured = report.runs.find((run) => run.size === entry.size);
    return measured ? { ...entry, status: measured.status, reason: measured.limitation || entry.reason, preflight: measured.preflight || null } : entry;
  });
  const measuredCount = report.runs.filter((run) => run.status === "measured").length;
  report.status = measuredCount === options.sizes.length ? "complete" : measuredCount > 0 ? "partial" : "blocked";
  report.outputDir = outputDir;
  validateHardeningReport(report);
  await writeFile(join(outputDir, "hardening-report.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(join(outputDir, "hardening-report.md"), `${renderMarkdown(report)}\n`, "utf8");
  if (options.requireReleaseWasm && releaseWasm.status !== "passed") throw new Error(`release WASM requirement failed: ${releaseWasm.reason || releaseWasm.status}; report: ${join(outputDir, "hardening-report.json")}`);
  return report;
}

async function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) { console.log(usage()); return; }
    const report = await run(options);
    console.log(JSON.stringify({ outputDir: report.outputDir, status: report.status, releaseWasm: report.releaseWasm.status, scalePlan: report.scalePlan.map((entry) => ({ size: entry.size, status: entry.status })), cancellation: report.cancellation.status }, null, 2));
  } catch (error) {
    console.error(`large-vault hardening failed: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
