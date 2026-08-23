#!/usr/bin/env node

/**
 * Offline end-to-end validation for the clustering orchestration.
 *
 * This intentionally runs the same clusterEmbeddings() entry point used by
 * the worker.  It does not call an embedding provider: the input is an
 * already-materialized Gemini embedding dataset.  When wasm-core/pkg exists,
 * its generated Rust/WASM bindings are initialized and passed through the
 * same WasmNumericKernel adapter used by the packaged worker.
 */

import { build } from "esbuild";
import { gunzip } from "node:zlib";
import { promisify } from "node:util";
import { readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, dirname, resolve, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const gunzipAsync = promisify(gunzip);
const scriptDir = dirname(fileURLToPath(import.meta.url));
const pluginDir = resolve(scriptDir, "..");
const repositoryDir = resolve(pluginDir, "..");
const defaultInput = join(repositoryDir, "dbpedia_gemini_embeddings.json.gz");

export function usage() {
  return `Usage: node scripts/offline-e2e.mjs [options]

Options:
  --input-json PATH             Gemini embedding JSON or .json.gz (default: repository dataset)
  --dataset-sample-size N      Deterministically cluster at most N rows
  --dataset-sample-seed N      Sampling seed (alias: --seed)
  --seed N                     Clustering and sampling seed (default: 42)
  --fast                       Use reduced PCA/UMAP settings for quick validation
  --output PATH                Write the JSON report (default: /tmp/atomic-clusters-offline-e2e-*.json)
  --help                       Show this help
`;
}

export function parseArgs(argv) {
  const options = {
    input: defaultInput,
    sampleSize: undefined,
    seed: 42,
    fast: false,
    output: undefined
  };
  const valueFor = (arg, index) => {
    const inline = arg.indexOf("=");
    if (inline >= 0) return [arg.slice(inline + 1), index];
    if (index + 1 >= argv.length || argv[index + 1].startsWith("--")) {
      throw new Error(`${arg} requires a value`);
    }
    return [argv[index + 1], index + 1];
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") return { help: true };
    if (arg === "--fast") { options.fast = true; continue; }
    if (arg === "--input-json" || arg.startsWith("--input-json=")) {
      const [value, next] = valueFor(arg, index); options.input = value; index = next; continue;
    }
    if (arg === "--output" || arg === "--output-json" || arg.startsWith("--output=") || arg.startsWith("--output-json=")) {
      const [value, next] = valueFor(arg, index); options.output = value; index = next; continue;
    }
    if (arg === "--dataset-sample-size" || arg === "--sample-size" || arg.startsWith("--dataset-sample-size=") || arg.startsWith("--sample-size=")) {
      const [value, next] = valueFor(arg, index); options.sampleSize = positiveInteger(value, "dataset sample size"); index = next; continue;
    }
    if (arg === "--seed" || arg === "--dataset-sample-seed" || arg.startsWith("--seed=") || arg.startsWith("--dataset-sample-seed=")) {
      const [value, next] = valueFor(arg, index); options.seed = safeInteger(value, "seed"); index = next; continue;
    }
    throw new Error(`Unknown option: ${arg}`);
  }
  return options;
}

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

async function readDataset(inputPath) {
  const absolute = resolve(inputPath);
  if (basename(absolute) === "dbpedia_label_embeddings.json") {
    throw new Error("Refusing dbpedia_label_embeddings.json; use the 3,000-record Gemini embedding dataset.");
  }
  if (!existsSync(absolute)) throw new Error(`Input dataset does not exist: ${absolute}`);
  const started = performance.now();
  const bytes = await readFile(absolute);
  const raw = absolute.endsWith(".gz") ? await gunzipAsync(bytes) : bytes;
  let parsed;
  try { parsed = JSON.parse(raw.toString("utf8")); } catch (error) {
    throw new Error(`Input dataset is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!Array.isArray(parsed) || parsed.length === 0) throw new Error("Input dataset must be a non-empty JSON array of records.");
  const dimension = parsed[0]?.embedding?.length;
  if (!Number.isSafeInteger(dimension) || dimension < 1) throw new Error("Records must contain a non-empty embedding array.");
  const records = parsed.map((record, index) => {
    if (!record || !Array.isArray(record.embedding) || record.embedding.length !== dimension) {
      throw new Error(`Embedding at record ${index} is not rectangular (expected ${dimension} values).`);
    }
    if (!record.embedding.every((value) => typeof value === "number" && Number.isFinite(value))) {
      throw new Error(`Embedding at record ${index} contains a non-finite or non-numeric value.`);
    }
    return {
      id: `dbpedia-${index}`,
      class: typeof record.class === "string" ? record.class : null,
      classHierarchy: Array.isArray(record.class_hierarchy) ? record.class_hierarchy : [],
      embedding: record.embedding
    };
  });
  return { absolute, records, dimension, loadMs: performance.now() - started };
}

export function sampleRecords(records, sampleSize, seed) {
  const count = Math.min(sampleSize ?? records.length, records.length);
  if (count < 3) throw new Error("At least 3 records are required for clustering.");
  // A full-dataset validation must mirror the real plugin path, which keeps
  // vault order and lets clusterEmbeddings apply its own PCA sample ordering.
  if (count === records.length) return records.slice();
  const indices = Array.from({ length: records.length }, (_, index) => index);
  let state = seed >>> 0;
  for (let index = indices.length - 1; index > 0; index--) {
    state = Math.imul(state ^ (state >>> 16), 2246822519) >>> 0;
    const swap = state % (index + 1);
    [indices[index], indices[swap]] = [indices[swap], indices[index]];
  }
  return indices.slice(0, count).map((index) => records[index]);
}

async function loadOrchestration() {
  // esbuild turns the repository's TypeScript orchestration into a temporary
  // in-memory ESM module, avoiding generated source files in the plugin tree.
  const clusteringPath = resolve(pluginDir, "src/clustering.ts");
  const kernelPath = resolve(pluginDir, "src/wasm-kernel.ts");
  const source = `import { clusterEmbeddings } from ${JSON.stringify(clusteringPath)};
import { WasmNumericKernel } from ${JSON.stringify(kernelPath)};
export { clusterEmbeddings, WasmNumericKernel };`;
  const result = await build({
    stdin: { contents: source, resolveDir: pluginDir, sourcefile: "offline-e2e-entry.ts", loader: "ts" },
    bundle: true,
    format: "esm",
    platform: "node",
    target: "node20",
    write: false,
    sourcemap: false,
    logLevel: "silent"
  });
  const code = result.outputFiles[0].text;
  return import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}`);
}

async function loadWasm(plugin) {
  const gluePath = resolve(pluginDir, "wasm-core/pkg/atomic_clusters_wasm_core.js");
  const wasmPath = resolve(pluginDir, "wasm-core/pkg/atomic_clusters_wasm_core_bg.wasm");
  if (!existsSync(gluePath) || !existsSync(wasmPath)) return { kernel: undefined, loaded: false, exports: [] };
  const wasm = await import(pathToFileURL(gluePath).href);
  wasm.initSync({ module: new WebAssembly.Module(await readFile(wasmPath)) });
  const names = ["normalize", "matmul", "pca", "randomized_pca", "cosine_distances", "exact_knn", "exact_knn_cosine_tiled", "euclidean_mutual_reachability_mst", "mst", "mutual_reachability_mst", "hdbscan_extract", "HnswIndex"];
  const bindings = Object.fromEntries(names.filter((name) => typeof wasm[name] === "function").map((name) => [name, wasm[name]]));
  return { kernel: new plugin.WasmNumericKernel(bindings), loaded: true, exports: Object.keys(bindings) };
}

function clusteringConfig(options, count, dimension) {
  if (!options.fast) return { seed: options.seed };
  return {
    seed: options.seed,
    pcaSampleSize: Math.min(500, count),
    pcaMaxComponents: Math.min(64, dimension, Math.max(1, count - 1)),
    umapComponents: Math.min(10, Math.max(2, count - 1)),
    umapNeighbors: Math.min(10, Math.max(2, count - 1)),
    minClusterSize: Math.min(5, Math.max(2, Math.floor(count / 10)))
  };
}

function summarize(result, records) {
  const sizes = new Map();
  result.leafLabels.forEach((label) => { if (label >= 0) sizes.set(label, (sizes.get(label) || 0) + 1); });
  const clusterSizes = [...sizes.values()].sort((a, b) => b - a);
  const noiseCount = result.leafLabels.filter((label) => label < 0).length;
  return {
    recordCount: records.length,
    dimension: records[0].embedding.length,
    clusterCount: sizes.size,
    clusteredCount: records.length - noiseCount,
    noiseCount,
    noiseRate: noiseCount / records.length,
    clusterSizes,
    largestClusterSize: clusterSizes[0] || 0,
    medianClusterSize: clusterSizes.length ? clusterSizes[Math.floor((clusterSizes.length - 1) / 2)] : 0,
    probabilityMean: mean(result.probabilities),
    outlierProxyMean: mean(result.outlierProxy),
    pca: result.pca,
    hierarchy: {
      leafCount: result.hierarchy.leaves.length,
      mergeCount: result.hierarchy.merges.length,
      root: result.hierarchy.root
    }
  };
}

function mean(values) { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0; }

export async function run(options) {
  const started = performance.now();
  const dataset = await readDataset(options.input);
  const records = sampleRecords(dataset.records, options.sampleSize, options.seed);
  const config = clusteringConfig(options, records.length, dataset.dimension);
  const orchestrationStarted = performance.now();
  const plugin = await loadOrchestration();
  const wasm = await loadWasm(plugin);
  const progress = [];
  const result = await plugin.clusterEmbeddings(
    records.map((record) => record.id),
    records.map((record) => record.embedding),
    config,
    { kernel: wasm.kernel, onProgress: (phase, value) => progress.push({ phase, progress: value, atMs: performance.now() - orchestrationStarted }) }
  );
  const orchestrationMs = performance.now() - orchestrationStarted;
  const report = {
    schemaVersion: 1,
    runner: "atomic-clusters-offline-e2e",
    generatedAt: new Date().toISOString(),
    dataset: {
      path: dataset.absolute,
      sourceRecords: dataset.records.length,
      sampledRecords: records.length,
      dimension: dataset.dimension,
      samplingSeed: options.seed,
      sampleSizeRequested: options.sampleSize ?? null
    },
    options: { seed: options.seed, fast: options.fast, config },
    runtime: { wasmLoaded: wasm.loaded, wasmExports: wasm.exports, orchestration: "clusterEmbeddings" },
    timings: { loadMs: dataset.loadMs, orchestrationMs, clusterTotalMs: result.timings.totalMs, totalMs: performance.now() - started },
    progress,
    summary: summarize(result, records),
    assignments: records.map((record, index) => ({ id: record.id, class: record.class, classHierarchy: record.classHierarchy, label: result.leafLabels[index], probability: result.probabilities[index], outlierProxy: result.outlierProxy[index] }))
  };
  return report;
}

async function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) { console.log(usage()); return; }
    const report = await run(options);
    const output = resolve(options.output || `/tmp/atomic-clusters-offline-e2e-${Date.now()}.json`);
    await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    console.log(JSON.stringify({ output, ...report.summary, wasmLoaded: report.runtime.wasmLoaded, timings: report.timings }, null, 2));
  } catch (error) {
    console.error(`offline E2E validation failed: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
