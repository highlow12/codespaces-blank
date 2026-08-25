#!/usr/bin/env node

/**
 * Compare the authoritative Python HDBSCAN result with the packaged WASM
 * extractor on the same UMAP coordinates. This is an audit report, not a
 * parity claim: umap-learn/umap-js differences are intentionally outside its
 * scope. Use --strict to turn supplied metric limits into a CI gate.
 */
import { build } from "esbuild";
import { readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { basename, dirname, resolve, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pluginDir = resolve(scriptDir, "..");
const repositoryDir = resolve(pluginDir, "..");
const defaultInput = join(repositoryDir, "dbpedia_gemini_embeddings.json.gz");

export function parseArgs(argv) {
  const options = { input: defaultInput, sampleSize: 100, sampleSeed: 42, seed: 42, fast: false, output: undefined, strict: false, minLabelAgreement: undefined, maxProbabilityMae: undefined, maxOutlierMae: undefined };
  const valueFor = (arg, index) => { const inline = arg.indexOf("="); if (inline >= 0) return [arg.slice(inline + 1), index]; if (index + 1 >= argv.length || argv[index + 1].startsWith("--")) throw new Error(`${arg} requires a value`); return [argv[index + 1], index + 1]; };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") return { help: true };
    if (arg === "--fast") { options.fast = true; continue; }
    if (arg === "--strict") { options.strict = true; continue; }
    const names = [
      ["--input-json", "input"], ["--output-json", "output"], ["--output", "output"],
      ["--dataset-sample-size", "sampleSize"], ["--dataset-sample-seed", "sampleSeed"], ["--seed", "seed"],
      ["--min-label-agreement", "minLabelAgreement"], ["--max-probability-mae", "maxProbabilityMae"], ["--max-outlier-mae", "maxOutlierMae"]
    ];
    const match = names.find(([name]) => arg === name || arg.startsWith(`${name}=`));
    if (match) { const [value, next] = valueFor(arg, index); options[match[1]] = match[1] === "input" || match[1] === "output" ? value : number(value, match[1]); index = next; continue; }
    throw new Error(`Unknown option: ${arg}`);
  }
  return options;
}

function number(value, name) { const parsed = Number(value); if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${name} must be a non-negative number`); return parsed; }

export function usage() {
  return `Usage: node scripts/hdbscan-parity-audit.mjs [options]\n\nOptions:\n  --input-json PATH             Gemini embeddings (.json or .json.gz)\n  --dataset-sample-size N       Python sample size (default: 100)\n  --dataset-sample-seed N       Python sample seed (default: 42)\n  --seed N                      PCA/UMAP seed (default: 42)\n  --fast                        Use small deterministic PCA/UMAP settings\n  --output-json PATH            Audit report path (default: /tmp/atomic-clusters-hdbscan-parity-*.json)\n  --strict                      Fail if supplied metric limits are exceeded\n  --min-label-agreement N       Strict label agreement lower bound\n  --max-probability-mae N       Strict probability MAE upper bound\n  --max-outlier-mae N           Strict outlier MAE upper bound\n`;
}

async function loadParityModule() {
  const source = `import { compareHdbscanOutputs } from ${JSON.stringify(resolve(pluginDir, "src/hdbscan-parity.ts"))};\nimport { WasmNumericKernel } from ${JSON.stringify(resolve(pluginDir, "src/wasm-kernel.ts"))};\nexport { compareHdbscanOutputs, WasmNumericKernel };`;
  const result = await build({ stdin: { contents: source, resolveDir: pluginDir, sourcefile: "hdbscan-parity-audit-entry.ts", loader: "ts" }, bundle: true, format: "esm", platform: "node", target: "node20", write: false, logLevel: "silent" });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].contents).toString("base64")}`);
}

async function loadWasm() {
  const gluePath = resolve(pluginDir, "wasm-core/pkg/atomic_clusters_wasm_core.js");
  const wasmPath = resolve(pluginDir, "wasm-core/pkg/atomic_clusters_wasm_core_bg.wasm");
  if (!existsSync(gluePath) || !existsSync(wasmPath)) throw new Error("WASM parity audit requires wasm-core/pkg; run npm run build:wasm first");
  const wasm = await import(pathToFileURL(gluePath).href);
  wasm.initSync({ module: new WebAssembly.Module(await readFile(wasmPath)) });
  return wasm;
}

function runPython(options, referencePath) {
  if (basename(resolve(options.input)) === "dbpedia_label_embeddings.json") throw new Error("Refusing dbpedia_label_embeddings.json; use the Gemini dataset");
  const args = ["hdbscan_wasm_parity_audit.py", "--input-json", resolve(options.input), "--output-json", referencePath, "--dataset-sample-size", String(options.sampleSize), "--dataset-sample-seed", String(options.sampleSeed), "--seed", String(options.seed)];
  if (options.fast) args.push("--fast");
  const processResult = spawnSync(resolve(repositoryDir, ".venv/bin/python"), args, { cwd: repositoryDir, encoding: "utf8" });
  if (processResult.status !== 0) throw new Error(`Python parity fixture failed:\n${processResult.stdout}\n${processResult.stderr}`);
}

export async function run(options) {
  const referencePath = `/tmp/atomic-clusters-hdbscan-reference-${process.pid}.json`;
  runPython(options, referencePath);
  const reference = JSON.parse(await readFile(referencePath, "utf8"));
  const { compareHdbscanOutputs, WasmNumericKernel } = await loadParityModule();
  const wasm = await loadWasm();
  const names = ["normalize", "matmul", "pca", "randomized_pca", "cosine_distances", "exact_knn", "exact_knn_cosine_tiled", "euclidean_mutual_reachability_mst", "mst", "mutual_reachability_mst", "hdbscan_extract", "HnswIndex"];
  const bindings = Object.fromEntries(names.filter((name) => typeof wasm[name] === "function").map((name) => [name, wasm[name]]));
  const config = reference.configuration;
  const candidate = new WasmNumericKernel(bindings).hdbscan(reference.features, config.min_cluster_size, config.min_samples);
  const metrics = compareHdbscanOutputs(reference.reference, candidate);
  const report = {
    schemaVersion: 1,
    contract: "hdbscan-membership-v1",
    parityClaim: false,
    comparisonScope: reference.comparisonScope,
    dataset: reference.dataset,
    configuration: config,
    metrics,
    interpretation: "Metrics compare HDBSCAN only on identical Python UMAP coordinates. They do not establish end-to-end Python/plugin parity; full soft cross-cluster memberships are authoritative only on the Python side.",
    strict: options.strict,
    wasmExports: Object.keys(bindings)
  };
  const failures = [];
  if (options.minLabelAgreement !== undefined && metrics.labelAgreement < options.minLabelAgreement) failures.push(`labelAgreement ${metrics.labelAgreement} < ${options.minLabelAgreement}`);
  if (options.maxProbabilityMae !== undefined && metrics.probabilityMae > options.maxProbabilityMae) failures.push(`probabilityMae ${metrics.probabilityMae} > ${options.maxProbabilityMae}`);
  if (options.maxOutlierMae !== undefined && metrics.outlierMae > options.maxOutlierMae) failures.push(`outlierMae ${metrics.outlierMae} > ${options.maxOutlierMae}`);
  report.strictFailures = failures;
  if (options.strict && failures.length) { report.status = "failed"; } else report.status = "informational";
  const output = resolve(options.output || `/tmp/atomic-clusters-hdbscan-parity-${Date.now()}.json`);
  await writeFile(output, `${JSON.stringify({ ...report, output }, null, 2)}\n`, "utf8");
  if (options.strict && failures.length) throw new Error(`HDBSCAN parity audit failed: ${failures.join("; ")}`);
  return { ...report, output };
}

async function main() {
  try { const options = parseArgs(process.argv.slice(2)); if (options.help) { console.log(usage()); return; } const report = await run(options); console.log(JSON.stringify({ output: report.output, status: report.status, metrics: report.metrics }, null, 2)); }
  catch (error) { console.error(`HDBSCAN parity audit failed: ${error instanceof Error ? error.message : String(error)}`); process.exitCode = 1; }
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
