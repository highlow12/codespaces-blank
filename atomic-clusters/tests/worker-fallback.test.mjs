import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadFallbackWorker() {
  const source = await readFile(new URL("../src/worker-client.ts", import.meta.url), "utf8");
  const stubbed = source
    .replace('import { Worker as NodeWorker } from "worker_threads";', 'const NodeWorker = class { constructor() { throw new Error("worker_threads unavailable"); } };')
    .replace('import { clusterEmbeddings } from "./clustering";', 'const clusterEmbeddings = async (ids, embeddings, config, options) => { options.onProgress?.("pca", 0.5); await new Promise((resolve) => setTimeout(resolve, 0)); if (options.signal?.cancelled) throw new Error("Clustering cancelled"); options.onProgress?.("complete", 1); return { schemaVersion: 1, ids, leafLabels: embeddings.map(() => -1), probabilities: embeddings.map(() => 0), outlierProxy: embeddings.map(() => 1), pca: { selected: 1, explainedVariance: 1, totalVariance: 0, candidates: [1], sampleSize: embeddings.length, varianceTarget: 0.9 }, hierarchy: { leaves: [], merges: [], root: null }, timings: { totalMs: 0 } }; };')
    .replace('import { ClusteringConfig, ClusterResult, WorkerRequest, WorkerResponse } from "./types";', '')
    .replace('import { loadWasmKernel } from "./wasm-loader";', 'const loadWasmKernel = () => undefined;');
  const result = await transform(stubbed, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

test("in-process clustering fallback reports progress and returns a result", async () => {
  const { NodeClusteringWorker, InProcessClusteringWorker } = await loadFallbackWorker();
  await assert.rejects(() => new NodeClusteringWorker("broken worker source").init(), /worker_threads unavailable/);
  const worker = new InProcessClusteringWorker();
  await worker.init();
  const progress = [];
  const result = await worker.run(["a", "b"], [[1], [2]], {}, (phase, value) => progress.push([phase, value]));
  assert.deepEqual(result.ids, ["a", "b"]);
  assert.deepEqual(progress, [["pca", 0.5], ["complete", 1]]);
});

test("in-process clustering fallback cancels before starting work", async () => {
  const { InProcessClusteringWorker } = await loadFallbackWorker();
  const worker = new InProcessClusteringWorker();
  const run = worker.run(["a"], [[1]], {});
  worker.cancel();
  await assert.rejects(run, /Clustering cancelled/);
});
