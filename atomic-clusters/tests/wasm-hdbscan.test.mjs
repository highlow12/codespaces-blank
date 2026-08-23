import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { build } from "esbuild";

async function loadWasmKernel() {
  const result = await build({
    entryPoints: ["src/wasm-kernel.ts"], bundle: true, format: "esm", platform: "node",
    target: "es2020", write: false, logLevel: "silent"
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].contents).toString("base64")}`);
}

async function loadClustering() {
  const result = await build({
    entryPoints: ["src/clustering.ts"], bundle: true, format: "esm", platform: "node",
    target: "es2020", write: false, logLevel: "silent"
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].contents).toString("base64")}`);
}

test("default provider chooses a WASM HDBSCAN implementation when present", async () => {
  const { DeterministicHdbscanProvider } = await loadClustering();
  const calls = [];
  const kernel = { hdbscan(rows, minClusterSize, minSamples) { calls.push([rows, minClusterSize, minSamples]); return { labels: [0, 0], probabilities: [1, 1], outlierProxy: [0, 0] }; } };
  const result = new DeterministicHdbscanProvider().fit([[1], [2]], 7, 3, kernel);
  assert.deepEqual(result.labels, [0, 0]);
  assert.deepEqual(calls[0].slice(1), [7, 3]);
});

test("WASM HDBSCAN contract forwards distinct minSamples and returns typed output", async () => {
  const { WasmNumericKernel } = await loadWasmKernel();
  const calls = [];
  const wasm = {
    normalize: (rows) => rows, matmul: () => [], pca: () => ({ projected: [], explained: [] }), cosine_distances: () => [], exact_knn: () => [], mst: () => [],
    euclidean_mutual_reachability_mst(rows, rowCount, dimension, minSamples, tile) {
      calls.push(["euclidean-mst", rows.length, rowCount, dimension, minSamples, tile]);
      return { edge_count: rowCount - 1, edges: new Float32Array([0, 1, 0.1, 1, 2, 0.1, 2, 3, 1, 3, 4, 0.1, 4, 5, 0.1]) };
    },
    hdbscan_extract(edges, rowCount, minClusterSize, selectionMethod, allowSingleCluster) {
      calls.push(["extract", Array.from(edges), rowCount, minClusterSize, selectionMethod, allowSingleCluster]);
      return { labels: new Int32Array([0, 0, 0, 1, 1, 1]), probabilities: new Float32Array(6).fill(1), outlier_scores: new Float32Array(6), cluster_count: 2 };
    }
  };
  const result = new WasmNumericKernel(wasm).hdbscan([[1, 0], [0.9, 0.1], [0.8, 0.2], [-1, 0], [-0.9, -0.1], [-0.8, -0.2]], 3, 2);
  assert.deepEqual(result.labels, [0, 0, 0, 1, 1, 1]);
  assert.deepEqual(calls.find(([name]) => name === "euclidean-mst").slice(2), [6, 2, 2, 256]);
  assert.deepEqual(calls.find(([name]) => name === "extract").slice(2), [6, 3, 1, false]);
});

test("Euclidean HDBSCAN sends its complete mutual-reachability MST directly to extraction", async () => {
  const { WasmNumericKernel } = await loadWasmKernel();
  let extracted = null;
  const wasm = {
    normalize: (rows) => rows, matmul: () => [], pca: () => ({ projected: [], explained: [] }), cosine_distances: () => [], exact_knn: () => [], mst: () => [],
    euclidean_mutual_reachability_mst(_rows, rowCount) { return { edge_count: rowCount - 1, edges: new Float32Array([0, 1, 0.1, 1, 2, 1, 2, 3, 0.1]) }; },
    hdbscan_extract(edges, rowCount) { extracted = { edges: Array.from(edges), rowCount }; return { labels: new Int32Array(rowCount).fill(-1), probabilities: new Float32Array(rowCount), outlier_scores: new Float32Array(rowCount).fill(1), cluster_count: 0 }; }
  };
  const rows = [[1, 0], [0.99, 0.1], [-1, 0], [-0.99, -0.1]];
  new WasmNumericKernel(wasm).hdbscan(rows, 2, 2);
  assert.equal(extracted.rowCount, 4);
  assert.equal(extracted.edges.length, 9);
  assert.deepEqual(extracted.edges.slice(0, 3), [0, 1, extracted.edges[2]]);
});

test("packaged WASM + TypeScript adapter extracts two synthetic HDBSCAN clusters", async () => {
  const { WasmNumericKernel } = await loadWasmKernel();
  const wasm = await import("../wasm-core/pkg/atomic_clusters_wasm_core.js");
  wasm.initSync({ module: await readFile(new URL("../wasm-core/pkg/atomic_clusters_wasm_core_bg.wasm", import.meta.url)) });
  const rows = [[1, 0], [0.995, 0.1], [0.98, 0.2], [-1, 0], [-0.995, -0.1], [-0.98, -0.2]];
  const result = new WasmNumericKernel(wasm).hdbscan(rows, 3, 2);
  assert.deepEqual(result.labels, [0, 0, 0, 1, 1, 1]);
  assert.deepEqual(result.probabilities, [1, 1, 1, 1, 1, 1]);
  assert.deepEqual(result.outlierProxy, [0, 0, 0, 0, 0, 0]);
});
