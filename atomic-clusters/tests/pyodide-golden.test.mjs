import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { build } from "esbuild";

async function loadWasmKernel() {
  const result = await build({ entryPoints: ["src/wasm-kernel.ts"], bundle: true, format: "esm", platform: "node", target: "es2020", write: false, logLevel: "silent" });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].contents).toString("base64")}`);
}

test("Python/WASM golden fixture defines explicit tolerances and contracts", async () => {
  const fixture = JSON.parse(await readFile(new URL("./fixtures/python-wasm-golden.json", import.meta.url)));
  assert.equal(fixture.schema_version, 1);
  assert.ok(fixture.tolerances.pca_abs > 0 && fixture.tolerances.umap_abs > 0 && fixture.tolerances.hdbscan_abs > 0);
  assert.equal(fixture.embeddings.length, fixture.ids.length);
  assert.equal(fixture.expected.pca.selected_dimension, fixture.config.pca.components);
  assert.equal(fixture.expected.discovery.leaf_labels.length, fixture.embeddings.length);
  assert.equal(fixture.expected.discovery.cluster_count, 3);
});

test("WASM PCA and HDBSCAN satisfy the Python golden shape/value contract", async () => {
  const fixture = JSON.parse(await readFile(new URL("./fixtures/python-wasm-golden.json", import.meta.url)));
  const { WasmNumericKernel } = await loadWasmKernel();
  const wasm = await import("../wasm-core/pkg/atomic_clusters_wasm_core.js");
  wasm.initSync({ module: await readFile(new URL("../wasm-core/pkg/atomic_clusters_wasm_core_bg.wasm", import.meta.url)) });
  const kernel = new WasmNumericKernel(wasm);
  const pca = kernel.pca(fixture.embeddings, fixture.expected.pca.selected_dimension);
  assert.equal(pca.projected.length, fixture.embeddings.length);
  assert.equal(pca.projected[0].length, fixture.expected.pca.selected_dimension);
  assert.ok(pca.projected.flat().every(Number.isFinite));
  assert.ok(pca.explained.every((value) => value >= -fixture.tolerances.pca_abs));
  const hdbscan = fixture.wasm_hdbscan;
  const result = kernel.hdbscan(hdbscan.features, hdbscan.min_cluster_size, hdbscan.min_samples);
  assert.deepEqual(result.labels, hdbscan.expected_labels);
  assert.deepEqual(result.probabilities, hdbscan.expected_probabilities);
  assert.deepEqual(result.outlierProxy, hdbscan.expected_outlier_proxy);
});

test("the plugin source keeps the Python reference external", async () => {
  const sourceNames = await readdir(new URL("../src/", import.meta.url));
  const types = await Promise.all(sourceNames.filter((name) => name.endsWith(".ts")).map((name) => readFile(new URL(`../src/${name}`, import.meta.url), "utf8")));
  assert.ok(types.length > 0);
  assert.ok(types.every((source) => !/pyodide/i.test(source)));
  const pythonReference = await readFile(new URL("../../pyodide_core/atomic_clustering/__init__.py", import.meta.url), "utf8");
  assert.match(pythonReference, /cluster_documents/);
});
