import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadClustering() {
  const source = await readFile(new URL("../src/clustering.ts", import.meta.url));
  const stub = `const UMAP = class { constructor(options) { this.options = options; } setSupervisedProjection(labels, params) { this.labels = labels; this.supervised = params; } async fitAsync(rows, callback) { callback(rows.length * 5); return rows.map((row) => [row[0], row[1]]); } };`;
  const replaced = source.toString().replace('import { UMAP } from "umap-js";', stub);
  const result = await transform(replaced, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

test("visualization projection uses the bounded 2D supervised UMAP contract", async () => {
  const { projectVisualization } = await loadClustering();
  const rows = Array.from({ length: 25 }, (_, index) => [index, index * 2, 1]);
  const labels = rows.map((_, index) => index < 12 ? 0 : 1);
  const progress = [];
  const result = await projectVisualization(rows, labels, { seed: 42, onProgress: (phase, value) => progress.push([phase, value]) });
  assert.equal(result.coordinates.length, rows.length);
  assert.ok(result.coordinates.every((point) => point.length === 2 && point.every(Number.isFinite)));
  assert.deepEqual(result.labels, labels);
  assert.deepEqual(result.configuration, { runtime: "umap-js", seed: 42, nComponents: 2, nNeighbors: 24, minDist: 1, spread: 1.8, targetMetric: "categorical", targetWeight: 0.01 });
  assert.deepEqual(progress.at(-1), ["visualization", 1]);
});

test("visualization projection omits coordinates for too-small datasets", async () => {
  const { projectVisualization } = await loadClustering();
  assert.equal(await projectVisualization([[1, 2], [3, 4]], [0, 1]), undefined);
});

test("visualization projection observes cancellation before starting UMAP", async () => {
  const { projectVisualization } = await loadClustering();
  await assert.rejects(() => projectVisualization([[1], [2], [3]], [0, 0, 1], { signal: { cancelled: true } }), /Clustering cancelled/);
});
