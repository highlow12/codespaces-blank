import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadVisualization() {
  const source = await readFile(new URL("../src/visualization.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

test("visualization helpers preserve aspect ratio and center degenerate coordinates", async () => {
  const { scaleVisualizationPoints } = await loadVisualization();
  const points = scaleVisualizationPoints([[0, 0], [10, 5]], 220, 120, 10);
  assert.deepEqual(points[0], [10, 110]);
  assert.deepEqual(points[1], [210, 10]);
  const flat = scaleVisualizationPoints([[2, 4], [2, 4]], 100, 80, 10);
  assert.deepEqual(flat, [[50, 40], [50, 40]]);
  assert.ok(flat.flat().every(Number.isFinite));
});

test("visualization colors are stable and noise is neutral", async () => {
  const { visualizationColor, VISUALIZATION_NOISE_COLOR } = await loadVisualization();
  assert.equal(visualizationColor(-1), VISUALIZATION_NOISE_COLOR);
  assert.equal(visualizationColor(0), "#1f77b4");
  assert.equal(visualizationColor(20), visualizationColor(0));
  assert.equal(visualizationColor(Number.NaN), VISUALIZATION_NOISE_COLOR);
});

test("nearest-point hit testing returns only the closest point within radius", async () => {
  const { findNearestVisualizationPoint } = await loadVisualization();
  assert.equal(findNearestVisualizationPoint([[10, 10], [12, 10], [50, 50]], 11.5, 10, 3), 1);
  assert.equal(findNearestVisualizationPoint([[10, 10]], 20, 20, 3), null);
});
