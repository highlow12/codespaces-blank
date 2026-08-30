import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadVisualization() {
  const source = await readFile(new URL("../src/visualization.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2022" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

test("global depth cut retains shallow leaves and dims non-selected branches", async () => {
  const { buildVisualizationTree, visualizationGlobalDepthFrontier } = await loadVisualization();
  const tree = buildVisualizationTree({ leaves: [0, 1, 2], merges: [{ id: 3, left: 0, right: 1, distance: 1, mass: 2 }], root: 9, children: { "9": [3, 2] } }, [0, 1, 2, -1]);
  const first = visualizationGlobalDepthFrontier(tree, 0);
  assert.deepEqual(first.map((entry) => entry.node.id), ["node:3", "node:2"]);
  const second = visualizationGlobalDepthFrontier(tree, 1, "node:3");
  assert.deepEqual(second.map((entry) => entry.node.id), ["node:0", "node:1", "node:2"]);
  assert.deepEqual(second.map((entry) => entry.opacity), [1, 1, .2]);
  assert.equal(new Set(second.flatMap((entry) => entry.pointIndices)).size, 3);
});

test("camera zoom keeps the pointer world coordinate fixed and clamps zoom", async () => {
  const { visualizationFitCameraState, visualizationCameraFromState, visualizationScreenToWorld, zoomVisualizationCameraAt } = await loadVisualization();
  const state = visualizationFitCameraState([[0, 0], [10, 10]], 400, 300);
  const point = [123, 87];
  const before = visualizationScreenToWorld(visualizationCameraFromState(state), point);
  const zoomed = zoomVisualizationCameraAt(state, point[0], point[1], 1000);
  const after = visualizationScreenToWorld(visualizationCameraFromState(zoomed), point);
  assert.equal(zoomed.zoom, 16);
  assert.ok(Math.abs(before[0] - after[0]) < 1e-9);
  assert.ok(Math.abs(before[1] - after[1]) < 1e-9);
});

test("terminal paths distinguish leaves, internal residuals, and root noise", async () => {
  const { buildVisualizationTree, visualizationNoteTerminalPath, visualizationGlobalDepthFrontier } = await loadVisualization();
  const tree = buildVisualizationTree({ leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 1, mass: 2 }], root: 2 }, [0, 1, -1]);
  const placements = [{ kind: "leaf", nodeId: 0, confidence: 1 }, { kind: "residual", nodeId: 2, confidence: .4 }, { kind: "residual", nodeId: null, confidence: 0 }];
  assert.deepEqual(visualizationNoteTerminalPath(tree, 0, [0, 1, -1], placements), ["root", "node:2", "node:0"]);
  assert.deepEqual(visualizationNoteTerminalPath(tree, 1, [0, 1, -1], placements), ["root", "node:2"]);
  assert.deepEqual(visualizationNoteTerminalPath(tree, 2, [0, 1, -1], placements), ["root"]);
  const entries = visualizationGlobalDepthFrontier(tree, 0, null, undefined, undefined, placements);
  assert.equal(entries[0].pointIndices.includes(1), false);
});
