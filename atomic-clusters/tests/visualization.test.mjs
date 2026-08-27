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
  const { scaleVisualizationPoints, VISUALIZATION_POINT_PADDING } = await loadVisualization();
  const points = scaleVisualizationPoints([[0, 0], [10, 5]], 220, 120, 10);
  assert.deepEqual(points[0], [10, 110]);
  assert.deepEqual(points[1], [210, 10]);
  const flat = scaleVisualizationPoints([[2, 4], [2, 4]], 100, 80, 10);
  assert.deepEqual(flat, [[50, 40], [50, 40]]);
  assert.ok(flat.flat().every(Number.isFinite));
  const padded = scaleVisualizationPoints([[0, 0], [10, 5]], 220, 120);
  assert.equal(VISUALIZATION_POINT_PADDING, 18);
  assert.ok(padded.flatMap(([x, y]) => [x, y]).every((value) => value >= 18 && value <= 202));
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

test("adapter exposes a virtual root and generalized children with soft residuals", async () => {
  const { buildVisualizationTree, residualPointIndices, blendVisualizationColor, validateVisualizationData, visualizationCloudGeometry, visualizationPath, visualizationParent } = await loadVisualization();
  const hierarchy = { leaves: [0, 1, 2], merges: [{ id: 3, left: 0, right: 1, distance: 1, mass: 2 }, { id: 4, left: 3, right: 2, distance: 2, mass: 3 }], root: 4 };
  const tree = buildVisualizationTree(hierarchy, [0, 1, 2, -1]);
  assert.equal(tree.id, "root");
  assert.equal(tree.children.length, 1);
  assert.deepEqual(tree.children[0].children[0].leafLabels, [0, 1]);
  assert.deepEqual(residualPointIndices(tree, [0, 1, 2, -1], [[.8, .2, 0], [.5, .5, 0], [0, 0, 1], [0, 0, 0]], [0, 1, 2]), [3]);
  assert.deepEqual(residualPointIndices(tree.children[0], [0, 1, 2], [[.5, .5, 0], [.5, .5, 0], [.5, .5, 0]], [0, 1, 2]), []);
  assert.notEqual(blendVisualizationColor([.4, .3, 0], [0, 1, 2]), "#9aa0a6");
  assert.ok(blendVisualizationColor([.8, .8, .8], [0, 1, 2]).startsWith("#"));
  assert.deepEqual(visualizationCloudGeometry([[0, 0], [10, 0]], [0, 1], 100, 100), { x: 5, y: 0, radius: 32.5 });
  assert.deepEqual(visualizationPath(tree, tree.children[0].children[0].children[0].id).map((node) => node.id), ["root", "node:4", "node:3", "node:0"]);
  assert.equal(visualizationParent(tree, "node:3").id, "node:4");
  const valid = { schemaVersion: 4, ids: ["a"], leafLabels: [0], leafOrdering: [0], memberships: [[.6]], hierarchy: { leaves: [0], merges: [], root: 0 }, visualization: { coordinates: [[1, 2]], labels: [0], leafOrdering: [0], memberships: [[.6]], configuration: { runtime: "test", seed: 1, nComponents: 2, nNeighbors: 1, minDist: 0, spread: 1 } } };
  assert.equal(validateVisualizationData(valid), true);
  assert.equal(validateVisualizationData({ ...valid, memberships: [[1.2]] }), false);
  assert.equal(validateVisualizationData({ ...valid, memberships: [[.5, .6]] }), false);
  assert.equal(validateVisualizationData({ ...valid, leafOrdering: undefined }), false);
  assert.equal(validateVisualizationData({ ...valid, memberships: [[Number.NaN]] }), false);
  assert.equal(validateVisualizationData({ ...valid, visualization: { ...valid.visualization, labels: [99] } }), false);
});

test("n-ary adapter rejects cycles and preserves immediate children", async () => {
  const { buildVisualizationTree } = await loadVisualization();
  const tree = buildVisualizationTree({ leaves: [0, 1, 2], root: 9, children: { "9": [0, 1, 2] } }, [0, 1, 2]);
  assert.equal(tree.children[0].children.length, 3);
  assert.throws(() => buildVisualizationTree({ leaves: [0], root: 9, children: { "9": [9] } }, [0]), /cycle/);
  assert.throws(() => buildVisualizationTree({ leaves: [0], root: 9, children: { "9": [0, 0] } }, [0]), /duplicate/);
  assert.throws(() => buildVisualizationTree({ leaves: [0, 0], root: 0, children: {} }, [0]), /duplicate/);
  assert.throws(() => buildVisualizationTree({ leaves: [0], root: 9, children: { "9": ["bad"] } }, [0]), /malformed/);
});

test("bandwidth is robust, clamped, and uses the third neighbor", async () => {
  const { visualizationBaseBandwidth } = await loadVisualization();
  const bandwidth = visualizationBaseBandwidth([[0, 0], [1, 0], [0, 1], [1, 1], [10, 10]]);
  assert.ok(Number.isFinite(bandwidth) && bandwidth > 0);
  assert.ok(bandwidth >= Math.hypot(10, 10) / 500 && bandwidth <= Math.hypot(10, 10) / 30);
  assert.ok(Number.isFinite(visualizationBaseBandwidth([[2, 2], [2, 2], [2, 2], [2, 2]])));
});

test("stage sigma is monotonic, capped, and gives leaves a crisp lower bound", async () => {
  const { visualizationStageSigma } = await loadVisualization();
  assert.equal(visualizationStageSigma(2, 0), 2);
  assert.ok(visualizationStageSigma(2, 2) > visualizationStageSigma(2, 1));
  assert.equal(visualizationStageSigma(2, 100), 8);
  assert.equal(visualizationStageSigma(2, 100, true), 1.3);
});

test("kernel scale has a bounded default multiplier and preserves depth adaptation", async () => {
  const { clampVisualizationKernelScale, visualizationScaledStageSigma, visualizationStageSigma, VISUALIZATION_KERNEL_SCALE_DEFAULT, VISUALIZATION_KERNEL_SCALE_MIN, VISUALIZATION_KERNEL_SCALE_MAX } = await loadVisualization();
  assert.equal(VISUALIZATION_KERNEL_SCALE_DEFAULT, 0.65);
  assert.equal(clampVisualizationKernelScale(-1), VISUALIZATION_KERNEL_SCALE_MIN);
  assert.equal(clampVisualizationKernelScale(4), VISUALIZATION_KERNEL_SCALE_MAX);
  assert.equal(clampVisualizationKernelScale(Number.NaN), VISUALIZATION_KERNEL_SCALE_DEFAULT);
  assert.equal(visualizationScaledStageSigma(2, 2, false, 0.5), visualizationStageSigma(2, 2) * 0.5);
  assert.ok(visualizationScaledStageSigma(2, 2, false, 0.5) > visualizationScaledStageSigma(2, 1, false, 0.5));
});

test("tiny membership mass retains its hue while unexplained mass controls amplitude", async () => {
  const { blendVisualizationColor, visualizationMembershipAmplitude } = await loadVisualization();
  assert.equal(blendVisualizationColor([0.001, 0], [0, 1]), "#1f77b4");
  assert.ok(visualizationMembershipAmplitude([0.001, 0], 1) < visualizationMembershipAmplitude([1, 0], 1));
});

test("density accumulates separate lobes and clips at three sigma", async () => {
  const { accumulateVisualizationDensity, visualizationDensityAt, visualizationDensityAlpha } = await loadVisualization();
  const splats = [{ x: 2, y: 2, sigma: 1, color: [255, 0, 0], amplitude: 1 }, { x: 7, y: 2, sigma: 1, color: [0, 0, 255], amplitude: 1 }];
  const field = accumulateVisualizationDensity(splats, 10, 5);
  assert.ok(field.density[2 * 10 + 2] > field.density[2 * 10 + 4]);
  assert.ok(field.density[2 * 10 + 7] > field.density[2 * 10 + 4]);
  assert.equal(visualizationDensityAt([splats[0]], 6, 2), 0);
  assert.equal(visualizationDensityAlpha(1), 1 - Math.exp(-1));
});

test("frontier expands only the active cloud and emits residuals once at the boundary", async () => {
  const { buildVisualizationTree, visualizationFrontier } = await loadVisualization();
  const hierarchy = { leaves: [0, 1, 2], merges: [{ id: 3, left: 0, right: 1, distance: 1, mass: 2 }, { id: 4, left: 3, right: 2, distance: 2, mass: 3 }], root: 4 };
  const tree = buildVisualizationTree(hierarchy, [0, 1, 2, -1, 0, 2]);
  const rows = [[.5, .5, 0], [.5, .5, 0], [0, 0, 1], [0, 0, 0], [.5, .5, 0], [0, 0, 0]];
  const expanded = visualizationFrontier(tree, ["root"], rows, [0, 1, 2]);
  assert.deepEqual(expanded.map((entry) => entry.node.id), ["node:3", "node:2"]);
  assert.deepEqual(expanded.flatMap((entry) => entry.residualIndices), [3, 5]);
  assert.deepEqual(expanded[1].pointIndices, [2]);
  assert.equal(expanded[1].actualPoints, false);
  const deeper = visualizationFrontier(tree, ["root", "node:3"], rows, [0, 1, 2]);
  assert.deepEqual(deeper.map((entry) => entry.node.id), ["node:0", "node:1", "node:2"]);
  assert.deepEqual(deeper.flatMap((entry) => entry.residualIndices), [3, 5]);
  const selectedLeaf = visualizationFrontier(tree, ["root", "node:2"], rows, [0, 1, 2]);
  assert.equal(selectedLeaf.find((entry) => entry.node.id === "node:2").actualPoints, true);
  assert.deepEqual(selectedLeaf.find((entry) => entry.node.id === "node:2").pointIndices, [2]);
});

test("camera remains invertible in global UMAP coordinates and density picks the strongest cloud", async () => {
  const { visualizationCameraTransform, visualizationWorldToScreen, visualizationScreenToWorld, pickVisualizationCloud } = await loadVisualization();
  const camera = visualizationCameraTransform({ minX: 10, maxX: 20, minY: -2, maxY: 2 }, 400, 200);
  const screen = visualizationWorldToScreen(camera, [12, 1]);
  assert.ok(Math.abs(visualizationScreenToWorld(camera, screen)[0] - 12) < 1e-8);
  assert.equal(pickVisualizationCloud([[{ x: 10, y: 10, sigma: 2, amplitude: 1, color: [0, 0, 0] }], [{ x: 11, y: 10, sigma: 2, amplitude: 2, color: [0, 0, 0] }]], 11, 10), 1);
});

test("first real hierarchy split classifies zero-free ambiguous rows against the actual root", async () => {
  const { buildVisualizationTree, visualizationFrontier } = await loadVisualization();
  const tree = buildVisualizationTree({ leaves: [0, 1, 2], root: 9, children: { "9": [0, 1, 2] } }, [0, 1, 2]);
  const frontier = visualizationFrontier(tree, ["root"], [[.2, .2, .2], [.2, .2, .2], [.2, .2, .2]], [0, 1, 2]);
  assert.deepEqual(frontier.map((entry) => entry.node.id), ["node:0", "node:1", "node:2"]);
  assert.deepEqual(frontier.flatMap((entry) => entry.residualIndices), [0, 1, 2]);
  assert.ok(frontier.every((entry) => entry.pointIndices.length === 0));
});
