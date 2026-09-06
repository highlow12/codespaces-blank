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

test("top membership summaries rank named clusters for regular and noise points", async () => {
  const { visualizationTopMemberships } = await loadVisualization();
  const result = { hierarchy: { leaves: [2, 0, 1, 3] }, leafOrdering: [2, 0, 1, 3], memberships: [[.1, .8, .3, .05]], titles: { "0": "Alpha", "1": "Beta" } };
  assert.deepEqual(visualizationTopMemberships(result, 0), [
    { leafId: 0, title: "Alpha", value: .8 },
    { leafId: 1, title: "Beta", value: .3 },
    { leafId: 2, title: "Cluster 2", value: .1 }
  ]);
  const noise = { ...result, memberships: [[.2, 0, .4, .1]] };
  assert.deepEqual(visualizationTopMemberships(noise, 0, 2), [
    { leafId: 1, title: "Beta", value: .4 },
    { leafId: 2, title: "Cluster 2", value: .2 }
  ]);
});

test("cluster label helpers choose titles, contrast, bounds, and non-overlapping placements", async () => {
  const { buildVisualizationTree, visualizationFrontier, visualizationClusterLabelText, visualizationLabelContrast, layoutVisualizationClusterLabels } = await loadVisualization();
  const tree = buildVisualizationTree({ leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 1, mass: 2 }], root: 2 }, [0, 1]);
  const frontier = visualizationFrontier(tree, ["root"], [[1, 0], [0, 1]], [0, 1]);
  assert.equal(visualizationClusterLabelText(frontier[0], { "0": "  Topics  " }), "Topics");
  assert.equal(visualizationClusterLabelText(frontier[0], {}), "Cluster 0");
  assert.equal(visualizationClusterLabelText({ node: tree, actualPoints: false }, { "": "Root" }), null);
  assert.deepEqual(visualizationLabelContrast("#111111"), { foreground: "#ffffff", background: "#000000" });
  assert.deepEqual(visualizationLabelContrast("#eeeeee"), { foreground: "#000000", background: "#ffffff" });
  const placements = layoutVisualizationClusterLabels(frontier, [[5, 5], [5, 5]], {}, new Map([["node:0", "#111111"], ["node:1", "#eeeeee"]]), 80, 40, { measureText: () => 70 });
  assert.equal(placements.length, 2);
  assert.ok(placements.every(({ x, y, width, height }) => x >= 8 && y >= 8 && x + width <= 72 && y + height <= 32));
  const separate = layoutVisualizationClusterLabels(frontier.concat(frontier), [[5, 5], [5, 5]], {}, new Map([["node:0", "#111111"], ["node:1", "#eeeeee"]]), 200, 100, { measureText: () => 40 });
  assert.ok(separate[1].y >= separate[0].y + separate[0].height || separate[0].y >= separate[1].y + separate[1].height);
});

test("camera fit and resize preserve label-box and hover-radius edge padding", async () => {
  const { buildVisualizationTree, visualizationFrontier, measureVisualizationClusterLabelBoxes, layoutVisualizationClusterLabels, visualizationFitCameraState, visualizationCameraFromState, visualizationWorldToScreen, resizeVisualizationCameraState, VISUALIZATION_POINT_PADDING } = await loadVisualization();
  const tree = buildVisualizationTree({ leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 1, mass: 2 }], root: 2 }, [0, 1]);
  const frontier = visualizationFrontier(tree, ["root"], [[1, 0], [0, 1]], [0, 1]);
  const titles = { "0": "A deliberately long cluster title" };
  const measureText = (text) => text.length * 8;
  const labelBoxes = measureVisualizationClusterLabelBoxes(frontier, titles, measureText, 24);
  const labelMargin = 12;
  const hoverPointRadius = 28;
  const options = { pointRadius: 4, hoverPointRadius, labelBoxes, labelMargin };
  const labelPadding = Math.max(...labelBoxes.map((box) => Math.max(box.width / 2, box.height / 2))) + labelMargin;
  const fit = visualizationFitCameraState([[0, 0], [100, 0]], 480, 320, VISUALIZATION_POINT_PADDING, options);
  assert.ok(labelPadding > VISUALIZATION_POINT_PADDING);
  assert.ok(labelPadding > hoverPointRadius);
  assert.equal(fit.padding, labelPadding);
  assert.equal(fit.renderedPointRadius, hoverPointRadius);

  const fitCamera = visualizationCameraFromState(fit);
  const fitPoints = [[0, 0], [100, 0]].map((point) => visualizationWorldToScreen(fitCamera, point));
  const fitLabels = layoutVisualizationClusterLabels(frontier, fitPoints, titles, new Map([["node:0", "#111111"], ["node:1", "#eeeeee"]]), fit.width, fit.height, { margin: fit.padding, labelHeight: 24, measureText });
  assert.ok(fitLabels.every((label) => label.x >= fit.padding && label.y >= fit.padding && label.x + label.width <= fit.width - fit.padding && label.y + label.height <= fit.height - fit.padding));

  const narrow = resizeVisualizationCameraState(fit, [[0, 0], [100, 0]], 260, 180, options);
  assert.equal(narrow.width, 260);
  assert.equal(narrow.height, 180);
  assert.equal(narrow.padding, 90);
  assert.equal(narrow.renderedPointRadius, hoverPointRadius);
  const narrowCamera = visualizationCameraFromState(narrow);
  const narrowPoints = [[0, 0], [100, 0]].map((point) => visualizationWorldToScreen(narrowCamera, point));
  const narrowLabels = layoutVisualizationClusterLabels(frontier, narrowPoints, titles, new Map([["node:0", "#111111"], ["node:1", "#eeeeee"]]), narrow.width, narrow.height, { margin: narrow.padding, labelHeight: 24, measureText });
  assert.ok(narrowLabels.every((label) => label.x >= narrow.padding && label.y >= narrow.padding && label.x + label.width <= narrow.width - narrow.padding && label.y + label.height <= narrow.height - narrow.padding));

  const expanded = resizeVisualizationCameraState(narrow, [[0, 0], [100, 0]], 600, 400, options);
  assert.equal(expanded.padding, labelPadding);
  assert.equal(expanded.renderedPointRadius, hoverPointRadius);
});

test("hierarchy palette spaces three top-level clusters and varies descendants in HSL", async () => {
  const { buildVisualizationTree, visualizationColorScheme } = await loadVisualization();
  const hierarchy = { leaves: [0, 1, 2, 3, 6], merges: [{ id: 4, left: 0, right: 1, distance: 1, mass: 2 }, { id: 5, left: 2, right: 3, distance: 1, mass: 2 }], root: 9, children: { "9": [4, 5, 6] } };
  const tree = buildVisualizationTree(hierarchy, [0, 1, 2, 3]);
  const first = visualizationColorScheme(tree); const second = visualizationColorScheme(tree);
  assert.deepEqual([...first.nodeColors], [...second.nodeColors]);
  assert.deepEqual([...first.leafColors], [...second.leafColors]);
  assert.deepEqual([...first.nodeHsl], [...second.nodeHsl]);
  const topHues = ["node:4", "node:5", "node:6"].map((id) => first.nodeHsl.get(id).hue);
  const clockwise = (topHues[1] - topHues[0] + 360) % 360;
  const nextClockwise = (topHues[2] - topHues[1] + 360) % 360;
  assert.ok(Math.abs(clockwise - 120) < 1e-8);
  assert.ok(Math.abs(nextClockwise - 120) < 1e-8);
  const parent = first.nodeHsl.get("node:4"); const child = first.nodeHsl.get("node:0");
  const hueDelta = Math.abs((child.hue - parent.hue + 540) % 360 - 180);
  assert.ok(hueDelta > 0 && hueDelta <= 16);
  assert.ok(child.lightness > parent.lightness && child.lightness - parent.lightness <= 8);
  assert.equal(first.leafColors.get(0), first.nodeColors.get("node:0"));
  assert.equal(first.leafColors.get(3), first.nodeColors.get("node:3"));
  assert.notEqual(first.nodeColors.get("node:0"), first.nodeColors.get("node:1"));
  assert.match(first.nodeColors.get("node:4"), /^#[0-9a-f]{6}$/);
});

test("nearest-point hit testing returns only the closest point within radius", async () => {
  const { findNearestVisualizationPoint } = await loadVisualization();
  assert.equal(findNearestVisualizationPoint([[10, 10], [12, 10], [50, 50]], 11.5, 10, 3), 1);
  assert.equal(findNearestVisualizationPoint([[10, 10]], 20, 20, 3), null);
});

test("screen-space spatial index matches deterministic nearest-point semantics without scanning all points", async () => {
  const { buildVisualizationPointSpatialIndex, findNearestVisualizationPoint } = await loadVisualization();
  const points = [[0, 0], [12, 0], [24, 0], [12, 12], [100, 100], [Number.NaN, 4]];
  const index = buildVisualizationPointSpatialIndex(points, 12);
  for (const [x, y, radius] of [[0, 0, 0], [11.5, 0, 3], [18, 5, 10], [99, 99, 3], [50, 50, 5]]) {
    assert.equal(index.queryNearest(x, y, radius), findNearestVisualizationPoint(points, x, y, radius));
  }
  assert.deepEqual(index.queryRect(-1, -1, 25, 13).sort((a, b) => a - b), [0, 1, 2, 3]);
  assert.equal(index.queryNearest(Number.NaN, 0, 10), null);
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

test("frontier shows only the focused node's children and reveals notes at a selected leaf", async () => {
  const { buildVisualizationTree, visualizationFrontier } = await loadVisualization();
  const hierarchy = { leaves: [0, 1, 2], merges: [{ id: 3, left: 0, right: 1, distance: 1, mass: 2 }, { id: 4, left: 3, right: 2, distance: 2, mass: 3 }], root: 4 };
  const tree = buildVisualizationTree(hierarchy, [0, 1, 2, -1, 0, 2]);
  const rows = [[.5, .5, 0], [.5, .5, 0], [0, 0, 1], [0, 0, 0], [.5, .5, 0], [0, 0, 0]];
  const expanded = visualizationFrontier(tree, ["root"], rows, [0, 1, 2]);
  assert.deepEqual(expanded.map((entry) => entry.node.id), ["node:3", "node:2"]);
  assert.deepEqual(expanded.flatMap((entry) => entry.residualIndices), [3, 5]);
  assert.equal(new Set(expanded.flatMap((entry) => entry.residualIndices)).size, 2);
  assert.deepEqual(expanded[1].pointIndices, [2]);
  assert.equal(expanded[1].actualPoints, false);
  const deeper = visualizationFrontier(tree, ["node:3"], rows, [0, 1, 2]);
  assert.deepEqual(deeper.map((entry) => entry.node.id), ["node:0", "node:1"]);
  assert.deepEqual(deeper.flatMap((entry) => entry.residualIndices), [3, 5]);
  assert.ok(deeper.every((entry) => !entry.actualPoints));
  const selectedLeaf = visualizationFrontier(tree, ["root", "node:2"], rows, [0, 1, 2]);
  assert.equal(selectedLeaf.find((entry) => entry.node.id === "node:2").actualPoints, true);
  assert.deepEqual(selectedLeaf.find((entry) => entry.node.id === "node:2").pointIndices, [2]);
  assert.deepEqual(selectedLeaf.find((entry) => entry.node.id === "node:2").residualIndices, [3, 5]);
});

test("top-level clouds use one cluster color while notes stay leaf-only", async () => {
  const { buildVisualizationTree, visualizationColorScheme, visualizationCloudColor, visualizationFrontier, visualizationParent } = await loadVisualization();
  const hierarchy = { leaves: [0, 1, 2], merges: [{ id: 3, left: 0, right: 1, distance: 1, mass: 2 }, { id: 4, left: 3, right: 2, distance: 2, mass: 3 }], root: 4 };
  const tree = buildVisualizationTree(hierarchy, [0, 1, 2]);
  const palette = visualizationColorScheme(tree);
  const top = visualizationFrontier(tree, [], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2]);
  assert.deepEqual(top.map((entry) => entry.node.id), ["node:3", "node:2"]);
  assert.ok(top.every((entry) => !entry.actualPoints));
  assert.equal(visualizationCloudColor(top[0].node, palette), palette.nodeColors.get("node:3"));
  assert.equal(visualizationCloudColor(top[1].node, palette), palette.nodeColors.get("node:2"));
  const middle = visualizationFrontier(tree, ["node:3"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2]);
  assert.deepEqual(middle.map((entry) => entry.node.id), ["node:0", "node:1"]);
  assert.ok(middle.every((entry) => !entry.actualPoints));
  assert.deepEqual(middle.flatMap((entry) => entry.residualIndices), []);
  const leaf = visualizationFrontier(tree, ["node:0"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2]);
  assert.equal(leaf.length, 1);
  assert.equal(leaf[0].actualPoints, true);
  assert.deepEqual(leaf[0].pointIndices, [0]);
  assert.equal(visualizationParent(tree, "node:0").id, "node:3");
  assert.deepEqual(visualizationFrontier(tree, [visualizationParent(tree, "node:0").id], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 1, 2]).map((entry) => entry.node.id), ["node:0", "node:1"]);
});

test("camera remains invertible in global UMAP coordinates and density picks the strongest cloud", async () => {
  const { visualizationCameraTransform, visualizationWorldToScreen, visualizationScreenToWorld, pickVisualizationCloud } = await loadVisualization();
  const camera = visualizationCameraTransform({ minX: 10, maxX: 20, minY: -2, maxY: 2 }, 400, 200);
  const screen = visualizationWorldToScreen(camera, [12, 1]);
  assert.ok(Math.abs(visualizationScreenToWorld(camera, screen)[0] - 12) < 1e-8);
  assert.equal(pickVisualizationCloud([[{ x: 10, y: 10, sigma: 2, amplitude: 1, color: [0, 0, 0] }], [{ x: 11, y: 10, sigma: 2, amplitude: 2, color: [0, 0, 0] }]], 11, 10), 1);
});

test("camera transition easing is monotonic and camera-layer transform reaches exact endpoints", async () => {
  const { visualizationEaseInOut, visualizationCameraLayerTransform } = await loadVisualization();
  assert.equal(visualizationEaseInOut(-1), 0);
  assert.equal(visualizationEaseInOut(0), 0);
  assert.equal(visualizationEaseInOut(1), 1);
  assert.equal(visualizationEaseInOut(2), 1);
  let previous = 0;
  for (let step = 0; step <= 100; step++) {
    const value = visualizationEaseInOut(step / 100);
    assert.ok(value >= previous, `easing must be monotonic at ${step / 100}`);
    previous = value;
  }
  const from = { scale: 2, offsetX: 30, offsetY: -14, width: 400, height: 280, worldRegion: { minX: 0, maxX: 1, minY: 0, maxY: 1 } };
  const to = { scale: 8, offsetX: -12, offsetY: 25, width: 400, height: 280, worldRegion: { minX: 2, maxX: 3, minY: 2, maxY: 3 } };
  const start = visualizationCameraLayerTransform(from, to, 0);
  const middle = visualizationCameraLayerTransform(from, to, 0.5);
  const end = visualizationCameraLayerTransform(from, to, 1);
  assert.deepEqual(start, { scale: from.scale / to.scale, translateX: from.offsetX - (from.scale / to.scale) * to.offsetX, translateY: from.offsetY - (from.scale / to.scale) * to.offsetY });
  assert.deepEqual(end, { scale: 1, translateX: 0, translateY: 0 });
  assert.ok(middle.scale > start.scale && middle.scale < end.scale);
  assert.ok(middle.translateX > end.translateX && middle.translateX < start.translateX);
  assert.ok(middle.translateY > Math.min(start.translateY, end.translateY) && middle.translateY < Math.max(start.translateY, end.translateY));
});

test("outgoing camera transform starts at identity and ends at source-to-target mapping", async () => {
  const { visualizationOutgoingLayerTransform } = await loadVisualization();
  const from = { scale: 2, offsetX: 30, offsetY: -14, width: 400, height: 280, worldRegion: { minX: 0, maxX: 1, minY: 0, maxY: 1 } };
  const to = { scale: 8, offsetX: -12, offsetY: 25, width: 400, height: 280, worldRegion: { minX: 2, maxX: 3, minY: 2, maxY: 3 } };
  assert.deepEqual(visualizationOutgoingLayerTransform(from, to, 0), { scale: 1, translateX: 0, translateY: 0 });
  const end = visualizationOutgoingLayerTransform(from, to, 1);
  assert.deepEqual(end, { scale: to.scale / from.scale, translateX: to.offsetX - (to.scale / from.scale) * from.offsetX, translateY: to.offsetY - (to.scale / from.scale) * from.offsetY });
  const middle = visualizationOutgoingLayerTransform(from, to, .5);
  assert.notDeepEqual(middle, { scale: 1, translateX: 0, translateY: 0 });
  assert.notDeepEqual(middle, end);
});

test("first real hierarchy split classifies zero-free ambiguous rows against the actual root", async () => {
  const { buildVisualizationTree, visualizationFrontier } = await loadVisualization();
  const tree = buildVisualizationTree({ leaves: [0, 1, 2], root: 9, children: { "9": [0, 1, 2] } }, [0, 1, 2]);
  const frontier = visualizationFrontier(tree, ["root"], [[.2, .2, .2], [.2, .2, .2], [.2, .2, .2]], [0, 1, 2]);
  assert.deepEqual(frontier.map((entry) => entry.node.id), ["node:0", "node:1", "node:2"]);
  assert.deepEqual(frontier.flatMap((entry) => entry.residualIndices), [0, 1, 2]);
  assert.ok(frontier.every((entry) => entry.pointIndices.length === 0 && !entry.actualPoints));
});

test("v6 placements render every root and local residual exactly once at each visible stage", async () => {
  const { buildVisualizationTree, visualizationFrontier } = await loadVisualization();
  const hierarchy = {
    leaves: [0, 1, 2, 3],
    merges: [
      { id: 4, left: 0, right: 1, distance: 1, mass: 2 },
      { id: 5, left: 2, right: 3, distance: 1, mass: 2 },
      { id: 6, left: 4, right: 5, distance: 2, mass: 4 }
    ],
    root: 6,
    nodes: [
      { id: 0, children: [], descendantLeaves: [0] },
      { id: 1, children: [], descendantLeaves: [1] },
      { id: 2, children: [], descendantLeaves: [2] },
      { id: 3, children: [], descendantLeaves: [3] },
      { id: 4, children: [0, 1], descendantLeaves: [0, 1] },
      { id: 5, children: [2, 3], descendantLeaves: [2, 3] },
      { id: 6, children: [4, 5], descendantLeaves: [0, 1, 2, 3] }
    ],
    rootChildren: [4, 5]
  };
  const labels = [0, 1, 2, 3, -1, 1, 2];
  const placements = [
    { kind: "leaf", nodeId: 0, confidence: 0.8 },
    { kind: "residual", nodeId: 4, confidence: 0.5 },
    { kind: "residual", nodeId: 5, confidence: 0.5 },
    { kind: "leaf", nodeId: 3, confidence: 0.9 },
    { kind: "residual", nodeId: null, confidence: 0 },
    { kind: "leaf", nodeId: 1, confidence: 0.7 },
    { kind: "residual", nodeId: 5, confidence: 0 }
  ];
  const tree = buildVisualizationTree(hierarchy, labels);
  const rootStage = visualizationFrontier(tree, ["root"], undefined, undefined, placements);
  const rootClouds = rootStage.flatMap((entry) => entry.pointIndices);
  const rootResiduals = rootStage.flatMap((entry) => entry.residualIndices);
  assert.deepEqual(rootClouds, [0, 5, 3]);
  assert.deepEqual(rootResiduals, [1, 2, 4, 6]);
  assert.deepEqual(rootStage.flatMap((entry) => entry.directResidualIndices), []);
  assert.deepEqual(rootStage.flatMap((entry) => entry.descendantResidualIndices), [1, 2, 4, 6]);
  assert.deepEqual([...rootClouds, ...rootResiduals].sort((a, b) => a - b), labels.map((_, index) => index));
  assert.equal(new Set([...rootClouds, ...rootResiduals]).size, labels.length);

  const focused = visualizationFrontier(tree, ["node:5"], undefined, undefined, placements);
  assert.deepEqual(focused.flatMap((entry) => entry.pointIndices), [3]);
  assert.deepEqual(focused.flatMap((entry) => entry.residualIndices), [2, 6]);
  assert.deepEqual(focused.flatMap((entry) => entry.directResidualIndices), [2, 6]);
  assert.deepEqual(focused.flatMap((entry) => entry.descendantResidualIndices), []);
});

test("v6 all-noise visualization keeps every note as an interactive residual point", async () => {
  const { buildVisualizationTree, visualizationFrontier, validateVisualizationData } = await loadVisualization();
  const result = {
    schemaVersion: 6,
    ids: ["a.md", "b.md", "c.md"],
    leafLabels: [-1, -1, -1],
    leafOrdering: [],
    memberships: [[], [], []],
    hierarchyPlacements: [
      { kind: "residual", nodeId: null, confidence: 0 },
      { kind: "residual", nodeId: null, confidence: 0 },
      { kind: "residual", nodeId: null, confidence: 0 }
    ],
    hierarchy: { leaves: [], merges: [], root: null, nodes: [], rootChildren: [] },
    visualization: {
      coordinates: [[-1, 0], [0, 0], [1, 0]],
      labels: [-1, -1, -1],
      leafOrdering: [],
      memberships: [[], [], []],
      configuration: { runtime: "test", seed: 42, nComponents: 2, nNeighbors: 2, minDist: 1, spread: 1 }
    }
  };
  assert.equal(validateVisualizationData(result), true);
  const tree = buildVisualizationTree(result.hierarchy, result.leafLabels);
  const stage = visualizationFrontier(tree, ["root"], result.memberships, result.leafOrdering, result.hierarchyPlacements);
  assert.equal(stage.length, 1);
  assert.deepEqual(stage[0].pointIndices, []);
  assert.deepEqual(stage[0].residualIndices, [0, 1, 2]);
  assert.deepEqual(stage[0].directResidualIndices, [0, 1, 2]);
});
