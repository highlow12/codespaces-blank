import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadClustering() {
  let source = await readFile(new URL("../src/clustering.ts", import.meta.url), "utf8");
  source = source.replace(
    'import { UMAP } from "umap-js";',
    "const UMAP = class { async fitAsync(points) { return points; } };"
  );
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

function unitRows(count) {
  return Array.from({ length: count }, (_, row) =>
    Array.from({ length: count }, (_, column) => row === column ? 1 : 0)
  );
}

function indexTree(tree) {
  assert.ok(Array.isArray(tree.nodes), "v6 hierarchy must persist nodes");
  return new Map(tree.nodes.map((node) => [node.id, node]));
}

function assertTreeShape(tree) {
  assert.equal(tree.splitMethod, "distance-knee-2-5");
  assert.ok(Array.isArray(tree.rootChildren));
  const byId = indexTree(tree);
  const seen = new Set();
  const visit = (id) => {
    assert.equal(seen.has(id), false, `node ${id} must occur in one subtree only`);
    seen.add(id);
    const node = byId.get(id);
    assert.ok(node, `missing node ${id}`);
    assert.ok(Number.isFinite(node.distance));
    assert.ok(Number.isFinite(node.mass) && node.mass >= 0);
    assert.ok(Array.isArray(node.children));
    assert.ok(Array.isArray(node.descendantLeaves));
    if (!node.children.length) {
      assert.deepEqual(node.descendantLeaves, [node.id]);
      return new Set(node.descendantLeaves);
    }
    assert.ok(node.children.length >= 2 && node.children.length <= 5, `node ${id} has ${node.children.length} children`);
    const union = new Set();
    for (const child of node.children) {
      const childLeaves = visit(child);
      for (const leaf of childLeaves) {
        assert.equal(union.has(leaf), false, `node ${id} children overlap at leaf ${leaf}`);
        union.add(leaf);
      }
    }
    assert.deepEqual([...union].sort((a, b) => a - b), [...node.descendantLeaves].sort((a, b) => a - b));
    return union;
  };
  const rootLeaves = new Set();
  const rootNode = tree.root === null ? undefined : byId.get(tree.root);
  if (tree.leaves.length > 1) assert.deepEqual(tree.rootChildren, rootNode.children);
  const rootIds = tree.root === null ? tree.rootChildren : [tree.root];
  for (const child of rootIds) {
    const childLeaves = visit(child);
    for (const leaf of childLeaves) {
      assert.equal(rootLeaves.has(leaf), false, `root children overlap at leaf ${leaf}`);
      rootLeaves.add(leaf);
    }
  }
  assert.deepEqual([...rootLeaves].sort((a, b) => a - b), [...tree.leaves].sort((a, b) => a - b));
  assert.equal(seen.size, tree.nodes.length);
}

function assertMonotoneMerges(tree) {
  const heights = new Map();
  for (const merge of tree.merges) {
    const leftHeight = heights.get(merge.left) || 0;
    const rightHeight = heights.get(merge.right) || 0;
    assert.ok(merge.distance >= leftHeight, `merge ${merge.id} is below its left child`);
    assert.ok(merge.distance >= rightHeight, `merge ${merge.id} is below its right child`);
    assert.ok(Number.isFinite(merge.distance));
    heights.set(merge.id, merge.distance);
  }
}

test("v6 small-result schema stores an explicit root residual placement", async () => {
  const { clusterEmbeddings } = await loadClustering();
  const result = await clusterEmbeddings(["a.md", "b.md"], [[1, 0], [0, 1]]);
  assert.equal(result.schemaVersion, 6);
  assert.deepEqual(result.hierarchy, {
    leaves: [],
    merges: [],
    root: null,
    nodes: [],
    rootChildren: [],
    splitMethod: "distance-knee-2-5"
  });
  assert.deepEqual(result.hierarchyPlacements, [
    { kind: "residual", nodeId: null, confidence: 0 },
    { kind: "residual", nodeId: null, confidence: 0 }
  ]);
  assert.deepEqual(result.memberships, [[], []]);
  assert.deepEqual(result.leafOrdering, []);
  assert.deepEqual(result.visualization.coordinates, [[-0.5, 0], [0.5, 0]]);
  assert.deepEqual(result.visualization.labels, [-1, -1]);
});

test("generated v6 hierarchy is deterministic, monotone, disjoint, and exhaustive", async () => {
  const { buildHierarchy } = await loadClustering();
  const rows = unitRows(7).flatMap((row) => [row, row]);
  const labels = rows.map((_, index) => Math.floor(index / 2));
  const probabilities = rows.map(() => 1);
  const first = buildHierarchy(rows, labels, probabilities);
  const second = buildHierarchy(rows, labels, probabilities);
  assert.deepEqual(second, first);
  assert.deepEqual(first.leaves, [0, 1, 2, 3, 4, 5, 6]);
  assertTreeShape(first);
  assertMonotoneMerges(first);
});

test("distance knees deterministically select each supported child count from 2 through 5", async () => {
  const { chooseNaryFrontier } = await loadClustering();
  const chain = (leafCount) => {
    const merges = new Map(); let current = leafCount;
    merges.set(current, { id: current, left: 0, right: 1, distance: 1, mass: 2 });
    for (let leaf = 2; leaf < leafCount; leaf++) {
      const id = leafCount + leaf - 1;
      const distance = 11 - leafCount + leaf;
      merges.set(id, { id, left: leaf, right: current, distance, mass: leaf + 1 });
      current = id;
    }
    return { root: current, merges, masses: new Map(Array.from({ length: leafCount }, (_, leaf) => [leaf, 1])) };
  };
  for (let expected = 2; expected <= 5; expected++) {
    const { root, merges, masses } = chain(expected + 1);
    const first = chooseNaryFrontier(root, merges, masses, 1);
    const second = chooseNaryFrontier(root, merges, new Map(Array.from({ length: expected + 1 }, (_, leaf) => [leaf, 1])), 1);
    assert.equal(first.length, expected);
    assert.deepEqual(second, first);
  }
});

test("binary tie breaks are stable by active node order and heights remain monotone", async () => {
  const { buildHierarchy } = await loadClustering();
  const tree = buildHierarchy([[1, 0], [1, 0], [1, 0]], [0, 1, 2], [1, 1, 1]);
  assert.deepEqual(tree.merges.map(({ id, left, right }) => ({ id, left, right })), [
    { id: 3, left: 0, right: 1 },
    { id: 4, left: 2, right: 3 }
  ]);
  assertMonotoneMerges(tree);
  assertTreeShape(tree);
});

test("zero and one non-noise leaves do not create empty hierarchy children", async () => {
  const { buildHierarchy } = await loadClustering();
  const empty = buildHierarchy([[1, 0], [0, 1]], [-1, -1], [0, 0]);
  assert.deepEqual(empty.leaves, []);
  assert.deepEqual(empty.merges, []);
  assert.equal(empty.root, null);
  assert.deepEqual(empty.rootChildren, []);
  assert.deepEqual(empty.nodes, []);

  const one = buildHierarchy([[1, 0], [0, 1], [1, 0]], [4, -1, 4], [1, 0, 1]);
  assert.deepEqual(one.leaves, [4]);
  assert.equal(one.root, 4);
  assert.deepEqual(one.nodes, [{ id: 4, children: [], descendantLeaves: [4], distance: 0, mass: 2 }]);
  assert.ok(Array.isArray(one.rootChildren));
  assert.equal(one.nodes.some((node) => node.children.length > 0), false);
});

test("minimum cluster mass prevents promoting undersized binary splits", async () => {
  const { buildHierarchy } = await loadClustering();
  const rows = [[1, 0], [0.99, 0.1], [-1, 0], [-0.99, -0.1]];
  const tree = buildHierarchy(rows, [0, 1, 2, 3], [1, 1, 1, 1], undefined, 2);
  assertTreeShape(tree);
  const byId = indexTree(tree);
  const root = byId.get(tree.root);
  assert.ok(root);
  assert.equal(root.children.length, 2);
  assert.ok(root.children.every((child) => byId.get(child).mass >= 2));
});

function placementHierarchy() {
  return {
    leaves: [0, 1, 2, 3],
    merges: [
      { id: 4, left: 0, right: 1, distance: 1, mass: 2 },
      { id: 5, left: 2, right: 3, distance: 1, mass: 2 },
      { id: 6, left: 4, right: 5, distance: 2, mass: 4 }
    ],
    root: 6,
    nodes: [
      { id: 0, children: [], descendantLeaves: [0], distance: 0, mass: 1 },
      { id: 1, children: [], descendantLeaves: [1], distance: 0, mass: 1 },
      { id: 2, children: [], descendantLeaves: [2], distance: 0, mass: 1 },
      { id: 3, children: [], descendantLeaves: [3], distance: 0, mass: 1 },
      { id: 4, children: [0, 1], descendantLeaves: [0, 1], distance: 1, mass: 2 },
      { id: 5, children: [2, 3], descendantLeaves: [2, 3], distance: 1, mass: 2 },
      { id: 6, children: [4, 5], descendantLeaves: [0, 1, 2, 3], distance: 2, mass: 4 }
    ],
    rootChildren: [4, 5],
    splitMethod: "distance-knee-2-5"
  };
}

test("placements descend only on strict majority and retain one aligned terminal per row", async () => {
  const { applyHierarchyPlacementMasses, computeHierarchyPlacements } = await loadClustering();
  const labels = [0, 1, 2, -1, 3];
  const probabilities = [0.8, 0.5, 0.4, 0, 0.6];
  const memberships = [
    [0.8, 0.2, 0, 0],
    [0.5, 0.5, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0.4, 0.6]
  ];
  const placements = computeHierarchyPlacements(labels, probabilities, memberships, placementHierarchy());
  assert.equal(placements.length, labels.length);
  assert.deepEqual(placements, [
    { kind: "leaf", nodeId: 0, confidence: 0.8 },
    { kind: "residual", nodeId: 4, confidence: 0.5 },
    { kind: "residual", nodeId: 6, confidence: 0 },
    { kind: "residual", nodeId: null, confidence: 0 },
    { kind: "leaf", nodeId: 3, confidence: 0.6 }
  ]);
  for (const placement of placements) {
    assert.ok(placement.kind === "leaf" || placement.kind === "residual");
    assert.ok(placement.nodeId === null || Number.isSafeInteger(placement.nodeId));
    assert.ok(Number.isFinite(placement.confidence) && placement.confidence >= 0 && placement.confidence <= 1);
  }
  const hierarchy = applyHierarchyPlacementMasses(placementHierarchy(), placements);
  assert.deepEqual(
    hierarchy.nodes.map(({ id, mass }) => [id, mass]),
    [[0, 1], [1, 0], [2, 0], [3, 1], [4, 2], [5, 1], [6, 4]]
  );
});

test("a strict-majority path reaches its terminal leaf with bounded confidence", async () => {
  const { computeHierarchyPlacements } = await loadClustering();
  const hierarchy = placementHierarchy();
  const placements = computeHierarchyPlacements(
    [0],
    [0.51],
    [[0.51, 0.49, 0, 0]],
    hierarchy
  );
  assert.deepEqual(placements, [{ kind: "leaf", nodeId: 0, confidence: 0.51 }]);
});
