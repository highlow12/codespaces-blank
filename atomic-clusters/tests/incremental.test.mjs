import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadIncremental() {
  const source = await readFile(new URL("../src/incremental.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

function sleep(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }

function baseResult() {
  return {
    schemaVersion: 6,
    ids: ["a.md", "b.md"],
    leafLabels: [0, 1],
    probabilities: [0.9, 0.9],
    outlierProxy: [0.1, 0.1],
    pca: {
      selected: 2,
      model: { modelHash: "pca-1", inputDimension: 2, outputDimension: 2, normalization: "none", mean: [0, 0], components: [[1, 0], [0, 1]], explainedVariance: [1, 1], provider: "local", model: "m" }
    },
    hierarchy: {
      leaves: [0, 1],
      merges: [{ id: 2, left: 0, right: 1, distance: 0.5, mass: 2 }],
      root: 2,
      nodes: [
        { id: 0, children: [], descendantLeaves: [0], distance: 0, mass: 1 },
        { id: 1, children: [], descendantLeaves: [1], distance: 0, mass: 1 },
        { id: 2, children: [0, 1], descendantLeaves: [0, 1], distance: 0.5, mass: 2 }
      ],
      rootChildren: [0, 1],
      splitMethod: "distance-knee-2-5"
    },
    hierarchyPlacements: [
      { kind: "leaf", nodeId: 0, confidence: 0.9 },
      { kind: "leaf", nodeId: 1, confidence: 0.9 }
    ],
    leafOrder: [0, 1],
    leafOrdering: [0, 1],
    memberships: [[1, 0], [0, 1]],
    softMemberships: [[1, 0], [0, 1]],
    timings: {},
    embeddingProvider: "local",
    embeddingModel: "m",
    incremental: { mode: "full", generatedAt: "2026-08-31T00:00:00.000Z", changedPaths: [], provisionalPaths: [], fullRebuildRecommended: false, cumulativeChangedCount: 0, lastFullRebuildAt: "2026-08-31T00:00:00.000Z" }
  };
}

test("vault change queue coalesces an event storm and keeps events arriving during a drain", async () => {
  const { VaultChangeQueue } = await loadIncremental();
  const snapshots = [];
  let queue;
  queue = new VaultChangeQueue({ delayMs: 8, maxDelayMs: 40, onReady: () => { const snapshot = queue.drain(); if (snapshot) snapshots.push(snapshot); } });
  queue.enqueueModified("a.md"); queue.enqueueModified("a.md"); queue.enqueueModified("b.md");
  await sleep(25);
  queue.enqueueModified("during-refresh.md");
  await sleep(25);
  assert.equal(snapshots.length, 2);
  assert.deepEqual([...snapshots[0].modified].sort(), ["a.md", "b.md"]);
  assert.deepEqual([...snapshots[1].modified], ["during-refresh.md"]);
  queue.dispose();
});

test("incremental policy distinguishes path-only, soft, and full refreshes", async () => {
  const { decideIncrementalRefresh } = await loadIncremental();
  const result = baseResult();
  assert.deepEqual(decideIncrementalRefresh({ result, activeNoteCount: 100, changedNoteCount: 0, deletedNoteCount: 0, pathOnly: true, provider: "local", model: "m" }), { mode: "no-op", reason: "path_only_change" });
  assert.equal(decideIncrementalRefresh({ result, activeNoteCount: 100, changedNoteCount: 1, deletedNoteCount: 0, pathOnly: false, provider: "local", model: "m" }).mode, "soft");
  assert.equal(decideIncrementalRefresh({ result, activeNoteCount: 100, changedNoteCount: 21, deletedNoteCount: 0, pathOnly: false, provider: "local", model: "m" }).mode, "full");
  assert.equal(decideIncrementalRefresh({ result, activeNoteCount: 100, changedNoteCount: 1, deletedNoteCount: 0, pathOnly: false, provider: "local", model: "other" }).reason, "embedding_model_changed");
});

test("soft refresh reuses the hierarchy and marks a newly placed note provisional", async () => {
  const { buildSoftRefresh } = await loadIncremental();
  const result = baseResult();
  const notes = [
    { path: "a.md", title: "A", content: "a", mtime: 1, hash: "a" },
    { path: "b.md", title: "B", content: "b", mtime: 1, hash: "b" },
    { path: "new.md", title: "New", content: "new", mtime: 2, hash: "new" }
  ];
  const refreshed = buildSoftRefresh({
    result,
    notes,
    vectorsByPath: new Map([["a.md", [1, 0]], ["b.md", [0, 1]], ["new.md", [0.98, 0.1]]]),
    existingCoordinates: new Map([["a.md", [1, 0]], ["b.md", [0, 1]]]),
    changedPaths: new Set(["new.md"]),
    deletedPaths: new Set(),
    provider: "local",
    model: "m"
  });
  assert.deepEqual(refreshed.result.ids, ["a.md", "b.md", "new.md"]);
  assert.equal(refreshed.result.leafLabels[2], 0);
  assert.deepEqual(refreshed.result.provisionalPaths, ["new.md"]);
  assert.equal(refreshed.result.incremental.mode, "soft");
  assert.equal(refreshed.result.visualization, undefined);
  assert.deepEqual(refreshed.projectedPaths, ["new.md"]);
});

test("path-only rename mapping follows chained renames without changing row count", async () => {
  const { renameClusterResultPaths } = await loadIncremental();
  const result = { ...baseResult(), ids: ["a.md"], provisionalPaths: ["a.md"], incremental: { ...baseResult().incremental, provisionalPaths: ["a.md"] } };
  const renamed = renameClusterResultPaths(result, new Map([["a.md", "b.md"], ["b.md", "c.md"]]));
  assert.deepEqual(renamed.ids, ["c.md"]);
  assert.deepEqual(renamed.provisionalPaths, ["c.md"]);
});

test("the plugin build keeps the Python reference outside the shipped bundle", async () => {
  const buildScript = await readFile(new URL("../build.mjs", import.meta.url), "utf8");
  assert.doesNotMatch(buildScript, /pyodide/i);
  const main = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
  assert.doesNotMatch(main, /PyodideClusteringWorker|pyodide-worker-client/i);
});

test("vault rename handling preserves excluded-boundary and folder child changes", async () => {
  const main = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
  assert.match(main, /const enqueueRename = \(from: string, to: string\)/);
  assert.match(main, /pendingVaultChanges!\.enqueueDeleted\(from\)/);
  assert.match(main, /pendingVaultChanges!\.enqueueCreated\(to\)/);
  assert.match(main, /const markdownChildren =/);
  assert.match(main, /childPaths = markdownChildren\(folder\)/);
});
