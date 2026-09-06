import test from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";
import { readFile } from "node:fs/promises";

async function loadNoteDetail() {
  const result = await build({ entryPoints: [new URL("../src/note-detail.ts", import.meta.url).pathname], bundle: true, format: "esm", platform: "browser", write: false });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString("base64")}`);
}

function baseResult() {
  return {
    schemaVersion: 6,
    ids: ["alpha.md", "beta.md", "noise.md"],
    leafLabels: [0, 1, -1],
    probabilities: [0.91, 0.72, 0.05],
    outlierProxy: [0.09, 0.28, 0.95],
    leafOrder: [0, 1],
    leafOrdering: [0, 1],
    memberships: [[0.91, 0.09], [0.2, 0.8], [0, 0]],
    softMemberships: [[0.91, 0.09], [0.2, 0.8], [0, 0]],
    hierarchy: {
      leaves: [0, 1],
      merges: [{ id: 2, left: 0, right: 1, distance: 1, mass: 2 }],
      root: 2,
      nodes: [
        { id: 0, children: [], descendantLeaves: [0], distance: 0, mass: 1 },
        { id: 1, children: [], descendantLeaves: [1], distance: 0, mass: 1 },
        { id: 2, children: [0, 1], descendantLeaves: [0, 1], distance: 1, mass: 2 },
      ],
      rootChildren: [0, 1],
      splitMethod: "distance-knee-2-5",
    },
    hierarchyPlacements: [
      { kind: "leaf", nodeId: 0, confidence: 0.91 },
      { kind: "leaf", nodeId: 1, confidence: 0.72 },
      { kind: "residual", nodeId: 2, confidence: 0.05 },
    ],
    titles: { "0": "Alpha cluster", "1": "Beta cluster", "2": "Parent topic" },
    titleGeneration: {
      method: "keywords", algorithmVersion: "test", inputFingerprint: "test", nodeCount: 3, durationMs: 0, statuses: {},
      scores: { "0": [{ keyword: "alpha", score: 1 }], "1": [{ keyword: "beta", score: 1 }], "2": [{ keyword: "shared topic", score: 1 }] },
    },
    visualization: { coordinates: [[0, 0], [1, 0], [100, 100]], labels: [0, 1, -1], leafOrdering: [0, 1], memberships: [[0.91, 0.09], [0.2, 0.8], [0, 0]], configuration: {} },
    provisionalPaths: ["beta.md"],
    incremental: { mode: "soft", generatedAt: "2026-09-03T00:00:00.000Z", changedPaths: ["beta.md"], provisionalPaths: ["beta.md"], fullRebuildRecommended: false },
    pca: { selected: 2 },
    timings: {},
  };
}

const notes = [
  { path: "alpha.md", title: "Alpha", content: "alpha note", mtime: 1, hash: "a" },
  { path: "beta.md", title: "Beta", content: "beta note", mtime: 2, hash: "b" },
  { path: "noise.md", title: "Noise", content: "noise note", mtime: 3, hash: "n" },
];
const manualCorrections = { titleOverrides: [], notePreferences: [{ notePath: "alpha.md", preferredClusterKey: "1", createdAt: "2026-09-03T00:00:00.000Z" }], groups: [], feedback: [] };

test("note detail derives hierarchy, confidence, state, preference fallback, keywords, and related notes", async () => {
  const { buildNoteDetail } = await loadNoteDetail();
  const detail = buildNoteDetail(baseResult(), notes, "alpha.md", manualCorrections);
  assert.equal(detail.title, "Alpha");
  assert.equal(detail.path, "alpha.md");
  assert.deepEqual(detail.automaticLeaf, { id: 0, title: "Alpha cluster" });
  assert.deepEqual(detail.ancestors.map((item) => item.title), ["All notes", "Parent topic", "Alpha cluster"]);
  assert.equal(detail.probability, 0.91);
  assert.equal(detail.strongestMembership, 0.91);
  assert.equal(detail.noise, false);
  assert.equal(detail.residual, false);
  assert.equal(detail.provisional, false);
  assert.deepEqual(detail.manualPreferredCluster, { key: "1", title: "Beta cluster" });
  assert.deepEqual(detail.clusterKeywords, ["shared topic", "alpha"]);
  assert.equal(detail.relatedNotes[0].path, "beta.md");
  assert.ok(detail.relatedNotes[0].similarity > detail.relatedNotes[1].similarity);
});

test("note detail marks noise/residual and gracefully falls back when selection is gone", async () => {
  const { buildNoteDetail } = await loadNoteDetail();
  const detail = buildNoteDetail(baseResult(), notes, "noise.md");
  assert.equal(detail.automaticLeaf, null);
  assert.equal(detail.noise, true);
  assert.equal(detail.residual, true);
  assert.equal(detail.provisional, false);
  assert.equal(detail.manualPreferredCluster, null);
  assert.deepEqual(detail.ancestors.map((item) => item.title), ["All notes", "Parent topic"]);
  assert.equal(buildNoteDetail(baseResult(), notes, "missing.md"), null);
});

test("preferred-cluster candidates are relevant, use effective titles, and stay capped at five", async () => {
  const { getPreferredClusterCandidates } = await loadNoteDetail();
  const leafCount = 7;
  const result = {
    schemaVersion: 6,
    ids: Array.from({ length: leafCount }, (_, index) => `candidate-${index}.md`),
    leafLabels: Array.from({ length: leafCount }, (_, index) => index),
    probabilities: Array.from({ length: leafCount }, () => 1),
    outlierProxy: Array.from({ length: leafCount }, () => 0),
    leafOrder: Array.from({ length: leafCount }, (_, index) => index),
    hierarchy: { leaves: Array.from({ length: leafCount }, (_, index) => index), merges: [], root: 0 },
    memberships: [[0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]],
    titles: Object.fromEntries(Array.from({ length: leafCount }, (_, index) => [String(index), `Generated ${index}`])),
    pca: { selected: 2 },
  };
  const selected = "candidate-0.md";
  const first = getPreferredClusterCandidates(result, selected);
  assert.equal(first.length, 5);
  assert.deepEqual(first.map((candidate) => candidate.leafId), [1, 2, 3, 4, 5]);
  const corrected = getPreferredClusterCandidates(result, selected, {
    titleOverrides: [{ stableClusterKey: first[0].key, title: "Manual candidate", createdAt: "2026-09-04T00:00:00.000Z", updatedAt: "2026-09-04T00:00:00.000Z" }],
    notePreferences: [], groups: [], feedback: [],
  });
  assert.equal(corrected[0].title, "Manual candidate");
});

test("Explorer exposes a keyboard-accessible selected-note detail panel and Open note action", async () => {
  const [view, css] = await Promise.all([
    readFile(new URL("../src/view.ts", import.meta.url), "utf8"),
    readFile(new URL("../styles.css", import.meta.url), "utf8"),
  ]);
  assert.match(view, /selectedNotePath/);
  assert.match(view, /selectNote\(path: string\)/);
  assert.match(view, /renderNoteDetailPanel/);
  assert.match(view, /Selected note details/);
  assert.match(view, /Open note/);
  assert.match(view, /None recorded; using automatic/);
  assert.match(view, /aria-label.*Open note/);
  assert.match(view, /this\.selectNote\(path\)/);
  assert.match(view, /this\.selectNote\(result\.ids\[index\]\)/);
  assert.match(view, /onAccessiblePointClick/);
  assert.match(view, /buildNoteDetail\(this\.result, this\.searchNotes, this\.selectedNotePath\)/);
  assert.match(css, /\.atomic-clusters-note-detail \{/);
  assert.match(css, /\.atomic-clusters-note-detail-related/);
});
