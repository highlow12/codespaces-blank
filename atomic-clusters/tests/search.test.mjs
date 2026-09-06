import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadSearch() {
  let source = await readFile(new URL("../src/search.ts", import.meta.url), "utf8");
  source = source.replace(/import type \{[\s\S]*?\} from "\.\/types";\n/, "");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

test("SearchIndex parses plain terms, phrases, and field qualifiers", async () => {
  const { parseSearchQuery, SearchIndex } = await loadSearch();
  const parsed = parseSearchQuery('"design system" tag:#ui path:Projects/ cluster:frontend');
  assert.deepEqual(parsed.phrases, ["design system"]);
  assert.deepEqual(parsed.tags, ["ui"]);
  assert.deepEqual(parsed.paths, ["projects/"]);
  assert.deepEqual(parsed.clusters, ["frontend"]);
  const index = new SearchIndex([
    { path: "Projects/ui.md", title: "UI notes", body: "Design system guidance", tags: ["ui"], aliases: ["frontend"], clusterIds: ["0", "2", "root"], clusterTerms: ["Frontend", "design system"], leafLabel: 0, mtime: Date.now() },
    { path: "Archive/old.md", title: "Old", body: "Design system", tags: ["archive"], aliases: [], clusterIds: ["1", "2", "root"], clusterTerms: ["Backend"], leafLabel: 1, mtime: 0 }
  ], [{ id: "0", title: "Frontend", keywords: ["design system"] }, { id: "1", title: "Backend" }, { id: "2", title: "All work" }, { id: "root", title: "All notes" }]);
  const match = index.search(parsed);
  assert.deepEqual(match.notePaths, ["Projects/ui.md"]);
  assert.ok(match.clusterIds.includes("0"));
  assert.ok(match.clusterIds.includes("2"));
});

test("Search filters compose and preserve optional metadata fallbacks", async () => {
  const { SearchIndex } = await loadSearch();
  const index = new SearchIndex([
    { path: "a.md", title: "A", body: "alpha", tags: ["one"], aliases: ["first"], clusterIds: ["0", "root"], clusterTerms: ["Alpha"], leafLabel: 0, provisional: true, manuallyAdjusted: false, mtime: 2_000 },
    { path: "b.md", title: "B", body: "alpha", tags: [], aliases: [], clusterIds: ["1", "root"], clusterTerms: ["Beta"], leafLabel: -1, provisional: false, manuallyAdjusted: false, mtime: 0 },
    { path: "c.md", title: "C", body: "alpha", tags: [], aliases: [], clusterIds: ["0", "root"], clusterTerms: ["Alpha"], leafLabel: 0, provisional: false, manuallyAdjusted: true, mtime: 1_900 }
  ]);
  assert.deepEqual(index.search("alpha", { currentClusterId: "0", provisional: true }).notePaths, ["a.md"]);
  assert.deepEqual(index.search("", { noise: true }).notePaths, ["b.md"]);
  assert.deepEqual(index.search("", { manuallyAdjusted: true }).notePaths, ["c.md"]);
  assert.deepEqual(index.search("", { recentlyChanged: true, now: 2_000, recentlyChangedWindowMs: 500 }).notePaths, ["a.md", "c.md"]);
  assert.deepEqual(index.search("tag:one").notePaths, ["a.md"]);
  assert.deepEqual(index.search("path:b.md").notePaths, ["b.md"]);
});

test("10,000 metadata documents remain indexed deterministically", async () => {
  const { SearchIndex } = await loadSearch();
  const documents = Array.from({ length: 10_000 }, (_, index) => ({
    path: `Projects/${String(index).padStart(5, "0")}.md`, title: `Note ${index}`, body: index === 9876 ? "needle phrase for benchmark" : `body ${index % 50}`,
    tags: [`tag-${index % 20}`], aliases: [], clusterIds: [String(index % 20), "root"], clusterTerms: [`Cluster ${index % 20}`], leafLabel: index % 20, mtime: index
  }));
  const index = new SearchIndex(documents);
  const started = performance.now();
  const result = index.search('"needle phrase" path:Projects/ tag:tag-16');
  const duration = performance.now() - started;
  assert.deepEqual(result.notePaths, ["Projects/09876.md"]);
  assert.equal(result.matchedNotes.length, 1);
  // This is deliberately generous for loaded CI hosts; the assertion catches
  // accidental quadratic query paths without making a hardware claim.
  assert.ok(duration < 1_000, `10k metadata query took ${duration.toFixed(1)}ms`);
});

test("buildSearchDocuments extracts optional frontmatter and ancestor context", async () => {
  const { buildSearchDocuments } = await loadSearch();
  const result = {
    ids: ["note.md"], leafLabels: [0], probabilities: [1], outlierProxy: [0], schemaVersion: 6, pca: { selected: 2 }, timings: {},
    hierarchy: { leaves: [0], merges: [{ id: 1, left: 0, right: 0, distance: 1, mass: 1 }], root: 1, nodes: [{ id: 0, children: [], descendantLeaves: [0], distance: 0, mass: 1 }, { id: 1, children: [0], descendantLeaves: [0], distance: 1, mass: 1 }], rootChildren: [0], splitMethod: "distance-knee-2-5" },
    hierarchyPlacements: [{ kind: "leaf", nodeId: 0, confidence: 1 }], titles: { "0": "Frontend", "1": "Product" }, titleGeneration: { method: "keywords", algorithmVersion: "test", inputFingerprint: "test", statuses: {}, nodeCount: 2, durationMs: 0, scores: { "0": [{ keyword: "design system", score: 1 }] } }
  };
  const built = buildSearchDocuments([{ path: "note.md", title: "Note", mtime: 1, hash: "hash", content: "---\ntags: [ui, design]\naliases: [front page]\n---\nbody" }], result);
  assert.deepEqual(built.documents[0].tags, ["ui", "design"]);
  assert.deepEqual(built.documents[0].aliases, ["front page"]);
  assert.deepEqual(built.documents[0].clusterIds.sort(), ["0", "1", "root"]);
  assert.ok(built.documents[0].clusterTerms.includes("Product"));
});

test("buildSearchDocuments keeps generated titles separate while indexing persisted corrections", async () => {
  const { buildSearchDocuments, SearchIndex } = await loadSearch();
  const result = {
    ids: ["note.md"], leafLabels: [0], probabilities: [1], outlierProxy: [0], schemaVersion: 6, pca: { selected: 2 }, timings: {},
    hierarchy: { leaves: [0], merges: [], root: 0, nodes: [{ id: 0, children: [], descendantLeaves: [0], distance: 0, mass: 1 }], rootChildren: [0], splitMethod: "distance-knee-2-5" },
    hierarchyPlacements: [{ kind: "leaf", nodeId: 0, confidence: 1 }], titles: { "0": "Generated title" },
  };
  const notes = [{ path: "note.md", title: "Note", mtime: 1, hash: "hash", content: "body" }];
  const generated = buildSearchDocuments(notes, result);
  const stableKey = generated.clusters.find((cluster) => cluster.id === "0").stableClusterKey;
  const corrected = buildSearchDocuments(notes, result, {
    titleOverrides: [{ stableClusterKey: stableKey, title: "Manual title", createdAt: "2026-09-04T00:00:00.000Z", updatedAt: "2026-09-04T00:00:00.000Z" }],
    notePreferences: [{ notePath: "note.md", preferredClusterKey: stableKey, createdAt: "2026-09-04T00:00:00.000Z" }],
    groups: [{ groupId: "group", title: "Manual group", childClusterKeys: [stableKey], createdAt: "2026-09-04T00:00:00.000Z", updatedAt: "2026-09-04T00:00:00.000Z" }],
    feedback: [],
  });
  assert.equal(result.titles["0"], "Generated title");
  assert.equal(corrected.clusters.find((cluster) => cluster.id === "0").title, "Manual title");
  assert.equal(corrected.documents[0].manuallyAdjusted, true);
  assert.equal(corrected.documents[0].manualPreferredClusterKey, stableKey);
  assert.ok(corrected.documents[0].clusterTerms.includes("Manual group"));
  assert.ok(new SearchIndex(corrected.documents, corrected.clusters).search("manual title").notePaths.includes("note.md"));
});
