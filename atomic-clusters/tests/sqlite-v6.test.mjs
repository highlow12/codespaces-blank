import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import initSqlJs from "sql.js";
import { transform } from "esbuild";

async function loadStorage() {
  let source = await readFile(new URL("../src/sqlite-storage.ts", import.meta.url), "utf8");
  source = source.replace(
    'import { contentHash } from "./hash";',
    `async function contentHash(value) {
      let hash = 2166136261;
      for (let index = 0; index < value.length; index++) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
      }
      return "fnv1a-" + (hash >>> 0).toString(16);
    }`
  );
  source = source.replace(/import \{[\s\S]*?\} from "\.\/types";/, "");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

class MemoryAdapter {
  files = new Map();
  flushes = 0;
  activeWrites = 0;
  maxActiveWrites = 0;
  yieldWrites = false;
  failWrites = false;

  async exists(path) { return this.files.has(path); }
  async readBinary(path) { return this.files.get(path); }
  async writeBinary(path, bytes) {
    if (this.failWrites) throw new Error("disk full");
    if (path.endsWith(".tmp") || path.endsWith("atomic-clusters.sqlite")) this.flushes++;
    this.activeWrites++;
    this.maxActiveWrites = Math.max(this.maxActiveWrites, this.activeWrites);
    if (this.yieldWrites) await Promise.resolve();
    this.files.set(path, bytes);
    this.activeWrites--;
  }
  async mkdir() {}
  async rename(from, to) {
    this.files.set(to, this.files.get(from));
    this.files.delete(from);
  }
  async remove(path) { this.files.delete(path); }
}

async function openStore(Storage, adapter = new MemoryAdapter(), options = {}) {
  const SQL = await initSqlJs();
  const store = await new Storage.SqliteClusterStore(adapter, SQL, options).open();
  return { store, adapter, SQL };
}

function notes(count, prefix = "note") {
  return Array.from({ length: count }, (_, index) => ({
    path: `${prefix}-${index}.md`, title: `${prefix}-${index}`, mtime: index,
    content: `content ${index}`, hash: `hash-${index}`
  }));
}

function v6Result() {
  return {
    schemaVersion: 6,
    ids: ["a.md", "b.md", "c.md"],
    leafLabels: [0, 1, -1],
    probabilities: [0.9, 0.8, 0],
    outlierProxy: [0.1, 0.2, 1],
    softMemberships: [[0.9, 0.1], [0.5, 0.5], [0, 0]],
    leafOrder: [0, 1],
    leafOrdering: [0, 1],
    memberships: [[0.9, 0.1], [0.5, 0.5], [0, 0]],
    pca: { selected: 2 },
    hierarchy: {
      leaves: [0, 1],
      merges: [{ id: 2, left: 0, right: 1, distance: 0.3, mass: 2 }],
      root: 2,
      nodes: [
        { id: 0, children: [], descendantLeaves: [0], distance: 0, mass: 1 },
        { id: 1, children: [], descendantLeaves: [1], distance: 0, mass: 1 },
        { id: 2, children: [0, 1], descendantLeaves: [0, 1], distance: 0.3, mass: 2 }
      ],
      rootChildren: [0, 1],
      splitMethod: "distance-knee-2-5"
    },
    hierarchyPlacements: [
      { kind: "leaf", nodeId: 0, confidence: 0.9 },
      { kind: "residual", nodeId: 2, confidence: 0.5 },
      { kind: "residual", nodeId: null, confidence: 0 }
    ],
    titles: { "0": "Old leaf", "2": "Old root" },
    timings: {}
  };
}

async function seedNotes(store, count = 3, prefix = "note") {
  const entries = notes(count, prefix);
  if (prefix === "") entries.forEach((note, index) => { note.path = `${String.fromCharCode(97 + index)}.md`; });
  await store.upsertNotes(entries);
}

test("Float32 persistence is little-endian and round-trips embedding values", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const blob = Storage.float32Blob([1, -2.5, 0.125]);
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  assert.equal(view.getFloat32(0, true), 1);
  assert.equal(view.getFloat32(4, true), -2.5);
  assert.equal(view.getFloat32(8, true), 0.125);
  assert.deepEqual(Storage.float32Values(blob), [1, -2.5, 0.125]);

  await store.upsertNote({ path: "a.md", title: "A", mtime: 0, content: "a", hash: "a" });
  await store.putEmbedding({ path: "a.md", provider: "p", model: "m", hash: "a", vector: [1, -2.5, 0.125] });
  assert.equal(store.query("SELECT typeof(vector_blob) AS kind FROM embeddings")[0].kind, "blob");
  const embedding = await store.getEmbedding("a.md", "p", "m", "a");
  assert.deepEqual(embedding.vector, [1, -2.5, 0.125]);
  store.close();
});

test("v6 hierarchy nodes, ordered children, root children, memberships, and placements round-trip", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  await seedNotes(store, 3, "");
  const result = v6Result();
  const resultId = await store.saveResult(result, { resultId: "v6" });

  const metadata = JSON.parse(store.query("SELECT result_json AS resultJson FROM results WHERE result_id=?", [resultId])[0].resultJson);
  assert.equal(metadata._normalizedV6.memberships, true);
  assert.equal(metadata.ids, undefined, "row-aligned ids must have one normalized source");
  assert.equal(metadata.hierarchy.nodes, undefined, "hierarchy structure must not be duplicated in JSON");
  assert.equal(metadata.softMemberships, undefined, "dense memberships must not be duplicated in JSON");

  assert.deepEqual(store.getHierarchyNodes(resultId), result.hierarchy.nodes);
  assert.deepEqual(store.query("SELECT child_id AS childId FROM root_children WHERE result_id=? ORDER BY ordinal", [resultId]), [{ childId: 0 }, { childId: 1 }]);
  assert.deepEqual(store.query("SELECT parent_id AS parentId,ordinal,child_id AS childId FROM hierarchy_children WHERE result_id=? ORDER BY parent_id,ordinal", [resultId]), [
    { parentId: 2, ordinal: 0, childId: 0 },
    { parentId: 2, ordinal: 1, childId: 1 }
  ]);
  assert.deepEqual(store.getHierarchyPlacements(resultId), [
    { path: "a.md", kind: "leaf", nodeId: 0, confidence: 0.9 },
    { path: "b.md", kind: "residual", nodeId: 2, confidence: 0.5 },
    { path: "c.md", kind: "residual", nodeId: null, confidence: 0 }
  ]);
  const membershipRows = store.getMembershipRows(resultId);
  assert.deepEqual(membershipRows.map(({ path }) => path), ["a.md", "b.md", "c.md"]);
  const expectedMemberships = [[0.9, 0.1], [0.5, 0.5], [0, 0]];
  membershipRows.forEach((row, index) => row.memberships.forEach((value, column) => {
    assert.ok(Math.abs(value - expectedMemberships[index][column]) <= 1e-6);
  }));
  const hydrated = await store.getResult(resultId);
  assert.equal(hydrated.schemaVersion, 6);
  assert.deepEqual(hydrated.ids, result.ids);
  assert.deepEqual(hydrated.hierarchy, result.hierarchy);
  assert.deepEqual(hydrated.hierarchyPlacements, result.hierarchyPlacements);
  store.close();
});

test("putEmbeddings persists 100 rows with one durable flush", async () => {
  const Storage = await loadStorage();
  const { store, adapter } = await openStore(Storage);
  const entries = notes(100, "embedding");
  await store.upsertNotes(entries);
  adapter.flushes = 0;
  const hashes = await store.putEmbeddings(entries.map((note, index) => ({
    path: note.path, provider: "provider", model: "model", hash: note.hash,
    vector: [index + 0.25, index + 0.5, index + 0.75]
  })));
  assert.equal(hashes.length, 100);
  assert.equal(adapter.flushes, 1);
  assert.equal(store.query("SELECT COUNT(*) AS count FROM embeddings")[0].count, 100);
  store.close();
});

test("projectMany persists 100 rows with one durable flush", async () => {
  const Storage = await loadStorage();
  const { store, adapter } = await openStore(Storage);
  const entries = notes(100, "project");
  await store.upsertNotes(entries);
  const model = { modelHash: "model", inputDimension: 3, outputDimension: 2, normalization: "none", mean: [0, 0, 0], components: [[1, 0, 0], [0, 1, 0]], explainedVariance: [1, 1] };
  await store.savePcaModel(model);
  adapter.flushes = 0;
  const coordinates = await store.projectMany(entries.map((note, index) => ({ path: note.path, vector: [index, index + 1, index + 2] })), model);
  assert.equal(coordinates.length, 100);
  assert.deepEqual(coordinates[0], [0, 1]);
  assert.deepEqual(coordinates[99], [99, 100]);
  assert.equal(adapter.flushes, 1);
  assert.equal(store.query("SELECT COUNT(*) AS count FROM pca_coordinates")[0].count, 100);
  store.close();
});

test("batch PCA coordinate reads preserve ordered paths and missing rows", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const entries = notes(3, "batch-pca");
  await store.upsertNotes(entries);
  const model = { modelHash: "batch-model", inputDimension: 2, outputDimension: 2, normalization: "none", mean: [0, 0], components: [[1, 0], [0, 1]], explainedVariance: [1, 1] };
  await store.savePcaModel(model);
  await store.projectMany(entries.map((note, index) => ({ path: note.path, vector: [index + 1, -(index + 1)] })), model);
  const coordinates = await store.getPcaCoordinatesMany([entries[2].path, "missing.md", entries[0].path, entries[2].path], model.modelHash);
  assert.deepEqual(coordinates[0], [3, -3]);
  assert.equal(coordinates[1], undefined);
  assert.deepEqual(coordinates[2], [1, -1]);
  assert.deepEqual(coordinates[3], [3, -3]);
  store.close();
});

test("store mutex serializes concurrent writes", async () => {
  const Storage = await loadStorage();
  const adapter = new MemoryAdapter();
  adapter.yieldWrites = true;
  const { store } = await openStore(Storage, adapter);
  await Promise.all(notes(24, "mutex").map((note) => store.upsertNote(note)));
  assert.equal(adapter.maxActiveWrites, 1);
  assert.equal(store.query("SELECT COUNT(*) AS count FROM notes")[0].count, 24);
  store.close();
});

test("only the latest embedding, result, and embedding log are retained", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage, new MemoryAdapter(), { now: (() => { let tick = 0; return () => `2020-01-01T00:00:00.00${tick++}Z`; })() });
  await store.upsertNote({ path: "a.md", title: "A", mtime: 0, content: "a", hash: "a" });
  await store.putEmbedding({ path: "a.md", provider: "p", model: "m", hash: "a", vector: [1] });
  await store.putEmbedding({ path: "a.md", provider: "p", model: "m", hash: "a", vector: [2] });
  assert.equal(store.query("SELECT COUNT(*) AS count FROM embeddings WHERE path='a.md' AND provider='p' AND model='m'")[0].count, 1);
  assert.deepEqual((await store.getEmbedding("a.md", "p", "m", "a")).vector, [2]);

  const first = v6Result();
  const second = { ...v6Result(), titles: { "0": "New leaf", "2": "New root" } };
  await store.upsertNotes(notes(3, ""));
  await store.saveResult(first, { resultId: "first" });
  await store.saveResult(second, { resultId: "second" });
  assert.equal(store.query("SELECT COUNT(*) AS count FROM results")[0].count, 1);
  assert.equal((await store.getResult()).titles["0"], "New leaf");

  const log = (provider) => ({ version: 1, startedAt: provider, completedAt: provider, provider, model: "m", total: 0, succeeded: 0, failed: 0, cached: 0, entries: [] });
  await store.saveEmbeddingLog(log("old"));
  await store.saveEmbeddingLog(log("new"));
  assert.equal(store.query("SELECT COUNT(*) AS count FROM embedding_logs")[0].count, 1);
  assert.equal((await store.loadLatestEmbeddingLog()).provider, "new");
  store.close();
});

test("placement length mismatch is rejected before persistence", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const result = v6Result();
  const malformed = { ...result, hierarchyPlacements: result.hierarchyPlacements.slice(0, 2) };
  assert.throws(() => Storage.validateClusterResultAlignment(malformed), /align/);
  await assert.rejects(store.saveResult(malformed, { resultId: "malformed" }), /align/);
  assert.equal(store.query("SELECT COUNT(*) AS count FROM results")[0].count, 0);
  store.close();
});

test("title-only patch preserves normalized structural row counts", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  await seedNotes(store, 3, "");
  const resultId = await store.saveResult(v6Result(), { resultId: "titles" });
  const relations = ["assignments", "hierarchy_merges", "hierarchy_nodes", "hierarchy_children", "root_children", "hierarchy_placements", "soft_memberships", "membership_rows"];
  const before = Object.fromEntries(relations.map((relation) => [relation, store.query(`SELECT COUNT(*) AS count FROM ${relation} WHERE result_id=?`, [resultId])[0].count]));
  assert.equal(typeof store.patchResultTitles, "function", "v6 acceptance requires an atomic title-only patch API named patchResultTitles");
  await store.patchResultTitles(resultId, { "0": "Patched leaf", "2": "Patched root" });
  const after = Object.fromEntries(relations.map((relation) => [relation, store.query(`SELECT COUNT(*) AS count FROM ${relation} WHERE result_id=?`, [resultId])[0].count]));
  assert.deepEqual(after, before);
  assert.deepEqual((await store.getResult(resultId)).titles, { "0": "Patched leaf", "2": "Patched root" });
  store.close();
});

test("lazy visualization patch round-trips and preserves structural rows", async () => {
  const Storage = await loadStorage();
  const { store, adapter } = await openStore(Storage);
  await seedNotes(store, 3, "");
  const resultId = await store.saveResult(v6Result(), { resultId: "visualization-patch" });
  const relations = ["assignments", "hierarchy_merges", "hierarchy_nodes", "hierarchy_children", "root_children", "hierarchy_placements", "soft_memberships", "membership_rows"];
  const before = Object.fromEntries(relations.map((relation) => [relation, store.query(`SELECT COUNT(*) AS count FROM ${relation} WHERE result_id=?`, [resultId])[0].count]));
  adapter.flushes = 0;
  await store.patchResultVisualization(resultId, {
    coordinates: [[10, 11], [20, 21], [30, 31]],
    labels: [0, 1, -1],
    configuration: { runtime: "umap-js", seed: 42, nComponents: 2, nNeighbors: 3, minDist: 0.1, spread: 1 },
    timings: { umapMs: 12 }
  });
  assert.equal(adapter.flushes, 1);
  const after = Object.fromEntries(relations.map((relation) => [relation, store.query(`SELECT COUNT(*) AS count FROM ${relation} WHERE result_id=?`, [resultId])[0].count]));
  assert.deepEqual(after, before);
  assert.deepEqual(store.getVisualization(resultId), [
    { path: "a.md", x: 10, y: 11, leafLabel: 0 },
    { path: "b.md", x: 20, y: 21, leafLabel: 1 },
    { path: "c.md", x: 30, y: 31, leafLabel: -1 }
  ]);
  const result = await store.getResult(resultId);
  assert.deepEqual(result.visualization.coordinates, [[10, 11], [20, 21], [30, 31]]);
  assert.deepEqual(result.visualization.labels, [0, 1, -1]);
  assert.deepEqual(result.hierarchyPlacements, v6Result().hierarchyPlacements);
  store.close();
});

test("lazy visualization patch rejects misaligned rows without changing metadata or structure", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  await seedNotes(store, 3, "");
  const resultId = await store.saveResult(v6Result(), { resultId: "bad-visualization-patch" });
  const before = store.query("SELECT result_json FROM results WHERE result_id=?", [resultId])[0].result_json;
  await assert.rejects(store.patchResultVisualization(resultId, {
    coordinates: [[1, 2]], labels: [0],
    configuration: { runtime: "umap-js", seed: 1, nComponents: 2, nNeighbors: 2, minDist: 0.1, spread: 1 }
  }), /align/);
  assert.equal(store.query("SELECT result_json FROM results WHERE result_id=?", [resultId])[0].result_json, before);
  assert.equal(store.query("SELECT COUNT(*) AS count FROM visualization_points WHERE result_id=?", [resultId])[0].count, 0);
  store.close();
});

test("v5 to v6 migration leaves the old file intact on flush failure and succeeds on retry", async () => {
  const Storage = await loadStorage();
  const SQL = await initSqlJs();
  const legacy = new SQL.Database();
  legacy.run(Storage.SQLITE_SCHEMA);
  legacy.run("INSERT INTO metadata(key,value) VALUES('schema_version','5')");
  legacy.run("INSERT INTO notes(path,title,mtime,content_hash,content) VALUES('a.md','A',0,'a','a')");
  legacy.run("INSERT INTO embeddings(path,provider,model,embedding_hash,note_content_hash,dimension,vector_json,vector_blob,created_at) VALUES('a.md','p','m','e','a',2,'[1.25,-2.5]',NULL,'2020')");
  const adapter = new MemoryAdapter();
  adapter.files.set(Storage.SQLITE_PATH, legacy.export().slice().buffer);
  legacy.close();

  adapter.failWrites = true;
  await assert.rejects(new Storage.SqliteClusterStore(adapter, SQL).open(), /disk full/);
  const unchanged = new SQL.Database(new Uint8Array(adapter.files.get(Storage.SQLITE_PATH)));
  assert.equal(unchanged.exec("SELECT value FROM metadata WHERE key='schema_version'")[0].values[0][0], "5");
  assert.equal(unchanged.exec("SELECT vector_blob FROM embeddings")[0].values[0][0], null);
  unchanged.close();

  adapter.failWrites = false;
  const migrated = await new Storage.SqliteClusterStore(adapter, SQL).open();
  assert.equal(migrated.query("SELECT value FROM metadata WHERE key='schema_version'")[0].value, "6");
  assert.equal(migrated.query("SELECT typeof(vector_blob) AS kind FROM embeddings")[0].kind, "blob");
  assert.equal(migrated.query("SELECT vector_json AS vectorJson FROM embeddings")[0].vectorJson, null, "converted numeric JSON is removed when the column is nullable");
  assert.deepEqual((await migrated.getEmbedding("a.md", "p", "m", "a")).vector, [1.25, -2.5]);
  migrated.close();
});

test("v5 latest result is converted to normalized v6 hierarchy and terminal placements", async () => {
  const Storage = await loadStorage();
  const SQL = await initSqlJs();
  const legacy = new SQL.Database();
  legacy.run(Storage.SQLITE_SCHEMA);
  legacy.run("INSERT INTO metadata(key,value) VALUES('schema_version','5')");
  for (const note of ["a.md", "b.md", "c.md", "d.md"]) legacy.run("INSERT INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?)", [note, note, 0, note, note]);
  const result = {
    schemaVersion: 5, ids: ["a.md", "b.md", "c.md", "d.md"], leafLabels: [0, 1, -1, 1],
    probabilities: [.9, .5, 0, .8], outlierProxy: [.1, .5, 1, .2],
    softMemberships: [[.9, .1], [.5, .5], [0, 0], [.2, .8]], leafOrder: [0, 1],
    pca: {}, hierarchy: { leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: .3, mass: 3 }], root: 2 },
    visualization: { coordinates: [[1, 2], [3, 4], [5, 6], [7, 8]], labels: [0, 1, -1, 1], configuration: { runtime: "umap-js", seed: 42, nComponents: 2, nNeighbors: 2, minDist: .1, spread: 1 } },
    titles: { "0": "Old leaf", "2": "Old root" }, timings: {}
  };
  legacy.run("INSERT INTO results(result_id,schema_version,created_at,result_json) VALUES(?,?,?,?)", ["legacy-v5", 5, "2020", JSON.stringify(result)]);
  const adapter = new MemoryAdapter(); adapter.files.set(Storage.SQLITE_PATH, legacy.export().slice().buffer); legacy.close();

  const migrated = await new Storage.SqliteClusterStore(adapter, SQL).open();
  const hydrated = await migrated.getResult("legacy-v5");
  assert.equal(hydrated.schemaVersion, 6);
  assert.deepEqual(hydrated.ids, result.ids);
  assert.deepEqual(hydrated.leafLabels, result.leafLabels);
  assert.equal(hydrated.hierarchy.splitMethod, "distance-knee-2-5");
  assert.deepEqual(hydrated.hierarchy.rootChildren, [0, 1]);
  assert.deepEqual(hydrated.hierarchyPlacements, [
    { kind: "leaf", nodeId: 0, confidence: .9 },
    { kind: "residual", nodeId: 2, confidence: .5 },
    { kind: "residual", nodeId: null, confidence: 0 },
    { kind: "leaf", nodeId: 1, confidence: .8 }
  ]);
  assert.deepEqual(migrated.query("SELECT schema_version AS schemaVersion FROM results"), [{ schemaVersion: 6 }]);
  assert.equal(migrated.query("SELECT COUNT(*) AS count FROM hierarchy_nodes")[0].count, 3);
  assert.equal(migrated.query("SELECT COUNT(*) AS count FROM hierarchy_children")[0].count, 2);
  assert.equal(migrated.query("SELECT COUNT(*) AS count FROM root_children")[0].count, 2);
  assert.equal(migrated.query("SELECT COUNT(*) AS count FROM hierarchy_placements")[0].count, 4);
  assert.equal(migrated.query("SELECT mass FROM hierarchy_nodes WHERE node_id=2")[0].mass, 3);
  assert.equal(migrated.query("SELECT COUNT(*) AS count FROM membership_rows")[0].count, 4);
  assert.equal(migrated.query("SELECT COUNT(*) AS count FROM soft_memberships")[0].count, 8);
  assert.equal(migrated.query("SELECT COUNT(*) AS count FROM visualization_points")[0].count, 4);
  const metadata = JSON.parse(migrated.query("SELECT result_json AS resultJson FROM results")[0].resultJson);
  assert.equal(metadata.ids, undefined);
  assert.equal(metadata.hierarchy.nodes, undefined);
  migrated.close();
});

test("legacy JSON archive rename is retried after the data was already imported", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const path = ".obsidian/plugins/atomic-clusters/embedding-log.json";
  const files = new Map([[path, JSON.stringify({ version: 1, startedAt: "s", completedAt: "e", provider: "p", model: "m", total: 0, succeeded: 0, failed: 0, cached: 0, entries: [] })]]);
  let failRename = true; let renameAttempts = 0;
  const legacyAdapter = {
    async read(name) { if (!files.has(name)) throw new Error("missing"); return files.get(name); },
    async rename(from, to) { renameAttempts++; if (failRename) throw new Error("busy"); files.set(to, files.get(from)); files.delete(from); }
  };
  assert.deepEqual((await Storage.migrateLegacyAdapter(store, legacyAdapter)).migrated, ["embedding-log.json"]);
  assert.equal(files.has(path), true);
  failRename = false;
  assert.deepEqual((await Storage.migrateLegacyAdapter(store, legacyAdapter)).migrated, []);
  assert.equal(renameAttempts, 2);
  assert.equal(files.has(`${path}.legacy`), true);
  store.close();
});

test("legacy JSON imports are normalized to v6 and embedding vectors get a Float32 blob", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const legacyResult = {
    schemaVersion: 5,
    ids: ["a.md", "b.md"],
    leafLabels: [0, 1],
    probabilities: [0.9, 0.8],
    outlierProxy: [0.1, 0.2],
    softMemberships: [[0.9, 0], [0, 0.8]],
    leafOrder: [0, 1],
    pca: {},
    hierarchy: { leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 0.4, mass: 2 }], root: 2 },
    timings: {}
  };
  const migrated = await Storage.migrateLegacyJson(store, {
    embeddingCache: JSON.stringify({ embeddings: [{ path: "a.md", provider: "p", model: "m", hash: "a", vector: [1.25, -2.5] }] }),
    result: JSON.stringify(legacyResult)
  });
  assert.deepEqual(migrated.migrated, ["embedding-cache.json", "cluster-result.json"]);
  assert.equal(store.query("SELECT typeof(vector_blob) AS kind FROM embeddings")[0].kind, "blob");
  assert.deepEqual((await store.getEmbedding("a.md", "p", "m", "a")).vector, [1.25, -2.5]);
  const hydrated = await store.getResult();
  assert.equal(hydrated.schemaVersion, 6);
  assert.equal(hydrated.hierarchy.splitMethod, "distance-knee-2-5");
  assert.equal(hydrated.hierarchyPlacements.length, 2);
  assert.deepEqual(store.query("SELECT schema_version AS schemaVersion FROM results"), [{ schemaVersion: 6 }]);
  store.close();
});
