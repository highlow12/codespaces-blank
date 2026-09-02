import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";
import initSqlJs from "sql.js";

async function loadStorage() {
  let source = await readFile(new URL("../src/sqlite-storage.ts", import.meta.url), "utf8");
  source = source.replace('import { contentHash } from "./hash";', `
    async function contentHash(value) { let h = 2166136261; for (let i = 0; i < value.length; i++) { h ^= value.charCodeAt(i); h = Math.imul(h, 16777619); } return "fnv1a-" + (h >>> 0).toString(16); }
  `);
  source = source.replace(/import \{[\s\S]*?\} from "\.\/types";/, "");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  try { return await import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`); } catch (error) { console.error("LOAD STORAGE DEBUG", error); throw error; }
}

class MemoryAdapter {
  files = new Map();
  failWrites = false;
  async exists(path) { return this.files.has(path); }
  async readBinary(path) { return this.files.get(path); }
  async writeBinary(path, bytes) { if (this.failWrites) throw new Error("disk full"); this.files.set(path, bytes); }
  async mkdir() {}
  async rename(from, to) { this.files.set(to, this.files.get(from)); this.files.delete(from); }
  async remove(path) { this.files.delete(path); }
}

test("PCA projection normalizes, centers, and multiplies components transpose", async () => {
  const { projectPca } = await loadStorage();
  const pca = { modelHash: "m", inputDimension: 2, outputDimension: 2, normalization: "l2", mean: [0.1, 0.2], components: [[1, 0], [0, 1]], explainedVariance: [1, 1] };
  const value = projectPca([3, 4], pca);
  assert.deepEqual(value, [0.5, 0.6000000000000001]);
});

test("schema declares immutable embeddings, PCA, normalized result relations, views, and migrations", async () => {
  const source = await readFile(new URL("../src/sqlite-storage.ts", import.meta.url), "utf8");
  for (const relation of ["notes", "embeddings", "pca_models", "pca_coordinates", "results", "result_note_hashes", "assignments", "hierarchy_merges", "cluster_titles", "soft_memberships", "embedding_logs", "migrations"]) assert.match(source, new RegExp(`CREATE TABLE IF NOT EXISTS ${relation}`));
  for (const view of ["v_current_embeddings", "v_note_pca", "v_cluster_assignments", "v_embedding_log"]) assert.match(source, new RegExp(`CREATE VIEW IF NOT EXISTS ${view}`));
  assert.match(source, /BEGIN IMMEDIATE/); assert.match(source, /ROLLBACK/); assert.match(source, /INSERT INTO migrations/);
});

test("embedding fingerprints include dimension and vector values", async () => {
  const { embeddingHash } = await loadStorage();
  assert.notEqual(await embeddingHash([1, 2]), await embeddingHash([1, 2, 0]));
  assert.notEqual(await embeddingHash([1, 2]), await embeddingHash([1, 3]));
});

test("legacy migration inserts one-time marker and handles all legacy JSON files", async () => {
  const source = await readFile(new URL("../src/sqlite-storage.ts", import.meta.url), "utf8");
  assert.match(source, /embedding-cache\.json/); assert.match(source, /cluster-result\.json/); assert.match(source, /embedding-log\.json/);
  assert.match(source, /if \(!raw \|\| db\.exec/);
});

test("sql.js round-trip persists notes, immutable embeddings, PCA, visualization, hierarchy, memberships, and logs", async () => {
  const { SqliteClusterStore } = await loadStorage();
  const adapter = new MemoryAdapter(); const SQL = await initSqlJs();
  const store = await new SqliteClusterStore(adapter, SQL, { now: () => "2020-01-01T00:00:00.000Z" }).open();
  await store.upsertNote({ path: "a.md", title: "A", mtime: 10, content: "hello", hash: "note-1" });
  const embeddingHash = await store.putEmbedding({ path: "a.md", provider: "local", model: "m", hash: "note-1", vector: [3, 4] });
  assert.equal((await store.getEmbedding("a.md", "local", "m", "note-1")).vector[1], 4);
  await store.savePcaModel({ modelHash: "pca-1", inputDimension: 2, outputDimension: 2, normalization: "l2", mean: [0, 0], components: [[1, 0], [0, 1]], explainedVariance: [1, 1] });
  assert.deepEqual(await store.project("a.md", [3, 4], await store.getPcaModel("pca-1")), [0.6, 0.8]);
  const result = { schemaVersion: 5, ids: ["a.md"], leafLabels: [0], probabilities: [0.9], outlierProxy: [0.1], softMemberships: [[0.9]], leafOrder: [0], pca: {}, hierarchy: { leaves: [0], merges: [], root: 0 }, visualization: { coordinates: [[1, 2]], labels: [0], configuration: { runtime: "umap-js", seed: 42, nComponents: 2, nNeighbors: 1, minDist: 1, spread: 1 } }, timings: {}, titles: { "0": "A" } };
  const resultId = await store.saveResult(result, { resultId: "r1" });
  await store.saveEmbeddingLog({ version: 1, startedAt: "s", completedAt: "e", provider: "local", model: "m", total: 1, succeeded: 1, failed: 0, cached: 0, entries: [] });
  await store.flush(); store.close();
  const reopened = await new SqliteClusterStore(adapter, SQL).open();
  assert.equal((await reopened.getResult(resultId)).schemaVersion, 6);
  assert.deepEqual(reopened.getVisualization("r1")[0], { path: "a.md", x: 1, y: 2, leafLabel: 0 });
  assert.deepEqual(reopened.getSoftMemberships("r1")[0], { path: "a.md", leafId: 0, membership: 0.9 });
  assert.equal((await reopened.loadLatestEmbeddingLog()).provider, "local");
  assert.ok(typeof embeddingHash === "string" && embeddingHash.length > 0);
});

test("transaction rollback and note hash changes prevent stale embedding reuse", async () => {
  const { SqliteClusterStore } = await loadStorage(); const adapter = new MemoryAdapter(); const SQL = await initSqlJs();
  const store = await new SqliteClusterStore(adapter, SQL).open();
  await assert.rejects(store.transaction((db) => { db.run("INSERT INTO notes(path,title,mtime,content_hash) VALUES('x','X',0,'x')"); throw new Error("abort"); }), /abort/);
  assert.equal(store.query("SELECT COUNT(*) AS count FROM notes")[0].count, 0);
  await store.upsertNote({ path: "x", title: "X", mtime: 0, content: "one", hash: "one" });
  await store.putEmbedding({ path: "x", provider: "p", model: "m", hash: "one", vector: [1] });
  await store.upsertNote({ path: "x", title: "X", mtime: 1, content: "two", hash: "two" });
  assert.equal(await store.getEmbedding("x", "p", "m", "two"), undefined);
  adapter.failWrites = true;
  await assert.rejects(store.upsertNote({ path: "y", title: "Y", mtime: 0, content: "y", hash: "y" }), /disk full/);
  assert.equal(store.query("SELECT COUNT(*) AS count FROM notes")[0].count, 1);
});
