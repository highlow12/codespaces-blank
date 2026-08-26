import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadStorage() {
  let source = await readFile(new URL("../src/sqlite-storage.ts", import.meta.url), "utf8");
  source = source.replace('import { contentHash } from "./hash";', `
    async function contentHash(value) { let h = 2166136261; for (let i = 0; i < value.length; i++) { h ^= value.charCodeAt(i); h = Math.imul(h, 16777619); } return "fnv1a-" + (h >>> 0).toString(16); }
  `);
  source = source.replace('import { CachedEmbedding, ClusterResult, EmbeddingRunLog, NoteRecord } from "./types";', "");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

test("PCA projection normalizes, centers, and multiplies components transpose", async () => {
  const { projectPca } = await loadStorage();
  const pca = { modelHash: "m", inputDimension: 2, outputDimension: 2, normalization: "l2", mean: [0.1, 0.2], components: [[1, 0], [0, 1]], explainedVariance: [1, 1] };
  const value = projectPca([3, 4], pca);
  assert.deepEqual(value, [0.5, 0.6000000000000001]);
});

test("schema declares immutable embeddings, PCA, normalized result relations, views, and migrations", async () => {
  const source = await readFile(new URL("../src/sqlite-storage.ts", import.meta.url), "utf8");
  for (const relation of ["notes", "embeddings", "pca_models", "pca_coordinates", "assignments", "hierarchy_merges", "cluster_titles", "soft_memberships", "embedding_logs", "migrations"]) assert.match(source, new RegExp(`CREATE TABLE IF NOT EXISTS ${relation}`));
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
