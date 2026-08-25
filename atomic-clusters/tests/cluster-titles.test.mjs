import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";
import { createHash } from "node:crypto";

async function loadTitle() {
  let source = await readFile(new URL("../src/title.ts", import.meta.url), "utf8");
  source = source.replace('import { requestUrl } from "obsidian";', 'const requestUrl = async () => ({ arrayBuffer: new TextEncoder().encode("asset").buffer });').replace('import { ClusterResult, ClusterTitleCacheEntry, ClusterTitleStatus, HierarchyMerge, NoteRecord } from "./types";', '');
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}
class MemoryStorage { files = new Map(); reads = []; async exists(path) { return this.files.has(path); } async read(path) { this.reads.push(path); return this.files.get(path); } async write(path, data) { this.files.set(path, data); } async remove(path) { this.files.delete(path); } }

test("title model manager writes manifest last and distinguishes incomplete installs", async () => {
  const { TitleModelManager, TITLE_MODEL_DESCRIPTOR } = await loadTitle(); const storage = new MemoryStorage(); const manager = new TitleModelManager(storage, { ...TITLE_MODEL_DESCRIPTOR, modelSha256: createHash("sha256").update("asset").digest("hex") });
  assert.equal(await manager.status(), "missing");
  await manager.downloadModel(async () => true);
  assert.equal(await manager.status(), "installed");
  const manifest = [...storage.files.keys()].at(-1); assert.match(manifest, /manifest\.json$/);
  storage.files.delete(manifest); assert.equal(await manager.status(), "incomplete");
});

test("title model manager rejects a model whose bytes do not match the pinned SHA", async () => {
  const { TitleModelManager } = await loadTitle(); const storage = new MemoryStorage(); const manager = new TitleModelManager(storage);
  await assert.rejects(() => manager.downloadModel(async () => true), /SHA-256 mismatch/);
  assert.equal(await manager.status(), "missing");
});

test("title worker uses the bundled Transformers.js pipeline with local-only WebGPU assets", async () => {
  const worker = await readFile(new URL("../src/title-worker.ts", import.meta.url), "utf8");
  assert.match(worker, /@huggingface\/transformers/);
  assert.match(worker, /allowRemoteModels\s*=\s*false/);
  assert.match(worker, /localModelPath/);
  assert.match(worker, /device:\s*"webgpu"/);
  assert.match(worker, /dtype:\s*"q4f16"/);
  assert.match(worker, /wasmPaths\s*=\s*"https:\/\/atomic-clusters\.local\/ort\//);
  assert.match(worker, /wasmBinary\s*=\s*event\.data\.ortWasm/);
  assert.match(worker, /ort-wasm-simd-threaded\.jsep\.wasm/);
  assert.match(worker, /blocked non-local asset request/);
  assert.match(worker, /input instanceof Request \? input\.url/);
});

test("title prompt selection uses probability-ranked notes and bounded cleaned snippets", async () => {
  const { buildTitlePrompts, sanitizeTitle } = await loadTitle();
  const result = { schemaVersion: 2, ids: ["low.md", "high.md"], leafLabels: [0, 0], probabilities: [0.1, 0.9], outlierProxy: [0.9, 0.1], hierarchy: { leaves: [0], merges: [], root: 0 }, pca: {}, timings: {} };
  const notes = [{ path: "low.md", title: "Low", content: "*low*", hash: "a" }, { path: "high.md", title: "High", content: "# high", hash: "b" }];
  const prompt = buildTitlePrompts(result, notes)[0]; assert.match(prompt.text, /High/); assert.ok(prompt.text.indexOf("High") < prompt.text.indexOf("Low"));
  assert.equal(sanitizeTitle('"Title:  Hello\nworld"'), "Hello world"); assert.ok(prompt.text.length < 5000);
});

test("title generation failure is recorded without failing the cluster result", async () => {
  const { LocalClusterTitleGenerator, TitleModelManager, TITLE_MODEL_DESCRIPTOR } = await loadTitle(); const storage = new MemoryStorage(); const manager = new TitleModelManager(storage, { ...TITLE_MODEL_DESCRIPTOR, modelSha256: createHash("sha256").update("asset").digest("hex") });
  await manager.downloadModel(async () => true);
  const result = { schemaVersion: 2, ids: ["a.md", "b.md", "c.md"], leafLabels: [0, 0, 1], probabilities: [1, 1, 1], outlierProxy: [0, 0, 0], hierarchy: { leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 0.1, mass: 3 }], root: 2 }, pca: {}, timings: {} };
  const notes = result.ids.map((path, i) => ({ path, title: path, content: "body", hash: String(i) }));
  const generator = new LocalClusterTitleGenerator(manager, async () => { throw new Error("GPU initialization failed"); });
  const titled = await generator.generate(result, notes); assert.equal(titled.schemaVersion, 2); assert.equal(titled.titleGeneration.backend, "unavailable"); assert.equal(Object.keys(titled.titleGeneration.statuses).length, 3); assert.match(Object.values(titled.titleGeneration.errors)[0], /GPU/);
});
