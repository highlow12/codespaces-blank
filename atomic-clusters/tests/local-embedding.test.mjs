import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadEmbedding() {
  const source = await readFile(new URL("../src/embedding.ts", import.meta.url), "utf8");
  const stubbed = source
    .replace('import { requestUrl } from "obsidian";', 'const requestUrl = async ({ url }) => ({ arrayBuffer: new TextEncoder().encode(url.includes("tokenizer") ? "tokenizer" : "onnx").buffer });')
    .replace('import * as ort from "onnxruntime-web/wasm";', 'const ort = { env: { wasm: {} }, Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }, InferenceSession: { async create() { return { inputNames: ["input_ids", "attention_mask"], outputNames: ["last_hidden_state"], async run(feeds) { const batch = feeds.input_ids.dims[0]; const sequence = feeds.input_ids.dims[1]; return { last_hidden_state: { dims: [batch, sequence, 2], data: new Float32Array(batch * sequence * 2).fill(1) } }; } }; } } };');
  const result = await transform(`${stubbed}\nexport { ort };`, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

async function loadOrtAssets() {
  const source = await readFile(new URL("../src/ort-assets.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020", platform: "node" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

class MemoryStorage {
  files = new Map();
  async exists(path) { return this.files.has(path); }
  async read(path) { if (!this.files.has(path)) throw new Error(`missing ${path}`); return this.files.get(path); }
  async write(path, value) { this.files.set(path, value); }
  async remove(path) { this.files.delete(path); }
}

test("local model installation is explicit, integrity checked, and deletable", async () => {
  const { LocalModelManager } = await loadEmbedding();
  const storage = new MemoryStorage();
  const manager = new LocalModelManager(storage);
  assert.equal(await manager.status(), "missing");
  await assert.rejects(() => manager.downloadModel(async () => false), /cancelled/);
  const progress = [];
  await manager.downloadModel(async () => true, (update) => progress.push(update));
  assert.deepEqual(progress.map((update) => update.phase), ["consent", "consent", "model", "model", "tokenizer", "tokenizer", "verify", "install", "complete"]);
  assert.equal(progress.at(-1).progress, 1);
  assert.equal(await manager.status(), "installed");
  const artifact = await manager.load();
  assert.ok(artifact.model.byteLength > 0);
  storage.files.set("multilingual-e5-small/2024-05-01/model.onnx", new Uint8Array([0]).buffer);
  assert.equal(await manager.status(), "corrupt");
  await manager.deleteModel();
  assert.equal(await manager.status(), "missing");
});

test("local provider versions cache keys and validates 384-dimensional vectors", async () => {
  const { LocalEmbeddingProvider, LOCAL_MODEL_VERSION } = await loadEmbedding();
  const settings = { localModel: "multilingual-e5-small" };
  const notes = [{ path: "a.md", title: "A", content: "hello", hash: "h", mtime: 1 }];
  const provider = new LocalEmbeddingProvider(settings, async () => [new Array(384).fill(0).map((_, index) => index / 384)]);
  const result = await provider.embed(notes);
  assert.equal(result[0].model, `multilingual-e5-small@${LOCAL_MODEL_VERSION}`);
  assert.deepEqual(await new LocalEmbeddingProvider(settings, async () => [[1, 2]]).embed(notes), []);
});

test("local provider logs per-note failures without exposing content or vectors", async () => {
  const { LocalEmbeddingProvider } = await loadEmbedding();
  const settings = { localModel: "multilingual-e5-small" };
  const notes = [
    { path: "ok.md", title: "OK", content: "private content", hash: "a", mtime: 1 },
    { path: "bad.md", title: "Bad", content: "secret note", hash: "b", mtime: 2 }
  ];
  const entries = [];
  const provider = new LocalEmbeddingProvider(settings, async (texts) => { if (texts[0].includes("Bad")) { const error = new Error("provider failed secret=should-not-leak"); error.cause = new Error("backend detail"); throw error; } return [new Array(384).fill(0.1)]; });
  const vectors = await provider.embed(notes, undefined, (entry) => entries.push(entry));
  assert.equal(vectors.length, 1);
  assert.deepEqual(entries.map((entry) => [entry.path, entry.status]), [["ok.md", "success"], ["bad.md", "failure"]]);
  assert.match(entries[1].error, /redacted/);
  assert.match(entries[1].error, /backend detail/);
  assert.doesNotMatch(JSON.stringify(entries), /private content|secret note|should-not-leak/);
  assert.ok(entries.every((entry) => entry.provider === "local" && entry.model.includes("multilingual-e5-small") && typeof entry.timestamp === "string" && typeof entry.durationMs === "number"));
});

test("local provider fails fast on systemic ONNX backend initialization errors", async () => {
  const { LocalEmbeddingProvider, LocalInferenceBackendError } = await loadEmbedding();
  const notes = Array.from({ length: 3 }, (_, index) => ({ path: `${index}.md`, title: `${index}`, content: "note", hash: `${index}`, mtime: index }));
  let attempts = 0;
  const provider = new LocalEmbeddingProvider({ localModel: "multilingual-e5-small" }, undefined, { async load() { return { model: new ArrayBuffer(1), tokenizer: new ArrayBuffer(1) }; } }, async () => ({ async embed() { attempts++; throw new LocalInferenceBackendError("ORT asset load failed"); } }));
  await assert.rejects(() => provider.embed(notes), /ORT asset load failed/);
  assert.equal(attempts, 1);
});

test("ORT runtime mean-pools masked tokens and emits unit vectors", async () => {
  const { OrtEmbeddingRuntime } = await loadEmbedding();
  const fakeSession = { inputNames: ["input_ids", "attention_mask"], outputNames: ["last_hidden_state"], async run() { return { last_hidden_state: { dims: [1, 2, 2], data: new Float32Array([3, 0, 0, 4]) } }; } };
  const ort = { Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }, InferenceSession: { async create() { return fakeSession; } } };
  const runtime = new OrtEmbeddingRuntime(ort, { encode() { return { inputIds: [[1, 2]], attentionMask: [[1, 0]] }; } });
  const [vector] = await runtime.embed(["passage: test"], { model: new ArrayBuffer(1) });
  assert.equal(vector.length, 2);
  assert.ok(Math.abs(Math.hypot(...vector) - 1) < 1e-6);
  assert.ok(vector[0] > 0.99 && Math.abs(vector[1]) < 1e-6);
});

test("ORT runtime supplies zero token_type_ids when the model requires them", async () => {
  const { OrtEmbeddingRuntime } = await loadEmbedding();
  let captured;
  const ort = { Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }, InferenceSession: { async create() { return { inputNames: ["input_ids", "attention_mask", "token_type_ids"], outputNames: ["last_hidden_state"], async run(feeds) { captured = feeds.token_type_ids; return { last_hidden_state: { dims: [1, 2, 2], data: new Float32Array([1, 0, 0, 1]) } }; } }; } } };
  const runtime = new OrtEmbeddingRuntime(ort, { encode() { return { inputIds: [[1, 2]], attentionMask: [[1, 1]] }; } });
  await runtime.embed(["probe"], { model: new ArrayBuffer(1) });
  assert.deepEqual(Array.from(captured.data), [0n, 0n]);
  assert.deepEqual(captured.dims, [1, 2]);
});

test("ORT runtime rejects unknown required model inputs clearly", async () => {
  const { OrtEmbeddingRuntime } = await loadEmbedding();
  const ort = { Tensor: class {}, InferenceSession: { async create() { return { inputNames: ["input_ids", "attention_mask", "position_ids"], outputNames: ["last_hidden_state"] }; } } };
  const runtime = new OrtEmbeddingRuntime(ort, { encode() { return { inputIds: [[1]], attentionMask: [[1]] }; } });
  await assert.rejects(() => runtime.embed(["probe"], { model: new ArrayBuffer(1) }), /Unsupported local ONNX required inputs: position_ids/);
});

test("ORT runtime reports every inference batch", async () => {
  const { OrtEmbeddingRuntime } = await loadEmbedding();
  const ort = { Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }, InferenceSession: { async create() { return { inputNames: ["input_ids", "attention_mask"], outputNames: ["last_hidden_state"], async run(feeds) { const batch = feeds.input_ids.dims[0]; const sequence = feeds.input_ids.dims[1]; return { last_hidden_state: { dims: [batch, sequence, 2], data: new Float32Array(batch * sequence * 2).fill(1) } }; } }; } } };
  const progress = [];
  const runtime = new OrtEmbeddingRuntime(ort, { encode(texts) { return { inputIds: texts.map(() => [1]), attentionMask: texts.map(() => [1]) }; } }, 2);
  const vectors = await runtime.embed(["a", "b", "c"], { model: new ArrayBuffer(1) }, (done, total) => progress.push([done, total]));
  assert.equal(vectors.length, 3);
  assert.deepEqual(progress, [[0, 3], [2, 3], [2, 3], [3, 3]]);
});

test("ORT runtime cancellation stops before the next batch", async () => {
  const { OrtEmbeddingRuntime } = await loadEmbedding();
  const controller = new AbortController();
  let runs = 0;
  const ort = { Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }, InferenceSession: { async create() { return { inputNames: ["input_ids", "attention_mask"], outputNames: ["last_hidden_state"], async run(feeds) { runs++; controller.abort(); const batch = feeds.input_ids.dims[0]; const sequence = feeds.input_ids.dims[1]; return { last_hidden_state: { dims: [batch, sequence, 2], data: new Float32Array(batch * sequence * 2).fill(1) } }; } }; } } };
  const runtime = new OrtEmbeddingRuntime(ort, { encode(texts) { return { inputIds: texts.map(() => [1]), attentionMask: texts.map(() => [1]) }; } }, 1);
  await assert.rejects(() => runtime.embed(["a", "b"], { model: new ArrayBuffer(1) }, undefined, controller.signal), /Clustering cancelled/);
  assert.equal(runs, 1);
});

test("local provider cancellation does not retry notes after a cancelled batch", async () => {
  const { LocalEmbeddingProvider } = await loadEmbedding();
  const controller = new AbortController();
  let calls = 0;
  const provider = new LocalEmbeddingProvider({ localModel: "multilingual-e5-small" }, async () => {
    calls++;
    controller.abort();
    return [new Array(384).fill(0.1), new Array(384).fill(0.1)];
  });
  const notes = [
    { path: "a.md", title: "A", content: "a", hash: "a", mtime: 1 },
    { path: "b.md", title: "B", content: "b", hash: "b", mtime: 2 }
  ];
  await assert.rejects(() => provider.embed(notes, undefined, undefined, controller.signal), /Clustering cancelled/);
  assert.equal(calls, 1);
});

test("Unigram tokenizer bounds very long notes before dynamic programming", async () => {
  const { UnigramTokenizer } = await loadEmbedding();
  const tokenizer = new UnigramTokenizer(JSON.stringify({ model: { type: "Unigram", unk_id: 3, vocab: [["▁", -0.1], ["a", -0.2], ["<unk>", -5]] }, added_tokens: [{ id: 0, content: "<s>" }, { id: 1, content: "<pad>" }, { id: 2, content: "</s>" }] }));
  const batch = tokenizer.encode(["a".repeat(2_000_000)], 512);
  assert.equal(batch.inputIds.length, 1);
  assert.ok(batch.inputIds[0].length <= 512);
  assert.equal(batch.attentionMask[0].length, batch.inputIds[0].length);
});

test("ORT runtime caches the initialized session for preflight and bulk inference", async () => {
  const { OrtEmbeddingRuntime } = await loadEmbedding();
  let creates = 0;
  const ort = { Tensor: class { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }, InferenceSession: { async create() { creates++; return { inputNames: ["input_ids", "attention_mask"], outputNames: ["last_hidden_state"], async run(feeds) { const batch = feeds.input_ids.dims[0]; const sequence = feeds.input_ids.dims[1]; return { last_hidden_state: { dims: [batch, sequence, 2], data: new Float32Array(batch * sequence * 2).fill(1) } }; } }; } } };
  const runtime = new OrtEmbeddingRuntime(ort, { encode(texts) { return { inputIds: texts.map(() => [1]), attentionMask: texts.map(() => [1]) }; } });
  const artifact = { model: new ArrayBuffer(1) };
  await runtime.embed(["probe"], artifact);
  await runtime.embed(["bulk"], artifact);
  assert.equal(creates, 1);
});

test("local runtime preflight probes once and reuses the runtime for embedding", async () => {
  const { LocalEmbeddingProvider } = await loadEmbedding();
  let factoryCalls = 0;
  let inferenceCalls = 0;
  const provider = new LocalEmbeddingProvider({ localModel: "multilingual-e5-small" }, undefined, { async load() { return { model: new ArrayBuffer(1), tokenizer: new ArrayBuffer(1) }; } }, async () => { factoryCalls++; return { async embed() { inferenceCalls++; return [new Array(384).fill(0.1)]; } }; });
  const phases = [];
  await provider.preflight((update) => phases.push(update.phase));
  await provider.embed([{ path: "a.md", title: "A", content: "hello", hash: "h", mtime: 1 }]);
  assert.deepEqual(phases, ["configuration", "model", "session", "complete"]);
  assert.equal(factoryCalls, 1);
  assert.equal(inferenceCalls, 2);
});

test("default factory uses bundled Unigram tokenizer and ORT runtime without an override", async () => {
  const { configureLocalOrtAssets, defaultLocalRuntimeFactory } = await loadEmbedding();
  configureLocalOrtAssets("file:///vault/.obsidian/plugins/atomic-clusters/");
  const tokenizer = JSON.stringify({ model: { type: "Unigram", unk_id: 3, vocab: [["▁", -0.1], ["hello", -0.2], ["world", -0.3], ["<unk>", -5]] }, added_tokens: [{ id: 0, content: "<s>" }, { id: 1, content: "<pad>" }, { id: 2, content: "</s>" }] });
  const runtime = await defaultLocalRuntimeFactory({ model: new ArrayBuffer(2), tokenizer: new TextEncoder().encode(tokenizer).buffer });
  const vectors = await runtime.embed(["hello world"], { model: new ArrayBuffer(2), tokenizer: new TextEncoder().encode(tokenizer).buffer });
  assert.equal(vectors.length, 1);
  assert.equal(vectors[0].length, 2);
  assert.ok(Math.abs(Math.hypot(...vectors[0]) - 1) < 1e-6);
});

test("default factory wires renderer-safe ORT blob module and wasm binary overrides", async () => {
  const { ort: __ort, configureLocalOrtAssets, defaultLocalRuntimeFactory } = await loadEmbedding();
  const wasmBinary = new ArrayBuffer(8);
  configureLocalOrtAssets("file:///vault/.obsidian/plugins/atomic-clusters/", { mjs: "blob:atomic-ort-module", wasmBinary });
  const tokenizer = JSON.stringify({ model: { type: "Unigram", unk_id: 3, vocab: [["▁", -0.1], ["hello", -0.2], ["<unk>", -5]] }, added_tokens: [{ id: 0, content: "<s>" }, { id: 1, content: "<pad>" }, { id: 2, content: "</s>" }] });
  await defaultLocalRuntimeFactory({ model: new ArrayBuffer(2), tokenizer: new TextEncoder().encode(tokenizer).buffer });
  assert.deepEqual(__ort.env.wasm.wasmPaths, { mjs: "blob:atomic-ort-module" });
  assert.equal(__ort.env.wasm.wasmBinary, wasmBinary);
});

test("ORT blob override is revoked when the plugin unloads", async () => {
  const { configureLocalOrtAssets, disposeLocalOrtAssets } = await loadEmbedding();
  let revoked = 0;
  configureLocalOrtAssets("file:///vault/.obsidian/plugins/atomic-clusters/", { mjs: "blob:atomic-ort-module", revoke: () => revoked++ });
  disposeLocalOrtAssets();
  assert.equal(revoked, 1);
});

test("ORT assets resolve from a Windows vault path with spaces and manifest.dir", async () => {
  const { resolveLocalOrtAssetPrefix } = await loadOrtAssets();
  const prefix = resolveLocalOrtAssetPrefix("C:\\Users\\Alice\\Vault With Spaces", ".obsidian\\plugins\\atomic-clusters", "atomic-clusters");
  assert.equal(prefix, "file:///C:/Users/Alice/Vault%20With%20Spaces/.obsidian/plugins/atomic-clusters/");
  const fallback = resolveLocalOrtAssetPrefix("C:\\Users\\Alice\\Vault With Spaces", "atomic-clusters", "atomic-clusters");
  assert.equal(fallback, prefix);
});

test("pinned ORT renderer transform disables both Electron Node-detection branches", async () => {
  const { prepareLocalOrtRendererModule } = await loadOrtAssets();
  const source = 'var B="object"==typeof process&&"object"==typeof process.versions&&"string"==typeof process.versions.node,C=0;if(B){const {createRequire:a}=await import("module");var D=require("worker_threads");}if(B){var fs=require("fs");}if(C){if(B){var port=require("worker_threads");}}var isNode = typeof globalThis.process?.versions?.node == \'string\';if (isNode) isPthread = (await import(\'worker_threads\')).workerData === \'em-pthread\';var wasm=(new URL("ort-wasm-simd-threaded.wasm",import.meta.url)).href;';
  const transformed = prepareLocalOrtRendererModule(source);
  assert.match(transformed, /B=false/);
  assert.equal((transformed.match(/if\(false\)\{/g) || []).length, 3);
  assert.match(transformed, /var isNode = false/);
  assert.match(transformed, /if \(false\) isPthread/);
  assert.match(transformed, /data:application\/wasm;base64/);
  assert.doesNotMatch(transformed, /new URL\("ort-wasm-simd-threaded\.wasm",import\.meta\.url\)/);
  assert.doesNotMatch(transformed, /if\(B\)\{|if \(isNode\)/);
});

test("ORT renderer transform rejects an unexpected asset format", async () => {
  const { prepareLocalOrtRendererModule } = await loadOrtAssets();
  assert.throws(() => prepareLocalOrtRendererModule("not an ORT module"), /Unsupported ORT/);
});

test("desktop bundle resolves ORT assets from vault instead of eval-loader __dirname", async () => {
  const { configureLocalOrtAssets, getLocalOrtAssetPrefix } = await loadEmbedding();
  configureLocalOrtAssets("file:///vault/.obsidian/plugins/atomic-clusters");
  assert.equal(getLocalOrtAssetPrefix(), "file:///vault/.obsidian/plugins/atomic-clusters/");
  const main = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
  assert.doesNotMatch(main, /__dirname/);
  assert.match(main, /this\.manifest\.dir/);
  assert.match(main, /getBasePath/);
  assert.match(main, /configureLocalOrtAssets/);
});
