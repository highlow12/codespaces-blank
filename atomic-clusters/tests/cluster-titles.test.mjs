import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { build, transform } from "esbuild";
import { createHash } from "node:crypto";
import vm from "node:vm";

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
  assert.match(worker, /atomic-clusters-title-onnxruntime-web/);
  assert.match(worker, /Symbol\.for\("onnxruntime"\)/);
  assert.match(worker, /configureOnnxRuntime\(event\.data\.ortWasm\)/);
  assert.match(worker, /allowRemoteModels\s*=\s*false/);
  assert.match(worker, /localModelPath/);
  assert.match(worker, /device:\s*"webgpu"/);
  assert.match(worker, /dtype:\s*"q4f16"/);
  assert.match(worker, /wasmPaths\s*=\s*"https:\/\/atomic-clusters\.local\/ort\//);
  assert.match(worker, /wasmBinary\s*=\s*event\.data\.ortWasm/);
  assert.match(worker, /ort-wasm-simd-threaded\.jsep\.wasm/);
  assert.match(worker, /blocked non-local asset request/);
  assert.match(worker, /input instanceof Request \? input\.url/);
  // Transformers.js fills env.backends.onnx only when its backend module is
  // first loaded; checking it before pipeline() regresses to a false failure.
  assert.doesNotMatch(worker, /const onnxWasm = env\.backends\?\.onnx\?\.wasm/);
});

test("title build aliases the explicit ORT import to Transformers.js' pinned runtime", async () => {
  const build = await readFile(new URL("../build.mjs", import.meta.url), "utf8");
  assert.match(build, /transformersOrtWeb/);
  assert.match(build, /onnxruntime-web\/dist\/ort\.webgpu\.mjs/);
  assert.match(build, /titleOrtAliasPlugin/);
});

test("built title worker hides Electron process before Transformers.js evaluates", async () => {
  const fakeRuntimePlugin = {
    name: "fake-title-runtime",
    setup(plugin) {
      plugin.onResolve({ filter: /^atomic-clusters-title-onnxruntime-web$/ }, () => ({ path: "fake-ort", namespace: "fake-title-runtime" }));
      plugin.onLoad({ filter: /.*/, namespace: "fake-title-runtime" }, () => ({
        loader: "js",
        contents: `export const env = { wasm: {} }; export const InferenceSession = {};`
      }));
      plugin.onResolve({ filter: /^@huggingface\/transformers$/ }, () => ({ path: "fake-transformers", namespace: "fake-title-transformers" }));
      plugin.onLoad({ filter: /.*/, namespace: "fake-title-transformers" }, () => ({
        loader: "js",
        contents: `
          const hasRuntimeOverride = Symbol.for("onnxruntime") in globalThis;
          const detectedDevices = !hasRuntimeOverride && typeof process === "undefined" && typeof navigator !== "undefined" && "gpu" in navigator ? ["webgpu", "wasm"] : ["dml", "cpu"];
          globalThis.__titleDetectedDevices = detectedDevices;
          export const env = { backends: { onnx: { wasm: {} } } };
          export async function pipeline(_task, _model, options) {
            if (!detectedDevices.includes(options.device)) throw new Error("Unsupported device: \\"" + options.device + "\\"");
            globalThis.__titlePipelineDevice = options.device;
            return async () => {
              // ORT initializes lazily. This runs after the dynamic import has
              // completed and catches a bootstrap that restores Electron's
              // process too early.
              if (typeof process !== "undefined") throw new Error("process restored before delayed title generation");
              return [{ generated_text: "WebGPU title" }];
            };
          }
        `
      }));
    }
  };
  const result = await build({
    bundle: true,
    platform: "browser",
    target: "es2020",
    format: "iife",
    entryPoints: [new URL("../src/title-worker-bootstrap.ts", import.meta.url).pathname],
    plugins: [fakeRuntimePlugin],
    write: false
  });
  const source = new TextDecoder().decode(result.outputFiles[0].contents);
  const context = vm.createContext({
    process: { platform: "win32", release: { name: "electron" } },
    navigator: { gpu: {} },
    postMessage: (message) => posted.push(message),
    console,
    setTimeout,
    clearTimeout
  });
  const posted = [];
  vm.runInContext(source, context);
  for (let attempt = 0; attempt < 20 && typeof context.onmessage !== "function"; attempt++) await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(typeof context.onmessage, "function");
  const buffer = () => vm.runInContext("new ArrayBuffer(1)", context);
  await context.onmessage({ data: { type: "INIT", model: buffer(), tokenizer: buffer(), config: buffer(), generationConfig: buffer(), tokenizerConfig: buffer(), ortWasm: buffer() } });
  assert.equal(Array.from(context.__titleDetectedDevices).join(","), "webgpu,wasm");
  assert.equal(context.__titlePipelineDevice, "webgpu");
  assert.deepEqual(posted.map((message) => ({ type: message.type, backend: message.backend })), [{ type: "READY", backend: "webgpu" }]);
  assert.equal(vm.runInContext("typeof process", context), "undefined");
  await context.onmessage({ data: { type: "GENERATE", id: 1, prompts: ["Name this cluster"], maxNewTokens: 12 } });
  assert.equal(posted.at(-1).type, "RESULT");
  assert.equal(posted.at(-1).id, 1);
  assert.deepEqual(Array.from(posted.at(-1).values), ["WebGPU title"]);
  assert.equal(vm.runInContext("typeof process", context), "undefined");
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

test("forced title regeneration bypasses cache reads but refreshes cache entries", async () => {
  const { LocalClusterTitleGenerator, TitleModelManager, TITLE_MODEL_DESCRIPTOR } = await loadTitle();
  const storage = new MemoryStorage();
  const manager = new TitleModelManager(storage, { ...TITLE_MODEL_DESCRIPTOR, modelSha256: createHash("sha256").update("asset").digest("hex") });
  await manager.downloadModel(async () => true);
  const result = { schemaVersion: 2, ids: ["a.md"], leafLabels: [0], probabilities: [1], outlierProxy: [0], hierarchy: { leaves: [0], merges: [], root: 0 }, pca: {}, timings: {} };
  const notes = [{ path: "a.md", title: "A", content: "body", hash: "a" }];
  let runtimeCalls = 0;
  const runtime = async () => ({ generate: async () => { runtimeCalls++; return ["Fresh title"]; }, diagnostics: { backend: "webgpu" } });
  const cache = { get: () => ({ key: "same", title: "Cached title", nodeMembersFingerprint: "x", savedAt: "" }), set: () => {} };
  const generator = new LocalClusterTitleGenerator(manager, runtime);
  const cached = await generator.generate(result, notes, { cache });
  assert.equal(cached.titles["0"], "Cached title");
  assert.equal(runtimeCalls, 0);
  const regenerated = await generator.generate(result, notes, { cache, forceRegenerate: true });
  assert.equal(regenerated.titles["0"], "Fresh title");
  assert.equal(regenerated.titleGeneration.statuses["0"], "generated");
  assert.equal(runtimeCalls, 1);
});
