#!/usr/bin/env node
/** Build a self-contained, offline demo vault with a populated plugin database. */
import { build, transform } from "esbuild";
import initSqlJs from "sql.js";
import { cp, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const pluginDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const vaultDir = resolve(process.argv[2] || "/tmp/atomic-clusters-demo-vault");
const noteCount = 240;
const dimensions = 48;
const clusterCount = 6;
const provider = "synthetic";
const model = "atomic-clusters-demo-v1";

function rng(seed = 20260904) { let state = seed >>> 0; return () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32); }
function normal(random) { const a = Math.max(random(), 1e-12); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * random()); }
function unit(values) { const length = Math.hypot(...values) || 1; return values.map((value) => value / length); }
function slug(value) { return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""); }

function synthesize() {
  const random = rng();
  const topics = ["Architecture", "Research", "Planning", "Operations", "Learning", "Writing"];
  const centers = topics.map((_, topic) => unit(Array.from({ length: dimensions }, (_, index) => (index === topic ? 1 : normal(random) * 0.12))));
  return Array.from({ length: noteCount }, (_, index) => {
    const isOutlier = index >= noteCount - 24;
    const topic = topics[index % topics.length];
    const vector = isOutlier
      ? unit(Array.from({ length: dimensions }, () => normal(random)))
      : unit(centers[index % topics.length].map((value) => value + normal(random) * 0.055));
    const path = `Demo/${isOutlier ? "Unsorted" : topic}/${String(index + 1).padStart(3, "0")}-${slug(topic)}.md`;
    const content = `# ${isOutlier ? "Unsorted" : topic} note ${index + 1}\n\nThis is synthetic offline demonstration content for the ${topic} cluster.\n`;
    return { path, title: path.split("/").at(-1).replace(/\.md$/, ""), content, mtime: 1767571200000 + index, hash: `demo-${index}-${topic}`, vector };
  });
}

async function loadModules() {
  const source = `import { clusterEmbeddings } from ${JSON.stringify(resolve(pluginDir, "src/clustering.ts"))};\nimport { WasmNumericKernel } from ${JSON.stringify(resolve(pluginDir, "src/wasm-kernel.ts"))};\nexport { clusterEmbeddings, WasmNumericKernel };`;
  const bundled = await build({ stdin: { contents: source, resolveDir: pluginDir, loader: "ts" }, bundle: true, format: "esm", platform: "node", target: "node20", write: false, logLevel: "silent" });
  const clustering = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString("base64")}`);
  let storageSource = await readFile(join(pluginDir, "src/sqlite-storage.ts"), "utf8");
  storageSource = storageSource.replace('import { contentHash } from "./hash";', 'async function contentHash(value) { let h = 2166136261; for (let i = 0; i < value.length; i++) { h ^= value.charCodeAt(i); h = Math.imul(h, 16777619); } return `fnv1a-${(h >>> 0).toString(16)}`; }');
  storageSource = storageSource.replace(/import \{[\s\S]*?\} from "\.\/types";/, "");
  const transformed = await transform(storageSource, { loader: "ts", format: "esm", target: "es2020" });
  const storage = await import(`data:text/javascript;base64,${Buffer.from(transformed.code).toString("base64")}`);
  return { ...clustering, ...storage };
}

async function loadWasm(WasmNumericKernel) {
  const glue = join(pluginDir, "wasm-core/pkg/atomic_clusters_wasm_core.js");
  const binary = join(pluginDir, "wasm-core/pkg/atomic_clusters_wasm_core_bg.wasm");
  if (!existsSync(glue) || !existsSync(binary)) return undefined;
  const wasm = await import(pathToFileURL(glue).href);
  wasm.initSync({ module: new WebAssembly.Module(await readFile(binary)) });
  const names = ["normalize", "matmul", "pca", "randomized_pca", "cosine_distances", "exact_knn", "exact_knn_cosine_tiled", "euclidean_mutual_reachability_mst", "mst", "mutual_reachability_mst", "hdbscan_extract", "hdbscan_extract_with_rows", "HnswIndex"];
  return new WasmNumericKernel(Object.fromEntries(names.filter((name) => typeof wasm[name] === "function").map((name) => [name, wasm[name]])));
}

class VaultAdapter {
  constructor(root) { this.root = root; }
  file(path) { return join(this.root, path); }
  async exists(path) { return existsSync(this.file(path)); }
  async readBinary(path) { const bytes = await readFile(this.file(path)); return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength); }
  async writeBinary(path, data) { const output = this.file(path); await mkdir(dirname(output), { recursive: true }); await writeFile(output, new Uint8Array(data)); }
  async mkdir(path) { await mkdir(this.file(path), { recursive: true }); }
  async rename(from, to) { const target = this.file(to); await mkdir(dirname(target), { recursive: true }); await rename(this.file(from), target); }
  async remove(path) { await rm(this.file(path), { force: true }); }
}

async function main() {
  await rm(vaultDir, { recursive: true, force: true });
  await mkdir(vaultDir, { recursive: true });
  const notes = synthesize();
  await Promise.all(notes.map(async (note) => { const output = join(vaultDir, note.path); await mkdir(dirname(output), { recursive: true }); await writeFile(output, note.content); }));
  await mkdir(join(vaultDir, ".obsidian/plugins/atomic-clusters"), { recursive: true });
  for (const asset of ["main.js", "manifest.json", "styles.css", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm", "sql-wasm.wasm"]) await cp(join(pluginDir, "dist", asset), join(vaultDir, ".obsidian/plugins/atomic-clusters", asset));
  await writeFile(join(vaultDir, ".obsidian/plugins/atomic-clusters/data.json"), JSON.stringify({ embeddingProvider: provider, geminiModel: model, localModel: model, automaticRefresh: false, clusterTitlesEnabled: true }, null, 2));
  const modules = await loadModules();
  const kernel = await loadWasm(modules.WasmNumericKernel);
  const result = await modules.clusterEmbeddings(notes.map((note) => note.path), notes.map((note) => note.vector), { seed: 42, pcaSampleSize: noteCount, pcaMaxComponents: 32, umapComponents: 10, umapNeighbors: 12, minClusterSize: 10, minSamples: 5 }, { kernel });
  result.embeddingProvider = provider; result.embeddingModel = model;
  const SQL = await initSqlJs({ locateFile: (file) => join(pluginDir, "node_modules/sql.js/dist", file) });
  const store = await new modules.SqliteClusterStore(new VaultAdapter(vaultDir), SQL).open();
  await store.upsertNotes(notes);
  await store.putEmbeddings(notes.map((note) => ({ path: note.path, hash: note.hash, provider, model, vector: note.vector })));
  await store.savePcaModel(result.pca.model);
  await store.projectMany(notes.map((note) => ({ path: note.path, vector: note.vector })), result.pca.model);
  const resultId = await store.saveResult(result, { noteHashes: new Map(notes.map((note) => [note.path, note.hash])) });
  await store.saveEmbeddingLog({ version: 1, startedAt: new Date().toISOString(), completedAt: new Date().toISOString(), provider, model, total: notes.length, succeeded: notes.length, failed: 0, cached: 0, entries: [], status: "completed", stage: "clustering" });
  const counts = store.query("SELECT (SELECT COUNT(*) FROM notes) AS notes, (SELECT COUNT(*) FROM embeddings) AS embeddings, (SELECT COUNT(*) FROM assignments WHERE result_id=?) AS assignments", [resultId])[0];
  store.close();
  await writeFile(join(vaultDir, "README.md"), "# Atomic Clusters demo vault\n\nThis vault contains 240 synthetic notes, their synthetic embeddings, and a completed Atomic Clusters result. Open it as an Obsidian vault and run **Open cluster explorer**. No API key or source note data is included.\n");
  console.log(JSON.stringify({ vaultDir, resultId, counts, clusters: result.hierarchy.leaves.length, wasm: !!kernel }, null, 2));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
