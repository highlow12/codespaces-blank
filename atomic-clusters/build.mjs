import { build, context } from "esbuild";
import { cp, mkdir, readFile, readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { verifyWasmAsset } from "./scripts/verify-wasm.mjs";

const common = {
  bundle: true,
  platform: "browser",
  target: "es2020",
  format: "cjs",
  sourcemap: false,
  external: ["obsidian", "electron", "worker_threads", "node:*"],
  logLevel: "info"
};

async function run() {
  await mkdir("dist", { recursive: true });
  const generatedDir = resolve("wasm-core/pkg");
  const gluePath = resolve(generatedDir, "atomic_clusters_wasm_core.js");
  const wasmPath = resolve(generatedDir, "atomic_clusters_wasm_core_bg.wasm");
  const requireWasm = process.argv.includes("--require-wasm");
  if (requireWasm && (!existsSync(gluePath) || !existsSync(wasmPath))) {
    throw new Error("Release build requires wasm-core/pkg. Run npm run build:wasm first.");
  }
  if (requireWasm) await verifyWasmAsset(gluePath, wasmPath);
  const wasmBootstrap = {
    name: "atomic-clusters-wasm-bootstrap",
    setup(plugin) {
      plugin.onResolve({ filter: /^atomic-clusters-wasm-bootstrap$/ }, () => ({ path: "bootstrap", namespace: "atomic-wasm" }));
      plugin.onLoad({ filter: /.*/, namespace: "atomic-wasm" }, async () => {
        if (!existsSync(gluePath) || !existsSync(wasmPath)) return { contents: "// Development build: deterministic JS fallback remains enabled." };
        const encoded = (await readFile(wasmPath)).toString("base64");
        return { resolveDir: generatedDir, contents: `
          import { initSync, normalize, matmul, pca, randomized_pca,
            cosine_distances, exact_knn, exact_knn_cosine_tiled,
            euclidean_mutual_reachability_mst, mst,
            mutual_reachability_mst, hdbscan_extract, HnswIndex } from ${JSON.stringify(gluePath)};
          const bytes = Uint8Array.from(Buffer.from(${JSON.stringify(encoded)}, "base64"));
          initSync({ module: new WebAssembly.Module(bytes) });
          globalThis.__ATOMIC_CLUSTERS_WASM__ = { normalize, matmul, pca,
            randomized_pca, cosine_distances, exact_knn,
            exact_knn_cosine_tiled, euclidean_mutual_reachability_mst,
            mst, mutual_reachability_mst,
            hdbscan_extract, HnswIndex };
        ` };
      });
    }
  };
  const pyodideCoreSource = {};
  async function collectPythonSources(directory, prefix = "") {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) await collectPythonSources(path, relative);
      else if (entry.name.endsWith(".py")) pyodideCoreSource[relative] = await readFile(path, "utf8");
    }
  }
  await collectPythonSources(resolve("..", "pyodide_core"));
  const pyodideCorePlugin = {
    name: "embedded-pyodide-core",
    setup(plugin) {
      plugin.onResolve({ filter: /^\.\/pyodide-core-source$/ }, () => ({ path: "pyodide-core-source", namespace: "embedded-pyodide-core" }));
      plugin.onLoad({ filter: /.*/, namespace: "embedded-pyodide-core" }, () => ({ contents: `export const PYODIDE_CORE_SOURCE = ${JSON.stringify(pyodideCoreSource)};`, loader: "js" }));
    }
  };
  const workerBuild = await build({ ...common, entryPoints: ["src/worker.ts"], platform: "node", plugins: [wasmBootstrap], write: false });
  const workerSource = new TextDecoder().decode(workerBuild.outputFiles[0].contents);
  const browserWorkerBuild = await build({ ...common, format: "iife", platform: "browser", entryPoints: ["src/browser-worker.ts"], plugins: [wasmBootstrap], write: false });
  const browserWorkerSource = new TextDecoder().decode(browserWorkerBuild.outputFiles[0].contents);
  // Transformers.js bundles its own (currently 1.22.x) ORT Web dependency.
  // Point the title worker's explicit runtime import at that exact copy so
  // its env object is the one selected by the lazy Transformers backend.
  const transformersOrtWeb = resolve("node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist/ort.webgpu.mjs");
  const titleOrtAliasPlugin = { name: "transformers-title-onnxruntime-web", setup(plugin) {
    plugin.onResolve({ filter: /^atomic-clusters-title-onnxruntime-web$/ }, () => ({ path: transformersOrtWeb }));
  } };
  const titleWorkerBuild = await build({ ...common, format: "iife", platform: "browser", entryPoints: ["src/title-worker.ts"], plugins: [titleOrtAliasPlugin], write: false });
  const titleWorkerSource = new TextDecoder().decode(titleWorkerBuild.outputFiles[0].contents);
  const pyodideWorkerBuild = await build({ ...common, format: "iife", platform: "browser", entryPoints: ["src/pyodide-worker.ts"], plugins: [wasmBootstrap, pyodideCorePlugin], write: false });
  const pyodideWorkerSource = new TextDecoder().decode(pyodideWorkerBuild.outputFiles[0].contents);
  const workerPlugin = { name: "embedded-worker", setup(plugin) {
    plugin.onResolve({ filter: /^\.\/worker-source$/ }, () => ({ path: "atomic-clusters-worker-source", namespace: "embedded-worker" }));
    plugin.onLoad({ filter: /.*/, namespace: "embedded-worker" }, () => ({ contents: `export default ${JSON.stringify(workerSource)};`, loader: "js" }));
  } };
  const pyodideWorkerPlugin = { name: "embedded-pyodide-worker", setup(plugin) {
    plugin.onResolve({ filter: /^\.\/pyodide-worker-source$/ }, () => ({ path: "pyodide-worker-source", namespace: "embedded-pyodide-worker" }));
    plugin.onLoad({ filter: /.*/, namespace: "embedded-pyodide-worker" }, () => ({ contents: `export default ${JSON.stringify(pyodideWorkerSource)};`, loader: "js" }));
  } };
  const browserWorkerPlugin = { name: "embedded-browser-worker", setup(plugin) {
    plugin.onResolve({ filter: /^\.\/browser-worker-source$/ }, () => ({ path: "atomic-clusters-browser-worker-source", namespace: "embedded-browser-worker" }));
    plugin.onLoad({ filter: /.*/, namespace: "embedded-browser-worker" }, () => ({ contents: `export default ${JSON.stringify(browserWorkerSource)};`, loader: "js" }));
  } };
  const titleWorkerPlugin = { name: "embedded-title-worker", setup(plugin) {
    plugin.onResolve({ filter: /^\.\/title-worker-source$/ }, () => ({ path: "atomic-clusters-title-worker-source", namespace: "embedded-title-worker" }));
    plugin.onLoad({ filter: /.*/, namespace: "embedded-title-worker" }, () => ({ contents: `export default ${JSON.stringify(titleWorkerSource)};`, loader: "js" }));
  } };
  // The package export points at ort.webgpu.bundle.min.mjs, whose embedded
  // Emscripten factory evaluates new URL(..., import.meta.url) even when
  // wasmBinary is supplied. Use the non-bundle build so the renderer-safe
  // JSEP module is loaded through wasmPaths and transformed into a Blob.
  const ortWebGpuAliasPlugin = { name: "onnxruntime-webgpu-renderer-safe", setup(plugin) {
    plugin.onResolve({ filter: /^onnxruntime-web\/webgpu$/ }, () => ({ path: resolve("node_modules/onnxruntime-web/dist/ort.webgpu.mjs") }));
  } };
  const mainBuild = { ...common, plugins: [workerPlugin, pyodideWorkerPlugin, browserWorkerPlugin, titleWorkerPlugin, ortWebGpuAliasPlugin], entryPoints: ["src/main.ts"], outfile: "dist/main.js" };
  if (process.argv.includes("--watch")) {
    const buildContext = await context(mainBuild);
    await buildContext.watch();
    console.log("watching");
    return;
  }
  await build(mainBuild);
  await cp("manifest.json", "dist/manifest.json");
  await cp("styles.css", "dist/styles.css");
  // onnxruntime-web resolves its worker and binary next to the plugin bundle.
  // Ship both CPU WASM and the JSEP/WebGPU assets; inference remains local
  // after the model has been explicitly downloaded.
  const ortAssets = resolve("node_modules/onnxruntime-web/dist");
  await cp(resolve(ortAssets, "ort-wasm-simd-threaded.mjs"), "dist/ort-wasm-simd-threaded.mjs");
  await cp(resolve(ortAssets, "ort-wasm-simd-threaded.wasm"), "dist/ort-wasm-simd-threaded.wasm");
  await cp(resolve(ortAssets, "ort-wasm-simd-threaded.jsep.mjs"), "dist/ort-wasm-simd-threaded.jsep.mjs");
  await cp(resolve(ortAssets, "ort-wasm-simd-threaded.jsep.wasm"), "dist/ort-wasm-simd-threaded.jsep.wasm");
}
run().catch((error) => { console.error(error); process.exitCode = 1; });
