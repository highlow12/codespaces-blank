import { build, context } from "esbuild";
import { cp, mkdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

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
  const workerBuild = await build({ ...common, entryPoints: ["src/worker.ts"], platform: "node", plugins: [wasmBootstrap], write: false });
  const workerSource = new TextDecoder().decode(workerBuild.outputFiles[0].contents);
  const workerPlugin = { name: "embedded-worker", setup(plugin) {
    plugin.onResolve({ filter: /^\.\/worker-source$/ }, () => ({ path: "atomic-clusters-worker-source", namespace: "embedded-worker" }));
    plugin.onLoad({ filter: /.*/, namespace: "embedded-worker" }, () => ({ contents: `export default ${JSON.stringify(workerSource)};`, loader: "js" }));
  } };
  const mainBuild = { ...common, plugins: [workerPlugin], entryPoints: ["src/main.ts"], outfile: "dist/main.js" };
  if (process.argv.includes("--watch")) {
    const buildContext = await context(mainBuild);
    await buildContext.watch();
    console.log("watching");
    return;
  }
  await build(mainBuild);
  await cp("manifest.json", "dist/manifest.json");
  await cp("styles.css", "dist/styles.css");
}
run().catch((error) => { console.error(error); process.exitCode = 1; });
