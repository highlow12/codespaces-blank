import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const buildSource = await readFile(new URL("../build.mjs", import.meta.url), "utf8");
const offlineSource = await readFile(new URL("../scripts/offline-e2e.mjs", import.meta.url), "utf8");

// Keep the two bootstrap paths in lockstep with WasmKernelModule. A missing
// export silently forces the worker onto a slower fallback or changes results.
const requiredExports = [
  "normalize", "matmul", "pca", "randomized_pca", "cosine_distances",
  "exact_knn", "exact_knn_cosine_tiled", "euclidean_mutual_reachability_mst",
  "mst", "mutual_reachability_mst", "hdbscan_extract", "hdbscan_extract_with_rows", "HnswIndex"
];

test("WASM bootstrap paths bind every WasmKernelModule export", () => {
  for (const name of requiredExports) {
    assert.match(buildSource, new RegExp(`\\b${name}\\b`), `build bootstrap omits ${name}`);
    assert.match(offlineSource, new RegExp(`\\b${name}\\b`), `offline runner omits ${name}`);
  }
});
