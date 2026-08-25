import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export const REQUIRED_EXPORTS = [
  "normalize", "matmul", "pca", "randomized_pca", "cosine_distances",
  "exact_knn", "exact_knn_cosine_tiled", "euclidean_mutual_reachability_mst",
  "mst", "mutual_reachability_mst", "hdbscan_extract", "HnswIndex"
];

function fail(message) { throw new Error(`WASM asset verification failed: ${message}`); }

/** Validate generated wasm-bindgen glue and execute deterministic smoke probes. */
export async function verifyWasmAsset(gluePath, wasmPath) {
  const bytes = await readFile(wasmPath);
  if (bytes.length < 8 || bytes[0] !== 0 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6d || bytes[4] !== 1 || bytes[5] !== 0 || bytes[6] !== 0 || bytes[7] !== 0) fail("asset is not a WebAssembly MVP module");
  let module;
  try { module = new WebAssembly.Module(bytes); } catch (error) { fail(`module compilation failed: ${error instanceof Error ? error.message : String(error)}`); }
  let glue;
  try { glue = await import(pathToFileURL(gluePath).href); } catch (error) { fail(`wasm-bindgen glue import failed: ${error instanceof Error ? error.message : String(error)}`); }
  for (const name of REQUIRED_EXPORTS) if (typeof glue[name] !== "function") fail(`missing generated export ${name}`);
  if (typeof glue.initSync !== "function") fail("missing initSync export");
  try { glue.initSync({ module }); } catch (error) { fail(`wasm-bindgen initialization failed: ${error instanceof Error ? error.message : String(error)}`); }

  const normalized = Array.from(glue.normalize([3, 4], 1, 2));
  if (normalized.length !== 2 || Math.abs(normalized[0] - 0.6) > 1e-5 || Math.abs(normalized[1] - 0.8) > 1e-5) fail("normalize smoke probe returned an unexpected result");
  const distances = Array.from(glue.cosine_distances([1, 0, 0, 1], 2, 2, 1));
  if (distances.length !== 4 || distances.some((value, index) => !Number.isFinite(value) || (index === 0 || index === 3 ? Math.abs(value) > 1e-6 : Math.abs(value - 1) > 1e-6))) fail("cosine distance smoke probe returned an unexpected result");
  const pca = glue.pca([1, 0, 0, 1], 2, 2, 1);
  if (!pca || pca.projected?.length !== 2 || pca.explained?.length !== 1) fail("PCA smoke probe returned an unexpected shape");
  const index = new glue.HnswIndex([1, 0, 0, 1], 2, 2, 2, 42);
  if (!index || typeof index.search !== "function" || index.search([1, 0], 1).length !== 1) fail("HNSW smoke probe failed");
  index.free?.();
  return { bytes: bytes.length, exports: REQUIRED_EXPORTS.slice() };
}

if (import.meta.url === pathToFileURL(process.argv[1] || "").href) {
  const [gluePath, wasmPath] = process.argv.slice(2);
  if (!gluePath || !wasmPath) { console.error("Usage: node scripts/verify-wasm.mjs GLUE.js CORE.wasm"); process.exitCode = 2; }
  else verifyWasmAsset(gluePath, wasmPath).then((result) => console.log(JSON.stringify(result))).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
