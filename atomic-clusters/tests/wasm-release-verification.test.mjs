import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { verifyWasmAsset, REQUIRED_EXPORTS } from "../scripts/verify-wasm.mjs";

test("release verifier compiles the wasm asset, checks exports, and runs smoke probes", async () => {
  const result = await verifyWasmAsset(
    new URL("../wasm-core/pkg/atomic_clusters_wasm_core.js", import.meta.url).pathname,
    new URL("../wasm-core/pkg/atomic_clusters_wasm_core_bg.wasm", import.meta.url).pathname
  );
  assert.ok(result.bytes > 1000);
  assert.deepEqual(result.exports, REQUIRED_EXPORTS);
});

test("release build invokes the practical verifier after checking asset presence", async () => {
  const source = await readFile(new URL("../build.mjs", import.meta.url), "utf8");
  assert.match(source, /verifyWasmAsset\(gluePath, wasmPath\)/);
  assert.match(source, /requireWasm/);
  assert.match(source, /ort-wasm-simd-threaded\.jsep\.mjs/);
  assert.match(source, /ort-wasm-simd-threaded\.jsep\.wasm/);
  assert.match(source, /onnxruntime-webgpu-renderer-safe/);
  assert.match(source, /ort\.webgpu\.mjs/);
});
