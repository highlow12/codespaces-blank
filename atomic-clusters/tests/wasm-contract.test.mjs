import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadContract() {
  const source = await readFile(new URL("../src/engine/wasm-contract.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

function mockBindings(log) {
  return {
    matmul(a, b, m, k, n) {
      log.push(["matmul", a, b, m, k, n]);
      return new Float32Array([4, 4, 10, 8]);
    },
    pca(rows, rowCount, dimension, components) {
      log.push(["pca", rows, rowCount, dimension, components]);
      return { projected: new Float32Array(rowCount * components), explained: new Float32Array([3, 1].slice(0, components)) };
    },
    cosine_distances(rows, rowCount, dimension, tile) {
      log.push(["cosine_distances", rows, rowCount, dimension, tile]);
      return new Float32Array(rowCount * rowCount);
    },
    exact_knn(distances, rowCount, k) {
      log.push(["exact_knn", distances, rowCount, k]);
      return new Uint32Array(rowCount * k).fill(1);
    },
    mst(distances, rowCount) {
      log.push(["mst", distances, rowCount]);
      return new Float32Array([0, 1, 0.25]);
    },
    HnswIndex: class {
      constructor(points, count, dimension, m, seed) {
        log.push(["hnsw_build", points, count, dimension, m, seed]);
      }
      search(query, k) {
        log.push(["hnsw_query", query, k]);
        return new Uint32Array([1, 0].slice(0, k));
      }
      free() {
        log.push(["hnsw_free"]);
      }
    }
  };
}

test("WASM wrapper passes flat typed arrays and preserves output shapes", async () => {
  const { WasmNumericKernel, matrix } = await loadContract();
  const log = [];
  const kernel = new WasmNumericKernel(mockBindings(log), { cosineTile: 64 });
  const rows = matrix(new Float32Array([1, 2, 3, 4, 5, 6]), 2, 3);
  const rhs = matrix([1, 0, 0, 1, 1, 1], 3, 2);

  const product = kernel.matmul(rows, rhs);
  assert.deepEqual([...product.data], [4, 4, 10, 8]);
  assert.deepEqual([product.rows, product.cols], [2, 2]);
  const pca = kernel.pca(rows, 2);
  assert.deepEqual([pca.projected.rows, pca.projected.cols], [2, 2]);
  const distances = kernel.cosineDistances(rows);
  assert.deepEqual([distances.rows, distances.cols], [2, 2]);
  assert.equal(kernel.exactKnn(distances, 1).length, 2);
  assert.equal(kernel.mst(distances).length, 3);

  const index = kernel.hnswBuild(rows);
  assert.deepEqual([...index.query(new Float32Array([1, 2, 3]), 2)], [1, 0]);
  index.free();
  assert.equal(log.every((entry) => entry[1] === undefined || entry[1] instanceof Float32Array), true);
  assert.equal(log.find((entry) => entry[0] === "cosine_distances")[4], 64);
  assert.deepEqual(log.find((entry) => entry[0] === "hnsw_build").slice(4), [16, 42]);
});

test("WASM wrapper uses configured deterministic randomized PCA when available", async () => {
  const { WasmNumericKernel, matrix } = await loadContract();
  const calls = [];
  const bindings = {
    ...mockBindings(calls),
    randomized_pca(rows, count, dimension, components, oversamples, powerIterations, seed) {
      calls.push(["randomized_pca", rows, count, dimension, components, oversamples, powerIterations, seed]);
      return { projected: new Float32Array(count * components), explained: new Float32Array(components) };
    }
  };
  const kernel = new WasmNumericKernel(bindings, { pcaOversamples: 16, pcaPowerIterations: 3, pcaSeed: 42 });
  kernel.pca(matrix([1, 0, 0, 1], 2, 2), 2);
  assert.deepEqual(calls.find((entry) => entry[0] === "randomized_pca").slice(4), [2, 16, 3, 42]);
  assert.equal(calls.some((entry) => entry[0] === "pca"), false);
});

test("function-style HNSW exports are supported", async () => {
  const { WasmNumericKernel, matrix } = await loadContract();
  const calls = [];
  const bindings = {
    matmul: () => [], pca: () => ({ projected: [], explained: [] }), cosine_distances: () => [], exact_knn: () => [], mst: () => [],
    hnsw_build(points, count, dimension) { calls.push(["build", points, count, dimension]); return 7; },
    hnsw_query(handle, query, k) { calls.push(["query", handle, query, k]); return new Uint32Array([0]); },
    hnsw_free(handle) { calls.push(["free", handle]); }
  };
  const index = new WasmNumericKernel(bindings).hnswBuild(matrix([1, 0, 0, 1], 2, 2));
  assert.deepEqual([...index.query([1, 0], 1)], [0]);
  index.free();
  assert.deepEqual(calls.map(([name]) => name), ["build", "query", "free"]);
});

test("wrapper rejects malformed, non-finite, and incompatible buffers", async () => {
  const { WasmNumericKernel, WasmContractError, matrix } = await loadContract();
  const log = [];
  const kernel = new WasmNumericKernel(mockBindings(log));
  assert.throws(() => matrix([1, 2, 3], 2, 2), WasmContractError);
  assert.throws(() => matrix([1, Number.NaN], 1, 2), WasmContractError);
  const a = matrix([1, 0, 0, 1], 2, 2);
  assert.throws(() => kernel.matmul(a, matrix([1, 2, 3], 3, 1)), WasmContractError);
  assert.throws(() => kernel.exactKnn(matrix([0, 1, 1], 1, 3), 1), WasmContractError);
  assert.throws(() => kernel.hnswBuild(a).query([1], 1), WasmContractError);
});
