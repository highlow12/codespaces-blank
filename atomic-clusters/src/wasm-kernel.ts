import type { HdbscanOutput, NumericKernel } from "./clustering";
import {
  ExactKnnCosineTiledWasmResult,
  HdbscanExtractWasmResult,
  MutualReachabilityMstWasmResult,
  NumericInput,
  WasmBindings,
  WasmNumericKernel as FlatWasmKernel,
  matrix
} from "./engine/wasm-contract";

export interface WasmKernelModule extends WasmBindings {
  normalize(rows: NumericInput, rowCount: number, dimension: number): NumericInput;
  exact_knn_cosine_tiled(rows: NumericInput, rowCount: number, dimension: number, k: number, tile: number): ExactKnnCosineTiledWasmResult;
  euclidean_mutual_reachability_mst(rows: NumericInput, rowCount: number, dimension: number, minSamples: number, tile: number): MutualReachabilityMstWasmResult;
  mutual_reachability_mst(indices: NumericInput, distances: NumericInput, rowCount: number, k: number, minSamples: number): MutualReachabilityMstWasmResult;
  hdbscan_extract(mstEdges: NumericInput, rowCount: number, minClusterSize: number, selectionMethod: number, allowSingleCluster: boolean): HdbscanExtractWasmResult;
}

/** Adapts the flat TypedArray WASM contract to the worker orchestration API. */
export class WasmNumericKernel implements NumericKernel {
  private readonly flat: FlatWasmKernel;

  constructor(private readonly wasm: WasmKernelModule) {
    // On the real production ordering, 16 oversamples and 3 power iterations
    // reproduce the Python full-PCA 96-dimensional preservation knee without
    // requiring a full SVD or an exhaustive 512-component pilot.
    this.flat = new FlatWasmKernel(wasm, {
      cosineTile: 256, hnswM: 16, hnswSeed: 42,
      pcaOversamples: 16, pcaPowerIterations: 3, pcaSeed: 42
    });
  }

  normalize(rows: number[][]): number[][] {
    const input = flatten(rows);
    return nested(Float32Array.from(this.wasm.normalize(input.data, input.rows, input.cols)), input.rows, input.cols);
  }

  pca(rows: number[][], components: number): { projected: number[][]; explained: number[] } {
    const result = this.flat.pca(flatten(rows), components);
    return {
      projected: nested(result.projected.data, result.projected.rows, result.projected.cols),
      explained: Array.from(result.explained)
    };
  }

  cosineDistances(rows: number[][]): number[][] {
    const result = this.flat.cosineDistances(flatten(rows));
    return nested(result.data, result.rows, result.cols);
  }

  exactKnn(rows: number[][], k: number): number[][] {
    const input = flatten(rows);
    const result = this.wasm.exact_knn_cosine_tiled(input.data, input.rows, input.cols, k, 256);
    return nestedIndices(Uint32Array.from(result.indices), result.rows, result.k);
  }

  mst(rows: number[][], k: number): Array<[number, number, number]> {
    const input = flatten(rows);
    const knn = this.wasm.exact_knn_cosine_tiled(input.data, input.rows, input.cols, k, 256);
    const result = this.wasm.mutual_reachability_mst(
      Uint32Array.from(knn.indices), Float32Array.from(knn.distances), input.rows, knn.k, k
    );
    const values = Float32Array.from(result.edges);
    const edges: Array<[number, number, number]> = [];
    for (let index = 0; index < values.length; index += 3) edges.push([values[index], values[index + 1], values[index + 2]]);
    return edges;
  }

  /** Production HDBSCAN path: exact Euclidean mutual-reachability MST -> WASM leaf extraction. */
  hdbscan(rows: number[][], minClusterSize: number, minSamples: number): HdbscanOutput {
    const input = flatten(rows);
    if (input.rows < minClusterSize) return { labels: new Array(input.rows).fill(-1), probabilities: new Array(input.rows).fill(0), outlierProxy: new Array(input.rows).fill(1) };
    const clusterSize = boundedInteger(minClusterSize, 2, input.rows, "minClusterSize");
    const samples = boundedInteger(minSamples, 1, input.rows - 1, "minSamples");
    // Keep all Euclidean pair work in WASM. The export computes the exact
    // minSamples core distances and a complete-graph Prim MST without an n²
    // JS allocation; cosine helpers remain available for non-HDBSCAN users.
    const mst = this.wasm.euclidean_mutual_reachability_mst(input.data, input.rows, input.cols, samples, 256);
    const output = this.wasm.hdbscan_extract(Float32Array.from(mst.edges), input.rows, clusterSize, 1, false);
    return hdbscanOutput(output, input.rows);
  }

  hnsw(rows: number[][]): ReturnType<FlatWasmKernel["hnswBuild"]> {
    return this.flat.hnswBuild(flatten(rows));
  }
}

function flatten(rows: number[][]) {
  if (!rows.length || !rows[0].length || rows.some((row) => row.length !== rows[0].length)) {
    throw new TypeError("WASM matrices must be non-empty and rectangular");
  }
  return matrix(Float32Array.from(rows.flat()), rows.length, rows[0].length);
}

function nested(values: Float32Array, rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, row) => Array.from(values.subarray(row * cols, (row + 1) * cols)));
}

function nestedIndices(values: Uint32Array, rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, row) => Array.from(values.subarray(row * cols, (row + 1) * cols)));
}

function boundedInteger(value: number, minimum: number, maximum: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new TypeError(`${name} must be an integer between ${minimum} and ${maximum}`);
  }
  return value;
}

function hdbscanOutput(output: HdbscanExtractWasmResult, count: number): HdbscanOutput {
  const labels = Array.from(output.labels);
  const probabilities = Array.from(output.probabilities);
  const outlierProxy = Array.from(output.outlier_scores);
  if (!Number.isSafeInteger(output.cluster_count) || output.cluster_count < 0 || labels.length !== count || probabilities.length !== count || outlierProxy.length !== count) {
    throw new TypeError("WASM HDBSCAN output has an invalid shape");
  }
  for (let index = 0; index < count; index++) {
    if (!Number.isInteger(labels[index]) || labels[index] < -1 || labels[index] >= output.cluster_count || !Number.isFinite(probabilities[index]) || probabilities[index] < 0 || probabilities[index] > 1 || !Number.isFinite(outlierProxy[index]) || outlierProxy[index] < 0 || outlierProxy[index] > 1) {
      throw new TypeError(`WASM HDBSCAN output is invalid at row ${index}`);
    }
  }
  const memberships = Array.from({ length: count }, (_, row) => {
    const values = new Array(output.cluster_count).fill(0);
    if (labels[row] >= 0) values[labels[row]] = probabilities[row];
    return values;
  });
  return { labels, probabilities, outlierProxy, memberships };
}

interface Edge { left: number; right: number; distance: number; }

/**
 * A sparse kNN mutual-reachability graph is normally connected. If it is not,
 * rebuild an exact complete-graph MST with deterministic Boruvka rounds. This
 * scans cosine pairs but never materializes an n² distance or edge matrix;
 * connected runs retain the O(nk) sparse path unchanged.
 */
function completeMutualReachabilityMst(
  rows: Float32Array, count: number, dimension: number, knn: ExactKnnCosineTiledWasmResult,
  sparse: MutualReachabilityMstWasmResult, minSamples: number
): Float32Array {
  const values = Float32Array.from(sparse.edges);
  if (values.length % 3 !== 0 || sparse.edge_count !== values.length / 3) throw new TypeError("WASM mutual-reachability MST has invalid triples");
  const uf = new UnionFind(count); const edges: Edge[] = [];
  for (let index = 0; index < values.length; index += 3) {
    const left = values[index]; const right = values[index + 1]; const distance = values[index + 2];
    if (!Number.isInteger(left) || !Number.isInteger(right) || left < 0 || right < 0 || left >= count || right >= count || left === right || !Number.isFinite(distance) || distance < 0) throw new TypeError("WASM mutual-reachability MST contains an invalid edge");
    if (uf.join(left, right)) edges.push({ left, right, distance });
  }
  if (edges.length === count - 1) return flattenEdges(edges);
  const k = knn.k;
  if (knn.rows !== count || k < minSamples || Float32Array.from(knn.distances).length !== count * k) throw new TypeError("WASM kNN output cannot provide HDBSCAN core distances");
  const distances = Float32Array.from(knn.distances);
  const core = Array.from({ length: count }, (_, point) => distances[point * k + Math.min(minSamples, k) - 1]);
  // The sparse edges are insufficient to prove the complete-graph MST when
  // they are disconnected. Restart from singleton components so the exact
  // fallback cannot retain a sparse edge that a lower complete-graph edge
  // would have displaced.
  edges.length = 0;
  const exact = new UnionFind(count);
  while (edges.length < count - 1) {
    const best = new Map<number, Edge>();
    for (let left = 0; left < count; left++) for (let right = left + 1; right < count; right++) {
      const leftRoot = exact.find(left); const rightRoot = exact.find(right);
      if (leftRoot === rightRoot) continue;
      const edge = { left, right, distance: Math.max(core[left], core[right], cosineDistance(rows, dimension, left, right)) };
      updateBest(best, leftRoot, edge); updateBest(best, rightRoot, edge);
    }
    const additions = [...best.values()].sort(compareEdges); let joined = 0;
    for (const edge of additions) if (exact.join(edge.left, edge.right)) { edges.push(edge); joined++; }
    if (!joined) throw new TypeError("Unable to bridge disconnected mutual-reachability components");
  }
  return flattenEdges(edges);
}

function updateBest(best: Map<number, Edge>, component: number, edge: Edge): void {
  const previous = best.get(component); if (!previous || compareEdges(edge, previous) < 0) best.set(component, edge);
}
function compareEdges(left: Edge, right: Edge): number {
  return left.distance - right.distance || left.left - right.left || left.right - right.right;
}
function flattenEdges(edges: Edge[]): Float32Array {
  return Float32Array.from(edges.flatMap((edge) => [edge.left, edge.right, edge.distance]));
}
function cosineDistance(rows: Float32Array, dimension: number, left: number, right: number): number {
  let dot = 0; let leftNorm = 0; let rightNorm = 0; const leftOffset = left * dimension; const rightOffset = right * dimension;
  for (let column = 0; column < dimension; column++) { const a = rows[leftOffset + column]; const b = rows[rightOffset + column]; dot += a * b; leftNorm += a * a; rightNorm += b * b; }
  const norm = Math.sqrt(leftNorm) * Math.sqrt(rightNorm); return norm <= 1e-12 ? 1 : Math.max(0, Math.min(2, 1 - dot / norm));
}
class UnionFind {
  private readonly parent: number[];
  constructor(count: number) { this.parent = Array.from({ length: count }, (_, index) => index); }
  find(index: number): number { if (this.parent[index] !== index) this.parent[index] = this.find(this.parent[index]); return this.parent[index]; }
  join(left: number, right: number): boolean { left = this.find(left); right = this.find(right); if (left === right) return false; if (left > right) [left, right] = [right, left]; this.parent[right] = left; return true; }
}
