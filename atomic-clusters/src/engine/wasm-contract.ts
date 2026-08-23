/**
 * Typed-array contract shared by the TypeScript orchestration layer and the
 * wasm-bindgen numerical core.  The wrapper deliberately accepts only flat
 * row-major matrices at the WASM boundary; this avoids allocating one JS
 * array for every row of a large embedding set.
 */

export type NumericInput = Float32Array | ArrayLike<number>;
export type IndexInput = Uint32Array | ArrayLike<number>;

export interface FlatMatrix {
  readonly data: Float32Array;
  readonly rows: number;
  readonly cols: number;
}

export interface PcaWasmResult {
  readonly projected: NumericInput;
  readonly explained: NumericInput;
}

export interface RandomizedPcaWasmResult extends PcaWasmResult {
  readonly basis: NumericInput;
  readonly mean: NumericInput;
}

export interface ExactKnnCosineTiledWasmResult {
  readonly indices: IndexInput;
  readonly distances: NumericInput;
  readonly rows: number;
  readonly k: number;
}

export interface MutualReachabilityMstWasmResult {
  readonly edges: NumericInput;
  readonly edge_count: number;
}

/** Flat HDBSCAN extraction result emitted by wasm-bindgen. */
export interface HdbscanExtractWasmResult {
  readonly labels: ArrayLike<number>;
  readonly probabilities: NumericInput;
  readonly outlier_scores: NumericInput;
  readonly cluster_count: number;
}

export interface HnswWasmIndex {
  search(query: NumericInput, k: number): IndexInput;
  free?: () => void;
}

/** The shape emitted by wasm-bindgen for the current Rust core. */
export interface WasmBindings {
  matmul(a: NumericInput, b: NumericInput, m: number, k: number, n: number): NumericInput;
  pca(rows: NumericInput, rowCount: number, dimension: number, components: number): PcaWasmResult;
  /** Optional faster/seeded PCA export; pca remains the required baseline. */
  randomized_pca?: (
    rows: NumericInput, rowCount: number, dimension: number, components: number,
    oversamples: number, powerIterations: number, seed: number
  ) => RandomizedPcaWasmResult;
  cosine_distances(rows: NumericInput, rowCount: number, dimension: number, tile: number): NumericInput;
  /** Optional combined tiled cosine + exact neighbor export. */
  exact_knn_cosine_tiled?: (
    rows: NumericInput, rowCount: number, dimension: number, k: number, tile: number
  ) => ExactKnnCosineTiledWasmResult;
  /** Exact Euclidean mutual-reachability MST for the HDBSCAN discovery path. */
  euclidean_mutual_reachability_mst?: (
    rows: NumericInput, rowCount: number, dimension: number, minSamples: number, tile: number
  ) => MutualReachabilityMstWasmResult;
  exact_knn(distances: NumericInput, rowCount: number, k: number): IndexInput;
  mst(distances: NumericInput, rowCount: number): NumericInput;
  /** Optional mutual-reachability MST export. */
  mutual_reachability_mst?: (
    indices: IndexInput, distances: NumericInput, rowCount: number, k: number, minSamples: number
  ) => MutualReachabilityMstWasmResult;
  /** Optional production HDBSCAN condensed-tree extraction export. */
  hdbscan_extract?: (
    mstEdges: NumericInput, rowCount: number, minClusterSize: number,
    selectionMethod: number, allowSingleCluster: boolean
  ) => HdbscanExtractWasmResult;

  /** Current wasm-core export. */
  HnswIndex?: new (points: NumericInput, count: number, dimension: number, m: number, seed: number) => HnswWasmIndex;
  /** Optional function-style export for a future core without JS classes. */
  hnsw_build?: (points: NumericInput, count: number, dimension: number) => number;
  hnsw_query?: (handle: number, query: NumericInput, k: number) => IndexInput;
  hnsw_free?: (handle: number) => void;
}

export interface WasmKernelOptions {
  readonly cosineTile?: number;
  readonly hnswM?: number;
  readonly hnswSeed?: number;
  /** Quality settings for the deterministic randomized PCA export. */
  readonly pcaOversamples?: number;
  readonly pcaPowerIterations?: number;
  readonly pcaSeed?: number;
}

export interface PcaResult {
  readonly projected: FlatMatrix;
  readonly explained: Float32Array;
}

export interface HnswIndex {
  query(query: NumericInput, k: number): Uint32Array;
  free(): void;
}

export interface WasmKernelContract {
  matmul(a: FlatMatrix, b: FlatMatrix): FlatMatrix;
  pca(rows: FlatMatrix, components: number): PcaResult;
  cosineDistances(rows: FlatMatrix, tile?: number): FlatMatrix;
  exactKnn(distances: FlatMatrix, k: number): Uint32Array;
  hnswBuild(points: FlatMatrix): HnswIndex;
  mst(distances: FlatMatrix): Float32Array;
}

export class WasmContractError extends TypeError {
  constructor(message: string) {
    super(`Invalid WASM contract input: ${message}`);
    this.name = "WasmContractError";
  }
}

export function matrix(data: NumericInput, rows: number, cols: number): FlatMatrix {
  const normalizedRows = positiveInteger(rows, "rows");
  const normalizedCols = positiveInteger(cols, "cols");
  const values = float32(data, "matrix data");
  if (values.length !== normalizedRows * normalizedCols) {
    throw new WasmContractError(`matrix data length ${values.length} does not equal ${normalizedRows} x ${normalizedCols}`);
  }
  return { data: values, rows: normalizedRows, cols: normalizedCols };
}

export function vector(data: NumericInput, expectedLength?: number): Float32Array {
  const values = float32(data, "vector data");
  if (expectedLength !== undefined && values.length !== expectedLength) {
    throw new WasmContractError(`vector length ${values.length} does not equal ${expectedLength}`);
  }
  return values;
}

export class WasmNumericKernel implements WasmKernelContract {
  private readonly cosineTile: number;
  private readonly hnswM: number;
  private readonly hnswSeed: number;
  private readonly pcaOversamples: number;
  private readonly pcaPowerIterations: number;
  private readonly pcaSeed: number;

  constructor(private readonly wasm: WasmBindings, options: WasmKernelOptions = {}) {
    for (const name of ["matmul", "pca", "cosine_distances", "exact_knn", "mst"] as const) {
      if (typeof wasm[name] !== "function") throw new Error(`WASM export ${name} is missing`);
    }
    this.cosineTile = positiveInteger(options.cosineTile ?? 256, "cosineTile");
    this.hnswM = positiveInteger(options.hnswM ?? 16, "hnswM");
    if (!Number.isSafeInteger(options.hnswSeed ?? 42)) throw new WasmContractError("hnswSeed must be a safe integer");
    this.hnswSeed = options.hnswSeed ?? 42;
    this.pcaOversamples = positiveInteger(options.pcaOversamples ?? 8, "pcaOversamples");
    this.pcaPowerIterations = positiveInteger(options.pcaPowerIterations ?? 2, "pcaPowerIterations");
    if (!Number.isSafeInteger(options.pcaSeed ?? 42)) throw new WasmContractError("pcaSeed must be a safe integer");
    this.pcaSeed = options.pcaSeed ?? 42;
  }

  matmul(a: FlatMatrix, b: FlatMatrix): FlatMatrix {
    if (a.cols !== b.rows) throw new WasmContractError(`matmul inner dimensions ${a.cols} and ${b.rows} differ`);
    const output = float32(this.wasm.matmul(a.data, b.data, a.rows, a.cols, b.cols), "matmul output");
    return matrix(output, a.rows, b.cols);
  }

  pca(rows: FlatMatrix, components: number): PcaResult {
    const count = boundedInteger(components, 1, Math.min(rows.rows, rows.cols), "components");
    const output = this.wasm.randomized_pca
      ? this.wasm.randomized_pca(rows.data, rows.rows, rows.cols, count, this.pcaOversamples, this.pcaPowerIterations, this.pcaSeed)
      : this.wasm.pca(rows.data, rows.rows, rows.cols, count);
    if (!output || typeof output !== "object") throw new WasmContractError("pca output must be an object");
    const projected = float32(output.projected, "pca projected output");
    const explained = float32(output.explained, "pca explained output");
    return { projected: matrix(projected, rows.rows, count), explained };
  }

  cosineDistances(rows: FlatMatrix, tile = this.cosineTile): FlatMatrix {
    const tileSize = positiveInteger(tile, "tile");
    const output = float32(this.wasm.cosine_distances(rows.data, rows.rows, rows.cols, tileSize), "cosine distance output");
    return matrix(output, rows.rows, rows.rows);
  }

  exactKnn(distances: FlatMatrix, k: number): Uint32Array {
    if (distances.rows !== distances.cols) throw new WasmContractError("exact k-NN requires a square distance matrix");
    const neighbors = boundedInteger(k, 1, Math.max(1, distances.rows - 1), "k");
    const output = indices(this.wasm.exact_knn(distances.data, distances.rows, neighbors), "exact k-NN output");
    if (output.length !== distances.rows * neighbors) throw new WasmContractError("exact k-NN output has an unexpected length");
    validateIndices(output, distances.rows, "exact k-NN output");
    return output;
  }

  hnswBuild(points: FlatMatrix): HnswIndex {
    if (this.wasm.HnswIndex) {
      const instance = new this.wasm.HnswIndex(points.data, points.rows, points.cols, this.hnswM, this.hnswSeed);
      return { query: (query, k) => queryHnsw(instance, query, points.rows, points.cols, k), free: () => instance.free?.() };
    }
    if (this.wasm.hnsw_build && this.wasm.hnsw_query) {
      const handle = this.wasm.hnsw_build(points.data, points.rows, points.cols);
      if (!Number.isSafeInteger(handle) || handle < 0) throw new WasmContractError("hnsw_build returned an invalid handle");
      return {
        query: (query, k) => {
          const neighbors = boundedInteger(k, 1, points.rows, "k");
          const output = indices(this.wasm.hnsw_query!(handle, vector(query, points.cols), neighbors), "hnsw query output");
          if (output.length !== neighbors) throw new WasmContractError("hnsw query output has an unexpected length");
          validateIndices(output, points.rows, "hnsw query output");
          return output;
        },
        free: () => this.wasm.hnsw_free?.(handle)
      };
    }
    throw new Error("WASM HNSW exports are missing (expected HnswIndex or hnsw_build/hnsw_query)");
  }

  mst(distances: FlatMatrix): Float32Array {
    if (distances.rows !== distances.cols) throw new WasmContractError("MST requires a square distance matrix");
    const output = float32(this.wasm.mst(distances.data, distances.rows), "MST output");
    if (output.length % 3 !== 0 || output.length > Math.max(0, distances.rows - 1) * 3) {
      throw new WasmContractError("MST output must contain up to rows - 1 triples");
    }
    for (let i = 0; i < output.length; i += 3) {
      if (!Number.isInteger(output[i]) || !Number.isInteger(output[i + 1]) || output[i] < 0 || output[i + 1] < 0 || output[i] >= distances.rows || output[i + 1] >= distances.rows) {
        throw new WasmContractError("MST edge endpoints must be valid integer row indices");
      }
    }
    return output;
  }
}

function queryHnsw(instance: HnswWasmIndex, query: NumericInput, count: number, dimension: number, k: number): Uint32Array {
  const neighbors = boundedInteger(k, 1, count, "k");
  const output = indices(instance.search(vector(query, dimension), neighbors), "hnsw query output");
  if (output.length !== neighbors) throw new WasmContractError("hnsw query output has an unexpected length");
  validateIndices(output, count, "hnsw query output");
  return output;
}

function float32(input: NumericInput, name: string): Float32Array {
  if (input === null || input === undefined || typeof (input as ArrayLike<number>).length !== "number") {
    throw new WasmContractError(`${name} must be an array-like numeric buffer`);
  }
  const values = input instanceof Float32Array ? new Float32Array(input) : Float32Array.from(input);
  for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i])) throw new WasmContractError(`${name} contains a non-finite value at ${i}`);
  return values;
}

function indices(input: IndexInput, name: string): Uint32Array {
  if (input === null || input === undefined || typeof (input as ArrayLike<number>).length !== "number") throw new WasmContractError(`${name} must be an array-like index buffer`);
  const values = input instanceof Uint32Array ? new Uint32Array(input) : Uint32Array.from(input);
  return values;
}

function validateIndices(values: Uint32Array, upperBound: number, name: string): void {
  for (let i = 0; i < values.length; i++) {
    if (values[i] >= upperBound) throw new WasmContractError(`${name} contains out-of-range index at ${i}`);
  }
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new WasmContractError(`${name} must be a positive integer`);
  return value;
}

function boundedInteger(value: number, min: number, max: number, name: string): number {
  positiveInteger(value, name);
  if (value < min || value > max) throw new WasmContractError(`${name} must be between ${min} and ${max}`);
  return value;
}
