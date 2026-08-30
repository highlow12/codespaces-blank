import { UMAP } from "umap-js";
import { ClusterResult, ClusterVisualization, ClusteringConfig, HierarchyMerge, HierarchyNode, HierarchyPlacement, HierarchyTree, PcaModelArtifact, PcaPreservationCandidate, PcaSelection, VisualizationCoordinate } from "./types";

export interface NumericKernel {
  normalize(rows: number[][]): number[][];
  pca(rows: number[][], components: number): { projected: number[][]; explained: number[]; mean?: number[]; components?: number[][] };
  cosineDistances(rows: number[][]): number[][];
  exactKnn(rows: number[][], k: number): number[][];
  /** Returns a minimum spanning tree of a mutual-reachability graph. */
  mst(rows: number[][], k: number): Array<[number, number, number]>;
  /** Present only in the packaged WASM kernel. */
  hdbscan?(rows: number[][], minClusterSize: number, minSamples: number): HdbscanOutput;
}

/**
 * Cross-runtime HDBSCAN result contract.  `probabilities` is the confidence
 * of the assigned leaf (`0` for noise); `outlierProxy` is a bounded
 * provider-specific outlier score (the WASM provider uses `1 - probability`).
 * Providers that expose the complete soft membership matrix may also return
 * `memberships`.  A missing matrix is represented by the assigned-membership
 * matrix by the parity helpers, so a provider cannot accidentally claim soft
 * cross-cluster memberships it did not compute.
 */
export interface HdbscanOutput { labels: number[]; probabilities: number[]; outlierProxy?: number[]; memberships?: number[][]; }

/** Provider boundary; the packaged WASM kernel supplies true HDBSCAN extraction. */
export interface HdbscanProvider { fit(rows: number[][], minClusterSize: number, minSamples: number, kernel: NumericKernel): HdbscanOutput; }

/** Stable seam for a separately maintained/native HDBSCAN implementation. */
export interface ExternalHdbscanProvider {
  readonly id: string;
  fit(rows: number[][], minClusterSize: number, minSamples: number): HdbscanOutput;
}

/** Adapt an external provider without letting it bypass the shared contract. */
export class ExternalHdbscanProviderAdapter implements HdbscanProvider {
  constructor(private readonly provider: ExternalHdbscanProvider) {}

  fit(rows: number[][], minClusterSize: number, minSamples: number): HdbscanOutput {
    const result = this.provider.fit(rows, minClusterSize, minSamples);
    return validateHdbscanOutput(result, rows.length, `external provider ${this.provider.id}`);
  }
}

export function validateHdbscanOutput(output: HdbscanOutput, rowCount: number, source = "HDBSCAN provider"): HdbscanOutput {
  if (!output || !Array.isArray(output.labels) || output.labels.length !== rowCount) throw new TypeError(`${source} labels must have ${rowCount} rows`);
  if (!Array.isArray(output.probabilities) || output.probabilities.length !== rowCount) throw new TypeError(`${source} probabilities must have ${rowCount} rows`);
  const outlierProxy = output.outlierProxy || output.probabilities.map((value) => 1 - value);
  if (!Array.isArray(outlierProxy) || outlierProxy.length !== rowCount) throw new TypeError(`${source} outlierProxy must have ${rowCount} rows`);
  const labels = output.labels.map((label, index) => {
    if (!Number.isSafeInteger(label) || label < -1) throw new TypeError(`${source} label ${index} must be -1 or a non-negative integer`);
    return label;
  });
  const nonNoise = [...new Set(labels.filter((label) => label >= 0))].sort((left, right) => left - right);
  if (nonNoise.some((label, index) => label !== index)) throw new TypeError(`${source} labels must be contiguous from zero (up to permutation)`);
  const probabilities = output.probabilities.map((value, index) => {
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new TypeError(`${source} probability ${index} must be between 0 and 1`);
    return value;
  });
  labels.forEach((label, index) => { if (label < 0 && probabilities[index] > 1e-6) throw new TypeError(`${source} noise probability ${index} must be zero`); });
  const outliers = outlierProxy.map((value, index) => {
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new TypeError(`${source} outlierProxy ${index} must be between 0 and 1`);
    return value;
  });
  if (output.memberships !== undefined) {
    if (!Array.isArray(output.memberships) || output.memberships.length !== rowCount) throw new TypeError(`${source} memberships must have ${rowCount} rows`);
    const width = output.memberships[0]?.length || 0;
    for (const row of output.memberships) {
      if (!Array.isArray(row) || row.length !== width || row.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) throw new TypeError(`${source} memberships must be rectangular probabilities`);
      if (row.reduce((sum, value) => sum + value, 0) > 1 + 1e-6) throw new TypeError(`${source} membership rows must sum to at most one`);
    }
    if (width !== nonNoise.length) throw new TypeError(`${source} memberships width must equal the number of clusters`);
  }
  return { labels, probabilities, outlierProxy: outliers, memberships: output.memberships };
}

export class DeterministicHdbscanProvider implements HdbscanProvider {
  fit(rows: number[][], minClusterSize: number, minSamples: number, kernel: NumericKernel): HdbscanOutput {
    const result = kernel.hdbscan ? kernel.hdbscan(rows, minClusterSize, minSamples) : hdbscanFallback(rows, minClusterSize, minSamples, kernel);
    return validateHdbscanOutput(result, rows.length, "WASM/TypeScript HDBSCAN provider");
  }
}

export interface ClusterProgress { (phase: string, progress: number): void; }

function throttledProgress(callback: ClusterProgress): ClusterProgress {
  let lastAt = 0; let lastValue = -Infinity; let lastPhase = "";
  return (phase, value) => {
    const now = Date.now(); const bounded = Math.max(0, Math.min(1, value));
    if (phase !== lastPhase || bounded >= 1 || bounded - lastValue >= 0.01 || now - lastAt >= 100) {
      lastAt = now; lastValue = bounded; lastPhase = phase; callback(phase, bounded);
    }
  };
}

const jsKernel: NumericKernel = {
  normalize, pca: jsPca, cosineDistances, exactKnn,
  mst: (rows, k) => minimumSpanningTree(rows, k)
};

export interface ClusterOptions { kernel?: NumericKernel; hdbscan?: HdbscanProvider; onProgress?: ClusterProgress; signal?: { cancelled: boolean }; }

export interface DiscoveryResult {
  umapFeatures: number[][];
  labels: number[];
  probabilities: number[];
  outlierProxy: number[];
  memberships?: number[][];
}

export interface VisualizationOptions { seed?: number; onProgress?: ClusterProgress; signal?: { cancelled: boolean }; }

function visualizationRandom(seed: number): () => number {
  let state = (seed >>> 0) || 1;
  return () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
}

/** Run the separate 2D explorer UMAP over selected PCA features. */
export async function projectVisualization(pcaFeatures: number[][], labels: number[], options: VisualizationOptions = {}): Promise<ClusterVisualization | undefined> {
  if (pcaFeatures.length < 3) return undefined;
  if (pcaFeatures.some((row) => row.length !== pcaFeatures[0].length)) throw new Error("Visualization features must be rectangular.");
  if (labels.length !== pcaFeatures.length) throw new Error("Visualization labels must align with features.");
  if (pcaFeatures.some((row) => row.some((value) => !Number.isFinite(value)))) throw new Error("Visualization features must be finite.");
  checkCancelled(options);
  const progress = throttledProgress(options.onProgress || (() => undefined));
  const seed = options.seed ?? 42;
  const nNeighbors = Math.min(24, pcaFeatures.length - 1);
  const minDist = 1.0;
  const spread = 1.8;
  const varyingLabels = new Set(labels.filter((label) => Number.isSafeInteger(label))).size > 1;
  const umap = new UMAP({ nComponents: 2, nNeighbors, minDist, spread, random: visualizationRandom(seed) });
  if (varyingLabels) umap.setSupervisedProjection(labels, { targetWeight: 0.01, targetNNeighbors: nNeighbors });
  progress("visualization", 0);
  const started = Date.now();
  const reduced = await umap.fitAsync(pcaFeatures, (epoch) => {
    progress("visualization", Math.min(1, epoch / Math.max(1, pcaFeatures.length * 5)));
    checkCancelled(options);
  });
  checkCancelled(options);
  const coordinates: VisualizationCoordinate[] = reduced.map((row, index) => {
    const coordinate: VisualizationCoordinate = [Number(row[0]), Number(row[1])];
    if (!Number.isFinite(coordinate[0]) || !Number.isFinite(coordinate[1])) throw new Error(`Visualization coordinate ${index} is not finite.`);
    return coordinate;
  });
  progress("visualization", 1);
  return {
    coordinates, labels: labels.slice(),
    configuration: {
      runtime: "umap-js", seed, nComponents: 2, nNeighbors, minDist, spread,
      ...(varyingLabels ? { targetMetric: "categorical" as const, targetWeight: 0.01 } : {})
    },
    timings: { totalMs: Date.now() - started }
  };
}

export async function clusterEmbeddings(ids: string[], input: number[][], config: ClusteringConfig = {}, options: ClusterOptions = {}): Promise<ClusterResult> {
  if (!input.length || input.some((row) => row.length !== input[0].length)) throw new Error("Embeddings must be a non-empty rectangular matrix.");
  if (input.length < 3) {
    const leafLabels = input.map(() => -1);
    const memberships = input.map(() => [] as number[]);
    const leafOrdering: number[] = [];
    const coordinates: VisualizationCoordinate[] = input.length === 1 ? [[0, 0]] : [[-0.5, 0], [0.5, 0]];
    return {
      schemaVersion: 6,
      ids,
      leafLabels,
      probabilities: input.map(() => 0),
      outlierProxy: input.map(() => 1),
      softMemberships: memberships,
      memberships,
      leafOrder: leafOrdering,
      leafOrdering,
      hierarchyPlacements: input.map(() => ({ kind: "residual", nodeId: null, confidence: 0 })),
      pca: { selected: 1, explainedVariance: 1, totalVariance: 0, candidates: [1], preservationCandidates: [], selectionReason: "small_dataset", sampleSize: input.length, varianceTarget: config.pcaVarianceTarget ?? 0.9 },
      hierarchy: { leaves: [], merges: [], root: null, nodes: [], rootChildren: [], splitMethod: "distance-knee-2-5" },
      visualization: {
        coordinates,
        labels: leafLabels.slice(),
        leafOrdering,
        memberships,
        configuration: { runtime: "deterministic-small", seed: config.seed ?? 42, nComponents: 2, nNeighbors: Math.max(1, input.length - 1), minDist: 1, spread: 1.8 },
        timings: { totalMs: 0 }
      },
      timings: { totalMs: 0 }
    };
  }
  const kernel = options.kernel || jsKernel;
  const progress = throttledProgress(options.onProgress || (() => undefined));
  const started = Date.now();
  const normalized = kernel.normalize(input);
  checkCancelled(options);
  progress("pca", 0.05);
  const sampleSize = Math.min(config.pcaSampleSize || 2000, normalized.length);
  const sample = deterministicSample(normalized, sampleSize, config.seed ?? 42);
  const maxComponents = Math.min(config.pcaMaxComponents || 512, normalized[0].length, normalized.length);
  const minComponents = Math.min(maxComponents, Math.max(1, config.pcaMinComponents ?? 32));
  // Fit one bounded pilot, then score prefixes of that same projection. This
  // deliberately avoids re-running randomized PCA at every candidate width.
  const pilotComponents = Math.min(maxComponents, Math.max(minComponents, config.pcaKneeProbeComponents ?? 256));
  const probe = kernel.pca(sample, pilotComponents);
  const selectedPca = selectPcaByPreservation(sample, probe.projected, candidateComponents(pilotComponents, minComponents), sampleSize, config.pcaVarianceTarget ?? 0.9, kernel);
  const pca = kernel.pca(normalized, selectedPca.selected);
  const fitted = pca.mean && pca.components ? { mean: pca.mean, components: pca.components } : jsPca(normalized, selectedPca.selected);
  const modelHash = modelFingerprint(normalized[0].length, selectedPca.selected, fitted.mean, fitted.components, pca.explained);
  selectedPca.totalVariance = centeredVarianceTrace(normalized);
  selectedPca.explainedVariance = explainedFraction(pca.explained, selectedPca.selected, selectedPca.totalVariance);
  selectedPca.model = { modelHash, inputDimension: normalized[0].length, outputDimension: selectedPca.selected, normalization: "l2", mean: fitted.mean, components: fitted.components, explainedVariance: pca.explained.slice(0, selectedPca.selected) };
  checkCancelled(options);
  progress("umap", 0.2);
  const discovery = await discoverPcaFeatures(pca.projected, config, { ...options, onProgress: progress });
  const hdbscan = discovery;
  /* The Python/Pyodide worker uses this same boundary after fitting its
   * authoritative PCA. Keeping discovery separate makes the JS callback a
   * real, testable replacement for Python's optional UMAP/HDBSCAN imports. */
  progress("hdbscan", 0.78);
  const hierarchy = buildHierarchy(pca.projected, hdbscan.labels, hdbscan.probabilities, kernel, config.minClusterSize || 1);
  progress("hierarchy", 0.86);
  const visualization = config.deferVisualization ? undefined : await projectVisualization(pca.projected, hdbscan.labels, {
    seed: config.seed ?? 42,
    signal: options.signal,
    onProgress: (phase, value) => progress(phase, 0.86 + value * 0.1)
  });
  progress("complete", 1);
  const leafOrder = hierarchy.leaves.slice();
  const clusterCount = leafOrder.length;
  const softMemberships = hdbscan.memberships || computeCentroidSoftMemberships(pca.projected, hdbscan.labels, hdbscan.probabilities, leafOrder);
  const memberships = softMemberships.map((row) => row.slice(0, clusterCount));
  const hierarchyPlacements = computeHierarchyPlacements(hdbscan.labels, hdbscan.probabilities, memberships, hierarchy);
  const placedHierarchy = applyHierarchyPlacementMasses(hierarchy, hierarchyPlacements);
  const completeVisualization = visualization ? { ...visualization, leafOrdering: leafOrder, memberships } : undefined;
  return {
    schemaVersion: 6, ids, leafLabels: hdbscan.labels, probabilities: hdbscan.probabilities,
    outlierProxy: hdbscan.outlierProxy, softMemberships: memberships, leafOrder, leafOrdering: leafOrder, memberships, pca: selectedPca, hierarchy: placedHierarchy, ...(completeVisualization ? { visualization: completeVisualization } : {}),
    hierarchyPlacements,
    timings: { totalMs: Date.now() - started }
  };
}

/** Exact cosine soft memberships for providers that expose labels only. */
export function computeCentroidSoftMemberships(rows: readonly number[][], labels: readonly number[], probabilities: readonly number[], leafOrder: readonly number[]): number[][] {
  const centers = new Map<number, number[]>();
  const dimension = rows[0]?.length || 0;
  for (const leaf of leafOrder) {
    const center = new Array<number>(dimension).fill(0); let total = 0; let memberCount = 0;
    // Avoid creating a member-index array for every leaf.  This also keeps
    // the centroid accumulation order identical to the old labels.map/filter
    // implementation, which is important for deterministic soft memberships.
    for (let index = 0; index < labels.length; index++) if (labels[index] === leaf) {
      memberCount++;
      const weight = Math.max(1e-6, probabilities[index] || 0); total += weight;
      const row = rows[index]; for (let d = 0; d < dimension; d++) center[d] += row[d] * weight;
    }
    if (!memberCount) { centers.set(leaf, center); continue; }
    for (let d = 0; d < dimension; d++) center[d] /= total;
    centers.set(leaf, normalizeVector(center));
  }
  return rows.map((row, index) => {
    const vector = normalizeVector(row); const scores = leafOrder.map((leaf) => Math.max(0, dot(vector, centers.get(leaf) || vector))); const total = scores.reduce((sum, score) => sum + score, 0); const probability = Math.max(0, Math.min(1, probabilities[index] || 0));
    if (total <= 1e-12) return leafOrder.map((leaf) => leaf === labels[index] ? probability : 0);
    return scores.map((score) => probability * score / total);
  });
}

/** Run the browser-side UMAP/HDBSCAN discovery boundary on PCA features. */
export async function discoverPcaFeatures(pcaFeatures: number[][], config: ClusteringConfig = {}, options: ClusterOptions = {}): Promise<DiscoveryResult> {
  if (!pcaFeatures.length || pcaFeatures.some((row) => row.length !== pcaFeatures[0].length)) throw new Error("PCA features must be a non-empty rectangular matrix.");
  const kernel = options.kernel || jsKernel;
  const progress = throttledProgress(options.onProgress || (() => undefined));
  checkCancelled(options);
  progress("umap", 0.2);
  // umap-js initializes its simplicial-set embedding from this seeded random
  // source; this matches the production umap-learn route's random init.
  const random = seededRandom(config.seed ?? 42);
  const umap = new UMAP({
    nComponents: config.umapComponents || 20,
    nNeighbors: Math.min(config.umapNeighbors || 15, Math.max(2, pcaFeatures.length - 1)),
    minDist: config.umapMinDist ?? 0.1,
    random
  });
  const reduced = await umap.fitAsync(pcaFeatures, (epoch) => {
    progress("umap", 0.2 + Math.min(0.55, epoch / Math.max(1, pcaFeatures.length * 5) * 0.55));
    checkCancelled(options);
  });
  checkCancelled(options);
  progress("hdbscan", 0.78);
  const minClusterSize = config.minClusterSize || 5;
  // This is intentionally distinct from minClusterSize: the authoritative
  // Python PCA -> UMAP -> HDBSCAN route uses min_samples=3.
  const minSamples = config.minSamples ?? 3;
  const hdbscan = (options.hdbscan || new DeterministicHdbscanProvider()).fit(reduced, minClusterSize, minSamples, kernel);
  return { umapFeatures: reduced, labels: hdbscan.labels, probabilities: hdbscan.probabilities, outlierProxy: hdbscan.outlierProxy || hdbscan.probabilities.map((value) => 1 - value), memberships: hdbscan.memberships };
}

export function candidateComponents(max: number, min = 32, step = 32): number[] {
  const cap = Math.max(1, Math.floor(max));
  const first = Math.min(cap, Math.max(1, Math.floor(min)));
  const increment = Math.max(1, Math.floor(step));
  const values: number[] = [];
  for (let value = first; value <= cap; value += increment) values.push(value);
  return values.length ? values : [cap];
}

/**
 * Original-project PCA selection. One pilot PCA projection is sliced at
 * 32-component prefixes; each normalized prefix is scored by exact cosine
 * k-NN preservation against the normalized input. A first local gain below
 * 0.05 proposes the previous prefix, then a monotonized global preservation
 * knee can move selection later. This is the same rule as pca_dimension_search.py.
 */
export function selectPcaByPreservation(originalRows: number[][], projected: number[][], candidates: number[], sampleSize: number, varianceTarget: number, kernel: NumericKernel): PcaSelection {
  const validCandidates = candidates.filter((dimension) => Number.isSafeInteger(dimension) && dimension >= 1 && dimension <= projected[0].length);
  if (!validCandidates.length) throw new Error("PCA preservation selection needs at least one valid prefix.");
  const kValues = neighborhoodKValues(originalRows.length);
  const maximumK = Math.max(...kValues);
  const reference = kernel.exactKnn(originalRows, maximumK);
  // Reuse one row workspace for every PCA prefix.  Calling `slice()` here
  // used to allocate a complete N x D nested matrix for every candidate,
  // even though normalization and k-NN consume it synchronously.  Keeping
  // the public NumericKernel boundary as number[][] preserves third-party
  // kernels and exact tie-breaking while removing those transient rows.
  const prefixWidth = Math.max(...validCandidates);
  const prefixRows = projected.map(() => new Array<number>(prefixWidth));
  const diagnostics: PcaPreservationCandidate[] = [];
  for (const dimension of validCandidates) {
    for (let rowIndex = 0; rowIndex < projected.length; rowIndex++) {
      const source = projected[rowIndex]; const target = prefixRows[rowIndex];
      for (let column = 0; column < dimension; column++) target[column] = source[column];
      target.length = dimension;
    }
    const prefix = kernel.normalize(prefixRows);
    const neighbors = kernel.exactKnn(prefix, maximumK);
    const byK: Record<number, number> = {};
    for (const k of kValues) byK[k] = meanNeighborPreservation(reference, neighbors, k);
    const mean = kValues.reduce((sum, k) => sum + byK[k], 0) / kValues.length;
    diagnostics.push({ dimension, meanNeighborPreservation: mean, neighborPreservationByK: byK, neighborPreservationGain: diagnostics.length ? mean - diagnostics[diagnostics.length - 1].meanNeighborPreservation : null });
  }
  const choice = choosePcaPreservationCandidate(diagnostics);
  return {
    selected: choice.selected.dimension, explainedVariance: 0, totalVariance: 0, candidates: validCandidates,
    preservationCandidates: diagnostics, selectionReason: choice.reason, sampleSize, varianceTarget
  };
}

export function choosePcaPreservationCandidate(diagnostics: PcaPreservationCandidate[]): { selected: PcaPreservationCandidate; reason: NonNullable<PcaSelection["selectionReason"]> } {
  if (!diagnostics.length) throw new Error("PCA preservation selection needs diagnostics.");
  let selected = diagnostics[diagnostics.length - 1];
  let reason: NonNullable<PcaSelection["selectionReason"]> = "all_gains_meet_minimum_use_maximum_dimension";
  for (let index = 1; index < diagnostics.length; index++) if (diagnostics[index].neighborPreservationGain! < 0.05) {
    selected = diagnostics[index - 1]; reason = "first_below_minimum_gain_use_previous_dimension"; break;
  }
  const globalKnee = globalPreservationKnee(diagnostics);
  if (globalKnee.dimension > selected.dimension) { selected = globalKnee; reason = "global_preservation_knee_after_local_plateau"; }
  return { selected, reason };
}

function neighborhoodKValues(rows: number): number[] {
  const values = [15, 30].filter((k) => k < rows);
  return values.length ? values : [Math.max(1, Math.min(rows - 1, Math.floor(rows / 2)))];
}

function meanNeighborPreservation(reference: number[][], candidate: number[][], k: number): number {
  let preserved = 0;
  const wanted = new Set<number>();
  for (let row = 0; row < reference.length; row++) {
    wanted.clear();
    for (let index = 0; index < k; index++) wanted.add(reference[row][index]);
    for (let index = 0; index < k; index++) if (wanted.has(candidate[row][index])) preserved++;
  }
  return preserved / (reference.length * k);
}

function globalPreservationKnee(candidates: PcaPreservationCandidate[]): PcaPreservationCandidate {
  if (candidates.length <= 2) return candidates[0];
  const firstDimension = candidates[0].dimension; const lastDimension = candidates[candidates.length - 1].dimension;
  const preservation = candidates.map((candidate) => candidate.meanNeighborPreservation);
  for (let index = 1; index < preservation.length; index++) preservation[index] = Math.max(preservation[index], preservation[index - 1]);
  const range = preservation[preservation.length - 1] - preservation[0];
  if (lastDimension === firstDimension || range <= 1e-12) return candidates[0];
  let bestIndex = 0; let bestStrength = -Infinity;
  for (let index = 0; index < candidates.length; index++) {
    const x = (candidates[index].dimension - firstDimension) / (lastDimension - firstDimension);
    const y = (preservation[index] - preservation[0]) / range;
    const strength = y - x;
    if (strength > bestStrength + 1e-12) { bestStrength = strength; bestIndex = index; }
  }
  return candidates[bestIndex];
}

function explainedFraction(explained: number[], selected: number, totalVariance: number): number {
  if (totalVariance <= Number.EPSILON) return 1;
  const captured = explained.slice(0, selected).reduce((sum, value) => sum + Math.max(0, value), 0);
  return Math.max(0, Math.min(1, captured / totalVariance));
}

/** Trace of the sample covariance: total variance around the component means. */
function centeredVarianceTrace(rows: number[][]): number {
  if (rows.length < 2) return 0;
  const means = new Array(rows[0].length).fill(0);
  for (const row of rows) for (let dimension = 0; dimension < row.length; dimension++) means[dimension] += row[dimension] / rows.length;
  let squaredDistance = 0;
  for (const row of rows) for (let dimension = 0; dimension < row.length; dimension++) {
    const delta = row[dimension] - means[dimension]; squaredDistance += delta * delta;
  }
  return squaredDistance / (rows.length - 1);
}

function buildBinaryHierarchy(rows: number[][], labels: number[], probabilities: number[], kernel: NumericKernel = jsKernel): HierarchyTree {
  const groups = new Map<number, number[]>();
  labels.forEach((label, index) => { if (label >= 0) groups.set(label, [...(groups.get(label) || []), index]); });
  const leaves = [...groups.keys()].sort((a, b) => a - b);
  if (leaves.length < 2) return { leaves, merges: [], root: leaves[0] ?? null };
  const centers = new Map<number, number[]>(); const masses = new Map<number, number>();
  for (const leaf of leaves) {
    const members = groups.get(leaf)!; const center = new Array(rows[0].length).fill(0); let mass = 0;
    for (const index of members) { const weight = Math.max(1e-6, probabilities[index]); mass += weight; for (let d = 0; d < center.length; d++) center[d] += rows[index][d] * weight; }
    centers.set(leaf, normalizeVector(center.map((value) => value / mass))); masses.set(leaf, mass);
  }
  const active = new Set(leaves); const merges: HierarchyMerge[] = []; const heights = new Map(leaves.map((leaf) => [leaf, 0])); let next = Math.max(...leaves) + 1;
  type Pair = [number, number, number]; const heap: Pair[] = [];
  const before = (a: Pair, b: Pair): boolean => a[2] < b[2] || (a[2] === b[2] && (a[0] < b[0] || (a[0] === b[0] && a[1] < b[1])));
  const push = (pair: Pair) => { heap.push(pair); let i = heap.length - 1; while (i) { const p = (i - 1) >> 1; if (!before(heap[i], heap[p])) break; [heap[i], heap[p]] = [heap[p], heap[i]]; i = p; } };
  const pop = (): Pair => { const first = heap[0]; const last = heap.pop()!; if (heap.length) { heap[0] = last; let i = 0; while (true) { const left = i * 2 + 1; const right = left + 1; let best = i; if (left < heap.length && before(heap[left], heap[best])) best = left; if (right < heap.length && before(heap[right], heap[best])) best = right; if (best === i) break; [heap[i], heap[best]] = [heap[best], heap[i]]; i = best; } } return first; };
  const addPair = (a: number, b: number) => { const left = Math.min(a, b), right = Math.max(a, b); push([left, right, Math.max(0, 1 - dot(centers.get(left)!, centers.get(right)!))]); };
  for (let i = 0; i < leaves.length; i++) for (let j = i + 1; j < leaves.length; j++) addPair(leaves[i], leaves[j]);
  while (active.size > 1) {
    let best = pop(); while (!active.has(best[0]) || !active.has(best[1])) best = pop();
    const [left, right, distance] = best; const mass = masses.get(left)! + masses.get(right)!;
    const merged = centers.get(left)!.map((value, i) => (value * masses.get(left)! + centers.get(right)![i] * masses.get(right)!) / mass);
    centers.set(next, normalizeVector(merged)); masses.set(next, mass); active.delete(left); active.delete(right); active.add(next);
    // Heights in a dendrogram must be monotone even when an upstream provider
    // returns a non-monotone pairwise distance sequence.
    const height = Math.max(distance, heights.get(left) || 0, heights.get(right) || 0); heights.set(next, height);
    merges.push({ id: next, left, right, distance: height, mass }); for (const other of active) if (other !== next) addPair(next, other); next++;
  }
  return { leaves, merges, root: [...active][0] };
}

function naryNodeMass(id: number, masses: Map<number, number>, merges: ReadonlyMap<number, HierarchyMerge>): number {
  const existing = masses.get(id); if (existing !== undefined) return existing;
  const merge = merges.get(id); if (!merge) return 0;
  const mass = naryNodeMass(merge.left, masses, merges) + naryNodeMass(merge.right, masses, merges); masses.set(id, mass); return mass;
}

export function chooseNaryFrontier(root: number, merges: ReadonlyMap<number, HierarchyMerge>, masses: Map<number, number>, minMass: number): number[] {
  const rootMerge = merges.get(root); if (!rootMerge) return [root];
  const frontier = [rootMerge.left, rootMerge.right];
  const candidates = new Map<number, number[]>([[2, frontier.slice()]]);
  const appliedHeights = new Map<number, number>([[2, rootMerge.distance]]);
  const height = (id: number): number => merges.get(id)?.distance || 0;
  const nextSplit = (items: readonly number[]): { index: number; id: number; height: number } | null => {
    let candidate = -1; let candidateHeight = -Infinity;
    for (let index = 0; index < items.length; index++) {
      const merge = merges.get(items[index]); if (!merge) continue;
      // A split that creates a tiny child is not a valid independent child.
      if (naryNodeMass(merge.left, masses, merges) < minMass || naryNodeMass(merge.right, masses, merges) < minMass) continue;
      const candidateId = items[index]; const candidateDistance = height(candidateId);
      if (candidateDistance > candidateHeight || (candidateDistance === candidateHeight && (candidate < 0 || candidateId < items[candidate]))) {
        candidate = index; candidateHeight = candidateDistance;
      }
    }
    return candidate < 0 ? null : { index: candidate, id: items[candidate], height: candidateHeight };
  };
  while (frontier.length < 5) {
    const split = nextSplit(frontier); if (!split) break;
    const merge = merges.get(split.id)!; frontier.splice(split.index, 1, merge.left, merge.right);
    candidates.set(frontier.length, frontier.slice()); appliedHeights.set(frontier.length, split.height);
  }
  let bestCount = 2; let bestGap = Number.NEGATIVE_INFINITY;
  for (const [count, items] of candidates) {
    const next = nextSplit(items); if (!next) continue;
    const applied = appliedHeights.get(count)!; const scale = Math.max(Math.abs(applied), Math.abs(next.height), 1e-12);
    const gap = (applied - next.height) / scale;
    if (gap > bestGap + 1e-12) { bestGap = gap; bestCount = count; }
  }
  // No last/next split pair means no valid knee; the deterministic fallback is binary.
  return (Number.isFinite(bestGap) ? candidates.get(bestCount) : candidates.get(2))!.slice();
}

/** Build the user-facing deterministic 2–5-ary hierarchy beside binary merges. */
export function buildHierarchy(rows: number[][], labels: number[], probabilities: number[], kernel: NumericKernel = jsKernel, minClusterSize = 1): HierarchyTree {
  const binary = buildBinaryHierarchy(rows, labels, probabilities, kernel);
  const merges = new Map(binary.merges.map((merge) => [merge.id, merge]));
  // Promotion is governed by note count (minClusterSize), while the legacy
  // binary merge mass continues to preserve its probability-weighted value.
  const massMemo = new Map<number, number>(); labels.forEach((label) => { if (label >= 0) massMemo.set(label, (massMemo.get(label) || 0) + 1); });
  const nodes = new Map<number, HierarchyNode>();
  const massOf = (id: number): number => naryNodeMass(id, massMemo, merges);
  const leavesMemo = new Map<number, number[]>();
  const leavesOf = (id: number): number[] => {
    const cached = leavesMemo.get(id); if (cached) return cached;
    const merge = merges.get(id); if (!merge) return [id];
    const leaves = [...new Set([...leavesOf(merge.left), ...leavesOf(merge.right)])].sort((a, b) => a - b); leavesMemo.set(id, leaves); return leaves;
  };
  const make = (id: number): void => {
    if (nodes.has(id)) return;
    const merge = merges.get(id);
    if (!merge) { nodes.set(id, { id, children: [], descendantLeaves: [id], distance: 0, mass: massOf(id) }); return; }
    const children = chooseNaryFrontier(id, merges, massMemo, Math.max(1, minClusterSize));
    children.forEach(make);
    nodes.set(id, { id, children, descendantLeaves: leavesOf(id), distance: merge.distance, mass: massOf(id) });
  };
  if (binary.root !== null) make(binary.root);
  const rootChildren = binary.root === null ? binary.leaves.slice() : (nodes.get(binary.root)?.children.length ? nodes.get(binary.root)!.children.slice() : [binary.root]);
  if (binary.root === null) rootChildren.forEach(make);
  return { ...binary, nodes: [...nodes.values()].sort((a, b) => a.id - b.id), rootChildren, splitMethod: "distance-knee-2-5" };
}

function hierarchyChildren(result: HierarchyTree): Map<number, number[]> {
  const map = new Map<number, number[]>(); for (const node of result.nodes || []) map.set(node.id, node.children.slice());
  if (!map.size) for (const merge of result.merges) map.set(merge.id, [merge.left, merge.right]);
  return map;
}

/** Deterministically place every row at one leaf or the residual boundary. */
export function computeHierarchyPlacements(labels: readonly number[], probabilities: readonly number[], memberships: readonly number[][], hierarchy: HierarchyTree): HierarchyPlacement[] {
  const ordering = hierarchy.leaves.slice(); const leafColumns = new Map(ordering.map((leaf, index) => [leaf, index])); const children = hierarchyChildren(hierarchy); const nodeMap = new Map((hierarchy.nodes || []).map((node) => [node.id, node])); const rootChildren = hierarchy.rootChildren || (hierarchy.root === null ? hierarchy.leaves : [hierarchy.root]);
  const hierarchyRoot = hierarchy.root === null ? null : nodeMap.get(hierarchy.root);
  const startsBelowRoot = !!hierarchyRoot && hierarchyRoot.children.length === rootChildren.length && hierarchyRoot.children.every((child, index) => child === rootChildren[index]);
  return labels.map((label, index) => {
    if (label < 0) return { kind: "residual", nodeId: null, confidence: 0 };
    const row = memberships[index] || []; let currentChildren = rootChildren; let parent: number | null = startsBelowRoot ? hierarchy.root : null; let confidence = Math.max(0, Math.min(1, probabilities[index] || 0));
    while (currentChildren.length) {
      const masses = currentChildren.map((child) => {
        const node = nodeMap.get(child); const descendants = node?.descendantLeaves || [child];
        return descendants.reduce((sum, leaf) => { const column = leafColumns.get(leaf); return sum + (column === undefined ? 0 : Math.max(0, Number(row[column]) || 0)); }, 0);
      });
      const total = masses.reduce((sum, value) => sum + value, 0); let best = 0; for (let i = 1; i < masses.length; i++) if (masses[i] > masses[best] || (masses[i] === masses[best] && currentChildren[i] < currentChildren[best])) best = i;
      if (total <= 1e-9 || masses[best] <= total * 0.5) return { kind: "residual", nodeId: parent, confidence: total > 0 ? masses[best] / total : 0 };
      confidence = masses[best] / total; const next = currentChildren[best]; const nextChildren = children.get(next) || [];
      if (!nextChildren.length) return { kind: "leaf", nodeId: next, confidence };
      parent = next; currentChildren = nextChildren;
    }
    return { kind: "leaf", nodeId: label, confidence };
  });
}

/** Recount user-facing node mass from terminal placements, excluding residuals from descendants. */
export function applyHierarchyPlacementMasses(hierarchy: HierarchyTree, placements: readonly HierarchyPlacement[]): HierarchyTree {
  if (!hierarchy.nodes) return hierarchy;
  const parent = new Map<number, number>(); for (const node of hierarchy.nodes) for (const child of node.children) parent.set(child, node.id);
  const masses = new Map(hierarchy.nodes.map((node) => [node.id, 0]));
  for (const placement of placements) {
    if (placement.nodeId === null) continue;
    let current: number | undefined = placement.nodeId; const seen = new Set<number>();
    while (current !== undefined && !seen.has(current)) { seen.add(current); masses.set(current, (masses.get(current) || 0) + 1); current = parent.get(current); }
  }
  return { ...hierarchy, nodes: hierarchy.nodes.map((node) => ({ ...node, mass: masses.get(node.id) || 0 })) };
}

function hdbscanFallback(rows: number[][], minClusterSize: number, minSamples: number, kernel: NumericKernel): { labels: number[]; probabilities: number[] } {
  const n = rows.length; if (n < minClusterSize * 2) return { labels: new Array(n).fill(-1), probabilities: new Array(n).fill(0) };
  if (n >= 512) throw new Error("The exact JavaScript HDBSCAN fallback is limited to fewer than 512 rows; use the packaged WASM kernel.");
  const k = Math.max(2, Math.min(minSamples, n - 1));
  // The production HDBSCAN provider consumes the WASM MST. This deterministic
  // fallback uses its edge weights as a density graph when the asset is absent.
  const edges = kernel.mst(rows, k); const distances = edges.map((edge) => edge[2]).sort((a, b) => a - b);
  const threshold = distances[Math.floor(Math.max(0, distances.length - 1) * 0.65)] ?? 0.5;
  const parent = Array.from({ length: n }, (_, i) => i); const find = (x: number): number => parent[x] === x ? x : (parent[x] = find(parent[x]));
  const join = (a: number, b: number) => { a = find(a); b = find(b); if (a !== b) parent[b] = a; };
  edges.forEach(([a, b, distance]) => { if (distance <= threshold) join(a, b); });
  const groups = new Map<number, number[]>(); for (let i = 0; i < n; i++) groups.set(find(i), [...(groups.get(find(i)) || []), i]);
  const labels = new Array(n).fill(-1); const probabilities = new Array(n).fill(0); let label = 0;
  for (const members of groups.values()) if (members.length >= minClusterSize) {
    members.forEach((index) => { labels[index] = label; probabilities[index] = 1; }); label++;
  }
  return { labels, probabilities };
}

function normalize(rows: number[][]): number[][] {
  const normalized: number[][] = new Array(rows.length);
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) normalized[rowIndex] = normalizeVector(rows[rowIndex]);
  return normalized;
}
function normalizeVector(row: number[]): number[] { const norm = Math.sqrt(row.reduce((sum, value) => sum + value * value, 0)) || 1; return row.map((value) => value / norm); }
function dot(a: number[], b: number[]): number { return a.reduce((sum, value, i) => sum + value * b[i], 0); }
function cosineDistances(rows: number[][]): number[][] { const normalized = normalize(rows); const result = Array.from({ length: normalized.length }, () => new Array(normalized.length).fill(0)); for (let i = 0; i < normalized.length; i++) for (let j = i; j < normalized.length; j++) { const distance = Math.max(0, 1 - dot(normalized[i], normalized[j])); result[i][j] = distance; result[j][i] = distance; } return result; }
function exactKnnFromDistances(distances: number[][], k: number): number[][] {
  const result = new Array<number[]>(distances.length); const candidates = new Array<number>(Math.max(0, distances.length - 1));
  for (let rowIndex = 0; rowIndex < distances.length; rowIndex++) {
    const row = distances[rowIndex]; let count = 0;
    for (let index = 0; index < row.length; index++) if (index !== rowIndex) candidates[count++] = index;
    candidates.length = count;
    candidates.sort((left, right) => row[left] - row[right] || left - right);
    const width = Math.min(k, count); const neighbors = new Array<number>(width);
    for (let index = 0; index < width; index++) neighbors[index] = candidates[index];
    result[rowIndex] = neighbors;
  }
  return result;
}
function exactKnn(rows: number[][], k: number): number[][] { return exactKnnFromDistances(cosineDistances(rows), k); }
function minimumSpanningTree(rows: number[][], k: number): Array<[number, number, number]> {
  const distances = cosineDistances(rows); const neighbors = exactKnnFromDistances(distances, k); const edges: Array<[number, number, number]> = []; for (let i = 0; i < rows.length; i++) neighbors[i].forEach((j) => edges.push([i, j, distances[i][j]]));
  edges.sort((a, b) => a[2] - b[2]); const parent = Array.from({ length: rows.length }, (_, i) => i); const find = (x: number): number => parent[x] === x ? x : (parent[x] = find(parent[x]));
  const result: Array<[number, number, number]> = []; for (const edge of edges) { const a = find(edge[0]); const b = find(edge[1]); if (a !== b) { parent[a] = b; result.push(edge); if (result.length === rows.length - 1) break; } } return result;
}
function deterministicSample(rows: number[][], size: number, seed: number): number[][] { const indices = Array.from({ length: rows.length }, (_, i) => i); let state = seed >>> 0; for (let i = indices.length - 1; i > 0; i--) { state = Math.imul(state ^ (state >>> 16), 2246822519) >>> 0; const j = state % (i + 1); [indices[i], indices[j]] = [indices[j], indices[i]]; } return indices.slice(0, size).map((i) => rows[i]); }
function seededRandom(seed: number): () => number { let state = seed >>> 0; return () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; }; }
function checkCancelled(options: ClusterOptions): void { if (options.signal?.cancelled) throw new Error("Clustering cancelled"); }

function jsPca(rows: number[][], components: number): { projected: number[][]; explained: number[]; mean: number[]; components: number[][] } {
  const n = rows.length; const dimensions = rows[0].length; const means = new Array(dimensions).fill(0); rows.forEach((row) => row.forEach((value, i) => means[i] += value / n));
  const centered = rows.map((row) => row.map((value, i) => value - means[i])); const covariance = Array.from({ length: dimensions }, () => new Array(dimensions).fill(0));
  for (const row of centered) for (let i = 0; i < dimensions; i++) for (let j = i; j < dimensions; j++) covariance[i][j] += row[i] * row[j] / Math.max(1, n - 1);
  for (let i = 0; i < dimensions; i++) for (let j = i + 1; j < dimensions; j++) covariance[j][i] = covariance[i][j];
  const vectors: number[][] = []; const values: number[] = []; for (let component = 0; component < Math.min(components, dimensions); component++) { let vector: number[] = Array.from({ length: dimensions }, (_, i) => i === component ? 1 : 0); for (let iteration = 0; iteration < 30; iteration++) { const next = covariance.map((row) => dot(row, vector)); const norm = Math.sqrt(dot(next, next)) || 1; vector = next.map((value) => value / norm); } const eigenvalue = Math.max(0, dot(vector, covariance.map((row) => dot(row, vector)))); values.push(eigenvalue); vectors.push(vector); for (let i = 0; i < dimensions; i++) for (let j = 0; j < dimensions; j++) covariance[i][j] -= eigenvalue * vector[i] * vector[j]; }
  return { projected: centered.map((row) => vectors.map((vector) => dot(row, vector))), explained: values, mean: means, components: vectors };
}

function modelFingerprint(inputDimension: number, outputDimension: number, mean: number[], components: number[][], explainedVariance: number[]): string {
  const value = JSON.stringify({ inputDimension, outputDimension, normalization: "l2", mean, components, explainedVariance });
  let hash = 2166136261; for (let index = 0; index < value.length; index++) { hash ^= value.charCodeAt(index); hash = Math.imul(hash, 16777619); }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}
