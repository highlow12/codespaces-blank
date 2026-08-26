import { UMAP } from "umap-js";
import { ClusterResult, ClusterVisualization, ClusteringConfig, HierarchyMerge, HierarchyTree, PcaPreservationCandidate, PcaSelection, VisualizationCoordinate } from "./types";

export interface NumericKernel {
  normalize(rows: number[][]): number[][];
  pca(rows: number[][], components: number): { projected: number[][]; explained: number[] };
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
  const progress = options.onProgress || (() => undefined);
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
    return { schemaVersion: 5, ids, leafLabels: input.map(() => -1), probabilities: input.map(() => 0), outlierProxy: input.map(() => 1), softMemberships: input.map(() => []), leafOrder: [], pca: { selected: 1, explainedVariance: 1, totalVariance: 0, candidates: [1], preservationCandidates: [], selectionReason: "small_dataset", sampleSize: input.length, varianceTarget: config.pcaVarianceTarget ?? 0.9 }, hierarchy: { leaves: [], merges: [], root: null }, timings: { totalMs: 0 } };
  }
  const kernel = options.kernel || jsKernel;
  const progress = options.onProgress || (() => undefined);
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
  selectedPca.totalVariance = centeredVarianceTrace(normalized);
  selectedPca.explainedVariance = explainedFraction(pca.explained, selectedPca.selected, selectedPca.totalVariance);
  checkCancelled(options);
  progress("umap", 0.2);
  const discovery = await discoverPcaFeatures(pca.projected, config, options);
  const hdbscan = discovery;
  /* The Python/Pyodide worker uses this same boundary after fitting its
   * authoritative PCA. Keeping discovery separate makes the JS callback a
   * real, testable replacement for Python's optional UMAP/HDBSCAN imports. */
  progress("hdbscan", 0.78);
  const hierarchy = buildHierarchy(pca.projected, hdbscan.labels, hdbscan.probabilities, kernel);
  progress("hierarchy", 0.86);
  const visualization = await projectVisualization(pca.projected, hdbscan.labels, {
    seed: config.seed ?? 42,
    signal: options.signal,
    onProgress: (phase, value) => progress(phase, 0.86 + value * 0.1)
  });
  progress("complete", 1);
  const leafOrder = hierarchy.leaves.slice();
  const clusterCount = leafOrder.length;
  const softMemberships = hdbscan.memberships || hdbscan.labels.map((label, index) => leafOrder.map((leaf) => label === leaf ? hdbscan.probabilities[index] : 0));
  return {
    schemaVersion: 5, ids, leafLabels: hdbscan.labels, probabilities: hdbscan.probabilities,
    outlierProxy: hdbscan.outlierProxy, softMemberships: softMemberships.map((row) => row.slice(0, clusterCount)), leafOrder, pca: selectedPca, hierarchy, ...(visualization ? { visualization } : {}),
    timings: { totalMs: Date.now() - started }
  };
}

/** Run the browser-side UMAP/HDBSCAN discovery boundary on PCA features. */
export async function discoverPcaFeatures(pcaFeatures: number[][], config: ClusteringConfig = {}, options: ClusterOptions = {}): Promise<DiscoveryResult> {
  if (!pcaFeatures.length || pcaFeatures.some((row) => row.length !== pcaFeatures[0].length)) throw new Error("PCA features must be a non-empty rectangular matrix.");
  const kernel = options.kernel || jsKernel;
  const progress = options.onProgress || (() => undefined);
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
  const diagnostics: PcaPreservationCandidate[] = [];
  for (const dimension of validCandidates) {
    const prefix = kernel.normalize(projected.map((row) => row.slice(0, dimension)));
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
  for (let row = 0; row < reference.length; row++) {
    const wanted = new Set(reference[row].slice(0, k));
    for (const neighbor of candidate[row].slice(0, k)) if (wanted.has(neighbor)) preserved++;
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

export function buildHierarchy(rows: number[][], labels: number[], probabilities: number[], kernel: NumericKernel = jsKernel): HierarchyTree {
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
  const active = new Set(leaves); const merges: HierarchyMerge[] = []; let next = Math.max(...leaves) + 1;
  while (active.size > 1) {
    const ids = [...active]; let best: [number, number, number] | null = null;
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
      const distance = 1 - dot(centers.get(ids[i])!, centers.get(ids[j])!);
      if (!best || distance < best[2]) best = [ids[i], ids[j], distance];
    }
    const [left, right, distance] = best!; const mass = masses.get(left)! + masses.get(right)!;
    const merged = centers.get(left)!.map((value, i) => (value * masses.get(left)! + centers.get(right)![i] * masses.get(right)!) / mass);
    centers.set(next, normalizeVector(merged)); masses.set(next, mass); active.delete(left); active.delete(right); active.add(next);
    merges.push({ id: next, left, right, distance, mass }); next++;
  }
  return { leaves, merges, root: [...active][0] };
}

function hdbscanFallback(rows: number[][], minClusterSize: number, minSamples: number, kernel: NumericKernel): { labels: number[]; probabilities: number[] } {
  const n = rows.length; if (n < minClusterSize * 2) return { labels: new Array(n).fill(-1), probabilities: new Array(n).fill(0) };
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

function normalize(rows: number[][]): number[][] { return rows.map((row) => normalizeVector(row)); }
function normalizeVector(row: number[]): number[] { const norm = Math.sqrt(row.reduce((sum, value) => sum + value * value, 0)) || 1; return row.map((value) => value / norm); }
function dot(a: number[], b: number[]): number { return a.reduce((sum, value, i) => sum + value * b[i], 0); }
function cosineDistances(rows: number[][]): number[][] { const normalized = normalize(rows); return normalized.map((a) => normalized.map((b) => 1 - dot(a, b))); }
function exactKnn(rows: number[][], k: number): number[][] { return cosineDistances(rows).map((row, i) => row.map((distance, index) => [distance, index]).filter((entry) => entry[1] !== i).sort((a, b) => a[0] - b[0]).slice(0, k).map((entry) => entry[1])); }
function minimumSpanningTree(rows: number[][], k: number): Array<[number, number, number]> {
  const distances = cosineDistances(rows); const edges: Array<[number, number, number]> = []; for (let i = 0; i < rows.length; i++) exactKnn(rows, k)[i].forEach((j) => edges.push([i, j, distances[i][j]]));
  edges.sort((a, b) => a[2] - b[2]); const parent = Array.from({ length: rows.length }, (_, i) => i); const find = (x: number): number => parent[x] === x ? x : (parent[x] = find(parent[x]));
  const result: Array<[number, number, number]> = []; for (const edge of edges) { const a = find(edge[0]); const b = find(edge[1]); if (a !== b) { parent[a] = b; result.push(edge); if (result.length === rows.length - 1) break; } } return result;
}
function deterministicSample(rows: number[][], size: number, seed: number): number[][] { const indices = Array.from({ length: rows.length }, (_, i) => i); let state = seed >>> 0; for (let i = indices.length - 1; i > 0; i--) { state = Math.imul(state ^ (state >>> 16), 2246822519) >>> 0; const j = state % (i + 1); [indices[i], indices[j]] = [indices[j], indices[i]]; } return indices.slice(0, size).map((i) => rows[i]); }
function seededRandom(seed: number): () => number { let state = seed >>> 0; return () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; }; }
function checkCancelled(options: ClusterOptions): void { if (options.signal?.cancelled) throw new Error("Clustering cancelled"); }

function jsPca(rows: number[][], components: number): { projected: number[][]; explained: number[] } {
  const n = rows.length; const dimensions = rows[0].length; const means = new Array(dimensions).fill(0); rows.forEach((row) => row.forEach((value, i) => means[i] += value / n));
  const centered = rows.map((row) => row.map((value, i) => value - means[i])); const covariance = Array.from({ length: dimensions }, () => new Array(dimensions).fill(0));
  for (const row of centered) for (let i = 0; i < dimensions; i++) for (let j = i; j < dimensions; j++) covariance[i][j] += row[i] * row[j] / Math.max(1, n - 1);
  for (let i = 0; i < dimensions; i++) for (let j = i + 1; j < dimensions; j++) covariance[j][i] = covariance[i][j];
  const vectors: number[][] = []; const values: number[] = []; for (let component = 0; component < Math.min(components, dimensions); component++) { let vector: number[] = Array.from({ length: dimensions }, (_, i) => i === component ? 1 : 0); for (let iteration = 0; iteration < 30; iteration++) { const next = covariance.map((row) => dot(row, vector)); const norm = Math.sqrt(dot(next, next)) || 1; vector = next.map((value) => value / norm); } const eigenvalue = Math.max(0, dot(vector, covariance.map((row) => dot(row, vector)))); values.push(eigenvalue); vectors.push(vector); for (let i = 0; i < dimensions; i++) for (let j = 0; j < dimensions; j++) covariance[i][j] -= eigenvalue * vector[i] * vector[j]; }
  return { projected: centered.map((row) => vectors.map((vector) => dot(row, vector))), explained: values };
}
