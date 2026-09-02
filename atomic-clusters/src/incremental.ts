import type {
  ClusterResult,
  HierarchyPlacement,
  NoteRecord,
  PcaModelArtifact,
  IncrementalRefreshMetadata,
} from "./types";

export interface PendingVaultChanges {
  created: Set<string>;
  modified: Set<string>;
  deleted: Set<string>;
  renamed: Map<string, string>;
  firstChangedAt: number;
  lastChangedAt: number;
}

export function createPendingVaultChanges(): PendingVaultChanges {
  return { created: new Set(), modified: new Set(), deleted: new Set(), renamed: new Map(), firstChangedAt: 0, lastChangedAt: 0 };
}

export function clonePendingVaultChanges(changes: PendingVaultChanges): PendingVaultChanges {
  return {
    created: new Set(changes.created),
    modified: new Set(changes.modified),
    deleted: new Set(changes.deleted),
    renamed: new Map(changes.renamed),
    firstChangedAt: changes.firstChangedAt,
    lastChangedAt: changes.lastChangedAt,
  };
}

export function pendingChangeCount(changes: PendingVaultChanges): number {
  return new Set([
    ...changes.created,
    ...changes.modified,
    ...changes.deleted,
    ...changes.renamed.keys(),
    ...changes.renamed.values(),
  ]).size;
}

/** Merge two snapshots without scheduling a refresh. */
export function mergePendingVaultChanges(target: PendingVaultChanges, source: PendingVaultChanges): void {
  for (const path of source.created) target.created.add(path);
  for (const path of source.modified) target.modified.add(path);
  for (const path of source.deleted) target.deleted.add(path);
  for (const [oldPath, newPath] of source.renamed) target.renamed.set(oldPath, newPath);
  if (source.firstChangedAt && (!target.firstChangedAt || source.firstChangedAt < target.firstChangedAt)) target.firstChangedAt = source.firstChangedAt;
  target.lastChangedAt = Math.max(target.lastChangedAt, source.lastChangedAt);
  // A later event wins over a stale earlier classification.
  for (const path of target.created) { target.modified.delete(path); target.deleted.delete(path); }
  for (const path of target.deleted) { target.created.delete(path); target.modified.delete(path); }
}

export interface VaultChangeQueueOptions {
  delayMs?: number;
  maxDelayMs?: number;
  now?: () => number;
  onReady?: () => void;
}

/**
 * Debounced, lossless event queue for Markdown vault changes.
 *
 * The queue deliberately separates `drain()` from the callback. A refresh
 * that is already running can therefore leave newly arrived events in a new
 * bucket, and a failed refresh can requeue only the snapshot it attempted.
 */
export class VaultChangeQueue {
  private pending = createPendingVaultChanges();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private delayMs: number;
  private readonly maxDelayMs: number;
  private readonly now: () => number;
  private readonly onReady?: () => void;

  constructor(options: VaultChangeQueueOptions = {}) {
    this.delayMs = Math.max(0, options.delayMs ?? 5000);
    this.maxDelayMs = Math.max(this.delayMs, options.maxDelayMs ?? 60000);
    this.now = options.now || (() => Date.now());
    this.onReady = options.onReady;
  }

  get hasChanges(): boolean { return pendingChangeCount(this.pending) > 0; }
  get size(): number { return pendingChangeCount(this.pending); }

  enqueueCreated(path: string): void {
    if (!path) return;
    this.pending.deleted.delete(path);
    this.pending.modified.delete(path);
    this.pending.created.add(path);
    this.touch();
  }

  enqueueModified(path: string): void {
    if (!path) return;
    this.pending.deleted.delete(path);
    if (!this.pending.created.has(path)) this.pending.modified.add(path);
    this.touch();
  }

  enqueueDeleted(path: string): void {
    if (!path) return;
    this.pending.created.delete(path);
    this.pending.modified.delete(path);
    for (const [oldPath, newPath] of this.pending.renamed) {
      if (oldPath === path || newPath === path) this.pending.renamed.delete(oldPath);
    }
    this.pending.deleted.add(path);
    this.touch();
  }

  enqueueRenamed(oldPath: string, newPath: string): void {
    if (!oldPath || !newPath) return;
    if (oldPath === newPath) { this.enqueueModified(newPath); return; }
    // Collapse a rename chain (a.md -> b.md -> c.md) into its final target so
    // one refresh never leaves an intermediate path in the result.
    for (const [source, target] of this.pending.renamed) if (target === oldPath) {
      this.pending.renamed.delete(source);
      oldPath = source;
    }
    const wasCreated = this.pending.created.delete(oldPath);
    const wasModified = this.pending.modified.delete(oldPath);
    this.pending.deleted.delete(oldPath);
    this.pending.created.delete(newPath);
    this.pending.modified.delete(newPath);
    this.pending.deleted.delete(newPath);
    this.pending.renamed.set(oldPath, newPath);
    if (wasCreated) this.pending.created.add(newPath);
    else if (wasModified) this.pending.modified.add(newPath);
    this.touch();
  }

  peek(): PendingVaultChanges | null { return this.hasChanges ? clonePendingVaultChanges(this.pending) : null; }

  drain(): PendingVaultChanges | null {
    if (!this.hasChanges) return null;
    const snapshot = clonePendingVaultChanges(this.pending);
    this.pending = createPendingVaultChanges();
    this.clearTimer();
    return snapshot;
  }

  requeue(changes: PendingVaultChanges): void {
    if (!pendingChangeCount(changes)) return;
    mergePendingVaultChanges(this.pending, changes);
  }

  clear(): void { this.pending = createPendingVaultChanges(); this.clearTimer(); }

  setDelay(delayMs: number): void {
    this.delayMs = Math.max(0, Math.min(this.maxDelayMs, delayMs));
    if (this.hasChanges) this.schedule();
  }

  /** Ask the consumer to process the current bucket immediately. */
  notifyReady(): void { if (this.hasChanges) this.onReady?.(); }

  dispose(): void { this.clearTimer(); }

  private touch(): void {
    const timestamp = this.now();
    if (!this.pending.firstChangedAt) this.pending.firstChangedAt = timestamp;
    this.pending.lastChangedAt = timestamp;
    this.schedule();
  }

  private schedule(): void {
    this.clearTimer();
    const elapsed = Math.max(0, this.now() - this.pending.firstChangedAt);
    const wait = Math.min(this.delayMs, Math.max(0, this.maxDelayMs - elapsed));
    this.timer = setTimeout(() => { this.timer = undefined; this.onReady?.(); }, wait);
  }

  private clearTimer(): void { if (this.timer !== undefined) clearTimeout(this.timer); this.timer = undefined; }
}

export interface IncrementalRefreshPolicy {
  softChangedFraction?: number;
  softChangedFloor?: number;
  maxDeletedFraction?: number;
  maxProvisionalFraction?: number;
  maxOutOfDistributionFraction?: number;
  maxCumulativeChangedFraction?: number;
  fullRebuildAfterMs?: number;
}

export interface IncrementalRefreshDecisionInput {
  result: ClusterResult | null;
  activeNoteCount: number;
  changedNoteCount: number;
  deletedNoteCount: number;
  /** True when events changed paths/metadata but not note content. */
  pathOnly: boolean;
  provider: string;
  model: string;
  now?: number;
  policy?: IncrementalRefreshPolicy;
}

export interface IncrementalRefreshDecision {
  mode: "no-op" | "soft" | "full";
  reason: string;
}

const DEFAULT_POLICY: Required<IncrementalRefreshPolicy> = {
  softChangedFraction: 0.02,
  softChangedFloor: 20,
  maxDeletedFraction: 0.01,
  maxProvisionalFraction: 0.05,
  maxOutOfDistributionFraction: 0.05,
  maxCumulativeChangedFraction: 0.10,
  fullRebuildAfterMs: 7 * 24 * 60 * 60 * 1000,
};

function policyOf(policy?: IncrementalRefreshPolicy): Required<IncrementalRefreshPolicy> {
  return { ...DEFAULT_POLICY, ...(policy || {}) };
}

function resultProvisionalPaths(result: ClusterResult): string[] {
  return [...new Set(result.provisionalPaths || result.incremental?.provisionalPaths || [])];
}

export function decideIncrementalRefresh(input: IncrementalRefreshDecisionInput): IncrementalRefreshDecision {
  const policy = policyOf(input.policy);
  const active = Math.max(0, input.activeNoteCount);
  const result = input.result;
  if (!result) return { mode: "full", reason: "no_saved_result" };

  const embeddingSpaceKnown = !!result.embeddingProvider && !!result.embeddingModel;
  const embeddingSpaceChanged = embeddingSpaceKnown && (result.embeddingProvider !== input.provider || result.embeddingModel !== input.model);
  if (embeddingSpaceChanged) return { mode: "full", reason: "embedding_model_changed" };

  if (input.changedNoteCount === 0 && input.deletedNoteCount === 0) {
    // A path-only rename can safely reuse the existing result even for a
    // legacy result that predates embedding-space metadata. A model change
    // with content changes is still caught below as a conservative full run.
    return { mode: "no-op", reason: input.pathOnly ? "path_only_change" : "metadata_unchanged" };
  }

  if (!result.umap || result.umap.coordinates.length !== result.ids.length) return { mode: "full", reason: "umap_coordinates_unavailable" };
  if (!result.pca.model) return { mode: "full", reason: "pca_model_unavailable" };
  if (!embeddingSpaceKnown) return { mode: "full", reason: "embedding_space_metadata_unavailable" };
  if (result.incremental?.fullRebuildRecommended) return { mode: "full", reason: "previous_refresh_recommended_rebuild" };

  const provisionalCount = resultProvisionalPaths(result).length;
  if (active > 0 && provisionalCount > active * policy.maxProvisionalFraction) return { mode: "full", reason: "provisional_ratio_exceeded" };
  const outOfDistributionCount = result.incremental?.outOfDistributionPaths?.length || 0;
  if (active > 0 && outOfDistributionCount > active * policy.maxOutOfDistributionFraction) return { mode: "full", reason: "umap_ood_ratio_exceeded" };

  const changedLimit = Math.max(policy.softChangedFloor, Math.ceil(active * policy.softChangedFraction));
  if (input.changedNoteCount > changedLimit) return { mode: "full", reason: "changed_note_threshold_exceeded" };
  if (active > 0 && input.deletedNoteCount > active * policy.maxDeletedFraction) return { mode: "full", reason: "deleted_note_threshold_exceeded" };

  const cumulative = (result.incremental?.cumulativeChangedCount || 0) + input.changedNoteCount + input.deletedNoteCount;
  if (active > 0 && cumulative > active * policy.maxCumulativeChangedFraction) return { mode: "full", reason: "cumulative_change_threshold_exceeded" };

  const lastFull = result.incremental?.lastFullRebuildAt ? Date.parse(result.incremental.lastFullRebuildAt) : NaN;
  if (Number.isFinite(lastFull) && (input.now ?? Date.now()) - lastFull >= policy.fullRebuildAfterMs) return { mode: "full", reason: "full_rebuild_age_exceeded" };

  return { mode: "soft", reason: "small_change_reuse_saved_structure" };
}

export interface SoftRefreshInput {
  result: ClusterResult;
  notes: NoteRecord[];
  vectorsByPath: ReadonlyMap<string, number[]>;
  /** Saved PCA coordinates keyed by note path. */
  existingCoordinates: ReadonlyMap<string, number[]>;
  /** Saved full-build UMAP coordinates keyed by note path. */
  existingUmapCoordinates: ReadonlyMap<string, number[]>;
  changedPaths: ReadonlySet<string>;
  deletedPaths?: ReadonlySet<string>;
  now?: string;
  provider: string;
  model: string;
  /** Minimum weighted UMAP-neighbour support required to admit a leaf. */
  minSupport?: number;
  /** Minimum distance-adjusted probability required to admit a leaf. */
  minProbability?: number;
  /** Multiplier applied to the saved per-leaf p95 kNN-distance gate. */
  maxDistanceFactor?: number;
  /** @deprecated retained only for callers compiled against PR #3. */
  minSimilarity?: number;
}

export interface SoftRefreshOutput {
  result: ClusterResult;
  /** Paths whose PCA rows need to be written after this refresh. */
  projectedPaths: string[];
}

function clamp(value: number, min = 0, max = 1): number { return Math.max(min, Math.min(max, value)); }

function euclideanDistance(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) return Number.POSITIVE_INFINITY;
  let squared = 0;
  for (let index = 0; index < left.length; index++) { const delta = left[index] - right[index]; squared += delta * delta; }
  return Math.sqrt(squared);
}

interface NearestPoint { path: string; distance: number; }

function nearestPoints(point: readonly number[], paths: readonly string[], coordinates: ReadonlyMap<string, number[]>, limit: number): NearestPoint[] {
  const nearest = paths.map((path) => ({ path, distance: euclideanDistance(point, coordinates.get(path) || []) }));
  nearest.sort((left, right) => left.distance - right.distance || left.path.localeCompare(right.path));
  return nearest.slice(0, Math.max(1, Math.min(limit, nearest.length)));
}

/** Apply the saved PCA-to-UMAP transform seam without fitting UMAP again. */
function transformWithSavedUmap(
  pcaCoordinate: readonly number[],
  sourcePaths: readonly string[],
  pcaCoordinates: ReadonlyMap<string, number[]>,
  umapCoordinates: ReadonlyMap<string, number[]>,
  neighbors: number,
): number[] {
  const nearest = nearestPoints(pcaCoordinate, sourcePaths, pcaCoordinates, neighbors);
  const dimensions = umapCoordinates.get(nearest[0]?.path || "")?.length || 0;
  if (!dimensions) throw new Error("Saved UMAP coordinates are empty");
  const weights = nearest.map(({ distance }) => 1 / Math.max(1e-6, distance));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const transformed = new Array(dimensions).fill(0);
  nearest.forEach(({ path }, index) => {
    const coordinate = umapCoordinates.get(path);
    if (!coordinate || coordinate.length !== dimensions) throw new Error("Saved UMAP coordinates are not rectangular");
    const weight = weights[index] / total;
    for (let dimension = 0; dimension < dimensions; dimension++) transformed[dimension] += coordinate[dimension] * weight;
  });
  return transformed;
}

interface UmapVoteResult {
  label: number;
  probability: number;
  outlier: number;
  memberships: number[];
  support: number;
  distance: number;
  outOfDistribution: boolean;
}

/**
 * Assign a transformed point against the saved UMAP neighbourhood. Votes are
 * weighted by both HDBSCAN probability and UMAP distance. A candidate must
 * have majority support, useful distance-adjusted probability, and remain
 * within the saved per-leaf p95 kNN-distance envelope; otherwise it remains
 * provisional noise instead of being forced into a leaf.
 */
function assignInSavedUmap(
  coordinate: readonly number[],
  sourcePaths: readonly string[],
  umapCoordinates: ReadonlyMap<string, number[]>,
  result: ClusterResult,
  leafOrder: readonly number[],
  leafIndex: ReadonlyMap<number, number>,
  neighbors: number,
  p95ByLeaf: Readonly<Record<string, number>>,
  minSupport: number,
  minProbability: number,
  maxDistanceFactor: number,
): UmapVoteResult {
  const nearest = nearestPoints(coordinate, sourcePaths, umapCoordinates, neighbors);
  const oldIndex = new Map(result.ids.map((path, index) => [path, index] as const));
  const votes = new Map<number, { weight: number; weightedDistance: number }>();
  let totalWeight = 0;
  for (const candidate of nearest) {
    const index = oldIndex.get(candidate.path);
    if (index === undefined) continue;
    const probability = clamp(Number(result.probabilities[index]) || 0);
    const weight = (1 / (1 + candidate.distance)) * Math.max(0.05, probability);
    totalWeight += weight;
    const label = result.leafLabels[index];
    if (label >= 0) {
      const current = votes.get(label) || { weight: 0, weightedDistance: 0 };
      current.weight += weight; current.weightedDistance += weight * candidate.distance; votes.set(label, current);
    }
  }
  const best = [...votes.entries()].sort((left, right) => right[1].weight - left[1].weight || left[0] - right[0])[0];
  const bestLabel = best?.[0] ?? -1;
  const bestVote = best?.[1];
  const support = bestVote && totalWeight > 0 ? bestVote.weight / totalWeight : 0;
  const distance = bestVote && bestVote.weight > 0 ? bestVote.weightedDistance / bestVote.weight : Number.POSITIVE_INFINITY;
  const p95 = bestLabel >= 0 ? Number(p95ByLeaf[String(bestLabel)]) : Number.POSITIVE_INFINITY;
  const finiteP95 = Number.isFinite(p95);
  const distanceLimit = finiteP95 ? Math.max(1e-6, p95 * maxDistanceFactor) : Number.POSITIVE_INFINITY;
  const distanceConfidence = !Number.isFinite(distance) ? 0 : finiteP95 ? clamp(1 - distance / distanceLimit) : clamp(1 / (1 + distance));
  const probability = clamp(support * distanceConfidence);
  const accepted = bestLabel >= 0 && support >= minSupport && probability >= minProbability && distance <= distanceLimit;
  const memberships = new Array(leafOrder.length).fill(0);
  if (totalWeight > 0) for (const [label, vote] of votes) {
    const column = leafIndex.get(label); if (column !== undefined) memberships[column] = probability * vote.weight / totalWeight;
  }
  return {
    label: accepted ? bestLabel : -1,
    probability: accepted ? probability : 0,
    outlier: accepted ? 1 - probability : 1,
    memberships,
    support,
    distance,
    outOfDistribution: !accepted,
  };
}

export function projectPcaVector(vector: readonly number[], model: PcaModelArtifact): number[] {
  if (vector.length !== model.inputDimension) throw new Error(`PCA input dimension mismatch: expected ${model.inputDimension}, got ${vector.length}`);
  const input = vector.slice();
  if (model.normalization === "l2") {
    const norm = Math.sqrt(input.reduce((sum, value) => sum + value * value, 0));
    if (norm > 1e-12) for (let index = 0; index < input.length; index++) input[index] /= norm;
  }
  const centered = input.map((value, index) => value - model.mean[index]);
  return model.components.map((component) => {
    if (component.length !== model.inputDimension) throw new Error("PCA component dimension mismatch");
    return component.reduce((sum, value, index) => sum + value * centered[index], 0);
  });
}

function fallbackPlacement(label: number, probability: number): HierarchyPlacement {
  return label >= 0 ? { kind: "leaf", nodeId: label, confidence: clamp(probability) } : { kind: "residual", nodeId: null, confidence: 0 };
}

function recountHierarchyMasses(result: ClusterResult, placements: readonly HierarchyPlacement[]): ClusterResult["hierarchy"] {
  const hierarchy = result.hierarchy;
  if (!hierarchy.nodes) return hierarchy;
  const parent = new Map<number, number>();
  for (const node of hierarchy.nodes) for (const child of node.children) parent.set(child, node.id);
  const masses = new Map<number, number>(hierarchy.nodes.map((node) => [node.id, 0] as const));
  for (const placement of placements) {
    if (placement.nodeId === null) continue;
    const seen = new Set<number>(); let current: number | undefined = placement.nodeId;
    while (current !== undefined && !seen.has(current)) {
      seen.add(current); masses.set(current, (masses.get(current) || 0) + 1); current = parent.get(current);
    }
  }
  return {
    ...hierarchy,
    merges: hierarchy.merges.map((merge) => ({ ...merge, mass: masses.get(merge.id) || 0 })),
    nodes: hierarchy.nodes.map((node) => ({ ...node, mass: masses.get(node.id) || 0 }))
  };
}

/**
 * Reuse the fitted PCA and saved UMAP coordinates for a small change set.
 * This is a placement refresh, not an online HDBSCAN update: changed points
 * are first projected with the saved PCA-to-UMAP kNN transform, then voted
 * into the existing UMAP space. The structural hierarchy is never refit.
 */
export function buildSoftRefresh(input: SoftRefreshInput): SoftRefreshOutput {
  const { result, notes, vectorsByPath, existingCoordinates, existingUmapCoordinates, changedPaths } = input;
  const deletedPaths = input.deletedPaths || new Set<string>();
  const model = result.pca.model;
  if (!model) throw new Error("Soft refresh requires a saved PCA model");
  const savedUmap = result.umap;
  if (!savedUmap || savedUmap.coordinates.length !== result.ids.length) throw new Error("Soft refresh requires saved full-build UMAP coordinates");
  if (savedUmap.sourcePcaModelHash && savedUmap.sourcePcaModelHash !== model.modelHash) throw new Error("Saved UMAP transform does not match the PCA model");
  const leafOrder = (result.leafOrder || result.leafOrdering || result.hierarchy.leaves).slice();
  const leafIndex = new Map(leafOrder.map((leaf, index) => [leaf, index] as const));
  const oldIndex = new Map(result.ids.map((path, index) => [path, index] as const));
  const memberships = result.memberships || result.softMemberships;
  const previousProvisional = new Set(resultProvisionalPaths(result));
  const previousOutOfDistribution = new Set(result.incremental?.outOfDistributionPaths || []);
  const projected = new Map<string, number[]>();
  const projectedUmap = new Map<string, number[]>();
  const projectedPaths: string[] = [];
  const activePaths = new Set(notes.map((note) => note.path));
  const isChanged = (path: string): boolean => changedPaths.has(path) || !oldIndex.has(path);

  for (const note of notes) {
    if (!isChanged(note.path)) {
      const coordinate = existingCoordinates.get(note.path);
      const umapCoordinate = existingUmapCoordinates.get(note.path);
      if (!coordinate || !umapCoordinate) throw new Error(`Saved PCA/UMAP coordinates are incomplete for ${note.path}`);
      projected.set(note.path, coordinate.slice());
      projectedUmap.set(note.path, umapCoordinate.slice());
    } else {
      const vector = vectorsByPath.get(note.path);
      if (!vector) throw new Error(`Missing embedding for changed note ${note.path}`);
      const coordinate = projectPcaVector(vector, model);
      projected.set(note.path, coordinate);
      projectedPaths.push(note.path);
    }
  }

  // Only stable, unchanged rows may vote. This prevents a stream of
  // provisional placements from drifting the saved structure before a full
  // rebuild is scheduled.
  const sourcePaths = result.ids.filter((path) => activePaths.has(path) && !deletedPaths.has(path) && !isChanged(path) && !previousProvisional.has(path) && existingCoordinates.has(path) && existingUmapCoordinates.has(path));
  if (sourcePaths.length < 2) throw new Error("Soft refresh requires at least two unchanged UMAP anchor notes");
  const transformNeighbors = Math.max(1, Math.min(savedUmap.transform?.neighbors || savedUmap.nNeighbors || 15, sourcePaths.length));
  const voteNeighbors = Math.max(1, Math.min(savedUmap.nNeighbors || 15, sourcePaths.length));

  const labels: number[] = []; const probabilities: number[] = []; const outliers: number[] = [];
  const nextMemberships: number[][] = []; const placements: HierarchyPlacement[] = []; const provisionalPaths = new Set<string>();
  // Carry forward unresolved OOD notes that are still active. A subsequent
  // refresh must measure growth across refreshes, rather than forgetting an
  // earlier OOD row when no new note is changed in the current batch.
  const outOfDistributionPaths = new Set([...previousOutOfDistribution].filter((path) => activePaths.has(path) && !deletedPaths.has(path) && !isChanged(path)));
  const knnSupport: Record<string, number> = {}; const knnDistance: Record<string, number> = {};
  for (const note of notes) {
    const index = oldIndex.get(note.path);
    const changed = isChanged(note.path);
    if (index !== undefined && !changed) {
      labels.push(result.leafLabels[index]); probabilities.push(result.probabilities[index]); outliers.push(result.outlierProxy[index]);
      const row = memberships?.[index]?.slice() || leafOrder.map((leaf) => leaf === result.leafLabels[index] ? result.probabilities[index] : 0);
      nextMemberships.push(row);
      placements.push(result.hierarchyPlacements?.[index] ? { ...result.hierarchyPlacements[index] } : fallbackPlacement(result.leafLabels[index], result.probabilities[index]));
      if (previousProvisional.has(note.path)) provisionalPaths.add(note.path);
      continue;
    }

    const pcaCoordinate = projected.get(note.path);
    if (!pcaCoordinate) throw new Error(`Missing PCA projection for changed note ${note.path}`);
    const umapCoordinate = transformWithSavedUmap(pcaCoordinate, sourcePaths, existingCoordinates, existingUmapCoordinates, transformNeighbors);
    projectedUmap.set(note.path, umapCoordinate);
    const vote = assignInSavedUmap(
      umapCoordinate, sourcePaths, existingUmapCoordinates, result, leafOrder, leafIndex, voteNeighbors,
      savedUmap.leafKnnDistanceP95, input.minSupport ?? 0.55, input.minProbability ?? 0.35, input.maxDistanceFactor ?? 1.5
    );
    labels.push(vote.label); probabilities.push(vote.probability); outliers.push(vote.outlier); nextMemberships.push(vote.memberships);
    placements.push(vote.label >= 0 ? { kind: "leaf", nodeId: vote.label, confidence: vote.probability } : { kind: "residual", nodeId: null, confidence: 0 });
    provisionalPaths.add(note.path);
    knnSupport[note.path] = vote.support; knnDistance[note.path] = vote.distance;
    if (vote.outOfDistribution) outOfDistributionPaths.add(note.path);
  }

  const sortedProvisionalPaths = [...provisionalPaths].sort();
  const sortedOutOfDistributionPaths = [...outOfDistributionPaths].sort();
  const previousIncremental = result.incremental;
  const generatedAt = input.now || new Date().toISOString();
  const cumulativeChangedCount = (previousIncremental?.cumulativeChangedCount || 0) + changedPaths.size + deletedPaths.size;
  const previousOccupancy = new Map<number, number>();
  const nextOccupancy = new Map<number, number>();
  for (const label of result.leafLabels) if (label >= 0) previousOccupancy.set(label, (previousOccupancy.get(label) || 0) + 1);
  for (const label of labels) if (label >= 0) nextOccupancy.set(label, (nextOccupancy.get(label) || 0) + 1);
  const leafOccupancyDrift = leafOrder.some((leaf) => {
    const before = previousOccupancy.get(leaf) || 0;
    const after = nextOccupancy.get(leaf) || 0;
    return before > 0 && Math.abs(after - before) / before > 0.30;
  });
  const provisionalRatioExceeded = notes.length > 0 && sortedProvisionalPaths.length > notes.length * DEFAULT_POLICY.maxProvisionalFraction;
  const oodRatioExceeded = notes.length > 0 && sortedOutOfDistributionPaths.length > notes.length * DEFAULT_POLICY.maxOutOfDistributionFraction;
  const cumulativeThresholdExceeded = cumulativeChangedCount > notes.length * DEFAULT_POLICY.maxCumulativeChangedFraction;
  const fullRebuildRecommended = provisionalRatioExceeded || oodRatioExceeded || cumulativeThresholdExceeded || leafOccupancyDrift;
  const incremental: IncrementalRefreshMetadata = {
    mode: "soft",
    generatedAt,
    changedPaths: [...changedPaths].filter((path) => activePaths.has(path)).sort(),
    provisionalPaths: sortedProvisionalPaths,
    fullRebuildRecommended,
    reason: fullRebuildRecommended ? (oodRatioExceeded ? "umap_ood_growth" : provisionalRatioExceeded ? "provisional_ratio_reached" : cumulativeThresholdExceeded ? "cumulative_change_threshold_reached" : "leaf_occupancy_drift") : "small_change_reuse_saved_structure",
    cumulativeChangedCount,
    ...(sortedOutOfDistributionPaths.length ? { outOfDistributionPaths: sortedOutOfDistributionPaths } : {}),
    ...(Object.keys(knnSupport).length ? { knnSupport, knnDistance } : {}),
    ...(previousIncremental?.lastFullRebuildAt ? { lastFullRebuildAt: previousIncremental.lastFullRebuildAt } : {}),
  };
  const umapCoordinates = notes.map((note) => projectedUmap.get(note.path));
  if (umapCoordinates.some((coordinate) => !coordinate)) throw new Error("Incremental UMAP coordinates are incomplete");
  const next: ClusterResult = {
    ...result,
    embeddingProvider: input.provider,
    embeddingModel: input.model,
    pca: { ...result.pca, model: { ...model, provider: input.provider, model: input.model } },
    ids: notes.map((note) => note.path),
    leafLabels: labels,
    probabilities,
    outlierProxy: outliers,
    softMemberships: nextMemberships,
    memberships: nextMemberships,
    hierarchyPlacements: placements,
    hierarchy: recountHierarchyMasses({ ...result, hierarchyPlacements: placements }, placements),
    umap: { ...savedUmap, coordinates: umapCoordinates as number[][] },
    visualization: undefined,
    provisionalPaths: sortedProvisionalPaths,
    incremental,
    timings: { ...result.timings },
  };
  return { result: next, projectedPaths };
}

/** Apply a path-only rename while keeping row order and visualization intact. */
export function renameClusterResultPaths(result: ClusterResult, renames: ReadonlyMap<string, string>): ClusterResult {
  if (!renames.size) return result;
  const mapPath = (path: string): string => {
    let current = path; const seen = new Set<string>();
    while (renames.has(current) && !seen.has(current)) { seen.add(current); current = renames.get(current)!; }
    return current;
  };
  const incremental = result.incremental ? {
    ...result.incremental,
    changedPaths: result.incremental.changedPaths.map(mapPath),
    provisionalPaths: result.incremental.provisionalPaths.map(mapPath),
    ...(result.incremental.outOfDistributionPaths ? { outOfDistributionPaths: result.incremental.outOfDistributionPaths.map(mapPath) } : {}),
    ...(result.incremental.knnSupport ? { knnSupport: Object.fromEntries(Object.entries(result.incremental.knnSupport).map(([path, value]) => [mapPath(path), value])) } : {}),
    ...(result.incremental.knnDistance ? { knnDistance: Object.fromEntries(Object.entries(result.incremental.knnDistance).map(([path, value]) => [mapPath(path), value])) } : {}),
  } : undefined;
  const provisionalPaths = result.provisionalPaths?.map(mapPath);
  return {
    ...result,
    ids: result.ids.map(mapPath),
    ...(provisionalPaths ? { provisionalPaths } : {}),
    ...(incremental ? { incremental } : {}),
  };
}

export function markFullRebuildResult(result: ClusterResult, provider: string, model: string, now = new Date().toISOString()): ClusterResult {
  return {
    ...result,
    pca: result.pca.model ? { ...result.pca, model: { ...result.pca.model, provider, model } } : result.pca,
    embeddingProvider: provider,
    embeddingModel: model,
    provisionalPaths: [],
    incremental: {
      mode: "full",
      generatedAt: now,
      changedPaths: [],
      provisionalPaths: [],
      fullRebuildRecommended: false,
      cumulativeChangedCount: 0,
      lastFullRebuildAt: now,
    },
  };
}
