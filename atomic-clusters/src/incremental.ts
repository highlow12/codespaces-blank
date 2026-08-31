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

  if (!result.pca.model) return { mode: "full", reason: "pca_model_unavailable" };
  if (!embeddingSpaceKnown) return { mode: "full", reason: "embedding_space_metadata_unavailable" };
  if (result.incremental?.fullRebuildRecommended) return { mode: "full", reason: "previous_refresh_recommended_rebuild" };

  const provisionalCount = resultProvisionalPaths(result).length;
  if (active > 0 && provisionalCount > active * policy.maxProvisionalFraction) return { mode: "full", reason: "provisional_ratio_exceeded" };

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
  existingCoordinates: ReadonlyMap<string, number[]>;
  changedPaths: ReadonlySet<string>;
  deletedPaths?: ReadonlySet<string>;
  now?: string;
  provider: string;
  model: string;
  minSimilarity?: number;
}

export interface SoftRefreshOutput {
  result: ClusterResult;
  /** Paths whose PCA rows need to be written after this refresh. */
  projectedPaths: string[];
}

function clamp(value: number, min = 0, max = 1): number { return Math.max(min, Math.min(max, value)); }

function normalizeVector(values: readonly number[]): number[] {
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  return norm > 1e-12 ? values.map((value) => value / norm) : values.map(() => 0);
}

function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) return -1;
  const a = normalizeVector(left); const b = normalizeVector(right);
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
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
 * Reuse a fitted PCA and the existing hierarchy for a small change set.
 * This is intentionally a placement refresh, not an online HDBSCAN update.
 */
export function buildSoftRefresh(input: SoftRefreshInput): SoftRefreshOutput {
  const { result, notes, vectorsByPath, existingCoordinates, changedPaths } = input;
  const deletedPaths = input.deletedPaths || new Set<string>();
  const model = result.pca.model;
  if (!model) throw new Error("Soft refresh requires a saved PCA model");
  const leafOrder = (result.leafOrder || result.leafOrdering || result.hierarchy.leaves).slice();
  const leafIndex = new Map(leafOrder.map((leaf, index) => [leaf, index] as const));
  const oldIndex = new Map(result.ids.map((path, index) => [path, index] as const));
  const memberships = result.memberships || result.softMemberships;
  const previousProvisional = new Set(resultProvisionalPaths(result));
  const projected = new Map<string, number[]>();
  const projectedPaths: string[] = [];
  const activePaths = new Set(notes.map((note) => note.path));
  const isChanged = (path: string): boolean => changedPaths.has(path) || !oldIndex.has(path);

  for (const note of notes) {
    if (!isChanged(note.path)) {
      const coordinate = existingCoordinates.get(note.path);
      if (coordinate) projected.set(note.path, coordinate.slice());
    } else {
      const vector = vectorsByPath.get(note.path);
      if (!vector) throw new Error(`Missing embedding for changed note ${note.path}`);
      const coordinate = projectPcaVector(vector, model);
      projected.set(note.path, coordinate);
      projectedPaths.push(note.path);
    }
  }

  const centers = leafOrder.map(() => [] as number[]);
  const masses = leafOrder.map(() => 0);
  const addToCenters = (allowProvisional: boolean): void => {
    for (const [path, index] of oldIndex) {
      if (!activePaths.has(path) || deletedPaths.has(path) || isChanged(path)) continue;
      if (!allowProvisional && previousProvisional.has(path)) continue;
      const coordinate = projected.get(path) || existingCoordinates.get(path);
      if (!coordinate) continue;
      const row = memberships?.[index];
      if (row && row.length === leafOrder.length) {
        row.forEach((weight, column) => {
          const value = Math.max(0, Number(weight) || 0);
          if (!value) return;
          if (!centers[column].length) centers[column] = new Array(coordinate.length).fill(0);
          for (let dimension = 0; dimension < coordinate.length; dimension++) centers[column][dimension] += coordinate[dimension] * value;
          masses[column] += value;
        });
      } else {
        const label = result.leafLabels[index]; const column = leafIndex.get(label);
        if (column === undefined) return;
        if (!centers[column].length) centers[column] = new Array(coordinate.length).fill(0);
        for (let dimension = 0; dimension < coordinate.length; dimension++) centers[column][dimension] += coordinate[dimension];
        masses[column] += 1;
      }
    }
  };
  addToCenters(false);
  // A leaf with only provisional historical members is still better served by
  // its last known center than by treating every new note as root noise.
  if (masses.some((mass) => mass <= 1e-12)) {
    centers.forEach((center) => { center.length = 0; });
    masses.fill(0);
    addToCenters(true);
  }
  const normalizedCenters = centers.map((center, index) => masses[index] > 1e-12 ? center.map((value) => value / masses[index]) : center);

  const labels: number[] = []; const probabilities: number[] = []; const outliers: number[] = [];
  const nextMemberships: number[][] = []; const placements: HierarchyPlacement[] = []; const provisionalPaths = new Set<string>();
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

    const coordinate = projected.get(note.path);
    if (!coordinate) throw new Error(`Missing PCA projection for changed note ${note.path}`);
    let bestColumn = -1; let bestSimilarity = -1;
    normalizedCenters.forEach((center, column) => {
      if (!center.length || masses[column] <= 1e-12) return;
      const similarity = cosineSimilarity(coordinate, center);
      if (similarity > bestSimilarity || similarity === bestSimilarity && leafOrder[column] < leafOrder[bestColumn]) { bestSimilarity = similarity; bestColumn = column; }
    });
    const confidence = clamp(bestSimilarity);
    const accepted = bestColumn >= 0 && bestSimilarity >= (input.minSimilarity ?? 0.55);
    const label = accepted ? leafOrder[bestColumn] : -1;
    labels.push(label); probabilities.push(accepted ? confidence : 0); outliers.push(accepted ? 1 - confidence : 1);
    const row = new Array(leafOrder.length).fill(0); if (accepted) row[bestColumn] = confidence; nextMemberships.push(row);
    placements.push(accepted ? { kind: "leaf", nodeId: label, confidence } : { kind: "residual", nodeId: null, confidence: 0 });
    provisionalPaths.add(note.path);
  }

  const sortedProvisionalPaths = [...provisionalPaths].sort();
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
  const cumulativeThresholdExceeded = cumulativeChangedCount > notes.length * DEFAULT_POLICY.maxCumulativeChangedFraction;
  const fullRebuildRecommended = provisionalRatioExceeded || cumulativeThresholdExceeded || leafOccupancyDrift;
  const incremental: IncrementalRefreshMetadata = {
    mode: "soft",
    generatedAt,
    changedPaths: [...changedPaths].filter((path) => activePaths.has(path)).sort(),
    provisionalPaths: sortedProvisionalPaths,
    fullRebuildRecommended,
    reason: fullRebuildRecommended ? (provisionalRatioExceeded ? "provisional_ratio_reached" : cumulativeThresholdExceeded ? "cumulative_change_threshold_reached" : "leaf_occupancy_drift") : "small_change_reuse_saved_structure",
    cumulativeChangedCount,
    ...(previousIncremental?.lastFullRebuildAt ? { lastFullRebuildAt: previousIncremental.lastFullRebuildAt } : {}),
  };
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
