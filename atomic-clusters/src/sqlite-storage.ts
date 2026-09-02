/**
 * Durable, self contained storage for Atomic Clusters.
 *
 * The module intentionally talks to the small part of the sql.js API used by
 * the plugin.  This keeps the database usable in Node tests and makes the
 * sql.js initialiser (and its wasm asset path) an explicit application concern.
 */
import { contentHash } from "./hash";
import { CachedEmbedding, ClusterResult, ClusterVisualization, EmbeddingRunLog, HierarchyPlacement, NoteRecord } from "./types";

export interface SqlValue { readonly [key: string]: unknown; }
export interface SqlStatement {
  bind(values?: unknown[] | Record<string, unknown>): void;
  step(): boolean;
  getAsObject(): Record<string, unknown>;
  free(): void;
}
export interface SqlDatabase {
  run(sql: string, params?: unknown[] | Record<string, unknown>): void;
  prepare(sql: string): SqlStatement;
  exec(sql: string): Array<{ columns: string[]; values: unknown[][] }>;
  export(): Uint8Array;
  close(): void;
}
export interface SqlJsStatic { Database: new (data?: ArrayLike<number>) => SqlDatabase; }

export interface BinaryAdapter {
  readBinary(path: string): Promise<ArrayBuffer>;
  writeBinary(path: string, data: ArrayBuffer): Promise<void>;
  exists(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
  rename?(oldPath: string, newPath: string): Promise<void>;
  remove?(path: string): Promise<void>;
}

export const SQLITE_PATH = ".obsidian/plugins/atomic-clusters/atomic-clusters.sqlite";
export const SQLITE_TEMP_PATH = `${SQLITE_PATH}.tmp`;
export const SQLITE_SCHEMA_VERSION = 6;

const SCHEMA = `
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
  path TEXT PRIMARY KEY, title TEXT NOT NULL, mtime INTEGER NOT NULL,
  content_hash TEXT NOT NULL, content TEXT, active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS embeddings (
  path TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  embedding_hash TEXT NOT NULL, note_content_hash TEXT NOT NULL, dimension INTEGER NOT NULL, vector_json TEXT, vector_blob BLOB,
  created_at TEXT NOT NULL, PRIMARY KEY(path, provider, model, embedding_hash),
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS embeddings_current ON embeddings(path, provider, model, created_at DESC);
CREATE TABLE IF NOT EXISTS pca_models (
  model_hash TEXT PRIMARY KEY, provider TEXT, model TEXT, input_dimension INTEGER NOT NULL,
  output_dimension INTEGER NOT NULL, normalization TEXT NOT NULL, mean_json TEXT,
  components_json TEXT, explained_variance_json TEXT,
  mean_blob BLOB, components_blob BLOB, explained_variance_blob BLOB,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pca_coordinates (
  path TEXT NOT NULL, model_hash TEXT NOT NULL, coordinates_json TEXT, coordinate_blob BLOB,
  PRIMARY KEY(path, model_hash), FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE,
  FOREIGN KEY(model_hash) REFERENCES pca_models(model_hash) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS results (
  result_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, created_at TEXT NOT NULL,
  result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS result_note_hashes (
  result_id TEXT NOT NULL, path TEXT NOT NULL, content_hash TEXT NOT NULL,
  PRIMARY KEY(result_id, path),
  FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS assignments (
  result_id TEXT NOT NULL, path TEXT NOT NULL, leaf_label INTEGER NOT NULL,
  probability REAL NOT NULL, outlier_score REAL NOT NULL, provisional INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(result_id, path),
  FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS hierarchy_merges (
  result_id TEXT NOT NULL, id INTEGER NOT NULL, left_id INTEGER NOT NULL,
  right_id INTEGER NOT NULL, distance REAL NOT NULL, mass INTEGER NOT NULL,
  PRIMARY KEY(result_id, id), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS hierarchy_leaves (
  result_id TEXT NOT NULL, ordinal INTEGER NOT NULL, leaf_id INTEGER NOT NULL,
  PRIMARY KEY(result_id, ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS leaf_order (
  result_id TEXT NOT NULL, ordinal INTEGER NOT NULL, leaf_id INTEGER NOT NULL,
  PRIMARY KEY(result_id, ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS visualization_points (
  result_id TEXT NOT NULL, ordinal INTEGER NOT NULL, path TEXT NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, leaf_label INTEGER NOT NULL,
  PRIMARY KEY(result_id, ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS cluster_titles (
  result_id TEXT NOT NULL, node_id INTEGER NOT NULL, title TEXT NOT NULL,
  PRIMARY KEY(result_id, node_id), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS soft_memberships (
  result_id TEXT NOT NULL, path TEXT NOT NULL, leaf_id INTEGER NOT NULL, membership REAL NOT NULL,
  PRIMARY KEY(result_id, path, leaf_id), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS membership_rows (
  result_id TEXT NOT NULL, path TEXT NOT NULL, width INTEGER NOT NULL, values_blob BLOB NOT NULL,
  PRIMARY KEY(result_id, path), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS hierarchy_nodes (
  result_id TEXT NOT NULL, node_id INTEGER NOT NULL, distance REAL NOT NULL, mass REAL NOT NULL,
  descendant_leaves_blob BLOB NOT NULL, PRIMARY KEY(result_id,node_id),
  FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS hierarchy_children (
  result_id TEXT NOT NULL, parent_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, child_id INTEGER NOT NULL,
  PRIMARY KEY(result_id,parent_id,ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS root_children (
  result_id TEXT NOT NULL, ordinal INTEGER NOT NULL, child_id INTEGER NOT NULL,
  PRIMARY KEY(result_id,ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS hierarchy_placements (
  result_id TEXT NOT NULL, path TEXT NOT NULL, ordinal INTEGER NOT NULL, kind TEXT NOT NULL,
  node_id INTEGER, confidence REAL NOT NULL, PRIMARY KEY(result_id,path),
  FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS embedding_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, completed_at TEXT NOT NULL,
  provider TEXT NOT NULL, model TEXT NOT NULL, status TEXT, log_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS migrations (
  name TEXT PRIMARY KEY, completed_at TEXT NOT NULL
);
CREATE VIEW IF NOT EXISTS v_current_embeddings AS
  SELECT e.path, n.title, n.mtime, n.content_hash, e.provider, e.model,
    e.embedding_hash, e.note_content_hash, e.dimension, e.vector_json, e.created_at
  FROM embeddings e JOIN notes n USING(path)
  WHERE e.rowid = (SELECT MAX(e2.rowid) FROM embeddings e2
                   WHERE e2.path=e.path AND e2.provider=e.provider AND e2.model=e.model)
    AND n.active=1;
CREATE VIEW IF NOT EXISTS v_note_pca AS
  SELECT c.path, n.title, n.mtime, n.content_hash, c.model_hash, c.coordinates_json
  FROM pca_coordinates c JOIN notes n USING(path);
CREATE VIEW IF NOT EXISTS v_cluster_assignments AS
  SELECT a.result_id, a.path, n.title, a.leaf_label, a.probability, a.outlier_score
  FROM assignments a JOIN notes n USING(path);
CREATE VIEW IF NOT EXISTS v_embedding_log AS
  SELECT id, started_at, completed_at, provider, model, status, log_json FROM embedding_logs;
`;

export interface PcaModel {
  modelHash: string;
  inputDimension: number;
  outputDimension: number;
  /** One mean value per input dimension. */
  mean: number[];
  /** Components are rows: outputDimension × inputDimension. */
  components: number[][];
  explainedVariance: number[];
  normalization: "l2" | "none" | string;
  provider?: string;
  model?: string;
}

/** Compact little-endian Float32 storage used for all large numeric rows. */
export function float32Blob(values: readonly number[]): Uint8Array {
  const bytes = new Uint8Array(values.length * 4); const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, Number(value) || 0, true)); return bytes;
}
export function float32Values(value: unknown): number[] {
  if (value instanceof Uint8Array || value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    const bytes = value instanceof Uint8Array ? value : value instanceof ArrayBuffer ? new Uint8Array(value) : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength); const result: number[] = [];
    for (let index = 0; index + 4 <= bytes.byteLength; index += 4) result.push(view.getFloat32(index, true)); return result;
  }
  return typeof value === "string" ? JSON.parse(value) : Array.isArray(value) ? value.map(Number) : [];
}
function flattenNumbers(rows: readonly number[][]): number[] { return rows.reduce((all, row) => all.concat(row), []); }
function flattenJsonNumbers(value: unknown): number[] { return Array.isArray(value) ? value.flatMap(flattenJsonNumbers) : [Number(value) || 0]; }

/**
 * Convert the last pre-v6 JSON result without importing the clustering module.
 * Storage is loaded by the worker and importing clustering here would create a
 * large (and runtime-dependent) cycle.  The conversion deliberately uses only
 * the binary dendrogram and row-aligned data already persisted in the result.
 */
function migrateLegacyResult(raw: unknown): ClusterResult | null {
  if (!raw || typeof raw !== "object") return null;
  const legacy = raw as Record<string, unknown>;
  const hierarchy = (legacy.hierarchy && typeof legacy.hierarchy === "object" ? legacy.hierarchy : {}) as Record<string, unknown>;
  const ids = Array.isArray(legacy.ids) ? legacy.ids.map(String) : [];
  const labels = Array.isArray(legacy.leafLabels) ? legacy.leafLabels.map(Number) : ids.map(() => -1);
  if (ids.length !== labels.length) return null;
  const probabilities = Array.isArray(legacy.probabilities) && legacy.probabilities.length === ids.length ? legacy.probabilities.map(Number) : ids.map(() => 0);
  const outlierProxy = Array.isArray(legacy.outlierProxy) && legacy.outlierProxy.length === ids.length ? legacy.outlierProxy.map(Number) : probabilities.map((value) => 1 - value);
  const oldLeaves = Array.isArray(hierarchy.leaves) ? hierarchy.leaves.map(Number).filter(Number.isSafeInteger) : [];
  const leaves = [...new Set([...oldLeaves, ...labels.filter((label) => label >= 0 && Number.isSafeInteger(label))])].sort((a, b) => a - b);
  const merges = Array.isArray(hierarchy.merges) ? hierarchy.merges.map((item) => {
    const merge = item as Record<string, unknown>;
    return { id: Number(merge.id), left: Number(merge.left), right: Number(merge.right), distance: Number(merge.distance) || 0, mass: Number(merge.mass) || 0 };
  }).filter((merge) => Number.isSafeInteger(merge.id) && Number.isSafeInteger(merge.left) && Number.isSafeInteger(merge.right)) : [];
  const mergeById = new Map(merges.map((merge) => [merge.id, merge]));
  const countMass = new Map<number, number>();
  labels.forEach((label) => { if (label >= 0) countMass.set(label, (countMass.get(label) || 0) + 1); });
  const massOf = (id: number): number => {
    const known = countMass.get(id); if (known !== undefined) return known;
    const merge = mergeById.get(id); if (!merge) return 0;
    const mass = massOf(merge.left) + massOf(merge.right); countMass.set(id, mass); return mass;
  };
  const chooseChildren = (root: number): number[] => {
    const rootMerge = mergeById.get(root); if (!rootMerge) return [root];
    const frontier = [rootMerge.left, rootMerge.right]; const candidates = new Map<number, number[]>([[2, frontier.slice()]]); const applied = new Map<number, number>([[2, rootMerge.distance]]);
    const nextSplit = (items: readonly number[]): { index: number; height: number } | null => {
      let index = -1; let best = -Infinity;
      items.forEach((id, itemIndex) => { const merge = mergeById.get(id); if (!merge || massOf(merge.left) < 1 || massOf(merge.right) < 1) return; if (merge.distance > best || merge.distance === best && id < items[index]) { index = itemIndex; best = merge.distance; } });
      return index < 0 ? null : { index, height: best };
    };
    while (frontier.length < 5) { const split = nextSplit(frontier); if (!split) break; const merge = mergeById.get(frontier[split.index])!; frontier.splice(split.index, 1, merge.left, merge.right); candidates.set(frontier.length, frontier.slice()); applied.set(frontier.length, split.height); }
    let bestCount = 2; let bestGap = -Infinity;
    for (const [count, items] of candidates) { const next = nextSplit(items); if (!next) continue; const scale = Math.max(Math.abs(applied.get(count) || 0), Math.abs(next.height), 1e-12); const gap = ((applied.get(count) || 0) - next.height) / scale; if (gap > bestGap + 1e-12 || gap === bestGap && count < bestCount) { bestGap = gap; bestCount = count; } }
    return (candidates.get(bestCount) || candidates.get(2) || frontier.slice()).slice();
  };
  const requestedRoot = hierarchy.root == null ? (merges.length ? merges[merges.length - 1].id : leaves[0] ?? null) : Number(hierarchy.root);
  // A malformed/early v5 result occasionally had multiple leaves but no
  // binary root. Keep all leaves visible instead of making the first leaf a
  // misleading root and dropping its siblings from the tree.
  const root = leaves.length > 1 && !mergeById.has(requestedRoot as number) ? null : requestedRoot;
  const descendants = new Map<number, number[]>();
  const leavesOf = (id: number): number[] => { const cached = descendants.get(id); if (cached) return cached; const merge = mergeById.get(id); const result = merge ? [...new Set([...leavesOf(merge.left), ...leavesOf(merge.right)])].sort((a, b) => a - b) : [id]; descendants.set(id, result); return result; };
  const nodes = new Map<number, { id: number; children: number[]; descendantLeaves: number[]; distance: number; mass: number }>();
  const make = (id: number): void => { if (nodes.has(id)) return; const merge = mergeById.get(id); if (!merge) { nodes.set(id, { id, children: [], descendantLeaves: [id], distance: 0, mass: massOf(id) }); return; } const children = chooseChildren(id); children.forEach(make); nodes.set(id, { id, children, descendantLeaves: leavesOf(id), distance: Math.max(merge.distance, ...children.map((child) => nodes.get(child)?.distance || 0)), mass: massOf(id) }); };
  if (root !== null) make(root);
  const rootChildren = root === null ? leaves.slice() : ((nodes.get(root)?.children?.length ? nodes.get(root)!.children : [root])).slice();
  if (root === null) rootChildren.forEach(make);
  const nodeMap = new Map([...nodes.values()].map((node) => [node.id, node]));
  const ordering = Array.isArray(legacy.leafOrder) ? legacy.leafOrder.map(Number) : Array.isArray(legacy.leafOrdering) ? legacy.leafOrdering.map(Number) : leaves.slice();
  const leafOrder = leaves.filter((leaf) => ordering.includes(leaf)).concat(leaves.filter((leaf) => !ordering.includes(leaf)));
  const rawMemberships = Array.isArray(legacy.softMemberships) ? legacy.softMemberships : Array.isArray(legacy.memberships) ? legacy.memberships : [];
  const memberships = ids.map((_id, index) => { const row = Array.isArray(rawMemberships[index]) ? (rawMemberships[index] as unknown[]).map(Number) : []; return leafOrder.map((_leaf, column) => Number.isFinite(row[column]) ? Math.max(0, row[column]) : 0); });
  const childrenOf = (id: number): number[] => nodeMap.get(id)?.children || [];
  const placements = labels.map((label, index) => {
    if (label < 0) return { kind: "residual" as const, nodeId: null, confidence: 0 };
    const row = memberships[index] || []; let current = rootChildren.slice(); let parent: number | null = root !== null && nodeMap.get(root)?.children.join(",") === rootChildren.join(",") ? root : null; let confidence = Math.max(0, Math.min(1, probabilities[index] || 0));
    while (current.length) {
      const scores = current.map((child) => (nodeMap.get(child)?.descendantLeaves || [child]).reduce((sum, leaf) => { const column = leafOrder.indexOf(leaf); return sum + (column < 0 ? 0 : row[column] || 0); }, 0)); const total = scores.reduce((sum, value) => sum + value, 0); let best = 0; for (let item = 1; item < scores.length; item++) if (scores[item] > scores[best] || scores[item] === scores[best] && current[item] < current[best]) best = item;
      if (total <= 1e-9 || scores[best] <= total * 0.5) return { kind: "residual" as const, nodeId: parent, confidence: total > 0 ? scores[best] / total : 0 };
      confidence = scores[best] / total; const next = current[best]; const nextChildren = childrenOf(next); if (!nextChildren.length) return { kind: "leaf" as const, nodeId: next, confidence }; parent = next; current = nextChildren;
    }
    return { kind: "leaf" as const, nodeId: label, confidence };
  });
  const placementMass = new Map<number, number>(); const parent = new Map<number, number>(); for (const node of nodes.values()) for (const child of node.children) parent.set(child, node.id);
  for (const placement of placements) { if (placement.nodeId === null) continue; let current: number | undefined = placement.nodeId; while (current !== undefined) { placementMass.set(current, (placementMass.get(current) || 0) + 1); current = parent.get(current); } }
  const finalNodes = [...nodes.values()].sort((a, b) => a.id - b.id).map((node) => ({ ...node, mass: placementMass.get(node.id) || 0 }));
  const migrated = { ...legacy, schemaVersion: 6 as const, ids, leafLabels: labels, probabilities, outlierProxy, hierarchy: { leaves, merges, root, nodes: finalNodes, rootChildren, splitMethod: "distance-knee-2-5" as const }, hierarchyPlacements: placements, leafOrder, leafOrdering: leafOrder, ...(rawMemberships.length === ids.length ? { memberships, softMemberships: memberships } : {}) } as ClusterResult;
  return migrated;
}

/** Keep large v6 arrays in normalized tables and only lightweight metadata in JSON. */
function compactV6Result(result: ClusterResult): string {
  const modelHash = result.pca.model?.modelHash;
  return JSON.stringify({
    schemaVersion: 6,
    pca: { ...result.pca, model: modelHash ? { modelHash, provider: result.pca.model?.provider, model: result.pca.model?.model } : undefined },
    hierarchy: { root: result.hierarchy.root, splitMethod: result.hierarchy.splitMethod },
    timings: result.timings,
    titleGeneration: result.titleGeneration,
    embeddingProvider: result.embeddingProvider,
    embeddingModel: result.embeddingModel,
    // UMAP coordinates are the durable insertion space. They are kept in the
    // compact metadata JSON because they must remain row-aligned with the
    // result even when the optional 2D visualization is lazily regenerated.
    umap: result.umap,
    incremental: result.incremental ? { ...result.incremental, provisionalPaths: [] } : undefined,
    visualization: result.visualization ? { configuration: result.visualization.configuration, timings: result.visualization.timings } : undefined,
    _normalizedV6: { memberships: !!(result.softMemberships || result.memberships), softMemberships: !!result.softMemberships, titles: result.titles !== undefined }
  });
}

/** Apply the exact projection used by the browser and Python implementations. */
export function projectPca(vector: number[], pca: PcaModel): number[] {
  if (vector.length !== pca.inputDimension || pca.mean.length !== pca.inputDimension) {
    throw new Error(`PCA input dimension mismatch: expected ${pca.inputDimension}, got ${vector.length}`);
  }
  const normalized = vector.slice();
  if (pca.normalization === "l2") {
    const norm = Math.sqrt(normalized.reduce((sum, value) => sum + value * value, 0));
    if (norm > 0) for (let i = 0; i < normalized.length; i++) normalized[i] /= norm;
  }
  const centered = normalized.map((value, index) => value - pca.mean[index]);
  return pca.components.map((component) => {
    if (component.length !== pca.inputDimension) throw new Error("PCA component dimension mismatch");
    return component.reduce((sum, value, index) => sum + value * centered[index], 0);
  });
}

/** Stable vector fingerprint; includes dimensionality and IEEE-754 values. */
export async function embeddingHash(vector: number[]): Promise<string> {
  return contentHash(JSON.stringify({ dimension: vector.length, vector }));
}

export async function pcaModelHash(model: Omit<PcaModel, "modelHash">): Promise<string> {
  return contentHash(JSON.stringify({ inputDimension: model.inputDimension, outputDimension: model.outputDimension,
    normalization: model.normalization, mean: model.mean, components: model.components,
    explainedVariance: model.explainedVariance, provider: model.provider || "", model: model.model || "" }));
}

export function validateClusterResultAlignment(result: ClusterResult): void {
  const n = result.ids.length;
  if (new Set(result.ids).size !== n) throw new Error("Cluster result ids must be unique");
  for (const [name, values] of [["leafLabels", result.leafLabels], ["probabilities", result.probabilities], ["outlierProxy", result.outlierProxy]] as const) if (values.length !== n) throw new Error(`Cluster result ${name} must align with ids`);
  const leaves = result.leafOrder || result.leafOrdering || result.hierarchy.leaves;
  if (new Set(leaves).size !== leaves.length || leaves.some((leaf) => !Number.isSafeInteger(leaf) || leaf < 0)) throw new Error("Cluster result leaf order is invalid");
  const memberships = result.softMemberships || result.memberships;
  if (memberships && (memberships.length !== n || memberships.some((row) => row.length !== leaves.length))) throw new Error("Cluster result memberships must align with ids and leaf order");
  if (result.schemaVersion === 6 && (!result.hierarchy.nodes || !result.hierarchy.rootChildren || result.hierarchy.splitMethod !== "distance-knee-2-5" || !result.hierarchyPlacements)) throw new Error("Schema v6 requires its n-ary hierarchy and placements");
  if (result.hierarchyPlacements && (result.hierarchyPlacements.length !== n || result.hierarchyPlacements.some((placement) => !placement || (placement.kind !== "leaf" && placement.kind !== "residual") || !Number.isFinite(placement.confidence) || placement.confidence < 0 || placement.confidence > 1))) throw new Error("Cluster hierarchy placements must align with ids");
  if (result.hierarchy.nodes) {
    const nodes = new Map(result.hierarchy.nodes.map((node) => [node.id, node]));
    if (nodes.size !== result.hierarchy.nodes.length) throw new Error("Cluster hierarchy node ids must be unique");
    for (const node of result.hierarchy.nodes) {
      if (!Number.isSafeInteger(node.id) || node.children.length > 0 && (node.children.length < 2 || node.children.length > 5) || new Set(node.descendantLeaves).size !== node.descendantLeaves.length) throw new Error("Cluster hierarchy n-ary nodes are malformed");
      if (node.children.some((child) => !nodes.has(child)) || node.descendantLeaves.some((leaf) => !result.hierarchy.leaves.includes(leaf))) throw new Error("Cluster hierarchy references an unknown child or leaf");
      const union = node.children.flatMap((child) => nodes.get(child)?.descendantLeaves || [child]);
      if (new Set(union).size !== union.length || node.children.length && (union.length !== node.descendantLeaves.length || union.some((leaf) => !node.descendantLeaves.includes(leaf)))) throw new Error("Cluster hierarchy children must be disjoint and exhaustive");
    }
    for (const leaf of result.hierarchy.leaves) if (!nodes.has(leaf) || nodes.get(leaf)!.children.length || nodes.get(leaf)!.descendantLeaves.length !== 1 || nodes.get(leaf)!.descendantLeaves[0] !== leaf) throw new Error("Cluster hierarchy leaf nodes are malformed");
    const rootChildren = result.hierarchy.rootChildren || []; const rootLeaves = rootChildren.flatMap((child) => nodes.get(child)?.descendantLeaves || [child]);
    if (new Set(rootLeaves).size !== rootLeaves.length || rootLeaves.length !== result.hierarchy.leaves.length || rootLeaves.some((leaf) => !result.hierarchy.leaves.includes(leaf))) throw new Error("Cluster hierarchy root children must be disjoint and exhaustive");
    if (result.hierarchy.leaves.length > 1 && (rootChildren.length < 2 || rootChildren.length > 5)) throw new Error("Cluster hierarchy root must have 2 to 5 children");
    if (result.hierarchy.leaves.length > 1) { const root = result.hierarchy.root === null ? undefined : nodes.get(result.hierarchy.root); if (!root || root.children.length !== rootChildren.length || root.children.some((child, index) => child !== rootChildren[index])) throw new Error("Cluster hierarchy root children must match the root node"); }
    const reachable = new Set<number>(); const visit = (id: number): void => { if (reachable.has(id)) throw new Error("Cluster hierarchy contains a cycle or duplicate subtree"); reachable.add(id); nodes.get(id)?.children.forEach(visit); }; if (result.hierarchy.root !== null) visit(result.hierarchy.root); if (reachable.size !== nodes.size) throw new Error("Cluster hierarchy contains unreachable nodes");
    if (result.hierarchyPlacements?.some((placement) => placement.nodeId !== null && (!nodes.has(placement.nodeId) || placement.kind === "leaf" !== (nodes.get(placement.nodeId)!.children.length === 0)))) throw new Error("Cluster hierarchy placement target is invalid");
  }
  if (result.visualization && (result.visualization.coordinates.length !== n || result.visualization.labels.length !== n)) throw new Error("Cluster visualization must align with ids");
  if (result.visualization?.coordinates.some((point) => point.length !== 2 || point.some((value) => !Number.isFinite(value)))) throw new Error("Cluster visualization coordinates must be finite 2D points");
  if (result.umap) {
    if (result.umap.coordinates.length !== n) throw new Error("Saved UMAP coordinates must align with ids");
    if (!Number.isSafeInteger(result.umap.inputDimension) || result.umap.inputDimension < 1 || !Number.isSafeInteger(result.umap.outputDimension) || result.umap.outputDimension < 1) throw new Error("Saved UMAP dimensions are invalid");
    if (result.umap.coordinates.some((row) => row.length !== result.umap!.outputDimension || row.some((value) => !Number.isFinite(value)))) throw new Error("Saved UMAP coordinates must be finite and rectangular");
    if (Object.values(result.umap.leafKnnDistanceP95 || {}).some((value) => !Number.isFinite(value) && value !== Number.POSITIVE_INFINITY || Number(value) < 0)) throw new Error("Saved UMAP leaf kNN envelopes are invalid");
  }
}

/** Write a converted result while the open-time migration transaction is held. */
function persistMigratedResult(db: SqlDatabase, result: ClusterResult, resultId: string, now: string, pcaJsonRequired: boolean): void {
  validateClusterResultAlignment(result);
  const pcaModel = result.pca.model;
  if (pcaModel) db.run("INSERT OR REPLACE INTO pca_models(model_hash,provider,model,input_dimension,output_dimension,normalization,mean_json,components_json,explained_variance_json,mean_blob,components_blob,explained_variance_blob,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", [pcaModel.modelHash, pcaModel.provider || null, pcaModel.model || null, pcaModel.inputDimension, pcaModel.outputDimension, pcaModel.normalization, pcaJsonRequired ? JSON.stringify(pcaModel.mean) : null, pcaJsonRequired ? JSON.stringify(pcaModel.components) : null, pcaJsonRequired ? JSON.stringify(pcaModel.explainedVariance) : null, float32Blob(pcaModel.mean), float32Blob(flattenNumbers(pcaModel.components)), float32Blob(pcaModel.explainedVariance), now]);
  db.run("DELETE FROM result_note_hashes WHERE result_id=?", [resultId]); db.run("DELETE FROM assignments WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_merges WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_leaves WHERE result_id=?", [resultId]); db.run("DELETE FROM leaf_order WHERE result_id=?", [resultId]); db.run("DELETE FROM visualization_points WHERE result_id=?", [resultId]); db.run("DELETE FROM cluster_titles WHERE result_id=?", [resultId]); db.run("DELETE FROM soft_memberships WHERE result_id=?", [resultId]); db.run("DELETE FROM membership_rows WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_nodes WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_children WHERE result_id=?", [resultId]); db.run("DELETE FROM root_children WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_placements WHERE result_id=?", [resultId]);
  result.ids.forEach((path, index) => {
    db.run("INSERT OR IGNORE INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?)", [path, path.split("/").pop() || path, 0, "legacy", null]);
    const provisional = new Set(result.provisionalPaths || result.incremental?.provisionalPaths || []);
    db.run("INSERT INTO assignments(result_id,path,leaf_label,probability,outlier_score,provisional) VALUES(?,?,?,?,?,?)", [resultId, path, result.leafLabels[index], result.probabilities[index], result.outlierProxy[index], provisional.has(path) ? 1 : 0]);
  });
  result.hierarchy.leaves.forEach((leaf, ordinal) => db.run("INSERT INTO hierarchy_leaves(result_id,ordinal,leaf_id) VALUES(?,?,?)", [resultId, ordinal, leaf]));
  (result.leafOrder || result.leafOrdering || result.hierarchy.leaves).forEach((leaf, ordinal) => db.run("INSERT INTO leaf_order(result_id,ordinal,leaf_id) VALUES(?,?,?)", [resultId, ordinal, leaf]));
  result.hierarchy.merges.forEach((merge) => db.run("INSERT INTO hierarchy_merges(result_id,id,left_id,right_id,distance,mass) VALUES(?,?,?,?,?,?)", [resultId, merge.id, merge.left, merge.right, merge.distance, merge.mass]));
  for (const node of result.hierarchy.nodes || []) { db.run("INSERT INTO hierarchy_nodes(result_id,node_id,distance,mass,descendant_leaves_blob) VALUES(?,?,?,?,?)", [resultId, node.id, node.distance, node.mass, float32Blob(node.descendantLeaves)]); node.children.forEach((child, ordinal) => db.run("INSERT INTO hierarchy_children(result_id,parent_id,ordinal,child_id) VALUES(?,?,?,?)", [resultId, node.id, ordinal, child])); }
  (result.hierarchy.rootChildren || []).forEach((child, ordinal) => db.run("INSERT INTO root_children(result_id,ordinal,child_id) VALUES(?,?,?)", [resultId, ordinal, child]));
  if (result.visualization && result.visualization.coordinates.length === result.ids.length && result.visualization.labels.length === result.ids.length) result.visualization.coordinates.forEach((point, ordinal) => db.run("INSERT INTO visualization_points(result_id,ordinal,path,x,y,leaf_label) VALUES(?,?,?,?,?,?)", [resultId, ordinal, result.ids[ordinal], point[0], point[1], result.visualization!.labels[ordinal]]));
  for (const [node, title] of Object.entries(result.titles || {})) db.run("INSERT INTO cluster_titles(result_id,node_id,title) VALUES(?,?,?)", [resultId, Number(node), title]);
  const memberships = result.softMemberships || result.memberships;
  if (memberships) memberships.forEach((row, ordinal) => { db.run("INSERT INTO membership_rows(result_id,path,width,values_blob) VALUES(?,?,?,?)", [resultId, result.ids[ordinal], row.length, float32Blob(row)]); row.forEach((membership, leafIndex) => db.run("INSERT INTO soft_memberships(result_id,path,leaf_id,membership) VALUES(?,?,?,?)", [resultId, result.ids[ordinal], (result.leafOrder || result.leafOrdering || result.hierarchy.leaves)[leafIndex], membership])); });
  (result.hierarchyPlacements || []).forEach((placement, ordinal) => db.run("INSERT INTO hierarchy_placements(result_id,path,ordinal,kind,node_id,confidence) VALUES(?,?,?,?,?,?)", [resultId, result.ids[ordinal], ordinal, placement.kind, placement.nodeId, placement.confidence]));
  db.run("UPDATE results SET schema_version=?,result_json=? WHERE result_id=?", [6, compactV6Result(result), resultId]);
}

export interface SqliteStorageOptions { path?: string; now?: () => string; }

export class SqliteClusterStore {
  readonly path: string;
  private db!: SqlDatabase;
  private opened = false;
  private readonly now: () => string;
  private transactionTail: Promise<unknown> = Promise.resolve();
  private embeddingJsonRequired = false;
  private pcaJsonRequired = false;
  private coordinateJsonRequired = false;
  constructor(private readonly adapter: BinaryAdapter, private readonly sql: SqlJsStatic, options: SqliteStorageOptions = {}) {
    this.path = options.path || SQLITE_PATH;
    this.now = options.now || (() => new Date().toISOString());
  }

  async open(): Promise<this> {
    let bytes: ArrayBuffer | undefined;
    if (await this.adapter.exists(this.path)) bytes = await this.adapter.readBinary(this.path);
    this.db = new this.sql.Database(bytes ? new Uint8Array(bytes) : undefined);
    let previousVersion = 0;
    try { previousVersion = Number(this.db.exec("SELECT value FROM metadata WHERE key='schema_version'")[0]?.values[0]?.[0]) || 0; } catch { /* pre-metadata database */ }
    this.db.run(SCHEMA);
    // Databases created by an early development build did not carry the
    // note-content column. Keep opening them safe and let the next write
    // populate the richer immutable embedding records.
    for (const statement of [
      "ALTER TABLE notes ADD COLUMN content TEXT",
      "ALTER TABLE notes ADD COLUMN active INTEGER NOT NULL DEFAULT 1",
      "ALTER TABLE assignments ADD COLUMN provisional INTEGER NOT NULL DEFAULT 0",
      "ALTER TABLE embeddings ADD COLUMN note_content_hash TEXT NOT NULL DEFAULT ''",
      "ALTER TABLE embeddings ADD COLUMN vector_blob BLOB",
      "ALTER TABLE pca_models ADD COLUMN mean_blob BLOB",
      "ALTER TABLE pca_models ADD COLUMN components_blob BLOB",
      "ALTER TABLE pca_models ADD COLUMN explained_variance_blob BLOB",
      "ALTER TABLE pca_coordinates ADD COLUMN coordinate_blob BLOB",
    ]) try { this.db.run(statement); } catch { /* idempotent migration */ }
    // CREATE VIEW IF NOT EXISTS does not replace a view created by an older
    // plugin build, so refresh the active-only cache view after adding the
    // note activity flag.
    try { this.db.run("DROP VIEW IF EXISTS v_current_embeddings"); this.db.run(SCHEMA); } catch { /* retain the already-open schema */ }
    const requires = (table: string, columns: string[]): boolean => {
      const info = this.db.exec(`PRAGMA table_info(${table})`)[0]?.values || [];
      return info.some((row) => columns.includes(String(row[1])) && Number(row[3]) === 1);
    };
    this.embeddingJsonRequired = requires("embeddings", ["vector_json"]);
    this.pcaJsonRequired = requires("pca_models", ["mean_json", "components_json", "explained_variance_json"]);
    this.coordinateJsonRequired = requires("pca_coordinates", ["coordinates_json"]);
    const migrateBlob = (table: string, key: string, jsonColumn: string, blobColumn: string, clearJson: boolean): void => {
      const rows = this.db.exec(`SELECT ${key},${jsonColumn} FROM ${table} WHERE ${blobColumn} IS NULL AND ${jsonColumn} IS NOT NULL`)[0]?.values || [];
      for (const row of rows) { try { this.db.run(`UPDATE ${table} SET ${blobColumn}=?${clearJson ? `,${jsonColumn}=NULL` : ""} WHERE ${key}=?`, [float32Blob(flattenJsonNumbers(JSON.parse(String(row[1])))), row[0]]); } catch { /* retain readable legacy JSON */ } }
    };
    try {
      this.db.run("BEGIN IMMEDIATE");
      if (bytes && previousVersion < SQLITE_SCHEMA_VERSION) {
        this.db.run("DELETE FROM results WHERE rowid NOT IN (SELECT rowid FROM results ORDER BY created_at DESC,rowid DESC LIMIT 1)");
        this.db.run("DELETE FROM embedding_logs WHERE id NOT IN (SELECT id FROM embedding_logs ORDER BY id DESC LIMIT 1)");
        this.db.run("DELETE FROM embeddings WHERE rowid NOT IN (SELECT MAX(rowid) FROM embeddings GROUP BY path,provider,model)");
        migrateBlob("embeddings", "rowid", "vector_json", "vector_blob", !this.embeddingJsonRequired);
        migrateBlob("pca_coordinates", "rowid", "coordinates_json", "coordinate_blob", !this.coordinateJsonRequired);
        const models = this.db.exec("SELECT model_hash,mean_json,components_json,explained_variance_json FROM pca_models WHERE mean_blob IS NULL OR components_blob IS NULL OR explained_variance_blob IS NULL")[0]?.values || [];
        for (const row of models) try { this.db.run(`UPDATE pca_models SET mean_blob=?,components_blob=?,explained_variance_blob=?${this.pcaJsonRequired ? "" : ",mean_json=NULL,components_json=NULL,explained_variance_json=NULL"} WHERE model_hash=?`, [float32Blob(flattenJsonNumbers(JSON.parse(String(row[1])))), float32Blob(flattenJsonNumbers(JSON.parse(String(row[2])))), float32Blob(flattenJsonNumbers(JSON.parse(String(row[3])))), row[0]]); } catch { /* retain readable legacy JSON */ }
      }
      // A previous interrupted/early migration, or a legacy JSON import into
      // an already-v6 database, can leave the newest result row at v5. Check
      // the row independently of the metadata marker so startup can repair
      // that state as well.
      if (bytes) {
        const latest = this.db.exec("SELECT result_id,result_json,schema_version FROM results ORDER BY created_at DESC,rowid DESC LIMIT 1")[0]?.values[0];
        if (latest?.[0] != null && Number(latest[2]) < 6) {
          try {
            const converted = migrateLegacyResult(JSON.parse(String(latest[1])));
            if (converted) persistMigratedResult(this.db, converted, String(latest[0]), this.now(), this.pcaJsonRequired);
          } catch (error) {
            throw new Error(`Unable to migrate latest v5 cluster result: ${error instanceof Error ? error.message : String(error)}`);
          }
        }
      }
      this.db.run("INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)", [String(SQLITE_SCHEMA_VERSION)]);
      this.db.run("INSERT OR IGNORE INTO migrations(name,completed_at) VALUES(?,?)", ["schema-v6-normalized", this.now()]);
      this.db.run("COMMIT"); this.opened = true;
      if (bytes && previousVersion < SQLITE_SCHEMA_VERSION) await this.flush();
    } catch (error) {
      try { this.db.run("ROLLBACK"); } catch { /* preserve migration error */ }
      try { this.db.close(); } catch { /* preserve migration error */ }
      this.opened = false; throw error;
    }
    return this;
  }
  get database(): SqlDatabase { this.requireOpen(); return this.db; }
  close(): void { if (this.opened) { this.db.close(); this.opened = false; } }
  private requireOpen(): void { if (!this.opened) throw new Error("SQLite store is not open"); }

  async flush(): Promise<void> {
    this.requireOpen();
    const parent = this.path.slice(0, this.path.lastIndexOf("/"));
    if (parent && !(await this.adapter.exists(parent))) await this.adapter.mkdir(parent);
    const bytes = this.db.export();
    const durableBytes = bytes.slice().buffer as ArrayBuffer;
    await this.adapter.writeBinary(`${this.path}.tmp`, durableBytes);
    if (this.adapter.rename) {
      const temporaryPath = `${this.path}.tmp`;
      try {
        await this.adapter.rename(temporaryPath, this.path);
      } catch (error) {
        // Obsidian's desktop adapter can reject rename-over-existing on some
        // mounted filesystems. The complete database is already durable at
        // temporaryPath, so remove only the old destination and retry the
        // replacement when the adapter exposes the required operation.
        const errorCode = typeof error === "object" && error !== null && "code" in error ? error.code : undefined;
        const errorMessage = error instanceof Error ? error.message : String(error);
        const destinationExistsError = errorCode === "EEXIST" || /already exists|file exists/i.test(errorMessage);
        if (!destinationExistsError || !this.adapter.remove || !(await this.adapter.exists(this.path))) throw error;
        await this.adapter.remove(this.path);
        await this.adapter.rename(temporaryPath, this.path);
      }
    }
    else {
      await this.adapter.writeBinary(this.path, durableBytes);
      await this.adapter.remove?.(`${this.path}.tmp`);
    }
  }

  /** Execute a mutating operation and persist it as one transaction. */
  async transaction<T>(operation: (db: SqlDatabase) => T | Promise<T>): Promise<T> {
    this.requireOpen();
    const run = async (): Promise<T> => {
      const before = this.db.export(); this.db.run("BEGIN IMMEDIATE");
      try { const result = await operation(this.db); this.db.run("COMMIT"); await this.flush(); return result; }
      catch (error) {
        try { this.db.run("ROLLBACK"); } catch { /* commit or operation may already have failed */ }
        try { this.db.close(); this.db = new this.sql.Database(before); this.db.run(SCHEMA); } catch { /* preserve original adapter error */ }
        throw error;
      }
    };
    const result = this.transactionTail.then(run, run);
    this.transactionTail = result.then(() => undefined, () => undefined);
    return result;
  }

  async upsertNote(note: NoteRecord): Promise<void> {
    await this.transaction((db) => db.run("INSERT INTO notes(path,title,mtime,content_hash,content,active) VALUES(?,?,?,?,?,1) ON CONFLICT(path) DO UPDATE SET title=excluded.title,mtime=excluded.mtime,content_hash=excluded.content_hash,content=excluded.content,active=1", [note.path, note.title, note.mtime, note.hash, note.content]));
  }
  async upsertNotes(notes: NoteRecord[]): Promise<void> {
    await this.transaction((db) => { for (const note of notes) db.run("INSERT INTO notes(path,title,mtime,content_hash,content,active) VALUES(?,?,?,?,?,1) ON CONFLICT(path) DO UPDATE SET title=excluded.title,mtime=excluded.mtime,content_hash=excluded.content_hash,content=excluded.content,active=1", [note.path, note.title, note.mtime, note.hash, note.content]); });
  }
  /** Synchronize the active note set without deleting historical cache rows. */
  async syncActiveNotes(notes: NoteRecord[]): Promise<void> {
    await this.transaction((db) => {
      db.run("UPDATE notes SET active=0");
      for (const note of notes) db.run("INSERT INTO notes(path,title,mtime,content_hash,content,active) VALUES(?,?,?,?,?,1) ON CONFLICT(path) DO UPDATE SET title=excluded.title,mtime=excluded.mtime,content_hash=excluded.content_hash,content=excluded.content,active=1", [note.path, note.title, note.mtime, note.hash, note.content]);
    });
  }
  async listNoteMetadata(activeOnly = false): Promise<Array<{ path: string; hash: string; mtime: number; active: boolean; content?: string }>> {
    this.requireOpen();
    const rows = this.query(`SELECT path,content_hash AS hash,mtime,active,content FROM notes${activeOnly ? " WHERE active=1" : ""} ORDER BY path`);
    return rows.map((row) => ({ path: String(row.path), hash: String(row.hash), mtime: Number(row.mtime), active: Number(row.active) !== 0, content: row.content == null ? undefined : String(row.content) }));
  }
  async listNotes(activeOnly = false): Promise<NoteRecord[]> {
    this.requireOpen();
    const rows = this.query(`SELECT path,title,mtime,content_hash AS hash,content FROM notes${activeOnly ? " WHERE active=1" : ""} ORDER BY path`);
    return rows.map((row) => ({ path: String(row.path), title: String(row.title), mtime: Number(row.mtime), hash: String(row.hash), content: row.content == null ? "" : String(row.content) }));
  }
  /** Move all path-keyed cache and result rows in one transaction. */
  async renameNote(oldPath: string, newPath: string): Promise<boolean> {
    if (!oldPath || !newPath || oldPath === newPath) return oldPath === newPath;
    return this.transaction((db) => {
      const oldRows = db.exec(`SELECT path FROM notes WHERE path=${sqlQuote(oldPath)}`);
      const newRows = db.exec(`SELECT path FROM notes WHERE path=${sqlQuote(newPath)}`);
      // A previous refresh may have moved the normalized rows successfully
      // and failed only while publishing the compact result. Treat the same
      // rename as already applied so the queued event can be retried safely.
      if (!oldRows[0]?.values.length) return !!newRows[0]?.values.length;
      if (newRows[0]?.values.length) return false;
      // Child rows must move before the referenced parent row while foreign
      // keys are enabled. The target was checked above, so each update keeps
      // its table's primary key unique.
      for (const table of ["embeddings", "pca_coordinates", "result_note_hashes", "assignments", "visualization_points", "soft_memberships", "membership_rows", "hierarchy_placements"]) db.run(`UPDATE ${table} SET path=? WHERE path=?`, [newPath, oldPath]);
      db.run("UPDATE notes SET path=? WHERE path=?", [newPath, oldPath]);
      return true;
    });
  }
  async putEmbedding(entry: CachedEmbedding): Promise<string> {
    const hash = await embeddingHash(entry.vector);
    await this.transaction((db) => {
      db.run("DELETE FROM embeddings WHERE path=? AND provider=? AND model=?", [entry.path, entry.provider, entry.model]);
      db.run("INSERT INTO embeddings(path,provider,model,embedding_hash,note_content_hash,dimension,vector_json,vector_blob,created_at) VALUES(?,?,?,?,?,?,?,?,?)", [entry.path, entry.provider, entry.model, hash, entry.hash, entry.vector.length, this.embeddingJsonRequired ? JSON.stringify(entry.vector) : null, float32Blob(entry.vector), this.now()]);
    });
    return hash;
  }
  /** Persist a batch under one SQLite transaction and one durable flush. */
  async putEmbeddings(entries: CachedEmbedding[]): Promise<string[]> {
    const hashes = await Promise.all(entries.map((entry) => embeddingHash(entry.vector)));
    await this.transaction((db) => entries.forEach((entry, index) => {
      db.run("DELETE FROM embeddings WHERE path=? AND provider=? AND model=?", [entry.path, entry.provider, entry.model]);
      db.run("INSERT INTO embeddings(path,provider,model,embedding_hash,note_content_hash,dimension,vector_json,vector_blob,created_at) VALUES(?,?,?,?,?,?,?,?,?)", [entry.path, entry.provider, entry.model, hashes[index], entry.hash, entry.vector.length, this.embeddingJsonRequired ? JSON.stringify(entry.vector) : null, float32Blob(entry.vector), this.now()]);
    }));
    return hashes;
  }
  async getEmbedding(path: string, provider: string, model: string, noteHash?: string): Promise<CachedEmbedding | undefined> {
    this.requireOpen();
    const statement = this.db.prepare("SELECT e.path,e.provider,e.model,e.embedding_hash,e.note_content_hash,e.vector_json,e.vector_blob,n.content_hash FROM embeddings e JOIN notes n USING(path) WHERE e.path=? AND e.provider=? AND e.model=? AND n.active=1 ORDER BY e.created_at DESC");
    try { statement.bind([path, provider, model]); while (statement.step()) { const row = statement.getAsObject(); if (!noteHash || row.note_content_hash === noteHash) return { path: String(row.path), provider: String(row.provider), model: String(row.model), hash: String(row.note_content_hash), vector: row.vector_blob ? float32Values(row.vector_blob) : JSON.parse(String(row.vector_json)) }; } }
    finally { statement.free(); }
    return undefined;
  }
  async loadEmbeddings(provider?: string, model?: string): Promise<Map<string, CachedEmbedding>> {
    this.requireOpen();
    const predicates = [provider ? `provider=${sqlQuote(provider)}` : "1=1", model ? `model=${sqlQuote(model)}` : "1=1"];
    const rows = this.db.exec(`SELECT e.path,e.provider,e.model,e.note_content_hash,e.vector_json,e.vector_blob FROM embeddings e JOIN notes n USING(path) WHERE ${predicates.join(" AND ")} AND n.active=1 AND e.rowid=(SELECT MAX(e2.rowid) FROM embeddings e2 WHERE e2.path=e.path AND e2.provider=e.provider AND e2.model=e.model)`);
    const map = new Map<string, CachedEmbedding>();
    for (const row of rows[0]?.values || []) { const [path, itemProvider, itemModel, hash, vector, blob] = row; const entry = { path: String(path), provider: String(itemProvider), model: String(itemModel), hash: String(hash), vector: blob ? float32Values(blob) : JSON.parse(String(vector)) as number[] }; map.set(`${entry.provider}:${entry.model}:${entry.path}`, entry); }
    return map;
  }
  async savePcaModel(model: PcaModel): Promise<void> {
    await this.transaction((db) => db.run("INSERT OR REPLACE INTO pca_models(model_hash,provider,model,input_dimension,output_dimension,normalization,mean_json,components_json,explained_variance_json,mean_blob,components_blob,explained_variance_blob,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", [model.modelHash, model.provider || null, model.model || null, model.inputDimension, model.outputDimension, model.normalization, this.pcaJsonRequired ? JSON.stringify(model.mean) : null, this.pcaJsonRequired ? JSON.stringify(model.components) : null, this.pcaJsonRequired ? JSON.stringify(model.explainedVariance) : null, float32Blob(model.mean), float32Blob(flattenNumbers(model.components)), float32Blob(model.explainedVariance), this.now()]));
  }
  async getPcaModel(modelHash: string): Promise<PcaModel | undefined> {
    this.requireOpen(); const rows = this.db.exec("SELECT * FROM pca_models WHERE model_hash=" + sqlQuote(modelHash));
    const row = rows[0]?.values[0]; if (!row) return undefined;
    const columns = rows[0].columns; const object = Object.fromEntries(columns.map((key, i) => [key, row[i]]));
    const mean = object.mean_blob ? float32Values(object.mean_blob) : JSON.parse(String(object.mean_json)); const flatComponents = object.components_blob ? float32Values(object.components_blob) : [];
    const components = object.components_blob ? Array.from({ length: Number(object.output_dimension) }, (_value, index) => flatComponents.slice(index * Number(object.input_dimension), (index + 1) * Number(object.input_dimension))) : JSON.parse(String(object.components_json));
    return { modelHash: String(object.model_hash), provider: object.provider == null ? undefined : String(object.provider), model: object.model == null ? undefined : String(object.model), inputDimension: Number(object.input_dimension), outputDimension: Number(object.output_dimension), normalization: String(object.normalization), mean, components, explainedVariance: object.explained_variance_blob ? float32Values(object.explained_variance_blob) : JSON.parse(String(object.explained_variance_json)) };
  }
  async project(path: string, vector: number[], model: PcaModel): Promise<number[]> {
    const coordinates = projectPca(vector, model);
    await this.transaction((db) => db.run("INSERT OR REPLACE INTO pca_coordinates(path,model_hash,coordinates_json,coordinate_blob) VALUES(?,?,?,?)", [path, model.modelHash, this.coordinateJsonRequired ? JSON.stringify(coordinates) : null, float32Blob(coordinates)]));
    return coordinates;
  }
  /** Project and persist a row-aligned batch with one transaction/flush. */
  async projectMany(rows: Array<{ path: string; vector: number[] }>, model: PcaModel): Promise<number[][]> {
    const coordinates = rows.map((row) => projectPca(row.vector, model));
    await this.transaction((db) => rows.forEach((row, index) => db.run("INSERT OR REPLACE INTO pca_coordinates(path,model_hash,coordinates_json,coordinate_blob) VALUES(?,?,?,?)", [row.path, model.modelHash, this.coordinateJsonRequired ? JSON.stringify(coordinates[index]) : null, float32Blob(coordinates[index])] )));
    return coordinates;
  }
  async saveResult(result: ClusterResult, options: { resultId?: string; coordinates?: Record<string, number[]>; softMemberships?: Record<string, Record<number, number>>; noteHashes?: ReadonlyMap<string, string> } = {}): Promise<string> {
    validateClusterResultAlignment(result);
    for (const [path, point] of Object.entries(options.coordinates || {})) if (!result.ids.includes(path) || point.length !== 2 || point.some((value) => !Number.isFinite(value))) throw new Error(`Visualization coordinate for ${path} is invalid or unaligned`);
    const resultId = options.resultId || await contentHash(JSON.stringify(result));
    await this.transaction((db) => {
      const pcaModel = result.pca.model;
      if (pcaModel) db.run("INSERT OR REPLACE INTO pca_models(model_hash,provider,model,input_dimension,output_dimension,normalization,mean_json,components_json,explained_variance_json,mean_blob,components_blob,explained_variance_blob,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", [pcaModel.modelHash, pcaModel.provider || null, pcaModel.model || null, pcaModel.inputDimension, pcaModel.outputDimension, pcaModel.normalization, this.pcaJsonRequired ? JSON.stringify(pcaModel.mean) : null, this.pcaJsonRequired ? JSON.stringify(pcaModel.components) : null, this.pcaJsonRequired ? JSON.stringify(pcaModel.explainedVariance) : null, float32Blob(pcaModel.mean), float32Blob(flattenNumbers(pcaModel.components)), float32Blob(pcaModel.explainedVariance), this.now()]);
      db.run("INSERT OR REPLACE INTO results(result_id,schema_version,created_at,result_json) VALUES(?,?,?,?)", [resultId, result.schemaVersion, this.now(), result.schemaVersion === 6 ? compactV6Result(result) : JSON.stringify(result)]);
      db.run("DELETE FROM result_note_hashes WHERE result_id=?", [resultId]); db.run("DELETE FROM assignments WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_merges WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_leaves WHERE result_id=?", [resultId]); db.run("DELETE FROM leaf_order WHERE result_id=?", [resultId]); db.run("DELETE FROM visualization_points WHERE result_id=?", [resultId]); db.run("DELETE FROM cluster_titles WHERE result_id=?", [resultId]); db.run("DELETE FROM soft_memberships WHERE result_id=?", [resultId]); db.run("DELETE FROM membership_rows WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_nodes WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_children WHERE result_id=?", [resultId]); db.run("DELETE FROM root_children WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_placements WHERE result_id=?", [resultId]);
      const provisional = new Set(result.provisionalPaths || result.incremental?.provisionalPaths || []);
      result.ids.forEach((path, i) => db.run("INSERT INTO assignments(result_id,path,leaf_label,probability,outlier_score,provisional) VALUES(?,?,?,?,?,?)", [resultId, path, result.leafLabels[i], result.probabilities[i], result.outlierProxy[i], provisional.has(path) ? 1 : 0]));
      if (options.noteHashes) for (const path of result.ids) {
        const hash = options.noteHashes.get(path);
        if (hash !== undefined) db.run("INSERT INTO result_note_hashes(result_id,path,content_hash) VALUES(?,?,?)", [resultId, path, hash]);
      }
      result.hierarchy.leaves.forEach((leaf, ordinal) => db.run("INSERT INTO hierarchy_leaves(result_id,ordinal,leaf_id) VALUES(?,?,?)", [resultId, ordinal, leaf]));
      (result.leafOrder || result.leafOrdering || result.hierarchy.leaves).forEach((leaf, ordinal) => db.run("INSERT INTO leaf_order(result_id,ordinal,leaf_id) VALUES(?,?,?)", [resultId, ordinal, leaf]));
      result.hierarchy.merges.forEach((merge) => db.run("INSERT INTO hierarchy_merges(result_id,id,left_id,right_id,distance,mass) VALUES(?,?,?,?,?,?)", [resultId, merge.id, merge.left, merge.right, merge.distance, merge.mass]));
      for (const node of result.hierarchy.nodes || []) {
        db.run("INSERT INTO hierarchy_nodes(result_id,node_id,distance,mass,descendant_leaves_blob) VALUES(?,?,?,?,?)", [resultId, node.id, node.distance, node.mass, float32Blob(node.descendantLeaves)]);
        node.children.forEach((child, ordinal) => db.run("INSERT INTO hierarchy_children(result_id,parent_id,ordinal,child_id) VALUES(?,?,?,?)", [resultId, node.id, ordinal, child]));
      }
      (result.hierarchy.rootChildren || []).forEach((child, ordinal) => db.run("INSERT INTO root_children(result_id,ordinal,child_id) VALUES(?,?,?)", [resultId, ordinal, child]));
      const suppliedCoordinates = result.visualization?.coordinates || result.ids.map((path) => options.coordinates?.[path]);
      suppliedCoordinates.forEach((point, ordinal) => { if (!point) return; db.run("INSERT INTO visualization_points(result_id,ordinal,path,x,y,leaf_label) VALUES(?,?,?,?,?,?)", [resultId, ordinal, result.ids[ordinal], point[0], point[1], result.visualization?.labels[ordinal] ?? result.leafLabels[ordinal]]); });
      for (const [node, title] of Object.entries(result.titles || {})) db.run("INSERT INTO cluster_titles(result_id,node_id,title) VALUES(?,?,?)", [resultId, Number(node), title]);
      const memberships = result.softMemberships || result.memberships;
      if (memberships) memberships.forEach((row, ordinal) => {
        db.run("INSERT INTO membership_rows(result_id,path,width,values_blob) VALUES(?,?,?,?)", [resultId, result.ids[ordinal], row.length, float32Blob(row)]);
        if (result.schemaVersion < 6) row.forEach((membership, leafIndex) => db.run("INSERT INTO soft_memberships(result_id,path,leaf_id,membership) VALUES(?,?,?,?)", [resultId, result.ids[ordinal], (result.leafOrder || result.leafOrdering || result.hierarchy.leaves)[leafIndex], membership]));
      });
      for (const [path, memberships] of Object.entries(options.softMemberships || {})) for (const [leaf, membership] of Object.entries(memberships)) db.run("INSERT INTO soft_memberships(result_id,path,leaf_id,membership) VALUES(?,?,?,?)", [resultId, path, Number(leaf), membership]);
      (result.hierarchyPlacements || []).forEach((placement, ordinal) => db.run("INSERT INTO hierarchy_placements(result_id,path,ordinal,kind,node_id,confidence) VALUES(?,?,?,?,?,?)", [resultId, result.ids[ordinal], ordinal, placement.kind, placement.nodeId, placement.confidence]));
      // The explorer only needs the current structural result; keeping old
      // snapshots here otherwise dominates vault size for large vaults.
      db.run("DELETE FROM results WHERE result_id<>?", [resultId]);
    });
    return resultId;
  }
  async getResult(resultId?: string): Promise<ClusterResult | null> {
    this.requireOpen();
    const where = resultId ? ` WHERE result_id=${sqlQuote(resultId)}` : "";
    const rows = this.db.exec(`SELECT result_id,schema_version,result_json FROM results${where} ORDER BY created_at DESC,rowid DESC LIMIT 1`); const row = rows[0]?.values[0];
    if (!row?.[2]) return null;
    const stored = JSON.parse(String(row[2])) as ClusterResult & { _normalizedV6?: { memberships?: boolean; softMemberships?: boolean; titles?: boolean } };
    if (Number(row[1]) !== 6 || !stored._normalizedV6) return stored;
    const id = String(row[0]);
    const assignments = this.query("SELECT a.path,a.leaf_label AS leafLabel,a.probability,a.outlier_score AS outlierScore,a.provisional FROM hierarchy_placements p JOIN assignments a ON a.result_id=p.result_id AND a.path=p.path WHERE p.result_id=? ORDER BY p.ordinal", [id]);
    const ids = assignments.map((item) => String(item.path));
    const provisionalPaths = assignments.filter((item) => Number(item.provisional) !== 0).map((item) => String(item.path));
    const leaves = this.query("SELECT leaf_id AS leafId FROM hierarchy_leaves WHERE result_id=? ORDER BY ordinal", [id]).map((item) => Number(item.leafId));
    const leafOrdering = this.query("SELECT leaf_id AS leafId FROM leaf_order WHERE result_id=? ORDER BY ordinal", [id]).map((item) => Number(item.leafId));
    const membershipByPath = new Map(this.getMembershipRows(id).map((item) => [item.path, item.memberships])); const memberships = ids.map((path) => membershipByPath.get(path) || []);
    const hierarchyPlacements = this.getHierarchyPlacements(id).map(({ path: _path, ...placement }) => placement);
    const merges = this.query("SELECT id,left_id AS leftId,right_id AS rightId,distance,mass FROM hierarchy_merges WHERE result_id=? ORDER BY id", [id]).map((item) => ({ id: Number(item.id), left: Number(item.leftId), right: Number(item.rightId), distance: Number(item.distance), mass: Number(item.mass) }));
    const rootChildren = this.query("SELECT child_id AS childId FROM root_children WHERE result_id=? ORDER BY ordinal", [id]).map((item) => Number(item.childId));
    const points = this.getVisualization(id); const visualizationMetadata = stored.visualization;
    const titles = Object.fromEntries(this.query("SELECT node_id AS nodeId,title FROM cluster_titles WHERE result_id=? ORDER BY node_id", [id]).map((item) => [String(item.nodeId), String(item.title)]));
    const pcaStub = stored.pca.model as unknown as { modelHash?: string } | undefined; const pcaModel = pcaStub?.modelHash ? await this.getPcaModel(pcaStub.modelHash) : undefined;
    const umap = stored.umap ? {
      ...stored.umap,
      // JSON encodes an infinite singleton-leaf p95 as null. Restore the
      // sentinel so a singleton remains an intentionally open distance gate
      // after a SQLite round trip instead of becoming a zero-radius leaf.
      leafKnnDistanceP95: Object.fromEntries(Object.entries(stored.umap.leafKnnDistanceP95 || {}).map(([leaf, value]) => [leaf, value == null ? Number.POSITIVE_INFINITY : Number(value)])),
    } : undefined;
    return {
      schemaVersion: 6,
      ids,
      leafLabels: assignments.map((item) => Number(item.leafLabel)),
      probabilities: assignments.map((item) => Number(item.probability)),
      outlierProxy: assignments.map((item) => Number(item.outlierScore)),
      pca: { ...stored.pca, model: pcaModel ? { ...pcaModel, normalization: pcaModel.normalization === "l2" ? "l2" : "none" } : undefined },
      hierarchy: { leaves, merges, root: stored.hierarchy.root, nodes: this.getHierarchyNodes(id), rootChildren, splitMethod: "distance-knee-2-5" },
      hierarchyPlacements,
      leafOrder: leafOrdering,
      leafOrdering,
      ...(stored._normalizedV6.memberships ? { memberships, ...(stored._normalizedV6.softMemberships ? { softMemberships: memberships.map((values) => values.slice()) } : {}) } : {}),
      ...(visualizationMetadata ? { visualization: { ...visualizationMetadata, coordinates: points.map((point) => [point.x, point.y] as [number, number]), labels: points.map((point) => point.leafLabel), leafOrdering, memberships } } : {}),
      ...(stored._normalizedV6.titles ? { titles } : {}),
      ...(stored.titleGeneration ? { titleGeneration: stored.titleGeneration } : {}),
      ...(stored.embeddingProvider ? { embeddingProvider: String(stored.embeddingProvider) } : {}),
      ...(stored.embeddingModel ? { embeddingModel: String(stored.embeddingModel) } : {}),
      ...(umap ? { umap } : {}),
      ...(provisionalPaths.length ? { provisionalPaths } : {}),
      ...(stored.incremental ? { incremental: { ...stored.incremental, provisionalPaths } } : {}),
      timings: stored.timings || {}
    };
  }
  /** Return the note hashes captured by the last successfully saved result. */
  async getResultNoteHashes(resultId?: string): Promise<Map<string, string>> {
    this.requireOpen();
    const id = resultId || await this.getLatestResultId();
    if (!id) return new Map();
    return new Map(this.query("SELECT path,content_hash AS hash FROM result_note_hashes WHERE result_id=? ORDER BY path", [id]).map((row) => [String(row.path), String(row.hash)]));
  }
  async getNote(path: string): Promise<NoteRecord | undefined> {
    this.requireOpen(); const rows = this.db.exec(`SELECT path,title,mtime,content_hash,content FROM notes WHERE path=${sqlQuote(path)}`); const row = rows[0]?.values[0];
    if (!row) return undefined; const object = Object.fromEntries(rows[0].columns.map((key, i) => [key, row[i]]));
    return { path: String(object.path), title: String(object.title), mtime: Number(object.mtime), hash: String(object.content_hash), content: object.content == null ? "" : String(object.content) };
  }
  async getPcaCoordinates(path: string, modelHash: string): Promise<number[] | undefined> {
    this.requireOpen(); const rows = this.db.exec(`SELECT coordinates_json,coordinate_blob FROM pca_coordinates WHERE path=${sqlQuote(path)} AND model_hash=${sqlQuote(modelHash)}`);
    return rows[0]?.values[0]?.[1] ? float32Values(rows[0].values[0][1]) : rows[0]?.values[0]?.[0] ? JSON.parse(String(rows[0].values[0][0])) : undefined;
  }
  /**
   * Read PCA coordinates in the caller's path order. Missing paths remain
   * undefined, which lets incremental/lazy visualization callers distinguish
   * an absent projection from a valid zero-valued coordinate.
   */
  async getPcaCoordinatesMany(paths: readonly string[], modelHash: string): Promise<Array<number[] | undefined>> {
    this.requireOpen();
    if (!paths.length) return [];
    const placeholders = paths.map(() => "?").join(",");
    const rows = this.query(`SELECT path,coordinates_json,coordinate_blob FROM pca_coordinates WHERE model_hash=? AND path IN (${placeholders})`, [modelHash, ...paths]);
    const byPath = new Map(rows.map((row) => [String(row.path), row.coordinate_blob ? float32Values(row.coordinate_blob) : row.coordinates_json ? JSON.parse(String(row.coordinates_json)) as number[] : undefined]));
    return paths.map((path) => byPath.get(path));
  }
  getVisualization(resultId: string): Array<{ path: string; x: number; y: number; leafLabel: number }> {
    return this.query("SELECT path,x,y,leaf_label AS leafLabel FROM visualization_points WHERE result_id=? ORDER BY ordinal", [resultId]).map((row) => ({ path: String(row.path), x: Number(row.x), y: Number(row.y), leafLabel: Number(row.leafLabel) }));
  }
  getSoftMemberships(resultId: string): Array<{ path: string; leafId: number; membership: number }> {
    const scalar = this.query("SELECT path,leaf_id AS leafId,membership FROM soft_memberships WHERE result_id=? ORDER BY path,leaf_id", [resultId]).map((row) => ({ path: String(row.path), leafId: Number(row.leafId), membership: Number(row.membership) }));
    if (scalar.length) return scalar;
    const leaves = this.query("SELECT leaf_id AS leafId FROM leaf_order WHERE result_id=? ORDER BY ordinal", [resultId]).map((row) => Number(row.leafId));
    return this.getMembershipRows(resultId).flatMap((row) => row.memberships.map((membership, index) => ({ path: row.path, leafId: leaves[index], membership })));
  }
  getMembershipRows(resultId: string): Array<{ path: string; memberships: number[] }> {
    return this.query("SELECT m.path,m.values_blob,m.width FROM membership_rows m LEFT JOIN hierarchy_placements p ON p.result_id=m.result_id AND p.path=m.path WHERE m.result_id=? ORDER BY CASE WHEN p.ordinal IS NULL THEN 1 ELSE 0 END,p.ordinal,m.path", [resultId]).map((row) => ({ path: String(row.path), memberships: float32Values(row.values_blob).slice(0, Number(row.width)) }));
  }
  getHierarchyPlacements(resultId: string): Array<HierarchyPlacement & { path: string }> {
    return this.query("SELECT path,kind,node_id AS nodeId,confidence FROM hierarchy_placements WHERE result_id=? ORDER BY ordinal", [resultId]).map((row) => ({ path: String(row.path), kind: String(row.kind) as "leaf" | "residual", nodeId: row.nodeId == null ? null : Number(row.nodeId), confidence: Number(row.confidence) }));
  }
  getHierarchyNodes(resultId: string): Array<{ id: number; children: number[]; descendantLeaves: number[]; distance: number; mass: number }> {
    const nodes = this.query("SELECT node_id AS id,distance,mass,descendant_leaves_blob FROM hierarchy_nodes WHERE result_id=? ORDER BY node_id", [resultId]).map((row) => ({ id: Number(row.id), children: [] as number[], descendantLeaves: float32Values(row.descendant_leaves_blob).map(Math.trunc), distance: Number(row.distance), mass: Number(row.mass) }));
    for (const child of this.query("SELECT parent_id AS parentId,ordinal,child_id AS childId FROM hierarchy_children WHERE result_id=? ORDER BY parent_id,ordinal", [resultId])) nodes.find((node) => node.id === Number(child.parentId))?.children.push(Number(child.childId));
    return nodes;
  }
  /** Atomically replace titles while leaving every structural relation intact. */
  async getLatestResultId(): Promise<string | null> {
    this.requireOpen(); const rows = this.db.exec("SELECT result_id FROM results ORDER BY created_at DESC,rowid DESC LIMIT 1");
    return rows[0]?.values[0]?.[0] == null ? null : String(rows[0].values[0][0]);
  }
  async patchResultTitles(resultId: string, titles: Record<string, string>, titleGeneration?: ClusterResult["titleGeneration"]): Promise<void> {
    await this.transaction((db) => {
      const rows = db.exec(`SELECT result_json FROM results WHERE result_id=${sqlQuote(resultId)}`); const raw = rows[0]?.values[0]?.[0]; if (!raw) throw new Error(`Cluster result ${resultId} not found`);
      const result = JSON.parse(String(raw)) as ClusterResult & { _normalizedV6?: { titles?: boolean } }; const patched = result._normalizedV6 ? { ...result, _normalizedV6: { ...result._normalizedV6, titles: true }, ...(titleGeneration ? { titleGeneration } : {}) } : { ...result, titles: { ...titles }, ...(titleGeneration ? { titleGeneration } : {}) };
      db.run("UPDATE results SET result_json=? WHERE result_id=?", [JSON.stringify(patched), resultId]); db.run("DELETE FROM cluster_titles WHERE result_id=?", [resultId]);
      for (const [node, title] of Object.entries(titles)) db.run("INSERT INTO cluster_titles(result_id,node_id,title) VALUES(?,?,?)", [resultId, Number(node), title]);
    });
  }
  /**
   * Atomically attach a lazily-created v6 visualization to an existing
   * structural result. Only compact metadata and visualization point rows are
   * replaced; hierarchy, assignments, memberships, and placements are left
   * untouched so Explorer entry never duplicates the expensive result graph.
   */
  async patchResultVisualization(resultId: string, visualization: ClusterVisualization): Promise<void> {
    if (visualization.coordinates.length !== visualization.labels.length) throw new Error("Visualization coordinates and labels must align");
    if (visualization.coordinates.some((point) => point.length !== 2 || point.some((value) => !Number.isFinite(value)))) throw new Error("Visualization coordinates must be finite 2D points");
    await this.transaction((db) => {
      const rows = db.exec(`SELECT result_json,schema_version FROM results WHERE result_id=${sqlQuote(resultId)}`);
      const storedRow = rows[0]?.values[0];
      if (!storedRow?.[0]) throw new Error(`Cluster result ${resultId} not found`);
      if (Number(storedRow[1]) !== 6) throw new Error("Lazy visualization patches require schema v6");
      const stored = JSON.parse(String(storedRow[0])) as ClusterResult & { _normalizedV6?: Record<string, unknown> };
      if (!stored._normalizedV6) throw new Error("Cluster result is not normalized v6");
      const ids = this.query("SELECT p.path FROM hierarchy_placements p WHERE p.result_id=? ORDER BY p.ordinal", [resultId]).map((row) => String(row.path));
      if (ids.length !== visualization.coordinates.length) throw new Error(`Visualization rows must align with ${ids.length} result notes`);
      const patched = { ...stored, visualization: { configuration: visualization.configuration, timings: visualization.timings } };
      db.run("UPDATE results SET result_json=? WHERE result_id=?", [JSON.stringify(patched), resultId]);
      db.run("DELETE FROM visualization_points WHERE result_id=?", [resultId]);
      visualization.coordinates.forEach((point, ordinal) => db.run("INSERT INTO visualization_points(result_id,ordinal,path,x,y,leaf_label) VALUES(?,?,?,?,?,?)", [resultId, ordinal, ids[ordinal], point[0], point[1], visualization.labels[ordinal]]));
    });
  }
  query(sql: string, params: unknown[] = []): Array<Record<string, unknown>> {
    this.requireOpen(); const statement = this.db.prepare(sql); const result: Array<Record<string, unknown>> = [];
    try { statement.bind(params); while (statement.step()) result.push(statement.getAsObject()); } finally { statement.free(); } return result;
  }
  async saveEmbeddingLog(log: EmbeddingRunLog): Promise<void> { await this.transaction((db) => { db.run("DELETE FROM embedding_logs"); db.run("INSERT INTO embedding_logs(started_at,completed_at,provider,model,status,log_json) VALUES(?,?,?,?,?,?)", [log.startedAt, log.completedAt, log.provider, log.model, log.status || null, JSON.stringify(log)]); }); }
  async loadLatestEmbeddingLog(): Promise<EmbeddingRunLog | null> { this.requireOpen(); const rows = this.db.exec("SELECT log_json FROM embedding_logs ORDER BY id DESC LIMIT 1"); return rows[0]?.values[0]?.[0] ? JSON.parse(String(rows[0].values[0][0])) : null; }
}

/** Convenience factory for the async `initSqlJs({ locateFile })` API. */
export async function createSqliteStore(
  adapter: BinaryAdapter,
  initializer: SqlJsStatic | Promise<SqlJsStatic> | (() => Promise<SqlJsStatic>),
  options: SqliteStorageOptions = {}
): Promise<SqliteClusterStore> {
  const sql = typeof initializer === "function" ? await initializer() : await initializer;
  return new SqliteClusterStore(adapter, sql, options).open();
}

function sqlQuote(value: string): string { return `'${value.split("'").join("''")}'`; }

export interface LegacyJsonSources { embeddingCache?: string; result?: string; embeddingLog?: string; }
/** Import the old three JSON files exactly once. Invalid or absent files are ignored. */
export async function migrateLegacyJson(store: SqliteClusterStore, sources: LegacyJsonSources): Promise<{ migrated: string[] }> {
  const migrated: string[] = [];
  await store.transaction((db) => {
    const importOne = (name: string, raw: string | undefined, action: () => void) => {
      if (!raw || db.exec("SELECT 1 FROM migrations WHERE name=" + sqlQuote(name))[0]?.values.length) return;
      try { JSON.parse(raw); action(); db.run("INSERT INTO migrations(name,completed_at) VALUES(?,?)", [name, new Date().toISOString()]); migrated.push(name); } catch { /* leave it available for a later repair */ }
    };
    importOne("embedding-cache.json", sources.embeddingCache, () => {
      const document = JSON.parse(sources.embeddingCache!); for (const entry of document.embeddings || []) {
        // Legacy cache entries predate note metadata. A zero-metadata note is
        // upgraded in place when the vault is scanned next.
        db.run("INSERT OR IGNORE INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?)", [entry.path, entry.path.split("/").pop() || entry.path, 0, entry.hash || "legacy", null]);
        db.run("INSERT OR IGNORE INTO embeddings(path,provider,model,embedding_hash,note_content_hash,dimension,vector_json,vector_blob,created_at) VALUES(?,?,?,?,?,?,?,?,?)", [entry.path, entry.provider, entry.model, entry.hash, entry.hash, entry.vector.length, JSON.stringify(entry.vector), float32Blob(entry.vector), new Date().toISOString()]);
      }
    });
    importOne("cluster-result.json", sources.result, () => {
      const parsed = JSON.parse(sources.result!); const result = Number(parsed.schemaVersion) < 6 ? migrateLegacyResult(parsed) : parsed; if (!result) throw new Error("legacy cluster result is malformed");
      const id = `legacy-${Date.now()}`; const timestamp = new Date().toISOString();
      db.run("INSERT OR IGNORE INTO results(result_id,schema_version,created_at,result_json) VALUES(?,?,?,?)", [id, result.schemaVersion || 3, timestamp, result.schemaVersion === 6 ? compactV6Result(result) : JSON.stringify(result)]);
      for (const [i, path] of (result.ids || []).entries()) {
        db.run("INSERT OR IGNORE INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?)", [path, String(path).split("/").pop() || path, 0, "legacy", null]);
        db.run("INSERT OR IGNORE INTO assignments(result_id,path,leaf_label,probability,outlier_score) VALUES(?,?,?,?,?)", [id, path, result.leafLabels?.[i] ?? -1, result.probabilities?.[i] ?? 0, result.outlierProxy?.[i] ?? 0]);
      }
      if (result.schemaVersion === 6) persistMigratedResult(db, result, id, timestamp, false);
      else {
        for (const [ordinal, leaf] of (result.hierarchy?.leaves || []).entries()) db.run("INSERT OR IGNORE INTO hierarchy_leaves(result_id,ordinal,leaf_id) VALUES(?,?,?)", [id, ordinal, leaf]);
        for (const merge of result.hierarchy?.merges || []) db.run("INSERT OR IGNORE INTO hierarchy_merges(result_id,id,left_id,right_id,distance,mass) VALUES(?,?,?,?,?,?)", [id, merge.id, merge.left, merge.right, merge.distance, merge.mass]);
        for (const [node, title] of Object.entries(result.titles || {})) db.run("INSERT OR IGNORE INTO cluster_titles(result_id,node_id,title) VALUES(?,?,?)", [id, Number(node), title]);
      }
    });
    importOne("embedding-log.json", sources.embeddingLog, () => { const log = JSON.parse(sources.embeddingLog!); db.run("INSERT INTO embedding_logs(started_at,completed_at,provider,model,status,log_json) VALUES(?,?,?,?,?,?)", [log.startedAt || new Date().toISOString(), log.completedAt || new Date().toISOString(), log.provider || "unknown", log.model || "unknown", log.status || null, JSON.stringify(log)]); });
  });
  return { migrated };
}

/** Read legacy files from an Obsidian adapter and perform one-time migration. */
export async function migrateLegacyAdapter(store: SqliteClusterStore, adapter: { read(path: string): Promise<string>; rename?(oldPath: string, newPath: string): Promise<void> }): Promise<{ migrated: string[] }> {
  const read = async (path: string) => { try { return await adapter.read(path); } catch { return undefined; } };
  const paths: Record<string, string> = { "embedding-cache.json": ".obsidian/plugins/atomic-clusters/embedding-cache.json", "cluster-result.json": ".obsidian/plugins/atomic-clusters/cluster-result.json", "embedding-log.json": ".obsidian/plugins/atomic-clusters/embedding-log.json" };
  const source = { embeddingCache: await read(paths["embedding-cache.json"]), result: await read(paths["cluster-result.json"]), embeddingLog: await read(paths["embedding-log.json"]) };
  const migrated = await migrateLegacyJson(store, source);
  if (adapter.rename) for (const [name, path] of Object.entries(paths)) {
    const raw = name === "embedding-cache.json" ? source.embeddingCache : name === "cluster-result.json" ? source.result : source.embeddingLog;
    const imported = raw !== undefined && store.query("SELECT 1 AS imported FROM migrations WHERE name=?", [name]).length > 0;
    if (imported) try { await adapter.rename(path, `${path}.legacy`); } catch { /* a later startup retries this idempotent archive step */ }
  }
  return migrated;
}

export { SCHEMA as SQLITE_SCHEMA };
