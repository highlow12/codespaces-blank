import type { ClusterResult, GeneratedClusterSnapshot, ManualCorrectionsState, NoteRecord } from "./types";
import { buildVisualizationTree, visualizationNoteTerminalPath, visualizationTopMemberships } from "./visualization";

export interface NoteDetailMembership {
  leafId: number;
  title: string;
  value: number;
}

export interface NoteDetailAncestor {
  id: string;
  title: string;
}

export interface NoteDetailRelatedNote {
  path: string;
  title: string;
  similarity: number;
}

export interface NoteDetailPreferredCandidate {
  key: string;
  title: string;
  value: number;
  leafId: number;
  automatic: boolean;
}

export interface NoteDetailModel {
  path: string;
  title: string;
  automaticLeaf: { id: number; title: string } | null;
  ancestors: NoteDetailAncestor[];
  probability: number | null;
  strongestMembership: number | null;
  memberships: NoteDetailMembership[];
  noise: boolean;
  residual: boolean;
  provisional: boolean;
  manualPreferredCluster: { key: string; title: string } | null;
  preferredClusterCandidates: NoteDetailPreferredCandidate[];
  relatedNotes: NoteDetailRelatedNote[];
  clusterKeywords: string[];
}

const EMPTY_MANUAL_CORRECTIONS: ManualCorrectionsState = { titleOverrides: [], notePreferences: [], groups: [], feedback: [] };

function fallbackNoteTitle(path: string): string {
  return path.split("/").pop()?.replace(/\.md$/i, "") || path;
}

function normalizeMemberPaths(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => String(value || "").normalize("NFKC").trim().replace(/\\/g, "/").replace(/^\.\/+/, "").replace(/^\/+/, "").replace(/\/{2,}/g, "/").replace(/\/+$/, "").trim()).filter(Boolean))].sort();
}

function stableClusterFingerprint(memberPaths: readonly string[]): string {
  const canonical = JSON.stringify(normalizeMemberPaths(memberPaths));
  let hash = 2166136261;
  for (let index = 0; index < canonical.length; index++) { hash ^= canonical.charCodeAt(index); hash = Math.imul(hash, 16777619); }
  return `cluster-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function generatedSnapshots(result: ClusterResult): GeneratedClusterSnapshot[] {
  const membersById = new Map<number, Set<string>>();
  const add = (id: number, paths: Iterable<string>): void => {
    const members = membersById.get(id) || new Set<string>();
    for (const path of paths) { const normalized = normalizeMemberPaths([path])[0]; if (normalized) members.add(normalized); }
    membersById.set(id, members);
  };
  result.leafLabels.forEach((label, index) => { if (Number.isSafeInteger(label) && label >= 0) add(label, [result.ids[index]]); });
  for (const node of result.hierarchy.nodes || []) add(node.id, node.descendantLeaves.flatMap((leaf) => [...(membersById.get(leaf) || [])]));
  const merges = new Map(result.hierarchy.merges.map((merge) => [merge.id, merge]));
  const visiting = new Set<number>();
  const visit = (id: number): Set<string> => {
    const existing = membersById.get(id); if (existing?.size) return existing;
    if (visiting.has(id)) return new Set();
    const merge = merges.get(id); if (!merge) return existing || new Set();
    visiting.add(id); const paths = new Set<string>([...visit(merge.left), ...visit(merge.right)]); visiting.delete(id); membersById.set(id, paths); return paths;
  };
  for (const merge of result.hierarchy.merges) visit(merge.id);
  const ids = new Set<number>([...(result.hierarchy.leaves || []), ...result.hierarchy.merges.map((merge) => merge.id), ...(result.hierarchy.nodes || []).map((node) => node.id)]);
  const unique = new Map<string, GeneratedClusterSnapshot>();
  for (const nodeId of [...ids].sort((left, right) => left - right)) {
    const memberPaths = normalizeMemberPaths([...(membersById.get(nodeId) || [])]);
    if (!memberPaths.length) continue;
    const stableClusterKey = stableClusterFingerprint(memberPaths);
    if (!unique.has(stableClusterKey)) unique.set(stableClusterKey, { stableClusterKey, nodeId, memberPaths });
  }
  return [...unique.values()];
}

function effectiveTitleMap(result: ClusterResult, manualCorrections: ManualCorrectionsState): { titles: Record<string, string>; snapshots: GeneratedClusterSnapshot[] } {
  const snapshots = generatedSnapshots(result);
  const titles = { ...(result.titles || {}) };
  const nodeByKey = new Map(snapshots.map((snapshot) => [snapshot.stableClusterKey, snapshot.nodeId]));
  for (const override of manualCorrections.titleOverrides || []) {
    const nodeId = nodeByKey.get(override.stableClusterKey);
    if (nodeId !== undefined && !override.orphaned && override.title.trim()) titles[String(nodeId)] = override.title.trim();
  }
  return { titles, snapshots };
}

function nodeTitle(result: ClusterResult, id: string, titleMap: { titles: Record<string, string>; snapshots: GeneratedClusterSnapshot[] }): string {
  if (id === "root") return "All notes";
  return titleMap.titles[id.replace(/^node:/, "")]?.trim() || `Cluster ${id.replace(/^node:/, "")}`;
}

function preferredClusterFor(manualCorrections: ManualCorrectionsState, path: string): string | null {
  return manualCorrections.notePreferences.find((preference) => preference.notePath === path)?.preferredClusterKey || null;
}

function finiteProbability(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : null;
}

function finiteCoordinates(result: ClusterResult): number[][] | null {
  const visualization = result.visualization?.coordinates;
  if (visualization?.length === result.ids.length && visualization.every((row) => row.length >= 2 && row.every(Number.isFinite))) return visualization.map((row) => row.slice());
  const umap = result.umap?.coordinates;
  if (umap?.length === result.ids.length && umap.length > 0 && umap.every((row) => row.length > 0 && row.every(Number.isFinite))) return umap.map((row) => row.slice());
  return null;
}

function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  const width = Math.min(left.length, right.length);
  let dot = 0; let leftNorm = 0; let rightNorm = 0;
  for (let index = 0; index < width; index++) { const a = Number(left[index]) || 0; const b = Number(right[index]) || 0; dot += a * b; leftNorm += a * a; rightNorm += b * b; }
  if (!leftNorm || !rightNorm) return 0;
  return Math.max(0, Math.min(1, dot / Math.sqrt(leftNorm * rightNorm)));
}

function relatedNotes(result: ClusterResult, notes: readonly NoteRecord[], selectedIndex: number, limit: number): NoteDetailRelatedNote[] {
  const selectedPath = result.ids[selectedIndex];
  const noteTitles = new Map(notes.map((note) => [note.path, note.title || fallbackNoteTitle(note.path)]));
  const coordinates = finiteCoordinates(result);
  const memberships = result.memberships || result.softMemberships;
  const selectedCoordinates = coordinates?.[selectedIndex];
  const selectedMembership = memberships?.[selectedIndex];
  const scored = result.ids.map((path, index) => {
    if (index === selectedIndex) return null;
    let similarity: number | null = null;
    if (selectedCoordinates && coordinates?.[index]) {
      const other = coordinates[index]; const width = Math.min(selectedCoordinates.length, other.length); let distanceSquared = 0;
      for (let dimension = 0; dimension < width; dimension++) distanceSquared += (selectedCoordinates[dimension] - other[dimension]) ** 2;
      similarity = 1 / (1 + Math.sqrt(distanceSquared));
    } else if (selectedMembership && memberships?.[index]) similarity = cosineSimilarity(selectedMembership, memberships[index]);
    if (similarity === null || !Number.isFinite(similarity)) return null;
    return { path, title: noteTitles.get(path) || fallbackNoteTitle(path), similarity: Math.max(0, Math.min(1, similarity)) };
  }).filter((item): item is NoteDetailRelatedNote => !!item);
  return scored.sort((left, right) => right.similarity - left.similarity || left.path.localeCompare(right.path)).slice(0, Math.max(0, Math.trunc(limit)));
}

function ancestorPath(result: ClusterResult, index: number): string[] {
  try {
    const root = buildVisualizationTree(result.hierarchy, result.leafLabels);
    return visualizationNoteTerminalPath(root, index, result.leafLabels, result.hierarchyPlacements);
  } catch {
    const leaf = result.leafLabels[index];
    return Number.isSafeInteger(leaf) && leaf >= 0 ? ["root", `node:${leaf}`] : ["root"];
  }
}

function clusterKeywords(result: ClusterResult, ancestors: readonly NoteDetailAncestor[], titleMap: { titles: Record<string, string>; snapshots: GeneratedClusterSnapshot[] }): string[] {
  const keywords: string[] = [];
  const add = (value: string): void => { const normalized = value.trim(); if (normalized && !keywords.some((item) => item.toLocaleLowerCase() === normalized.toLocaleLowerCase())) keywords.push(normalized); };
  for (const ancestor of ancestors) {
    const id = ancestor.id.replace(/^node:/, "");
    for (const score of result.titleGeneration?.scores?.[id] || []) add(score.keyword);
  }
  if (!keywords.length) for (const ancestor of ancestors) if (ancestor.id !== "root") for (const word of (titleMap.titles[ancestor.id.replace(/^node:/, "")] || "").split(" · ")) add(word);
  return keywords.slice(0, 12);
}

function preferredClusterCandidates(result: ClusterResult, selectedPath: string, limit: number, titleMap: { titles: Record<string, string>; snapshots: GeneratedClusterSnapshot[] }): NoteDetailPreferredCandidate[] {
  const index = result.ids.indexOf(selectedPath);
  if (index < 0) return [];
  const ordering = result.leafOrdering || result.leafOrder || result.hierarchy.leaves;
  const row = result.memberships?.[index] || result.softMemberships?.[index] || [];
  const snapshotsByNode = new Map(titleMap.snapshots.map((snapshot) => [snapshot.nodeId, snapshot]));
  const automaticLeaf = Number.isSafeInteger(result.leafLabels[index]) ? result.leafLabels[index] : -1;
  const candidates = ordering.map((leafId, column) => {
    const snapshot = snapshotsByNode.get(leafId);
    if (!snapshot) return null;
    const value = Number.isFinite(row[column]) ? Math.max(0, Math.min(1, Number(row[column]))) : leafId === automaticLeaf ? 1 : 0;
    return { key: snapshot.stableClusterKey, title: titleMap.titles[String(leafId)]?.trim() || `Cluster ${leafId}`, value, leafId, automatic: leafId === automaticLeaf } satisfies NoteDetailPreferredCandidate;
  }).filter((candidate): candidate is NoteDetailPreferredCandidate => !!candidate);
  return candidates.sort((left, right) => right.value - left.value || Number(right.automatic) - Number(left.automatic) || left.leafId - right.leafId).slice(0, Math.min(5, Math.max(1, Math.trunc(limit))));
}

/** Return at most five relevant leaf candidates for a persisted note preference picker. */
export function getPreferredClusterCandidates(result: ClusterResult, selectedPath: string, manualCorrections: ManualCorrectionsState = EMPTY_MANUAL_CORRECTIONS, limit = 5): NoteDetailPreferredCandidate[] {
  return preferredClusterCandidates(result, selectedPath, Math.min(5, Math.max(1, Math.trunc(limit))), effectiveTitleMap(result, manualCorrections));
}

/** Build a read-only detail model from the saved result and in-memory metadata. */
export function buildNoteDetail(result: ClusterResult, notes: readonly NoteRecord[], selectedPath: string, relatedLimitOrCorrections: number | ManualCorrectionsState = 5, suppliedManualCorrections?: ManualCorrectionsState): NoteDetailModel | null {
  const relatedLimit = typeof relatedLimitOrCorrections === "number" ? relatedLimitOrCorrections : 5;
  const manualCorrections = typeof relatedLimitOrCorrections === "number" ? suppliedManualCorrections || EMPTY_MANUAL_CORRECTIONS : relatedLimitOrCorrections;
  const index = result.ids.indexOf(selectedPath);
  if (index < 0) return null;
  const note = notes.find((item) => item.path === selectedPath);
  const title = note?.title || fallbackNoteTitle(selectedPath);
  const leafId = Number.isSafeInteger(result.leafLabels[index]) && result.leafLabels[index] >= 0 ? result.leafLabels[index] : null;
  const placement = result.hierarchyPlacements?.[index];
  const noise = leafId === null;
  const residual = placement?.kind === "residual";
  const provisional = new Set(result.provisionalPaths || result.incremental?.provisionalPaths || []).has(selectedPath);
  const pathIds = ancestorPath(result, index);
  const titleMap = effectiveTitleMap(result, manualCorrections);
  const ancestors = pathIds.map((id) => ({ id, title: nodeTitle(result, id, titleMap) }));
  const displayResult = { ...result, titles: titleMap.titles };
  const memberships = visualizationTopMemberships(displayResult, index, 5).map((item) => ({ leafId: item.leafId, title: item.title, value: Math.max(0, Math.min(1, item.value)) }));
  const preferredKey = preferredClusterFor(manualCorrections, selectedPath);
  const preferredSnapshot = preferredKey ? titleMap.snapshots.find((snapshot) => snapshot.stableClusterKey === preferredKey) : undefined;
  const preferredNodeId = preferredSnapshot?.nodeId ?? (/^-?\d+$/.test(preferredKey || "") ? Number(preferredKey) : undefined);
  const preferredTitle = preferredNodeId === undefined ? preferredKey || "" : titleMap.titles[String(preferredNodeId)] || preferredKey || "";
  return {
    path: selectedPath,
    title,
    automaticLeaf: leafId === null ? null : { id: leafId, title: nodeTitle(result, `node:${leafId}`, titleMap) },
    ancestors,
    probability: finiteProbability(result.probabilities[index]),
    strongestMembership: memberships[0]?.value ?? null,
    memberships,
    noise,
    residual,
    provisional,
    manualPreferredCluster: preferredKey ? { key: preferredKey, title: preferredTitle || preferredKey } : null,
    preferredClusterCandidates: getPreferredClusterCandidates(result, selectedPath, manualCorrections, 5),
    relatedNotes: relatedNotes(result, notes, index, relatedLimit),
    clusterKeywords: clusterKeywords(result, ancestors, titleMap),
  };
}
