import type { ClusterResult, NoteRecord } from "./types";
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
  relatedNotes: NoteDetailRelatedNote[];
  clusterKeywords: string[];
}

interface PreferenceCarrier {
  manualPreferredClusters?: Readonly<Record<string, unknown>>;
  manualPreferred?: Readonly<Record<string, unknown>>;
  manualPreferences?: Readonly<Record<string, unknown>>;
  notePreferences?: Readonly<Record<string, unknown>>;
  manualAdjustments?: Readonly<Record<string, unknown>>;
}

function resultWithOptionalPreferences(result: ClusterResult): ClusterResult & PreferenceCarrier {
  return result as ClusterResult & PreferenceCarrier;
}

function fallbackNoteTitle(path: string): string {
  return path.split("/").pop()?.replace(/\.md$/i, "") || path;
}

function nodeTitle(result: ClusterResult, id: string): string {
  if (id === "root") return "All notes";
  return result.titles?.[id.replace(/^node:/, "")]?.trim() || `Cluster ${id.replace(/^node:/, "")}`;
}

function preferenceKey(value: unknown): string | null {
  if (typeof value === "string") return value.trim() || null;
  if (!value || typeof value !== "object") return null;
  const object = value as { preferredClusterKey?: unknown; clusterKey?: unknown; key?: unknown };
  for (const candidate of [object.preferredClusterKey, object.clusterKey, object.key]) if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  return null;
}

function preferredClusterFor(result: ClusterResult, path: string): string | null {
  const carrier = resultWithOptionalPreferences(result);
  for (const source of [carrier.manualPreferredClusters, carrier.manualPreferred, carrier.notePreferences, carrier.manualPreferences, carrier.manualAdjustments]) {
    const key = preferenceKey(source?.[path]);
    if (key) return key;
  }
  return null;
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

function clusterKeywords(result: ClusterResult, ancestors: readonly NoteDetailAncestor[]): string[] {
  const keywords: string[] = [];
  const add = (value: string): void => { const normalized = value.trim(); if (normalized && !keywords.some((item) => item.toLocaleLowerCase() === normalized.toLocaleLowerCase())) keywords.push(normalized); };
  for (const ancestor of ancestors) {
    const id = ancestor.id.replace(/^node:/, "");
    for (const score of result.titleGeneration?.scores?.[id] || []) add(score.keyword);
  }
  if (!keywords.length) for (const ancestor of ancestors) if (ancestor.id !== "root") for (const word of (result.titles?.[ancestor.id.replace(/^node:/, "")] || "").split(" · ")) add(word);
  return keywords.slice(0, 12);
}

/** Build a read-only detail model from the saved result and in-memory metadata. */
export function buildNoteDetail(result: ClusterResult, notes: readonly NoteRecord[], selectedPath: string, relatedLimit = 5): NoteDetailModel | null {
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
  const ancestors = pathIds.map((id) => ({ id, title: nodeTitle(result, id) }));
  const memberships = visualizationTopMemberships(result, index, 5).map((item) => ({ leafId: item.leafId, title: item.title, value: Math.max(0, Math.min(1, item.value)) }));
  const preferredKey = preferredClusterFor(result, selectedPath);
  const preferredTitle = preferredKey ? nodeTitle(result, preferredKey.startsWith("node:") ? preferredKey : `node:${preferredKey}`) : "";
  return {
    path: selectedPath,
    title,
    automaticLeaf: leafId === null ? null : { id: leafId, title: nodeTitle(result, `node:${leafId}`) },
    ancestors,
    probability: finiteProbability(result.probabilities[index]),
    strongestMembership: memberships[0]?.value ?? null,
    memberships,
    noise,
    residual,
    provisional,
    manualPreferredCluster: preferredKey ? { key: preferredKey, title: preferredTitle || preferredKey } : null,
    relatedNotes: relatedNotes(result, notes, index, relatedLimit),
    clusterKeywords: clusterKeywords(result, ancestors),
  };
}
