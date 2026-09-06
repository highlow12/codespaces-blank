import type { ClusterResult, GeneratedClusterSnapshot, ManualCorrectionsState, NoteRecord } from "./types";

export type SearchFilter = "all" | "current-cluster" | "noise" | "provisional" | "manually-adjusted" | "recently-changed";

export interface SearchDocument {
  path: string;
  title: string;
  body?: string;
  tags?: string[];
  aliases?: string[];
  clusterIds: string[];
  clusterTerms?: string[];
  leafLabel?: number;
  provisional?: boolean;
  manuallyAdjusted?: boolean;
  manualPreferredClusterKey?: string;
  stableClusterKeys?: string[];
  manualGroupIds?: string[];
  mtime?: number;
}

export interface SearchClusterDocument {
  id: string;
  title: string;
  keywords?: string[];
  parentId?: string;
  stableClusterKey?: string;
  manualGroupId?: string;
}

export interface SearchFilters {
  currentClusterId?: string | null;
  noise?: boolean;
  provisional?: boolean;
  manuallyAdjusted?: boolean;
  recentlyChanged?: boolean;
  now?: number;
  recentlyChangedWindowMs?: number;
}

export interface ParsedSearchQuery {
  raw: string;
  terms: string[];
  phrases: string[];
  tags: string[];
  paths: string[];
  clusters: string[];
}

export interface SearchResult {
  query: ParsedSearchQuery;
  notePaths: string[];
  clusterIds: string[];
  matchedNotes: SearchDocument[];
  matchedClusters: SearchClusterDocument[];
}

const QUALIFIERS = new Set(["tag", "path", "cluster"]);
const DEFAULT_RECENT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

function normalize(value: unknown): string {
  return String(value ?? "").normalize("NFKC").replace(/\s+/g, " ").trim().toLocaleLowerCase();
}

function normalizeTag(value: string): string {
  return normalize(value).replace(/^#/, "");
}

function addToken(target: ParsedSearchQuery, value: string, quoted: boolean): void {
  const normalized = normalize(value);
  if (!normalized) return;
  if (quoted) target.phrases.push(normalized);
  else target.terms.push(normalized);
}

/** Parse the intentionally small Explorer query language without throwing on partial input. */
export function parseSearchQuery(raw: string): ParsedSearchQuery {
  const result: ParsedSearchQuery = { raw, terms: [], phrases: [], tags: [], paths: [], clusters: [] };
  const text = String(raw || "");
  let index = 0;
  while (index < text.length) {
    while (/\s/.test(text[index] || "")) index++;
    if (index >= text.length) break;
    const start = index;
    while (index < text.length && !/\s/.test(text[index])) index++;
    const atom = text.slice(start, index);
    const colon = atom.indexOf(":");
    const prefix = colon > 0 ? normalize(atom.slice(0, colon)) : "";
    let value = colon > 0 ? atom.slice(colon + 1) : atom;
    let quoted = false;
    // Accept tag:"two words"/cluster:"two words" as a convenience while a
    // user is still typing a qualifier. The closing quote is optional.
    if (QUALIFIERS.has(prefix) && value.startsWith('"')) {
      quoted = true;
      value = value.slice(1);
      if (!value.endsWith('"')) {
        const quote = text.indexOf('"', index);
        if (quote >= 0) { value += ` ${text.slice(index, quote)}`; index = quote + 1; }
        else { value += text.slice(index); index = text.length; }
      } else value = value.slice(0, -1);
    } else if (atom.startsWith('"')) {
      quoted = true;
      value = atom.slice(1);
      if (value.endsWith('"')) value = value.slice(0, -1);
      else {
        const quote = text.indexOf('"', index);
        if (quote >= 0) { value += ` ${text.slice(index, quote)}`; index = quote + 1; }
        else { value += text.slice(index); index = text.length; }
      }
    }
    if (QUALIFIERS.has(prefix) && value) {
      if (prefix === "tag") result.tags.push(normalizeTag(value));
      else if (prefix === "path") result.paths.push(normalize(value).replace(/^\.\//, ""));
      else result.clusters.push(normalize(value));
    } else addToken(result, value, quoted);
  }
  return result;
}

function frontmatterList(body: string, key: "tags" | "aliases"): string[] {
  const match = body.match(/^---\s*\n([\s\S]*?)\n---(?:\s*\n|$)/);
  if (!match) return [];
  const line = match[1].split(/\r?\n/).find((item) => new RegExp(`^\\s*${key}\\s*:`, "i").test(item));
  if (!line) return [];
  const value = line.replace(new RegExp(`^\\s*${key}\\s*:\\s*`, "i"), "").trim();
  if (!value) return [];
  const unwrapped = value.replace(/^\[/, "").replace(/\]$/, "");
  return unwrapped.split(",").map((item) => item.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean).map((item) => key === "tags" ? normalizeTag(item) : normalize(item));
}

function metadataFromNote(note: NoteRecord): { tags: string[]; aliases: string[] } {
  return { tags: frontmatterList(note.content || "", "tags"), aliases: frontmatterList(note.content || "", "aliases") };
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

/** Derive the same generated evidence used by SQLite without changing the result. */
export function generatedSearchClusterSnapshots(result: ClusterResult): GeneratedClusterSnapshot[] {
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

const EMPTY_MANUAL_CORRECTIONS: ManualCorrectionsState = { titleOverrides: [], notePreferences: [], groups: [], feedback: [] };

function effectiveTitles(result: ClusterResult, manualCorrections: ManualCorrectionsState): Record<string, string> {
  const titles = { ...(result.titles || {}) };
  const snapshots = generatedSearchClusterSnapshots(result);
  const nodesByKey = new Map(snapshots.map((snapshot) => [snapshot.stableClusterKey, snapshot.nodeId]));
  for (const override of manualCorrections.titleOverrides || []) {
    const nodeId = nodesByKey.get(override.stableClusterKey);
    if (nodeId !== undefined && !override.orphaned && override.title.trim()) titles[String(nodeId)] = override.title.trim();
  }
  return titles;
}

function clusterHierarchy(result: ClusterResult, manualCorrections: ManualCorrectionsState = EMPTY_MANUAL_CORRECTIONS): { clusters: SearchClusterDocument[]; parents: Map<string, string>; snapshots: GeneratedClusterSnapshot[]; nodeKeys: Map<string, string>; titles: Record<string, string> } {
  const parents = new Map<string, string>();
  const nodes = result.hierarchy.nodes || [];
  const snapshots = generatedSearchClusterSnapshots(result);
  const nodeKeys = new Map(snapshots.map((snapshot) => [String(snapshot.nodeId), snapshot.stableClusterKey]));
  const titles = effectiveTitles(result, manualCorrections);
  for (const node of nodes) for (const child of node.children) parents.set(String(child), String(node.id));
  const clusters: SearchClusterDocument[] = [];
  const ids = new Set<number>([
    ...(result.hierarchy.leaves || []),
    ...(result.hierarchy.merges || []).map((merge) => merge.id),
    ...nodes.map((node) => node.id),
  ]);
  for (const id of [...ids].sort((left, right) => left - right)) {
    clusters.push({
      id: String(id),
      title: titles[String(id)] || `Cluster ${id}`,
      keywords: result.titleGeneration?.scores?.[String(id)]?.map((item) => item.keyword) || [],
      parentId: parents.get(String(id)),
      stableClusterKey: nodeKeys.get(String(id)),
    });
  }
  clusters.push({ id: "root", title: "All notes", keywords: [] });
  for (const group of manualCorrections.groups || []) {
    if (!group.title.trim()) continue;
    clusters.push({ id: `manual-group:${group.groupId}`, title: group.title, keywords: [], manualGroupId: group.groupId });
  }
  return { clusters, parents, snapshots, nodeKeys, titles };
}

function ancestorIds(id: string, parents: ReadonlyMap<string, string>): string[] {
  const values: string[] = [];
  const seen = new Set<string>();
  let current: string | undefined = id;
  while (current && !seen.has(current)) { seen.add(current); values.push(current); current = parents.get(current); }
  values.push("root");
  return [...new Set(values)];
}

/** Build searchable note/cluster metadata from the currently persisted result. */
export function buildSearchDocuments(notes: readonly NoteRecord[], result: ClusterResult, suppliedManualCorrections?: ManualCorrectionsState): { documents: SearchDocument[]; clusters: SearchClusterDocument[] } {
  const manualCorrections = suppliedManualCorrections || EMPTY_MANUAL_CORRECTIONS;
  const { clusters, parents, snapshots, nodeKeys, titles } = clusterHierarchy(result, manualCorrections);
  const notesByPath = new Map(notes.map((note) => [note.path, note] as const));
  const provisional = new Set(result.provisionalPaths || result.incremental?.provisionalPaths || []);
  const overrideKeys = new Set((manualCorrections.titleOverrides || []).filter((override) => !override.orphaned).map((override) => override.stableClusterKey));
  const preferences = new Map((manualCorrections.notePreferences || []).map((preference) => [preference.notePath, preference.preferredClusterKey]));
  const groups = (manualCorrections.groups || []).filter((group) => group.title.trim());
  const documents = result.ids.map((path, index) => {
    const note = notesByPath.get(path);
    const placement = result.hierarchyPlacements?.[index];
    const terminal = placement?.nodeId == null ? "root" : String(placement.nodeId);
    const leaf = result.leafLabels[index];
    const clusterIds = terminal === "root" ? ["root"] : ancestorIds(terminal, parents);
    if (leaf >= 0 && !clusterIds.includes(String(leaf))) clusterIds.unshift(...ancestorIds(String(leaf), parents));
    const metadata = note ? metadataFromNote(note) : { tags: [], aliases: [] };
    const stableClusterKeys = [...new Set(clusterIds.map((id) => nodeKeys.get(id)).filter((key): key is string => !!key))];
    const manualPreferredClusterKey = preferences.get(path);
    const manualGroupIds = groups.filter((group) => group.childClusterKeys.some((key) => stableClusterKeys.includes(key))).map((group) => group.groupId);
    const clusterTerms = clusterIds.flatMap((id) => {
      const cluster = clusters.find((item) => item.id === id);
      return cluster ? [cluster.title, ...(cluster.keywords || [])] : [];
    });
    const preferredSnapshot = manualPreferredClusterKey ? snapshots.find((snapshot) => snapshot.stableClusterKey === manualPreferredClusterKey) : undefined;
    const legacyPreferredNodeId = manualPreferredClusterKey && /^-?\d+$/.test(manualPreferredClusterKey) ? Number(manualPreferredClusterKey) : undefined;
    const preferredTitle = preferredSnapshot ? titles[String(preferredSnapshot.nodeId)] : legacyPreferredNodeId === undefined ? undefined : titles[String(legacyPreferredNodeId)];
    if (preferredTitle && !clusterTerms.some((term) => term.toLocaleLowerCase() === preferredTitle.toLocaleLowerCase())) clusterTerms.push(preferredTitle);
    for (const groupId of manualGroupIds) {
      const group = groups.find((item) => item.groupId === groupId);
      if (group) clusterTerms.push(group.title);
    }
    const manuallyAdjusted = !!manualPreferredClusterKey || stableClusterKeys.some((key) => overrideKeys.has(key)) || manualGroupIds.length > 0;
    return {
      path,
      title: note?.title || path.split("/").pop()?.replace(/\.md$/i, "") || path,
      body: note?.content || "",
      tags: metadata.tags,
      aliases: metadata.aliases,
      clusterIds: [...new Set(clusterIds)],
      clusterTerms,
      leafLabel: leaf,
      provisional: provisional.has(path),
      manuallyAdjusted,
      ...(manualPreferredClusterKey ? { manualPreferredClusterKey } : {}),
      ...(stableClusterKeys.length ? { stableClusterKeys } : {}),
      ...(manualGroupIds.length ? { manualGroupIds } : {}),
      mtime: note?.mtime,
    } satisfies SearchDocument;
  });
  for (const group of groups) {
    const cluster = clusters.find((item) => item.manualGroupId === group.groupId);
    if (cluster) cluster.keywords = group.childClusterKeys.slice();
  }
  return { documents, clusters };
}

function fieldText(document: SearchDocument): string {
  return normalize([
    document.path,
    document.title,
    document.body || "",
    ...(document.tags || []),
    ...(document.aliases || []),
    ...(document.clusterTerms || []),
  ].join(" "));
}

function clusterFieldText(cluster: SearchClusterDocument): string {
  return normalize([cluster.id, cluster.title, ...(cluster.keywords || [])].join(" "));
}

function recentTimestamp(mtime: number | undefined): number {
  if (!Number.isFinite(mtime)) return 0;
  // Obsidian stats are milliseconds; accepting seconds keeps fixtures and
  // older adapters useful without making the filter silently empty.
  return Number(mtime) < 1e11 ? Number(mtime) * 1000 : Number(mtime);
}

function matchesFilters(document: SearchDocument, filters: SearchFilters): boolean {
  if (filters.currentClusterId && !document.clusterIds.includes(filters.currentClusterId)) return false;
  if (filters.noise && (document.leafLabel ?? -1) >= 0) return false;
  if (filters.provisional && !document.provisional) return false;
  if (filters.manuallyAdjusted && !document.manuallyAdjusted) return false;
  if (filters.recentlyChanged) {
    const now = filters.now ?? Date.now();
    const windowMs = filters.recentlyChangedWindowMs ?? DEFAULT_RECENT_WINDOW_MS;
    if (recentTimestamp(document.mtime) < now - windowMs) return false;
  }
  return true;
}

function matchesQuery(document: SearchDocument, query: ParsedSearchQuery, indexedText = fieldText(document)): boolean {
  const text = indexedText;
  if (query.terms.some((term) => !text.includes(term))) return false;
  if (query.phrases.some((phrase) => !text.includes(phrase))) return false;
  if (query.tags.some((tag) => !(document.tags || []).some((item) => normalizeTag(item) === tag || normalizeTag(item).includes(tag)))) return false;
  if (query.paths.some((path) => !normalize(document.path).includes(path))) return false;
  if (query.clusters.some((cluster) => !(document.clusterTerms || []).some((item) => normalize(item).includes(cluster)))) return false;
  return true;
}

export class SearchIndex {
  private documents: SearchDocument[];
  private clusters: SearchClusterDocument[];
  private readonly textByPath = new Map<string, string>();
  constructor(documents: readonly SearchDocument[] = [], clusters: readonly SearchClusterDocument[] = []) {
    this.documents = [];
    this.clusters = [];
    this.replace(documents, clusters);
  }
  replace(documents: readonly SearchDocument[], clusters: readonly SearchClusterDocument[] = []): void {
    this.documents = documents.slice();
    this.clusters = clusters.slice();
    this.textByPath.clear();
    for (const document of this.documents) this.textByPath.set(document.path, fieldText(document));
  }
  get size(): number { return this.documents.length; }
  search(raw: string | ParsedSearchQuery, filters: SearchFilters = {}): SearchResult {
    const query = typeof raw === "string" ? parseSearchQuery(raw) : raw;
    const matchedNotes = this.documents.filter((document) => matchesFilters(document, filters) && matchesQuery(document, query, this.textByPath.get(document.path) || ""));
    const matchedNotePaths = new Set(matchedNotes.map((document) => document.path));
    const queryMatchesCluster = (cluster: SearchClusterDocument): boolean => {
      // Note-only qualifiers cannot be answered from cluster metadata. Their
      // matching cluster IDs are added below from matching documents instead
      // of incorrectly highlighting every cluster.
      if (query.tags.length || query.paths.length) return false;
      const text = clusterFieldText(cluster);
      return query.terms.every((term) => text.includes(term)) && query.phrases.every((phrase) => text.includes(phrase)) && query.clusters.every((term) => text.includes(term));
    };
    const matchedClusters = this.clusters.filter((cluster) => queryMatchesCluster(cluster) && (!filters.currentClusterId || cluster.id === filters.currentClusterId || matchedNotes.some((note) => note.clusterIds.includes(cluster.id))));
    const clusterIds = new Set<string>(matchedClusters.map((cluster) => cluster.id));
    for (const note of matchedNotes) for (const id of note.clusterIds) clusterIds.add(id);
    return { query, notePaths: [...matchedNotePaths].sort(), clusterIds: [...clusterIds].sort(), matchedNotes, matchedClusters };
  }
}

export { DEFAULT_RECENT_WINDOW_MS };
