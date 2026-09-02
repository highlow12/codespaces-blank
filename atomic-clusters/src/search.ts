import type { ClusterResult, NoteRecord } from "./types";

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
  mtime?: number;
}

export interface SearchClusterDocument {
  id: string;
  title: string;
  keywords?: string[];
  parentId?: string;
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

function clusterHierarchy(result: ClusterResult): { clusters: SearchClusterDocument[]; parents: Map<string, string> } {
  const parents = new Map<string, string>();
  const nodes = result.hierarchy.nodes || [];
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
      title: result.titles?.[String(id)] || `Cluster ${id}`,
      keywords: result.titleGeneration?.scores?.[String(id)]?.map((item) => item.keyword) || [],
      parentId: parents.get(String(id)),
    });
  }
  clusters.push({ id: "root", title: "All notes", keywords: [] });
  return { clusters, parents };
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
export function buildSearchDocuments(notes: readonly NoteRecord[], result: ClusterResult): { documents: SearchDocument[]; clusters: SearchClusterDocument[] } {
  const { clusters, parents } = clusterHierarchy(result);
  const notesByPath = new Map(notes.map((note) => [note.path, note] as const));
  const provisional = new Set(result.provisionalPaths || result.incremental?.provisionalPaths || []);
  const manualAdjustments = (result as ClusterResult & { manualAdjustments?: Record<string, unknown> }).manualAdjustments || {};
  const documents = result.ids.map((path, index) => {
    const note = notesByPath.get(path);
    const placement = result.hierarchyPlacements?.[index];
    const terminal = placement?.nodeId == null ? "root" : String(placement.nodeId);
    const leaf = result.leafLabels[index];
    const clusterIds = terminal === "root" ? ["root"] : ancestorIds(terminal, parents);
    if (leaf >= 0 && !clusterIds.includes(String(leaf))) clusterIds.unshift(...ancestorIds(String(leaf), parents));
    const metadata = note ? metadataFromNote(note) : { tags: [], aliases: [] };
    const clusterTerms = clusterIds.flatMap((id) => {
      const cluster = clusters.find((item) => item.id === id);
      return cluster ? [cluster.title, ...(cluster.keywords || [])] : [];
    });
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
      manuallyAdjusted: Boolean(manualAdjustments[path]),
      mtime: note?.mtime,
    } satisfies SearchDocument;
  });
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
