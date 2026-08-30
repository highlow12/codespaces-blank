import { ClusterResult, NoteRecord } from "./types";

export const KEYWORD_TITLE_ALGORITHM_VERSION = "contrastive-keyphrases-v2";
export const KEYWORD_TITLE_METHOD = "keywords" as const;

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for", "from", "has", "have", "how", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "those", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "about", "after", "before", "can", "do", "does", "done", "not", "than", "then", "also", "just", "very", "note", "notes", "md",
  "그리고", "그러나", "대한", "대해", "때문", "있는", "있다", "있음", "없는", "없다", "하는", "하다", "했다", "한다", "것", "수", "및", "등", "더", "또한", "이것", "저것", "그것", "오늘", "내일", "어제", "정리", "메모", "노트"
]);
const KOREAN_PARTICLES = /(?:으로부터|에서부터|으로서|으로|에서|에게|한테|까지|부터|처럼|보다|마다|만큼|이라도|라도|이며|이고|이랑|랑|과|와|은|는|이|가|을|를|에|의|도|만|로)$/u;

export function cleanText(raw: string, limit = 100_000): string {
  let text = String(raw ?? "").normalize("NFKC");
  text = text.replace(/^\uFEFF?---\s*(?:\r?\n)[\s\S]*?(?:\r?\n)---\s*(?:\r?\n|$)/, " ");
  text = text.replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, " ");
  text = text.replace(/!\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/g, " ");
  text = text.replace(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]/g, (_m, target: string, alias?: string) => alias || target.split(/[\\/]/).pop() || "");
  text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, " ").replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
  text = text.replace(/https?:\/\/\S+|www\.\S+/gi, " ");
  text = text.replace(/(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+/g, " ");
  text = text.replace(/^\s*(?:[-*+•]|\d+[.)])\s*(?:\[[ xX]\]\s*)?/gm, "");
  text = text.replace(/(^|\s)[#>*_~]+(?=\S)/g, "$1").replace(/[\*_`~]+/g, " ");
  return text.replace(/\s+/g, " ").trim().slice(0, limit);
}

export function cleanNoteText(note: NoteRecord, limit = 100_000): string { return cleanText(note.content, limit); }
export function cleanNoteTitle(title: string): string { return cleanText(title.replace(/\.(?:md|markdown)$/i, ""), 500).replace(/[|:：]+/g, " ").trim(); }

export function normalizeKeyword(raw: string): string {
  let token = String(raw ?? "").normalize("NFKC").trim().toLocaleLowerCase();
  token = token.replace(/^['"“”‘’「」『』]+|['"“”‘’「」『』]+$/g, "");
  if (/^[가-힣]+$/u.test(token)) token = token.replace(KOREAN_PARTICLES, "");
  if (/^[a-z][a-z'-]{2,}$/i.test(token)) {
    if (token.endsWith("ies") && token.length > 4) token = `${token.slice(0, -3)}y`;
    else if (/(ches|shes|xes|zes|sses)$/.test(token)) token = token.slice(0, -2);
    else if (token.endsWith("s") && !token.endsWith("ss") && token.length > 3) token = token.slice(0, -1);
  }
  return token;
}

export function tokenizeKeywords(raw: string): string[] {
  const text = cleanText(raw);
  const rawTokens = text.match(/[A-Za-z][A-Za-z'’-]*|[가-힣]{2,}|[一-鿿ぁ-ゟァ-ヿ]{2,}/gu) || [];
  return rawTokens.map(normalizeKeyword).filter((token) => token.length >= 2 && !STOP_WORDS.has(token) && !/^\d+$/.test(token));
}

export interface KeywordScore { keyword: string; score: number; prevalence: number; titleCount: number; documentFrequency: number; }
interface NoteTerms { all: Map<string, number>; title: Map<string, number>; }
interface KeywordIndex { terms: NoteTerms[]; memberIndex: number[]; df: Map<string, number>; totalDocuments: number; }

function noteTerms(note: NoteRecord): NoteTerms {
  const titleTokens = tokenizeKeywords(cleanNoteTitle(note.title)); const bodyTokens = tokenizeKeywords(note.content);
  const all = new Map<string, number>(); const title = new Map<string, number>();
  for (const token of titleTokens) { title.set(token, (title.get(token) || 0) + 1); all.set(token, (all.get(token) || 0) + 3); }
  for (const token of bodyTokens) all.set(token, (all.get(token) || 0) + 1);
  return { all, title };
}
function buildIndex(notes: NoteRecord[], ids: string[]): KeywordIndex {
  const terms = notes.map(noteTerms); const byPath = new Map(notes.map((note, index) => [note.path, index]));
  const memberIndex = ids.map((id) => byPath.get(id) ?? -1); const df = new Map<string, number>();
  terms.forEach(({ all }) => all.forEach((_value, token) => df.set(token, (df.get(token) || 0) + 1)));
  return { terms, memberIndex, df, totalDocuments: notes.length };
}

export function nodeMembers(result: ClusterResult): Map<number, number[]> {
  const groups = new Map<number, number[]>(); result.leafLabels.forEach((label, index) => { if (label >= 0) groups.set(label, [...(groups.get(label) || []), index]); });
  const merges = new Map(result.hierarchy.merges.map((merge) => [merge.id, merge]));
  const visit = (id: number): number[] => { if (groups.has(id)) return groups.get(id)!; const merge = merges.get(id); if (!merge) return []; const members = [...visit(merge.left), ...visit(merge.right)].sort((a, b) => a - b); groups.set(id, members); return members; };
  if (result.hierarchy.root !== null) visit(result.hierarchy.root); return groups;
}
export function stableFingerprint(values: string[]): string { let hash = 2166136261; for (const value of values) for (let i = 0; i < value.length; i++) { hash ^= value.charCodeAt(i); hash = Math.imul(hash, 16777619); } return (hash >>> 0).toString(16).padStart(8, "0"); }

function scoreNode(index: KeywordIndex, members: number[], probabilities: number[]): KeywordScore[] {
  if (!members.length) return []; const counts = new Map<string, number>(); const titleCounts = new Map<string, number>();
  for (const member of members) { const terms = index.terms[index.memberIndex[member]]; if (!terms) continue; const probability = Math.max(0, Math.min(1, Number(probabilities[member]) || 0)); terms.all.forEach((value, token) => counts.set(token, (counts.get(token) || 0) + value * (0.5 + probability * 0.5))); terms.title.forEach((value, token) => titleCounts.set(token, (titleCounts.get(token) || 0) + value)); }
  const size = members.length;
  return [...counts.entries()].map(([keyword, weightedCount]) => { const prevalence = [...members].filter((member) => index.terms[index.memberIndex[member]]?.all.has(keyword)).length / size; const df = index.df.get(keyword) || 0; const idf = Math.log((index.totalDocuments + 1) / (df + 1)) + 1; const tf = weightedCount / (weightedCount + 1.2); return { keyword, score: tf * idf * (0.35 + prevalence * 0.65), prevalence, titleCount: titleCounts.get(keyword) || 0, documentFrequency: df }; }).sort((a, b) => b.score - a.score || b.prevalence - a.prevalence || b.titleCount - a.titleCount || (a.keyword < b.keyword ? -1 : a.keyword > b.keyword ? 1 : 0));
}

function titleForNode(nodeId: number, result: ClusterResult, index: KeywordIndex, membersByNode: Map<number, number[]>): { title: string; scores: KeywordScore[] } {
  const merge = result.hierarchy.merges.find((item) => item.id === nodeId); const scores = scoreNode(index, membersByNode.get(nodeId) || [], result.probabilities);
  if (!merge) return { title: scores.slice(0, 3).map((item) => item.keyword).join(" · "), scores };
  const left = new Set(scoreNode(index, membersByNode.get(merge.left) || [], result.probabilities).map((item) => item.keyword)); const right = new Set(scoreNode(index, membersByNode.get(merge.right) || [], result.probabilities).map((item) => item.keyword));
  const common = scores.filter((item) => left.has(item.keyword) && right.has(item.keyword));
  const leftScores = scoreNode(index, membersByNode.get(merge.left) || [], result.probabilities); const rightScores = scoreNode(index, membersByNode.get(merge.right) || [], result.probabilities);
  const childBest = new Map<string, number>();
  for (const child of [leftScores, rightScores]) { const max = child[0]?.score || 1; for (const item of child) childBest.set(item.keyword, (childBest.get(item.keyword) || 0) + item.score / max); }
  const fallback = scores.filter((item) => !common.some((other) => other.keyword === item.keyword)).sort((a, b) => (childBest.get(b.keyword) || 0) - (childBest.get(a.keyword) || 0) || b.score - a.score || (a.keyword < b.keyword ? -1 : a.keyword > b.keyword ? 1 : 0));
  const selected = [...common, ...fallback].slice(0, 3);
  return { title: selected.map((item) => item.keyword).join(" · "), scores: selected };
}

export interface KeywordTitleOptions { signal?: AbortSignal; onProgress?: (done: number, total: number) => void; onNode?: (nodeId: number, title: string, scores: KeywordScore[]) => void; }
function generateLegacyKeywordTitles(result: ClusterResult, notes: NoteRecord[], options: KeywordTitleOptions = {}): ClusterResult {
  const started = Date.now(); const membersByNode = nodeMembers(result); const index = buildIndex(notes, result.ids); const merges = new Map(result.hierarchy.merges.map((merge) => [merge.id, merge]));
  const depth = (id: number): number => { const merge = merges.get(id); return merge ? Math.max(depth(merge.left), depth(merge.right)) + 1 : 0; }; const nodes = [...membersByNode.keys()].sort((a, b) => depth(a) - depth(b) || a - b);
  const titles: Record<string, string> = {}; const statuses: Record<string, "generated" | "empty"> = {}; const scores: Record<string, Array<{ keyword: string; score: number }>> = {};
  nodes.forEach((nodeId, done) => { if (options.signal?.aborted) throw new Error("Clustering cancelled"); const selected = titleForNode(nodeId, result, index, membersByNode); titles[String(nodeId)] = selected.title; statuses[String(nodeId)] = selected.title ? "generated" : "empty"; scores[String(nodeId)] = selected.scores.slice(0, 3).map((item) => ({ keyword: item.keyword, score: Number(item.score.toFixed(8)) })); options.onNode?.(nodeId, selected.title, selected.scores); options.onProgress?.(done + 1, nodes.length); });
  const inputFingerprint = stableFingerprint(notes.map((note) => `${note.path}:${note.hash}`).concat(result.ids));
  return { ...result, schemaVersion: result.schemaVersion, titles, titleGeneration: { method: KEYWORD_TITLE_METHOD, algorithmVersion: KEYWORD_TITLE_ALGORITHM_VERSION, inputFingerprint, generatedAt: new Date().toISOString(), nodeCount: nodes.length, statuses, scores, durationMs: Date.now() - started } };
}
export class KeywordClusterTitleGenerator { generate(result: ClusterResult, notes: NoteRecord[], options: KeywordTitleOptions = {}): ClusterResult { return generateKeywordTitles(result, notes, options); } }
export const LocalClusterTitleGenerator = KeywordClusterTitleGenerator;
export function validateTitle(raw: string): { valid: boolean; reason?: string } { const title = String(raw || "").trim(); if (!title) return { valid: false, reason: "empty output" }; const parts = title.split(/\s*·\s*/).filter(Boolean); if (parts.length > 3 || new Set(parts).size !== parts.length) return { valid: false, reason: "duplicate or too many keywords" }; if (parts.some((part) => !tokenizeKeywords(part).length)) return { valid: false, reason: "invalid keyword" }; return { valid: true }; }

interface V2PhraseStats { phrase: string; score: number; prevalence: number; coverage: number; documentFrequency: number; }
interface V2Terms { phrases: Map<string, number>; title: string; }

const V2_FIELD_BOOSTS: Record<string, number> = { filename: 4.5, h1: 4, h2: 3.2, alias: 3.5, tag: 3, title: 3, body: 1 };

/** Tokenization for v2 deliberately does not stem or strip Korean particles. */
function tokenizeV2(raw: string): string[] {
  const text = String(raw ?? "").normalize("NFKC");
  return (text.match(/[A-Za-z][A-Za-z'’-]*|[가-힣]+|[一-鿿ぁ-ゟァ-ヿ]+|\d+/gu) || [])
    .map((token) => token.toLocaleLowerCase()).filter((token) => token.length >= 2 && !STOP_WORDS.has(token));
}

function v2Ngrams(tokens: string[]): string[] {
  const result: string[] = [];
  for (let index = 0; index < tokens.length; index++) for (let width = 1; width <= 3 && index + width <= tokens.length; width++) result.push(tokens.slice(index, index + width).join(" "));
  return result;
}

function v2FieldTerms(value: string, field: string): Map<string, number> {
  const output = new Map<string, number>(); const boost = V2_FIELD_BOOSTS[field] || 1;
  for (const phrase of v2Ngrams(tokenizeV2(value))) output.set(phrase, (output.get(phrase) || 0) + boost);
  return output;
}

function extractV2Terms(note: NoteRecord): V2Terms {
  const text = String(note.content || ""); const fields: Array<[string, string]> = [];
  const filename = String(note.path || note.title || "").split(/[\\/]/).pop()?.replace(/\.(?:md|markdown)$/i, "") || "";
  fields.push(["filename", filename], ["title", note.title || filename]);
  for (const match of text.matchAll(/^\s{0,3}(#{1,2})\s+(.+?)\s*#*\s*$/gmu)) fields.push([match[1].length === 1 ? "h1" : "h2", match[2]]);
  for (const match of text.matchAll(/!??\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]/gu)) fields.push(["alias", match[2] || match[1].split(/[\\/]/).pop() || match[1]]);
  const frontmatter = /^\uFEFF?---\s*(?:\r?\n)([\s\S]*?)(?:\r?\n)---\s*(?:\r?\n|$)/u.exec(text)?.[1] || "";
  for (const match of frontmatter.matchAll(/^\s*(?:tags?|keywords?)\s*:\s*(.*?)\s*$/gim)) for (const tag of match[1].replace(/[\[\]{}]/g, "").split(/[\s,]+/).filter(Boolean)) fields.push(["tag", tag.replace(/^#/, "")]);
  const body = text.replace(/^\uFEFF?---\s*(?:\r?\n)[\s\S]*?(?:\r?\n)---\s*(?:\r?\n|$)/u, " ").replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, " ").replace(/!?\[([^\]]+)\]\([^)]*\)/g, "$1");
  fields.push(["body", body]);
  const phrases = new Map<string, number>(); for (const [field, value] of fields) for (const [phrase, weight] of v2FieldTerms(value, field)) phrases.set(phrase, (phrases.get(phrase) || 0) + weight);
  return { phrases, title: cleanNoteTitle(note.title || filename) };
}

function v2NodeMembers(result: ClusterResult): Map<number, number[]> {
  const placements = result.hierarchyPlacements || [];
  const groups = new Map<number, number[]>(); const hasAlignedPlacements = placements.length === result.ids.length;
  result.leafLabels.forEach((label, index) => { if (label >= 0 && (!hasAlignedPlacements || placements[index]?.kind === "leaf")) groups.set(label, [...(groups.get(label) || []), index]); });
  const directResidual = new Map<number, number[]>(); placements.forEach((placement, index) => { if (placement.kind === "residual" && placement.nodeId !== null) directResidual.set(placement.nodeId, [...(directResidual.get(placement.nodeId) || []), index]); });
  const nodes = result.hierarchy.nodes || []; const merges = new Map(result.hierarchy.merges.map((merge) => [merge.id, merge]));
  const children = new Map<number, number[]>(); for (const node of nodes) children.set(node.id, node.children); for (const merge of merges.values()) if (!children.has(merge.id)) children.set(merge.id, [merge.left, merge.right]);
  const visit = (id: number): number[] => { const childIds = children.get(id) || []; if (!childIds.length) return groups.get(id) || []; const members = [...new Set(childIds.flatMap(visit).concat(directResidual.get(id) || []))].sort((a, b) => a - b); if (members.length) groups.set(id, members); return members; };
  const roots = result.hierarchy.root === null ? (result.hierarchy.rootChildren || []) : [result.hierarchy.root]; roots.forEach(visit);
  return groups;
}

function v2NodeChildren(result: ClusterResult): Map<number, number[]> {
  const output = new Map<number, number[]>(); for (const node of result.hierarchy.nodes || []) output.set(node.id, node.children); for (const merge of result.hierarchy.merges) if (!output.has(merge.id)) output.set(merge.id, [merge.left, merge.right]); return output;
}

function v2ScoreNode(index: V2Terms[], members: number[], childMembers: number[][], siblingMembers: number[][], totalDocuments: number): V2PhraseStats[] {
  if (!members.length) return [];
  const counts = new Map<string, number>(); const df = new Map<string, number>();
  for (const member of members) { const terms = index[member]; if (!terms) continue; for (const [phrase, count] of terms.phrases) { counts.set(phrase, (counts.get(phrase) || 0) + count); df.set(phrase, (df.get(phrase) || 0) + 1); } }
  const childSets = childMembers.map((child) => { const set = new Set<string>(); for (const member of child) index[member]?.phrases.forEach((_value, phrase) => set.add(phrase)); return set; });
  const contrastDocuments = [...new Set(siblingMembers.flat())];
  return [...counts].map(([phrase, weighted]) => {
    const documentFrequency = df.get(phrase) || 0; const prevalence = documentFrequency / members.length; const idf = Math.log((totalDocuments + 1) / (documentFrequency + 1)) + 1; const tf = weighted / (weighted + 1.25);
    const childHits = childSets.filter((set) => set.has(phrase)).length; const coverage = childSets.length ? childHits / childSets.length : 0;
    const outsidePresent = contrastDocuments.filter((member) => index[member]?.phrases.has(phrase)).length;
    const logOdds = contrastDocuments.length ? Math.log(((documentFrequency + 0.5) / (members.length - documentFrequency + 0.5)) / ((outsidePresent + 0.5) / (contrastDocuments.length - outsidePresent + 0.5))) : 0;
    return { phrase, score: tf * idf * (0.35 + 0.65 * prevalence) + 0.3 * logOdds + 0.45 * coverage, prevalence, coverage, documentFrequency };
  }).sort((a, b) => b.score - a.score || b.prevalence - a.prevalence || b.coverage - a.coverage || a.phrase.localeCompare(b.phrase));
}

function phraseSimilarity(left: string, right: string): number {
  const a = new Set(left.split(" ")); const b = new Set(right.split(" ")); const intersection = [...a].filter((token) => b.has(token)).length; return intersection / Math.max(1, new Set([...a, ...b]).size);
}

function v2Select(scores: V2PhraseStats[], limit = 2): V2PhraseStats[] {
  const chosen: V2PhraseStats[] = []; const remaining = scores.slice();
  while (chosen.length < limit && remaining.length) { let best = remaining[0]; let bestValue = -Infinity; for (const candidate of remaining) { const redundancy = chosen.length ? Math.max(...chosen.map((item) => phraseSimilarity(item.phrase, candidate.phrase))) : 0; const value = candidate.score - 0.35 * redundancy; if (value > bestValue + 1e-12 || (Math.abs(value - bestValue) <= 1e-12 && candidate.phrase < best.phrase)) { best = candidate; bestValue = value; } } chosen.push(best); remaining.splice(remaining.indexOf(best), 1); }
  return chosen;
}

function generateV2Titles(result: ClusterResult, notes: NoteRecord[], options: KeywordTitleOptions = {}): ClusterResult {
  const started = Date.now(); const byPath = new Map(notes.map((note) => [note.path, note])); const alignedNotes = result.ids.map((id) => byPath.get(id) || ({ path: id, title: id, content: "", hash: "", mtime: 0 } as NoteRecord)); const terms = alignedNotes.map(extractV2Terms); const members = v2NodeMembers(result); const children = v2NodeChildren(result); const parents = new Map<number, number>(); for (const [parent, childIds] of children) for (const child of childIds) parents.set(child, parent); const nodes = [...members.keys()].sort((a, b) => a - b); const titles: Record<string, string> = {}; const statuses: Record<string, "generated" | "empty"> = {}; const scores: Record<string, Array<{ keyword: string; score: number }>> = {};
  for (const [done, nodeId] of nodes.entries()) {
    if (options.signal?.aborted) throw new Error("Clustering cancelled");
    const nodeMembers = members.get(nodeId) || []; const childMembers = (children.get(nodeId) || []).map((child) => (members.get(child) || []).filter((member) => !result.hierarchyPlacements?.[member] || result.hierarchyPlacements[member].kind === "leaf")); const parent = parents.get(nodeId); const siblingMembers = parent === undefined ? [] : (children.get(parent) || []).filter((sibling) => sibling !== nodeId).map((sibling) => members.get(sibling) || []); const ranked = v2ScoreNode(terms, nodeMembers, childMembers, siblingMembers, alignedNotes.length); const selected = v2Select(ranked); let title = selected.map((item) => item.phrase).join(" · ");
    if (!title) {
      const coordinates = result.visualization?.coordinates;
      const validMembers = nodeMembers.filter((member) => alignedNotes[member]);
      let representative: number | undefined;
      if (coordinates && validMembers.length) {
        const center = validMembers.reduce<[number, number]>(([x, y], member) => [x + coordinates[member][0] / validMembers.length, y + coordinates[member][1] / validMembers.length], [0, 0]);
        representative = validMembers.slice().sort((a, b) => ((coordinates[a][0] - center[0]) ** 2 + (coordinates[a][1] - center[1]) ** 2) - ((coordinates[b][0] - center[0]) ** 2 + (coordinates[b][1] - center[1]) ** 2) || (result.probabilities[b] || 0) - (result.probabilities[a] || 0) || alignedNotes[a].path.localeCompare(alignedNotes[b].path))[0];
      } else representative = validMembers.slice().sort((a, b) => (result.probabilities[b] || 0) - (result.probabilities[a] || 0) || alignedNotes[a].path.localeCompare(alignedNotes[b].path))[0];
      title = representative === undefined ? "" : cleanNoteTitle(alignedNotes[representative]?.title || "");
    }
    titles[String(nodeId)] = title; statuses[String(nodeId)] = title ? "generated" : "empty"; scores[String(nodeId)] = selected.map((item) => ({ keyword: item.phrase, score: Number(item.score.toFixed(8)) })); options.onNode?.(nodeId, title, ranked.map((item) => ({ keyword: item.phrase, score: item.score, prevalence: item.prevalence, titleCount: 0, documentFrequency: item.documentFrequency }))); options.onProgress?.(done + 1, nodes.length);
  }
  // Keep this result pure: callers can atomically publish it only after every
  // node succeeds, so cancellation/error leaves the previously saved titles.
  const inputFingerprint = stableFingerprint(notes.map((note) => `${note.path}:${note.hash}`).concat(result.ids));
  return { ...result, titles, titleGeneration: { method: KEYWORD_TITLE_METHOD, algorithmVersion: KEYWORD_TITLE_ALGORITHM_VERSION, inputFingerprint, generatedAt: new Date().toISOString(), nodeCount: nodes.length, statuses, scores, durationMs: Date.now() - started } };
}

/** Generate v2 titles for v6/n-ary results; retain the legacy contract for old saved results. */
export function generateKeywordTitles(result: ClusterResult, notes: NoteRecord[], options: KeywordTitleOptions = {}): ClusterResult {
  return result.schemaVersion >= 6 || !!result.hierarchy.nodes ? generateV2Titles(result, notes, options) : generateLegacyKeywordTitles(result, notes, options);
}
