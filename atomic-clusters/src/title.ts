import { requestUrl } from "obsidian";
import { ClusterResult, ClusterTitleCacheEntry, ClusterTitleStatus, HierarchyMerge, NoteRecord } from "./types";

/** The title model is deliberately separate from embedding models and is never fetched implicitly. */
export const TITLE_MODEL_ID = "qwen2.5-0.5b-instruct";
export const TITLE_MODEL_REVISION = "516c8d04add8a80c5228f32102b57953b8d421a9";
export const TITLE_MODEL_MODEL_SHA256 = "b11c1dd99efd57e6c6e5bc4443a019931a5fbd5dd500d48644d8225f5ce0b2cb";
export const TITLE_MODEL_PROMPT_VERSION = "cluster-title-v1";
export const TITLE_MODEL_DESCRIPTOR = {
  id: TITLE_MODEL_ID,
  revision: TITLE_MODEL_REVISION,
  modelUrl: `https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/${TITLE_MODEL_REVISION}/onnx/model_q4f16.onnx`,
  tokenizerUrl: `https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/${TITLE_MODEL_REVISION}/tokenizer.json`,
  configUrl: `https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/${TITLE_MODEL_REVISION}/config.json`,
  generationConfigUrl: `https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/${TITLE_MODEL_REVISION}/generation_config.json`,
  tokenizerConfigUrl: `https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/${TITLE_MODEL_REVISION}/tokenizer_config.json`,
  /** The manifest records the downloaded digest; this field is a fixed descriptor identity. */
  quantization: "q4f16",
  device: "webgpu",
  modelSha256: TITLE_MODEL_MODEL_SHA256
} as const;

export interface TitleModelProgress { phase: "consent" | "model" | "tokenizer" | "config" | "verify" | "install" | "complete"; progress: number; loadedBytes?: number; totalBytes?: number; detail?: string; }
export interface TitleModelArtifact {
  model: ArrayBuffer;
  tokenizer: ArrayBuffer;
  config: ArrayBuffer;
  generationConfig: ArrayBuffer;
  tokenizerConfig: ArrayBuffer;
  modelSha256: string;
  tokenizerSha256: string;
  configSha256: string;
  generationConfigSha256: string;
  tokenizerConfigSha256: string;
  revision: string;
}
export interface TitleModelStorage { exists(path: string): Promise<boolean>; read(path: string): Promise<ArrayBuffer>; write(path: string, data: ArrayBuffer): Promise<void>; remove(path: string): Promise<void>; }
interface TitleManifest { id: string; revision: string; quantization: string; modelSha256: string; tokenizerSha256: string; configSha256: string; generationConfigSha256: string; tokenizerConfigSha256: string; }

export class VaultTitleModelStorage implements TitleModelStorage {
  constructor(private readonly adapter: { exists(path: string): Promise<boolean>; readBinary(path: string): Promise<ArrayBuffer>; writeBinary(path: string, data: ArrayBuffer): Promise<void>; remove(path: string): Promise<void>; mkdir(path: string): Promise<void> }, private readonly prefix = ".obsidian/plugins/atomic-clusters/title-models") {}
  exists(path: string): Promise<boolean> { return this.adapter.exists(`${this.prefix}/${path}`); }
  read(path: string): Promise<ArrayBuffer> { return this.adapter.readBinary(`${this.prefix}/${path}`); }
  async write(path: string, data: ArrayBuffer): Promise<void> { const slash = path.lastIndexOf("/"); if (slash > 0) { const directory = `${this.prefix}/${path.slice(0, slash)}`; if (!(await this.adapter.exists(directory))) await this.adapter.mkdir(directory); } await this.adapter.writeBinary(`${this.prefix}/${path}`, data); }
  remove(path: string): Promise<void> { return this.adapter.remove(`${this.prefix}/${path}`); }
}

const MODEL_FILE = "model_q4f16.onnx";
const TOKENIZER_FILE = "tokenizer.json";
const CONFIG_FILE = "config.json";
const GENERATION_CONFIG_FILE = "generation_config.json";
const TOKENIZER_CONFIG_FILE = "tokenizer_config.json";
const MANIFEST_FILE = "manifest.json";

async function digest(data: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error("Title model integrity checks require Web Crypto SHA-256 support.");
  const bytes = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", data));
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}
async function fetchBinary(url: string, phase: TitleModelProgress["phase"], onProgress?: (update: TitleModelProgress) => void): Promise<ArrayBuffer> {
  onProgress?.({ phase, progress: 0, detail: "Downloading…" });
  const response = await requestUrl({ url, method: "GET" });
  const bytes = (response as unknown as { arrayBuffer?: ArrayBuffer }).arrayBuffer;
  if (!(bytes instanceof ArrayBuffer) || bytes.byteLength === 0) throw new Error(`Title model download returned no bytes: ${url}`);
  onProgress?.({ phase, progress: 1, loadedBytes: bytes.byteLength, detail: "Downloaded" });
  return bytes;
}

export type TitleModelStatus = "missing" | "incomplete" | "installed" | "corrupt";

export class TitleModelManager {
  constructor(private readonly storage: TitleModelStorage, private readonly descriptor = TITLE_MODEL_DESCRIPTOR) {}
  private path(file: string): string { return `${this.descriptor.id}/${this.descriptor.revision}/${file}`; }
  private async hasAssets(): Promise<boolean> { return (await Promise.all([MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE, GENERATION_CONFIG_FILE, TOKENIZER_CONFIG_FILE].map((file) => this.storage.exists(this.path(file))))).every(Boolean); }
  async status(): Promise<TitleModelStatus> {
    const hasManifest = await this.storage.exists(this.path(MANIFEST_FILE));
    const hasAssets = await this.hasAssets();
    if (!hasManifest) return hasAssets ? "incomplete" : "missing";
    if (!hasAssets) return "corrupt";
    try { this.validateManifest(JSON.parse(new TextDecoder().decode(await this.storage.read(this.path(MANIFEST_FILE)))) as TitleManifest); return "installed"; } catch { return "corrupt"; }
  }
  async verifyStatus(): Promise<TitleModelStatus> { const state = await this.status(); if (state === "missing" || state === "incomplete") return state; try { await this.load(); return "installed"; } catch { return "corrupt"; } }
  async downloadModel(confirm: () => Promise<boolean>, onProgress?: (update: TitleModelProgress) => void): Promise<void> {
    onProgress?.({ phase: "consent", progress: 0, detail: "Waiting for confirmation" });
    if (!(await confirm())) throw new Error("Title model download cancelled");
    onProgress?.({ phase: "consent", progress: 0.05, detail: "Download approved" });
    const model = await fetchBinary(this.descriptor.modelUrl, "model", (u) => onProgress?.({ ...u, progress: 0.05 + u.progress * 0.65 }));
    const tokenizer = await fetchBinary(this.descriptor.tokenizerUrl, "tokenizer", (u) => onProgress?.({ ...u, progress: 0.70 + u.progress * 0.10 }));
    const config = await fetchBinary(this.descriptor.configUrl, "config", (u) => onProgress?.({ ...u, progress: 0.80 + u.progress * 0.05 }));
    const generationConfig = await fetchBinary(this.descriptor.generationConfigUrl, "config", (u) => onProgress?.({ ...u, progress: 0.85 + u.progress * 0.025 }));
    const tokenizerConfig = await fetchBinary(this.descriptor.tokenizerConfigUrl, "config", (u) => onProgress?.({ ...u, progress: 0.875 + u.progress * 0.025 }));
    onProgress?.({ phase: "verify", progress: 0.89, detail: "Calculating SHA-256 integrity hashes" });
    const modelSha256 = await digest(model);
    if (modelSha256 !== this.descriptor.modelSha256) throw new Error(`Title model SHA-256 mismatch: expected ${this.descriptor.modelSha256}, received ${modelSha256}.`);
    const manifest: TitleManifest = { id: this.descriptor.id, revision: this.descriptor.revision, quantization: this.descriptor.quantization, modelSha256, tokenizerSha256: await digest(tokenizer), configSha256: await digest(config), generationConfigSha256: await digest(generationConfig), tokenizerConfigSha256: await digest(tokenizerConfig) };
    // Remove the manifest before replacing assets. It is written last, so a
    // killed download is reported as incomplete instead of looking installed.
    if (await this.storage.exists(this.path(MANIFEST_FILE))) await this.storage.remove(this.path(MANIFEST_FILE));
    onProgress?.({ phase: "install", progress: 0.92, detail: "Installing model assets" });
    await this.storage.write(this.path(MODEL_FILE), model);
    await this.storage.write(this.path(TOKENIZER_FILE), tokenizer);
    await this.storage.write(this.path(CONFIG_FILE), config);
    await this.storage.write(this.path(GENERATION_CONFIG_FILE), generationConfig);
    await this.storage.write(this.path(TOKENIZER_CONFIG_FILE), tokenizerConfig);
    await this.storage.write(this.path(MANIFEST_FILE), new TextEncoder().encode(JSON.stringify(manifest)).buffer);
    onProgress?.({ phase: "complete", progress: 1, detail: "Title model installed" });
  }
  async deleteModel(): Promise<void> { for (const file of [MANIFEST_FILE, MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE, GENERATION_CONFIG_FILE, TOKENIZER_CONFIG_FILE]) if (await this.storage.exists(this.path(file))) await this.storage.remove(this.path(file)); }
  async load(): Promise<TitleModelArtifact> {
    const manifest = JSON.parse(new TextDecoder().decode(await this.storage.read(this.path(MANIFEST_FILE)))) as TitleManifest;
    this.validateManifest(manifest);
    const [model, tokenizer, config, generationConfig, tokenizerConfig] = await Promise.all([MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE, GENERATION_CONFIG_FILE, TOKENIZER_CONFIG_FILE].map((file) => this.storage.read(this.path(file))));
    const [modelSha256, tokenizerSha256, configSha256, generationConfigSha256, tokenizerConfigSha256] = await Promise.all([model, tokenizer, config, generationConfig, tokenizerConfig].map(digest));
    if (modelSha256 !== this.descriptor.modelSha256 || modelSha256 !== manifest.modelSha256 || tokenizerSha256 !== manifest.tokenizerSha256 || configSha256 !== manifest.configSha256 || generationConfigSha256 !== manifest.generationConfigSha256 || tokenizerConfigSha256 !== manifest.tokenizerConfigSha256) throw new Error("Installed title model failed its pinned SHA-256 integrity check; delete and download it again.");
    return { model, tokenizer, config, generationConfig, tokenizerConfig, modelSha256, tokenizerSha256, configSha256, generationConfigSha256, tokenizerConfigSha256, revision: manifest.revision };
  }
  private validateManifest(manifest: TitleManifest): void { if (manifest.id !== this.descriptor.id || manifest.revision !== this.descriptor.revision || manifest.quantization !== this.descriptor.quantization || manifest.modelSha256 !== this.descriptor.modelSha256 || !/^[a-f0-9]{64}$/.test(manifest.modelSha256) || !/^[a-f0-9]{64}$/.test(manifest.tokenizerSha256) || !/^[a-f0-9]{64}$/.test(manifest.configSha256) || !/^[a-f0-9]{64}$/.test(manifest.generationConfigSha256) || !/^[a-f0-9]{64}$/.test(manifest.tokenizerConfigSha256)) throw new Error("Installed title model manifest is stale or invalid; download the current model again."); }
}

export interface TitlePrompt { nodeId: number; members: number[]; memberFingerprint: string; text: string; }
export interface TitleGenerationRuntime { generate(prompts: string[], options: { maxNewTokens: number; doSample: boolean; temperature: number; signal?: AbortSignal }): Promise<string[]>; diagnostics?: { backend: "webgpu" | "unavailable" }; }

/** A runtime seam keeps model orchestration testable and makes GPU failure non-fatal. */
export type TitleRuntimeFactory = (artifact: TitleModelArtifact) => Promise<TitleGenerationRuntime>;
export const unavailableTitleRuntime: TitleRuntimeFactory = async () => { throw new Error("Transformers.js WebGPU title runtime is unavailable"); };

export function sanitizeTitle(raw: string): string {
  return raw.replace(/```[\s\S]*?```/g, "").replace(/[\r\n]+/g, " ").replace(/^["'“”‘’]+|["'“”‘’]+$/g, "").replace(/^\s*(?:title|제목)\s*[:：-]\s*/i, "").replace(/\s+/g, " ").trim().slice(0, 48).trim();
}

export function cleanNoteText(note: NoteRecord): string {
  return note.content.replace(/^---[\s\S]*?---\s*/m, "").replace(/!\[[^\]]*\]\([^)]*\)/g, "").replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/[`*_>#~-]/g, " ").replace(/\s+/g, " ").trim().slice(0, 300);
}
export function selectRepresentativeNotes(notes: NoteRecord[], members: number[], probabilities: number[], limit = 6): NoteRecord[] {
  return members.map((index) => ({ index, note: notes[index] })).filter((entry) => entry.note).sort((a, b) => (probabilities[b.index] ?? 0) - (probabilities[a.index] ?? 0) || a.note.path.localeCompare(b.note.path)).slice(0, limit).map((entry) => entry.note);
}

export function stableFingerprint(values: string[]): string {
  let hash = 2166136261;
  for (const value of values) for (let i = 0; i < value.length; i++) { hash ^= value.charCodeAt(i); hash = Math.imul(hash, 16777619); }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
export function nodeMembers(result: ClusterResult): Map<number, number[]> {
  const groups = new Map<number, number[]>(); result.leafLabels.forEach((label, index) => { if (label >= 0) groups.set(label, [...(groups.get(label) || []), index]); });
  const merges = new Map(result.hierarchy.merges.map((merge) => [merge.id, merge]));
  const visit = (id: number): number[] => { if (groups.has(id)) return groups.get(id)!; const merge = merges.get(id); if (!merge) return []; const members = [...visit(merge.left), ...visit(merge.right)].sort((a, b) => a - b); groups.set(id, members); return members; };
  if (result.hierarchy.root !== null) visit(result.hierarchy.root);
  return groups;
}

export function buildTitlePrompts(result: ClusterResult, notes: NoteRecord[], language = "auto"): TitlePrompt[] {
  const membersByNode = nodeMembers(result); const merges = new Map(result.hierarchy.merges.map((merge) => [merge.id, merge])); const prompts: TitlePrompt[] = [];
  const ordered = [...membersByNode.keys()].sort((a, b) => (merges.has(a) ? 1 : 0) - (merges.has(b) ? 1 : 0) || a - b);
  for (const nodeId of ordered) {
    const members = membersByNode.get(nodeId)!; const representative = selectRepresentativeNotes(notes, members, result.probabilities);
    const children = merges.get(nodeId) ? [merges.get(nodeId)!.left, merges.get(nodeId)!.right].map((id) => result.titles?.[String(id)]).filter(Boolean) : [];
    const snippets = representative.map((note) => `- ${note.title}: ${cleanNoteText(note)}`).join("\n");
    prompts.push({ nodeId, members, memberFingerprint: stableFingerprint(members.map((member) => `${result.ids[member]}:${notes[member]?.hash || ""}`)), text: `Generate exactly one concise title of 2-6 words. Follow the input language (${language === "auto" ? "detect automatically" : language}). Output title only, no quotes, punctuation, prefix, or explanation.\n${children.length ? `Child titles: ${children.join(" | ")}\n` : ""}Representative notes:\n${snippets}`.slice(0, 5000) });
  }
  return prompts;
}

export interface TitleCacheLike { get(key: string): ClusterTitleCacheEntry | undefined; set(entry: ClusterTitleCacheEntry): void; }
export interface GenerateTitlesOptions { language?: string; signal?: AbortSignal; onProgress?: (done: number, total: number) => void; onBatch?: (result: ClusterResult) => Promise<void> | void; cache?: TitleCacheLike; }

export class LocalClusterTitleGenerator {
  constructor(private readonly manager: TitleModelManager, private readonly runtimeFactory: TitleRuntimeFactory = unavailableTitleRuntime, private readonly modelStatus?: () => Promise<TitleModelStatus>) {}
  async generate(result: ClusterResult, notes: NoteRecord[], options: GenerateTitlesOptions = {}): Promise<ClusterResult> {
    const status = await (this.modelStatus ? this.modelStatus() : this.manager.status());
    const initialPrompts = buildTitlePrompts(result, notes, options.language || "auto");
    const statuses: Record<string, ClusterTitleStatus> = {}; const errors: Record<string, string> = {}; const durationsMs: Record<string, number> = {};
    if (status !== "installed") { initialPrompts.forEach((prompt) => { statuses[String(prompt.nodeId)] = "skipped"; durationsMs[String(prompt.nodeId)] = 0; }); return withTitleMetadata(result, statuses, options.language || "auto", "unavailable", {}, durationsMs); }
    let runtime: TitleGenerationRuntime;
    try { runtime = await this.runtimeFactory(await this.manager.load()); }
    catch (error) { const message = safeTitleError(error); initialPrompts.forEach((prompt) => { statuses[String(prompt.nodeId)] = "failed"; errors[String(prompt.nodeId)] = message; durationsMs[String(prompt.nodeId)] = 0; }); return withTitleMetadata(result, statuses, options.language || "auto", "unavailable", errors, durationsMs); }
    const output = { ...result, schemaVersion: 2 as const, titles: { ...(result.titles || {}) } }; const batchSize = 4;
    const mergeMap = new Map(result.hierarchy.merges.map((merge) => [merge.id, merge]));
    const depths = new Map<number, number>(); const depthOf = (id: number): number => { if (depths.has(id)) return depths.get(id)!; const merge = mergeMap.get(id); const depth = merge ? Math.max(depthOf(merge.left), depthOf(merge.right)) + 1 : 0; depths.set(id, depth); return depth; };
    const levelGroups = new Map<number, number[]>(); initialPrompts.forEach((prompt) => { const depth = depthOf(prompt.nodeId); levelGroups.set(depth, [...(levelGroups.get(depth) || []), prompt.nodeId]); });
    const orderedLevels = [...levelGroups.keys()].sort((a, b) => a - b);
    const totalNodes = initialPrompts.length;
    let completed = 0;
    for (const level of orderedLevels) for (let start = 0; start < levelGroups.get(level)!.length; start += batchSize) {
      if (options.signal?.aborted) throw new Error("Clustering cancelled");
      // Rebuild prompts for every batch so merge nodes see titles generated for
      // their children in earlier (bottom-up) batches.
      const currentPrompts = buildTitlePrompts(output, notes, options.language || "auto");
      const batch = levelGroups.get(level)!.slice(start, start + batchSize).map((id) => currentPrompts.find((prompt) => prompt.nodeId === id)!).filter(Boolean); const uncached: TitlePrompt[] = [];
      for (const prompt of batch) { const key = titleCacheKey(prompt, options.language || "auto"); const cached = options.cache?.get(key); if (cached) { output.titles![String(prompt.nodeId)] = cached.title; statuses[String(prompt.nodeId)] = "cached"; durationsMs[String(prompt.nodeId)] = 0; } else uncached.push(prompt); }
      if (uncached.length) try {
        const started = Date.now();
        const values = await runtime.generate(uncached.map((prompt) => prompt.text), { maxNewTokens: 12, doSample: false, temperature: 0, signal: options.signal });
        uncached.forEach((prompt, index) => { const title = sanitizeTitle(values[index] || ""); durationsMs[String(prompt.nodeId)] = Math.max(0, Math.round((Date.now() - started) / Math.max(1, uncached.length))); if (!title) { statuses[String(prompt.nodeId)] = "failed"; errors[String(prompt.nodeId)] = "Empty model output"; return; } output.titles![String(prompt.nodeId)] = title; statuses[String(prompt.nodeId)] = "generated"; options.cache?.set({ key: titleCacheKey(prompt, options.language || "auto"), title, nodeMembersFingerprint: prompt.memberFingerprint, savedAt: new Date().toISOString() }); });
      } catch (error) {
        const cancelled = options.signal?.aborted || (error instanceof Error && error.message.toLowerCase().includes("cancel"));
        if (cancelled) throw error;
        const message = safeTitleError(error); uncached.forEach((prompt) => { statuses[String(prompt.nodeId)] = "failed"; errors[String(prompt.nodeId)] = message; durationsMs[String(prompt.nodeId)] = 0; });
      }
      completed += batch.length; options.onProgress?.(completed, totalNodes);
      if (options.onBatch) await options.onBatch(withTitleMetadata(output, statuses, options.language || "auto", runtime.diagnostics?.backend || "webgpu", errors, durationsMs));
    }
    return withTitleMetadata(output, statuses, options.language || "auto", runtime.diagnostics?.backend || "webgpu", errors, durationsMs);
  }
}

function titleCacheKey(prompt: TitlePrompt, language: string): string { return `${TITLE_MODEL_REVISION}:${TITLE_MODEL_PROMPT_VERSION}:${language}:${prompt.memberFingerprint}`; }
function withTitleMetadata(result: ClusterResult, statuses: Record<string, ClusterTitleStatus>, language: string, backend: "webgpu" | "unavailable", errors: Record<string, string> = {}, durationsMs: Record<string, number> = {}): ClusterResult { return { ...result, schemaVersion: 2, titleGeneration: { modelRevision: TITLE_MODEL_REVISION, promptVersion: TITLE_MODEL_PROMPT_VERSION, language, inputFingerprint: stableFingerprint(Object.keys(statuses).sort()), backend, generatedAt: new Date().toISOString(), statuses, durationsMs, ...(Object.keys(errors).length ? { errors } : {}) } }; }
function safeTitleError(error: unknown): string { return (error instanceof Error ? error.message : String(error)).replace(/((?:^|[?&\s])(?:key|token|secret|authorization)=)[^&\s]+/gi, "$1[redacted]").slice(0, 240); }
