import { requestUrl } from "obsidian";
import { ClusterResult, ClusterTitleCacheEntry, ClusterTitleStatus, HierarchyMerge, NoteRecord } from "./types";

/** The title model is deliberately separate from embedding models and is never fetched implicitly. */
export const TITLE_MODEL_ID = "qwen3-0.6b-q4f16";
export const TITLE_MODEL_REVISION = "558750086ed49d78cb701ed6fa85af33fd16453f";
export const TITLE_MODEL_MODEL_SHA256 = "9e33a5911974174761d0dfdcc0bec975d9c45af0eae5e9eb647b8ba9442a8f91";
export const TITLE_MODEL_SIZE_BYTES = 569_789_750;
export const TITLE_MODEL_PROMPT_VERSION = "cluster-title-v5-qwen3-assistant-prefix-validated-retry";
export const TITLE_MODEL_DESCRIPTOR = {
  id: TITLE_MODEL_ID,
  revision: TITLE_MODEL_REVISION,
  modelUrl: `https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/${TITLE_MODEL_REVISION}/onnx/model_q4f16.onnx`,
  tokenizerUrl: `https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/${TITLE_MODEL_REVISION}/tokenizer.json`,
  configUrl: `https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/${TITLE_MODEL_REVISION}/config.json`,
  generationConfigUrl: `https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/${TITLE_MODEL_REVISION}/generation_config.json`,
  tokenizerConfigUrl: `https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/${TITLE_MODEL_REVISION}/tokenizer_config.json`,
  /** The manifest records the downloaded digest; this field is a fixed descriptor identity. */
  quantization: "q4f16",
  device: "webgpu",
  modelSizeBytes: TITLE_MODEL_SIZE_BYTES,
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
export type TitleGenerationMode = "title" | "diagnostic";
export interface TitleGenerationRuntime { generate(prompts: string[], options: { maxNewTokens: number; doSample: boolean; temperature: number; repetitionPenalty?: number; noRepeatNgramSize?: number; mode?: TitleGenerationMode; signal?: AbortSignal }): Promise<string[]>; diagnostics?: { backend: "webgpu" | "unavailable" }; }

/** A runtime seam keeps model orchestration testable and makes GPU failure non-fatal. */
export type TitleRuntimeFactory = (artifact: TitleModelArtifact) => Promise<TitleGenerationRuntime>;
export const unavailableTitleRuntime: TitleRuntimeFactory = async () => { throw new Error("Transformers.js WebGPU title runtime is unavailable"); };

/** Remove presentation syntax before it can become a title signal. */
export function cleanText(raw: string, limit = 300): string {
  let text = raw.replace(/^\uFEFF?---\s*(?:\r?\n)[\s\S]*?(?:\r?\n)---\s*(?:\r?\n|$)/, " ");
  text = text.replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, " ").replace(/`[^`\n]+`/g, " ");
  text = text.replace(/!\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]/g, " ");
  text = text.replace(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]/g, (_match, target: string, alias?: string) => alias || target.split("/").pop() || "");
  text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, " ").replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
  text = text.replace(/https?:\/\/\S+|www\.\S+/gi, " ");
  text = text.replace(/^\s*(?:[-*+•]|\d+[.)])\s*(?:\[[ xX]\]\s*)?/gm, "").replace(/^\s*\[[ xX]\]\s*/gm, "");
  text = text.replace(/(^|\s)[#>*_~]+(?=\S)/g, "$1").replace(/[\*_`~]+/g, " ");
  // Paths and repeated punctuation/tokens are common prompt-injection noise.
  text = text.replace(/(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+/g, " ").replace(/([!?.,;:])\1+/g, "$1");
  for (let i = 0; i < 3; i++) text = text.replace(/\b(\S+)(?:\s+\1){1,}\b/gi, "$1");
  return text.replace(/\s+/g, " ").trim().slice(0, limit).trim();
}

export function cleanNoteText(note: NoteRecord, limit = 260): string { return cleanText(note.content, limit); }
export function cleanNoteTitle(title: string): string { return cleanText(title.replace(/\.(?:md|markdown)$/i, ""), 100).replace(/[|:：]+/g, " ").replace(/\s+/g, " ").trim(); }

/** Normalize a model response without allowing markdown or a prompt label to leak into the UI. */
export function sanitizeTitle(raw: string): string {
  let title = String(raw || "").replace(/```[\s\S]*?```/g, " ").replace(/[\r\n]+/g, " ");
  // Qwen3 may still return a reasoning block when the soft /no_think switch
  // is ignored or a generation is truncated. It is never part of a title.
  title = title.replace(/<think>[\s\S]*?<\/think>/gi, " ").replace(/<think>[\s\S]*$/gi, " ");
  title = title.replace(/^["'“”‘’「」『』]+|["'“”‘’「」『』]+$/g, "").trim();
  title = title.replace(/^\s*(?:assistant|answer|title|제목)\s*[:：-]\s*/i, "");
  title = title.replace(/^\s*(?:[-*+•]|\d+[.)])\s*(?:\[[ xX]\]\s*)?/, "").replace(/^\s*\[[ xX]\]\s*/, "");
  title = title.replace(/!\[\[[^\]]+\]\]/g, " ").replace(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]/g, (_match, target: string, alias?: string) => alias || target.split("/").pop() || "");
  title = title.replace(/https?:\/\/\S+|www\.\S+/gi, "").replace(/[\*_`~#]+/g, " ");
  title = title.replace(/^["'“”‘’「」『』]+|["'“”‘’「」『』]+$/g, "");
  title = title.replace(/([!?.,;:])\1+/g, "$1").replace(/\s+/g, " ").trim();
  return title.slice(0, 48).trim();
}

export interface TitleValidation { valid: boolean; reason?: string; }
export function validateTitle(raw: string, prompt = ""): TitleValidation {
  const rawText = String(raw || "").trim();
  // Never sanitize control syntax away before validation. In particular,
  // Qwen3 can emit reasoning, ChatML, or HTML-like tags; accepting the text
  // after those tags would cache a polluted generation as a valid title.
  // The small Qwen3 model has also emitted slash-prefixed control words
  // (`/thought /Thought /Schooling ...`, `/thinking ...`). Match only the
  // known control-word family, so ordinary titles such as "Thought Process"
  // remain valid.
  if (/<\|[^>\r\n]{1,80}\|>|<\/?[^>\r\n]{1,80}>|(?:^|\s)\/(?:no[_-]?think|think(?:ing|er|s)?|thought(?:s|ful)?|thin(?:k|king)?|assistant|user|system)(?=$|[\s/:])/i.test(rawText)) return { valid: false, reason: "control, HTML, or ChatML tag residue" };
  const title = sanitizeTitle(rawText);
  if (!title) return { valid: false, reason: "empty output" };
  if (/\[[ xX]\]|```|\[\[[^\]]+\]\]|https?:\/\/|www\./i.test(title)) return { valid: false, reason: "markdown, checkbox, or URL residue" };
  if (/^(?:[-*+•]|\d+[.)])\s/.test(title) || /^(?:\d+\s*)+$/.test(title) || (!/[A-Za-z\u00c0-\uFFFF]/u.test(title) && /^[\d\W_]+$/.test(title))) return { valid: false, reason: "list or numeric garbage" };
  if (/\b(\S+)(?:\s+\1){1,}\b/i.test(title) || /(.)\1{4,}/u.test(title)) return { valid: false, reason: "repeated phrase" };
  const words = title.split(/\s+/).filter(Boolean);
  const cjk = /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/u.test(title);
  if ((!cjk && (words.length < 2 || words.length > 6)) || (cjk && (words.length > 6 || title.length < 2))) return { valid: false, reason: "title must contain 2-6 words" };
  const context = prompt.split(/Representative context[^\n]*:\s*/i)[1] || "";
  if (title.length >= 24 && context.includes(title)) return { valid: false, reason: "copied representative text" };
  return { valid: true };
}
export function selectRepresentativeNotes(notes: NoteRecord[], members: number[], probabilities: number[], limit = 6): NoteRecord[] {
  return members.map((index) => ({ index, note: notes[index] })).filter((entry) => entry.note).sort((a, b) => (probabilities[b.index] ?? 0) - (probabilities[a.index] ?? 0) || a.note.path.localeCompare(b.note.path)).slice(0, limit).map((entry) => entry.note);
}

export function stableFingerprint(values: string[]): string {
  let hash = 2166136261;
  for (const value of values) for (let i = 0; i < value.length; i++) { hash ^= value.charCodeAt(i); hash = Math.imul(hash, 16777619); }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function dominantLanguage(text: string): string {
  const korean = (text.match(/[가-힣]/g) || []).length;
  const japanese = (text.match(/[ぁ-ゟ゠-ヿ]/g) || []).length;
  const chinese = (text.match(/[一-鿿]/g) || []).length;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  if (korean > Math.max(japanese, chinese, latin)) return "Korean";
  if (japanese > Math.max(korean, chinese, latin)) return "Japanese";
  if (chinese > Math.max(korean, japanese, latin)) return "Chinese";
  if (latin) return "English";
  return "the dominant language of the input";
}

export const TITLE_SYSTEM_PROMPT = "You name knowledge clusters. Return only one useful, specific title. Never return an explanation, list, checkbox, markdown, URL, path, quotation, or label. Use 2-6 words and the requested input language.";

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
    const members = membersByNode.get(nodeId)!; const isMerge = merges.has(nodeId); const representative = selectRepresentativeNotes(notes, members, result.probabilities, isMerge ? 2 : 6);
    const children = merges.get(nodeId) ? [merges.get(nodeId)!.left, merges.get(nodeId)!.right].map((id) => result.titles?.[String(id)]).filter(Boolean) : [];
    const contextLanguage = language === "auto" ? dominantLanguage(representative.map((note) => `${note.title} ${cleanNoteText(note, 180)}`).join(" ")) : language;
    const snippets = representative.map((note) => `- ${cleanNoteTitle(note.title)}: ${cleanNoteText(note, isMerge ? 150 : 220)}`).join("\n");
    const childText = children.length ? `Child titles (prioritize these themes): ${children.map((child) => cleanNoteTitle(String(child))).join(" | ")}\n` : "";
    prompts.push({ nodeId, members, memberFingerprint: stableFingerprint(members.map((member) => `${result.ids[member]}:${notes[member]?.hash || ""}`)), text: `Create one title in ${contextLanguage}. It must be 2-6 words, specific to the shared theme, and output title only.\n${childText}Representative context (use only for disambiguation):\n${snippets}`.slice(0, 4000) });
  }
  return prompts;
}

/**
 * Make one deliberately small corrective prompt after polluted output. Keep
 * the representative evidence, but remove the long multi-part instruction
 * and child-title boilerplate that can encourage the tiny model to continue
 * a formatted response. The worker still wraps this in the same safe Qwen
 * ChatML envelope and appends /no_think.
 */
export function buildRetryTitlePrompt(prompt: TitlePrompt): string {
  const language = prompt.text.match(/^Create one title in ([^.]+)\./i)?.[1] || "the requested input language";
  const marker = "Representative context (use only for disambiguation):";
  const context = prompt.text.split(marker, 2)[1]?.trim() || "the shared themes in the cluster";
  return `Title only. 2-6 words in ${language}. No tags.\n${context}`.slice(0, 1800);
}

export interface TitleCacheLike { get(key: string): ClusterTitleCacheEntry | undefined; set(entry: ClusterTitleCacheEntry): void; }
export interface GenerateTitlesOptions {
  language?: string;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
  onBatch?: (result: ClusterResult) => Promise<void> | void;
  cache?: TitleCacheLike;
  /** Ignore cache reads while still writing newly generated titles to it. */
  forceRegenerate?: boolean;
}

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
      for (const prompt of batch) {
        const key = titleCacheKey(prompt, options.language || "auto");
        const cached = options.forceRegenerate ? undefined : options.cache?.get(key);
        if (cached) { output.titles![String(prompt.nodeId)] = cached.title; statuses[String(prompt.nodeId)] = "cached"; durationsMs[String(prompt.nodeId)] = 0; } else uncached.push(prompt);
      }
      if (uncached.length) try {
        const started = Date.now();
        let values = await runtime.generate(uncached.map((prompt) => prompt.text), { maxNewTokens: 12, doSample: false, temperature: 0, repetitionPenalty: 1.15, noRepeatNgramSize: 3, signal: options.signal });
        const invalid = uncached.filter((prompt, index) => !validateTitle(values[index] || "", prompt.text).valid);
        // A bad deterministic response is cheap to retry and should never be
        // persisted. The corrective prompt is still sent through the same
        // serialized WebGPU runtime, so this does not reintroduce GPU hangs.
        if (invalid.length) {
          try {
            const retryValues = await runtime.generate(invalid.map(buildRetryTitlePrompt), { maxNewTokens: 12, doSample: false, temperature: 0, repetitionPenalty: 1.2, noRepeatNgramSize: 3, signal: options.signal });
            const retryByNode = new Map(invalid.map((prompt, index) => [prompt.nodeId, retryValues[index] || ""]));
            values = values.map((value, index) => retryByNode.has(uncached[index].nodeId) ? retryByNode.get(uncached[index].nodeId)! : value);
          } catch (retryError) {
            if (options.signal?.aborted || (retryError instanceof Error && retryError.message.toLowerCase().includes("cancel"))) throw retryError;
            // Keep valid first-pass titles even if the corrective request
            // fails; only the malformed nodes should be reported as failed.
            values = values.map((value, index) => invalid.some((prompt) => prompt.nodeId === uncached[index].nodeId) ? "" : value);
          }
        }
        uncached.forEach((prompt, index) => {
          const raw = values[index] || ""; const title = sanitizeTitle(raw); const validation = validateTitle(raw, prompt.text);
          durationsMs[String(prompt.nodeId)] = Math.max(0, Math.round((Date.now() - started) / Math.max(1, uncached.length)));
          if (!validation.valid) { statuses[String(prompt.nodeId)] = "failed"; errors[String(prompt.nodeId)] = validation.reason || "Invalid model output"; return; }
          output.titles![String(prompt.nodeId)] = title; statuses[String(prompt.nodeId)] = "generated"; options.cache?.set({ key: titleCacheKey(prompt, options.language || "auto"), title, nodeMembersFingerprint: prompt.memberFingerprint, savedAt: new Date().toISOString() });
        });
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
