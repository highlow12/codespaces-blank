import { requestUrl } from "obsidian";
import * as ort from "onnxruntime-web/wasm";
import * as ortWebGpu from "onnxruntime-web/webgpu";
import { CachedEmbedding, EmbeddingLogEntry, EmbeddingProviderId, LocalExecutionProvider, NoteRecord, PluginSettings } from "./types";

export interface EmbeddingProvider {
  readonly id: EmbeddingProviderId;
  readonly model: string;
  embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void, onNote?: (entry: EmbeddingLogEntry) => void, signal?: AbortSignal): Promise<CachedEmbedding[]>;
}

function checkCancelled(signal?: AbortSignal): void { if (signal?.aborted) throw new Error("Clustering cancelled"); }

export type EmbeddingNoteLogger = (entry: EmbeddingLogEntry) => void;

function errorMessages(error: unknown): string[] {
  const messages: string[] = [];
  let current: unknown = error;
  for (let depth = 0; current && depth < 4; depth++) {
    messages.push(current instanceof Error ? current.message : String(current));
    current = (current as { cause?: unknown })?.cause;
  }
  return messages.filter(Boolean);
}

function safeError(error: unknown, note?: NoteRecord): string {
  const message = errorMessages(error).join(": ");
  if (note && (note.content && message.includes(note.content) || note.title && message.includes(note.title))) return "Embedding provider error (message redacted)";
  return message.replace(/((?:^|[?&\s])(?:key|token|secret|authorization)=)[^&\s]+/gi, "$1[redacted]").slice(0, 500);
}

function logEntry(note: NoteRecord, provider: string, model: string, status: EmbeddingLogEntry["status"], started: number, error?: unknown): EmbeddingLogEntry {
  return { path: note.path, timestamp: new Date().toISOString(), provider, model, status, durationMs: Math.max(0, Math.round(performance.now() - started)), ...(error ? { error: safeError(error, note) } : {}) };
}

export interface LocalModelProgress {
  phase: "consent" | "model" | "tokenizer" | "verify" | "install" | "complete";
  progress: number;
  loadedBytes?: number;
  totalBytes?: number;
  detail?: string;
}

export const LOCAL_MODEL_VERSION = "2024-05-01";
export const LOCAL_MODEL_DIMENSION = 384;
export const LOCAL_MODEL_DESCRIPTOR = {
  id: "multilingual-e5-small",
  version: LOCAL_MODEL_VERSION,
  dimension: LOCAL_MODEL_DIMENSION,
  modelUrl: "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx",
  tokenizerUrl: "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json"
} as const;

export const LOCAL_ORT_MJS_ASSET = "ort-wasm-simd-threaded.mjs";
export const LOCAL_ORT_WASM_ASSET = "ort-wasm-simd-threaded.wasm";
export const LOCAL_ORT_WEBGPU_MJS_ASSET = "ort-wasm-simd-threaded.jsep.mjs";
export const LOCAL_ORT_WEBGPU_WASM_ASSET = "ort-wasm-simd-threaded.jsep.wasm";
/** Conservative bound: SentencePiece normally uses far fewer than 8 chars/token. */
export const LOCAL_TOKENIZER_CHAR_FACTOR = 8;

export interface LocalOrtAssetOverrides { mjs?: string; wasm?: string; wasmBinary?: ArrayBuffer; webgpuMjs?: string; webgpuWasm?: string; webgpuWasmBinary?: ArrayBuffer; revoke?: () => void; }
let localOrtAssetPrefix: string | null = null;
let localOrtAssetOverrides: LocalOrtAssetOverrides | undefined;

/** Configure the directory containing the bundled ORT .mjs/.wasm assets. */
export function configureLocalOrtAssets(prefix: string, overrides?: LocalOrtAssetOverrides): void {
  localOrtAssetOverrides?.revoke?.();
  localOrtAssetPrefix = prefix.endsWith("/") ? prefix : `${prefix}/`;
  localOrtAssetOverrides = overrides;
}

export function getLocalOrtAssetPrefix(): string | null { return localOrtAssetPrefix; }

export function disposeLocalOrtAssets(): void {
  localOrtAssetOverrides?.revoke?.();
  localOrtAssetOverrides = undefined;
}

export interface LocalModelArtifact {
  descriptor: typeof LOCAL_MODEL_DESCRIPTOR;
  model: ArrayBuffer;
  tokenizer: ArrayBuffer;
  modelSha256: string;
  tokenizerSha256: string;
}

export interface LocalModelStorage {
  exists(path: string): Promise<boolean>;
  read(path: string): Promise<ArrayBuffer>;
  write(path: string, data: ArrayBuffer): Promise<void>;
  remove(path: string): Promise<void>;
}

/** Adapter-backed storage keeps model weights outside plugin settings/data JSON. */
export class VaultLocalModelStorage implements LocalModelStorage {
  constructor(private readonly adapter: { exists(path: string): Promise<boolean>; readBinary(path: string): Promise<ArrayBuffer>; writeBinary(path: string, data: ArrayBuffer): Promise<void>; remove(path: string): Promise<void>; mkdir(path: string): Promise<void> }, private readonly prefix = ".obsidian/plugins/atomic-clusters/models") {}
  exists(path: string): Promise<boolean> { return this.adapter.exists(`${this.prefix}/${path}`); }
  read(path: string): Promise<ArrayBuffer> { return this.adapter.readBinary(`${this.prefix}/${path}`); }
  async write(path: string, data: ArrayBuffer): Promise<void> { const separator = path.lastIndexOf("/"); if (separator > 0) { const directory = `${this.prefix}/${path.slice(0, separator)}`; if (!(await this.adapter.exists(directory))) await this.adapter.mkdir(directory); } await this.adapter.writeBinary(`${this.prefix}/${path}`, data); }
  remove(path: string): Promise<void> { return this.adapter.remove(`${this.prefix}/${path}`); }
}

interface LocalModelManifest { id: string; version: string; dimension: number; modelSha256: string; tokenizerSha256: string; }

async function sha256(data: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("Local model integrity checks require Web Crypto SHA-256 support.");
  const digest = await subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function downloadBinary(url: string, phase: LocalModelProgress["phase"], onProgress?: (progress: LocalModelProgress) => void): Promise<ArrayBuffer> {
  // This is called only by LocalModelManager.downloadModel(), which is an
  // explicit user action from Settings. embed() never reaches this function.
  onProgress?.({ phase, progress: 0, detail: "Downloading…" });
  const response = await requestUrl({ url, method: "GET" });
  const bytes = (response as unknown as { arrayBuffer?: ArrayBuffer }).arrayBuffer;
  if (!(bytes instanceof ArrayBuffer) || bytes.byteLength === 0) throw new Error(`Local model download returned no bytes: ${url}`);
  const headers = (response as unknown as { headers?: Record<string, string> }).headers || {};
  const totalHeader = Object.entries(headers).find(([name]) => name.toLowerCase() === "content-length")?.[1];
  const totalBytes = totalHeader ? Number(totalHeader) : undefined;
  onProgress?.({ phase, progress: 1, loadedBytes: bytes.byteLength, totalBytes: Number.isFinite(totalBytes) ? totalBytes : undefined, detail: "Downloaded" });
  return bytes;
}

export class LocalModelManager {
  constructor(private readonly storage: LocalModelStorage, private readonly descriptor = LOCAL_MODEL_DESCRIPTOR) {}
  private path(name: string): string { return `${this.descriptor.id}/${this.descriptor.version}/${name}`; }
  private manifestPath(): string { return this.path("manifest.json"); }

  async status(): Promise<"missing" | "installed" | "corrupt"> {
    if (!(await this.storage.exists(this.manifestPath())) || !(await this.storage.exists(this.path("model.onnx"))) || !(await this.storage.exists(this.path("tokenizer.json")))) return "missing";
    // Settings calls this during every render. Read only the small manifest;
    // hashing a 470 MB model here blocks the Obsidian settings UI.
    try { this.validateManifest(await this.readManifest()); return "installed"; } catch { return "corrupt"; }
  }

  /** Explicit, potentially expensive integrity check for the Check model button. */
  async verifyStatus(): Promise<"missing" | "installed" | "corrupt"> {
    if (!(await this.storage.exists(this.manifestPath())) || !(await this.storage.exists(this.path("model.onnx"))) || !(await this.storage.exists(this.path("tokenizer.json")))) return "missing";
    try { await this.load(); return "installed"; } catch { return "corrupt"; }
  }

  async downloadModel(confirm: () => Promise<boolean>, onProgress?: (progress: LocalModelProgress) => void): Promise<void> {
    onProgress?.({ phase: "consent", progress: 0, detail: "Waiting for confirmation" });
    if (!(await confirm())) throw new Error("Local model download cancelled");
    onProgress?.({ phase: "consent", progress: 0.08, detail: "Download approved" });
    const model = await downloadBinary(this.descriptor.modelUrl, "model", (update) => onProgress?.({ ...update, progress: 0.08 + update.progress * 0.42 }));
    const tokenizer = await downloadBinary(this.descriptor.tokenizerUrl, "tokenizer", (update) => onProgress?.({ ...update, progress: 0.5 + update.progress * 0.28 }));
    onProgress?.({ phase: "verify", progress: 0.82, detail: "Calculating SHA-256 integrity hashes" });
    const manifest: LocalModelManifest = { id: this.descriptor.id, version: this.descriptor.version, dimension: this.descriptor.dimension, modelSha256: await sha256(model), tokenizerSha256: await sha256(tokenizer) };
    onProgress?.({ phase: "install", progress: 0.92, detail: "Saving model, tokenizer, and manifest" });
    await this.storage.write(this.path("model.onnx"), model);
    await this.storage.write(this.path("tokenizer.json"), tokenizer);
    await this.storage.write(this.manifestPath(), new TextEncoder().encode(JSON.stringify(manifest)).buffer);
    onProgress?.({ phase: "complete", progress: 1, detail: "Local model installed" });
  }

  async deleteModel(): Promise<void> {
    for (const path of [this.manifestPath(), this.path("model.onnx"), this.path("tokenizer.json")]) if (await this.storage.exists(path)) await this.storage.remove(path);
  }

  async load(): Promise<LocalModelArtifact> {
    const manifest = await this.readManifest();
    this.validateManifest(manifest);
    const [model, tokenizer] = await Promise.all([this.storage.read(this.path("model.onnx")), this.storage.read(this.path("tokenizer.json"))]);
    const [modelSha256, tokenizerSha256] = await Promise.all([sha256(model), sha256(tokenizer)]);
    if (modelSha256 !== manifest.modelSha256 || tokenizerSha256 !== manifest.tokenizerSha256) throw new Error("Installed local model failed its SHA-256 integrity check; delete and download it again.");
    return { descriptor: this.descriptor, model, tokenizer, modelSha256, tokenizerSha256 };
  }

  private async readManifest(): Promise<LocalModelManifest> {
    const manifestBytes = await this.storage.read(this.manifestPath());
    return JSON.parse(new TextDecoder().decode(manifestBytes)) as LocalModelManifest;
  }

  private validateManifest(manifest: LocalModelManifest): void {
    if (manifest.id !== this.descriptor.id || manifest.version !== this.descriptor.version || manifest.dimension !== this.descriptor.dimension || typeof manifest.modelSha256 !== "string" || typeof manifest.tokenizerSha256 !== "string") throw new Error("Installed local model manifest is stale or invalid; download the current model again.");
  }
}

export interface LocalTokenBatch { inputIds: number[][]; attentionMask: number[][]; tokenTypeIds?: number[][]; }
export interface LocalTokenizer { encode(texts: string[], maxLength: number): LocalTokenBatch; }
export interface LocalInferenceRuntime { embed(texts: string[], artifact: LocalModelArtifact, onProgress?: (done: number, total: number) => void, signal?: AbortSignal): Promise<number[][]>; }

export interface LocalRuntimeProgress { phase: "configuration" | "model" | "session" | "probe" | "complete"; progress: number; detail?: string; }

export interface LocalRuntimeDiagnostics { backend: "webgpu" | "wasm"; fallbackReason?: string; }

/** Errors raised while initializing/loading the ONNX backend must not be retried per note. */
export class LocalInferenceBackendError extends Error {
  constructor(message: string, cause?: unknown) { super(message); this.name = "LocalInferenceBackendError"; if (cause !== undefined) (this as Error & { cause?: unknown }).cause = cause; }
}

interface UnigramEntry { token: string; id: number; score: number; length: number; }

/** Minimal offline tokenizer for the SentencePiece Unigram tokenizer used by e5-small. */
export class UnigramTokenizer implements LocalTokenizer {
  private readonly entriesByFirst = new Map<string, UnigramEntry[]>();
  private readonly unknown: UnigramEntry;
  private readonly bos: number;
  private readonly eos: number;
  private readonly pad: number;
  constructor(tokenizerJson: string) {
    const parsed = JSON.parse(tokenizerJson) as { model?: { type?: string; vocab?: Array<[string, number]> | Record<string, number>; unk_id?: number }; added_tokens?: Array<{ id: number; content: string }> };
    const model = parsed.model;
    if (!model || model.type !== "Unigram" || !model.vocab) throw new Error("Unsupported local tokenizer: expected a SentencePiece Unigram tokenizer.json.");
    const vocab = Array.isArray(model.vocab) ? model.vocab.map(([token, score], id) => [token, score, id] as const) : Object.entries(model.vocab).map(([token, id]) => [token, 0, id] as const);
    const entries = vocab.filter(([token]) => !token.startsWith("<")).map(([token, score, id]) => ({ token, score: Number(score), id: Number(id), length: Array.from(token).length }));
    for (const entry of entries) { const first = Array.from(entry.token)[0]; if (!first) continue; this.entriesByFirst.set(first, [...(this.entriesByFirst.get(first) || []), entry]); }
    this.unknown = entries.find((entry) => entry.token === "<unk>") || { token: "<unk>", id: Number(model.unk_id ?? 3), score: -100, length: 1 };
    const special = new Map((parsed.added_tokens || []).map((token) => [token.content, token.id]));
    this.bos = special.get("<s>") ?? 0; this.eos = special.get("</s>") ?? 2; this.pad = special.get("<pad>") ?? 1;
  }

  encode(texts: string[], maxLength: number): LocalTokenBatch {
    const charBudget = Math.max(maxLength, maxLength * LOCAL_TOKENIZER_CHAR_FACTOR);
    const sequences = texts.map((text) => { const boundedText = this.boundText(text, charBudget); const ids = [this.bos, ...this.segment(boundedText), this.eos]; return ids.length > maxLength ? [...ids.slice(0, Math.max(1, maxLength - 1)), this.eos] : ids; });
    const width = Math.max(1, ...sequences.map((sequence) => sequence.length));
    return { inputIds: sequences.map((sequence) => [...sequence, ...new Array(width - sequence.length).fill(this.pad)]), attentionMask: sequences.map((sequence) => [...new Array(sequence.length).fill(1), ...new Array(width - sequence.length).fill(0)]) };
  }

  private boundText(text: string, charBudget: number): string {
    let offset = 0; let count = 0;
    while (offset < text.length && count < charBudget) { const codePoint = text.codePointAt(offset)!; offset += codePoint > 0xffff ? 2 : 1; count++; }
    return text.slice(0, offset);
  }

  private segment(text: string): number[] {
    const normalized = `▁${text.normalize("NFKC").replace(/\s+/g, "▁")}`;
    const chars = Array.from(normalized); const best: Array<{ score: number; ids: number[] } | undefined> = new Array(chars.length + 1); best[0] = { score: 0, ids: [] };
    for (let position = 0; position < chars.length; position++) {
      const current = best[position]; if (!current) continue;
      const candidates = this.entriesByFirst.get(chars[position]) || [];
      let matched = false;
      for (const entry of candidates) if (chars.slice(position, position + entry.length).join("") === entry.token) {
        matched = true; const next = position + entry.length; const score = current.score + entry.score; if (!best[next] || score > best[next]!.score) best[next] = { score, ids: [...current.ids, entry.id] };
      }
      if (!matched) { const next = position + 1; const score = current.score + this.unknown.score; if (!best[next] || score > best[next]!.score) best[next] = { score, ids: [...current.ids, this.unknown.id] }; }
    }
    return best[chars.length]?.ids || [this.unknown.id];
  }
}

/** Runtime seam for the optional ONNX binding and tokenizer implementation. */
export class OrtEmbeddingRuntime implements LocalInferenceRuntime {
  private sessionPromise?: Promise<any>;
  readonly diagnostics: LocalRuntimeDiagnostics;
  constructor(private readonly ort: { Tensor: new (type: string, data: ArrayLike<number> | BigInt64Array, dims: number[]) => any; InferenceSession: { create(model: ArrayBuffer, options?: { executionProviders?: readonly string[] }): Promise<any> } }, private readonly tokenizer: LocalTokenizer, private readonly batchSize = 16, private readonly maxLength = 512, backend: "webgpu" | "wasm" = "wasm", fallbackReason?: string) { this.diagnostics = { backend, ...(fallbackReason ? { fallbackReason } : {}) }; }
  async initialize(artifact: LocalModelArtifact): Promise<void> { await this.getSession(artifact); }
  async embed(texts: string[], artifact: LocalModelArtifact, onProgress?: (done: number, total: number) => void, signal?: AbortSignal): Promise<number[][]> {
    const session = await this.getSession(artifact);
    const output: number[][] = [];
    for (let start = 0; start < texts.length; start += this.batchSize) {
      checkCancelled(signal);
      const batch = texts.slice(start, start + this.batchSize);
      onProgress?.(start, texts.length);
      const tokens = this.tokenizer.encode(batch, this.maxLength);
      checkCancelled(signal);
      const inputNames = Array.isArray(session.inputNames) ? session.inputNames as string[] : ["input_ids", "attention_mask"];
      const supportedInputs = new Set(["input_ids", "attention_mask", "token_type_ids"]);
      const unsupportedInputs = inputNames.filter((name) => !supportedInputs.has(name));
      if (unsupportedInputs.length) throw new Error(`Unsupported local ONNX required inputs: ${unsupportedInputs.join(", ")}. Expected input_ids, attention_mask, and optional token_type_ids.`);
      if (!inputNames.includes("input_ids") || !inputNames.includes("attention_mask")) throw new Error("Local ONNX model must require input_ids and attention_mask.");
      const ids = flattenBigInt(tokens.inputIds);
      const mask = flattenBigInt(tokens.attentionMask);
      const tokenWidth = tokens.inputIds[0]?.length || 0;
      const feeds: Record<string, unknown> = {
        input_ids: new this.ort.Tensor("int64", ids, [batch.length, tokenWidth]),
        attention_mask: new this.ort.Tensor("int64", mask, [batch.length, tokens.attentionMask[0].length])
      };
      if (inputNames.includes("token_type_ids")) {
        const tokenTypeIds = tokens.tokenTypeIds || tokens.inputIds.map((row) => new Array(row.length).fill(0));
        if (tokenTypeIds.length !== batch.length || tokenTypeIds.some((row) => row.length !== tokenWidth)) throw new Error("Local tokenizer token_type_ids shape does not match input_ids.");
        feeds.token_type_ids = new this.ort.Tensor("int64", flattenBigInt(tokenTypeIds), [batch.length, tokenWidth]);
      }
      const result = await session.run(feeds);
      checkCancelled(signal);
      const tensor = result[session.outputNames?.[0] || Object.keys(result)[0]];
      const dimensions = tensor?.dims as number[] | undefined;
      if (!tensor || !dimensions || (dimensions.length !== 2 && dimensions.length !== 3)) throw new Error("Local ONNX model output must be [batch, hidden] or [batch, sequence, hidden].");
      const hidden = dimensions[dimensions.length - 1];
      const sequence = dimensions.length === 3 ? dimensions[1] : 1;
      if (dimensions[0] !== batch.length || (dimensions.length === 3 && dimensions[1] !== tokens.attentionMask[0].length)) throw new Error("Local ONNX output shape does not match tokenizer inputs.");
      for (let row = 0; row < batch.length; row++) {
        const vector = new Array<number>(hidden).fill(0); let weight = 0;
        for (let token = 0; token < sequence; token++) { const tokenWeight = tokens.attentionMask[row][token]; weight += tokenWeight; for (let column = 0; column < hidden; column++) vector[column] += Number(tensor.data[(row * sequence + token) * hidden + column]) * tokenWeight; }
        if (weight > 0) for (let column = 0; column < hidden; column++) vector[column] /= weight;
        const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
        output.push(norm > 1e-12 ? vector.map((value) => value / norm) : vector.fill(0));
      }
      onProgress?.(Math.min(start + batch.length, texts.length), texts.length);
      if (start + batch.length < texts.length) { await yieldToEventLoop(); checkCancelled(signal); }
    }
    return output;
  }

  private async getSession(artifact: LocalModelArtifact): Promise<any> {
    if (!this.sessionPromise) {
      this.sessionPromise = this.ort.InferenceSession.create(artifact.model, { executionProviders: [this.diagnostics.backend] }).catch((error) => {
        this.sessionPromise = undefined;
        throw new LocalInferenceBackendError("Local ONNX backend initialization failed; verify the bundled ORT assets and installed model.", error);
      });
    }
    return this.sessionPromise;
  }
}

function flattenBigInt(rows: number[][]): BigInt64Array { return BigInt64Array.from(rows.flat().map((value) => BigInt(value))); }

/** MessageChannel queues a task without the aggressive background timer throttling used by Electron. */
function yieldToEventLoop(): Promise<void> {
  if (typeof MessageChannel === "function") return new Promise((resolve) => { const channel = new MessageChannel(); channel.port1.onmessage = () => { channel.port1.close(); channel.port2.close(); resolve(); }; channel.port2.postMessage(undefined); });
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export interface SecretResolver { getSecret(reference: string): Promise<string | null>; }

export class GeminiEmbeddingProvider implements EmbeddingProvider {
  readonly id = "gemini" as const;
  constructor(private readonly settings: PluginSettings, private readonly secrets: SecretResolver, private readonly confirmTransmission: (count: number) => Promise<boolean> = async () => true) {}
  get model(): string { return this.settings.geminiModel; }

  async embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void, onNote?: EmbeddingNoteLogger, signal?: AbortSignal): Promise<CachedEmbedding[]> {
    checkCancelled(signal);
    const key = await this.secrets.getSecret(this.settings.geminiSecretRef);
    if (!key) throw new Error("Gemini API key is not configured. Store it in Obsidian SecretStorage and set its reference in settings.");
    if (!(await this.confirmTransmission(notes.length))) throw new Error("Gemini transmission cancelled");
    const result: CachedEmbedding[] = [];
    const batchSize = 32;
    let processed = 0;
    const embedBatch = async (batch: NoteRecord[]): Promise<void> => {
      checkCancelled(signal);
      const started = performance.now();
      try {
        const response = await withRetry(async () => requestUrl({
        url: `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(this.model)}:batchEmbedContents?key=${encodeURIComponent(key)}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requests: batch.map((note) => ({ model: `models/${this.model}`, content: { parts: [{ text: `Represent this note for semantic clustering.\n\ntitle: ${note.title}\n\npassage: ${note.content}` }] }, outputDimensionality: 768 })) })
        }), signal);
        checkCancelled(signal);
        const values = (response.json?.embeddings || []) as Array<{ values: number[] }>;
        if (values.length !== batch.length) throw new Error(`Gemini returned ${values.length} embeddings for ${batch.length} notes.`);
        if (values.some((value) => value.values.length !== 768)) throw new Error("Gemini embedding contract violation: expected 768-dimensional vectors.");
        values.forEach((value, index) => { result.push({ path: batch[index].path, hash: batch[index].hash, provider: this.id, model: this.model, vector: value.values }); onNote?.(logEntry(batch[index], this.id, this.model, "success", started)); });
        processed += batch.length; onProgress?.(processed, notes.length);
      } catch (error) {
        // Batch APIs can reject one malformed note along with its neighbours.
        // Split failed batches recursively so healthy notes still complete;
        // a single-note failure is recorded and skipped.
        if (batch.length > 1) { checkCancelled(signal); const midpoint = Math.ceil(batch.length / 2); await embedBatch(batch.slice(0, midpoint)); await embedBatch(batch.slice(midpoint)); return; }
        processed++; onNote?.(logEntry(batch[0], this.id, this.model, "failure", started, error)); onProgress?.(processed, notes.length);
      }
    };
    for (let start = 0; start < notes.length; start += batchSize) { checkCancelled(signal); await embedBatch(notes.slice(start, start + batchSize)); }
    return result;
  }
}

async function withRetry<T>(request: () => Promise<T>, signal?: AbortSignal, attempts = 4): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    checkCancelled(signal);
    try { return await request(); } catch (error) {
      lastError = error;
      const status = (error as { status?: number; response?: { status?: number } })?.status || (error as { response?: { status?: number } })?.response?.status;
      if (status !== 429 && (!status || status < 500) || attempt === attempts - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** attempt + Math.floor(Math.random() * 200)));
      checkCancelled(signal);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

export type LocalRuntimeFactory = (artifact: LocalModelArtifact, executionProvider?: LocalExecutionProvider) => Promise<LocalInferenceRuntime>;

function configureOrtBinding(binding: typeof ort, backend: "webgpu" | "wasm"): void {
  binding.env.wasm.numThreads = 1;
  const mjs = backend === "webgpu" ? localOrtAssetOverrides?.webgpuMjs : localOrtAssetOverrides?.mjs;
  const wasm = backend === "webgpu" ? localOrtAssetOverrides?.webgpuWasm : localOrtAssetOverrides?.wasm;
  const wasmBinary = backend === "webgpu" ? localOrtAssetOverrides?.webgpuWasmBinary : localOrtAssetOverrides?.wasmBinary;
  binding.env.wasm.wasmPaths = mjs ? { mjs, ...(wasm ? { wasm } : {}) } : (localOrtAssetPrefix || undefined);
  if (wasmBinary) binding.env.wasm.wasmBinary = wasmBinary;
}

function hasWebGpu(): boolean { return typeof navigator !== "undefined" && !!(navigator as Navigator & { gpu?: unknown }).gpu; }

export async function defaultLocalRuntimeFactory(artifact: LocalModelArtifact, executionProvider: LocalExecutionProvider = "auto"): Promise<LocalInferenceRuntime> {
  // The WASM ORT binding and tokenizer are bundled with the plugin. The hook
  // remains an override for alternate execution providers, but is no longer
  // required for a normal model-download -> offline-embed flow.
  const override = (globalThis as typeof globalThis & { __ATOMIC_CLUSTERS_LOCAL_ONNX__?: LocalInferenceRuntime }).__ATOMIC_CLUSTERS_LOCAL_ONNX__;
  if (override) return override;
  if (!localOrtAssetPrefix) throw new LocalInferenceBackendError("Local ORT assets are not configured for this vault/plugin installation.");
  const tokenizer = new UnigramTokenizer(new TextDecoder().decode(artifact.tokenizer));
  const create = async (backend: "webgpu" | "wasm", fallbackReason?: string): Promise<OrtEmbeddingRuntime> => {
    const binding = backend === "webgpu" ? ortWebGpu : ort;
    configureOrtBinding(binding, backend);
    const runtime = new OrtEmbeddingRuntime(binding, tokenizer, 16, 512, backend, fallbackReason);
    await runtime.initialize(artifact);
    return runtime;
  };
  if (executionProvider === "wasm") return create("wasm");
  if (executionProvider === "webgpu" && !hasWebGpu()) throw new LocalInferenceBackendError("WebGPU is unavailable in this Obsidian environment.");
  if (executionProvider === "auto" && !hasWebGpu()) return create("wasm", "WebGPU is unavailable; using WASM CPU.");
  try { return await create("webgpu"); }
  catch (error) {
    if (executionProvider === "webgpu") throw new LocalInferenceBackendError("WebGPU local ONNX session initialization failed.", error);
    return create("wasm", `WebGPU initialization failed; using WASM CPU: ${safeError(error)}`);
  }
}

export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly id = "local" as const;
  private runtimeState?: { artifact: LocalModelArtifact; runtime: LocalInferenceRuntime };
  constructor(private readonly settings: PluginSettings, private readonly runner?: (texts: string[], model: string) => Promise<number[][]>, private readonly manager?: LocalModelManager, private readonly runtimeFactory: LocalRuntimeFactory = defaultLocalRuntimeFactory) {}
  get model(): string { return `${this.settings.localModel}@${LOCAL_MODEL_VERSION}`; }
  get runtimeDiagnostics(): LocalRuntimeDiagnostics | undefined { return (this.runtimeState?.runtime as LocalInferenceRuntime & { diagnostics?: LocalRuntimeDiagnostics })?.diagnostics; }

  async preflight(onProgress?: (progress: LocalRuntimeProgress) => void, signal?: AbortSignal): Promise<void> {
    checkCancelled(signal);
    onProgress?.({ phase: "configuration", progress: 0.05, detail: "Checking bundled ORT renderer assets" });
    if (this.runner) {
      onProgress?.({ phase: "probe", progress: 0.5, detail: "Running a safe local embedding probe" });
      checkCancelled(signal);
      const vectors = await this.runner(["passage: Atomic Clusters local runtime preflight"], this.model);
      checkCancelled(signal);
      if (vectors.length !== 1) throw new Error("Local runtime preflight returned an invalid number of vectors.");
      if (vectors[0].length !== LOCAL_MODEL_DIMENSION || vectors[0].some((value) => !Number.isFinite(value))) throw new Error(`Local runtime preflight returned an invalid ${LOCAL_MODEL_DIMENSION}-dimensional vector.`);
    } else {
      onProgress?.({ phase: "model", progress: 0.25, detail: "Loading and verifying the installed model" });
      onProgress?.({ phase: "session", progress: 0.5, detail: "Initializing the configured local ONNX backend" });
      const runtimeState = await this.getRuntime();
      const diagnostics = (runtimeState.runtime as LocalInferenceRuntime & { diagnostics?: LocalRuntimeDiagnostics }).diagnostics;
      onProgress?.({ phase: "session", progress: 0.5, detail: diagnostics ? `${diagnostics.backend === "webgpu" ? "WebGPU" : "WASM CPU"} backend ready${diagnostics.fallbackReason ? ` · ${diagnostics.fallbackReason}` : ""}` : "ONNX session ready" });
      const vectors = await runtimeState.runtime.embed(["passage: Atomic Clusters local runtime preflight"], runtimeState.artifact, undefined, signal);
      checkCancelled(signal);
      if (vectors.length !== 1 || vectors[0].length !== LOCAL_MODEL_DIMENSION || vectors[0].some((value) => !Number.isFinite(value))) throw new Error(`Local runtime preflight returned an invalid ${LOCAL_MODEL_DIMENSION}-dimensional vector.`);
    }
    onProgress?.({ phase: "complete", progress: 1, detail: "Local runtime is ready" });
  }

  async embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void, onNote?: EmbeddingNoteLogger, signal?: AbortSignal): Promise<CachedEmbedding[]> {
    checkCancelled(signal);
    const texts = notes.map((note) => `passage: ${note.title}\n${note.content}`);
    let reportedProgress = 0;
    const reportProgress = (done: number): void => { reportedProgress = Math.max(reportedProgress, Math.min(notes.length, done)); onProgress?.(reportedProgress, notes.length); };
    reportProgress(0);
    const result: CachedEmbedding[] = [];
    const validateVector = (vector: number[]): void => { if (vector.length !== LOCAL_MODEL_DIMENSION || vector.some((value) => !Number.isFinite(value))) throw new Error(`Local embedding contract violation: expected finite ${LOCAL_MODEL_DIMENSION}-dimensional vector.`); };
    const appendOne = (index: number, vector: number[], started: number): void => {
      validateVector(vector);
      result.push({ path: notes[index].path, hash: notes[index].hash, provider: this.id, model: this.model, vector });
      onNote?.(logEntry(notes[index], this.id, this.model, "success", started));
    };
    const recoverIndividually = async (run: (index: number) => Promise<number[][]>): Promise<void> => {
      for (let index = 0; index < notes.length; index++) {
        checkCancelled(signal);
        const started = performance.now();
        try { const vectors = await run(index); checkCancelled(signal); if (vectors.length !== 1) throw new Error("Local embedding runtime returned an invalid number of vectors."); appendOne(index, vectors[0], started); }
        catch (error) { onNote?.(logEntry(notes[index], this.id, this.model, "failure", started, error)); }
        reportProgress(index + 1);
      }
    };
    if (this.runner) {
      const started = performance.now();
      try { const vectors = await this.runner(texts, this.model); checkCancelled(signal); if (vectors.length !== notes.length) throw new Error("Local embedding runner returned an invalid number of vectors."); vectors.forEach(validateVector); vectors.forEach((vector, index) => appendOne(index, vector, started)); reportProgress(notes.length); }
      catch { result.length = 0; await recoverIndividually((index) => this.runner!([texts[index]], this.model)); }
    } else {
      if (!this.manager) throw new Error("Local model is not configured. Download multilingual-e5-small from Settings first.");
      const { artifact, runtime } = await this.getRuntime();
      const started = performance.now();
      try { const vectors = await runtime.embed(texts, artifact, (done) => reportProgress(done), signal); checkCancelled(signal); if (vectors.length !== notes.length) throw new Error("Local embedding runtime returned an invalid number of vectors."); vectors.forEach(validateVector); vectors.forEach((vector, index) => appendOne(index, vector, started)); }
      catch (error) {
        // Backend initialization/asset failures are systemic. Retrying every
        // note would only repeat the same failure hundreds of times. Runtime
        // implementations may opt into isolation for input-specific errors.
        if (error instanceof LocalInferenceBackendError) throw error;
        result.length = 0; await recoverIndividually((index) => runtime.embed([texts[index]], artifact, undefined, signal));
      }
    }
    reportProgress(notes.length);
    return result;
  }

  private async getRuntime(): Promise<{ artifact: LocalModelArtifact; runtime: LocalInferenceRuntime }> {
    if (!this.runtimeState) {
      if (!this.manager) throw new Error("Local model is not configured. Download multilingual-e5-small from Settings first.");
      const artifact = await this.manager.load();
      const runtime = await this.runtimeFactory(artifact, this.settings.localExecutionProvider || "auto");
      this.runtimeState = { artifact, runtime };
    }
    return this.runtimeState;
  }

  get status(): "unavailable" | "ready" { return this.runner || this.manager ? "ready" : "unavailable"; }
  async downloadModel(confirm: () => Promise<boolean> = async () => false, onProgress?: (progress: LocalModelProgress) => void): Promise<void> { if (!this.manager) throw new Error("Local model storage is not configured."); await this.manager.downloadModel(confirm, onProgress); }
  async deleteModel(): Promise<void> { if (!this.manager) throw new Error("Local model storage is not configured."); await this.manager.deleteModel(); }
}
