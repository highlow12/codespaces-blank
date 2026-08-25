import { requestUrl } from "obsidian";
import * as ort from "onnxruntime-web/wasm";
import { CachedEmbedding, EmbeddingProviderId, NoteRecord, PluginSettings } from "./types";

export interface EmbeddingProvider {
  readonly id: EmbeddingProviderId;
  readonly model: string;
  embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void): Promise<CachedEmbedding[]>;
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

let localOrtAssetPrefix = "./";

/** Configure the directory containing the bundled ORT .mjs/.wasm assets. */
export function configureLocalOrtAssets(prefix: string): void {
  localOrtAssetPrefix = prefix.endsWith("/") ? prefix : `${prefix}/`;
}

export function getLocalOrtAssetPrefix(): string { return localOrtAssetPrefix; }

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

async function downloadBinary(url: string): Promise<ArrayBuffer> {
  // This is called only by LocalModelManager.downloadModel(), which is an
  // explicit user action from Settings. embed() never reaches this function.
  const response = await requestUrl({ url, method: "GET" });
  const bytes = (response as unknown as { arrayBuffer?: ArrayBuffer }).arrayBuffer;
  if (!(bytes instanceof ArrayBuffer) || bytes.byteLength === 0) throw new Error(`Local model download returned no bytes: ${url}`);
  return bytes;
}

export class LocalModelManager {
  constructor(private readonly storage: LocalModelStorage, private readonly descriptor = LOCAL_MODEL_DESCRIPTOR) {}
  private path(name: string): string { return `${this.descriptor.id}/${this.descriptor.version}/${name}`; }
  private manifestPath(): string { return this.path("manifest.json"); }

  async status(): Promise<"missing" | "installed" | "corrupt"> {
    if (!(await this.storage.exists(this.manifestPath())) || !(await this.storage.exists(this.path("model.onnx"))) || !(await this.storage.exists(this.path("tokenizer.json")))) return "missing";
    try { await this.load(); return "installed"; } catch { return "corrupt"; }
  }

  async downloadModel(confirm: () => Promise<boolean>): Promise<void> {
    if (!(await confirm())) throw new Error("Local model download cancelled");
    const [model, tokenizer] = await Promise.all([downloadBinary(this.descriptor.modelUrl), downloadBinary(this.descriptor.tokenizerUrl)]);
    const manifest: LocalModelManifest = { id: this.descriptor.id, version: this.descriptor.version, dimension: this.descriptor.dimension, modelSha256: await sha256(model), tokenizerSha256: await sha256(tokenizer) };
    await this.storage.write(this.path("model.onnx"), model);
    await this.storage.write(this.path("tokenizer.json"), tokenizer);
    await this.storage.write(this.manifestPath(), new TextEncoder().encode(JSON.stringify(manifest)).buffer);
  }

  async deleteModel(): Promise<void> {
    for (const path of [this.manifestPath(), this.path("model.onnx"), this.path("tokenizer.json")]) if (await this.storage.exists(path)) await this.storage.remove(path);
  }

  async load(): Promise<LocalModelArtifact> {
    const manifestBytes = await this.storage.read(this.manifestPath());
    const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as LocalModelManifest;
    if (manifest.id !== this.descriptor.id || manifest.version !== this.descriptor.version || manifest.dimension !== this.descriptor.dimension) throw new Error("Installed local model manifest is stale; download the current model again.");
    const [model, tokenizer] = await Promise.all([this.storage.read(this.path("model.onnx")), this.storage.read(this.path("tokenizer.json"))]);
    const [modelSha256, tokenizerSha256] = await Promise.all([sha256(model), sha256(tokenizer)]);
    if (modelSha256 !== manifest.modelSha256 || tokenizerSha256 !== manifest.tokenizerSha256) throw new Error("Installed local model failed its SHA-256 integrity check; delete and download it again.");
    return { descriptor: this.descriptor, model, tokenizer, modelSha256, tokenizerSha256 };
  }
}

export interface LocalTokenBatch { inputIds: number[][]; attentionMask: number[][]; tokenTypeIds?: number[][]; }
export interface LocalTokenizer { encode(texts: string[], maxLength: number): LocalTokenBatch; }
export interface LocalInferenceRuntime { embed(texts: string[], artifact: LocalModelArtifact): Promise<number[][]>; }

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
    const sequences = texts.map((text) => { const ids = [this.bos, ...this.segment(text), this.eos]; return ids.length > maxLength ? [...ids.slice(0, Math.max(1, maxLength - 1)), this.eos] : ids; });
    const width = Math.max(1, ...sequences.map((sequence) => sequence.length));
    return { inputIds: sequences.map((sequence) => [...sequence, ...new Array(width - sequence.length).fill(this.pad)]), attentionMask: sequences.map((sequence) => [...new Array(sequence.length).fill(1), ...new Array(width - sequence.length).fill(0)]) };
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
  constructor(private readonly ort: { Tensor: new (type: string, data: ArrayLike<number> | BigInt64Array, dims: number[]) => any; InferenceSession: { create(model: ArrayBuffer): Promise<any> } }, private readonly tokenizer: LocalTokenizer, private readonly batchSize = 16, private readonly maxLength = 512) {}
  async embed(texts: string[], artifact: LocalModelArtifact): Promise<number[][]> {
    const session = await this.ort.InferenceSession.create(artifact.model);
    const output: number[][] = [];
    for (let start = 0; start < texts.length; start += this.batchSize) {
      const batch = texts.slice(start, start + this.batchSize);
      const tokens = this.tokenizer.encode(batch, this.maxLength);
      const ids = flattenBigInt(tokens.inputIds);
      const mask = flattenBigInt(tokens.attentionMask);
      const feeds: Record<string, unknown> = {
        input_ids: new this.ort.Tensor("int64", ids, [batch.length, tokens.inputIds[0].length]),
        attention_mask: new this.ort.Tensor("int64", mask, [batch.length, tokens.attentionMask[0].length])
      };
      if (tokens.tokenTypeIds && session.inputNames?.includes("token_type_ids")) feeds.token_type_ids = new this.ort.Tensor("int64", flattenBigInt(tokens.tokenTypeIds), [batch.length, tokens.tokenTypeIds[0].length]);
      const result = await session.run(feeds);
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
    }
    return output;
  }
}

function flattenBigInt(rows: number[][]): BigInt64Array { return BigInt64Array.from(rows.flat().map((value) => BigInt(value))); }

export interface SecretResolver { getSecret(reference: string): Promise<string | null>; }

export class GeminiEmbeddingProvider implements EmbeddingProvider {
  readonly id = "gemini" as const;
  constructor(private readonly settings: PluginSettings, private readonly secrets: SecretResolver, private readonly confirmTransmission: (count: number) => Promise<boolean> = async () => true) {}
  get model(): string { return this.settings.geminiModel; }

  async embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void): Promise<CachedEmbedding[]> {
    const key = await this.secrets.getSecret(this.settings.geminiSecretRef);
    if (!key) throw new Error("Gemini API key is not configured. Store it in Obsidian SecretStorage and set its reference in settings.");
    if (!(await this.confirmTransmission(notes.length))) throw new Error("Gemini transmission cancelled");
    const result: CachedEmbedding[] = [];
    const batchSize = 32;
    for (let start = 0; start < notes.length; start += batchSize) {
      const batch = notes.slice(start, start + batchSize);
      const response = await withRetry(async () => requestUrl({
        url: `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(this.model)}:batchEmbedContents?key=${encodeURIComponent(key)}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requests: batch.map((note) => ({ model: `models/${this.model}`, content: { parts: [{ text: `Represent this note for semantic clustering.\n\ntitle: ${note.title}\n\npassage: ${note.content}` }] }, outputDimensionality: 768 })) })
      }));
      const values = (response.json?.embeddings || []) as Array<{ values: number[] }>;
      if (values.length !== batch.length) throw new Error(`Gemini returned ${values.length} embeddings for ${batch.length} notes.`);
      if (values.some((value) => value.values.length !== 768)) throw new Error("Gemini embedding contract violation: expected 768-dimensional vectors.");
      values.forEach((value, index) => result.push({ path: batch[index].path, hash: batch[index].hash, provider: this.id, model: this.model, vector: value.values }));
      onProgress?.(Math.min(start + batch.length, notes.length), notes.length);
    }
    return result;
  }
}

async function withRetry<T>(request: () => Promise<T>, attempts = 4): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try { return await request(); } catch (error) {
      lastError = error;
      const status = (error as { status?: number; response?: { status?: number } })?.status || (error as { response?: { status?: number } })?.response?.status;
      if (status !== 429 && (!status || status < 500) || attempt === attempts - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, 500 * 2 ** attempt + Math.floor(Math.random() * 200)));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

export type LocalRuntimeFactory = (artifact: LocalModelArtifact) => Promise<LocalInferenceRuntime>;

export async function defaultLocalRuntimeFactory(artifact: LocalModelArtifact): Promise<LocalInferenceRuntime> {
  // The WASM ORT binding and tokenizer are bundled with the plugin. The hook
  // remains an override for alternate execution providers, but is no longer
  // required for a normal model-download -> offline-embed flow.
  const override = (globalThis as typeof globalThis & { __ATOMIC_CLUSTERS_LOCAL_ONNX__?: LocalInferenceRuntime }).__ATOMIC_CLUSTERS_LOCAL_ONNX__;
  if (override) return override;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = localOrtAssetPrefix;
  const tokenizer = new UnigramTokenizer(new TextDecoder().decode(artifact.tokenizer));
  return new OrtEmbeddingRuntime(ort, tokenizer);
}

export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly id = "local" as const;
  constructor(private readonly settings: PluginSettings, private readonly runner?: (texts: string[], model: string) => Promise<number[][]>, private readonly manager?: LocalModelManager, private readonly runtimeFactory: LocalRuntimeFactory = defaultLocalRuntimeFactory) {}
  get model(): string { return `${this.settings.localModel}@${LOCAL_MODEL_VERSION}`; }

  async embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void): Promise<CachedEmbedding[]> {
    const texts = notes.map((note) => `passage: ${note.title}\n${note.content}`);
    let vectors: number[][];
    if (this.runner) vectors = await this.runner(texts, this.model);
    else {
      if (!this.manager) throw new Error("Local model is not configured. Download multilingual-e5-small from Settings first.");
      const artifact = await this.manager.load();
      vectors = await (await this.runtimeFactory(artifact)).embed(texts, artifact);
    }
    if (vectors.length !== notes.length) throw new Error("Local embedding runner returned an invalid number of vectors.");
    if (vectors.some((vector) => vector.length !== LOCAL_MODEL_DIMENSION || vector.some((value) => !Number.isFinite(value)))) throw new Error(`Local embedding contract violation: expected finite ${LOCAL_MODEL_DIMENSION}-dimensional vectors.`);
    const result = vectors.map((vector, index) => ({ path: notes[index].path, hash: notes[index].hash, provider: this.id, model: this.model, vector }));
    onProgress?.(notes.length, notes.length);
    return result;
  }

  get status(): "unavailable" | "ready" { return this.runner || this.manager ? "ready" : "unavailable"; }
  async downloadModel(confirm: () => Promise<boolean> = async () => false): Promise<void> { if (!this.manager) throw new Error("Local model storage is not configured."); await this.manager.downloadModel(confirm); }
  async deleteModel(): Promise<void> { if (!this.manager) throw new Error("Local model storage is not configured."); await this.manager.deleteModel(); }
}
