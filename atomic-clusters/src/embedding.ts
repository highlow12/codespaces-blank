import { requestUrl } from "obsidian";
import { CachedEmbedding, EmbeddingProviderId, NoteRecord, PluginSettings } from "./types";

export interface EmbeddingProvider {
  readonly id: EmbeddingProviderId;
  readonly model: string;
  embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void): Promise<CachedEmbedding[]>;
}

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

/** Local provider boundary. Model execution is intentionally injected by the desktop build. */
export class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly id = "local" as const;
  constructor(private readonly settings: PluginSettings, private readonly runner?: (texts: string[], model: string) => Promise<number[][]>) {}
  get model(): string { return this.settings.localModel; }

  async embed(notes: NoteRecord[], onProgress?: (done: number, total: number) => void): Promise<CachedEmbedding[]> {
    if (!this.runner) throw new Error("Local provider is unavailable in this build: no ONNX runner/model asset is bundled. Select Gemini API.");
    const vectors = await this.runner(notes.map((note) => `passage: ${note.title}\n${note.content}`), this.model);
    if (vectors.length !== notes.length) throw new Error("Local embedding runner returned an invalid number of vectors.");
    const result = vectors.map((vector, index) => ({ path: notes[index].path, hash: notes[index].hash, provider: this.id, model: this.model, vector }));
    onProgress?.(notes.length, notes.length);
    return result;
  }

  get status(): "unavailable" | "ready" { return this.runner ? "ready" : "unavailable"; }
  async downloadModel(): Promise<void> { throw new Error("Local multilingual-e5-small is unavailable in this build: no ONNX runtime/model asset is bundled."); }
}
