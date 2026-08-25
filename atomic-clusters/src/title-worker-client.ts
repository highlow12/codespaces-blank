import { TitleGenerationRuntime, TitleModelArtifact } from "./title";

interface TitleWorkerResponse { type: "READY" | "RESULT" | "ERROR"; backend?: "webgpu"; id?: number; values?: string[]; message?: string; }

/** Chromium Worker adapter; intentionally has no CPU fallback. */
export class BrowserTitleRuntime implements TitleGenerationRuntime {
  private worker: Worker | null = null;
  private objectUrl: string | null = null;
  private nextId = 1;
  private ready: Promise<void> | null = null;
  readonly diagnostics = { backend: "webgpu" as const };
  constructor(private readonly source: string, private readonly artifact: TitleModelArtifact, private readonly ortWasm: ArrayBuffer) {}
  async initialize(): Promise<void> {
    if (this.ready) return this.ready;
    if (typeof Worker !== "function" || typeof Blob !== "function" || typeof URL.createObjectURL !== "function") throw new Error("Chromium Worker API is unavailable for the title model.");
    this.objectUrl = URL.createObjectURL(new Blob([this.source], { type: "text/javascript" })); this.worker = new Worker(this.objectUrl);
    this.ready = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Title model worker initialization timed out.")), 15000);
      this.worker!.onmessage = (event) => { const response = event.data as TitleWorkerResponse; if (response.type === "READY") { clearTimeout(timer); resolve(); } else if (response.type === "ERROR") { clearTimeout(timer); reject(new Error(response.message || "Title model worker failed.")); } };
      this.worker!.onerror = () => { clearTimeout(timer); reject(new Error("Title model worker failed to initialize.")); };
      this.worker!.postMessage({ type: "INIT", model: this.artifact.model, tokenizer: this.artifact.tokenizer, config: this.artifact.config, generationConfig: this.artifact.generationConfig, tokenizerConfig: this.artifact.tokenizerConfig, ortWasm: this.ortWasm }, [this.artifact.model, this.artifact.tokenizer, this.artifact.config, this.artifact.generationConfig, this.artifact.tokenizerConfig, this.ortWasm]);
    });
    return this.ready;
  }
  async generate(prompts: string[], options: { maxNewTokens: number; doSample: boolean; temperature: number; signal?: AbortSignal }): Promise<string[]> {
    await this.initialize();
    if (options.signal?.aborted) throw new Error("Clustering cancelled");
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const worker = this.worker!;
      const onMessage = (event: MessageEvent<TitleWorkerResponse>) => { const response = event.data; if (response.id !== id) return; worker.removeEventListener("message", onMessage); if (response.type === "RESULT") resolve(response.values || []); else reject(new Error(response.message || "Title generation failed.")); };
      worker.addEventListener("message", onMessage); worker.postMessage({ type: "GENERATE", id, prompts, maxNewTokens: options.maxNewTokens, signal: options.signal?.aborted });
      if (options.signal) options.signal.addEventListener("abort", () => { worker.removeEventListener("message", onMessage); reject(new Error("Clustering cancelled")); }, { once: true });
    });
  }
  async terminate(): Promise<void> { this.worker?.terminate(); this.worker = null; this.ready = null; if (this.objectUrl) { URL.revokeObjectURL(this.objectUrl); this.objectUrl = null; } }
}
