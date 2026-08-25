import { TitleGenerationRuntime, TitleModelArtifact } from "./title";

interface TitleWorkerResponse { type: "READY" | "RESULT" | "ERROR"; backend?: "webgpu"; id?: number; values?: string[]; message?: string; }

export const TITLE_WORKER_INITIALIZATION_TIMEOUT_MS = 15_000;
export const TITLE_GENERATION_TIMEOUT_MS = 120_000;

export interface BrowserTitleRuntimeOptions {
  initializationTimeoutMs?: number;
  generationTimeoutMs?: number;
}

interface PendingGeneration {
  resolve: (values: string[]) => void;
  reject: (error: Error) => void;
  cleanup: () => void;
}

/** Chromium Worker adapter; intentionally has no CPU fallback. */
export class BrowserTitleRuntime implements TitleGenerationRuntime {
  private worker: Worker | null = null;
  private objectUrl: string | null = null;
  private nextId = 1;
  private ready: Promise<void> | null = null;
  private unusable = false;
  private readonly pending = new Map<number, PendingGeneration>();
  private readonly initializationTimeoutMs: number;
  private readonly generationTimeoutMs: number;
  readonly diagnostics = { backend: "webgpu" as const };
  constructor(private readonly source: string, private readonly artifact: TitleModelArtifact, private readonly ortWasm: ArrayBuffer, options: BrowserTitleRuntimeOptions = {}) {
    this.initializationTimeoutMs = options.initializationTimeoutMs ?? TITLE_WORKER_INITIALIZATION_TIMEOUT_MS;
    this.generationTimeoutMs = options.generationTimeoutMs ?? TITLE_GENERATION_TIMEOUT_MS;
  }
  async initialize(): Promise<void> {
    if (this.unusable) throw new Error("Title model worker is no longer available; create a new runtime.");
    if (this.ready) return this.ready;
    if (typeof Worker !== "function" || typeof Blob !== "function" || typeof URL.createObjectURL !== "function") throw new Error("Chromium Worker API is unavailable for the title model.");
    this.objectUrl = URL.createObjectURL(new Blob([this.source], { type: "text/javascript" }));
    const worker = new Worker(this.objectUrl);
    this.worker = worker;
    const ready = new Promise<void>((resolve, reject) => {
      let settled = false;
      const fail = (error: Error): void => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        worker.onmessage = null;
        worker.onerror = null;
        this.disposeWorker(true);
        reject(error);
      };
      const timer = setTimeout(() => fail(new Error(`Title model worker initialization timed out after ${this.initializationTimeoutMs}ms.`)), this.initializationTimeoutMs);
      worker.onmessage = (event) => {
        const response = event.data as TitleWorkerResponse;
        if (response.type === "READY") {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          worker.onmessage = this.handleWorkerMessage;
          worker.onerror = this.handleWorkerError;
          resolve();
        } else if (response.type === "ERROR") {
          fail(new Error(response.message || "Title model worker failed to initialize."));
        }
      };
      worker.onerror = () => fail(new Error("Title model worker failed to initialize."));
      try {
        worker.postMessage({ type: "INIT", model: this.artifact.model, tokenizer: this.artifact.tokenizer, config: this.artifact.config, generationConfig: this.artifact.generationConfig, tokenizerConfig: this.artifact.tokenizerConfig, ortWasm: this.ortWasm }, [this.artifact.model, this.artifact.tokenizer, this.artifact.config, this.artifact.generationConfig, this.artifact.tokenizerConfig, this.ortWasm]);
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)));
      }
    });
    this.ready = ready;
    return ready;
  }
  async generate(prompts: string[], options: { maxNewTokens: number; doSample: boolean; temperature: number; signal?: AbortSignal }): Promise<string[]> {
    await this.initialize();
    if (options.signal?.aborted) throw new Error("Clustering cancelled");
    const worker = this.worker;
    if (!worker) throw new Error("Title model worker is unavailable.");
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      let timer: ReturnType<typeof setTimeout>;
      const abort = (): void => {
        if (!this.pending.has(id)) return;
        this.disposeWorker(true, new Error("Clustering cancelled"));
      };
      const cleanup = (): void => {
        clearTimeout(timer);
        options.signal?.removeEventListener("abort", abort);
      };
      this.pending.set(id, { resolve, reject, cleanup });
      timer = setTimeout(() => {
        if (!this.pending.has(id)) return;
        this.disposeWorker(true, new Error(`Title model generation timed out after ${this.generationTimeoutMs}ms; worker terminated.`));
      }, this.generationTimeoutMs);
      options.signal?.addEventListener("abort", abort, { once: true });
      try {
        worker.postMessage({ type: "GENERATE", id, prompts, maxNewTokens: options.maxNewTokens, signal: options.signal?.aborted });
      } catch (error) {
        this.pending.delete(id);
        cleanup();
        this.disposeWorker(true);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  async terminate(): Promise<void> { this.disposeWorker(true, new Error("Title model worker terminated.")); }

  private readonly handleWorkerMessage = (event: MessageEvent<TitleWorkerResponse>): void => {
    const response = event.data;
    if (response.id === undefined) return;
    const pending = this.pending.get(response.id);
    if (!pending) return;
    this.pending.delete(response.id);
    pending.cleanup();
    if (response.type === "RESULT") pending.resolve(response.values || []);
    else pending.reject(new Error(response.message || "Title generation failed."));
  };

  private readonly handleWorkerError = (): void => {
    this.disposeWorker(true, new Error("Title model worker failed during generation; worker terminated."));
  };

  private disposeWorker(markUnusable: boolean, pendingError?: Error): void {
    const worker = this.worker;
    this.worker = null;
    if (worker) {
      worker.onmessage = null;
      worker.onerror = null;
      worker.terminate();
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    this.ready = null;
    if (markUnusable) this.unusable = true;
    if (pendingError) {
      const pending = [...this.pending.values()];
      this.pending.clear();
      pending.forEach((request) => { request.cleanup(); request.reject(pendingError); });
    }
  }
}
