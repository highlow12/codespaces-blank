import { Worker as NodeWorker } from "worker_threads";
import { clusterEmbeddings } from "./clustering";
import { ClusteringConfig, ClusterResult, WorkerRequest, WorkerResponse } from "./types";
import { loadWasmKernel } from "./wasm-loader";

export interface WorkerProgress { (phase: string, progress: number): void; }
export interface ClusteringWorker { init(): Promise<void>; run(ids: string[], embeddings: number[][], config: ClusteringConfig, onProgress?: WorkerProgress): Promise<ClusterResult>; cancel(): void; terminate(): Promise<void>; }

export class NodeClusteringWorker implements ClusteringWorker {
  private worker: NodeWorker | null = null;
  private pending: { resolve: (result: ClusterResult) => void; reject: (error: Error) => void; jobId: string; onProgress?: WorkerProgress } | null = null;
  constructor(private readonly workerSource: string) {}

  async init(): Promise<void> {
    if (this.worker) return;
    this.worker = new NodeWorker(this.workerSource, { eval: true });
    this.worker.on("message", (response: WorkerResponse) => this.onMessage(response));
    this.worker.on("error", (error) => { this.pending?.reject(error); this.pending = null; });
    await new Promise<void>((resolve, reject) => { const timer = setTimeout(() => reject(new Error("Clustering worker initialization timed out.")), 10000); const original = this.onReady; this.onReady = () => { clearTimeout(timer); this.onReady = original; resolve(); }; this.worker!.postMessage({ type: "INIT", version: 1 } satisfies WorkerRequest); });
  }

  run(ids: string[], embeddings: number[][], config: ClusteringConfig, onProgress?: WorkerProgress): Promise<ClusterResult> {
    if (!this.worker) return Promise.reject(new Error("Clustering worker is not initialized."));
    if (this.pending) return Promise.reject(new Error("A clustering job is already running."));
    const jobId = `cluster-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    return new Promise((resolve, reject) => { this.pending = { resolve, reject, jobId, onProgress }; this.worker!.postMessage({ type: "CLUSTER", jobId, ids, embeddings, config } satisfies WorkerRequest); });
  }

  cancel(): void {
    const pending = this.pending;
    if (!pending) return;
    pending.reject(new Error("Clustering cancelled"));
    this.pending = null;
    void this.terminate().then(() => this.init().catch(() => undefined));
  }
  async terminate(): Promise<void> { const worker = this.worker; this.worker = null; if (worker) await worker.terminate(); this.pending = null; }
  private onReady: () => void = () => undefined;
  private onMessage(response: WorkerResponse): void { if (response.type === "READY") { this.onReady(); return; } if (!this.pending || response.jobId !== this.pending.jobId) return; if (response.type === "PROGRESS") this.pending.onProgress?.(response.phase, response.progress); else if (response.type === "RESULT") { this.pending.resolve(response.result); this.pending = null; } else if (response.type === "ERROR") { this.pending.reject(new Error(response.message)); this.pending = null; } }
}

/** Chromium Worker fallback for Electron platforms where worker_threads is unavailable. */
export class BrowserClusteringWorker implements ClusteringWorker {
  private worker: globalThis.Worker | null = null;
  private objectUrl: string | null = null;
  private pending: { resolve: (result: ClusterResult) => void; reject: (error: Error) => void; jobId: string; onProgress?: WorkerProgress } | null = null;
  private initReject: ((error: Error) => void) | null = null;
  constructor(private readonly workerSource: string) {}

  async init(): Promise<void> {
    if (this.worker) return;
    if (typeof globalThis.Worker !== "function" || typeof Blob !== "function" || typeof URL.createObjectURL !== "function") throw new Error("Chromium Worker API is unavailable.");
    this.objectUrl = URL.createObjectURL(new Blob([this.workerSource], { type: "text/javascript" }));
    this.worker = new globalThis.Worker(this.objectUrl);
    this.worker.onmessage = (event) => this.onMessage(event.data as WorkerResponse);
    this.worker.onerror = (event) => { const error = new Error(event.message || "Browser clustering worker failed."); this.initReject?.(error); this.initReject = null; this.pending?.reject(error); this.pending = null; };
    await new Promise<void>((resolve, reject) => { const timer = setTimeout(() => { this.initReject = null; reject(new Error("Browser clustering worker initialization timed out.")); }, 10000); this.initReject = (error) => { clearTimeout(timer); reject(error); }; const original = this.onReady; this.onReady = () => { clearTimeout(timer); this.initReject = null; this.onReady = original; resolve(); }; this.worker!.postMessage({ type: "INIT", version: 1 } satisfies WorkerRequest); });
  }

  run(ids: string[], embeddings: number[][], config: ClusteringConfig, onProgress?: WorkerProgress): Promise<ClusterResult> {
    if (!this.worker) return Promise.reject(new Error("Browser clustering worker is not initialized."));
    if (this.pending) return Promise.reject(new Error("A clustering job is already running."));
    const jobId = `cluster-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    return new Promise((resolve, reject) => { this.pending = { resolve, reject, jobId, onProgress }; this.worker!.postMessage({ type: "CLUSTER", jobId, ids, embeddings, config } satisfies WorkerRequest); });
  }

  cancel(): void { if (this.pending) { this.pending.reject(new Error("Clustering cancelled")); this.pending = null; } this.worker?.postMessage({ type: "CANCEL" }); }
  async terminate(): Promise<void> { const worker = this.worker; this.worker = null; this.pending = null; if (worker) worker.terminate(); if (this.objectUrl) { URL.revokeObjectURL(this.objectUrl); this.objectUrl = null; } }
  private onReady: () => void = () => undefined;
  private onMessage(response: WorkerResponse): void { if (response.type === "READY") { this.onReady(); return; } if (!this.pending || response.jobId !== this.pending.jobId) return; if (response.type === "PROGRESS") this.pending.onProgress?.(response.phase, response.progress); else if (response.type === "RESULT") { this.pending.resolve(response.result); this.pending = null; } else if (response.type === "ERROR") { this.pending.reject(new Error(response.message)); this.pending = null; } }
}

/**
 * Electron builds can expose worker_threads while their V8 platform refuses
 * to construct a Worker. Keep clustering usable in that environment by
 * running the same orchestration asynchronously in the renderer. UMAP's
 * fitAsync yields between epochs; the signal is checked at every available
 * clustering boundary and cancellation never discards the embedding cache.
 */
export class InProcessClusteringWorker implements ClusteringWorker {
  private pending: { reject: (error: Error) => void } | null = null;
  private cancelled = false;
  private running = false;

  async init(): Promise<void> { return undefined; }

  run(ids: string[], embeddings: number[][], config: ClusteringConfig, onProgress?: WorkerProgress): Promise<ClusterResult> {
    if (this.running) return Promise.reject(new Error("A clustering job is already running."));
    this.running = true;
    this.cancelled = false;
    const worker = this;
    return new Promise((resolve, reject) => {
      this.pending = { reject };
      void yieldToEventLoop().then(async () => {
        try {
          if (worker.cancelled) throw new Error("Clustering cancelled");
          const result = await clusterEmbeddings(ids, embeddings, config, {
            kernel: loadWasmKernel(),
            signal: { get cancelled() { return worker.cancelled; } },
            onProgress
          });
          if (!this.cancelled) resolve(result);
        } catch (error) {
          reject(this.cancelled ? new Error("Clustering cancelled") : (error instanceof Error ? error : new Error(String(error))));
        } finally {
          this.running = false;
          this.pending = null;
        }
      });
    });
  }

  cancel(): void {
    this.cancelled = true;
    // Keep `running` true until the async orchestration observes cancellation
    // and finishes. This prevents a second build from racing the old one.
  }

  async terminate(): Promise<void> { this.cancel(); }
}

function yieldToEventLoop(): Promise<void> {
  if (typeof MessageChannel === "function") return new Promise((resolve) => { const channel = new MessageChannel(); channel.port1.onmessage = () => { channel.port1.close(); channel.port2.close(); resolve(); }; channel.port2.postMessage(undefined); });
  return new Promise((resolve) => setTimeout(resolve, 0));
}
