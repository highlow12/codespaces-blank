import { Worker } from "worker_threads";
import { ClusteringConfig, ClusterResult, WorkerRequest, WorkerResponse } from "./types";

export interface WorkerProgress { (phase: string, progress: number): void; }
export interface ClusteringWorker { init(): Promise<void>; run(ids: string[], embeddings: number[][], config: ClusteringConfig, onProgress?: WorkerProgress): Promise<ClusterResult>; cancel(): void; terminate(): Promise<void>; }

export class NodeClusteringWorker implements ClusteringWorker {
  private worker: Worker | null = null;
  private pending: { resolve: (result: ClusterResult) => void; reject: (error: Error) => void; jobId: string; onProgress?: WorkerProgress } | null = null;
  constructor(private readonly workerSource: string) {}

  async init(): Promise<void> {
    if (this.worker) return;
    this.worker = new Worker(this.workerSource, { eval: true });
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
