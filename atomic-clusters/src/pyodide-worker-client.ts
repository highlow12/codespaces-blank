import { ClusteringConfig, ClusterResult, WorkerResponse } from "./types";
import workerSource from "./pyodide-worker-source";

export interface PyodideWorkerOptions {
  pyodideUrl?: string;
  indexURL?: string;
  workerFactory?: (source: string) => PyodideWorkerLike;
}
export interface PyodideWorkerLike {
  postMessage(message: unknown): void;
  terminate(): void | Promise<void>;
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
}
export interface PyodideProgress { (phase: string, progress: number): void; }

function browserWorker(source: string): PyodideWorkerLike {
  if (typeof Worker === "undefined" || typeof Blob === "undefined" || typeof URL === "undefined") throw new Error("Pyodide clustering requires a browser Worker runtime.");
  const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  const worker = new Worker(url);
  return {
    postMessage: (message) => worker.postMessage(message), terminate: () => { worker.terminate(); URL.revokeObjectURL(url); },
    get onmessage() { return worker.onmessage as ((event: MessageEvent<WorkerResponse>) => void) | null; },
    set onmessage(value) { worker.onmessage = value; },
    get onerror() { return worker.onerror as ((event: ErrorEvent) => void) | null; },
    set onerror(value) { worker.onerror = value; }
  };
}

export class PyodideClusteringWorker {
  private worker: PyodideWorkerLike | null = null;
  private pending: { resolve: (result: ClusterResult) => void; reject: (error: Error) => void; jobId: string; onProgress?: PyodideProgress } | null = null;
  constructor(private readonly options: PyodideWorkerOptions = {}) {}

  async init(): Promise<void> {
    if (this.worker) return;
    this.worker = (this.options.workerFactory || browserWorker)(workerSource);
    this.worker.onmessage = (event) => this.onMessage(event.data);
    this.worker.onerror = (event) => { this.pending?.reject(new Error(event.message || "Pyodide worker failed.")); this.pending = null; };
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Pyodide worker initialization timed out.")), 30000);
      const onReady = (response: WorkerResponse) => { if (response.type !== "READY") return; clearTimeout(timer); this.ready = undefined; resolve(); };
      this.ready = onReady;
      this.worker!.postMessage({ type: "INIT", version: 1, pyodideUrl: this.options.pyodideUrl, indexURL: this.options.indexURL });
    });
  }

  run(ids: string[], embeddings: number[][], config: ClusteringConfig = {}, onProgress?: PyodideProgress): Promise<ClusterResult> {
    if (!this.worker) return Promise.reject(new Error("Pyodide worker is not initialized."));
    if (this.pending) return Promise.reject(new Error("A clustering job is already running."));
    const jobId = `pyodide-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    return new Promise((resolve, reject) => { this.pending = { resolve, reject, jobId, onProgress }; this.worker!.postMessage({ type: "CLUSTER", jobId, ids, embeddings, config }); });
  }

  cancel(): void { if (!this.pending || !this.worker) return; this.worker.postMessage({ type: "CANCEL", jobId: this.pending.jobId }); this.pending.reject(new Error("Clustering cancelled")); this.pending = null; }
  async terminate(): Promise<void> { const worker = this.worker; this.worker = null; this.pending = null; if (worker) await worker.terminate(); }
  private ready: ((response: WorkerResponse) => void) | undefined;
  private onMessage(response: WorkerResponse): void {
    if (response.type === "READY") { this.ready?.(response); return; }
    if (!this.pending || response.jobId !== this.pending.jobId) return;
    if (response.type === "PROGRESS") this.pending.onProgress?.(response.phase, response.progress);
    else if (response.type === "RESULT") { this.pending.resolve(response.result); this.pending = null; }
    else if (response.type === "ERROR") { this.pending.reject(new Error(response.message)); this.pending = null; }
  }
}
