import "atomic-clusters-wasm-bootstrap";
import { clusterEmbeddings } from "./clustering";
import { WorkerRequest, WorkerResponse } from "./types";
import { loadWasmKernel } from "./wasm-loader";

let cancelled = false;
const scope = globalThis as typeof globalThis & { onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null; postMessage(message: WorkerResponse): void };
const wasmKernel = loadWasmKernel();

scope.onmessage = async (event) => {
  const request = event.data;
  if (request.type === "INIT") { scope.postMessage({ type: "READY", version: 1 } satisfies WorkerResponse); return; }
  if (request.type === "CANCEL") { cancelled = true; return; }
  cancelled = false;
  try {
    const result = await clusterEmbeddings(request.ids, request.embeddings, request.config, { kernel: wasmKernel, signal: { get cancelled() { return cancelled; } }, onProgress: (phase, progress) => scope.postMessage({ type: "PROGRESS", jobId: request.jobId, phase, progress } satisfies WorkerResponse) });
    scope.postMessage({ type: "RESULT", jobId: request.jobId, result } satisfies WorkerResponse);
  } catch (error) {
    scope.postMessage({ type: "ERROR", jobId: request.jobId, code: cancelled ? "CANCELLED" : "CLUSTER_FAILED", message: error instanceof Error ? error.message : String(error) } satisfies WorkerResponse);
  }
};
