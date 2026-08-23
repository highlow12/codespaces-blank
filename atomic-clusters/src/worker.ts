import { parentPort } from "worker_threads";
import "atomic-clusters-wasm-bootstrap";
import { clusterEmbeddings } from "./clustering";
import { WorkerRequest, WorkerResponse } from "./types";
import { loadWasmKernel } from "./wasm-loader";

if (!parentPort) throw new Error("Atomic Clusters worker requires worker_threads.");
let cancelled = false;
const wasmKernel = loadWasmKernel();
parentPort.on("message", async (request: WorkerRequest) => {
  if (request.type === "INIT") { parentPort!.postMessage({ type: "READY", version: 1 } satisfies WorkerResponse); return; }
  if (request.type === "CANCEL") { cancelled = true; return; }
  cancelled = false;
  try {
    const result = await clusterEmbeddings(request.ids, request.embeddings, request.config, { kernel: wasmKernel, signal: { get cancelled() { return cancelled; } }, onProgress: (phase, progress) => parentPort!.postMessage({ type: "PROGRESS", jobId: request.jobId, phase, progress } satisfies WorkerResponse) });
    parentPort!.postMessage({ type: "RESULT", jobId: request.jobId, result } satisfies WorkerResponse);
  } catch (error) {
    parentPort!.postMessage({ type: "ERROR", jobId: request.jobId, code: cancelled ? "CANCELLED" : "CLUSTER_FAILED", message: error instanceof Error ? error.message : String(error) } satisfies WorkerResponse);
  }
});
