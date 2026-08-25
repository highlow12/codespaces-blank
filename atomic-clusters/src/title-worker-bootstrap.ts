/*
 * This is intentionally a separate entry point from title-worker.ts.
 *
 * Transformers.js computes its environment (including IS_NODE_ENV and the
 * list of supported execution providers) while its modules are evaluated.
 * Obsidian's Chromium workers can expose Electron's `process` object, which
 * makes that detection choose the Node/DirectML branch before the worker has
 * a chance to configure ONNX Runtime Web. Keep the Transformers import in a
 * later dynamic import so the browser-shaped environment is in place first.
 */
import * as ortWeb from "atomic-clusters-title-onnxruntime-web";

type RuntimeGlobal = typeof globalThis & {
  [key: symbol]: unknown;
  process?: unknown;
  postMessage?: (value: unknown) => void;
};

const scope = globalThis as RuntimeGlobal;
// Keep the exact ORT WebGPU module in this worker bundle. Do not register it
// under Symbol.for("onnxruntime") here: Transformers.js uses that symbol as a
// complete-runtime override and skips its browser supported-device detection
// when it is present. title-worker.ts registers the runtime only after the
// Transformers backend has evaluated and selected WebGPU.
void ortWeb;

function hideElectronProcess(): void {
  const target = globalThis as unknown as { process?: unknown };
  const descriptor = Object.getOwnPropertyDescriptor(target, "process");

  try {
    if (descriptor?.configurable) {
      if ("value" in descriptor) Object.defineProperty(target, "process", { ...descriptor, value: undefined });
      else Object.defineProperty(target, "process", { configurable: true, enumerable: descriptor.enumerable, get: () => undefined, set: descriptor.set });
    } else if (descriptor?.writable || !descriptor) {
      target.process = undefined;
    }
    // A non-configurable, read-only process property cannot be safely hidden.
    // Refuse initialization instead of allowing Transformers.js to select a
    // CPU/DirectML backend and violating the WebGPU-only title contract.
    if (target.process !== undefined) throw new Error("Electron process global cannot be hidden from the title model worker.");
  } catch (error) {
    throw error instanceof Error ? error : new Error(String(error));
  }
}

void (async () => {
  try {
    // Keep Electron's process global hidden for the entire worker lifetime.
    // Transformers.js loads the ONNX backend lazily: restoring process after
    // this import would make a later pipeline/generator call detect Node and
    // try to import the unavailable `worker_threads` module.
    hideElectronProcess();
    // esbuild lowers this dynamic import into a deferred module initializer
    // for the inline IIFE, preserving the ordering above in the built worker.
    await import("./title-worker");
  } catch (error) {
    scope.postMessage?.({ type: "ERROR", message: error instanceof Error ? error.message : String(error) });
  }
})();
