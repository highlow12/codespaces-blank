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

type ProcessRestore = () => void;

function hideElectronProcess(): ProcessRestore {
  const target = globalThis as unknown as { process?: unknown };
  const hadOwnProcess = Object.prototype.hasOwnProperty.call(target, "process");
  const descriptor = Object.getOwnPropertyDescriptor(target, "process");
  const previous = target.process;

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

  return () => {
    try {
      if (descriptor) Object.defineProperty(target, "process", descriptor);
      else if (hadOwnProcess) target.process = previous;
      else delete (target as { [key: string]: unknown }).process;
    } catch {
      // Restoring is best effort. The worker is no longer usable if module
      // initialization failed, and leaving process hidden is safer than
      // changing the host's process object to an unexpected value.
    }
  };
}

void (async () => {
  let restoreProcess: ProcessRestore | undefined;
  try {
    restoreProcess = hideElectronProcess();
    // esbuild lowers this dynamic import into a deferred module initializer
    // for the inline IIFE, preserving the ordering above in the built worker.
    await import("./title-worker");
  } catch (error) {
    scope.postMessage?.({ type: "ERROR", message: error instanceof Error ? error.message : String(error) });
  } finally {
    restoreProcess?.();
  }
})();
