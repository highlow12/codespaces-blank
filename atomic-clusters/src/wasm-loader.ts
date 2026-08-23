import { NumericKernel } from "./clustering";
import { WasmKernelModule, WasmNumericKernel } from "./wasm-kernel";

/**
 * The packaged build can assign the generated wasm-bindgen exports to this
 * hook before starting the worker. Keeping loading optional makes fixtures and
 * development builds deterministic without silently downloading anything.
 */
export function loadWasmKernel(): NumericKernel | undefined {
  const module = (globalThis as typeof globalThis & { __ATOMIC_CLUSTERS_WASM__?: WasmKernelModule }).__ATOMIC_CLUSTERS_WASM__;
  return module ? new WasmNumericKernel(module) : undefined;
}
