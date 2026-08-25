/* Dedicated Chromium worker. Transformers.js is bundled into this source by
 * build.mjs; no renderer-global or network-hosted runtime is accepted. */
import { env, pipeline } from "@huggingface/transformers";

type Message = { type: "INIT"; model: ArrayBuffer; tokenizer: ArrayBuffer; config: ArrayBuffer; generationConfig: ArrayBuffer; tokenizerConfig: ArrayBuffer; ortWasm: ArrayBuffer } | { type: "GENERATE"; id: number; prompts: string[]; maxNewTokens: number; signal?: boolean };
const scope = globalThis as typeof globalThis & { postMessage?: (value: unknown) => void; onmessage?: (event: MessageEvent<Message>) => void; fetch: typeof fetch };
let generator: any;
const assetFiles = new Map<string, ArrayBuffer>();
scope.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
  const url = typeof Request !== "undefined" && input instanceof Request ? input.url : String(input); const file = [...assetFiles.keys()].find((name) => url.endsWith(`/${name}`) || url.endsWith(name));
  if (file) return new Response(assetFiles.get(file)!.slice(0), { status: 200, headers: { "content-type": file.endsWith(".json") ? "application/json" : "application/octet-stream" } });
  throw new Error(`Title worker blocked non-local asset request: ${url}`);
};

scope.onmessage = async (event: MessageEvent<Message>) => {
  try {
    if (event.data.type === "INIT") {
      assetFiles.clear();
      if (!(event.data.ortWasm instanceof ArrayBuffer) || event.data.ortWasm.byteLength === 0) throw new Error("Bundled ONNX WebGPU/WASM asset is unavailable; title runtime remains offline-only.");
      assetFiles.set("onnx/model_q4f16.onnx", event.data.model); assetFiles.set("tokenizer.json", event.data.tokenizer); assetFiles.set("config.json", event.data.config); assetFiles.set("generation_config.json", event.data.generationConfig); assetFiles.set("tokenizer_config.json", event.data.tokenizerConfig); assetFiles.set("ort-wasm-simd-threaded.jsep.wasm", event.data.ortWasm);
      // Transformers.js resolves every file through env.localModelPath. The
      // fetch shim maps that virtual path to the transferred, vault-local
      // buffers and rejects any unexpected remote request.
      env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = "https://atomic-clusters.local/models/";
      const onnxWasm = env.backends?.onnx?.wasm;
      if (!onnxWasm) throw new Error("Transformers.js ONNX WASM environment is unavailable.");
      onnxWasm.wasmPaths = "https://atomic-clusters.local/ort/";
      onnxWasm.wasmBinary = event.data.ortWasm;
      generator = await pipeline("text-generation", "atomic-title", { device: "webgpu", dtype: "q4f16" });
      scope.postMessage?.({ type: "READY", backend: "webgpu" });
      return;
    }
    if (!generator) throw new Error("Title worker is not initialized.");
    const request = event.data as Extract<Message, { type: "GENERATE" }>;
    const values = await Promise.all(request.prompts.map(async (prompt) => {
      const output = await generator(prompt, { max_new_tokens: request.maxNewTokens, do_sample: false, temperature: 0, return_full_text: false });
      const item = Array.isArray(output) ? output[0] : output;
      return typeof item === "string" ? item : String(item?.generated_text || item?.text || "");
    }));
    scope.postMessage?.({ type: "RESULT", id: request.id, values });
  } catch (error) { scope.postMessage?.({ type: "ERROR", id: (event.data as any).id, message: error instanceof Error ? error.message : String(error) }); }
};
