/* Dedicated Chromium worker. Transformers.js is bundled into this source by
 * build.mjs; no renderer-global or network-hosted runtime is accepted. */
import { env, pipeline } from "@huggingface/transformers";
// build.mjs aliases this specifier to the same ORT version bundled by
// Transformers.js. Keeping this import explicit avoids relying on the lazy
// env.backends.onnx compatibility property.
import * as ortWeb from "atomic-clusters-title-onnxruntime-web";
import { extractAssistantContent } from "./title-output";

type Message = { type: "INIT"; model: ArrayBuffer; tokenizer: ArrayBuffer; config: ArrayBuffer; generationConfig: ArrayBuffer; tokenizerConfig: ArrayBuffer; ortWasm: ArrayBuffer } | { type: "GENERATE"; id: number; prompts: string[]; maxNewTokens: number; signal?: boolean };
const scope = globalThis as typeof globalThis & { postMessage?: (value: unknown) => void; onmessage?: (event: MessageEvent<Message>) => void; fetch: typeof fetch };
let generator: any;
let generationQueue: Promise<void> = Promise.resolve();
const TITLE_SYSTEM_PROMPT = "You name knowledge clusters. Return only one useful, specific title. Never return an explanation, list, checkbox, markdown, URL, path, quotation, or label. Use 2-6 words and the requested input language.";
const assetFiles = new Map<string, ArrayBuffer>();
type OnnxRuntimeEnvironment = { wasm?: { wasmPaths?: string | Record<string, string>; wasmBinary?: ArrayBuffer; proxy?: boolean } };
type OnnxRuntimeModule = { env?: OnnxRuntimeEnvironment };
const ORT_SYMBOL = Symbol.for("onnxruntime");

/**
 * Qwen's instruct tokenizer uses ChatML.  The virtual local model used by
 * the title worker does not reliably expose tokenizer.chat_template to
 * Transformers.js, so passing a message array makes pipeline() fail with
 * "tokenizer.chat_template is not set".  Render the small, official Qwen
 * ChatML envelope ourselves and pass a plain string to text-generation.
 *
 * Notes are model input, not trusted prompt syntax.  Strip ChatML/control
 * markers before inserting them so a note cannot terminate the user turn or
 * manufacture an assistant turn.
 */
function stripChatMLDelimiters(value: string): string {
  return String(value || "").replace(/<\|[^>\r\n]{1,80}\|>/g, "");
}

function renderQwenChatML(userPrompt: string): string {
  const system = stripChatMLDelimiters(TITLE_SYSTEM_PROMPT);
  // Qwen3 supports the documented soft switch and its no-thinking template
  // also emits an empty think block before the assistant answer. Keep both
  // markers: this prevents the small generation budget from being spent on
  // hidden reasoning while remaining compatible with ONNX tokenizer assets.
  const user = stripChatMLDelimiters(userPrompt).trimEnd() + "\n/no_think";
  return `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n<think></think>\n`;
}

function getOnnxRuntime(): OnnxRuntimeModule {
  const globalRuntime = (globalThis as typeof globalThis & { [key: symbol]: OnnxRuntimeModule })[ORT_SYMBOL];
  return globalRuntime ?? (ortWeb as unknown as OnnxRuntimeModule);
}

function configureOnnxRuntime(ortWasm: ArrayBuffer): OnnxRuntimeEnvironment {
  const runtimeEnv = getOnnxRuntime().env;
  if (!runtimeEnv?.wasm) throw new Error("Bundled ONNX Runtime Web environment is unavailable.");
  runtimeEnv.wasm.wasmPaths = "https://atomic-clusters.local/ort/";
  runtimeEnv.wasm.wasmBinary = ortWasm;
  runtimeEnv.wasm.proxy = false;
  return runtimeEnv;
}

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
      // Transformers.js resolves its ONNX backend lazily. Seed the shared
      // runtime symbol before pipeline() so that the backend selects this
      // exact, configured local runtime instead of an unrelated host copy.
      const runtime = getOnnxRuntime();
      (globalThis as typeof globalThis & { [key: symbol]: OnnxRuntimeModule })[ORT_SYMBOL] = runtime;
      configureOnnxRuntime(event.data.ortWasm);
      generator = await pipeline("text-generation", "atomic-title", { device: "webgpu", dtype: "q4f16" });
      // Keep compatibility with bundles that expose a separate backend env
      // after lazy initialization. Inference still uses the shared runtime.
      const backendEnvironment = env.backends?.onnx as OnnxRuntimeEnvironment | undefined;
      if (backendEnvironment?.wasm && backendEnvironment !== runtime.env) {
        backendEnvironment.wasm.wasmPaths = "https://atomic-clusters.local/ort/";
        backendEnvironment.wasm.wasmBinary = event.data.ortWasm;
        backendEnvironment.wasm.proxy = false;
      }
      scope.postMessage?.({ type: "READY", backend: "webgpu" });
      return;
    }
    if (!generator) throw new Error("Title worker is not initialized.");
    const request = event.data as Extract<Message, { type: "GENERATE" }>;
    // WebGPU inference is stateful and can retain a sizeable activation set.
    // Starting several generations at once can deadlock Chromium's WebGPU
    // queue (and leaves the client waiting forever), so keep the public batch
    // API but issue exactly one generation at a time, including across
    // overlapping requests.
    const queuedGeneration = generationQueue.catch(() => undefined).then(async () => {
      const values: string[] = [];
      for (const prompt of request.prompts) {
        const output = await generator(renderQwenChatML(prompt), { max_new_tokens: request.maxNewTokens, do_sample: false, temperature: 0, repetition_penalty: 1.15, no_repeat_ngram_size: 3, return_full_text: false });
        values.push(extractAssistantContent(output));
      }
      scope.postMessage?.({ type: "RESULT", id: request.id, values });
    });
    generationQueue = queuedGeneration;
    await queuedGeneration;
  } catch (error) { scope.postMessage?.({ type: "ERROR", id: (event.data as any).id, message: error instanceof Error ? error.message : String(error) }); }
};
