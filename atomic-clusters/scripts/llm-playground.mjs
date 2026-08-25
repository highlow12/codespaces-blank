#!/usr/bin/env node

/*
 * Standalone Qwen3 diagnostic runner.
 *
 * This intentionally lives outside the Obsidian plugin entry point. It does
 * not import cluster storage, title generation, or any vault code. The only
 * persistent input is the model directory supplied by the caller; prompt
 * text and model output stay in memory and stdout/stderr.
 */
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import { stdin, stdout, stderr } from "node:process";

export const DEFAULT_MAX_NEW_TOKENS = 64;
export const MAX_MAX_NEW_TOKENS = 256;
export const DEFAULT_MAX_PROMPT_CHARS = 12_000;
export const MODEL_FILES = ["model_q4f16.onnx", "tokenizer.json", "config.json", "generation_config.json", "tokenizer_config.json"];

export function usage() {
  return `Usage:
  npm run llm:playground -- --model-dir <directory> --prompt "<prompt>"
  cat prompt.txt | npm run llm:playground -- --model-dir <directory>

Options:
  --model-dir <path>       Qwen3 revision directory containing the five model assets
  --prompt <text>          Prompt; when omitted, read UTF-8 text from stdin
  --device <cpu|webgpu>    Backend to request (default: cpu; no implicit fallback)
  --max-new-tokens <n>     Deterministic generation limit (default: 64, maximum: 256)
  --max-prompt-chars <n>   Input bound (default: 12000)
  --help                   Show this help
`;
}

export function parseArgs(argv) {
  const options = { device: "cpu", maxNewTokens: DEFAULT_MAX_NEW_TOKENS, maxPromptChars: DEFAULT_MAX_PROMPT_CHARS };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") return { help: true, ...options };
    if (!argument.startsWith("--")) throw new Error(`Unknown argument: ${argument}`);
    const name = argument.slice(2);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`Missing value for --${name}`);
    index += 1;
    if (name === "model-dir") options.modelDir = value;
    else if (name === "prompt") options.prompt = value;
    else if (name === "device") options.device = value;
    else if (name === "max-new-tokens") options.maxNewTokens = parseBoundedInteger(value, 1, MAX_MAX_NEW_TOKENS, "max-new-tokens");
    else if (name === "max-prompt-chars") options.maxPromptChars = parseBoundedInteger(value, 1, 100_000, "max-prompt-chars");
    else throw new Error(`Unknown option: --${name}`);
  }
  if (options.device !== "webgpu" && options.device !== "cpu") throw new Error(`Unsupported device "${options.device}"; choose webgpu or cpu.`);
  if (options.modelDir) options.modelDir = resolve(options.modelDir);
  return options;
}

function parseBoundedInteger(value, minimum, maximum, label) {
  if (!/^\d+$/.test(value)) throw new Error(`--${label} must be an integer.`);
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) throw new Error(`--${label} must be between ${minimum} and ${maximum}.`);
  return parsed;
}

export function validateModelDirectory(modelDir) {
  if (!modelDir) throw new Error("--model-dir is required; no remote model lookup is performed.");
  const directory = resolve(modelDir);
  if (!existsSync(directory)) throw new Error(`Model directory does not exist: ${directory}`);
  const missing = MODEL_FILES.filter((file) => !existsSync(resolve(directory, file)));
  if (missing.length) throw new Error(`Model directory is missing: ${missing.join(", ")}`);
  return directory;
}

export function renderDiagnosticChatML(prompt, maxPromptChars = DEFAULT_MAX_PROMPT_CHARS) {
  const bounded = String(prompt ?? "").slice(0, maxPromptChars);
  if (!bounded.trim()) throw new Error("Prompt is empty.");
  // Keep the diagnostic envelope equivalent to the plugin's neutral mode, but
  // never treat user text as ChatML. This is not title prompting or cleanup.
  const user = bounded.replace(/<\|[^>\r\n]{1,80}\|>/g, "").trimEnd() + "\n/no_think";
  return `<|im_start|>system\nAnswer the user's request directly and concisely.<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n`;
}

export function extractRawOutput(result) {
  const item = Array.isArray(result) ? result[0] : result;
  const generated = item && typeof item === "object" && "generated_text" in item ? item.generated_text : item;
  if (Array.isArray(generated)) {
    const assistant = [...generated].reverse().find((message) => message && typeof message === "object" && message.role === "assistant");
    return String(assistant?.content ?? generated.at(-1)?.content ?? "");
  }
  return String(generated ?? "");
}

export async function readPrompt(options) {
  if (options.prompt !== undefined) return options.prompt;
  if (stdin.isTTY) throw new Error("No --prompt was provided and stdin is a TTY; pipe a prompt or pass --prompt.");
  return readFile(0, "utf8");
}

export async function runDiagnostic(options) {
  const modelDir = validateModelDirectory(options.modelDir);
  const prompt = await readPrompt(options);
  const input = renderDiagnosticChatML(prompt, options.maxPromptChars);
  const started = performance.now();
  let generator;
  try {
    const { env, pipeline } = await import("@huggingface/transformers");
    // No hub access and no Transformers.js filesystem cache: this command is
    // a pure diagnostic of the explicitly supplied local model directory.
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.useFSCache = false;
    env.useBrowserCache = false;
    env.localModelPath = modelDir;
    generator = await pipeline("text-generation", modelDir, {
      device: options.device,
      dtype: "q4f16",
      subfolder: "",
      model_file_name: "model",
      local_files_only: true,
    });
    const output = await generator(input, {
      max_new_tokens: options.maxNewTokens,
      do_sample: false,
      temperature: 0,
      repetition_penalty: 1.15,
      no_repeat_ngram_size: 3,
      return_full_text: false,
    });
    return { backend: options.device, durationMs: Math.max(0, Math.round(performance.now() - started)), output: extractRawOutput(output), promptChars: String(prompt).length, inputChars: input.length };
  } finally {
    // Transformers.js exposes dispose on the model; avoid retaining a large
    // ONNX session when this one-shot command is embedded by another runner.
    await generator?.dispose?.().catch(() => undefined);
  }
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  if (options.help) { stdout.write(usage()); return 0; }
  const result = await runDiagnostic(options);
  stdout.write(`backend: ${result.backend}\nduration_ms: ${result.durationMs}\nprompt_chars: ${result.promptChars}\ninput_chars: ${result.inputChars}\n--- raw output ---\n`);
  stdout.write(result.output);
  if (!result.output.endsWith("\n")) stdout.write("\n");
  stdout.write("--- end raw output ---\n");
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => { stderr.write(`llm-playground error: ${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
}
