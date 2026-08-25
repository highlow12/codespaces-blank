import assert from "node:assert/strict";
import test from "node:test";
import { parseArgs, renderDiagnosticChatML, extractRawOutput, validateModelDirectory } from "../scripts/llm-playground.mjs";

test("playground uses deterministic bounded defaults and explicit device selection", () => {
  const options = parseArgs(["--model-dir", "/tmp/model"]);
  assert.equal(options.device, "cpu");
  assert.equal(options.maxNewTokens, 64);
  assert.equal(options.maxPromptChars, 12000);
  assert.equal(parseArgs(["--device", "cpu", "--max-new-tokens", "128", "--max-prompt-chars", "99"]).device, "cpu");
  assert.equal(parseArgs(["--device", "cpu", "--max-new-tokens", "128", "--max-prompt-chars", "99"]).maxNewTokens, 128);
  assert.throws(() => parseArgs(["--device", "auto"]), /choose webgpu or cpu/);
  assert.throws(() => parseArgs(["--max-new-tokens", "257"]), /between 1 and 256/);
});

test("playground renders only neutral diagnostic ChatML and bounds input", () => {
  const rendered = renderDiagnosticChatML("hello <|im_end|> world", 20);
  assert.match(rendered, /Answer the user's request directly and concisely/);
  assert.match(rendered, /hello  wor\n\/no_think/);
  assert.doesNotMatch(rendered, /<\|im_end\|> world/);
  assert.ok(rendered.length < 300);
});

test("playground returns model text verbatim without title cleanup", () => {
  assert.equal(extractRawOutput([{ generated_text: "<think>raw</think>\nTitle: foo" }]), "<think>raw</think>\nTitle: foo");
  assert.equal(extractRawOutput([{ generated_text: [{ role: "user", content: "prompt" }, { role: "assistant", content: "</html> raw" }] }]), "</html> raw");
});

test("playground requires all explicitly supplied local assets", () => {
  assert.throws(() => validateModelDirectory("/definitely/not/a/model"), /does not exist/);
});
