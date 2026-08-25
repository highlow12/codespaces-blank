/** Remove Qwen3 reasoning blocks before title validation/sanitization. */
export function stripThinkingContent(value: unknown): string {
  let text = String(value ?? "");
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, " ");
  // A truncated generation can leave an unterminated reasoning block. Never
  // expose its internal reasoning as a cluster title.
  text = text.replace(/<think>[\s\S]*$/gi, " ");
  return text.trim();
}

/** Extract the final assistant message from Transformers.js text-generation output. */
export function extractAssistantContent(output: unknown): string {
  const item = Array.isArray(output) ? output[0] : output;
  const generated = (item && typeof item === "object" ? (item as { generated_text?: unknown; text?: unknown }).generated_text ?? (item as { text?: unknown }).text : undefined) ?? item;
  if (Array.isArray(generated)) {
    const messages = generated as Array<{ role?: string; content?: unknown; text?: unknown }>;
    const assistant = [...messages].reverse().find((message) => message && message.role === "assistant") || [...messages].reverse().find((message) => message && typeof message.content === "string");
    // Preserve control/tag output for the title validator. Stripping a think
    // block here would turn polluted model output into a false success and
    // could cache the text that followed it as if it were intentional.
    return assistant ? String(assistant.content ?? assistant.text ?? "") : "";
  }
  if (generated && typeof generated === "object") {
    const message = generated as { content?: unknown; text?: unknown; generated_text?: unknown };
    return String(message.content ?? message.text ?? message.generated_text ?? "");
  }
  return typeof generated === "string" ? generated : String(generated ?? "");
}
