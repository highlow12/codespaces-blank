/** Extract the final assistant message from Transformers.js text-generation output. */
export function extractAssistantContent(output: unknown): string {
  const item = Array.isArray(output) ? output[0] : output;
  const generated = (item && typeof item === "object" ? (item as { generated_text?: unknown; text?: unknown }).generated_text ?? (item as { text?: unknown }).text : undefined) ?? item;
  if (Array.isArray(generated)) {
    const messages = generated as Array<{ role?: string; content?: unknown; text?: unknown }>;
    const assistant = [...messages].reverse().find((message) => message && message.role === "assistant") || [...messages].reverse().find((message) => message && typeof message.content === "string");
    return assistant ? String(assistant.content ?? assistant.text ?? "") : "";
  }
  if (generated && typeof generated === "object") {
    const message = generated as { content?: unknown; text?: unknown; generated_text?: unknown };
    return String(message.content ?? message.text ?? message.generated_text ?? "");
  }
  return typeof generated === "string" ? generated : String(generated ?? "");
}
