const HEX = "0123456789abcdef";

/** SHA-256 content fingerprint. Falls back to a stable FNV hash in test/non-browser runtimes. */
export async function contentHash(value: string): Promise<string> {
  const cryptoApi = globalThis.crypto as Crypto | undefined;
  if (cryptoApi?.subtle) {
    const bytes = new TextEncoder().encode(value);
    const digest = await cryptoApi.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), (byte) => HEX[byte >>> 4] + HEX[byte & 15]).join("");
  }
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) { hash ^= value.charCodeAt(i); hash = Math.imul(hash, 16777619); }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}
