import { isAbsolute, join, resolve, sep, win32 } from "node:path";
import { pathToFileURL } from "node:url";

/** The shipped ORT asset is locked to the dependency version in package-lock.json. */
export const LOCAL_ORT_RENDERER_ASSET_VERSION = "1.20.1";

function countOccurrences(source: string, marker: string): number {
  return source.split(marker).length - 1;
}

/**
 * Make the pinned ORT WASM module safe for Electron's renderer. Electron
 * exposes process.versions.node even though this is not a Node worker; the
 * upstream module would otherwise import `module`/`worker_threads`. The
 * pinned module also constructs a relative WASM URL at evaluation time; a
 * blob: import has no hierarchical base, so that expression is replaced with
 * a harmless absolute data URL while wasmBinary supplies the real bytes.
 */
export function prepareLocalOrtRendererModule(source: string, wasmAsset = "ort-wasm-simd-threaded.wasm"): string {
  const legacyNodeMarker = 'B="object"==typeof process&&"object"==typeof process.versions&&"string"==typeof process.versions.node';
  const jsepNodeMarker = 'D="object"==typeof process&&"object"==typeof process.versions&&"string"==typeof process.versions.node';
  const modernNodeMarker = "var isNode = typeof globalThis.process?.versions?.node == 'string';";
  const nodeBranchMarker = "if(B){";
  const jsepBranchMarker = "if(D){";
  const modernBranchMarker = "if (isNode) isPthread = (await import('worker_threads')).workerData === 'em-pthread';";
  const relativeWasmUrlMarker = `(new URL("${wasmAsset}",import.meta.url)).href`;
  const nodeBaseUrlMarker = 'new URL("./",import.meta.url)';
  const workerModuleUrlMarker = "new URL(import.meta.url)";
  const legacyCount = countOccurrences(source, legacyNodeMarker);
  const branchCount = countOccurrences(source, nodeBranchMarker);
  const jsepCount = countOccurrences(source, jsepNodeMarker);
  const jsepBranchCount = countOccurrences(source, jsepBranchMarker);
  const hasKnownLegacyFormat = (legacyCount === 1 && branchCount === 3 && jsepCount === 0 && jsepBranchCount === 0) || (legacyCount === 0 && branchCount === 0 && jsepCount === 1 && jsepBranchCount === 3) || (legacyCount === 0 && branchCount === 0 && jsepCount === 0 && jsepBranchCount === 0);
  const nodeBaseUrlCount = countOccurrences(source, nodeBaseUrlMarker);
  const workerModuleUrlCount = countOccurrences(source, workerModuleUrlMarker);
  if (!hasKnownLegacyFormat || countOccurrences(source, modernNodeMarker) !== 1 || countOccurrences(source, modernBranchMarker) !== 1 || countOccurrences(source, relativeWasmUrlMarker) !== 1 || nodeBaseUrlCount > 1 || workerModuleUrlCount > 1) {
    throw new Error(`Unsupported ORT ${LOCAL_ORT_RENDERER_ASSET_VERSION} renderer asset format; refusing unsafe Node-branch transformation.`);
  }
  const transformed = source
    .replace(legacyNodeMarker, legacyCount ? "B=false" : legacyNodeMarker)
    .split(nodeBranchMarker).join(legacyCount ? "if(false){" : nodeBranchMarker)
    .replace(jsepNodeMarker, jsepCount ? "D=false" : jsepNodeMarker)
    .split(jsepBranchMarker).join(jsepCount ? "if(false){" : jsepBranchMarker)
    .replace(modernNodeMarker, "var isNode = false;")
    .replace(modernBranchMarker, "if (false) isPthread = (await import('worker_threads')).workerData === 'em-pthread';")
    .replace(relativeWasmUrlMarker, '"data:application/wasm;base64,"')
    .replace(nodeBaseUrlMarker, 'new URL("data:text/javascript,")')
    .replace(workerModuleUrlMarker, 'new URL("data:text/javascript,")');
  if (transformed.includes(nodeBranchMarker) || transformed.includes(jsepBranchMarker) || transformed.includes(modernNodeMarker) || transformed.includes(modernBranchMarker) || transformed.includes(relativeWasmUrlMarker) || transformed.includes(nodeBaseUrlMarker) || transformed.includes(workerModuleUrlMarker)) {
    throw new Error("ORT renderer asset Node-branch transformation did not apply completely.");
  }
  return transformed;
}

/** Resolve bundled ORT assets from the installed vault, never the eval loader's cwd. */
export function resolveLocalOrtAssetPrefix(basePath: string | undefined, manifestDir: string | undefined, pluginId = "atomic-clusters"): string {
  if (!basePath) throw new Error("Obsidian did not provide the vault base path.");
  const windowsPath = /^[A-Za-z]:[\\/]/.test(basePath);
  const pathApi = windowsPath ? win32 : undefined;
  const root = pathApi ? pathApi.resolve(basePath) : resolve(basePath);
  const fallback = pathApi ? pathApi.resolve(root, ".obsidian", "plugins", pluginId) : resolve(root, ".obsidian/plugins", pluginId);
  const rawManifestDir = manifestDir?.trim();
  const looksLikePath = !!rawManifestDir && (rawManifestDir.startsWith(".") || rawManifestDir.includes("/") || rawManifestDir.includes("\\"));
  const candidate = rawManifestDir
    ? (pathApi?.isAbsolute(rawManifestDir) || (!pathApi && isAbsolute(rawManifestDir))
      ? (pathApi ? pathApi.resolve(rawManifestDir) : resolve(rawManifestDir))
      : (pathApi ? pathApi.resolve(root, looksLikePath ? rawManifestDir : ".obsidian\\plugins", looksLikePath ? "" : rawManifestDir) : resolve(root, looksLikePath ? rawManifestDir : ".obsidian/plugins", looksLikePath ? "" : rawManifestDir)))
    : fallback;
  const insideVault = (target: string): boolean => {
    const comparableRoot = windowsPath ? root.toLowerCase() : root;
    const comparableTarget = windowsPath ? target.toLowerCase() : target;
    return comparableTarget === comparableRoot || comparableTarget.startsWith(`${comparableRoot}${pathApi?.sep || sep}`);
  };
  const pluginDir = insideVault(candidate) ? candidate : fallback;
  if (!insideVault(pluginDir)) throw new Error("Atomic Clusters plugin directory is outside the vault.");
  if (!windowsPath) return pathToFileURL(join(pluginDir, "/")).href;
  const normalized = pluginDir.replace(/\\/g, "/");
  const drive = normalized.slice(0, 2);
  const rest = normalized.slice(2).split("/").filter(Boolean).map((part) => encodeURIComponent(part)).join("/");
  return `file:///${drive}${rest ? `/${rest}` : ""}/`;
}
