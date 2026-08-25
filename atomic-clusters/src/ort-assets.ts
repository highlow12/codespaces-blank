import { isAbsolute, join, resolve, sep, win32 } from "node:path";
import { pathToFileURL } from "node:url";

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
