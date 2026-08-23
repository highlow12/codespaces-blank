import { normalizePath, Plugin, Vault } from "obsidian";
import { CachedEmbedding, ClusterResult, NoteRecord } from "./types";

interface CacheDocument { version: 1; embeddings: CachedEmbedding[]; }

export class EmbeddingCache {
  private readonly path = normalizePath(".obsidian/plugins/atomic-clusters/embedding-cache.json");
  constructor(private readonly vault: Vault) {}

  async load(): Promise<Map<string, CachedEmbedding>> {
    try {
      const raw = await this.vault.adapter.read(this.path);
      const document = JSON.parse(raw) as CacheDocument;
      return new Map((document.embeddings || []).map((item) => [this.key(item.path, item.provider, item.model), item]));
    } catch { return new Map(); }
  }

  async save(entries: Iterable<CachedEmbedding>): Promise<void> {
    const parent = ".obsidian/plugins/atomic-clusters";
    if (!(await this.vault.adapter.exists(parent))) await this.vault.adapter.mkdir(parent);
    await this.vault.adapter.write(this.path, JSON.stringify({ version: 1, embeddings: [...entries] } satisfies CacheDocument));
  }

  get(notes: NoteRecord, provider: string, model: string, map: Map<string, CachedEmbedding>): CachedEmbedding | undefined {
    const item = map.get(this.key(notes.path, provider, model));
    return item?.hash === notes.hash ? item : undefined;
  }

  private key(path: string, provider: string, model: string): string { return `${provider}:${model}:${path}`; }
}

export class NoteStore {
  constructor(private readonly vault: Vault) {}

  async collect(excludedFolders: string[] = []): Promise<NoteRecord[]> {
    const files = this.vault.getMarkdownFiles().filter((file) => !excludedFolders.some((folder) => file.path === folder || file.path.startsWith(`${folder}/`)));
    const notes: NoteRecord[] = [];
    for (const file of files) {
      const content = await this.vault.cachedRead(file);
      const { contentHash } = await import("./hash");
      notes.push({ path: file.path, title: file.basename, content, mtime: file.stat.mtime, hash: await contentHash(content) });
    }
    return notes;
  }
}

export class ClusterResultStore {
  private readonly path = normalizePath(".obsidian/plugins/atomic-clusters/cluster-result.json");
  constructor(private readonly vault: Vault) {}
  async load(): Promise<ClusterResult | null> { try { return JSON.parse(await this.vault.adapter.read(this.path)) as ClusterResult; } catch { return null; } }
  async save(result: ClusterResult): Promise<void> { const parent = ".obsidian/plugins/atomic-clusters"; if (!(await this.vault.adapter.exists(parent))) await this.vault.adapter.mkdir(parent); await this.vault.adapter.write(this.path, JSON.stringify(result)); }
}

export function pluginSettingsPath(plugin: Plugin): string {
  return normalizePath(`${plugin.manifest.dir}/settings.json`);
}
