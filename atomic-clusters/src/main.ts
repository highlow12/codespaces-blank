import { App, Modal, Notice, Plugin } from "obsidian";
import { configureLocalOrtAssets, GeminiEmbeddingProvider, LocalEmbeddingProvider, LocalModelManager, SecretResolver, VaultLocalModelStorage } from "./embedding";
import { ClusterResultStore, EmbeddingCache, NoteStore } from "./storage";
import { AtomicClustersSettingTab } from "./settings";
import { ClusterExplorerView, VIEW_TYPE_CLUSTER_EXPLORER } from "./view";
import { ClusteringConfig, ClusterResult, PluginSettings } from "./types";
import { NodeClusteringWorker } from "./worker-client";
import { PyodideClusteringWorker } from "./pyodide-worker-client";
import workerSource from "./worker-source";
import { pathToFileURL } from "node:url";

const DEFAULT_SETTINGS: PluginSettings = {
  embeddingProvider: "gemini", geminiModel: "gemini-embedding-2", geminiSecretRef: "gemini-api-key",
  localModel: "multilingual-e5-small", excludedFolders: [], minClusterSize: 5, minSamples: 3,
  umapNeighbors: 15, umapMinDist: 0.1, pcaVarianceTarget: 0.9
  , clusteringRuntime: "wasm", pyodideUrl: ""
};

export default class AtomicClustersPlugin extends Plugin {
  settings!: PluginSettings;
  private worker: NodeClusteringWorker | PyodideClusteringWorker | null = null;
  private latestResult: ClusterResult | null = null;
  private running = false;
  private localModelManager!: LocalModelManager;

  async onload(): Promise<void> {
    // Obsidian desktop loads main.js as a CommonJS plugin module. Resolve the
    // ORT assets beside that bundle instead of relying on document cwd.
    configureLocalOrtAssets(pathToFileURL(`${__dirname}/`).href);
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData() as Partial<PluginSettings> || {});
    this.localModelManager = new LocalModelManager(new VaultLocalModelStorage(this.app.vault.adapter));
    this.registerView(VIEW_TYPE_CLUSTER_EXPLORER, (leaf) => new ClusterExplorerView(leaf));
    this.addCommand({ id: "build-note-clusters", name: "Build note clusters", callback: () => void this.buildClusters() });
    this.addCommand({ id: "open-cluster-explorer", name: "Open cluster explorer", callback: () => void this.openExplorer() });
    this.addCommand({ id: "cancel-clustering", name: "Cancel clustering", callback: () => this.cancelClustering() });
    this.addSettingTab(new AtomicClustersSettingTab(this.app, this, this.settings, () => this.saveSettings(), this.localModelManager));
  }

  async onunload(): Promise<void> { await this.worker?.terminate(); this.worker = null; }

  private async buildClusters(): Promise<void> {
    if (this.running) { new Notice("Atomic Clusters is already running."); return; }
    this.running = true;
    try {
      const notes = await new NoteStore(this.app.vault).collect(this.settings.excludedFolders);
      if (!notes.length) throw new Error("No Markdown notes found in the selected vault folders.");
      new Notice(`Preparing embeddings for ${notes.length} notes…`);
      const cache = new EmbeddingCache(this.app.vault); const entries = await cache.load();
      const provider = this.settings.embeddingProvider === "gemini"
        ? new GeminiEmbeddingProvider(this.settings, this.secretResolver(), (count) => this.confirmGeminiTransmission(count))
        : new LocalEmbeddingProvider(this.settings, undefined, this.localModelManager);
      const fresh = notes.filter((note) => !cache.get(note, provider.id, provider.model, entries));
      const embedded = fresh.length ? await provider.embed(fresh, (done, total) => new Notice(`Embedding ${done}/${total}`)) : [];
      embedded.forEach((item) => entries.set(`${item.provider}:${item.model}:${item.path}`, item)); await cache.save(entries.values());
      const vectors = notes.map((note) => cache.get(note, provider.id, provider.model, entries)?.vector);
      if (vectors.some((vector) => !vector)) throw new Error("Embedding cache is incomplete; please run the build again.");
      const ids = notes.map((note) => note.path);
      const worker = await this.getWorker();
      const config: ClusteringConfig = { minClusterSize: this.settings.minClusterSize, minSamples: this.settings.minSamples, umapNeighbors: this.settings.umapNeighbors, umapMinDist: this.settings.umapMinDist, pcaVarianceTarget: this.settings.pcaVarianceTarget, seed: 42 };
      this.latestResult = await worker.run(ids, vectors as number[][], config, (phase, progress) => this.updateProgress(phase, progress));
      await new ClusterResultStore(this.app.vault).save(this.latestResult); await this.publishResult(this.latestResult); new Notice(`Built ${this.latestResult.hierarchy.leaves.length} clusters.`);
    } catch (error) { if (!(error instanceof Error && error.message === "Clustering cancelled")) new Notice(`Atomic Clusters failed: ${error instanceof Error ? error.message : String(error)}`); }
    finally { this.running = false; }
  }

  private async getWorker(): Promise<NodeClusteringWorker | PyodideClusteringWorker> {
    if (!this.worker) {
      if (this.settings.clusteringRuntime === "pyodide") this.worker = new PyodideClusteringWorker({ pyodideUrl: this.settings.pyodideUrl || undefined });
      else this.worker = new NodeClusteringWorker(workerSource);
      await this.worker.init();
    }
    return this.worker;
  }
  private cancelClustering(): void { if (!this.running) { new Notice("No clustering job is running."); return; } this.worker?.cancel(); new Notice("Clustering cancellation requested."); }
  private async openExplorer(): Promise<void> { const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER); const leaf = leaves[0] || this.app.workspace.getRightLeaf(false); if (!leaf) return; await leaf.setViewState({ type: VIEW_TYPE_CLUSTER_EXPLORER, active: true }); this.app.workspace.revealLeaf(leaf); if (!this.latestResult) this.latestResult = await new ClusterResultStore(this.app.vault).load(); if (this.latestResult) (leaf.view as ClusterExplorerView).setResult(this.latestResult); }
  private async publishResult(result: ClusterResult): Promise<void> { for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER)) (leaf.view as ClusterExplorerView).setResult(result); }
  private updateProgress(phase: string, progress: number): void { this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER).forEach((leaf) => { const view = leaf.view as ClusterExplorerView; view.contentEl.setAttribute("aria-label", `${phase} ${Math.round(progress * 100)}%`); view.setProgress(phase, progress); }); }
  private async saveSettings(): Promise<void> { await this.saveData(this.settings); }
  private secretResolver(): SecretResolver { const storage = (this.app as unknown as { secretStorage?: { getSecret?: (reference: string) => string | null } }).secretStorage; return { getSecret: async (reference) => storage?.getSecret?.(reference) || null }; }
  private confirmGeminiTransmission(count: number): Promise<boolean> { return new Promise((resolve) => new GeminiTransmissionModal(this.app, count, resolve).open()); }
}

class GeminiTransmissionModal extends Modal {
  private settled = false;
  constructor(app: App, private readonly count: number, private readonly resolveChoice: (value: boolean) => void) { super(app); }
  onOpen(): void {
    this.contentEl.createEl("h2", { text: "Send note text to Gemini?" });
    this.contentEl.createEl("p", { text: `Atomic Clusters will send the content of ${this.count} Markdown note${this.count === 1 ? "" : "s"} to Google Gemini to create embeddings. This is not an offline operation.` });
    const buttons = this.contentEl.createDiv({ cls: "modal-button-container" });
    buttons.createEl("button", { text: "Cancel" }).addEventListener("click", () => this.finish(false));
    buttons.createEl("button", { text: "Send note text", cls: "mod-cta" }).addEventListener("click", () => this.finish(true));
  }
  onClose(): void { this.finish(false); }
  private finish(value: boolean): void { if (this.settled) return; this.settled = true; this.resolveChoice(value); this.close(); }
}
