import { App, Modal, Notice, Plugin } from "obsidian";
import { configureLocalOrtAssets, disposeLocalOrtAssets, GeminiEmbeddingProvider, LocalEmbeddingProvider, LocalModelManager, LocalRuntimeProgress, LOCAL_ORT_MJS_ASSET, LOCAL_ORT_WASM_ASSET, SecretResolver, VaultLocalModelStorage } from "./embedding";
import { ClusterResultStore, EmbeddingCache, EmbeddingLogStore, NoteStore } from "./storage";
import { AtomicClustersSettingTab, ClusterRunControls, LocalRuntimeTest } from "./settings";
import { ClusterExplorerView, VIEW_TYPE_CLUSTER_EXPLORER } from "./view";
import { ClusteringConfig, ClusterResult, EmbeddingLogEntry, EmbeddingRunLog, PluginSettings } from "./types";
import { NodeClusteringWorker } from "./worker-client";
import { PyodideClusteringWorker } from "./pyodide-worker-client";
import workerSource from "./worker-source";
import { isAbsolute, resolve, sep } from "node:path";
import { shell } from "electron";
import { AtomicClustersProgress } from "./progress";
import { prepareLocalOrtRendererModule, resolveLocalOrtAssetPrefix } from "./ort-assets";

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
  private operationProgress: AtomicClustersProgress | null = null;
  private runAbortController: AbortController | null = null;

  async onload(): Promise<void> {
    // Obsidian may eval-load the plugin bundle, so loader-relative paths can
    // point into electron.asar. Resolve assets from the actual vault instead.
    const adapter = this.app.vault.adapter as typeof this.app.vault.adapter & { getBasePath?: () => string };
    try {
      const prefix = resolveLocalOrtAssetPrefix(adapter.getBasePath?.(), this.manifest.dir, this.manifest.id);
      configureLocalOrtAssets(prefix);
      await this.configureRendererOrtAssets(adapter, prefix);
    } catch { /* Local inference reports a useful error if this cannot be resolved. */ }
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData() as Partial<PluginSettings> || {});
    this.localModelManager = new LocalModelManager(new VaultLocalModelStorage(this.app.vault.adapter));
    this.registerView(VIEW_TYPE_CLUSTER_EXPLORER, (leaf) => new ClusterExplorerView(leaf));
    this.addCommand({ id: "build-note-clusters", name: "Build note clusters", callback: () => void this.buildClusters() });
    this.addCommand({ id: "open-cluster-explorer", name: "Open cluster explorer", callback: () => void this.openExplorer() });
    this.addCommand({ id: "open-embedding-log", name: "Open embedding log", callback: () => void this.openEmbeddingLog() });
    this.addCommand({ id: "cancel-clustering", name: "Cancel clustering", callback: () => this.cancelClustering() });
    const clusterRun: ClusterRunControls = { build: () => this.buildClusters(), cancel: () => this.cancelClustering(), isRunning: () => this.running };
    const testLocalRuntime: LocalRuntimeTest = (onProgress) => this.testLocalRuntime(onProgress);
    this.addSettingTab(new AtomicClustersSettingTab(this.app, this, this.settings, () => this.saveSettings(), this.localModelManager, () => this.openEmbeddingLog(), clusterRun, testLocalRuntime));
  }

  async onunload(): Promise<void> { await this.worker?.terminate(); this.worker = null; disposeLocalOrtAssets(); }

  private async configureRendererOrtAssets(adapter: { exists(path: string): Promise<boolean>; readBinary(path: string): Promise<ArrayBuffer> }, prefix: string): Promise<void> {
    const manifestDir = this.manifest.dir?.replace(/\\/g, "/").replace(/\/$/, "");
    const pluginDir = manifestDir?.startsWith(".obsidian/plugins/") ? manifestDir : `.obsidian/plugins/${this.manifest.id}`;
    const mjsPath = `${pluginDir}/${LOCAL_ORT_MJS_ASSET}`;
    const wasmPath = `${pluginDir}/${LOCAL_ORT_WASM_ASSET}`;
    if (!(await adapter.exists(mjsPath)) || !(await adapter.exists(wasmPath))) return;
    if (typeof Blob === "undefined" || typeof URL.createObjectURL !== "function") return;
    const [mjsBytes, wasmBinary] = await Promise.all([adapter.readBinary(mjsPath), adapter.readBinary(wasmPath)]);
    const mjsSource = prepareLocalOrtRendererModule(new TextDecoder().decode(mjsBytes));
    const mjsUrl = URL.createObjectURL(new Blob([mjsSource], { type: "text/javascript" }));
    configureLocalOrtAssets(prefix, { mjs: mjsUrl, wasmBinary, revoke: () => URL.revokeObjectURL(mjsUrl) });
  }

  private async buildClusters(): Promise<void> {
    if (this.running) { new Notice("Atomic Clusters is already running."); return; }
    this.running = true;
    this.runAbortController = new AbortController();
    const runSignal = this.runAbortController.signal;
    const progress = new AtomicClustersProgress("Atomic Clusters");
    this.operationProgress = progress;
    const startedAt = new Date().toISOString(); const logEntries: EmbeddingLogEntry[] = []; let persistedRunLog: EmbeddingRunLog | null = null; let runTotal = 0; let runStage: EmbeddingRunLog["stage"] = "embedding";
    const logStore = new EmbeddingLogStore(this.app.vault);
    const counts = () => ({ succeeded: logEntries.filter((entry) => entry.status === "success").length, failed: logEntries.filter((entry) => entry.status === "failure").length, cached: logEntries.filter((entry) => entry.status === "cached").length });
    let runProvider = this.settings.embeddingProvider; let runModel = this.settings.embeddingProvider === "gemini" ? this.settings.geminiModel : `${this.settings.localModel}@2024-05-01`;
    try {
      progress.update({ phase: "cache scan", progress: 0.02, detail: "Scanning Markdown notes" });
      const notes = await new NoteStore(this.app.vault).collect(this.settings.excludedFolders);
      if (!notes.length) throw new Error("No Markdown notes found in the selected vault folders.");
      runTotal = notes.length;
      progress.update({ phase: "cache scan", progress: 0.1, detail: `Scanned ${notes.length} notes` });
      const cache = new EmbeddingCache(this.app.vault); const entries = await cache.load();
      const provider = this.settings.embeddingProvider === "gemini"
        ? new GeminiEmbeddingProvider(this.settings, this.secretResolver(), (count) => this.confirmGeminiTransmission(count))
        : new LocalEmbeddingProvider(this.settings, undefined, this.localModelManager);
      runProvider = provider.id; runModel = provider.model;
      const fresh = notes.filter((note) => !cache.get(note, provider.id, provider.model, entries));
      notes.filter((note) => !fresh.includes(note)).forEach((note) => logEntries.push({ path: note.path, timestamp: new Date().toISOString(), provider: provider.id, model: provider.model, status: "cached", durationMs: 0 }));
      progress.update({ phase: "cache scan", progress: 0.15, detail: `${fresh.length} notes need embeddings · ${notes.length - fresh.length} cached` });
      if (provider.id === "local" && fresh.length) {
        runStage = "preflight";
        await (provider as LocalEmbeddingProvider).preflight((update) => progress.update({ phase: "preflight", progress: 0.15 + update.progress * 0.1, detail: update.detail || update.phase }), runSignal);
        runStage = "embedding";
      }
      let processedFresh = 0;
      const embedded = fresh.length ? await provider.embed(fresh, (done, total) => progress.update({ phase: "embedding", progress: 0.25 + (total ? done / total * 0.55 : 0), detail: `${done}/${total} notes processed` }), (entry) => { logEntries.push(entry); processedFresh++; progress.update({ phase: "embedding", progress: 0.25 + (processedFresh / Math.max(1, fresh.length)) * 0.55, detail: `${entry.status === "failure" ? "Failed" : "Embedded"}: ${entry.path}` }); }, runSignal) : [];
      embedded.forEach((item) => entries.set(`${item.provider}:${item.model}:${item.path}`, item)); await cache.save(entries.values());
      const activeNotes = notes.filter((note) => cache.get(note, provider.id, provider.model, entries));
      const freshEmbeddingFailed = fresh.length > 0 && embedded.length === 0;
      const embeddingError = freshEmbeddingFailed
        ? "All notes requiring fresh embeddings failed; fix the embedding provider or model and run the build again."
        : !activeNotes.length ? "No notes have usable embeddings; fix the failed notes and run the build again." : undefined;
      const embeddingCounts = counts();
      persistedRunLog = { version: 1, startedAt, completedAt: new Date().toISOString(), provider: provider.id, model: provider.model, total: notes.length, ...embeddingCounts, entries: logEntries, status: embeddingError ? "failed" : "completed", stage: "embedding", ...(embeddingError ? { error: safeRunError(embeddingError) } : {}) };
      await logStore.save(persistedRunLog);
      progress.update({ phase: "cache save", progress: 0.82, detail: `${persistedRunLog.succeeded} embedded · ${persistedRunLog.failed} failed · ${persistedRunLog.cached} cached` });
      if (embeddingError) throw new Error(embeddingError);
      // The persisted log currently describes a completed embedding phase;
      // remember that subsequent failures belong to clustering.
      persistedRunLog = { ...persistedRunLog, stage: "clustering" };
      const vectors = activeNotes.map((note) => cache.get(note, provider.id, provider.model, entries)?.vector);
      if (vectors.some((vector) => !vector)) throw new Error("Embedding cache is incomplete; please run the build again.");
      const ids = activeNotes.map((note) => note.path);
      const worker = await this.getWorker();
      progress.update({ phase: "clustering", progress: 0.84, detail: `Clustering ${activeNotes.length} notes` });
      const config: ClusteringConfig = { minClusterSize: this.settings.minClusterSize, minSamples: this.settings.minSamples, umapNeighbors: this.settings.umapNeighbors, umapMinDist: this.settings.umapMinDist, pcaVarianceTarget: this.settings.pcaVarianceTarget, seed: 42 };
      this.latestResult = await worker.run(ids, vectors as number[][], config, (phase, value) => { this.updateProgress(phase, value); progress.update({ phase, progress: 0.84 + value * 0.15, detail: `Clustering ${Math.round(value * 100)}%` }); });
      await new ClusterResultStore(this.app.vault).save(this.latestResult); await this.publishResult(this.latestResult); progress.complete(`Built ${this.latestResult.hierarchy.leaves.length} clusters`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const cancelled = message === "Clustering cancelled" || message.includes("cancelled");
      progress.fail(cancelled ? "Clustering cancelled" : `Build failed: ${message}`);
      const status = cancelled ? "cancelled" : "failed";
      const failedLog: EmbeddingRunLog = persistedRunLog
        ? { ...persistedRunLog, completedAt: new Date().toISOString(), status, error: safeRunError(error) }
        : { version: 1, startedAt, completedAt: new Date().toISOString(), provider: runProvider, model: runModel, total: runTotal, ...counts(), entries: logEntries, status, stage: runStage, error: safeRunError(error) };
      await logStore.save(failedLog).catch(() => undefined);
      if (!cancelled) new Notice(`Atomic Clusters failed: ${safeRunError(error)}`);
    }
    finally { this.running = false; this.operationProgress = null; this.runAbortController = null; }
  }

  private async testLocalRuntime(onProgress: (progress: LocalRuntimeProgress) => void): Promise<void> {
    const startedAt = new Date().toISOString();
    const provider = new LocalEmbeddingProvider(this.settings, undefined, this.localModelManager);
    const store = new EmbeddingLogStore(this.app.vault);
    try {
      await provider.preflight(onProgress);
      await store.save({ version: 1, startedAt, completedAt: new Date().toISOString(), provider: provider.id, model: provider.model, total: 0, succeeded: 0, failed: 0, cached: 0, entries: [], status: "completed", stage: "preflight" });
    } catch (error) {
      await store.save({ version: 1, startedAt, completedAt: new Date().toISOString(), provider: provider.id, model: provider.model, total: 0, succeeded: 0, failed: 0, cached: 0, entries: [], status: "failed", stage: "preflight", error: safeRunError(error) }).catch(() => undefined);
      throw error;
    }
  }

  private async getWorker(): Promise<NodeClusteringWorker | PyodideClusteringWorker> {
    if (!this.worker) {
      if (this.settings.clusteringRuntime === "pyodide") this.worker = new PyodideClusteringWorker({ pyodideUrl: this.settings.pyodideUrl || undefined });
      else this.worker = new NodeClusteringWorker(workerSource);
      await this.worker.init();
    }
    return this.worker;
  }
  private cancelClustering(): void { if (!this.running) { new Notice("No clustering job is running."); return; } this.runAbortController?.abort(); this.operationProgress?.fail("Cancellation requested"); this.worker?.cancel(); new Notice("Clustering cancellation requested."); }
  private async openExplorer(): Promise<void> { const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER); const leaf = leaves[0] || this.app.workspace.getRightLeaf(false); if (!leaf) return; await leaf.setViewState({ type: VIEW_TYPE_CLUSTER_EXPLORER, active: true }); this.app.workspace.revealLeaf(leaf); if (!this.latestResult) this.latestResult = await new ClusterResultStore(this.app.vault).load(); if (this.latestResult) (leaf.view as ClusterExplorerView).setResult(this.latestResult); }
  private async openEmbeddingLog(): Promise<void> {
    const relativePath = ".obsidian/plugins/atomic-clusters/embedding-log.json";
    try {
      const adapter = this.app.vault.adapter as typeof this.app.vault.adapter & { getBasePath?: () => string };
      const basePath = adapter.getBasePath?.();
      if (!basePath || !isAbsolute(basePath)) { new Notice("Cannot determine the absolute vault path."); return; }
      const vaultRoot = resolve(basePath);
      const absolutePath = resolve(vaultRoot, relativePath);
      if (absolutePath !== vaultRoot && !absolutePath.startsWith(`${vaultRoot}${sep}`)) {
        new Notice("Embedding log path is outside the vault."); return;
      }
      if (!(await adapter.exists(relativePath))) { new Notice("No embedding log is available yet."); return; }
      const openError = await shell.openPath(absolutePath);
      if (openError) new Notice(`Could not open embedding log: ${safeRunError(openError)}`);
    } catch (error) {
      new Notice(`Could not open embedding log: ${safeRunError(error instanceof Error ? error.message : String(error))}`);
    }
  }
  private async publishResult(result: ClusterResult): Promise<void> { for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER)) (leaf.view as ClusterExplorerView).setResult(result); }
  private updateProgress(phase: string, progress: number): void { this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER).forEach((leaf) => { const view = leaf.view as ClusterExplorerView; view.contentEl.setAttribute("aria-label", `${phase} ${Math.round(progress * 100)}%`); view.setProgress(phase, progress); }); }
  private async saveSettings(): Promise<void> { await this.saveData(this.settings); }
  private secretResolver(): SecretResolver { const storage = (this.app as unknown as { secretStorage?: { getSecret?: (reference: string) => string | null } }).secretStorage; return { getSecret: async (reference) => storage?.getSecret?.(reference) || null }; }
  private confirmGeminiTransmission(count: number): Promise<boolean> { return new Promise((resolve) => new GeminiTransmissionModal(this.app, count, resolve).open()); }
}

function safeRunError(error: unknown): string {
  const messages: string[] = [];
  let current: unknown = error;
  for (let depth = 0; current && depth < 4; depth++) {
    messages.push(current instanceof Error ? current.message : String(current));
    current = (current as { cause?: unknown })?.cause;
  }
  return messages.filter(Boolean).join(": ").replace(/((?:^|[?&\s])(?:key|token|secret|authorization)=)[^&\s]+/gi, "$1[redacted]").slice(0, 500);
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
