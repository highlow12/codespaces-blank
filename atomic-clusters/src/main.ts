import { App, Menu, Modal, Notice, Plugin, TAbstractFile, TFile, TFolder } from "obsidian";
import { configureLocalOrtAssets, disposeLocalOrtAssets, EmbeddingProvider, GeminiEmbeddingProvider, LocalEmbeddingProvider, LocalModelManager, LocalRuntimeDiagnostics, LocalRuntimeProgress, LOCAL_ORT_MJS_ASSET, LOCAL_ORT_WASM_ASSET, LOCAL_ORT_WEBGPU_MJS_ASSET, LOCAL_ORT_WEBGPU_WASM_ASSET, SecretResolver, VaultLocalModelStorage } from "./embedding";
import { createSqliteStore, migrateLegacyAdapter, SqliteClusterStore, KeywordTitleLogStore, NoteStore } from "./storage";
import { AtomicClustersSettingTab, ClusterRunControls, LocalRuntimeTest } from "./settings";
import { ClusterExplorerView, VIEW_TYPE_CLUSTER_EXPLORER } from "./view";
import { CachedEmbedding, ClusteringConfig, ClusterResult, ClusterVisualization, EmbeddingLogEntry, EmbeddingRunLog, normalizeExcludedPaths, normalizeVaultRelativePath, NoteRecord, pathMatchesExcludedFolder, PluginSettings } from "./types";
import { VaultChangeQueue, PendingVaultChanges, buildSoftRefresh, createPendingVaultChanges, decideIncrementalRefresh, markFullRebuildResult, renameClusterResultPaths } from "./incremental";
import { BrowserClusteringWorker, InProcessClusteringWorker, NodeClusteringWorker } from "./worker-client";
import workerSource from "./worker-source";
import browserWorkerSource from "./browser-worker-source";
import { AtomicClustersProgress } from "./progress";
import { preflightRendererClusteringMemory } from "./memory-preflight";
import { prepareLocalOrtRendererModule, resolveLocalOrtAssetPrefix } from "./ort-assets";
import { generateKeywordTitles, KEYWORD_TITLE_ALGORITHM_VERSION } from "./title";
import { projectVisualization } from "./clustering";
import { contentHash } from "./hash";
import initSqlJs from "sql.js";
// EmbeddingLogStore remains exported for legacy callers; runtime writes use
// the SQLite embedding_logs table through SqliteClusterStore.

const DEFAULT_SETTINGS: PluginSettings = {
  embeddingProvider: "gemini", geminiModel: "gemini-embedding-2", geminiSecretRef: "gemini-api-key",
  localModel: "multilingual-e5-small", localExecutionProvider: "auto", excludedFolders: [], excludedNotes: [], minClusterSize: 5, minSamples: 3,
  umapNeighbors: 15, umapMinDist: 0.1, pcaVarianceTarget: 0.9, clusterTitlesEnabled: true,
  automaticRefresh: true, refreshDelaySeconds: 5
};

interface PreparedIncrementalChanges {
  notes: NoteRecord[];
  result: ClusterResult | null;
  provider: EmbeddingProvider;
  noteHashes: Map<string, string>;
  changedPaths: Set<string>;
  deletedPaths: Set<string>;
  renames: Map<string, string>;
  existingCoordinates: Map<string, number[]>;
  existingUmapCoordinates: Map<string, number[]>;
  pathOnly: boolean;
}

interface StoredNoteMetadata { path: string; hash: string; mtime: number; active: boolean; content?: string; }

export default class AtomicClustersPlugin extends Plugin {
  settings!: PluginSettings;
  private worker: NodeClusteringWorker | BrowserClusteringWorker | InProcessClusteringWorker | null = null;
  private latestResult: ClusterResult | null = null;
  private running = false;
  private localModelManager!: LocalModelManager;
  // Keep the renderer's optional local embedding runtime assets in memory.
  private localOrtWebgpuWasmBinary: ArrayBuffer | null = null;
  private operationProgress: AtomicClustersProgress | null = null;
  private runAbortController: AbortController | null = null;
  private sqliteStore: SqliteClusterStore | null = null;
  private latestResultId: string | null = null;
  private explorerSearchNotes: NoteRecord[] = [];
  private resultRevision = 0;
  private readonly visualizationPromises = new WeakMap<readonly string[], Promise<ClusterVisualization>>();
  private pendingVaultChanges: VaultChangeQueue | null = null;
  private incrementalProcessing = false;
  private startupDiffRunning = false;
  private fullRebuildTimer: ReturnType<typeof setTimeout> | undefined;

  async onload(): Promise<void> {
    const adapter = this.app.vault.adapter as typeof this.app.vault.adapter & { getBasePath?: () => string };
    try {
      const prefix = resolveLocalOrtAssetPrefix(adapter.getBasePath?.(), this.manifest.dir, this.manifest.id);
      configureLocalOrtAssets(prefix);
      await this.configureRendererOrtAssets(adapter, prefix);
    } catch { }
    const storedSettings = await this.loadData() as Partial<PluginSettings> | null;
    this.settings = Object.assign({}, DEFAULT_SETTINGS, storedSettings || {});
    this.settings.excludedFolders = normalizeExcludedPaths(this.settings.excludedFolders);
    this.settings.excludedNotes = normalizeExcludedPaths(this.settings.excludedNotes);
    try {
      const sqlite = await this.openSqliteStore(adapter);
      this.latestResult = await sqlite.getResult();
      this.latestResultId = await sqlite.getLatestResultId();
      this.explorerSearchNotes = await sqlite.listNotes(true);
    }
    catch (error) { new Notice(`Atomic Clusters SQLite storage unavailable: ${safeRunError(error)}`); }
    this.localModelManager = new LocalModelManager(new VaultLocalModelStorage(this.app.vault.adapter));
    this.pendingVaultChanges = new VaultChangeQueue({ delayMs: this.refreshDelayMs(), maxDelayMs: 60000, onReady: () => { void this.processPendingVaultChanges(); } });
    this.registerVaultChangeListeners();
    this.registerFileContextMenus();
    this.registerView(VIEW_TYPE_CLUSTER_EXPLORER, (leaf) => new ClusterExplorerView(leaf, () => this.buildClusters(), this.latestResult, (result) => this.ensureVisualization(result), this.explorerSearchNotes));
    this.registerHoverLinkSource(VIEW_TYPE_CLUSTER_EXPLORER, { display: "Atomic Clusters", defaultMod: false });
    this.addRibbonIcon("scatter-chart", "Open Cluster Explorer", () => void this.openExplorer());
    this.addCommand({ id: "build-note-clusters", name: "Build note clusters", callback: () => void this.buildClusters() });
    this.addCommand({ id: "refresh-changed-notes", name: "Refresh changed notes", callback: () => void this.refreshChangedNotes() });
    this.addCommand({ id: "rebuild-all-clusters", name: "Rebuild all clusters", callback: () => void this.buildClusters() });
    this.addCommand({ id: "pause-automatic-refresh", name: "Pause automatic refresh", callback: () => void this.pauseAutomaticRefresh() });
    this.addCommand({ id: "regenerate-cluster-titles", name: "Regenerate cluster titles", callback: () => void this.regenerateTitles() });
    this.addCommand({ id: "open-cluster-explorer", name: "Open cluster explorer", callback: () => void this.openExplorer() });
    this.addCommand({ id: "open-embedding-log", name: "Open embedding log", callback: () => void this.openEmbeddingLog() });
    this.addCommand({ id: "cancel-clustering", name: "Cancel clustering", callback: () => this.cancelClustering() });
    const clusterRun: ClusterRunControls = { build: () => this.buildClusters(), regenerateTitles: () => this.regenerateTitles(), cancel: () => this.cancelClustering(), isRunning: () => this.running || this.incrementalProcessing };
    const testLocalRuntime: LocalRuntimeTest = (onProgress) => this.testLocalRuntime(onProgress);
    this.addSettingTab(new AtomicClustersSettingTab(this.app, this, this.settings, () => this.saveSettings(), this.localModelManager, () => this.openEmbeddingLog(), clusterRun, testLocalRuntime, () => { void this.configureAutomaticRefresh(); }, () => this.refreshAfterExclusionChange()));
    if (this.latestResult && this.settings.automaticRefresh !== false) void this.detectStartupChanges();
  }

  async onunload(): Promise<void> { this.pendingVaultChanges?.dispose(); this.pendingVaultChanges = null; if (this.fullRebuildTimer !== undefined) clearTimeout(this.fullRebuildTimer); this.fullRebuildTimer = undefined; this.operationProgress?.hide(); this.operationProgress = null; await this.worker?.terminate(); this.worker = null; this.sqliteStore?.close(); this.sqliteStore = null; disposeLocalOrtAssets(); }

  private async openSqliteStore(adapter: typeof this.app.vault.adapter & { getBasePath?: () => string }): Promise<SqliteClusterStore> {
    if (this.sqliteStore) return this.sqliteStore;
    const binaryAdapter = adapter as typeof adapter & { writeBinary(path: string, data: ArrayBuffer): Promise<void>; rename?(from: string, to: string): Promise<void>; remove?(path: string): Promise<void> };
    const pluginDir = this.manifest.dir?.replace(/\\/g, "/").replace(/\/$/, "") || `.obsidian/plugins/${this.manifest.id}`;
    const wasmPath = `${pluginDir}/sql-wasm.wasm`;
    const resourcePath = (adapter as typeof adapter & { getResourcePath?: (path: string) => string }).getResourcePath?.(wasmPath) || wasmPath;
    this.sqliteStore = await createSqliteStore(binaryAdapter, () => initSqlJs({ locateFile: () => resourcePath }), {});
    await migrateLegacyAdapter(this.sqliteStore, adapter);
    return this.sqliteStore;
  }

  private async getSqliteStore(): Promise<SqliteClusterStore> {
    if (this.sqliteStore) return this.sqliteStore;
    return this.openSqliteStore(this.app.vault.adapter as typeof this.app.vault.adapter & { getBasePath?: () => string });
  }

  private async configureRendererOrtAssets(adapter: { exists(path: string): Promise<boolean>; readBinary(path: string): Promise<ArrayBuffer> }, prefix: string): Promise<void> {
    const manifestDir = this.manifest.dir?.replace(/\\/g, "/").replace(/\/$/, "");
    const pluginDir = manifestDir?.startsWith(".obsidian/plugins/") ? manifestDir : `.obsidian/plugins/${this.manifest.id}`;
    const mjsPath = `${pluginDir}/${LOCAL_ORT_MJS_ASSET}`;
    const wasmPath = `${pluginDir}/${LOCAL_ORT_WASM_ASSET}`;
    const webgpuMjsPath = `${pluginDir}/${LOCAL_ORT_WEBGPU_MJS_ASSET}`;
    const webgpuWasmPath = `${pluginDir}/${LOCAL_ORT_WEBGPU_WASM_ASSET}`;
    this.localOrtWebgpuWasmBinary = null;
    if (!(await adapter.exists(mjsPath)) || !(await adapter.exists(wasmPath))) return;
    if (typeof Blob === "undefined" || typeof URL.createObjectURL !== "function") return;
    const [mjsBytes, wasmBinary] = await Promise.all([adapter.readBinary(mjsPath), adapter.readBinary(wasmPath)]);
    const mjsSource = prepareLocalOrtRendererModule(new TextDecoder().decode(mjsBytes));
    const mjsUrl = URL.createObjectURL(new Blob([mjsSource], { type: "text/javascript" }));
    let webgpuMjsUrl: string | undefined;
    let webgpuWasmBinary: ArrayBuffer | undefined;
    if (await adapter.exists(webgpuMjsPath) && await adapter.exists(webgpuWasmPath)) {
      const [webgpuMjsBytes, webgpuBinary] = await Promise.all([adapter.readBinary(webgpuMjsPath), adapter.readBinary(webgpuWasmPath)]);
      const webgpuSource = prepareLocalOrtRendererModule(new TextDecoder().decode(webgpuMjsBytes), LOCAL_ORT_WEBGPU_WASM_ASSET);
      webgpuMjsUrl = URL.createObjectURL(new Blob([webgpuSource], { type: "text/javascript" }));
      webgpuWasmBinary = webgpuBinary;
      this.localOrtWebgpuWasmBinary = webgpuBinary;
    }
    configureLocalOrtAssets(prefix, { mjs: mjsUrl, wasmBinary, ...(webgpuMjsUrl ? { webgpuMjs: webgpuMjsUrl, webgpuWasmBinary } : {}), revoke: () => { URL.revokeObjectURL(mjsUrl); if (webgpuMjsUrl) URL.revokeObjectURL(webgpuMjsUrl); } });
  }

  private async buildClusters(force = false): Promise<void> {
    if (this.running || (this.incrementalProcessing && !force)) { new Notice("Atomic Clusters is already running."); return; }
    if (this.fullRebuildTimer !== undefined) { clearTimeout(this.fullRebuildTimer); this.fullRebuildTimer = undefined; }
    const queuedBeforeBuild = this.pendingVaultChanges?.drain() || null;
    this.publishPendingChangeCount();
    this.running = true;
    this.runAbortController = new AbortController();
    const runSignal = this.runAbortController.signal;
    const progress = new AtomicClustersProgress("Atomic Clusters");
    this.operationProgress = progress;
    const startedAt = new Date().toISOString(); const logEntries: EmbeddingLogEntry[] = []; let persistedRunLog: EmbeddingRunLog | null = null; let runTotal = 0; let runStage: EmbeddingRunLog["stage"] = "embedding"; let runtimeDiagnostics: LocalRuntimeDiagnostics | undefined; let completed = false;
    const counts = () => ({ succeeded: logEntries.filter((entry) => entry.status === "success").length, failed: logEntries.filter((entry) => entry.status === "failure").length, cached: logEntries.filter((entry) => entry.status === "cached").length });
    let runProvider = this.settings.embeddingProvider; let runModel = this.settings.embeddingProvider === "gemini" ? this.settings.geminiModel : `${this.settings.localModel}@2024-05-01`;
    try {
      progress.update({ phase: "cache scan", progress: 0.02, detail: "Scanning Markdown notes" });
      const notes = await new NoteStore(this.app.vault).collect(this.settings.excludedFolders, this.settings.excludedNotes);
      if (!notes.length) throw new Error("No Markdown notes found in the selected vault folders.");
      const sqlite = await this.getSqliteStore();
      await sqlite.syncActiveNotes(notes);
      runTotal = notes.length;
      progress.update({ phase: "cache scan", progress: 0.1, detail: `Scanned ${notes.length} notes` });
      const entries = await sqlite.loadEmbeddings();
      const provider: EmbeddingProvider = this.settings.embeddingProvider === "gemini"
        ? new GeminiEmbeddingProvider(this.settings, this.secretResolver(), (count) => this.confirmGeminiTransmission(count))
        : new LocalEmbeddingProvider(this.settings, undefined, this.localModelManager);
      runProvider = provider.id; runModel = provider.model;
      const cached = (note: NoteRecord): CachedEmbedding | undefined => { const item = entries.get(`${provider.id}:${provider.model}:${note.path}`); return item?.hash === note.hash ? item : undefined; };
      const fresh = notes.filter((note) => !cached(note));
      const freshPaths = new Set(fresh.map((note) => note.path));
      notes.filter((note) => !freshPaths.has(note.path)).forEach((note) => logEntries.push({ path: note.path, timestamp: new Date().toISOString(), provider: provider.id, model: provider.model, status: "cached", durationMs: 0 }));
      progress.update({ phase: "cache scan", progress: 0.15, detail: `${fresh.length} notes need embeddings · ${notes.length - fresh.length} cached` });
      if (provider.id === "local" && fresh.length) {
        runStage = "preflight";
        const localProvider = provider as LocalEmbeddingProvider;
        await localProvider.preflight((update) => progress.update({ phase: "preflight", progress: 0.15 + update.progress * 0.1, detail: update.detail || update.phase }), runSignal);
        runtimeDiagnostics = localProvider.runtimeDiagnostics;
        runStage = "embedding";
      }
      let processedFresh = 0; const embedded: CachedEmbedding[] = []; let embeddingStreamError: unknown;
      const onEmbeddingProgress = (done: number, total: number) => progress.update({ phase: "embedding", progress: 0.25 + (total ? done / total * 0.55 : 0), detail: `${done}/${total} notes processed` });
      const onEmbeddingNote = (entry: EmbeddingLogEntry) => { logEntries.push(entry); processedFresh++; progress.update({ phase: "embedding", progress: 0.25 + (processedFresh / Math.max(1, fresh.length)) * 0.55, detail: `${entry.status === "failure" ? "Failed" : "Embedded"}: ${entry.path}` }); };
      if (fresh.length) {
        if (provider.embedBatches) {
          try { for await (const batch of provider.embedBatches(fresh, onEmbeddingProgress, onEmbeddingNote, runSignal)) embedded.push(...batch); }
          catch (error) { embeddingStreamError = error; }
        } else embedded.push(...await provider.embed(fresh, onEmbeddingProgress, onEmbeddingNote, runSignal));
      }
      if (embedded.length) { await sqlite.putEmbeddings(embedded); for (const item of embedded) entries.set(`${item.provider}:${item.model}:${item.path}`, item); }
      if (embeddingStreamError) throw embeddingStreamError;
      const activeNotes = notes.filter((note) => cached(note));
      const freshEmbeddingFailed = fresh.length > 0 && embedded.length === 0;
      const embeddingError = freshEmbeddingFailed
        ? "All notes requiring fresh embeddings failed; fix the embedding provider or model and run the build again."
        : !activeNotes.length ? "No notes have usable embeddings; fix the failed notes and run the build again." : undefined;
      const embeddingCounts = counts();
      persistedRunLog = { version: 1, startedAt, completedAt: new Date().toISOString(), provider: provider.id, model: provider.model, total: notes.length, ...embeddingCounts, entries: logEntries, status: embeddingError ? "failed" : "completed", stage: "embedding", ...(runtimeDiagnostics ? { runtime: runtimeDiagnostics } : {}), ...(embeddingError ? { error: safeRunError(embeddingError) } : {}) };
      await sqlite.saveEmbeddingLog(persistedRunLog);
      progress.update({ phase: "cache save", progress: 0.82, detail: `${persistedRunLog.succeeded} embedded · ${persistedRunLog.failed} failed · ${persistedRunLog.cached} cached` });
      if (embeddingError) throw new Error(embeddingError);
      persistedRunLog = { ...persistedRunLog, stage: "clustering" };
      const vectors = activeNotes.map((note) => cached(note)?.vector);
      if (vectors.some((vector) => !vector)) throw new Error("Embedding cache is incomplete; please run the build again.");
      const memoryPreflight = preflightRendererClusteringMemory(vectors as number[][]);
      progress.update({ phase: "memory preflight", progress: 0.83, detail: memoryPreflight.detail });
      if (!memoryPreflight.canProceed) throw new Error(memoryPreflight.error || memoryPreflight.detail);
      const ids = activeNotes.map((note) => note.path);
      const worker = await this.getWorker();
      progress.update({ phase: "clustering", progress: 0.84, detail: `${worker instanceof InProcessClusteringWorker ? "Worker APIs unavailable; clustering in process" : worker instanceof BrowserClusteringWorker ? "Using Chromium worker fallback" : "Clustering"} · ${activeNotes.length} notes` });
      const config: ClusteringConfig = { minClusterSize: this.settings.minClusterSize, minSamples: this.settings.minSamples, umapNeighbors: this.settings.umapNeighbors, umapMinDist: this.settings.umapMinDist, pcaVarianceTarget: this.settings.pcaVarianceTarget, seed: 42, deferVisualization: true };
      const workerResult = await worker.run(ids, vectors as number[][], config, (phase, value) => { this.updateProgress(phase, value); progress.update({ phase, progress: 0.84 + value * 0.15, detail: `Clustering ${Math.round(value * 100)}%` }); });
      const fullResult = markFullRebuildResult(workerResult, provider.id, provider.model);
      if (fullResult.pca.model) {
        await sqlite.savePcaModel(fullResult.pca.model);
        await sqlite.projectMany(activeNotes.map((note, index) => ({ path: note.path, vector: vectors[index] as number[] })), fullResult.pca.model);
      }
      const structuralResult = { ...fullResult, titles: undefined, titleGeneration: undefined };
      this.setLatestResult(structuralResult);
      const structuralResultId = await sqlite.saveResult(structuralResult, { noteHashes: new Map(notes.map((note) => [note.path, note.hash])) }); this.latestResultId = structuralResultId; await this.publishResult(structuralResult);
      completed = true;
      if (this.settings.clusterTitlesEnabled !== false) {
        try { await this.generateTitlesForResult(structuralResult, notes, runSignal, progress, false, structuralResultId); }
        catch (titleError) {
          if (runSignal.aborted || (titleError instanceof Error && titleError.message.toLowerCase().includes("cancel"))) throw titleError;
          progress.update({ phase: "cluster titles", progress: 1, detail: `Titles skipped: ${safeRunError(titleError)}` });
        }
      }
      progress.complete(`Built ${structuralResult.hierarchy.leaves.length} clusters`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const cancelled = message === "Clustering cancelled" || message.includes("cancelled");
      progress.fail(cancelled ? "Clustering cancelled" : `Build failed: ${message}`);
      const status = cancelled ? "cancelled" : "failed";
      const failedLog: EmbeddingRunLog = persistedRunLog
        ? { ...persistedRunLog, completedAt: new Date().toISOString(), status, error: safeRunError(error) }
        : { version: 1, startedAt, completedAt: new Date().toISOString(), provider: runProvider, model: runModel, total: runTotal, ...counts(), entries: logEntries, status, stage: runStage, error: safeRunError(error) };
      await this.getSqliteStore().then((store) => store.saveEmbeddingLog(failedLog)).catch(() => undefined);
      if (!cancelled) new Notice(`Atomic Clusters failed: ${safeRunError(error)}`);
    }
    finally {
      this.running = false; this.operationProgress = null; this.runAbortController = null;
      if (completed) this.pendingVaultChanges?.notifyReady();
      else if (queuedBeforeBuild) { this.pendingVaultChanges?.requeue(queuedBeforeBuild); this.publishPendingChangeCount(); }
    }
  }

  private async regenerateTitles(): Promise<void> {
    if (this.running || this.incrementalProcessing) { new Notice("Atomic Clusters is already running."); return; }
    this.running = true;
    this.runAbortController = new AbortController();
    const runSignal = this.runAbortController.signal;
    const progress = new AtomicClustersProgress("Regenerate cluster titles");
    this.operationProgress = progress;
    try {
      progress.update({ phase: "result load", progress: 0.02, detail: "Loading the saved cluster hierarchy" });
      const sqlite = await this.getSqliteStore(); const result = await sqlite.getResult();
      if (!result) throw new Error("No saved cluster result is available; build clusters first.");
      const notes = await new NoteStore(this.app.vault).collect(this.settings.excludedFolders, this.settings.excludedNotes);
      const byPath = new Map(notes.map((note) => [note.path, note]));
      const missing = result.ids.filter((path) => !byPath.has(path));
      if (missing.length) throw new Error(`${missing.length} note${missing.length === 1 ? " is" : "s are"} missing from the saved cluster result; build clusters again before regenerating titles.`);
      const orderedNotes = result.ids.map((path) => byPath.get(path)!);
      progress.update({ phase: "result load", progress: 0.1, detail: `${orderedNotes.length} current notes · ${result.hierarchy.leaves.length + result.hierarchy.merges.length} hierarchy nodes` });
      const resultId = await sqlite.getLatestResultId(); if (!resultId) throw new Error("Saved cluster result metadata is unavailable; build clusters again.");
      const titledResult = await this.generateTitlesForResult(result, orderedNotes, runSignal, progress, true, resultId);
      progress.complete(`Regenerated ${Object.keys(titledResult.titles || {}).length} keyword titles`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const cancelled = message.toLowerCase().includes("cancel");
      progress.fail(cancelled ? "Title regeneration cancelled" : `Title regeneration failed: ${message}`);
      if (!cancelled) new Notice(`Atomic Clusters: ${safeRunError(error)}`);
    } finally {
      this.running = false; this.operationProgress = null; this.runAbortController = null;
      if (this.settings.automaticRefresh !== false) this.pendingVaultChanges?.notifyReady();
    }
  }

  private async generateTitlesForResult(result: ClusterResult, notes: NoteRecord[], signal: AbortSignal, progress: AtomicClustersProgress, _forceRegenerate: boolean, resultId?: string): Promise<ClusterResult> {
    const started = Date.now();
    progress.update({ phase: "keyword titles", progress: 0.9, detail: "Selecting representative keywords" });
    const titled = generateKeywordTitles(result, notes, { signal, onProgress: (done, total) => progress.update({ phase: "keyword titles", progress: 0.9 + (total ? done / total * 0.09 : 0), detail: `${done}/${total} hierarchy nodes` }) });
    if (signal.aborted) throw new Error("Cluster title generation cancelled");
    const sqlite = await this.getSqliteStore(); const targetResultId = resultId || await sqlite.getLatestResultId();
    if (!targetResultId) throw new Error("Saved cluster result metadata is unavailable; build clusters again.");
    await sqlite.patchResultTitles(targetResultId, titled.titles || {}, titled.titleGeneration);
    this.setLatestResult(titled); await this.publishResult(titled);
    await new KeywordTitleLogStore(this.app.vault).save({ version: 1, method: "keywords", algorithmVersion: KEYWORD_TITLE_ALGORITHM_VERSION, startedAt: new Date(started).toISOString(), completedAt: new Date().toISOString(), durationMs: Date.now() - started, nodeCount: titled.titleGeneration?.nodeCount || 0, nodes: Object.fromEntries(Object.entries(titled.titles || {}).map(([id, title]) => [id, { title, scores: titled.titleGeneration?.scores?.[id] || [] }])) }).catch(() => undefined);
    return titled;
  }

  private async testLocalRuntime(onProgress: (progress: LocalRuntimeProgress) => void): Promise<void> {
    const startedAt = new Date().toISOString();
    const provider = new LocalEmbeddingProvider(this.settings, undefined, this.localModelManager);
    try {
      await provider.preflight(onProgress);
      await (await this.getSqliteStore()).saveEmbeddingLog({ version: 1, startedAt, completedAt: new Date().toISOString(), provider: provider.id, model: provider.model, total: 0, succeeded: 0, failed: 0, cached: 0, entries: [], status: "completed", stage: "preflight", ...(provider.runtimeDiagnostics ? { runtime: provider.runtimeDiagnostics } : {}) });
    } catch (error) {
      await (await this.getSqliteStore()).saveEmbeddingLog({ version: 1, startedAt, completedAt: new Date().toISOString(), provider: provider.id, model: provider.model, total: 0, succeeded: 0, failed: 0, cached: 0, entries: [], status: "failed", stage: "preflight", error: safeRunError(error) }).catch(() => undefined);
      throw error;
    }
  }

  private createEmbeddingProvider(): EmbeddingProvider {
    return this.settings.embeddingProvider === "gemini"
      ? new GeminiEmbeddingProvider(this.settings, this.secretResolver(), (count) => this.confirmGeminiTransmission(count))
      : new LocalEmbeddingProvider(this.settings, undefined, this.localModelManager);
  }

  private refreshDelayMs(): number {
    const seconds = Number(this.settings.refreshDelaySeconds ?? 5);
    return Math.round(Math.max(0, Math.min(60, Number.isFinite(seconds) ? seconds : 5)) * 1000);
  }

  private isMarkdownPath(path: string): boolean { return path.toLowerCase().endsWith(".md"); }
  private isExcludedFolderPath(path: string): boolean { return normalizeExcludedPaths(this.settings.excludedFolders).some((folder) => pathMatchesExcludedFolder(path, folder)); }
  private isExcludedNotePath(path: string): boolean { return new Set(normalizeExcludedPaths(this.settings.excludedNotes)).has(normalizeVaultRelativePath(path)); }
  private isExcludedPath(path: string): boolean { return this.isExcludedFolderPath(path) || this.isExcludedNotePath(path); }
  private hasIncludedMarkdownNotes(excludedFolders: string[], excludedNotes: string[]): boolean {
    const folders = normalizeExcludedPaths(excludedFolders);
    const notes = new Set(normalizeExcludedPaths(excludedNotes));
    return this.app.vault.getMarkdownFiles().some((file) => {
      const path = normalizeVaultRelativePath(file.path);
      return !!path && !notes.has(path) && !folders.some((folder) => pathMatchesExcludedFolder(path, folder));
    });
  }
  private markdownPath(file: { path?: string; extension?: string }): string | null {
    const path = file.path || "";
    return path && (file.extension?.toLowerCase() === "md" || this.isMarkdownPath(path)) ? path : null;
  }
  private currentMarkdownFiles() { return this.app.vault.getMarkdownFiles().filter((file) => !this.isExcludedPath(file.path)); }

  private async collectIncrementalNotes(sqlite: SqliteClusterStore, snapshot: PendingVaultChanges): Promise<{ notes: NoteRecord[]; stored: StoredNoteMetadata[] }> {
    const stored = await sqlite.listNoteMetadata(true);
    const previous = new Map(stored.map((note) => [note.path, note]));
    const hinted = new Set([...snapshot.created, ...snapshot.modified, ...snapshot.renamed.values()]);
    const notes = await Promise.all(this.currentMarkdownFiles().map(async (file) => {
      const old = previous.get(file.path);
      if (old && !hinted.has(file.path) && old.mtime === file.stat.mtime && old.content !== undefined && old.content !== "") return { path: file.path, title: file.basename, content: old.content, mtime: file.stat.mtime, hash: old.hash };
      const content = await this.app.vault.cachedRead(file);
      return { path: file.path, title: file.basename, content, mtime: file.stat.mtime, hash: await contentHash(content) };
    }));
    return { notes, stored };
  }

  private registerVaultChangeListeners(): void {
    const queuePath = (kind: "created" | "modified" | "deleted", path: string): void => {
      if (!this.pendingVaultChanges || this.settings.automaticRefresh === false || !this.isMarkdownPath(path) || this.isExcludedPath(path)) return;
      if (kind === "created") this.pendingVaultChanges.enqueueCreated(path);
      else if (kind === "modified") this.pendingVaultChanges.enqueueModified(path);
      else this.pendingVaultChanges.enqueueDeleted(path);
      this.publishPendingChangeCount();
    };
    this.registerEvent(this.app.vault.on("create", (file) => { const path = this.markdownPath(file); if (path) queuePath("created", path); }));
    this.registerEvent(this.app.vault.on("modify", (file) => { const path = this.markdownPath(file); if (path) queuePath("modified", path); }));
    this.registerEvent(this.app.vault.on("delete", (file) => { const path = file.path || ""; if (path) queuePath("deleted", path); }));
    this.registerEvent(this.app.vault.on("rename", (file, oldPath) => {
      const nextPath = this.markdownPath(file); const previousPath = String(oldPath || "");
      if (!this.pendingVaultChanges) return;
      const enqueueRename = (from: string, to: string): void => {
        const previouslyExcludedNote = this.isExcludedNotePath(from);
        const carriedNoteExclusion = previouslyExcludedNote && !this.isExcludedNotePath(to) && this.rewriteExcludedNoteRename(from, to);
        if (this.settings.automaticRefresh === false) return;
        const previousIncluded = this.isMarkdownPath(from) && !this.isExcludedFolderPath(from) && !previouslyExcludedNote;
        const nextIncluded = this.isMarkdownPath(to) && !this.isExcludedFolderPath(to) && !this.isExcludedNotePath(to) && !carriedNoteExclusion;
        if (previousIncluded && nextIncluded) this.pendingVaultChanges!.enqueueRenamed(from, to);
        else if (previousIncluded) this.pendingVaultChanges!.enqueueDeleted(from);
        else if (nextIncluded) this.pendingVaultChanges!.enqueueCreated(to);
      };
      if (nextPath) { enqueueRename(previousPath, nextPath); this.publishPendingChangeCount(); return; }
      const folder = file as { path?: string; children?: unknown[] };
      const currentFolderPath = String(folder.path || "");
      const markdownChildren = (node: { path?: string; children?: unknown[] }): string[] => {
        const path = String(node.path || ""); const children = node.children || [];
        if (children.length) return children.flatMap((child) => markdownChildren(child as { path?: string; children?: unknown[] }));
        return this.isMarkdownPath(path) ? [path] : [];
      };
      const childPaths = markdownChildren(folder);
      for (const childPath of childPaths) {
        const suffix = currentFolderPath && childPath.startsWith(`${currentFolderPath}/`) ? childPath.slice(currentFolderPath.length + 1) : childPath;
        enqueueRename(previousPath + (suffix ? `/${suffix}` : ""), childPath);
      }
      if (childPaths.length) this.publishPendingChangeCount();
    }));
  }

  private registerFileContextMenus(): void {
    this.registerEvent(this.app.workspace.on("file-menu", (menu, file) => this.addFileContextMenuItems(menu, file)));
  }

  private addFileContextMenuItems(menu: Menu, file: TAbstractFile): void {
    const path = normalizeVaultRelativePath(file.path);
    if (!path) return;
    if (file instanceof TFile) {
      if (!this.isMarkdownPath(path)) return;
      const directlyExcluded = this.isExcludedNotePath(path);
      const coveredByParent = this.isExcludedFolderPath(path);
      menu.addSeparator();
      menu.addItem((item) => {
        if (coveredByParent) {
          item.setTitle("Excluded by parent folder").setIcon("eye-off").setDisabled(true);
          return;
        }
        item
          .setTitle(directlyExcluded ? "Restore in Atomic Clusters" : "Exclude from Atomic Clusters")
          .setIcon(directlyExcluded ? "eye" : "eye-off")
          .setWarning(!directlyExcluded)
          .onClick(() => { void this.setNoteExcluded(path, !directlyExcluded).catch((error) => new Notice(`Atomic Clusters could not update note exclusion: ${safeRunError(error)}`)); });
      });
      return;
    }
    if (file instanceof TFolder) {
      const directlyExcluded = normalizeExcludedPaths(this.settings.excludedFolders).includes(path);
      const coveredByParent = this.isExcludedFolderPath(path) && !directlyExcluded;
      menu.addSeparator();
      menu.addItem((item) => {
        item.setTitle(directlyExcluded ? "Include folder in Atomic Clusters" : coveredByParent ? "Excluded by parent folder" : "Exclude folder").setIcon(directlyExcluded ? "folder-open" : "folder-minus");
        if (coveredByParent) item.setDisabled(true);
        else item.onClick(() => { void this.setFolderExcluded(path, !directlyExcluded).catch((error) => new Notice(`Atomic Clusters could not update folder exclusion: ${safeRunError(error)}`)); });
      });
    }
  }

  private async setNoteExcluded(path: string, excluded: boolean): Promise<void> {
    const normalized = normalizeVaultRelativePath(path);
    if (!normalized) return;
    const current = normalizeExcludedPaths(this.settings.excludedNotes);
    const next = normalizeExcludedPaths(excluded ? [...current, normalized] : current.filter((item) => item !== normalized));
    if (excluded && !this.hasIncludedMarkdownNotes(this.settings.excludedFolders, next)) {
      new Notice("Atomic Clusters must keep at least one Markdown note included. Restore another note or folder before excluding this one.");
      return;
    }
    this.settings.excludedNotes = next;
    await this.saveSettings();
    await this.refreshAfterExclusionChange();
    new Notice(excluded ? `Excluded from Atomic Clusters: ${normalized}` : `Restored in Atomic Clusters: ${normalized}`);
  }

  private async setFolderExcluded(path: string, excluded: boolean): Promise<void> {
    const normalized = normalizeVaultRelativePath(path);
    if (!normalized) return;
    const current = normalizeExcludedPaths(this.settings.excludedFolders);
    const next = normalizeExcludedPaths(excluded ? [...current, normalized] : current.filter((item) => item !== normalized));
    if (excluded && !this.hasIncludedMarkdownNotes(next, this.settings.excludedNotes)) {
      new Notice("Atomic Clusters must keep at least one Markdown note included. Restore another note or folder before excluding this folder.");
      return;
    }
    this.settings.excludedFolders = next;
    await this.saveSettings();
    await this.refreshAfterExclusionChange();
    new Notice(excluded ? `Excluded folder from Atomic Clusters: ${normalized}` : `Restored folder in Atomic Clusters: ${normalized}`);
  }

  private rewriteExcludedNoteRename(oldPath: string, newPath: string): boolean {
    const oldNormalized = normalizeVaultRelativePath(oldPath); const newNormalized = normalizeVaultRelativePath(newPath);
    const current = normalizeExcludedPaths(this.settings.excludedNotes);
    if (!oldNormalized || !newNormalized || !current.includes(oldNormalized) || current.includes(newNormalized)) return false;
    this.settings.excludedNotes = normalizeExcludedPaths(current.map((path) => path === oldNormalized ? newNormalized : path));
    void this.saveSettings().catch(() => undefined);
    return true;
  }

  private async refreshAfterExclusionChange(): Promise<void> {
    if (!this.latestResult || !this.pendingVaultChanges) return;
    await this.detectStartupChanges(true);
    if (!this.pendingVaultChanges.hasChanges) return;
    this.publishPendingChangeCount();
    if (this.running || this.incrementalProcessing) return;
    await this.processPendingVaultChanges(true);
  }

  private async configureAutomaticRefresh(): Promise<void> {
    this.pendingVaultChanges?.setDelay(this.refreshDelayMs());
    if (this.settings.automaticRefresh === false) { this.pendingVaultChanges?.clear(); this.publishPendingChangeCount(); return; }
    if (!this.latestResult) return;
    const provider = this.createEmbeddingProvider();
    if (this.latestResult.embeddingProvider && this.latestResult.embeddingModel && (this.latestResult.embeddingProvider !== provider.id || this.latestResult.embeddingModel !== provider.model)) {
      try { for (const file of this.currentMarkdownFiles()) this.pendingVaultChanges?.enqueueModified(file.path); }
      catch (error) { new Notice(`Atomic Clusters could not scan changed notes: ${safeRunError(error)}`); }
      return;
    }
    await this.detectStartupChanges();
  }

  private async detectStartupChanges(force = false): Promise<void> {
    if ((!force && this.settings.automaticRefresh === false) || this.startupDiffRunning || !this.latestResult || !this.pendingVaultChanges) return;
    this.startupDiffRunning = true;
    try {
      const sqlite = await this.getSqliteStore();
      const { notes, stored } = await this.collectIncrementalNotes(sqlite, createPendingVaultChanges());
      const current = new Map(notes.map((note) => [note.path, note]));
      const previous = new Map(stored.map((note) => [note.path, note]));
      const resultId = this.latestResultId || await sqlite.getLatestResultId();
      const savedHashes = resultId ? await sqlite.getResultNoteHashes(resultId) : new Map<string, string>();
      for (const [path, hash] of savedHashes) {
        const old = previous.get(path);
        previous.set(path, old ? { ...old, hash } : { path, hash, mtime: 0, active: true });
      }
      const unknownCurrent = notes.filter((note) => !previous.has(note.path));
      const missingPrevious = [...previous.values()].filter((note) => !current.has(note.path));
      const previousByHash = new Map<string, string[]>();
      const currentByHash = new Map<string, string[]>();
      for (const note of missingPrevious) previousByHash.set(note.hash, [...(previousByHash.get(note.hash) || []), note.path]);
      for (const note of unknownCurrent) currentByHash.set(note.hash, [...(currentByHash.get(note.hash) || []), note.path]);
      const inferredRenames = new Set<string>();
      for (const [hash, oldPaths] of previousByHash) {
        const newPaths = currentByHash.get(hash) || [];
        if (oldPaths.length === 1 && newPaths.length === 1) {
          this.pendingVaultChanges.enqueueRenamed(oldPaths[0], newPaths[0]);
          inferredRenames.add(oldPaths[0]); inferredRenames.add(newPaths[0]);
        }
      }
      for (const note of notes) {
        if (inferredRenames.has(note.path)) continue;
        const old = previous.get(note.path);
        if (!old) this.pendingVaultChanges.enqueueCreated(note.path);
        else if (old.hash !== note.hash) this.pendingVaultChanges.enqueueModified(note.path);
      }
      for (const note of previous.values()) if (!current.has(note.path) && !inferredRenames.has(note.path)) this.pendingVaultChanges.enqueueDeleted(note.path);
      for (const path of this.latestResult.ids) if (!current.has(path) && !previous.has(path) && !inferredRenames.has(path)) this.pendingVaultChanges.enqueueDeleted(path);
      this.publishPendingChangeCount();
    } catch (error) {
      if (force) new Notice(`Atomic Clusters could not detect changed notes: ${safeRunError(error)}`);
    } finally { this.startupDiffRunning = false; }
  }

  private async refreshChangedNotes(): Promise<void> {
    if (this.running || this.incrementalProcessing) { new Notice("Atomic Clusters is already running."); return; }
    if (!this.latestResult) { await this.buildClusters(); return; }
    await this.detectStartupChanges(true);
    if (!this.pendingVaultChanges?.hasChanges) { new Notice("Atomic Clusters found no changed Markdown notes."); return; }
    await this.processPendingVaultChanges(true);
  }

  private scheduleFullRebuild(): void {
    if (this.settings.automaticRefresh === false || this.fullRebuildTimer !== undefined) return;
    this.fullRebuildTimer = setTimeout(() => {
      this.fullRebuildTimer = undefined;
      if (this.settings.automaticRefresh === false) return;
      if (this.running || this.incrementalProcessing) { this.scheduleFullRebuild(); return; }
      void this.buildClusters();
    }, this.refreshDelayMs());
  }

  private async processPendingVaultChanges(force = false): Promise<void> {
    if ((!force && this.settings.automaticRefresh === false) || this.running || this.incrementalProcessing || !this.pendingVaultChanges) return;
    if (!this.latestResult) { this.pendingVaultChanges.clear(); return; }
    const snapshot = this.pendingVaultChanges.drain();
    if (!snapshot) return;
    this.publishPendingChangeCount();
    this.incrementalProcessing = true;
    let handedToFullBuild = false; let succeeded = false;
    try {
      const prepared = await this.prepareIncrementalChanges(snapshot);
      const decision = decideIncrementalRefresh({ result: prepared.result, activeNoteCount: prepared.notes.length, changedNoteCount: prepared.changedPaths.size, deletedNoteCount: prepared.deletedPaths.size, pathOnly: prepared.pathOnly, provider: prepared.provider.id, model: prepared.provider.model });
      if (decision.mode === "full") {
        this.pendingVaultChanges.requeue(snapshot); this.publishPendingChangeCount(); handedToFullBuild = true; this.incrementalProcessing = false;
        await this.buildClusters(true); return;
      }
      await this.applyIncrementalRefresh(prepared); succeeded = true;
    } catch (error) {
      if (!handedToFullBuild) { this.pendingVaultChanges.requeue(snapshot); this.publishPendingChangeCount(); }
      new Notice(`Atomic Clusters incremental refresh failed: ${safeRunError(error)}`);
    } finally {
      this.incrementalProcessing = false;
      if (succeeded && !handedToFullBuild && this.pendingVaultChanges.hasChanges && !this.running && (force || this.settings.automaticRefresh !== false)) this.pendingVaultChanges.notifyReady();
    }
  }

  private async prepareIncrementalChanges(snapshot: PendingVaultChanges): Promise<PreparedIncrementalChanges> {
    const sqlite = await this.getSqliteStore();
    const { notes, stored } = await this.collectIncrementalNotes(sqlite, snapshot);
    if (!notes.length) throw new Error("No Markdown notes found in the selected vault folders.");
    const result = this.latestResult || await sqlite.getResult();
    const provider = this.createEmbeddingProvider();
    const resultId = result ? (this.latestResultId || await sqlite.getLatestResultId()) : null;
    const savedHashes = resultId ? await sqlite.getResultNoteHashes(resultId) : new Map<string, string>();
    const previous = new Map(stored.map((note) => [note.path, note]));
    for (const [path, hash] of savedHashes) {
      const old = previous.get(path); previous.set(path, old ? { ...old, hash } : { path, hash, mtime: 0, active: true });
    }
    const current = new Map(notes.map((note) => [note.path, note]));
    const renames = new Map<string, string>(); const resultPaths = new Set(result?.ids || []);
    for (const [oldPath, newPath] of snapshot.renamed) {
      const normalRename = previous.has(oldPath) && current.has(newPath) && !previous.has(newPath);
      const retryAfterMove = resultPaths.has(oldPath) && current.has(newPath) && !current.has(oldPath);
      if (normalRename || retryAfterMove) renames.set(oldPath, newPath);
    }
    const renamedOld = new Set(renames.keys()); const changedPaths = new Set<string>(); const deletedPaths = new Set<string>();
    for (const note of notes) {
      const source = [...renames.entries()].find(([, target]) => target === note.path)?.[0];
      const old = source ? previous.get(source) || previous.get(note.path) : previous.get(note.path);
      const eventForcesContentCheck = snapshot.created.has(note.path) || snapshot.modified.has(note.path);
      if (!old || old.hash !== note.hash || eventForcesContentCheck && !savedHashes.has(note.path)) changedPaths.add(note.path);
    }
    for (const old of previous.values()) if (!current.has(old.path) && !renamedOld.has(old.path)) deletedPaths.add(old.path);
    for (const path of snapshot.deleted) if (!current.has(path) && !renamedOld.has(path) && result?.ids.includes(path)) deletedPaths.add(path);
    const entries = await sqlite.loadEmbeddings(provider.id, provider.model);
    for (const note of notes) {
      const source = [...renames.entries()].find(([, target]) => target === note.path)?.[0];
      const cachePath = source && previous.has(source) ? source : note.path;
      if (entries.get(`${provider.id}:${provider.model}:${cachePath}`)?.hash !== note.hash) changedPaths.add(note.path);
    }
    const existingCoordinates = new Map<string, number[]>(); const existingUmapCoordinates = new Map<string, number[]>(); const modelHash = result?.pca.model?.modelHash;
    if (modelHash && result) {
      const coordinates = await sqlite.getPcaCoordinatesMany(result.ids, modelHash);
      result.ids.forEach((path, index) => { const coordinate = coordinates[index]; if (coordinate) existingCoordinates.set(path, coordinate); });
    }
    if (result && result.umap?.coordinates.length === result.ids.length) result.ids.forEach((path, index) => { const coordinate = result.umap?.coordinates[index]; if (coordinate) existingUmapCoordinates.set(path, coordinate); });
    return { notes, result, provider, noteHashes: new Map(notes.map((note) => [note.path, note.hash])), changedPaths, deletedPaths, renames, existingCoordinates, existingUmapCoordinates, pathOnly: changedPaths.size === 0 && deletedPaths.size === 0 && renames.size > 0 };
  }

  private async applyIncrementalRefresh(prepared: PreparedIncrementalChanges): Promise<void> {
    const result = prepared.result; if (!result) throw new Error("No saved cluster result is available; build clusters first.");
    const sqlite = await this.getSqliteStore();
    this.running = true; this.runAbortController = new AbortController();
    const runSignal = this.runAbortController.signal; const progress = new AtomicClustersProgress("Atomic Clusters incremental refresh"); this.operationProgress = progress;
    const startedAt = new Date().toISOString(); const logEntries: EmbeddingLogEntry[] = []; let persistedRunLog: EmbeddingRunLog | null = null; let runStage: EmbeddingRunLog["stage"] = "embedding"; let runtimeDiagnostics: LocalRuntimeDiagnostics | undefined;
    const counts = () => ({ succeeded: logEntries.filter((entry) => entry.status === "success").length, failed: logEntries.filter((entry) => entry.status === "failure").length, cached: logEntries.filter((entry) => entry.status === "cached").length });
    try {
      progress.update({ phase: "change scan", progress: 0.05, detail: `${prepared.changedPaths.size} changed · ${prepared.deletedPaths.size} deleted` });
      const appliedRenames = new Map<string, string>();
      for (const [oldPath, newPath] of prepared.renames) { if (!await sqlite.renameNote(oldPath, newPath)) throw new Error(`Could not atomically move cached note state from ${oldPath} to ${newPath}.`); appliedRenames.set(oldPath, newPath); }
      await sqlite.syncActiveNotes(prepared.notes);
      let currentResult = renameClusterResultPaths(result, appliedRenames);
      if (!prepared.changedPaths.size && !prepared.deletedPaths.size) {
        const resultId = await sqlite.saveResult(currentResult, { noteHashes: prepared.noteHashes }); this.latestResultId = resultId; this.setLatestResult(currentResult); await this.publishResult(currentResult); progress.complete("No embedding changes"); return;
      }
      const entries = await sqlite.loadEmbeddings(prepared.provider.id, prepared.provider.model);
      const cached = (note: NoteRecord): CachedEmbedding | undefined => { const item = entries.get(`${prepared.provider.id}:${prepared.provider.model}:${note.path}`); return item?.hash === note.hash ? item : undefined; };
      const fresh = prepared.notes.filter((note) => !cached(note));
      for (const note of prepared.notes) if (!fresh.some((item) => item.path === note.path)) logEntries.push({ path: note.path, timestamp: new Date().toISOString(), provider: prepared.provider.id, model: prepared.provider.model, status: "cached", durationMs: 0 });
      progress.update({ phase: "cache scan", progress: 0.12, detail: `${fresh.length} notes need embeddings · ${prepared.notes.length - fresh.length} cached` });
      if (prepared.provider.id === "local" && fresh.length) {
        runStage = "preflight"; const localProvider = prepared.provider as LocalEmbeddingProvider;
        await localProvider.preflight((update) => progress.update({ phase: "preflight", progress: 0.12 + update.progress * 0.12, detail: update.detail || update.phase }), runSignal);
        runtimeDiagnostics = localProvider.runtimeDiagnostics; runStage = "embedding";
      }
      let processedFresh = 0; const embedded: CachedEmbedding[] = []; let embeddingStreamError: unknown;
      const onEmbeddingProgress = (done: number, total: number) => progress.update({ phase: "embedding", progress: 0.25 + (total ? done / total * 0.5 : 0), detail: `${done}/${total} notes processed` });
      const onEmbeddingNote = (entry: EmbeddingLogEntry) => { logEntries.push(entry); processedFresh++; progress.update({ phase: "embedding", progress: 0.25 + processedFresh / Math.max(1, fresh.length) * 0.5, detail: `${entry.status === "failure" ? "Failed" : "Embedded"}: ${entry.path}` }); };
      if (fresh.length) {
        if (prepared.provider.embedBatches) { try { for await (const batch of prepared.provider.embedBatches(fresh, onEmbeddingProgress, onEmbeddingNote, runSignal)) embedded.push(...batch); } catch (error) { embeddingStreamError = error; } }
        else embedded.push(...await prepared.provider.embed(fresh, onEmbeddingProgress, onEmbeddingNote, runSignal));
      }
      if (embedded.length) { await sqlite.putEmbeddings(embedded); for (const item of embedded) entries.set(`${item.provider}:${item.model}:${item.path}`, item); }
      if (embeddingStreamError) throw embeddingStreamError;
      const missingEmbeddings = fresh.filter((note) => !entries.get(`${prepared.provider.id}:${prepared.provider.model}:${note.path}`)?.hash || entries.get(`${prepared.provider.id}:${prepared.provider.model}:${note.path}`)!.hash !== note.hash);
      if (missingEmbeddings.length) throw new Error(`Embedding failed for ${missingEmbeddings.length} changed note${missingEmbeddings.length === 1 ? "" : "s"}; the previous cluster result was preserved.`);
      persistedRunLog = { version: 1, startedAt, completedAt: new Date().toISOString(), provider: prepared.provider.id, model: prepared.provider.model, total: prepared.notes.length, ...counts(), entries: logEntries, status: "completed", stage: "embedding", ...(runtimeDiagnostics ? { runtime: runtimeDiagnostics } : {}) };
      await sqlite.saveEmbeddingLog(persistedRunLog);
      progress.update({ phase: "cache save", progress: 0.78, detail: `${persistedRunLog.succeeded} embedded · ${persistedRunLog.failed} failed · ${persistedRunLog.cached} cached` });
      const vectorsByPath = new Map<string, number[]>(); for (const note of prepared.notes) { const cachedEntry = cached(note); if (!cachedEntry) throw new Error(`Embedding cache is incomplete for ${note.path}.`); vectorsByPath.set(note.path, cachedEntry.vector); }
      const existingCoordinates = new Map(prepared.existingCoordinates); const existingUmapCoordinates = new Map(prepared.existingUmapCoordinates);
      for (const [oldPath, newPath] of appliedRenames) { const coordinate = existingCoordinates.get(oldPath); if (coordinate) existingCoordinates.set(newPath, coordinate); existingCoordinates.delete(oldPath); }
      for (const [oldPath, newPath] of appliedRenames) { const coordinate = existingUmapCoordinates.get(oldPath); if (coordinate) existingUmapCoordinates.set(newPath, coordinate); existingUmapCoordinates.delete(oldPath); }
      const changedPaths = new Set([...prepared.changedPaths].map((path) => appliedRenames.get(path) || path));
      const deletedPaths = new Set(prepared.deletedPaths); for (const oldPath of appliedRenames.keys()) deletedPaths.delete(oldPath);
      if (currentResult.pca.model) currentResult = { ...currentResult, pca: { ...currentResult.pca, model: { ...currentResult.pca.model, provider: prepared.provider.id, model: prepared.provider.model } } };
      const soft = buildSoftRefresh({ result: currentResult, notes: prepared.notes, vectorsByPath, existingCoordinates, existingUmapCoordinates, changedPaths, deletedPaths, provider: prepared.provider.id, model: prepared.provider.model });
      if (soft.projectedPaths.length && soft.result.pca.model) await sqlite.projectMany(soft.projectedPaths.map((path) => ({ path, vector: vectorsByPath.get(path)! })), soft.result.pca.model);
      const resultId = await sqlite.saveResult(soft.result, { noteHashes: prepared.noteHashes }); this.latestResultId = resultId; this.setLatestResult(soft.result); await this.publishResult(soft.result);
      persistedRunLog = { ...persistedRunLog, completedAt: new Date().toISOString(), stage: "clustering" }; await sqlite.saveEmbeddingLog(persistedRunLog);
      if (soft.result.incremental?.fullRebuildRecommended) this.scheduleFullRebuild();
      progress.complete(`Refreshed ${changedPaths.size} notes`);
    } catch (error) {
      progress.fail(`Incremental refresh failed: ${safeRunError(error)}`);
      const failedLog: EmbeddingRunLog = persistedRunLog ? { ...persistedRunLog, completedAt: new Date().toISOString(), status: runSignal.aborted ? "cancelled" : "failed", error: safeRunError(error) } : { version: 1, startedAt, completedAt: new Date().toISOString(), provider: prepared.provider.id, model: prepared.provider.model, total: prepared.notes.length, ...counts(), entries: logEntries, status: runSignal.aborted ? "cancelled" : "failed", stage: runStage, error: safeRunError(error) };
      await sqlite.saveEmbeddingLog(failedLog).catch(() => undefined); throw error;
    } finally { this.running = false; this.operationProgress = null; this.runAbortController = null; }
  }

  private async pauseAutomaticRefresh(): Promise<void> {
    if (this.settings.automaticRefresh === false) { new Notice("Automatic refresh is already paused."); return; }
    this.settings.automaticRefresh = false; await this.saveSettings(); await this.configureAutomaticRefresh();
    new Notice("Automatic refresh paused. Enable it again in Settings when ready.");
  }

  private async getWorker(): Promise<NodeClusteringWorker | BrowserClusteringWorker | InProcessClusteringWorker> {
    if (!this.worker) {
      const nodeWorker = new NodeClusteringWorker(workerSource); this.worker = nodeWorker;
      try { await nodeWorker.init(); }
      catch (error) {
        await nodeWorker.terminate().catch(() => undefined);
        const browserWorker = new BrowserClusteringWorker(browserWorkerSource);
        try { await browserWorker.init(); this.worker = browserWorker; new Notice(`Node worker unavailable; using Chromium worker fallback: ${error instanceof Error ? error.message : String(error)}`); }
        catch (browserError) {
          await browserWorker.terminate().catch(() => undefined); const fallback = new InProcessClusteringWorker(); this.worker = fallback; await fallback.init();
          new Notice(`Worker APIs unavailable; using in-process fallback: ${browserError instanceof Error ? browserError.message : String(browserError)}`);
        }
      }
    }
    return this.worker;
  }
  private cancelClustering(): void { if (!this.running) { new Notice("No cluster operation is running."); return; } this.runAbortController?.abort(); this.operationProgress?.fail("Cancellation requested"); this.worker?.cancel(); new Notice("Cluster operation cancellation requested."); }
  private async ensureVisualization(result: ClusterResult): Promise<ClusterResult> {
    if (result.visualization) return result;
    const modelHash = result.pca.model?.modelHash; if (!modelHash) throw new Error("Saved PCA model is unavailable; rebuild clusters to create the Explorer projection.");
    const requestedRevision = this.resultRevision; const requestedResult = this.latestResult; let pending = this.visualizationPromises.get(result.ids);
    if (!pending) {
      pending = (async () => {
        const sqlite = await this.getSqliteStore(); const stored = await sqlite.getPcaCoordinatesMany(result.ids, modelHash);
        if (stored.some((row) => !row)) throw new Error("Saved PCA coordinates are incomplete; rebuild clusters to create the Explorer projection.");
        const projected = await projectVisualization(stored as number[][], result.leafLabels, { seed: 42 });
        if (!projected) throw new Error("Explorer projection requires at least three saved PCA rows.");
        return { ...projected, leafOrdering: (result.leafOrdering || result.leafOrder || result.hierarchy.leaves).slice(), memberships: (result.memberships || result.softMemberships || []).map((row) => row.slice()) };
      })();
      this.visualizationPromises.set(result.ids, pending); pending.catch(() => this.visualizationPromises.delete(result.ids));
    }
    const visualization = await pending; const current = this.latestResult; const sameResult = requestedResult === result && current === result && this.resultRevision === requestedRevision; const completed = { ...(sameResult && current ? current : result), visualization };
    if (sameResult) {
      const sqlite = await this.getSqliteStore(); const resultId = this.latestResultId || await sqlite.getLatestResultId();
      if (this.latestResult !== result || this.resultRevision !== requestedRevision) return completed;
      if (resultId) { await sqlite.patchResultVisualization(resultId, visualization); if (this.latestResult !== result || this.resultRevision !== requestedRevision) return completed; this.latestResultId = resultId; }
      this.setLatestResult(completed);
    }
    return completed;
  }
  private async openExplorer(): Promise<void> { const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER); const leaf = leaves[0] || this.app.workspace.getRightLeaf(false); if (!leaf) return; await leaf.setViewState({ type: VIEW_TYPE_CLUSTER_EXPLORER, active: true }); this.app.workspace.revealLeaf(leaf); if (!this.latestResult) { const loaded = await (await this.getSqliteStore()).getResult(); if (loaded) this.setLatestResult(loaded); } if (this.latestResult) (leaf.view as ClusterExplorerView).setResult(this.latestResult); }
  private async openEmbeddingLog(): Promise<void> {
    try { const log = await (await this.getSqliteStore()).loadLatestEmbeddingLog(); if (!log) { new Notice("No embedding log is available yet."); return; } const modal = new Modal(this.app); modal.titleEl.setText("Embedding log"); modal.contentEl.createEl("pre", { text: JSON.stringify(log, null, 2) }); modal.open(); }
    catch (error) { new Notice(`Could not open embedding log: ${safeRunError(error instanceof Error ? error.message : String(error))}`); }
  }
  private setLatestResult(result: ClusterResult): void { this.latestResult = result; this.resultRevision++; }
  private async publishResult(result: ClusterResult): Promise<void> { try { this.explorerSearchNotes = await (await this.getSqliteStore()).listNotes(true); } catch { } for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER)) { const view = leaf.view as ClusterExplorerView; view.setSearchNotes(this.explorerSearchNotes); view.setResult(result); } }
  private publishPendingChangeCount(): void { const count = this.pendingVaultChanges?.size || 0; for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER)) (leaf.view as ClusterExplorerView).setPendingChangeCount(count); }
  private updateProgress(phase: string, progress: number): void { this.app.workspace.getLeavesOfType(VIEW_TYPE_CLUSTER_EXPLORER).forEach((leaf) => { const view = leaf.view as ClusterExplorerView; view.contentEl.setAttribute("aria-label", `${phase} ${Math.round(progress * 100)}%`); view.setProgress(phase, progress); }); }
  private async saveSettings(): Promise<void> { await this.saveData(this.settings); }
  private secretResolver(): SecretResolver { const storage = (this.app as unknown as { secretStorage?: { getSecret?: (reference: string) => string | null } }).secretStorage; return { getSecret: async (reference) => storage?.getSecret?.(reference) || null }; }
  private confirmGeminiTransmission(count: number): Promise<boolean> { return new Promise((resolve) => new GeminiTransmissionModal(this.app, count, resolve).open()); }
}

function safeRunError(error: unknown): string {
  const messages: string[] = []; let current: unknown = error;
  for (let depth = 0; current && depth < 4; depth++) { messages.push(current instanceof Error ? current.message : String(current)); current = (current as { cause?: unknown })?.cause; }
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
