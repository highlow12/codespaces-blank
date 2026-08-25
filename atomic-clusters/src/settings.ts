import * as Obsidian from "obsidian";
import { App, Modal, Plugin, PluginSettingTab, Setting } from "obsidian";
import { LocalModelManager, LocalModelProgress, LocalRuntimeProgress } from "./embedding";
import { PluginSettings } from "./types";

export interface ClusterRunControls {
  build(): Promise<void>;
  cancel(): void;
  isRunning(): boolean;
}

export type LocalRuntimeTest = (onProgress: (progress: LocalRuntimeProgress) => void) => Promise<void>;

export class AtomicClustersSettingTab extends PluginSettingTab {
  constructor(app: App, plugin: Plugin, private readonly settings: PluginSettings, private readonly save: () => Promise<void>, private readonly localModels: LocalModelManager, private readonly openEmbeddingLog: () => Promise<void>, private readonly clusterRun: ClusterRunControls, private readonly testLocalRuntime: LocalRuntimeTest) { super(app, plugin); }
  display(): void {
    const { containerEl } = this; containerEl.empty(); containerEl.createEl("h2", { text: "Atomic Clusters" });
    new Setting(containerEl).setName("Embedding provider").setDesc("Gemini sends note text to Google; Local keeps inference on this device.").addDropdown((dropdown) => dropdown.addOption("gemini", "Gemini API").addOption("local", "Local multilingual-e5-small").setValue(this.settings.embeddingProvider).onChange(async (value) => { this.settings.embeddingProvider = value as PluginSettings["embeddingProvider"]; await this.save(); this.display(); }));
    this.renderSecretStorageControl(containerEl);
    new Setting(containerEl).setName("Gemini model").addText((text) => text.setValue(this.settings.geminiModel).onChange(async (value) => { this.settings.geminiModel = value.trim() || "gemini-embedding-2"; await this.save(); }));
    new Setting(containerEl).setName("HDBSCAN minimum samples").setDesc("Core-distance neighbourhood size (default: 3). Lower values retain finer leaf clusters.").addText((text) => text.setValue(String(this.settings.minSamples)).onChange(async (value) => { const parsed = Number.parseInt(value, 10); this.settings.minSamples = Number.isSafeInteger(parsed) && parsed >= 1 ? parsed : 3; await this.save(); }));
    new Setting(containerEl).setName("Clustering runtime").setDesc("Pyodide runs the Python reference API in a browser worker and loads its runtime on first use.").addDropdown((dropdown) => dropdown.addOption("wasm", "WASM (default)").addOption("pyodide", "Pyodide Python reference").setValue(this.settings.clusteringRuntime || "wasm").onChange(async (value) => { this.settings.clusteringRuntime = value as PluginSettings["clusteringRuntime"]; await this.save(); }));
    if (this.settings.clusteringRuntime === "pyodide") new Setting(containerEl).setName("Pyodide URL").setDesc("Optional pyodide.js URL. Leave blank to use the pinned CDN runtime.").addText((text) => text.setValue(this.settings.pyodideUrl || "").setPlaceholder("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js").onChange(async (value) => { this.settings.pyodideUrl = value.trim(); await this.save(); }));
    new Setting(containerEl).setName("Excluded folders").setDesc("Comma-separated vault-relative folder paths.").addText((text) => text.setValue(this.settings.excludedFolders.join(", ")).onChange(async (value) => { this.settings.excludedFolders = value.split(",").map((item) => item.trim()).filter(Boolean); await this.save(); }));
    this.renderClusterRunControl(containerEl);
    if (this.settings.embeddingProvider === "local") {
      new Setting(containerEl).setName("Local execution backend").setDesc("Auto tries WebGPU first and falls back to the WASM CPU backend when unavailable. WebGPU can be faster but depends on the graphics driver.").addDropdown((dropdown) => dropdown.addOption("auto", "Auto (WebGPU → WASM CPU)").addOption("webgpu", "WebGPU only").addOption("wasm", "WASM CPU only").setValue(this.settings.localExecutionProvider || "auto").onChange(async (value) => { this.settings.localExecutionProvider = value as PluginSettings["localExecutionProvider"]; await this.save(); }));
      const modelSetting = new Setting(containerEl).setName("Local multilingual-e5-small").setDesc("Download once with explicit consent; inference uses the installed files without network access.");
      const statusSetting = new Setting(containerEl).setName("Local model status").setDesc("Installation progress and integrity status.");
      const progress = statusSetting.controlEl.createEl("progress", { cls: "atomic-clusters-model-progress", attr: { max: "1", value: "0" } });
      const statusEl = statusSetting.controlEl.createDiv({ cls: "atomic-clusters-model-status", text: "Checking…" });
      const controls: Array<{ setDisabled(disabled: boolean): unknown }> = [];
      const setBusy = (busy: boolean) => controls.forEach((control) => control.setDisabled(busy));
      const updateProgress = (update: LocalModelProgress) => { progress.value = Math.max(0, Math.min(1, update.progress)); const bytes = update.loadedBytes !== undefined && update.totalBytes ? ` · ${formatBytes(update.loadedBytes)}/${formatBytes(update.totalBytes)}` : ""; statusEl.setText(`${update.phase} · ${Math.round(update.progress * 100)}%${update.detail ? ` · ${update.detail}` : ""}${bytes}`); };
      const updateRuntimeProgress = (update: LocalRuntimeProgress) => { progress.value = Math.max(0, Math.min(1, update.progress)); statusEl.setText(`runtime ${update.phase} · ${Math.round(update.progress * 100)}%${update.detail ? ` · ${update.detail}` : ""}`); };
      const refresh = async () => { try { const status = await this.localModels.status(); statusEl.setText(status === "installed" ? "Installed and integrity verified" : status === "corrupt" ? "Corrupt — delete and download again" : "Missing — download to enable offline inference"); progress.value = status === "installed" ? 1 : 0; } catch (error) { statusEl.setText(`Status check failed: ${safeUiError(error)}`); } };
      let checkButton: { setDisabled(disabled: boolean): unknown };
      modelSetting.addButton((button) => { checkButton = button; return button.setButtonText("Check model").onClick(async () => { setBusy(true); try { await refresh(); } finally { setBusy(false); } }); }); controls.push(checkButton!);
      let downloadButton: { setDisabled(disabled: boolean): unknown };
      modelSetting.addButton((button) => { downloadButton = button; return button.setButtonText("Download").onClick(async () => { setBusy(true); try { await this.localModels.downloadModel(() => confirmLocalModelDownload(this.app), updateProgress); } catch (error) { statusEl.setText(error instanceof Error && error.message.includes("cancelled") ? "Download cancelled" : `Download failed: ${safeUiError(error)}`); } finally { setBusy(false); await refresh(); } }); }); controls.push(downloadButton!);
      let deleteButton: { setDisabled(disabled: boolean): unknown };
      modelSetting.addButton((button) => { deleteButton = button; return button.setButtonText("Delete").onClick(async () => { setBusy(true); try { await this.localModels.deleteModel(); statusEl.setText("Deleted"); progress.value = 0; } catch (error) { statusEl.setText(`Delete failed: ${safeUiError(error)}`); } finally { setBusy(false); } }); }); controls.push(deleteButton!);
      let testButton: { setDisabled(disabled: boolean): unknown };
      modelSetting.addButton((button) => { testButton = button; return button.setButtonText("Test local runtime").onClick(async () => { setBusy(true); try { await this.testLocalRuntime(updateRuntimeProgress); statusEl.setText("Local runtime ready for offline inference"); progress.value = 1; } catch (error) { statusEl.setText(`Runtime test failed: ${safeUiError(error)}`); } finally { setBusy(false); } }); }); controls.push(testButton!);
      void refresh();
    }
    new Setting(containerEl).setName("Embedding log").setDesc("Open the latest per-note embedding diagnostics in the operating system's default text editor.").addButton((button) => button.setButtonText("Open embedding log").onClick(() => { void this.openEmbeddingLog().catch(() => undefined); }));
  }

  private renderClusterRunControl(containerEl: HTMLElement): void {
    const setting = new Setting(containerEl).setName("Build clusters").setDesc("Run the configured embedding and clustering pipeline. Progress remains visible in one persistent Notice while the job is running.");
    const statusEl = setting.controlEl.createDiv({ cls: "atomic-clusters-model-status", text: "Ready" });
    let buildButton: { setDisabled(disabled: boolean): unknown; setButtonText(text: string): unknown } | undefined;
    let cancelButton: { setDisabled(disabled: boolean): unknown } | undefined;
    const refresh = () => {
      const running = this.clusterRun.isRunning();
      if (buildButton) { buildButton.setDisabled(running); buildButton.setButtonText(running ? "Building…" : "Build clusters"); }
      cancelButton?.setDisabled(!running);
      statusEl.setText(running ? "Running — see the persistent progress Notice." : "Ready");
    };
    setting.addButton((button) => { buildButton = button; return button.setButtonText("Build clusters").onClick(() => { if (this.clusterRun.isRunning()) return; try { const run = this.clusterRun.build(); refresh(); void run.catch((error) => { statusEl.setText(`Build failed: ${safeUiError(error)}`); }).finally(refresh); } catch (error) { statusEl.setText(`Build failed: ${safeUiError(error)}`); refresh(); } }); });
    setting.addButton((button) => { cancelButton = button; return button.setButtonText("Cancel").setDisabled(true).onClick(() => { if (!this.clusterRun.isRunning()) return; this.clusterRun.cancel(); refresh(); }); });
    refresh();
  }

  private renderSecretStorageControl(containerEl: HTMLElement): void {
    const SecretCtor = (Obsidian as unknown as { SecretComponent?: new (app: App, el: HTMLElement, key: string) => { setValue?: (value: string) => unknown; onChange?: (callback: (value: string) => void) => unknown } }).SecretComponent;
    if (SecretCtor) {
      const wrapper = containerEl.createDiv(); wrapper.createEl("label", { text: "Gemini API key (Obsidian SecretStorage)" });
      try { const secret = new SecretCtor(this.app, wrapper, this.settings.geminiSecretRef || "gemini-api-key"); secret.onChange?.(() => { void this.save(); }); return; } catch { wrapper.empty(); }
    }
    new Setting(containerEl).setName("Gemini SecretStorage reference").setDesc("Name of the secret stored in Obsidian SecretStorage; never paste the key here.").addText((text) => text.setPlaceholder("gemini-api-key").setValue(this.settings.geminiSecretRef).onChange(async (value) => { this.settings.geminiSecretRef = value.trim() || "gemini-api-key"; await this.save(); }));
  }
}

function formatBytes(bytes: number): string { if (bytes < 1024) return `${bytes} B`; if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`; if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`; return `${(bytes / 1024 ** 3).toFixed(1)} GB`; }
function safeUiError(error: unknown): string { return (error instanceof Error ? error.message : String(error)).replace(/((?:^|[?&\s])(?:key|token|secret|authorization)=)[^&\s]+/gi, "$1[redacted]").slice(0, 240); }

class LocalModelConsentModal extends Modal {
  constructor(app: App, private readonly resolveConsent: (value: boolean) => void) { super(app); }
  onOpen(): void {
    this.contentEl.createEl("h3", { text: "Download local embedding model?" });
    this.contentEl.createEl("p", { text: "This downloads multilingual-e5-small and its tokenizer from Hugging Face. The model is stored in the plugin model directory and note text is not sent over the network during inference." });
    const row = this.contentEl.createDiv();
    row.createEl("button", { text: "Download" }).addEventListener("click", () => { this.resolveConsent(true); this.close(); });
    row.createEl("button", { text: "Cancel" }).addEventListener("click", () => { this.resolveConsent(false); this.close(); });
  }
  onClose(): void { this.resolveConsent(false); }
}

function confirmLocalModelDownload(app: App): Promise<boolean> {
  return new Promise((resolve) => { let settled = false; const modal = new LocalModelConsentModal(app, (value) => { if (!settled) { settled = true; resolve(value); } }); modal.open(); });
}
