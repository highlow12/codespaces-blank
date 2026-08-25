import * as Obsidian from "obsidian";
import { App, Modal, Plugin, PluginSettingTab, Setting } from "obsidian";
import { LocalModelManager } from "./embedding";
import { PluginSettings } from "./types";

export class AtomicClustersSettingTab extends PluginSettingTab {
  constructor(app: App, plugin: Plugin, private readonly settings: PluginSettings, private readonly save: () => Promise<void>, private readonly localModels: LocalModelManager) { super(app, plugin); }
  display(): void {
    const { containerEl } = this; containerEl.empty(); containerEl.createEl("h2", { text: "Atomic Clusters" });
    new Setting(containerEl).setName("Embedding provider").setDesc("Gemini sends note text to Google; Local keeps inference on this device.").addDropdown((dropdown) => dropdown.addOption("gemini", "Gemini API").addOption("local", "Local multilingual-e5-small").setValue(this.settings.embeddingProvider).onChange(async (value) => { this.settings.embeddingProvider = value as PluginSettings["embeddingProvider"]; await this.save(); this.display(); }));
    this.renderSecretStorageControl(containerEl);
    new Setting(containerEl).setName("Gemini model").addText((text) => text.setValue(this.settings.geminiModel).onChange(async (value) => { this.settings.geminiModel = value.trim() || "gemini-embedding-2"; await this.save(); }));
    new Setting(containerEl).setName("HDBSCAN minimum samples").setDesc("Core-distance neighbourhood size (default: 3). Lower values retain finer leaf clusters.").addText((text) => text.setValue(String(this.settings.minSamples)).onChange(async (value) => { const parsed = Number.parseInt(value, 10); this.settings.minSamples = Number.isSafeInteger(parsed) && parsed >= 1 ? parsed : 3; await this.save(); }));
    new Setting(containerEl).setName("Clustering runtime").setDesc("Pyodide runs the Python reference API in a browser worker and loads its runtime on first use.").addDropdown((dropdown) => dropdown.addOption("wasm", "WASM (default)").addOption("pyodide", "Pyodide Python reference").setValue(this.settings.clusteringRuntime || "wasm").onChange(async (value) => { this.settings.clusteringRuntime = value as PluginSettings["clusteringRuntime"]; await this.save(); }));
    if (this.settings.clusteringRuntime === "pyodide") new Setting(containerEl).setName("Pyodide URL").setDesc("Optional pyodide.js URL. Leave blank to use the pinned CDN runtime.").addText((text) => text.setValue(this.settings.pyodideUrl || "").setPlaceholder("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js").onChange(async (value) => { this.settings.pyodideUrl = value.trim(); await this.save(); }));
    new Setting(containerEl).setName("Excluded folders").setDesc("Comma-separated vault-relative folder paths.").addText((text) => text.setValue(this.settings.excludedFolders.join(", ")).onChange(async (value) => { this.settings.excludedFolders = value.split(",").map((item) => item.trim()).filter(Boolean); await this.save(); }));
    if (this.settings.embeddingProvider === "local") {
      const modelSetting = new Setting(containerEl).setName("Local multilingual-e5-small").setDesc("Download once with explicit consent; inference uses the installed files without network access.");
      modelSetting.addButton((button) => button.setButtonText("Check model").onClick(async () => { button.setDisabled(true); const status = await this.localModels.status(); button.setButtonText(status === "installed" ? "Installed" : status === "corrupt" ? "Corrupt" : "Missing"); button.setDisabled(false); }));
      modelSetting.addButton((button) => button.setButtonText("Download").onClick(async () => { button.setDisabled(true); try { await this.localModels.downloadModel(() => confirmLocalModelDownload(this.app)); button.setButtonText("Installed"); } catch (error) { button.setButtonText("Download"); if (!(error instanceof Error && error.message.includes("cancelled"))) throw error; } finally { button.setDisabled(false); } }));
      modelSetting.addButton((button) => button.setButtonText("Delete").onClick(async () => { button.setDisabled(true); await this.localModels.deleteModel(); button.setDisabled(false); }));
    }
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
