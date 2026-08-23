import * as Obsidian from "obsidian";
import { App, Plugin, PluginSettingTab, Setting } from "obsidian";
import { PluginSettings } from "./types";

export class AtomicClustersSettingTab extends PluginSettingTab {
  constructor(app: App, plugin: Plugin, private readonly settings: PluginSettings, private readonly save: () => Promise<void>) { super(app, plugin); }
  display(): void {
    const { containerEl } = this; containerEl.empty(); containerEl.createEl("h2", { text: "Atomic Clusters" });
    new Setting(containerEl).setName("Embedding provider").setDesc("Gemini sends note text to Google; Local keeps inference on this device.").addDropdown((dropdown) => dropdown.addOption("gemini", "Gemini API").addOption("local", "Local multilingual-e5-small").setValue(this.settings.embeddingProvider).onChange(async (value) => { this.settings.embeddingProvider = value as PluginSettings["embeddingProvider"]; await this.save(); this.display(); }));
    this.renderSecretStorageControl(containerEl);
    new Setting(containerEl).setName("Gemini model").addText((text) => text.setValue(this.settings.geminiModel).onChange(async (value) => { this.settings.geminiModel = value.trim() || "gemini-embedding-2"; await this.save(); }));
    new Setting(containerEl).setName("HDBSCAN minimum samples").setDesc("Core-distance neighbourhood size (default: 3). Lower values retain finer leaf clusters.").addText((text) => text.setValue(String(this.settings.minSamples)).onChange(async (value) => { const parsed = Number.parseInt(value, 10); this.settings.minSamples = Number.isSafeInteger(parsed) && parsed >= 1 ? parsed : 3; await this.save(); }));
    new Setting(containerEl).setName("Excluded folders").setDesc("Comma-separated vault-relative folder paths.").addText((text) => text.setValue(this.settings.excludedFolders.join(", ")).onChange(async (value) => { this.settings.excludedFolders = value.split(",").map((item) => item.trim()).filter(Boolean); await this.save(); }));
    if (this.settings.embeddingProvider === "local") new Setting(containerEl).setName("Local model").setDesc("Unavailable in this build: no ONNX runtime or model asset is bundled. Select Gemini API to build clusters.").addButton((button) => button.setButtonText("Unavailable").setDisabled(true));
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
