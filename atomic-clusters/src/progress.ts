import { Notice } from "obsidian";

export interface ProgressUpdate {
  phase: string;
  progress: number;
  detail?: string;
}

/** One persistent native Notice for a long-running plugin operation. */
export class AtomicClustersProgress {
  private readonly notice: Notice;
  private readonly labelEl: HTMLElement;
  private readonly detailEl: HTMLElement;
  private readonly barEl: HTMLProgressElement;
  private disposed = false;

  constructor(title: string) {
    this.notice = new Notice("", 0);
    this.notice.messageEl.empty();
    this.notice.messageEl.addClass("atomic-clusters-progress-notice");
    this.labelEl = this.notice.messageEl.createDiv({ cls: "atomic-clusters-progress-label", text: title });
    this.detailEl = this.notice.messageEl.createDiv({ cls: "atomic-clusters-progress-detail" });
    this.barEl = this.notice.messageEl.createEl("progress", { cls: "atomic-clusters-progress-bar", attr: { max: "1", value: "0" } });
  }

  update(update: ProgressUpdate): void {
    if (this.disposed) return;
    const value = Math.max(0, Math.min(1, update.progress));
    this.labelEl.setText(`${update.phase} · ${Math.round(value * 100)}%`);
    this.detailEl.setText(update.detail || "");
    this.barEl.value = value;
  }

  complete(message: string): void {
    if (this.disposed) return;
    this.labelEl.setText(message);
    this.detailEl.setText("");
    this.barEl.value = 1;
    window.setTimeout(() => this.hide(), 1600);
  }

  fail(message: string): void {
    if (this.disposed) return;
    this.labelEl.setText(message);
    this.detailEl.setText("");
    this.barEl.remove();
    window.setTimeout(() => this.hide(), 5000);
  }

  hide(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.notice.hide();
  }
}
