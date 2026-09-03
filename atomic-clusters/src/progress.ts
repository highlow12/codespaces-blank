import { Notice } from "obsidian";
import { buildProgressHeartbeatDetail, PROGRESS_HEARTBEAT_INTERVAL_MS } from "./progress-heartbeat";

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
  private lastUpdateAt = 0;
  private lastProgress = Number.NEGATIVE_INFINITY;
  private lastPhase = "";
  private readonly startedAt = Date.now();
  private lastDetail = "";
  private heartbeatTimer: number | null = null;
  private hideTimer: number | null = null;

  constructor(title: string) {
    this.notice = new Notice("", 0);
    this.notice.messageEl.empty();
    this.notice.messageEl.addClass("atomic-clusters-progress-notice");
    this.labelEl = this.notice.messageEl.createDiv({ cls: "atomic-clusters-progress-label", text: title });
    this.detailEl = this.notice.messageEl.createDiv({ cls: "atomic-clusters-progress-detail" });
    this.barEl = this.notice.messageEl.createEl("progress", { cls: "atomic-clusters-progress-bar", attr: { max: "1", value: "0" } });
    this.lastUpdateAt = this.startedAt;
    this.heartbeatTimer = window.setInterval(() => this.updateHeartbeat(), PROGRESS_HEARTBEAT_INTERVAL_MS);
  }

  update(update: ProgressUpdate): void {
    if (this.disposed) return;
    const value = Math.max(0, Math.min(1, update.progress));
    const now = Date.now();
    if (update.phase === this.lastPhase && value < 1 && value - this.lastProgress < 0.01 && now - this.lastUpdateAt < 100) return;
    this.lastUpdateAt = now; this.lastProgress = value; this.lastPhase = update.phase;
    this.lastDetail = update.detail || "";
    this.labelEl.setText(`${update.phase} · ${Math.round(value * 100)}%`);
    this.detailEl.setText(this.lastDetail);
    this.barEl.value = value;
  }

  complete(message: string): void {
    if (this.disposed) return;
    this.stopHeartbeat(); this.clearHideTimer();
    this.labelEl.setText(message);
    this.detailEl.setText("");
    this.barEl.value = 1;
    this.hideTimer = window.setTimeout(() => this.hide(), 1600);
  }

  fail(message: string): void {
    if (this.disposed) return;
    this.stopHeartbeat(); this.clearHideTimer();
    this.labelEl.setText(message);
    this.detailEl.setText("");
    this.barEl.remove();
    this.hideTimer = window.setTimeout(() => this.hide(), 5000);
  }

  hide(): void {
    if (this.disposed) return;
    this.stopHeartbeat(); this.clearHideTimer();
    this.disposed = true;
    this.notice.hide();
  }

  private updateHeartbeat(): void {
    if (this.disposed) return;
    const now = Date.now();
    this.detailEl.setText(buildProgressHeartbeatDetail(this.lastDetail, now - this.startedAt, now - this.lastUpdateAt));
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer === null) return;
    window.clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = null;
  }

  private clearHideTimer(): void {
    if (this.hideTimer === null) return;
    window.clearTimeout(this.hideTimer);
    this.hideTimer = null;
  }
}
