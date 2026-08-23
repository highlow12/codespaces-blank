import { ItemView, WorkspaceLeaf } from "obsidian";
import { ClusterResult } from "./types";

export const VIEW_TYPE_CLUSTER_EXPLORER = "atomic-clusters-explorer";

export class ClusterExplorerView extends ItemView {
  private result: ClusterResult | null = null;
  private progress: { phase: string; value: number } | null = null;
  constructor(leaf: WorkspaceLeaf) { super(leaf); }
  getViewType(): string { return VIEW_TYPE_CLUSTER_EXPLORER; }
  getDisplayText(): string { return "Atomic Clusters"; }
  async onOpen(): Promise<void> { this.render(); }
  async onClose(): Promise<void> { this.contentEl.empty(); }
  setResult(result: ClusterResult): void { this.result = result; this.render(); }
  setProgress(phase: string, value: number): void { this.progress = { phase, value }; this.render(); }
  private render(): void {
    this.contentEl.empty(); this.contentEl.addClass("atomic-clusters-view");
    this.contentEl.createEl("h3", { text: "Atomic Clusters" });
    if (this.progress && this.progress.value < 1) { this.contentEl.createDiv({ text: `${this.progress.phase} · ${Math.round(this.progress.value * 100)}%` }).addClass("atomic-clusters-status"); const bar = this.contentEl.createDiv({ cls: "atomic-clusters-progress" }); bar.createEl("span").style.width = `${Math.round(this.progress.value * 100)}%`; if (!this.result) return; }
    if (!this.result) { this.contentEl.createDiv({ text: "No clustering result yet. Run Build note clusters." }).addClass("atomic-clusters-status"); return; }
    this.contentEl.createDiv({ text: `${this.result.hierarchy.leaves.length} leaf clusters · ${this.result.hierarchy.merges.length} hierarchy merges · PCA ${this.result.pca.selected} dimensions` }).addClass("atomic-clusters-status");
    const list = this.contentEl.createDiv();
    const merges = new Map(this.result.hierarchy.merges.map((merge) => [merge.id, merge]));
    const renderNode = (id: number, parent: HTMLElement, depth: number): void => {
      const merge = merges.get(id);
      if (!merge) {
        const node = parent.createDiv({ cls: "atomic-clusters-node" }); node.createEl("strong", { text: `Leaf cluster ${id}` });
        const files = this.result!.ids.filter((_, index) => this.result!.leafLabels[index] === id).slice(0, 3);
        const representative = node.createDiv({ text: files.length ? "Representative notes:" : "No representative notes" });
        representative.addClass("atomic-clusters-status");
        for (const path of files) node.createEl("button", { text: path, attr: { type: "button" } }).addEventListener("click", () => void this.app.workspace.openLinkText(path, "", false));
        return;
      }
      const details = parent.createEl("details", { cls: "atomic-clusters-node" }) as HTMLDetailsElement; details.open = depth === 0;
      details.createEl("summary", { text: `Merge ${merge.id} · distance ${merge.distance.toFixed(3)}` });
      renderNode(merge.left, details, depth + 1); renderNode(merge.right, details, depth + 1);
    };
    if (this.result.hierarchy.root !== null) renderNode(this.result.hierarchy.root, list, 0);
    else list.createDiv({ text: "No non-noise clusters found." }).addClass("atomic-clusters-status");
  }
}
