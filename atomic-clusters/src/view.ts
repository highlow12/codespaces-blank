import { ItemView, WorkspaceLeaf } from "obsidian";
import { ClusterResult } from "./types";
import { findNearestVisualizationPoint, scaleVisualizationPoints, visualizationColor } from "./visualization";

export const VIEW_TYPE_CLUSTER_EXPLORER = "atomic-clusters-explorer";

export class ClusterExplorerView extends ItemView {
  private result: ClusterResult | null = null;
  private progress: { phase: string; value: number } | null = null;
  private visualizationPoints: [number, number][] = [];
  private visualizationHitElements: HTMLButtonElement[] = [];
  private visualizationResizeObserver: ResizeObserver | null = null;
  private visualizationCleanup: (() => void) | null = null;
  private hoveredVisualizationPoint: number | null = null;
  private hoveredVisualizationTarget: HTMLElement | null = null;
  constructor(leaf: WorkspaceLeaf) { super(leaf); }
  getViewType(): string { return VIEW_TYPE_CLUSTER_EXPLORER; }
  getDisplayText(): string { return "Atomic Clusters"; }
  async onOpen(): Promise<void> { this.render(); }
  async onClose(): Promise<void> { this.disposeVisualization(); this.contentEl.empty(); }
  setResult(result: ClusterResult): void { this.result = result; this.render(); }
  setProgress(phase: string, value: number): void { this.progress = { phase, value }; this.render(); }
  private render(): void {
    this.disposeVisualization(); this.contentEl.empty(); this.contentEl.addClass("atomic-clusters-view");
    const header = this.contentEl.createDiv({ cls: "atomic-clusters-view-header" });
    header.createEl("h3", { text: "Atomic Clusters" });
    if (this.progress && this.progress.value < 1) { header.createDiv({ text: `${this.progress.phase} · ${Math.round(this.progress.value * 100)}%` }).addClass("atomic-clusters-status"); const bar = header.createDiv({ cls: "atomic-clusters-progress" }); bar.createEl("span").style.width = `${Math.round(this.progress.value * 100)}%`; if (!this.result) return; }
    if (!this.result) { this.contentEl.createDiv({ text: "No clustering result yet. Run Build note clusters." }).addClass("atomic-clusters-status"); return; }
    header.createDiv({ text: `${this.result.hierarchy.leaves.length} leaf clusters · ${this.result.hierarchy.merges.length} hierarchy merges · PCA ${this.result.pca.selected} dimensions` }).addClass("atomic-clusters-status");
    this.renderVisualization();
    const tree = this.contentEl.createEl("details", { cls: "atomic-clusters-tree-panel" });
    tree.createEl("summary", { text: `Cluster hierarchy · ${this.result.hierarchy.leaves.length} leaves` });
    const list = tree.createDiv({ cls: "atomic-clusters-tree" }); const merges = new Map(this.result.hierarchy.merges.map((merge) => [merge.id, merge]));
    const renderNode = (id: number, parent: HTMLElement, depth: number): void => {
      const merge = merges.get(id);
      if (!merge) {
        const node = parent.createDiv({ cls: "atomic-clusters-node" }); node.createEl("strong", { text: `Leaf cluster ${id}` }); const title = this.result!.titles?.[String(id)]; node.querySelector("strong")?.setText(`${title ? `${title} · ` : ""}Leaf ${id}`);
        const files = this.result!.ids.map((path, index) => ({ path, index })).filter((item) => this.result!.leafLabels[item.index] === id).sort((a, b) => (this.result!.probabilities[b.index] || 0) - (this.result!.probabilities[a.index] || 0) || a.path.localeCompare(b.path)).slice(0, 3).map((item) => item.path);
        node.createDiv({ text: files.length ? "Representative notes:" : "No representative notes" }).addClass("atomic-clusters-status"); for (const path of files) node.createEl("button", { text: path, attr: { type: "button" } }).addEventListener("click", () => void this.app.workspace.openLinkText(path, "", false)); return;
      }
      const details = parent.createEl("details", { cls: "atomic-clusters-node" }) as HTMLDetailsElement; details.open = depth === 0; details.createEl("summary", { text: `${this.result!.titles?.[String(merge.id)] ? `${this.result!.titles![String(merge.id)]} · ` : ""}Merge ${merge.id} · distance ${merge.distance.toFixed(3)}` }); renderNode(merge.left, details, depth + 1); renderNode(merge.right, details, depth + 1);
    };
    if (this.result.hierarchy.root !== null) renderNode(this.result.hierarchy.root, list, 0); else list.createDiv({ text: "No non-noise clusters found." }).addClass("atomic-clusters-status");
  }
  private renderVisualization(): void {
    const visualization = this.result?.visualization;
    if (!visualization) { this.contentEl.createDiv({ text: "UMAP visualization unavailable for this saved result. Rebuild clusters to generate it." }).addClass("atomic-clusters-status"); return; }
    const valid = visualization.coordinates.length === this.result!.ids.length && visualization.labels.length === this.result!.ids.length && visualization.coordinates.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)) && visualization.labels.every(Number.isFinite);
    if (!valid) { this.contentEl.createDiv({ text: "UMAP visualization unavailable for this saved result. Rebuild clusters to generate it." }).addClass("atomic-clusters-status"); return; }
    const frame = this.contentEl.createDiv({ cls: "atomic-clusters-umap" }); const canvas = frame.createEl("canvas", { cls: "atomic-clusters-umap-canvas" }); canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "UMAP cluster visualization. Hover a note to preview it; click to open it.");
    const hitLayer = frame.createDiv({ cls: "atomic-clusters-umap-hit-layer" });
    this.visualizationHitElements = this.result!.ids.map((path, index) => {
      const hit = hitLayer.createEl("button", { cls: "atomic-clusters-umap-point-hit", attr: { type: "button", "aria-label": path, "data-point-index": String(index) } });
      hit.addEventListener("mouseenter", (event) => this.setHoveredVisualizationPoint(index, event, hit, draw));
      hit.addEventListener("click", (event) => { event.preventDefault(); void this.app.workspace.openLinkText(path, "", false); });
      return hit;
    });
    const draw = (): void => {
      const rect = canvas.getBoundingClientRect(); const width = Math.max(1, rect.width || frame.clientWidth || 320); const height = Math.max(1, rect.height || 280); const dpr = Math.max(1, typeof window === "undefined" ? 1 : window.devicePixelRatio || 1); canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
      const context = canvas.getContext("2d"); if (!context) return; context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, width, height); this.visualizationPoints = scaleVisualizationPoints(visualization.coordinates, width, height);
      const computedStyle = typeof getComputedStyle === "function" ? getComputedStyle(frame) : null;
      const backgroundColor = computedStyle?.getPropertyValue("--background-primary").trim() || "transparent";
      const textColor = computedStyle?.color || "currentColor";
      this.visualizationPoints.forEach(([x, y], index) => {
        const radius = index === this.hoveredVisualizationPoint ? 6 : 4;
        context.beginPath(); context.fillStyle = visualizationColor(visualization.labels[index]); context.globalAlpha = index === this.hoveredVisualizationPoint ? 1 : 0.86; context.arc(x, y, radius, 0, Math.PI * 2); context.fill();
        context.globalAlpha = 1; context.strokeStyle = backgroundColor; context.lineWidth = 1; context.stroke();
      });
      context.globalAlpha = 1;
      this.visualizationHitElements.forEach((hit, index) => { const point = this.visualizationPoints[index]; if (!point) return; hit.style.left = `${point[0]}px`; hit.style.top = `${point[1]}px`; });
      if (this.hoveredVisualizationPoint !== null) { const point = this.visualizationPoints[this.hoveredVisualizationPoint]; if (point) { context.beginPath(); context.strokeStyle = textColor; context.lineWidth = 1.5; context.arc(point[0], point[1], 10, 0, Math.PI * 2); context.stroke(); } }
    };
    const onMove = (event: MouseEvent): void => {
      const rect = canvas.getBoundingClientRect(); const point = findNearestVisualizationPoint(this.visualizationPoints, event.clientX - rect.left, event.clientY - rect.top, 14);
      const target = event.target instanceof HTMLElement && event.target.classList.contains("atomic-clusters-umap-point-hit") ? event.target : canvas;
      this.setHoveredVisualizationPoint(point, event, target, draw);
    };
    const onLeave = (): void => { this.setHoveredVisualizationPoint(null, null, null, draw); };
    frame.addEventListener("mousemove", onMove); frame.addEventListener("mouseleave", onLeave); this.visualizationCleanup = () => { frame.removeEventListener("mousemove", onMove); frame.removeEventListener("mouseleave", onLeave); this.visualizationHitElements = []; };
    if (typeof ResizeObserver === "function") { this.visualizationResizeObserver = new ResizeObserver(draw); this.visualizationResizeObserver.observe(frame); } draw();
  }
  private setHoveredVisualizationPoint(point: number | null, event: MouseEvent | null, target: HTMLElement | null, draw: () => void): void {
    if (point === this.hoveredVisualizationPoint && target === this.hoveredVisualizationTarget) return;
    this.hoveredVisualizationPoint = point; this.hoveredVisualizationTarget = target; draw();
    if (point !== null && event && this.result?.ids[point]) this.app.workspace.trigger("hover-link", { event, source: VIEW_TYPE_CLUSTER_EXPLORER, hoverParent: this.leaf, targetEl: target || this.visualizationHitElements[point], linktext: this.result.ids[point], sourcePath: "" });
  }
  private disposeVisualization(): void { this.visualizationCleanup?.(); this.visualizationCleanup = null; this.visualizationResizeObserver?.disconnect(); this.visualizationResizeObserver = null; this.visualizationPoints = []; this.visualizationHitElements = []; this.hoveredVisualizationPoint = null; this.hoveredVisualizationTarget = null; }
}
