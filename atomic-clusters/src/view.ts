import { ItemView, WorkspaceLeaf } from "obsidian";
import { ClusterResult } from "./types";
import { accumulateVisualizationDensity, blendVisualizationColor, buildVisualizationTree, clampVisualizationKernelScale, findNearestVisualizationPoint, pickVisualizationCloud, validateVisualizationData, visualizationBaseBandwidth, visualizationCameraLayerTransform, visualizationCameraTransform, visualizationColorVector, visualizationDensityAlpha, visualizationFrontier, visualizationLeafOrdering, visualizationMembershipAmplitude, visualizationPath, visualizationP95RowSum, visualizationRegion, visualizationScaledStageSigma, visualizationWorldToScreen, VisualizationCamera, VisualizationNode, VisualizationSplat, VISUALIZATION_KERNEL_SCALE_DEFAULT, VISUALIZATION_KERNEL_SCALE_MAX, VISUALIZATION_KERNEL_SCALE_MIN, VISUALIZATION_KERNEL_SCALE_STEP } from "./visualization";

export const VIEW_TYPE_CLUSTER_EXPLORER = "atomic-clusters-explorer";
const VISUALIZATION_CAMERA_TRANSITION_MS = 280;

export class ClusterExplorerView extends ItemView {
  private static visualizationControlsCounter = 0;
  private result: ClusterResult | null = null;
  private progress: { phase: string; value: number } | null = null;
  private visualizationPoints: [number, number][] = [];
  private visualizationHitElements: HTMLButtonElement[] = [];
  private visualizationResizeObserver: ResizeObserver | null = null;
  private visualizationCleanup: (() => void) | null = null;
  private hoveredVisualizationPoint: number | null = null;
  private hoveredVisualizationTarget: HTMLElement | null = null;
  private visualizationNodeId = "root";
  private visualizationExpandedIds = new Set<string>();
  private visualizationRoot: VisualizationNode | null = null;
  private visualizationKernelScale = VISUALIZATION_KERNEL_SCALE_DEFAULT;
  private visualizationControlsCollapsed = false;
  private visualizationLastCamera: VisualizationCamera | null = null;
  private visualizationDisplayedCamera: VisualizationCamera | null = null;
  private visualizationTransition: { fromCamera: VisualizationCamera } | null = null;
  private visualizationAnimationFrame: number | null = null;
  private visualizationAnimationToken = 0;
  private visualizationAnimating = false;
  private visualizationAnimationTarget: VisualizationCamera | null = null;
  private readonly visualizationControlsId = ++ClusterExplorerView.visualizationControlsCounter;
  private readonly rebuildClusters?: () => void | Promise<void>;
  constructor(leaf: WorkspaceLeaf, rebuildClusters?: () => void | Promise<void>) { super(leaf); this.rebuildClusters = rebuildClusters; }
  getViewType(): string { return VIEW_TYPE_CLUSTER_EXPLORER; }
  getDisplayText(): string { return "Atomic Clusters"; }
  async onOpen(): Promise<void> { this.render(); }
  async onClose(): Promise<void> { this.disposeVisualization(); this.contentEl.empty(); }
  setResult(result: ClusterResult): void { this.cancelVisualizationAnimation(false); this.result = result; this.visualizationNodeId = "root"; this.visualizationExpandedIds.clear(); this.visualizationTransition = null; this.visualizationLastCamera = null; this.visualizationDisplayedCamera = null; this.render(); }
  setProgress(phase: string, value: number): void { this.progress = { phase, value }; this.render(); }
  private render(): void {
    this.disposeVisualization(); this.contentEl.empty(); this.contentEl.addClass("atomic-clusters-view");
    const header = this.contentEl.createDiv({ cls: "atomic-clusters-view-header" }); header.createEl("h3", { text: "Atomic Clusters" });
    if (this.progress && this.progress.value < 1) { header.createDiv({ text: `${this.progress.phase} · ${Math.round(this.progress.value * 100)}%` }).addClass("atomic-clusters-status"); const bar = header.createDiv({ cls: "atomic-clusters-progress" }); bar.createEl("span").style.width = `${Math.round(this.progress.value * 100)}%`; if (!this.result) return; }
    if (!this.result) { this.contentEl.createDiv({ text: "No clustering result yet. Run Build note clusters." }).addClass("atomic-clusters-status"); return; }
    header.createDiv({ text: `${this.result.hierarchy.leaves.length} leaf clusters · ${this.result.hierarchy.merges.length} hierarchy merges · PCA ${this.result.pca.selected} dimensions` }).addClass("atomic-clusters-status");
    this.renderVisualization();
    const tree = this.contentEl.createEl("details", { cls: "atomic-clusters-tree-panel" }); tree.createEl("summary", { text: `Cluster hierarchy · ${this.result.hierarchy.leaves.length} leaves` });
    const list = tree.createDiv({ cls: "atomic-clusters-tree" }); const adapterRoot = buildVisualizationTree(this.result.hierarchy, this.result.leafLabels);
    const renderNode = (item: VisualizationNode, parent: HTMLElement, depth: number): void => { if (!item.children.length) { const id = item.sourceId ?? -1; const node = parent.createDiv({ cls: "atomic-clusters-node" }); const title = this.result!.titles?.[String(id)]; node.createEl("strong", { text: `${title ? `${title} · ` : ""}Leaf ${id}` }); const files = this.result!.ids.map((path, index) => ({ path, index })).filter((entry) => this.result!.leafLabels[entry.index] === id).sort((a, b) => (this.result!.probabilities[b.index] || 0) - (this.result!.probabilities[a.index] || 0) || a.path.localeCompare(b.path)).slice(0, 3).map((entry) => entry.path); node.createDiv({ text: files.length ? "Representative notes:" : "No representative notes" }).addClass("atomic-clusters-status"); for (const path of files) node.createEl("button", { text: path, attr: { type: "button" } }).addEventListener("click", () => void this.app.workspace.openLinkText(path, "", false)); return; } const details = parent.createEl("details", { cls: "atomic-clusters-node" }) as HTMLDetailsElement; details.open = depth === 0; const id = item.sourceId; details.createEl("summary", { text: `${id === null ? "All notes" : `${this.result!.titles?.[String(id)] ? `${this.result!.titles![String(id)]} · ` : ""}Merge ${id}`}` }); item.children.forEach((child) => renderNode(child, details, depth + 1)); };
    if (adapterRoot.children.length) adapterRoot.children.forEach((child) => renderNode(child, list, 0)); else list.createDiv({ text: "No non-noise clusters found." }).addClass("atomic-clusters-status");
  }
  private renderVisualization(): void {
    const result = this.result!; const visualization = result.visualization; const ordering = visualizationLeafOrdering(result);
    if (!validateVisualizationData(result) || !visualization) { this.contentEl.createDiv({ text: "Hierarchical Gaussian-cloud visualization is unavailable for this saved result. Rebuild clusters to create a v4 result with soft memberships." }).addClass("atomic-clusters-status"); const rebuild = this.contentEl.createEl("button", { text: "Rebuild clusters", attr: { type: "button" } }); rebuild.addEventListener("click", () => void this.rebuildClusters?.()); return; }
    const coordinates = visualization.coordinates; const memberships = result.memberships!; const labels = visualization.labels; this.visualizationRoot = buildVisualizationTree(result.hierarchy, labels); // Global-world baseline retained for compatibility: scaleVisualizationPoints(visualization.coordinates, width, height)
    const findNode = (node: VisualizationNode): VisualizationNode | null => node.id === this.visualizationNodeId ? node : node.children.reduce<VisualizationNode | null>((found, child) => found || findNode(child), null); const current = findNode(this.visualizationRoot) || this.visualizationRoot; this.visualizationNodeId = current.id;
    const frontier = visualizationFrontier(this.visualizationRoot, this.visualizationExpandedIds, memberships, ordering); const frame = this.contentEl.createDiv({ cls: "atomic-clusters-umap" }); const plot = frame.createDiv({ cls: "atomic-clusters-umap-plot" }); const navigation = plot.createDiv({ cls: "atomic-clusters-umap-navigation" }); const path = visualizationPath(this.visualizationRoot, current.id);
    const back = navigation.createEl("button", { text: "Back", attr: { type: "button", "aria-label": "Back to parent cluster" } }); back.disabled = path.length <= 1; back.addEventListener("click", () => { if (path.length > 1) { const parent = path[path.length - 2]; this.navigateVisualization(parent.id, () => { this.visualizationExpandedIds.delete(current.id); if (parent.id === "root") this.visualizationExpandedIds.delete("root"); }); } });
    const crumbs = navigation.createDiv({ cls: "atomic-clusters-breadcrumb" }); path.forEach((node, index) => { if (index) crumbs.createSpan({ text: " / " }); const title = node.sourceId === null ? "All notes" : result.titles?.[String(node.sourceId)]; const crumb = crumbs.createEl("button", { text: title || `Cluster ${node.sourceId}`, attr: { type: "button", "aria-current": index === path.length - 1 ? "page" : "false" } }); crumb.addEventListener("click", () => { this.navigateVisualization(node.id, () => { const allowed = new Set(visualizationPath(this.visualizationRoot!, node.id).map((item) => item.id)); this.visualizationExpandedIds = node.id === "root" ? new Set() : new Set([...this.visualizationExpandedIds].filter((id) => allowed.has(id))); if (!node.children.length) this.visualizationExpandedIds.add(node.id); }); }); });
    // Canvas and hit targets share one transformed layer. Controls/navigation
    // remain siblings so they never zoom or blur during camera movement.
    const visualLayer = plot.createDiv({ cls: "atomic-clusters-umap-visual-layer" });
    const canvas = visualLayer.createEl("canvas", { cls: "atomic-clusters-umap-canvas" }); canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "Hierarchical Gaussian cloud visualization. Click a cloud to zoom; hover a note to preview it; click a note to open it."); const hitLayer = visualLayer.createDiv({ cls: "atomic-clusters-umap-hit-layer" });
    // Keep all notes in the global UMAP world and fit only the selected node in the camera.
    const region = visualizationRegion(current, coordinates); const baseSigma = visualizationBaseBandwidth(coordinates); const p95 = visualizationP95RowSum(memberships); let cachedKey = ""; let cachedClouds: VisualizationSplat[][] = []; let cachedBitmap: HTMLCanvasElement | null = null; const pendingTransition = this.visualizationTransition; this.visualizationTransition = null;
    const rectSize = (): [number, number] => { const rect = canvas.getBoundingClientRect(); return [Math.max(1, rect.width || plot.clientWidth || 320), Math.max(1, rect.height || plot.clientHeight || 280)]; };
    let renderedCamera: VisualizationCamera | null = null;
    const clearLayerTransform = (): void => { visualLayer.style.transform = "none"; visualLayer.style.opacity = ""; visualLayer.classList.remove("is-animating"); };
    const draw = (): void => {
      const [width, height] = rectSize(); const dpr = Math.max(1, typeof window === "undefined" ? 1 : window.devicePixelRatio || 1); canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); const context = canvas.getContext("2d"); if (!context) return; context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, width, height);
      const camera = visualizationCameraTransform(region, width, height); renderedCamera = camera; const points = coordinates.map((point) => visualizationWorldToScreen(camera, point)); this.visualizationPoints = points; this.visualizationLastCamera = camera;
      const kernelScale = clampVisualizationKernelScale(this.visualizationKernelScale); this.visualizationKernelScale = kernelScale;
      const key = `${current.id}|${[...this.visualizationExpandedIds].sort().join(",")}|${Math.round(width)}x${Math.round(height)}|${baseSigma}|${kernelScale}`;
      if (key !== cachedKey) { cachedKey = key; cachedBitmap = null; const longAxis = Math.max(width, height); const rasterLong = Math.max(256, Math.min(512, Math.round(longAxis * .5))); const rasterScale = rasterLong / longAxis; const rasterWidth = Math.max(1, Math.round(width * rasterScale)); const rasterHeight = Math.max(1, Math.round(height * rasterScale)); cachedClouds = []; const allDots = new Set<number>();
        while (hitLayer.firstChild) hitLayer.removeChild(hitLayer.firstChild); for (const entry of frontier) { const isLeaf = !entry.node.children.length; const splats: VisualizationSplat[] = []; for (const index of entry.pointIndices) { const point = points[index]; const row = memberships[index] || []; if (!point) continue; const color = visualizationColorVector(row, ordering); splats.push({ x: point[0] * rasterScale, y: point[1] * rasterScale, sigma: visualizationScaledStageSigma(baseSigma, entry.remainingDepth, isLeaf, kernelScale) * camera.scale * rasterScale, color, amplitude: visualizationMembershipAmplitude(row, p95) }); } if (!entry.actualPoints) cachedClouds.push(splats); else for (const index of entry.pointIndices) allDots.add(index); for (const index of entry.residualIndices) allDots.add(index); }
        // A residual can be adjacent to multiple children, but is represented by one dot only.
        this.visualizationHitElements = [...allDots].sort((a, b) => a - b).map((index) => { const hit = hitLayer.createEl("button", { cls: "atomic-clusters-umap-point-hit", attr: { type: "button", "aria-label": result.ids[index], "data-point-index": String(index) } }); hit.addEventListener("mouseenter", (event) => this.setHoveredVisualizationPoint(index, event, hit, draw)); hit.addEventListener("click", (event) => { event.preventDefault(); void this.app.workspace.openLinkText(result.ids[index], "", false); }); return hit; });
        const field = accumulateVisualizationDensity(cachedClouds.flat(), rasterWidth, rasterHeight); const bitmap = typeof document === "undefined" ? null : document.createElement("canvas"); if (bitmap) { bitmap.width = rasterWidth; bitmap.height = rasterHeight; const bitmapContext = bitmap.getContext("2d"); if (bitmapContext) { const image = bitmapContext.createImageData(rasterWidth, rasterHeight); for (let offset = 0; offset < field.density.length; offset++) { const density = field.density[offset]; const alpha = visualizationDensityAlpha(density); image.data[offset * 4] = density > 0 ? Math.round(field.red[offset] / density) : 0; image.data[offset * 4 + 1] = density > 0 ? Math.round(field.green[offset] / density) : 0; image.data[offset * 4 + 2] = density > 0 ? Math.round(field.blue[offset] / density) : 0; image.data[offset * 4 + 3] = Math.round(alpha * 255); } bitmapContext.putImageData(image, 0, 0); cachedBitmap = bitmap; } }
      }
      if (cachedBitmap) { context.imageSmoothingEnabled = true; context.drawImage(cachedBitmap, 0, 0, width, height); }
      const computedStyle = typeof getComputedStyle === "function" ? getComputedStyle(frame) : null; const background = computedStyle?.getPropertyValue("--background-primary").trim() || "transparent"; const dotIndices = this.visualizationHitElements.map((hit) => Number(hit.dataset.pointIndex)); dotIndices.forEach((index) => { const point = points[index]; if (!point) return; const radius = index === this.hoveredVisualizationPoint ? 6 : 4; context.beginPath(); context.fillStyle = blendVisualizationColor(memberships[index], ordering); context.globalAlpha = index === this.hoveredVisualizationPoint ? 1 : .9; context.arc(point[0], point[1], radius, 0, Math.PI * 2); context.fill(); context.globalAlpha = 1; context.strokeStyle = background; context.stroke(); });
      this.visualizationHitElements.forEach((hit, index) => { const point = points[dotIndices[index]]; if (point) { hit.style.left = `${point[0]}px`; hit.style.top = `${point[1]}px`; } }); if (this.hoveredVisualizationPoint !== null) { const point = points[this.hoveredVisualizationPoint]; if (point) { context.beginPath(); context.strokeStyle = computedStyle?.color || "currentColor"; context.lineWidth = 1.5; context.arc(point[0], point[1], 10, 0, Math.PI * 2); context.stroke(); } } context.globalAlpha = 1;
    };
    const controlsBodyId = `atomic-clusters-umap-controls-body-${this.visualizationControlsId}`; const scaleInputId = `atomic-clusters-kernel-scale-${this.visualizationControlsId}`;
    const controls = frame.createDiv({ cls: "atomic-clusters-umap-controls", attr: { "aria-label": "Visualization adjustments" } });
    const toggle = controls.createEl("button", { cls: "atomic-clusters-umap-controls-toggle", text: this.visualizationControlsCollapsed ? "Show adjustments" : "Hide adjustments", attr: { type: "button", "aria-expanded": String(!this.visualizationControlsCollapsed), "aria-controls": controlsBodyId } });
    const controlsBody = controls.createDiv({ cls: "atomic-clusters-umap-controls-body" }); controlsBody.id = controlsBodyId;
    controlsBody.createEl("label", { text: "Gaussian kernel size", attr: { for: scaleInputId } });
    const scaleRow = controlsBody.createDiv({ cls: "atomic-clusters-umap-control-row" });
    const scaleInput = scaleRow.createEl("input", { attr: { id: scaleInputId, type: "range", min: String(VISUALIZATION_KERNEL_SCALE_MIN), max: String(VISUALIZATION_KERNEL_SCALE_MAX), step: String(VISUALIZATION_KERNEL_SCALE_STEP), value: String(this.visualizationKernelScale), "aria-label": "Gaussian kernel size multiplier" } });
    const scaleOutput = scaleRow.createEl("output", { text: `${this.visualizationKernelScale.toFixed(2)}×`, attr: { for: scaleInputId, "aria-live": "polite" } });
    if (this.visualizationControlsCollapsed) controls.addClass("is-collapsed");
    toggle.addEventListener("click", () => { this.visualizationControlsCollapsed = !this.visualizationControlsCollapsed; controls.toggleClass("is-collapsed", this.visualizationControlsCollapsed); toggle.setAttribute("aria-expanded", String(!this.visualizationControlsCollapsed)); toggle.setText(this.visualizationControlsCollapsed ? "Show adjustments" : "Hide adjustments"); });
    scaleInput.addEventListener("input", () => { this.visualizationKernelScale = clampVisualizationKernelScale(Number(scaleInput.value)); scaleOutput.setText(`${this.visualizationKernelScale.toFixed(2)}×`); cachedKey = ""; cachedBitmap = null; draw(); });
    frame.addEventListener("click", (event) => { if (this.visualizationAnimating || !(event.target instanceof HTMLCanvasElement)) return; const rect = canvas.getBoundingClientRect(); const x = event.clientX - rect.left, y = event.clientY - rect.top; const longAxis = Math.max(rect.width, rect.height); const rasterScale = Math.max(256, Math.min(512, Math.round(longAxis * .5))) / longAxis; const picked = pickVisualizationCloud(cachedClouds, x * rasterScale, y * rasterScale); if (picked !== null) { const cloudEntries = frontier.filter((entry) => !entry.actualPoints); const target = cloudEntries[picked]; if (target) { const targetId = target.node.id === "root" && this.visualizationRoot?.children.length === 1 ? this.visualizationRoot.children[0].id : target.node.id; this.navigateVisualization(targetId, () => { this.visualizationExpandedIds.add(target.node.id); }); } } });
    const onMove = (event: MouseEvent): void => { if (this.visualizationAnimating) return; const rect = canvas.getBoundingClientRect(); const dotIndices = this.visualizationHitElements.map((hit) => Number(hit.dataset.pointIndex)); const visible = dotIndices.map((index) => this.visualizationPoints[index]); const nearestVisible = findNearestVisualizationPoint(visible, event.clientX - rect.left, event.clientY - rect.top, 14); const index = nearestVisible === null ? null : dotIndices[nearestVisible]; const target = event.target instanceof HTMLElement && event.target.classList.contains("atomic-clusters-umap-point-hit") ? event.target : canvas; this.setHoveredVisualizationPoint(index, event, target, draw); }; const onLeave = (): void => this.setHoveredVisualizationPoint(null, null, null, draw); frame.addEventListener("mousemove", onMove); frame.addEventListener("mouseleave", onLeave); const onKey = (event: KeyboardEvent): void => { if (event.key === "Escape" && path.length > 1) { event.preventDefault(); const parent = path[path.length - 2]; this.navigateVisualization(parent.id, () => { this.visualizationExpandedIds.delete(current.id); if (parent.id === "root") this.visualizationExpandedIds.delete("root"); }); } }; const focusFrame = (): void => frame.focus(); frame.tabIndex = 0; frame.addEventListener("keydown", onKey); frame.addEventListener("pointerdown", focusFrame);
    this.visualizationCleanup = () => { frame.removeEventListener("mousemove", onMove); frame.removeEventListener("mouseleave", onLeave); frame.removeEventListener("keydown", onKey); frame.removeEventListener("pointerdown", focusFrame); this.visualizationHitElements = []; }; const initialPlotSize = rectSize(); let lastObservedPlotSize: [number, number] = initialPlotSize; if (typeof ResizeObserver === "function") { this.visualizationResizeObserver = new ResizeObserver(() => { const nextSize = rectSize(); const changed = nextSize[0] !== lastObservedPlotSize[0] || nextSize[1] !== lastObservedPlotSize[1]; if (!changed) return; lastObservedPlotSize = nextSize; if (this.visualizationAnimating) { this.cancelVisualizationAnimation(true); clearLayerTransform(); } draw(); }); this.visualizationResizeObserver.observe(plot); } draw();
    if (pendingTransition && renderedCamera) this.startVisualizationAnimation(visualLayer, pendingTransition.fromCamera, renderedCamera, clearLayerTransform);
  }
  private navigateVisualization(targetId: string, update: () => void): void {
    const fromCamera = this.visualizationDisplayedCamera || this.visualizationLastCamera;
    this.cancelVisualizationAnimation(false);
    this.visualizationTransition = fromCamera ? { fromCamera: { ...fromCamera, worldRegion: { ...fromCamera.worldRegion } } } : null;
    this.visualizationNodeId = targetId; update(); this.render();
  }
  private startVisualizationAnimation(layer: HTMLElement, from: VisualizationCamera, to: VisualizationCamera, clear: () => void): void {
    const sameViewport = from.width === to.width && from.height === to.height;
    const reduced = typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!sameViewport || reduced || (from.scale === to.scale && from.offsetX === to.offsetX && from.offsetY === to.offsetY)) { this.visualizationDisplayedCamera = to; clear(); return; }
    this.cancelVisualizationAnimation(false); this.visualizationAnimating = true; this.visualizationAnimationTarget = to; const token = ++this.visualizationAnimationToken; const started = typeof performance !== "undefined" ? performance.now() : Date.now();
    const initial = visualizationCameraLayerTransform(from, to, 0); layer.style.transformOrigin = "0 0"; layer.style.transform = `translate(${initial.translateX}px, ${initial.translateY}px) scale(${initial.scale})`; layer.classList.add("is-animating");
    const animationNow = (): number => typeof performance !== "undefined" ? performance.now() : Date.now();
    const requestFrame = (callback: FrameRequestCallback): number => typeof requestAnimationFrame === "function" ? requestAnimationFrame(callback) : setTimeout(() => callback(animationNow()), 16) as unknown as number;
    const tick = (now: number): void => {
      if (token !== this.visualizationAnimationToken) return;
      const progress = Math.max(0, Math.min(1, (now - started) / VISUALIZATION_CAMERA_TRANSITION_MS)); const transform = visualizationCameraLayerTransform(from, to, progress);
      layer.style.transformOrigin = "0 0"; layer.style.transform = `translate(${transform.translateX}px, ${transform.translateY}px) scale(${transform.scale})`;
      this.visualizationDisplayedCamera = { ...to, scale: to.scale * transform.scale, offsetX: transform.translateX + transform.scale * to.offsetX, offsetY: transform.translateY + transform.scale * to.offsetY };
      if (progress >= 1) { this.visualizationAnimating = false; this.visualizationAnimationFrame = null; this.visualizationAnimationTarget = null; this.visualizationDisplayedCamera = to; clear(); return; }
      this.visualizationAnimationFrame = requestFrame(tick);
    };
    this.visualizationAnimationFrame = requestFrame(tick);
  }
  private cancelVisualizationAnimation(snap: boolean): void {
    this.visualizationAnimationToken++;
    if (this.visualizationAnimationFrame !== null) { if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(this.visualizationAnimationFrame); else clearTimeout(this.visualizationAnimationFrame); this.visualizationAnimationFrame = null; }
    if (snap && this.visualizationAnimationTarget) this.visualizationDisplayedCamera = this.visualizationAnimationTarget;
    this.visualizationAnimationTarget = null; this.visualizationAnimating = false;
  }
  private setHoveredVisualizationPoint(point: number | null, event: MouseEvent | null, target: HTMLElement | null, draw: () => void): void { if (this.visualizationAnimating) return; if (point === this.hoveredVisualizationPoint && target === this.hoveredVisualizationTarget) return; this.hoveredVisualizationPoint = point; this.hoveredVisualizationTarget = target; draw(); if (point !== null && event && this.result?.ids[point]) this.app.workspace.trigger("hover-link", { event, source: VIEW_TYPE_CLUSTER_EXPLORER, hoverParent: this.leaf, targetEl: target || this.visualizationHitElements[point] || this.visualizationHitElements[0], linktext: this.result.ids[point], sourcePath: "" }); }
  private disposeVisualization(): void { this.cancelVisualizationAnimation(false); this.visualizationCleanup?.(); this.visualizationCleanup = null; this.visualizationResizeObserver?.disconnect(); this.visualizationResizeObserver = null; this.visualizationPoints = []; this.visualizationHitElements = []; this.hoveredVisualizationPoint = null; this.hoveredVisualizationTarget = null; }
}
