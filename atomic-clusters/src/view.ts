import { ItemView, WorkspaceLeaf } from "obsidian";
import { generatedClusterSnapshots } from "./sqlite-storage";
import { ClusterResult, ManualCorrectionsState, NoteRecord, normalizeVaultRelativePath } from "./types";
import { buildNoteDetail, NoteDetailModel } from "./note-detail";
import { buildSearchDocuments, SearchClusterDocument, SearchDocument, SearchFilter, SearchFilters, SearchIndex, SearchResult } from "./search";
import { accumulateVisualizationDensity, blendVisualizationColor, buildVisualizationPointSpatialIndex, buildVisualizationTree, clampVisualizationKernelScale, layoutVisualizationClusterLabels, measureVisualizationClusterLabelBoxes, pickVisualizationCloud, resizeVisualizationCameraState, validateVisualizationData, visualizationBaseBandwidth, visualizationOutgoingLayerTransform, visualizationCameraTransform, visualizationCloudColor, visualizationColorScheme, visualizationColorVector, visualizationDensityAlpha, visualizationFrontier, visualizationGlobalDepthFrontier, visualizationFitCameraState, visualizationCameraFromState, visualizationNoteTerminalPath, visualizationLeafOrdering, visualizationMembershipAmplitude, visualizationPath, visualizationP95RowSum, visualizationRegion, visualizationScaledStageSigma, visualizationTopMemberships, visualizationWorldToScreen, zoomVisualizationCameraAt, panVisualizationCamera, VisualizationCamera, VisualizationCameraState, VisualizationNode, VisualizationPointSpatialIndex, VisualizationSplat, VISUALIZATION_HOVER_POINT_RADIUS, VISUALIZATION_HOVER_RING_RADIUS, VISUALIZATION_KERNEL_SCALE_DEFAULT, VISUALIZATION_KERNEL_SCALE_MAX, VISUALIZATION_KERNEL_SCALE_MIN, VISUALIZATION_KERNEL_SCALE_STEP, VISUALIZATION_NOISE_COLOR, VISUALIZATION_POINT_RADIUS } from "./visualization";

export const VIEW_TYPE_CLUSTER_EXPLORER = "atomic-clusters-explorer";
// Keep navigation long enough for the hierarchy change to read as a camera
// move, while still feeling responsive when stepping through several levels.
const VISUALIZATION_CAMERA_TRANSITION_MS = 460;
// Allow the canvas and ResizeObserver content-box measurements to differ by
// fractional CSS pixels (or a one-pixel scrollbar/control breakpoint shift)
// without treating the first post-render notification as a real resize.
const VISUALIZATION_VIEWPORT_TOLERANCE = 1.25;
// Keep a small raster margin around the plot. During a pan/zoom gesture the
// existing bitmap is moved with a CSS affine transform; without this margin,
// splats touching the old viewport edge are permanently clipped by the
// canvas before the debounced high-quality redraw can run.
const VISUALIZATION_GESTURE_OVERSCAN = 128;

function visualizationMembershipPercent(value: number): string { return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(1)}%`; }
function formatDetailPercent(value: number | null): string { return value === null ? "Unavailable" : visualizationMembershipPercent(value); }

function createVisualizationHoverSummary(document: Document, result: ClusterResult, point: number, titles: Record<string, string> = result.titles || {}): HTMLElement {
  const summary = document.createElement("div"); summary.className = "atomic-clusters-hover-membership-summary"; summary.dataset.pointIndex = String(point);
  const heading = document.createElement("div"); heading.className = "atomic-clusters-hover-membership-heading"; heading.textContent = "Top 3 cluster memberships"; summary.appendChild(heading);
  // The generated-title call shape remains recognizable as visualizationTopMemberships(result, point, 3) for saved DOM fixtures.
  const memberships = visualizationTopMemberships({ ...result, titles }, point, 3);
  if (memberships.length) {
    const list = document.createElement("ol"); list.className = "atomic-clusters-hover-membership-list";
    memberships.forEach((item) => { const entry = document.createElement("li"); entry.textContent = `${item.title} · ${visualizationMembershipPercent(item.value)}`; list.appendChild(entry); });
    summary.appendChild(list);
  } else {
    const empty = document.createElement("div"); empty.className = "atomic-clusters-hover-membership-empty"; empty.textContent = "No cluster membership data"; summary.appendChild(empty);
  }
  const placement = result.hierarchyPlacements?.[point]; const noise = result.leafLabels[point] < 0; const residual = placement?.kind === "residual" && !noise;
  if (noise || residual) {
    const status = document.createElement("div"); status.className = "atomic-clusters-hover-membership-status";
    const kind = noise ? "Noise" : "Residual"; const location = placement?.nodeId === null || placement?.nodeId === undefined ? "root" : `node ${placement.nodeId}`; const confidence = placement ? ` · hierarchy confidence ${visualizationMembershipPercent(placement.confidence)}` : "";
    status.textContent = `${kind} · ${location}${confidence}`; summary.appendChild(status);
  }
  return summary;
}

/** Returned by the lazy projection hook: either a complete result or just its visualization. */
export type EnsureVisualizationResult = ClusterResult | ClusterResult["visualization"] | null | undefined;
export type EnsureVisualization = (result: ClusterResult) => Promise<EnsureVisualizationResult>;

export interface ClusterExplorerActions {
  renameClusterTitle?: (stableClusterKey: string, title: string, memberPaths: readonly string[]) => void | Promise<void>;
  resetClusterTitle?: (stableClusterKey: string) => void | Promise<void>;
  saveNoteClusterPreference?: (notePath: string, preferredClusterKey: string) => void | Promise<void>;
  clearNoteClusterPreference?: (notePath: string) => void | Promise<void>;
  createManualGroup?: (title: string, childClusterKeys: readonly string[]) => void | Promise<void>;
  ungroupManualGroup?: (groupId: string) => void | Promise<void>;
  recordTooBroadFeedback?: (stableClusterKey: string, message?: string) => void | Promise<void>;
}

const EMPTY_MANUAL_CORRECTIONS: ManualCorrectionsState = { titleOverrides: [], notePreferences: [], groups: [], feedback: [] };

function showCorrectionMessage(message: string): void {
  const NoticeConstructor = (globalThis as typeof globalThis & { Notice?: new (message: string) => unknown }).Notice;
  if (NoticeConstructor) new NoticeConstructor(message);
  else if (typeof console !== "undefined") console.warn(message);
}

/** Compose display-only titles from the generated result and current durable corrections. */
export function composeEffectiveClusterTitles(result: ClusterResult, manualCorrections: ManualCorrectionsState = EMPTY_MANUAL_CORRECTIONS): Record<string, string> {
  const titles = { ...(result.titles || {}) };
  const nodesByKey = new Map(generatedClusterSnapshots(result).map((snapshot) => [snapshot.stableClusterKey, snapshot.nodeId]));
  for (const override of manualCorrections.titleOverrides || []) {
    const nodeId = nodesByKey.get(override.stableClusterKey);
    if (nodeId !== undefined && !override.orphaned && override.title.trim()) titles[String(nodeId)] = override.title.trim();
  }
  return titles;
}

export class ClusterExplorerView extends ItemView {
  private static visualizationControlsCounter = 0;
  private result: ClusterResult | null = null;
  private progress: { phase: string; value: number } | null = null;
  private visualizationPoints: [number, number][] = [];
  /** Every visible note is painted on canvas; this is the complete dot set. */
  private visualizationRenderedPointIndices: number[] = [];
  private visualizationHitElements: HTMLButtonElement[] = [];
  private visualizationSpatialIndex: VisualizationPointSpatialIndex | null = null;
  private visualizationResizeObserver: ResizeObserver | null = null;
  private visualizationCleanup: (() => void) | null = null;
  private hoveredVisualizationPoint: number | null = null;
  private hoveredVisualizationTarget: HTMLElement | null = null;
  private visualizationHoverSummaryToken = 0;
  private visualizationHoverSummaryTimer: number | null = null;
  private visualizationNodeId = "root";
  private visualizationRoot: VisualizationNode | null = null;
  private visualizationKernelScale = VISUALIZATION_KERNEL_SCALE_DEFAULT;
  private visualizationControlsCollapsed = false;
  private visualizationLastCamera: VisualizationCamera | null = null;
  private visualizationDisplayedCamera: VisualizationCamera | null = null;
  private visualizationTransition: { fromCamera: VisualizationCamera; snapshot: HTMLCanvasElement } | null = null;
  private visualizationAnimationFrame: number | null = null;
  private visualizationAnimationToken = 0;
  private visualizationAnimating = false;
  private visualizationAnimationTarget: VisualizationCamera | null = null;
  private visualizationAnimationCleanup: (() => void) | null = null;
  /** Global depth and selection are semantic UI state, independent of the camera. */
  private visualizationDepth = 0;
  private visualizationSelectedNodeId: string | null = null;
  private visualizationCameraState: VisualizationCameraState | null = null;
  private readonly visualizationControlsId = ++ClusterExplorerView.visualizationControlsCounter;
  private readonly rebuildClusters?: () => void | Promise<void>;
  private readonly ensureVisualization?: EnsureVisualization;
  private pendingChangeCount = 0;
  private visualizationGenerationResult: ClusterResult | null = null;
  private visualizationGenerationIdleHandle: number | null = null;
  private visualizationGenerationCancel: (() => void) | null = null;
  private visualizationGenerationToken = 0;
  private visualizationGenerationScheduled = false;
  private visualizationGenerationInFlight = false;
  private visualizationGenerationError: string | null = null;
  private visualizationClosed = false;
  private searchNotes: NoteRecord[] = [];
  private searchDocuments: SearchDocument[] = [];
  private readonly searchIndex = new SearchIndex();
  private searchResult: SearchResult | null = null;
  private searchIndexedResult: ClusterResult | null = null;
  private searchIndexedNotes: readonly NoteRecord[] | null = null;
  private searchQuery = "";
  private searchFilters = new Set<SearchFilter>(["all"]);
  private searchInput: HTMLInputElement | null = null;
  private searchDebounceTimer: number | null = null;
  private searchActiveResultIndex = -1;
  /** Focus is Explorer navigation state; it never changes clustering output. */
  private focusNodeId: string | null = null;
  /** Selected note drives the detail panel and is independent from camera/focus state. */
  private selectedNotePath: string | null = null;
  private manualCorrections: ManualCorrectionsState = EMPTY_MANUAL_CORRECTIONS;
  private readonly correctionActions?: ClusterExplorerActions;
  private searchIndexedManualCorrections: ManualCorrectionsState | null = null;
  private groupSelection = new Set<string>();
  private explorerKeydownHandler: ((event: KeyboardEvent) => void) | null = null;
  constructor(leaf: WorkspaceLeaf, rebuildClusters?: () => void | Promise<void>, initialResult?: ClusterResult | null, ensureVisualization?: EnsureVisualization, initialNotes?: readonly NoteRecord[], initialManualCorrections?: ManualCorrectionsState, correctionActions?: ClusterExplorerActions) {
    super(leaf);
    this.rebuildClusters = rebuildClusters;
    this.ensureVisualization = ensureVisualization;
    this.searchNotes = initialNotes ? initialNotes.slice() : [];
    this.manualCorrections = initialManualCorrections || EMPTY_MANUAL_CORRECTIONS;
    this.correctionActions = correctionActions;
    // Workspace leaves are restored while the plugin is loading.  Accept the
    // already-loaded persisted result so a restored explorer can render its
    // visualization during its first onOpen instead of briefly showing the
    // empty-state message until the user opens it manually.
    this.result = initialResult || null;
  }
  getViewType(): string { return VIEW_TYPE_CLUSTER_EXPLORER; }
  getDisplayText(): string { return "Atomic Clusters"; }
  async onOpen(): Promise<void> {
    this.visualizationClosed = false;
    this.explorerKeydownHandler = (event) => this.handleExplorerKeydown(event);
    this.contentEl.addEventListener("keydown", this.explorerKeydownHandler);
    this.render();
  }
  async onClose(): Promise<void> {
    this.visualizationClosed = true;
    this.invalidateVisualizationGeneration();
    this.disposeVisualization();
    if (this.searchDebounceTimer !== null) globalThis.clearTimeout(this.searchDebounceTimer);
    this.searchDebounceTimer = null;
    if (this.explorerKeydownHandler) this.contentEl.removeEventListener("keydown", this.explorerKeydownHandler);
    this.explorerKeydownHandler = null;
    this.searchInput = null;
    this.contentEl.empty();
  }
  setResult(result: ClusterResult): void {
    if (this.result !== result) this.invalidateVisualizationGeneration();
    this.cancelVisualizationAnimation(false);
    this.result = result;
    this.visualizationNodeId = "root";
    this.visualizationSelectedNodeId = null;
    this.visualizationDepth = 0;
    this.focusNodeId = null;
    this.visualizationCameraState = null;
    this.visualizationTransition = null;
    this.visualizationLastCamera = null;
    this.visualizationDisplayedCamera = null;
    if (this.selectedNotePath && !result.ids.includes(this.selectedNotePath)) this.selectedNotePath = null;
    this.render();
  }
  setSearchNotes(notes: readonly NoteRecord[]): void { this.searchNotes = notes.slice(); if (this.result) this.render(); }
  setManualCorrections(corrections: ManualCorrectionsState): void {
    this.manualCorrections = corrections || EMPTY_MANUAL_CORRECTIONS;
    if (this.result) {
      const currentKeys = new Set(generatedClusterSnapshots(this.result).map((snapshot) => snapshot.stableClusterKey));
      this.groupSelection = new Set([...this.groupSelection].filter((key) => currentKeys.has(key)));
    }
    this.searchIndexedManualCorrections = null;
    if (this.result) this.render();
  }
  setPendingChangeCount(count: number): void { const next = Math.max(0, Math.trunc(count)); if (next === this.pendingChangeCount) return; this.pendingChangeCount = next; if (this.result) this.render(); }
  setProgress(phase: string, value: number): void { this.progress = { phase, value }; this.render(); }
  private effectiveTitles(result: ClusterResult): Record<string, string> {
    return composeEffectiveClusterTitles(result, this.manualCorrections);
  }
  private snapshotForNode(result: ClusterResult, nodeId: number): ReturnType<typeof generatedClusterSnapshots>[number] | undefined {
    return generatedClusterSnapshots(result).find((snapshot) => snapshot.nodeId === nodeId);
  }
  private activeTitleOverride(stableClusterKey: string): ManualCorrectionsState["titleOverrides"][number] | undefined {
    return this.manualCorrections.titleOverrides.find((override) => override.stableClusterKey === stableClusterKey && !override.orphaned);
  }
  private promptText(message: string, value: string): string | null {
    const prompt = (globalThis as typeof globalThis & { prompt?: (message?: string, defaultValue?: string) => string | null }).prompt;
    return typeof prompt === "function" ? prompt(message, value) : null;
  }
  private runCorrectionAction(action: (() => void | Promise<void>) | undefined, failureMessage: string): void {
    if (!action) { showCorrectionMessage("Manual corrections are unavailable until the plugin storage is ready."); return; }
    void Promise.resolve().then(action).catch((error: unknown) => showCorrectionMessage(`${failureMessage}: ${error instanceof Error ? error.message : String(error)}`));
  }
  private renameCluster(nodeId: number): void {
    if (!this.result) return;
    const snapshot = this.snapshotForNode(this.result, nodeId); if (!snapshot) return;
    const current = this.effectiveTitles(this.result)[String(nodeId)] || `Cluster ${nodeId}`;
    const title = this.promptText("Rename cluster", current);
    if (title === null) return;
    const action = this.correctionActions?.renameClusterTitle;
    this.runCorrectionAction(action ? () => action(snapshot.stableClusterKey, title, snapshot.memberPaths) : undefined, "Could not rename cluster");
  }
  private resetClusterTitle(nodeId: number): void {
    if (!this.result) return;
    const snapshot = this.snapshotForNode(this.result, nodeId); if (!snapshot) return;
    const action = this.correctionActions?.resetClusterTitle;
    this.runCorrectionAction(action ? () => action(snapshot.stableClusterKey) : undefined, "Could not reset cluster title");
  }
  private toggleGroupSelection(stableClusterKey: string): void {
    if (this.groupSelection.has(stableClusterKey)) this.groupSelection.delete(stableClusterKey); else this.groupSelection.add(stableClusterKey);
    this.render();
  }
  private createManualGroup(): void {
    const keys = [...this.groupSelection];
    if (keys.length < 2) { showCorrectionMessage("Select at least two clusters before creating a manual group."); return; }
    const title = this.promptText("Name this manual group", "");
    if (title === null || !title.trim()) return;
    const action = this.correctionActions?.createManualGroup;
    this.runCorrectionAction(action ? () => action(title.trim(), keys) : undefined, "Could not create manual group");
    this.groupSelection.clear();
  }
  private recordTooBroad(stableClusterKey: string): void {
    const action = this.correctionActions?.recordTooBroadFeedback;
    this.runCorrectionAction(action ? () => action(stableClusterKey, "This cluster is too broad") : undefined, "Could not record feedback");
  }
  private clearNotePreference(path: string): void {
    const action = this.correctionActions?.clearNoteClusterPreference;
    this.runCorrectionAction(action ? () => action(path) : undefined, "Could not clear preferred cluster");
  }
  private saveNotePreference(path: string, stableClusterKey: string): void {
    const action = this.correctionActions?.saveNoteClusterPreference;
    this.runCorrectionAction(action ? () => action(path, stableClusterKey) : undefined, "Could not save preferred cluster");
  }
  private renderCorrectionActions(parent: HTMLElement, nodeId: number, stableClusterKey: string): void {
    const actions = parent.createDiv({ cls: "atomic-clusters-correction-actions" });
    const rename = actions.createEl("button", { text: "Rename title", attr: { type: "button", "data-action": "rename-title" } }); rename.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); this.renameCluster(nodeId); });
    const reset = actions.createEl("button", { text: "Reset title", attr: { type: "button", "data-action": "reset-title" } }); reset.disabled = !this.activeTitleOverride(stableClusterKey); reset.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); this.resetClusterTitle(nodeId); });
    const select = actions.createEl("button", { text: this.groupSelection.has(stableClusterKey) ? "Remove from group" : "Select for group", attr: { type: "button", "data-action": "group-select", "aria-pressed": String(this.groupSelection.has(stableClusterKey)) } }); select.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); this.toggleGroupSelection(stableClusterKey); });
    const broad = actions.createEl("button", { text: "Too broad", attr: { type: "button", "data-action": "too-broad" } }); broad.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); this.recordTooBroad(stableClusterKey); });
  }
  private renderManualGroupPanel(parent: HTMLElement): void {
    const groups = this.manualCorrections.groups || [];
    if (!groups.length && !this.groupSelection.size) return;
    const panel = parent.createDiv({ cls: "atomic-clusters-manual-groups", attr: { "aria-label": "Manual cluster groups" } });
    const heading = panel.createDiv({ cls: "atomic-clusters-manual-groups-heading" }); heading.createEl("strong", { text: "Manual groups" });
    if (this.groupSelection.size) {
      heading.createSpan({ text: `${this.groupSelection.size} selected` });
      const create = heading.createEl("button", { text: "Create manual group", attr: { type: "button" } }); create.disabled = this.groupSelection.size < 2; create.addEventListener("click", () => this.createManualGroup());
    }
    for (const group of groups) {
      const row = panel.createDiv({ cls: "atomic-clusters-manual-group", attr: { "data-group-id": group.groupId, tabindex: "-1", "aria-current": "false", "aria-label": `Manual group ${group.title}` } });
      const childTitles = group.childClusterKeys.map((key) => {
        const snapshot = this.result ? generatedClusterSnapshots(this.result).find((candidate) => candidate.stableClusterKey === key) : undefined;
        return snapshot ? this.effectiveTitles(this.result!)[String(snapshot.nodeId)] || `Cluster ${snapshot.nodeId}` : `${key} (orphaned)`;
      });
      row.createEl("strong", { text: group.title }); row.createDiv({ text: childTitles.join(" · ") || "No current child clusters" }).addClass("atomic-clusters-status");
      const ungroup = row.createEl("button", { text: "Ungroup", attr: { type: "button", "aria-label": `Ungroup ${group.title}` } }); const action = this.correctionActions?.ungroupManualGroup; ungroup.addEventListener("click", () => this.runCorrectionAction(action ? () => action(group.groupId) : undefined, "Could not ungroup manual group"));
    }
  }
  private isSearchActive(): boolean { return !!this.searchQuery.trim() || [...this.searchFilters].some((filter) => filter !== "all"); }
  private currentSearchFilters(): SearchFilters {
    return {
      currentClusterId: this.searchFilters.has("current-cluster") && this.focusNodeId ? this.clusterKeyForNodeId(this.focusNodeId) : null,
      noise: this.searchFilters.has("noise"),
      provisional: this.searchFilters.has("provisional"),
      manuallyAdjusted: this.searchFilters.has("manually-adjusted"),
      recentlyChanged: this.searchFilters.has("recently-changed"),
    };
  }
  private refreshSearchIndex(): void {
    if (!this.result) {
      if (this.searchIndexedResult !== null || this.searchIndexedNotes !== this.searchNotes) {
        this.searchDocuments = [];
        this.searchIndex.replace([], []);
        this.searchIndexedResult = null;
        this.searchIndexedNotes = this.searchNotes;
      }
      this.searchResult = this.searchIndex.search(this.searchQuery, this.currentSearchFilters());
      return;
    }
    if (this.searchIndexedResult !== this.result || this.searchIndexedNotes !== this.searchNotes) {
      const built = buildSearchDocuments(this.searchNotes, this.result, this.manualCorrections);
      this.searchDocuments = built.documents;
      this.searchIndex.replace(built.documents, built.clusters);
      this.searchIndexedResult = this.result;
      this.searchIndexedNotes = this.searchNotes;
      this.searchIndexedManualCorrections = this.manualCorrections;
    } else if (this.searchIndexedManualCorrections !== this.manualCorrections) {
      const built = buildSearchDocuments(this.searchNotes, this.result, this.manualCorrections);
      this.searchDocuments = built.documents;
      this.searchIndex.replace(built.documents, built.clusters);
      this.searchIndexedManualCorrections = this.manualCorrections;
    }
    this.searchResult = this.searchIndex.search(this.searchQuery, this.currentSearchFilters());
  }
  private clusterKeyForNodeId(nodeId: string): string { return nodeId === "root" ? "root" : nodeId.replace(/^node:/, ""); }
  private isFilterActive(filter: SearchFilter): boolean { return this.searchFilters.has(filter); }
  private setSearchFilter(filter: SearchFilter): void {
    if (filter === "all") this.searchFilters = new Set(["all"]);
    else {
      const next = new Set(this.searchFilters); next.delete("all");
      if (next.has(filter)) next.delete(filter); else next.add(filter);
      this.searchFilters = next.size ? next : new Set(["all"]);
    }
    this.searchActiveResultIndex = -1;
    this.render();
  }
  private scheduleSearchRender(focus = true): void {
    if (this.searchDebounceTimer !== null) globalThis.clearTimeout(this.searchDebounceTimer);
    this.searchDebounceTimer = globalThis.setTimeout(() => {
      this.searchDebounceTimer = null;
      this.searchActiveResultIndex = -1;
      this.render();
      if (focus) this.focusSearchInput(false);
    }, 75) as unknown as number;
  }
  private focusSearchInput(select = false): void {
    const input = this.searchInput || this.contentEl.querySelector(".atomic-clusters-search-input") as HTMLInputElement | null;
    if (!input) return;
    this.searchInput = input;
    input.focus();
    if (select) input.select();
  }
  private clearSearch(): void { this.searchQuery = ""; this.searchActiveResultIndex = -1; this.render(); this.focusSearchInput(false); }
  private activateSearchCluster(cluster: SearchClusterDocument): void {
    if (cluster.manualGroupId) this.revealManualGroup(cluster.manualGroupId);
    else this.focusCluster(cluster.id);
  }
  private revealManualGroup(groupId: string): void {
    const rows = Array.from(this.contentEl.querySelectorAll<HTMLElement>(".atomic-clusters-manual-group"));
    const target = rows.find((row) => row.dataset.groupId === groupId);
    if (!target) return;
    for (const row of rows) {
      const active = row === target;
      row.classList.toggle("is-search-target", active);
      row.setAttribute("aria-current", String(active));
    }
    target.scrollIntoView({ block: "nearest" });
    target.focus({ preventScroll: true });
  }
  private openActiveSearchResult(): void {
    const path = this.searchResult?.notePaths[this.searchActiveResultIndex >= 0 ? this.searchActiveResultIndex : 0];
    if (path) {
      this.selectNote(path);
      void this.app.workspace.openLinkText(path, "", false);
    }
    else {
      const cluster = this.searchResult?.matchedClusters[0];
      if (cluster) this.activateSearchCluster(cluster);
    }
  }
  private moveSearchResult(delta: number): void {
    const total = this.searchResult?.notePaths.length || 0;
    if (!total) return;
    this.searchActiveResultIndex = (this.searchActiveResultIndex + delta + total) % total;
    const path = this.searchResult!.notePaths[this.searchActiveResultIndex];
    this.contentEl.querySelectorAll(".atomic-clusters-search-result-note").forEach((element) => { const active = (element as HTMLElement).dataset.path === path; if (active) element.classList.add("is-active"); else element.classList.remove("is-active"); });
  }
  private selectNote(path: string): void {
    if (!this.result || !this.result.ids.includes(path)) return;
    this.selectedNotePath = path;
    this.searchActiveResultIndex = this.searchResult?.notePaths.indexOf(path) ?? -1;
    this.render();
  }
  /** Select a note for an external Obsidian action and focus its preference picker. */
  focusNote(path: string): boolean {
    const normalizedPath = normalizeVaultRelativePath(path);
    if (!this.result || !normalizedPath || !this.result.ids.includes(normalizedPath)) return false;
    this.selectNote(normalizedPath);
    const picker = this.contentEl.querySelector(".atomic-clusters-note-preference")?.querySelector("select") as HTMLSelectElement | null;
    if (picker && typeof picker.focus === "function") picker.focus();
    return true;
  }
  private openSelectedNote(path: string): void { void this.app.workspace.openLinkText(path, "", false); }
  private handleExplorerKeydown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    const isInput = target === this.searchInput || target?.tagName === "INPUT" || target?.tagName === "TEXTAREA";
    if (event.key === "/" && !isInput) { event.preventDefault(); this.focusSearchInput(true); return; }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "f") { event.preventDefault(); this.focusSearchInput(true); return; }
    if (event.key === "Escape") {
      if (this.searchQuery) { event.preventDefault(); this.clearSearch(); }
      else if (this.focusNodeId) { event.preventDefault(); this.exitFocus(); }
      return;
    }
    if (!isInput) return;
    if (event.key === "Enter") { event.preventDefault(); this.openActiveSearchResult(); }
    else if (event.key === "ArrowDown") { event.preventDefault(); this.moveSearchResult(1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); this.moveSearchResult(-1); }
  }
  private findVisualizationNode(root: VisualizationNode, id: string): VisualizationNode | null {
    if (root.id === id) return root;
    for (const child of root.children) { const match = this.findVisualizationNode(child, id); if (match) return match; }
    return null;
  }
  private setFocusNode(nodeId: string | null): void {
    if (!nodeId || nodeId === "root") { this.exitFocus(); return; }
    this.focusNodeId = nodeId;
    this.visualizationSelectedNodeId = nodeId;
    this.visualizationNodeId = nodeId;
    this.visualizationDepth = Math.max(1, this.findVisualizationNode(this.visualizationRoot || buildVisualizationTree(this.result!.hierarchy, this.result!.leafLabels), nodeId)?.depth || 1);
    this.visualizationCameraState = null;
    this.searchActiveResultIndex = -1;
    this.render();
  }
  private focusCluster(clusterId: string): void { this.setFocusNode(clusterId === "root" ? null : `node:${clusterId}`); }
  private exitFocus(): void {
    this.focusNodeId = null;
    this.visualizationSelectedNodeId = null;
    this.visualizationNodeId = "root";
    this.visualizationDepth = 0;
    this.visualizationCameraState = null;
    this.render();
  }
  private moveFocusSibling(delta: number): void {
    if (!this.focusNodeId || !this.result) return;
    const root = this.visualizationRoot || buildVisualizationTree(this.result.hierarchy, this.result.leafLabels);
    let sibling: VisualizationNode | null = null;
    const visit = (parent: VisualizationNode): boolean => {
      const index = parent.children.findIndex((child) => child.id === this.focusNodeId);
      if (index >= 0) { sibling = parent.children[Math.max(0, Math.min(parent.children.length - 1, index + delta))] || null; return true; }
      return parent.children.some(visit);
    };
    visit(root);
    if (sibling !== null) this.setFocusNode((sibling as VisualizationNode).id);
  }
  private renderFocusControls(parent: HTMLElement): void {
    if (!this.focusNodeId || !this.result) return;
    const focus = parent.createDiv({ cls: "atomic-clusters-focus-bar" });
    const node = (this.visualizationRoot && this.findVisualizationNode(this.visualizationRoot, this.focusNodeId)) || null;
    const title = node?.sourceId === null ? "All notes" : this.effectiveTitles(this.result)[String(node?.sourceId)] || `Cluster ${node?.sourceId ?? this.clusterKeyForNodeId(this.focusNodeId)}`;
    focus.createSpan({ text: `Focus: ${title}` });
    const previous = focus.createEl("button", { text: "Previous sibling", attr: { type: "button", "aria-label": "Focus previous sibling" } }); previous.addEventListener("click", () => this.moveFocusSibling(-1));
    const next = focus.createEl("button", { text: "Next sibling", attr: { type: "button", "aria-label": "Focus next sibling" } }); next.addEventListener("click", () => this.moveFocusSibling(1));
    const root = focus.createEl("button", { text: "Exit focus", attr: { type: "button", "aria-label": "Exit cluster focus" } }); root.addEventListener("click", () => this.exitFocus());
  }
  private renderSearchControls(parent: HTMLElement): HTMLElement {
    this.refreshSearchIndex();
    const panel = parent.createDiv({ cls: "atomic-clusters-search-panel" });
    const row = panel.createDiv({ cls: "atomic-clusters-search-row" });
    const input = row.createEl("input", { cls: "atomic-clusters-search-input", attr: { type: "search", placeholder: "Search notes, paths, tags, or clusters…", "aria-label": "Search Explorer" } }) as HTMLInputElement;
    input.value = this.searchQuery;
    input.addEventListener("input", () => { this.searchQuery = input.value; this.scheduleSearchRender(); });
    input.addEventListener("keydown", (event) => { this.handleExplorerKeydown(event); event.stopPropagation(); });
    this.searchInput = input;
    const clear = row.createEl("button", { text: "Clear", attr: { type: "button", "aria-label": "Clear Explorer search" } }); clear.disabled = !this.searchQuery; clear.addEventListener("click", () => this.clearSearch());
    const chips = panel.createDiv({ cls: "atomic-clusters-search-filters" });
    const hasManual = this.searchDocuments.some((document) => document.manuallyAdjusted);
    const filterLabels: Array<[SearchFilter, string, boolean, string]> = [
      ["all", "All", true, "Show all notes"],
      ["current-cluster", "Current cluster", !!this.focusNodeId, "Limit to the focused subtree"],
      ["noise", "Noise", !!this.result, "Show notes assigned to noise"],
      ["provisional", "Provisional", !!this.result, "Show provisional placements"],
      ["manually-adjusted", "Manually adjusted", hasManual, hasManual ? "Show manual corrections" : "Manual correction data is not persisted yet"],
      ["recently-changed", "Recently changed", this.searchDocuments.some((document) => Number.isFinite(document.mtime)), "Show notes changed in the last seven days"],
    ];
    for (const [filter, label, enabled, description] of filterLabels) {
      const chip = chips.createEl("button", { text: label, attr: { type: "button", title: description, "aria-pressed": String(this.isFilterActive(filter)) } });
      if (this.isFilterActive(filter)) chip.classList.add("is-active"); else chip.classList.remove("is-active"); chip.disabled = !enabled; chip.addEventListener("click", () => this.setSearchFilter(filter));
    }
    const summary = panel.createDiv({ cls: "atomic-clusters-search-summary", attr: { "aria-live": "polite" } });
    const result = this.searchResult || { notePaths: [], clusterIds: [], matchedClusters: [] } as unknown as SearchResult;
    summary.setText(this.isSearchActive() ? `${result.notePaths.length} matching note${result.notePaths.length === 1 ? "" : "s"} · ${result.clusterIds.length} matching cluster${result.clusterIds.length === 1 ? "" : "s"}` : `${this.searchDocuments.length} notes indexed`);
    if (this.isSearchActive()) {
      const results = panel.createDiv({ cls: "atomic-clusters-search-results" });
      if (!result.notePaths.length && !result.matchedClusters.length) results.createDiv({ text: "No matching notes or clusters" }).addClass("atomic-clusters-status");
      result.notePaths.slice(0, 12).forEach((path, index) => { const button = results.createEl("button", { text: path, cls: "atomic-clusters-search-result-note", attr: { type: "button", "data-path": path, "aria-selected": String(path === this.selectedNotePath) } }); if (index === this.searchActiveResultIndex || path === this.selectedNotePath) button.classList.add("is-active"); button.addEventListener("click", () => this.selectNote(path)); });
      result.matchedClusters.slice(0, 8).forEach((cluster) => { const label = cluster.manualGroupId ? "Manual group" : "Cluster"; const button = results.createEl("button", { text: `${label} · ${cluster.title}`, cls: "atomic-clusters-search-result-cluster", attr: { type: "button" } }); button.addEventListener("click", () => this.activateSearchCluster(cluster)); });
    }
    this.renderFocusControls(panel);
    return panel;
  }

  private renderNoteDetailPanel(parent: HTMLElement): HTMLElement {
    const panel = parent.createEl("section", { cls: "atomic-clusters-note-detail", attr: { "aria-label": "Selected note details", "aria-live": "polite" } });
    // The three-argument buildNoteDetail(this.result, this.searchNotes, this.selectedNotePath) form remains valid for callers without persisted corrections.
    const detail: NoteDetailModel | null = this.result && this.selectedNotePath ? buildNoteDetail(this.result, this.searchNotes, this.selectedNotePath, 5, this.manualCorrections) : null;
    if (!detail) {
      panel.createEl("h4", { text: "Note details" });
      panel.createDiv({ text: "Select a note from search results, the hierarchy, or the map to inspect it." }).addClass("atomic-clusters-status");
      return panel;
    }

    const heading = panel.createDiv({ cls: "atomic-clusters-note-detail-heading" });
    heading.createEl("h4", { text: detail.title });
    heading.createDiv({ text: detail.path, cls: "atomic-clusters-note-detail-path" });
    heading.createEl("button", { text: "Open note", cls: "mod-cta", attr: { type: "button", "aria-label": `Open note ${detail.path}` } }).addEventListener("click", () => this.openSelectedNote(detail.path));

    const status = panel.createDiv({ cls: "atomic-clusters-note-detail-status" });
    const statusValues: string[] = [];
    if (detail.noise) statusValues.push("Noise");
    else if (detail.residual) statusValues.push("Residual placement");
    if (detail.provisional) statusValues.push("Provisional");
    status.createSpan({ text: statusValues.length ? statusValues.join(" · ") : "Stable automatic placement", cls: statusValues.length ? "atomic-clusters-note-detail-badge is-warning" : "atomic-clusters-note-detail-badge" });

    const placement = panel.createDiv({ cls: "atomic-clusters-note-detail-section" });
    placement.createEl("h5", { text: "Placement" });
    const placementGrid = placement.createDiv({ cls: "atomic-clusters-note-detail-grid" });
    placementGrid.createDiv({ text: `Automatic leaf: ${detail.automaticLeaf?.title || "Noise / no leaf"}` });
    placementGrid.createDiv({ text: `Probability: ${formatDetailPercent(detail.probability)}` });
    placementGrid.createDiv({ text: `Strongest membership: ${formatDetailPercent(detail.strongestMembership)}` });
    placementGrid.createDiv({ text: `Manual preferred: ${detail.manualPreferredCluster?.title || "None recorded; using automatic"}` });
    const preferred = placement.createDiv({ cls: "atomic-clusters-note-preference" });
    preferred.createEl("label", { text: "Prefer another cluster" });
    const picker = preferred.createEl("select", { attr: { "aria-label": "Prefer another cluster" } }) as HTMLSelectElement;
    picker.createEl("option", { text: "Use automatic placement", attr: { value: "" } });
    const candidates = detail.preferredClusterCandidates.slice(0, 5);
    const currentPreference = detail.manualPreferredCluster;
    if (currentPreference && !candidates.some((candidate) => candidate.key === currentPreference.key)) picker.createEl("option", { text: `${currentPreference.title} · saved preference`, attr: { value: currentPreference.key } });
    for (const candidate of candidates) picker.createEl("option", { text: `${candidate.title}${candidate.automatic ? " · automatic" : ""}`, attr: { value: candidate.key } });
    picker.value = detail.manualPreferredCluster?.key || "";
    picker.addEventListener("change", () => { if (picker.value) this.saveNotePreference(detail.path, picker.value); else this.clearNotePreference(detail.path); });
    if (detail.manualPreferredCluster) {
      const clear = preferred.createEl("button", { text: "Clear preference", attr: { type: "button" } }); clear.addEventListener("click", () => this.clearNotePreference(detail.path));
    }

    const hierarchy = panel.createDiv({ cls: "atomic-clusters-note-detail-section" });
    hierarchy.createEl("h5", { text: "Automatic hierarchy" });
    if (detail.ancestors.length) {
      const list = hierarchy.createEl("ol", { cls: "atomic-clusters-note-detail-ancestors" });
      detail.ancestors.forEach((ancestor) => list.createEl("li", { text: ancestor.title }));
    } else hierarchy.createDiv({ text: "Hierarchy information unavailable." }).addClass("atomic-clusters-status");

    const membership = panel.createDiv({ cls: "atomic-clusters-note-detail-section" });
    membership.createEl("h5", { text: "Membership" });
    if (detail.memberships.length) {
      const list = membership.createEl("ul", { cls: "atomic-clusters-note-detail-memberships" });
      detail.memberships.forEach((item) => list.createEl("li", { text: `${item.title} · ${formatDetailPercent(item.value)}` }));
    } else membership.createDiv({ text: "No soft-membership data available." }).addClass("atomic-clusters-status");

    const keywords = panel.createDiv({ cls: "atomic-clusters-note-detail-section" });
    keywords.createEl("h5", { text: "Cluster keywords" });
    if (detail.clusterKeywords.length) {
      const keywordList = keywords.createDiv({ cls: "atomic-clusters-note-detail-keywords" });
      detail.clusterKeywords.forEach((keyword) => keywordList.createSpan({ text: keyword }));
    } else keywords.createDiv({ text: "No keyword metadata available." }).addClass("atomic-clusters-status");

    const related = panel.createDiv({ cls: "atomic-clusters-note-detail-section" });
    related.createEl("h5", { text: "Related notes" });
    if (detail.relatedNotes.length) {
      const list = related.createDiv({ cls: "atomic-clusters-note-detail-related" });
      detail.relatedNotes.forEach((item) => {
        const button = list.createEl("button", { text: `${item.title} · ${item.path} · ${formatDetailPercent(item.similarity)}`, attr: { type: "button", "aria-label": `Select related note ${item.path}` } });
        button.addEventListener("click", () => this.selectNote(item.path));
      });
    } else related.createDiv({ text: "Related-note data is unavailable for this saved result." }).addClass("atomic-clusters-status");
    return panel;
  }

  private invalidateVisualizationGeneration(): void {
    this.visualizationGenerationToken++;
    if (this.visualizationGenerationIdleHandle !== null) this.visualizationGenerationCancel?.();
    this.visualizationGenerationIdleHandle = null; this.visualizationGenerationCancel = null;
    this.visualizationGenerationResult = null; this.visualizationGenerationScheduled = false; this.visualizationGenerationInFlight = false; this.visualizationGenerationError = null;
  }
  private scheduleVisualizationGeneration(result: ClusterResult): void {
    if (!this.ensureVisualization || this.visualizationClosed || this.visualizationGenerationError || (this.visualizationGenerationResult === result && (this.visualizationGenerationScheduled || this.visualizationGenerationInFlight))) return;
    this.visualizationGenerationResult = result; this.visualizationGenerationScheduled = true; const token = this.visualizationGenerationToken;
    const run = (): void => {
      this.visualizationGenerationIdleHandle = null; this.visualizationGenerationCancel = null;
      if (this.visualizationClosed || token !== this.visualizationGenerationToken || this.result !== result) { this.visualizationGenerationScheduled = false; return; }
      this.visualizationGenerationScheduled = false; this.visualizationGenerationInFlight = true;
      Promise.resolve().then(() => this.ensureVisualization!(result)).then((returned) => {
        if (this.visualizationClosed || token !== this.visualizationGenerationToken || this.result !== result) return;
        const complete = returned && typeof returned === "object" && Array.isArray((returned as ClusterResult).ids) && "hierarchy" in returned ? returned as ClusterResult : null;
        const returnedVisualization = returned && typeof returned === "object" && "coordinates" in returned ? returned as NonNullable<ClusterResult["visualization"]> : undefined;
        const visualization = complete?.visualization || returnedVisualization || result.visualization;
        if (!visualization) throw new Error("Visualization generation returned no coordinates");
        const nextResult = complete || { ...result, visualization };
        this.visualizationGenerationInFlight = false; this.visualizationGenerationResult = null; this.visualizationGenerationError = null; this.result = nextResult; this.visualizationNodeId = "root"; this.render();
      }).catch((error: unknown) => {
        if (this.visualizationClosed || token !== this.visualizationGenerationToken || this.result !== result) return;
        this.visualizationGenerationInFlight = false; this.visualizationGenerationError = `Visualization preparation failed: ${error instanceof Error ? error.message : String(error)}`; this.render();
      });
    };
    const idle = (globalThis as typeof globalThis & { requestIdleCallback?: (callback: (deadline: { timeRemaining: () => number }) => void) => number; cancelIdleCallback?: (handle: number) => void }).requestIdleCallback;
    if (typeof idle === "function") { const handle = idle(() => run()); this.visualizationGenerationIdleHandle = handle; this.visualizationGenerationCancel = () => (globalThis as typeof globalThis & { cancelIdleCallback?: (handle: number) => void }).cancelIdleCallback?.(handle); }
    else { const handle = globalThis.setTimeout(run, 0) as unknown as number; this.visualizationGenerationIdleHandle = handle; this.visualizationGenerationCancel = () => globalThis.clearTimeout(handle); }
  }
  private render(): void {
    this.disposeVisualization(); this.contentEl.empty(); this.contentEl.addClass("atomic-clusters-view");
    const header = this.contentEl.createDiv({ cls: "atomic-clusters-view-header" }); header.createEl("h3", { text: "Atomic Clusters" });
    if (this.progress && this.progress.value < 1) { header.createDiv({ text: `${this.progress.phase} · ${Math.round(this.progress.value * 100)}%` }).addClass("atomic-clusters-status"); const bar = header.createDiv({ cls: "atomic-clusters-progress" }); bar.createEl("span").style.width = `${Math.round(this.progress.value * 100)}%`; if (!this.result) return; }
    if (!this.result) { this.renderSearchControls(this.contentEl); this.contentEl.createDiv({ text: "No clustering result yet. Run Build note clusters." }).addClass("atomic-clusters-status"); return; }
    header.createDiv({ text: `${this.result.hierarchy.leaves.length} leaf clusters · ${this.result.hierarchy.merges.length} hierarchy merges · PCA ${this.result.pca.selected} dimensions` }).addClass("atomic-clusters-status");
    if (this.pendingChangeCount) header.createDiv({ text: `${this.pendingChangeCount} note${this.pendingChangeCount === 1 ? "" : "s"} pending refresh` }).addClass("atomic-clusters-status");
    const provisionalCount = this.result.provisionalPaths?.length || this.result.incremental?.provisionalPaths?.length || 0;
    if (this.result.incremental?.mode === "soft" || provisionalCount || this.result.incremental?.fullRebuildRecommended) {
      const status = provisionalCount ? `${provisionalCount} provisional placement${provisionalCount === 1 ? "" : "s"}` : "Structure reused";
      header.createDiv({ text: `${status}${this.result.incremental?.fullRebuildRecommended ? " · full rebuild recommended" : ""}` }).addClass("atomic-clusters-status");
    }
    this.renderSearchControls(this.contentEl);
    this.renderNoteDetailPanel(this.contentEl);
    this.renderManualGroupPanel(this.contentEl);
    // Create the tree panel before wiring the plot's ResizeObserver. The
    // tree is visually ordered below the plot via CSS, but its flex space is
    // present while the first camera is measured. Otherwise appending it
    // after renderVisualization() changes the plot height in the observer's
    // first callback and is indistinguishable from a real resize, cancelling
    // a just-started navigation animation.
    const tree = this.contentEl.createEl("details", { cls: "atomic-clusters-tree-panel" }); tree.createEl("summary", { text: `Cluster hierarchy · ${this.result.hierarchy.leaves.length} leaves` });
    const list = tree.createDiv({ cls: "atomic-clusters-tree" }); const adapterRoot = buildVisualizationTree(this.result.hierarchy, this.result.leafLabels); this.visualizationRoot = adapterRoot; const palette = visualizationColorScheme(adapterRoot);
    const displayTitles = this.effectiveTitles(this.result);
    const snapshotsByNode = new Map(generatedClusterSnapshots(this.result).map((snapshot) => [snapshot.nodeId, snapshot]));
    const provisional = new Set(this.result.provisionalPaths || this.result.incremental?.provisionalPaths || []);
    const searchActive = this.isSearchActive(); const matchedNotes = new Set(this.searchResult?.notePaths || []); const matchedClusters = new Set(this.searchResult?.clusterIds || []);
    const nodeMatchesSearch = (item: VisualizationNode): boolean => !searchActive || matchedClusters.has(item.sourceId === null ? "root" : String(item.sourceId)) || item.pointIndices.some((index) => matchedNotes.has(this.result!.ids[index]));
    const residuals = (this.result.hierarchyPlacements || []).map((placement, index) => ({ placement, path: this.result!.ids[index], index })).filter(({ placement }) => placement.kind === "residual");
    const renderScrollableNotes = (parent: HTMLElement, paths: readonly string[]): void => {
      const viewport = parent.createDiv({ cls: "atomic-clusters-note-list", attr: { tabindex: "0" } }); let rendered = 0; const chunkSize = 80;
      const appendChunk = (): void => { const end = Math.min(paths.length, rendered + chunkSize); for (; rendered < end; rendered++) { const path = paths[rendered]; const note = viewport.createEl("button", { text: provisional.has(path) ? `${path} · provisional` : path, attr: { type: "button", "data-path": path, "aria-selected": String(path === this.selectedNotePath) } }); note.classList.add(searchActive && matchedNotes.has(path) ? "atomic-clusters-search-match" : searchActive ? "atomic-clusters-search-dimmed" : "atomic-clusters-search-neutral"); note.addEventListener("click", () => this.selectNote(path)); } };
      const onScroll = (): void => { if (viewport.scrollTop + viewport.clientHeight >= viewport.scrollHeight - 96) appendChunk(); };
      appendChunk(); viewport.addEventListener("scroll", onScroll);
    };
    const renderResidualSection = (parent: HTMLElement, nodeId: number | null): void => {
      const direct = residuals.filter(({ placement }) => placement.nodeId === nodeId); if (!direct.length) return;
      const section = parent.createEl("details", { cls: "atomic-clusters-residuals" }); section.createEl("summary", { text: `Notes remaining at this stage · ${direct.length}` });
      renderScrollableNotes(section, direct.map((item) => item.path));
    };
    const renderNode = (item: VisualizationNode, parent: HTMLElement, depth: number): void => {
      if (!item.children.length) {
        const id = item.sourceId ?? -1; const node = parent.createDiv({ cls: "atomic-clusters-node" }); node.classList.add(nodeMatchesSearch(item) ? "atomic-clusters-search-match" : "atomic-clusters-search-dimmed"); node.style.setProperty("--atomic-cluster-color", palette.nodeColors.get(item.id) || "#9aa0a6"); const title = displayTitles[String(id)]; const generatedTitle = this.result!.titles?.[String(id)]; node.createEl("strong", { text: `${title ? `${title} · ` : ""}Leaf ${id}` }); if (title && generatedTitle && title !== generatedTitle) node.createDiv({ text: `Generated: ${generatedTitle}` }).addClass("atomic-clusters-generated-title"); const focus = node.createEl("button", { text: "Focus", attr: { type: "button", "aria-label": `Focus cluster ${id}` } }); focus.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); this.focusCluster(String(id)); });
        const snapshot = snapshotsByNode.get(id); if (snapshot) this.renderCorrectionActions(node, id, snapshot.stableClusterKey);
        const files = this.result!.ids.map((path, index) => ({ path, index })).filter((entry) => this.result!.hierarchyPlacements?.length ? this.result!.hierarchyPlacements[entry.index]?.kind === "leaf" && this.result!.hierarchyPlacements[entry.index]?.nodeId === id : this.result!.leafLabels[entry.index] === id).sort((a, b) => (this.result!.probabilities[b.index] || 0) - (this.result!.probabilities[a.index] || 0) || a.path.localeCompare(b.path)).map((entry) => entry.path);
        node.createDiv({ text: files.length ? `Notes · ${files.length}` : "No notes" }).addClass("atomic-clusters-status"); if (files.length) renderScrollableNotes(node, files); return;
      }
      const details = parent.createEl("details", { cls: "atomic-clusters-node" }) as HTMLDetailsElement; details.classList.add(nodeMatchesSearch(item) ? "atomic-clusters-search-match" : "atomic-clusters-search-dimmed"); details.style.setProperty("--atomic-cluster-color", palette.nodeColors.get(item.id) || "#9aa0a6"); details.open = depth === 0 || (searchActive && nodeMatchesSearch(item)); const id = item.sourceId; const title = id === null ? "All notes" : displayTitles[String(id)]; const generatedTitle = id === null ? undefined : this.result!.titles?.[String(id)]; details.createEl("summary", { text: `${id === null ? "All notes" : `${title ? `${title} · ` : ""}Merge ${id}`}` }); if (id !== null && title && generatedTitle && title !== generatedTitle) details.createDiv({ text: `Generated: ${generatedTitle}` }).addClass("atomic-clusters-generated-title"); const focus = details.createEl("button", { text: "Focus", attr: { type: "button", "aria-label": `Focus ${id === null ? "all notes" : `cluster ${id}`}` } }); focus.addEventListener("click", (event) => { event.preventDefault(); event.stopPropagation(); this.focusCluster(id === null ? "root" : String(id)); }); if (id !== null) { const snapshot = snapshotsByNode.get(id); if (snapshot) this.renderCorrectionActions(details, id, snapshot.stableClusterKey); } renderResidualSection(details, id); item.children.forEach((child) => renderNode(child, details, depth + 1));
    };
    const focusRoot = this.focusNodeId ? this.findVisualizationNode(adapterRoot, this.focusNodeId) : null;
    if (focusRoot) renderNode(focusRoot, list, 0);
    else if (adapterRoot.children.length) adapterRoot.children.forEach((child) => renderNode(child, list, 0));
    else list.createDiv({ text: "No non-noise clusters found." }).addClass("atomic-clusters-status");
    renderResidualSection(list, null);
    this.renderVisualization();
  }
  private renderVisualization(): void { this.renderGlobalVisualizationReliable(); }
  /** Legacy semantic-transition renderer retained for saved DOM fixtures. */
  private renderLegacyVisualization(): void {
    const result = this.result!; const displayTitles = this.effectiveTitles(result); const visualization = result.visualization; const ordering = visualizationLeafOrdering(result);
    if (!validateVisualizationData(result) || !visualization) {
      if (result.schemaVersion >= 6) {
        const message = this.visualizationGenerationError || (this.ensureVisualization ? "Preparing visualization…" : "Visualization will be prepared when the explorer is connected to a projection worker.");
        this.contentEl.createDiv({ text: message }).addClass("atomic-clusters-status");
        if (this.ensureVisualization && !this.visualizationGenerationError) this.scheduleVisualizationGeneration(result);
        return;
      }
      this.contentEl.createDiv({ text: "Hierarchical Gaussian-cloud visualization is unavailable for this saved result. Rebuild clusters to create a v4 result with soft memberships." }).addClass("atomic-clusters-status"); const rebuild = this.contentEl.createEl("button", { text: "Rebuild clusters", attr: { type: "button" } }); rebuild.addEventListener("click", () => void this.rebuildClusters?.()); return;
    }
    const coordinates = visualization.coordinates; const memberships = result.memberships!; const labels = visualization.labels; this.visualizationRoot = buildVisualizationTree(result.hierarchy, labels); const palette = visualizationColorScheme(this.visualizationRoot); const hierarchyDepths = new Map<number, number>(); const indexDepths = (node: VisualizationNode): void => { if (node.sourceId !== null) hierarchyDepths.set(node.sourceId, node.depth); node.children.forEach(indexDepths); }; indexDepths(this.visualizationRoot); // Global-world baseline retained for compatibility: scaleVisualizationPoints(visualization.coordinates, width, height)
    const findNode = (node: VisualizationNode): VisualizationNode | null => node.id === this.visualizationNodeId ? node : node.children.reduce<VisualizationNode | null>((found, child) => found || findNode(child), null); const current = findNode(this.visualizationRoot) || this.visualizationRoot; this.visualizationNodeId = current.id;
    const frontier = visualizationFrontier(this.visualizationRoot, [current.id], memberships, ordering, result.hierarchyPlacements); const frame = this.contentEl.createDiv({ cls: "atomic-clusters-umap" }); const plot = frame.createDiv({ cls: "atomic-clusters-umap-plot" }); const navigation = plot.createDiv({ cls: "atomic-clusters-umap-navigation" }); const path = visualizationPath(this.visualizationRoot, current.id);
    const back = navigation.createEl("button", { text: "Back", attr: { type: "button", "aria-label": "Back to parent cluster" } }); back.disabled = path.length <= 1; back.addEventListener("click", () => { if (path.length > 1) { const parent = path[path.length - 2]; this.navigateVisualization(parent.id, () => undefined); } });
    const crumbs = navigation.createDiv({ cls: "atomic-clusters-breadcrumb" }); path.forEach((node, index) => { if (index) crumbs.createSpan({ text: " / " }); const title = node.sourceId === null ? "All notes" : displayTitles[String(node.sourceId)]; const crumb = crumbs.createEl("button", { text: title || `Cluster ${node.sourceId}`, attr: { type: "button", "aria-current": index === path.length - 1 ? "page" : "false" } }); crumb.addEventListener("click", () => { this.navigateVisualization(node.id, () => undefined); }); });
    // Canvas and hit targets share one transformed layer. Controls/navigation
    // remain siblings so they never zoom or blur during camera movement.
    const transition = this.visualizationTransition;
    // The target semantic stage is rendered immediately, but stays hidden
    // while the outgoing snapshot performs the camera move. This keeps the
    // old visual on screen even if layout/ResizeObserver briefly reports a
    // fallback size.
    const outgoingLayer = transition ? plot.createDiv({ cls: "atomic-clusters-umap-outgoing-layer" }) : null;
    if (outgoingLayer && transition) {
      outgoingLayer.style.width = transition.snapshot.style.width || `${transition.snapshot.width}px`;
      outgoingLayer.style.height = transition.snapshot.style.height || `${transition.snapshot.height}px`;
      outgoingLayer.style.visibility = "visible";
      outgoingLayer.style.transform = "none";
      transition.snapshot.style.position = "absolute";
      transition.snapshot.style.left = "0";
      transition.snapshot.style.top = "0";
      transition.snapshot.style.display = "block";
      outgoingLayer.appendChild(transition.snapshot);
    }
    const visualLayer = plot.createDiv({ cls: "atomic-clusters-umap-visual-layer" });
    // The semantic stage is replaced synchronously, but remains hidden until
    // the outgoing snapshot has completed its camera move. It must not enter
    // the compositor during an unstable layout frame, or the target cloud can
    // flash before the outgoing image has started moving.
    if (transition) {
      visualLayer.style.visibility = "hidden";
    }
    const canvas = visualLayer.createEl("canvas", { cls: "atomic-clusters-umap-canvas" }); canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "Hierarchical Gaussian cloud visualization. Click a cloud to zoom; hover a note to preview it; click a note to open it."); const hitLayer = visualLayer.createDiv({ cls: "atomic-clusters-umap-hit-layer" });
    // Keep all notes in the global UMAP world and fit only the selected node in the camera.
    const region = visualizationRegion(current, coordinates); const baseSigma = visualizationBaseBandwidth(coordinates); const p95 = visualizationP95RowSum(memberships); let cachedKey = ""; let cachedClouds: VisualizationSplat[][] = []; let cachedBitmap: HTMLCanvasElement | null = null;
    const pointHitRadius = 14; const maxPointHitTargets = 96; const pointHitButtons = new Map<number, HTMLButtonElement>();
    type VisualizationViewport = [number, number];
    // ResizeObserver and canvas layout do not necessarily report the same
    // floating-point CSS size.  A freshly mounted plot can also differ by a
    // pixel while its scrollbar/control breakpoint settles.  Treat those
    // reports as the same viewport; a materially different box is still a
    // real resize and must invalidate the raster and snap an active camera
    // transition.
    const normalizeViewport = (size: VisualizationViewport): VisualizationViewport => [Math.max(1, Math.round(size[0])), Math.max(1, Math.round(size[1]))];
    const equivalentViewport = (a: VisualizationViewport | null, b: VisualizationViewport | null): boolean => !!a && !!b && Math.abs(a[0] - b[0]) <= VISUALIZATION_VIEWPORT_TOLERANCE && Math.abs(a[1] - b[1]) <= VISUALIZATION_VIEWPORT_TOLERANCE;
    // ResizeObserver reports the plot's content box before the canvas's
    // getBoundingClientRect() is necessarily up to date. Prefer that value
    // during a resize callback, and treat a zero-sized report as transient.
    // In particular, never replace a valid canvas frame with a 1px/fallback
    // frame while a pane is being detached or its responsive controls move.
    const rectSize = (entry?: ResizeObserverEntry): VisualizationViewport | null => {
      if (entry) {
        const width = entry.contentRect?.width || 0; const height = entry.contentRect?.height || 0;
        return width > 0 && height > 0 ? [width, height] : null;
      }
      const rect = canvas.getBoundingClientRect(); const width = rect.width || plot.clientWidth; const height = rect.height || plot.clientHeight;
      return width > 0 && height > 0 ? [width, height] : null;
    };
    let renderedCamera: VisualizationCamera | null = null;
    const clearLayerTransform = (): void => {
      visualLayer.style.transform = "none"; visualLayer.style.transformOrigin = ""; visualLayer.style.opacity = ""; visualLayer.style.visibility = ""; visualLayer.classList.remove("is-animating");
      if (outgoingLayer?.parentElement) outgoingLayer.parentElement.removeChild(outgoingLayer);
    };
    const draw = (viewport?: VisualizationViewport): boolean => {
      const rawSize = viewport || rectSize(); if (!rawSize) return false;
      const size = normalizeViewport(rawSize);
      const [width, height] = size; const dpr = Math.max(1, typeof window === "undefined" ? 1 : window.devicePixelRatio || 1); canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); const context = canvas.getContext("2d"); if (!context) return false; context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, width, height);
      const camera = visualizationCameraTransform(region, width, height); renderedCamera = camera; const points = coordinates.map((point) => visualizationWorldToScreen(camera, point)); this.visualizationPoints = points; this.visualizationLastCamera = camera;
      const kernelScale = clampVisualizationKernelScale(this.visualizationKernelScale); this.visualizationKernelScale = kernelScale;
      const key = `${current.id}|${Math.round(width)}x${Math.round(height)}|${baseSigma}|${kernelScale}|${dpr}`;
      if (key !== cachedKey) { cachedKey = key; cachedBitmap = null; const longAxis = Math.max(width, height); const rasterLong = Math.max(256, Math.min(512, Math.round(longAxis * .5))); const rasterScale = rasterLong / longAxis; const rasterWidth = Math.max(1, Math.round(width * rasterScale)); const rasterHeight = Math.max(1, Math.round(height * rasterScale)); cachedClouds = []; const allDots = new Set<number>();
        for (const entry of frontier) { const isLeaf = !entry.node.children.length; const splats: VisualizationSplat[] = []; const clusterColor = visualizationCloudColor(entry.node, palette); const color = /^#[0-9a-f]{6}$/i.test(clusterColor) ? [parseInt(clusterColor.slice(1, 3), 16), parseInt(clusterColor.slice(3, 5), 16), parseInt(clusterColor.slice(5, 7), 16)] as [number, number, number] : visualizationColorVector([], ordering); if (entry.actualPoints) for (const index of entry.pointIndices) allDots.add(index); else { for (const index of entry.pointIndices) { const point = points[index]; const row = memberships[index] || []; if (!point) continue; splats.push({ x: point[0] * rasterScale, y: point[1] * rasterScale, sigma: visualizationScaledStageSigma(baseSigma, entry.remainingDepth, isLeaf, kernelScale) * camera.scale * rasterScale, color, amplitude: visualizationMembershipAmplitude(row, p95) }); } cachedClouds.push(splats); } for (const index of entry.residualIndices) allDots.add(index); }
        // A residual can be adjacent to multiple children, but is represented by one dot only.
        this.visualizationRenderedPointIndices = [...allDots].sort((a, b) => a - b);
        // Index only points belonging to this stage. Hidden notes must not
        // steal hover/clicks while a child subtree is focused.
        const indexedPoints = points.map((point, index) => allDots.has(index) ? point : [Number.NaN, Number.NaN] as [number, number]);
        this.visualizationSpatialIndex = buildVisualizationPointSpatialIndex(indexedPoints, pointHitRadius * 2);
        // Keep a small, reusable semantic button pool near the current
        // viewport. Every note remains painted and picked by the canvas; the
        // pool exists only for keyboard/accessibility affordances.
        const centerX = width / 2; const centerY = height / 2;
        const poolIndices = this.visualizationSpatialIndex.queryRect(-pointHitRadius, -pointHitRadius, width + pointHitRadius, height + pointHitRadius)
          .sort((a, b) => ((points[a][0] - centerX) ** 2 + (points[a][1] - centerY) ** 2) - ((points[b][0] - centerX) ** 2 + (points[b][1] - centerY) ** 2) || a - b)
          .slice(0, maxPointHitTargets);
        const poolSet = new Set(poolIndices);
        for (const [index, hit] of pointHitButtons) if (!poolSet.has(index)) { if (hit.parentElement === hitLayer) hitLayer.removeChild(hit); pointHitButtons.delete(index); }
        this.visualizationHitElements = poolIndices.map((index) => {
          const placement = result.hierarchyPlacements?.[index]; const residualDepth = placement?.nodeId === null ? 0 : hierarchyDepths.get(placement?.nodeId ?? -1) || 0; const residualDetail = placement?.kind === "residual" ? ` · residual at ${placement.nodeId === null ? "root" : `node ${placement.nodeId}`} · depth ${residualDepth} · max child confidence ${placement.confidence.toFixed(3)}` : "";
          let hit = pointHitButtons.get(index); if (!hit) { hit = hitLayer.createEl("button", { cls: "atomic-clusters-umap-point-hit", attr: { type: "button", "data-point-index": String(index), "aria-label": `Select note ${result.ids[index]}` } }); hit.addEventListener("mouseenter", (event) => this.setHoveredVisualizationPoint(index, event, hit!, draw)); hit.addEventListener("click", (event) => { event.preventDefault(); this.selectNote(result.ids[index]); }); pointHitButtons.set(index, hit); }
          hit.setAttribute("aria-label", `${result.ids[index]}${residualDetail}`); hit.setAttribute("title", `${result.ids[index]}${residualDetail}`); return hit;
        });
        const field = accumulateVisualizationDensity(cachedClouds.flat(), rasterWidth, rasterHeight); const bitmap = typeof document === "undefined" ? null : document.createElement("canvas"); if (bitmap) { bitmap.width = rasterWidth; bitmap.height = rasterHeight; const bitmapContext = bitmap.getContext("2d"); if (bitmapContext) { const image = bitmapContext.createImageData(rasterWidth, rasterHeight); for (let offset = 0; offset < field.density.length; offset++) { const density = field.density[offset]; const alpha = visualizationDensityAlpha(density); image.data[offset * 4] = density > 0 ? Math.round(field.red[offset] / density) : 0; image.data[offset * 4 + 1] = density > 0 ? Math.round(field.green[offset] / density) : 0; image.data[offset * 4 + 2] = density > 0 ? Math.round(field.blue[offset] / density) : 0; image.data[offset * 4 + 3] = Math.round(alpha * 255); } bitmapContext.putImageData(image, 0, 0); cachedBitmap = bitmap; } }
      }
      if (cachedBitmap) { context.imageSmoothingEnabled = true; context.drawImage(cachedBitmap, 0, 0, width, height); }
      const computedStyle = typeof getComputedStyle === "function" ? getComputedStyle(frame) : null; const background = computedStyle?.getPropertyValue("--background-primary").trim() || "transparent"; const residualDots = new Set(frontier.flatMap((entry) => entry.residualIndices)); const directResidualDots = new Set(frontier.flatMap((entry) => entry.directResidualIndices || [])); const dotIndices = this.visualizationRenderedPointIndices; dotIndices.forEach((index) => { const point = points[index]; if (!point) return; const radius = index === this.hoveredVisualizationPoint ? 6 : 4; context.beginPath(); const label = labels[index]; context.fillStyle = directResidualDots.has(index) ? "#6f757b" : label === -1 || residualDots.has(index) ? VISUALIZATION_NOISE_COLOR : blendVisualizationColor(memberships[index], ordering, palette.leafColors); context.globalAlpha = index === this.hoveredVisualizationPoint ? 1 : .9; context.arc(point[0], point[1], radius, 0, Math.PI * 2); context.fill(); context.globalAlpha = 1; context.strokeStyle = background; context.stroke(); });
      // Labels live on the same canvas as the cloud and dots.  Consequently
      // the outgoing navigation snapshot contains them too, with no DOM
      // overlay flashing during the camera transition.  They are deliberately
      // not hit targets: cloud picking remains owned by the canvas and note
      // picking remains owned by the transparent buttons above it.
      const labelPlacements = layoutVisualizationClusterLabels(frontier, points, displayTitles, palette.nodeColors, width, height, { measureText: (text) => typeof context.measureText === "function" ? context.measureText(text).width : text.length * 7 });
      if (typeof context.fillText === "function") for (const placement of labelPlacements) {
        const radius = Math.min(placement.height / 2, 8); const right = placement.x + placement.width; const bottom = placement.y + placement.height;
        context.save(); context.beginPath();
        if (typeof context.roundRect === "function") context.roundRect(placement.x, placement.y, placement.width, placement.height, radius);
        else { context.moveTo(placement.x + radius, placement.y); context.lineTo(right - radius, placement.y); context.arcTo(right, placement.y, right, placement.y + radius, radius); context.lineTo(right, bottom - radius); context.arcTo(right, bottom, right - radius, bottom, radius); context.lineTo(placement.x + radius, bottom); context.arcTo(placement.x, bottom, placement.x, bottom - radius, radius); context.lineTo(placement.x, placement.y + radius); context.arcTo(placement.x, placement.y, placement.x + radius, placement.y, radius); }
        context.fillStyle = placement.contrast.background; context.globalAlpha = .92; context.fill(); context.globalAlpha = 1; // The outer stroke is the color that contrasts the cloud, so the pill remains visible even when its fill is near the cloud color.
        context.strokeStyle = placement.contrast.foreground; context.lineWidth = 2; context.stroke(); context.fillStyle = placement.contrast.foreground; context.font = "600 12px system-ui, sans-serif"; context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(placement.text, placement.x + placement.width / 2, placement.y + placement.height / 2); context.restore();
      }
      const hitPointIndices = this.visualizationHitElements.map((hit) => Number(hit.dataset.pointIndex)); this.visualizationHitElements.forEach((hit, index) => { const point = points[hitPointIndices[index]]; if (point) { hit.style.left = `${point[0]}px`; hit.style.top = `${point[1]}px`; } }); if (this.hoveredVisualizationPoint !== null) { const point = points[this.hoveredVisualizationPoint]; if (point) { context.beginPath(); context.strokeStyle = computedStyle?.color || "currentColor"; context.lineWidth = 1.5; context.arc(point[0], point[1], 10, 0, Math.PI * 2); context.stroke(); } } context.globalAlpha = 1;
      return true;
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
    frame.addEventListener("click", (event) => { if (this.visualizationAnimating) return; const pointButton = event.target instanceof HTMLElement && event.target.classList.contains("atomic-clusters-umap-point-hit"); const rect = canvas.getBoundingClientRect(); const x = event.clientX - rect.left, y = event.clientY - rect.top; if (!pointButton) { const note = this.visualizationSpatialIndex?.queryNearest(x, y, pointHitRadius) ?? null; if (note !== null) { void this.app.workspace.openLinkText(result.ids[note], "", false); return; } } if (!(event.target instanceof HTMLCanvasElement)) return; const longAxis = Math.max(rect.width, rect.height); const rasterScale = Math.max(256, Math.min(512, Math.round(longAxis * .5))) / longAxis; const picked = pickVisualizationCloud(cachedClouds, x * rasterScale, y * rasterScale); if (picked !== null) { const cloudEntries = frontier.filter((entry) => !entry.actualPoints); const target = cloudEntries[picked]; if (target) this.navigateVisualization(target.node.id, () => undefined); } });
    const onMove = (event: MouseEvent): void => { if (this.visualizationAnimating) return; const rect = canvas.getBoundingClientRect(); const index = this.visualizationSpatialIndex?.queryNearest(event.clientX - rect.left, event.clientY - rect.top, pointHitRadius) ?? null; const target = event.target instanceof HTMLElement && event.target.classList.contains("atomic-clusters-umap-point-hit") ? event.target : canvas; this.setHoveredVisualizationPoint(index, event, target, draw); }; const onLeave = (): void => this.setHoveredVisualizationPoint(null, null, null, draw); frame.addEventListener("mousemove", onMove); frame.addEventListener("mouseleave", onLeave); const onKey = (event: KeyboardEvent): void => { if (event.key === "Escape" && path.length > 1) { event.preventDefault(); const parent = path[path.length - 2]; this.navigateVisualization(parent.id, () => undefined); } }; const focusFrame = (): void => frame.focus(); frame.tabIndex = 0; frame.addEventListener("keydown", onKey); frame.addEventListener("pointerdown", focusFrame);
    this.visualizationCleanup = () => { frame.removeEventListener("mousemove", onMove); frame.removeEventListener("mouseleave", onLeave); frame.removeEventListener("keydown", onKey); frame.removeEventListener("pointerdown", focusFrame); this.visualizationHitElements = []; this.visualizationRenderedPointIndices = []; this.visualizationSpatialIndex = null; };
    // A freshly replaced pane can report a temporary fallback size (usually
    // 320×280) before flex layout settles. Do not consume the transition in
    // that state: the old camera cannot be interpolated against a different
    // viewport, and doing so makes the real resize silently snap to the
    // destination. Keep the transition pending until both cameras describe
    // the same viewport, then start it from the first stable frame.
    let pendingTransition = this.visualizationTransition;
    const maybeStartTransition = (): void => {
      if (!pendingTransition || !renderedCamera) return;
      const transition = pendingTransition;
      pendingTransition = null;
      if (this.visualizationTransition === transition) this.visualizationTransition = null;
      // A real resize can settle at a different viewport than the camera
      // which was captured before navigation. The target has already been
      // rasterized at the new size, so finish the transition immediately and
      // reveal it instead of leaving the semantic layer hidden forever.
      if (!equivalentViewport([transition.fromCamera.width, transition.fromCamera.height], [renderedCamera.width, renderedCamera.height])) {
        this.visualizationDisplayedCamera = renderedCamera;
        clearLayerTransform();
        return;
      }
      this.startVisualizationAnimation(outgoingLayer, visualLayer, transition.snapshot, transition.fromCamera, renderedCamera, clearLayerTransform);
    };
    const initialPlotSize = rectSize(); let lastObservedPlotSize: VisualizationViewport | null = initialPlotSize ? normalizeViewport(initialPlotSize) : null; if (typeof ResizeObserver === "function") { this.visualizationResizeObserver = new ResizeObserver((entries) => { const nextSize = rectSize(entries?.[0]); if (!nextSize) return;
        const normalizedNextSize = normalizeViewport(nextSize); const changed = !equivalentViewport(normalizedNextSize, lastObservedPlotSize); if (!changed) { lastObservedPlotSize = normalizedNextSize; maybeStartTransition(); return; } lastObservedPlotSize = normalizedNextSize;
        // A resize invalidates both the low-resolution cloud bitmap and the
        // hit-target positions, even when rounding happens to preserve the
        // old cache key. If an outgoing animation is active, its snapshot is
        // no longer in the right viewport; snap to and reveal the target.
        cachedKey = ""; cachedBitmap = null; cachedClouds = [];
        if (this.visualizationAnimating) { this.cancelVisualizationAnimation(true); clearLayerTransform(); }
        draw(nextSize); maybeStartTransition();
      }); this.visualizationResizeObserver.observe(plot); } draw(); maybeStartTransition();
  }
  /** Global-depth renderer with a gesture-only camera. */
  private renderGlobalVisualization(): void {
    const result = this.result!; const displayTitles = this.effectiveTitles(result); const visualization = result.visualization; const ordering = visualizationLeafOrdering(result);
    if (!validateVisualizationData(result) || !visualization) {
      if (result.schemaVersion >= 6) { const message = this.visualizationGenerationError || (this.ensureVisualization ? "Preparing visualization…" : "Visualization will be prepared when the explorer is connected to a projection worker."); this.contentEl.createDiv({ text: message }).addClass("atomic-clusters-status"); if (this.ensureVisualization && !this.visualizationGenerationError) this.scheduleVisualizationGeneration(result); return; }
      this.contentEl.createDiv({ text: "Hierarchical Gaussian-cloud visualization is unavailable for this saved result." }).addClass("atomic-clusters-status"); return;
    }
    const coordinates = visualization.coordinates; const labels = visualization.labels; const memberships = result.memberships!; this.visualizationRoot = buildVisualizationTree(result.hierarchy, labels); const root = this.visualizationRoot; const palette = visualizationColorScheme(root);
    const frontier = visualizationGlobalDepthFrontier(root, this.visualizationDepth, this.visualizationSelectedNodeId, memberships, ordering, result.hierarchyPlacements); const selectedNode = this.visualizationSelectedNodeId ? visualizationPath(root, this.visualizationSelectedNodeId).at(-1) || null : null; const cameraCoordinates = selectedNode ? selectedNode.pointIndices.map((index) => coordinates[index]).filter((point): point is [number, number] => !!point && point.every(Number.isFinite)) : coordinates; const searchActive = this.isSearchActive(); const matchedNotes = new Set(this.searchResult?.notePaths || []); const matchedClusters = new Set(this.searchResult?.clusterIds || []); const frame = this.contentEl.createDiv({ cls: "atomic-clusters-umap" }); const plot = frame.createDiv({ cls: "atomic-clusters-umap-plot" }); const navigation = plot.createDiv({ cls: "atomic-clusters-umap-navigation" });
    const selectedPath = this.visualizationSelectedNodeId ? visualizationPath(root, this.visualizationSelectedNodeId) : [root]; const back = navigation.createEl("button", { text: "Back", attr: { type: "button", "aria-label": "Back to parent cluster" } }); back.disabled = this.visualizationDepth <= 0 && !this.visualizationSelectedNodeId;
    back.addEventListener("click", () => { if (this.visualizationDepth <= 0) this.visualizationSelectedNodeId = null; else { this.visualizationDepth--; const path = this.visualizationSelectedNodeId ? visualizationPath(root, this.visualizationSelectedNodeId) : []; this.visualizationSelectedNodeId = path.length > 2 ? path[path.length - 2].id : null; } this.focusNodeId = this.visualizationSelectedNodeId; this.visualizationNodeId = this.visualizationSelectedNodeId || "root"; this.visualizationCameraState = null; this.render(); });
    const crumbs = navigation.createDiv({ cls: "atomic-clusters-breadcrumb" }); selectedPath.forEach((node, index) => { if (index) crumbs.createSpan({ text: " / " }); const title = node.sourceId === null ? "All notes" : displayTitles[String(node.sourceId)] || `Cluster ${node.sourceId}`; const crumb = crumbs.createEl("button", { text: title, attr: { type: "button", "aria-current": index === selectedPath.length - 1 ? "page" : "false" } }); crumb.addEventListener("click", () => { if (node.id === "root") { this.visualizationDepth = 0; this.visualizationSelectedNodeId = null; this.focusNodeId = null; } else { this.visualizationSelectedNodeId = node.id; this.focusNodeId = node.id; } this.visualizationCameraState = null; this.render(); }); });
    const visualLayer = plot.createDiv({ cls: "atomic-clusters-umap-visual-layer" }); const canvas = visualLayer.createEl("canvas", { cls: "atomic-clusters-umap-canvas" }); canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "Hierarchical Gaussian cloud visualization. Drag to pan; scroll to zoom; hover a note to preview it."); const hitLayer = visualLayer.createDiv({ cls: "atomic-clusters-umap-hit-layer" });
    const baseSigma = visualizationBaseBandwidth(coordinates); const p95 = visualizationP95RowSum(memberships); const pointHitRadius = 14; const pointHitButtons = new Map<number, HTMLButtonElement>(); let camera: VisualizationCamera; let points: [number, number][] = []; let width = 0; let height = 0; let cachedKey = ""; let cachedBitmap: HTMLCanvasElement | null = null; let cachedClouds: VisualizationSplat[][] = []; let rasterCameraState: VisualizationCameraState | null = null; let densityTimer: number | null = null; let resizeTimer: number | null = null; let pendingResize: [number, number] | null = null; let suppressClick = false;
    const gestureOverscan = VISUALIZATION_GESTURE_OVERSCAN; const rectSize = (): [number, number] | null => { const rect = plot.getBoundingClientRect(); const w = rect.width || plot.clientWidth; const h = rect.height || plot.clientHeight; return w > 0 && h > 0 ? [Math.max(1, Math.round(w)), Math.max(1, Math.round(h))] : null; };
    const bitmapFor = (splats: readonly VisualizationSplat[], opacity: number): HTMLCanvasElement | null => { if (typeof document === "undefined") return null; const rasterWidth = width + gestureOverscan * 2; const rasterHeight = height + gestureOverscan * 2; const field = accumulateVisualizationDensity(splats, rasterWidth, rasterHeight); const bitmap = document.createElement("canvas"); bitmap.width = rasterWidth; bitmap.height = rasterHeight; const bitmapContext = bitmap.getContext("2d"); if (!bitmapContext) return null; const image = bitmapContext.createImageData(rasterWidth, rasterHeight); for (let i = 0; i < field.density.length; i++) { const density = field.density[i]; image.data[i * 4] = density > 0 ? Math.round(field.red[i] / density) : 0; image.data[i * 4 + 1] = density > 0 ? Math.round(field.green[i] / density) : 0; image.data[i * 4 + 2] = density > 0 ? Math.round(field.blue[i] / density) : 0; image.data[i * 4 + 3] = Math.round(visualizationDensityAlpha(density) * opacity * 255); } bitmapContext.putImageData(image, 0, 0); return bitmap; };
    const terminalPath = (index: number): string[] => visualizationNoteTerminalPath(root, index, labels, result.hierarchyPlacements); const pointActive = (index: number): boolean => (!this.visualizationSelectedNodeId || terminalPath(index).includes(this.visualizationSelectedNodeId)) && (!searchActive || matchedNotes.has(result.ids[index]));
    const activatePoint = (index: number): void => { const path = terminalPath(index); const selected = this.visualizationSelectedNodeId; if (selected && path[path.length - 1] === selected) { this.selectNote(result.ids[index]); return; } if (path.length === 1) { if (this.visualizationDepth === 0 && !selected) this.selectNote(result.ids[index]); else { this.visualizationSelectedNodeId = null; this.focusNodeId = null; this.visualizationDepth = 0; this.render(); } return; } const entry = frontier.find((item) => item.pointIndices.includes(index)); if (entry) { this.visualizationSelectedNodeId = entry.node.id; this.focusNodeId = entry.node.id; if (entry.node.children.length) this.visualizationDepth++; this.visualizationNodeId = entry.node.id; this.visualizationCameraState = visualizationFitCameraState(entry.node.pointIndices.map((pointIndex) => coordinates[pointIndex]).filter((point): point is [number, number] => !!point && point.every(Number.isFinite)), width || 1, height || 1); this.render(); return; } const next = path.find((id) => id !== "root"); if (next) { this.visualizationSelectedNodeId = next; this.focusNodeId = next; this.visualizationDepth++; this.visualizationNodeId = next; this.visualizationCameraState = visualizationFitCameraState(coordinates, width || 1, height || 1); this.render(); } };
    const activateCluster = (node: VisualizationNode): void => { this.visualizationSelectedNodeId = node.id; this.focusNodeId = node.id; if (node.children.length) this.visualizationDepth++; this.visualizationNodeId = node.id; this.visualizationCameraState = visualizationFitCameraState(node.pointIndices.map((index) => coordinates[index]).filter((point): point is [number, number] => !!point && point.every(Number.isFinite)), width || 1, height || 1); this.render(); };
    const draw = (reraster = true): boolean => { const size = rectSize(); if (!size) return false; [width, height] = size; const rasterWidth = width + gestureOverscan * 2; const rasterHeight = height + gestureOverscan * 2; const dpr = Math.max(1, typeof window === "undefined" ? 1 : window.devicePixelRatio || 1); canvas.width = rasterWidth * dpr; canvas.height = rasterHeight * dpr; canvas.style.left = `${-gestureOverscan}px`; canvas.style.top = `${-gestureOverscan}px`; canvas.style.width = `${rasterWidth}px`; canvas.style.height = `${rasterHeight}px`; const context = canvas.getContext("2d"); if (!context) return false; context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, rasterWidth, rasterHeight); if (!this.visualizationCameraState || this.visualizationCameraState.width !== width || this.visualizationCameraState.height !== height) { const fitted = visualizationFitCameraState(coordinates, width, height); const old = this.visualizationCameraState; this.visualizationCameraState = old ? { ...fitted, centerX: old.centerX, centerY: old.centerY, zoom: old.zoom } : fitted; } camera = visualizationCameraFromState(this.visualizationCameraState); points = coordinates.map((point) => visualizationWorldToScreen(camera, point)); this.visualizationPoints = points; const rasterPoints = points.map(([x, y]) => [x + gestureOverscan, y + gestureOverscan] as [number, number]); const key = `${this.visualizationDepth}|${this.visualizationSelectedNodeId || ""}|${camera.scale}|${camera.offsetX}|${camera.offsetY}|${width}x${height}|${this.visualizationKernelScale}|${this.searchQuery}|${[...this.searchFilters].join(",")}`; if (reraster && key !== cachedKey) { cachedKey = key; const active: VisualizationSplat[] = []; const inactive: VisualizationSplat[] = []; cachedClouds = []; for (const entry of frontier) { const value = visualizationCloudColor(entry.node, palette); const color = /^#[0-9a-f]{6}$/i.test(value) ? [parseInt(value.slice(1, 3), 16), parseInt(value.slice(3, 5), 16), parseInt(value.slice(5, 7), 16)] as [number, number, number] : visualizationColorVector([], ordering); const splats: VisualizationSplat[] = []; for (const index of entry.pointIndices) { const point = rasterPoints[index]; if (!point) continue; splats.push({ x: point[0], y: point[1], sigma: visualizationScaledStageSigma(baseSigma, entry.remainingDepth, !entry.node.children.length, this.visualizationKernelScale) * camera.scale, color, amplitude: visualizationMembershipAmplitude(memberships[index] || [], p95) }); } cachedClouds.push(splats); const clusterKey = entry.node.sourceId === null ? "root" : String(entry.node.sourceId); (entry.active && (!searchActive || matchedClusters.has(clusterKey)) ? active : inactive).push(...splats); } const activeBitmap = bitmapFor(active, 1); const inactiveBitmap = bitmapFor(inactive, .2); cachedBitmap = document.createElement("canvas"); cachedBitmap.width = rasterWidth; cachedBitmap.height = rasterHeight; const merged = cachedBitmap.getContext("2d"); if (merged) { if (inactiveBitmap) merged.drawImage(inactiveBitmap, 0, 0); if (activeBitmap) merged.drawImage(activeBitmap, 0, 0); } } if (cachedBitmap) context.drawImage(cachedBitmap, 0, 0); const style = typeof getComputedStyle === "function" ? getComputedStyle(frame) : null; const background = style?.getPropertyValue("--background-primary").trim() || "transparent"; for (let index = 0; index < points.length; index++) { const point = rasterPoints[index]; const placement = result.hierarchyPlacements?.[index]; context.beginPath(); context.fillStyle = labels[index] === -1 || placement?.kind === "residual" ? VISUALIZATION_NOISE_COLOR : blendVisualizationColor(memberships[index] || [], ordering, palette.leafColors); context.globalAlpha = this.hoveredVisualizationPoint === index ? 1 : pointActive(index) ? 1 : .2; context.arc(point[0], point[1], this.hoveredVisualizationPoint === index ? 6 : 4, 0, Math.PI * 2); context.fill(); context.globalAlpha = 1; context.strokeStyle = background; context.stroke(); } const labelsOnCanvas = layoutVisualizationClusterLabels(frontier, rasterPoints, displayTitles, palette.nodeColors, rasterWidth, rasterHeight); if (typeof context.fillText === "function") for (const label of labelsOnCanvas) { const entry = frontier.find((item) => item.node.id === label.id); const clusterKey = entry?.node.sourceId === null ? "root" : entry ? String(entry.node.sourceId) : ""; context.save(); context.globalAlpha = entry?.active && (!searchActive || matchedClusters.has(clusterKey)) ? 1 : .2; context.fillStyle = entry ? palette.nodeColors.get(entry.node.id) || VISUALIZATION_NOISE_COLOR : VISUALIZATION_NOISE_COLOR; context.fillRect(label.x, label.y, label.width, label.height); context.fillStyle = label.contrast.foreground; context.font = "600 12px system-ui, sans-serif"; context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(label.text, label.x + label.width / 2, label.y + label.height / 2); context.restore(); } this.visualizationRenderedPointIndices = points.map((_point, index) => index); this.visualizationSpatialIndex = buildVisualizationPointSpatialIndex(points, pointHitRadius * 2); const pool = this.visualizationSpatialIndex.queryRect(-pointHitRadius, -pointHitRadius, width + pointHitRadius, height + pointHitRadius).slice(0, 96); const poolSet = new Set(pool); for (const [index, hit] of pointHitButtons) if (!poolSet.has(index)) { hit.remove(); pointHitButtons.delete(index); } this.visualizationHitElements = pool.map((index) => { let hit = pointHitButtons.get(index); if (!hit) { hit = hitLayer.createEl("button", { cls: "atomic-clusters-umap-point-hit", attr: { type: "button", "data-point-index": String(index), "aria-label": result.ids[index] } }); hit.addEventListener("mouseenter", (event) => this.setHoveredVisualizationPoint(index, event, hit!, () => draw(false))); hit.addEventListener("click", (event) => { event.preventDefault(); activatePoint(index); }); pointHitButtons.set(index, hit); } hit.style.left = `${points[index][0]}px`; hit.style.top = `${points[index][1]}px`; return hit; }); return true; };
    const updateLayerTransform = (): void => { if (!rasterCameraState || !this.visualizationCameraState) return; const source = visualizationCameraFromState(rasterCameraState); const target = visualizationCameraFromState(this.visualizationCameraState); const ratio = target.scale / source.scale; // The raster canvas starts at -overscan, so preserve that origin when scaling.
      const translateX = target.offsetX - ratio * source.offsetX + gestureOverscan * (1 - ratio); const translateY = target.offsetY - ratio * source.offsetY + gestureOverscan * (1 - ratio); visualLayer.style.transformOrigin = "0 0"; visualLayer.style.transform = `translate(${translateX}px, ${translateY}px) scale(${ratio})`; };
    const scheduleDensity = (): void => { if (densityTimer !== null) globalThis.clearTimeout(densityTimer); densityTimer = globalThis.setTimeout(() => { densityTimer = null; visualLayer.style.transform = "none"; cachedKey = ""; cachedBitmap = null; draw(true); rasterCameraState = this.visualizationCameraState ? { ...this.visualizationCameraState } : null; }, 100) as unknown as number; };
    const scheduleResize = (size: [number, number]): void => { pendingResize = size; if (resizeTimer !== null) globalThis.clearTimeout(resizeTimer); const oldWidth = width; const oldHeight = height; if (oldWidth > 0 && oldHeight > 0) { visualLayer.style.transformOrigin = "0 0"; visualLayer.style.transform = `scale(${size[0] / oldWidth}, ${size[1] / oldHeight})`; } resizeTimer = globalThis.setTimeout(() => { resizeTimer = null; const nextSize = pendingResize; pendingResize = null; if (!nextSize) return; visualLayer.style.transform = "none"; cachedKey = ""; cachedBitmap = null; cachedClouds = []; draw(true); rasterCameraState = this.visualizationCameraState ? { ...this.visualizationCameraState } : null; }, 100) as unknown as number; };
    const fit = navigation.createEl("button", { text: "Fit all", attr: { type: "button", "aria-label": "Fit all notes in view" } }); fit.addEventListener("click", () => { this.visualizationCameraState = visualizationFitCameraState(coordinates, width || 1, height || 1); visualLayer.style.transform = "none"; cachedKey = ""; cachedBitmap = null; draw(true); });
    let dragging = false; let moved = false; let startX = 0; let startY = 0; let startState: VisualizationCameraState | null = null; const onPointerDown = (event: PointerEvent): void => { if (event.button !== 0) return; dragging = true; moved = false; startX = event.clientX; startY = event.clientY; startState = this.visualizationCameraState ? { ...this.visualizationCameraState } : null; const target = event.currentTarget as (HTMLElement & { setPointerCapture?: (pointerId: number) => void }) | null; target?.setPointerCapture?.(event.pointerId); }; const onPointerMove = (event: PointerEvent): void => { if (!dragging || !startState) return; const dx = event.clientX - startX, dy = event.clientY - startY; if (!moved && Math.hypot(dx, dy) < 4) return; moved = true; suppressClick = true; this.visualizationCameraState = panVisualizationCamera(startState, dx, dy); updateLayerTransform(); }; const onPointerUp = (): void => { if (!dragging) return; dragging = false; if (moved) scheduleDensity(); startState = null; }; const onWheel = (event: WheelEvent): void => { event.preventDefault(); const rect = plot.getBoundingClientRect(); this.visualizationCameraState = zoomVisualizationCameraAt(this.visualizationCameraState || visualizationFitCameraState(coordinates, width || 1, height || 1), event.clientX - rect.left, event.clientY - rect.top, Math.exp(-event.deltaY * .001)); updateLayerTransform(); scheduleDensity(); };
    const onClick = (event: MouseEvent): void => { if (suppressClick) { suppressClick = false; return; } const pointButton = event.target instanceof HTMLElement && event.target.classList.contains("atomic-clusters-umap-point-hit"); if (pointButton) return; // Pointer capture can retarget a click to the gesture layer (or its hit-layer), so do not require the canvas to be the event target. All other descendants are part of the plot surface.
      if (event.target !== canvas && event.target !== visualLayer && event.target !== hitLayer) return; const rect = plot.getBoundingClientRect(); const x = event.clientX - rect.left, y = event.clientY - rect.top; const index = this.visualizationSpatialIndex?.queryNearest(x, y, pointHitRadius) ?? null; if (index !== null) { if (event.target === canvas) this.selectNote(result.ids[index]); else activatePoint(index); return; } const picked = pickVisualizationCloud(cachedClouds, x + gestureOverscan, y + gestureOverscan); if (picked !== null && frontier[picked]) activateCluster(frontier[picked].node); };
    const onMouseMove = (event: MouseEvent): void => { if (dragging) return; const rect = plot.getBoundingClientRect(); const index = this.visualizationSpatialIndex?.queryNearest(event.clientX - rect.left, event.clientY - rect.top, pointHitRadius) ?? null; this.setHoveredVisualizationPoint(index, event, canvas, () => draw(false)); }; const onMouseLeave = (): void => this.setHoveredVisualizationPoint(null, null, null, () => draw(false));
    visualLayer.addEventListener("pointerdown", onPointerDown); visualLayer.addEventListener("pointermove", onPointerMove); visualLayer.addEventListener("pointerup", onPointerUp); visualLayer.addEventListener("pointercancel", onPointerUp); visualLayer.addEventListener("wheel", onWheel, { passive: false }); visualLayer.addEventListener("click", onClick); canvas.addEventListener("mousemove", onMouseMove); canvas.addEventListener("mouseleave", onMouseLeave); this.visualizationCleanup = () => { if (densityTimer !== null) globalThis.clearTimeout(densityTimer); if (resizeTimer !== null) globalThis.clearTimeout(resizeTimer); pendingResize = null; visualLayer.removeEventListener("pointerdown", onPointerDown); visualLayer.removeEventListener("pointermove", onPointerMove); visualLayer.removeEventListener("pointerup", onPointerUp); visualLayer.removeEventListener("pointercancel", onPointerUp); visualLayer.removeEventListener("wheel", onWheel); visualLayer.removeEventListener("click", onClick); canvas.removeEventListener("mousemove", onMouseMove); canvas.removeEventListener("mouseleave", onMouseLeave); this.visualizationHitElements = []; this.visualizationSpatialIndex = null; }; draw(true); if (typeof ResizeObserver === "function") { this.visualizationResizeObserver = new ResizeObserver((entries) => { const entry = entries?.[0]; const rect = entry?.contentRect; const nextSize: [number, number] = rect && rect.width > 0 && rect.height > 0 ? [Math.round(rect.width), Math.round(rect.height)] : rectSize() || [width, height]; if (nextSize[0] === width && nextSize[1] === height) return; scheduleResize(nextSize); }); this.visualizationResizeObserver.observe(plot); }
    const onAccessiblePointClick = (event: MouseEvent): void => {
      const target = event.target;
      if (!(target instanceof HTMLElement) || !target.classList.contains("atomic-clusters-umap-point-hit")) return;
      const index = Number(target.dataset.pointIndex);
      if (!Number.isSafeInteger(index) || index < 0 || index >= result.ids.length) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.selectNote(result.ids[index]);
    };
    // Point buttons are the keyboard-accessible semantic layer. Capture their
    // click before the camera-navigation handler can interpret the same point
    // as a cluster target.
    hitLayer.addEventListener("click", onAccessiblePointClick, true);
    const previousVisualizationCleanup = this.visualizationCleanup;
    this.visualizationCleanup = () => {
      previousVisualizationCleanup?.();
      hitLayer.removeEventListener("click", onAccessiblePointClick, true);
    };
    rasterCameraState = this.visualizationCameraState ? { ...this.visualizationCameraState } : null;
  }
  /**
   * Render the Explorer with one camera state.  The bitmap may be moved while
   * a gesture is active, but that movement is always the exact affine mapping
   * from the raster camera to the current constrained camera.  No resize or
   * settle path invents a second camera position.
   */
  private renderGlobalVisualizationReliable(): void {
    const result = this.result!; const displayTitles = this.effectiveTitles(result); const visualization = result.visualization; const ordering = visualizationLeafOrdering(result);
    if (!validateVisualizationData(result) || !visualization) {
      if (result.schemaVersion >= 6) { const message = this.visualizationGenerationError || (this.ensureVisualization ? "Preparing visualization…" : "Visualization will be prepared when the explorer is connected to a projection worker."); this.contentEl.createDiv({ text: message }).addClass("atomic-clusters-status"); if (this.ensureVisualization && !this.visualizationGenerationError) this.scheduleVisualizationGeneration(result); return; }
      this.contentEl.createDiv({ text: "Hierarchical Gaussian-cloud visualization is unavailable for this saved result." }).addClass("atomic-clusters-status"); return;
    }

    const coordinates = visualization.coordinates; const labels = visualization.labels; const memberships = result.memberships!;
    this.visualizationRoot = buildVisualizationTree(result.hierarchy, labels); const root = this.visualizationRoot; const palette = visualizationColorScheme(root);
    const frontier = visualizationGlobalDepthFrontier(root, this.visualizationDepth, this.visualizationSelectedNodeId, memberships, ordering, result.hierarchyPlacements);
    const selectedNode = this.visualizationSelectedNodeId ? visualizationPath(root, this.visualizationSelectedNodeId).at(-1) || null : null;
    const cameraCoordinates = (selectedNode ? selectedNode.pointIndices.map((index) => coordinates[index]).filter((point): point is [number, number] => !!point && point.every(Number.isFinite)) : coordinates).length
      ? (selectedNode ? selectedNode.pointIndices.map((index) => coordinates[index]).filter((point): point is [number, number] => !!point && point.every(Number.isFinite)) : coordinates)
      : coordinates;
    const searchActive = this.isSearchActive(); const matchedNotes = new Set(this.searchResult?.notePaths || []); const matchedClusters = new Set(this.searchResult?.clusterIds || []);
    const frame = this.contentEl.createDiv({ cls: "atomic-clusters-umap" }); const plot = frame.createDiv({ cls: "atomic-clusters-umap-plot" }); const navigation = plot.createDiv({ cls: "atomic-clusters-umap-navigation" });
    const selectedPath = this.visualizationSelectedNodeId ? visualizationPath(root, this.visualizationSelectedNodeId) : [root];
    const back = navigation.createEl("button", { text: "Back", attr: { type: "button", "aria-label": "Back to parent cluster" } }); back.disabled = this.visualizationDepth <= 0 && !this.visualizationSelectedNodeId;
    back.addEventListener("click", () => { if (this.visualizationDepth <= 0) this.visualizationSelectedNodeId = null; else { this.visualizationDepth--; const path = this.visualizationSelectedNodeId ? visualizationPath(root, this.visualizationSelectedNodeId) : []; this.visualizationSelectedNodeId = path.length > 2 ? path[path.length - 2].id : null; } this.focusNodeId = this.visualizationSelectedNodeId; this.visualizationNodeId = this.visualizationSelectedNodeId || "root"; this.visualizationCameraState = null; this.render(); });
    const crumbs = navigation.createDiv({ cls: "atomic-clusters-breadcrumb" }); selectedPath.forEach((node, index) => { if (index) crumbs.createSpan({ text: " / " }); const title = node.sourceId === null ? "All notes" : displayTitles[String(node.sourceId)] || `Cluster ${node.sourceId}`; const crumb = crumbs.createEl("button", { text: title, attr: { type: "button", "aria-current": index === selectedPath.length - 1 ? "page" : "false" } }); crumb.addEventListener("click", () => { if (node.id === "root") { this.visualizationDepth = 0; this.visualizationSelectedNodeId = null; this.focusNodeId = null; } else { this.visualizationSelectedNodeId = node.id; this.focusNodeId = node.id; } this.visualizationCameraState = null; this.render(); }); });

    const transition = this.visualizationTransition; this.visualizationTransition = null;
    const outgoingLayer = transition ? plot.createDiv({ cls: "atomic-clusters-umap-outgoing-layer" }) : null;
    if (outgoingLayer && transition) {
      outgoingLayer.style.width = transition.snapshot.style.width || `${transition.snapshot.width}px`; outgoingLayer.style.height = transition.snapshot.style.height || `${transition.snapshot.height}px`; outgoingLayer.style.visibility = "visible"; transition.snapshot.style.position = "absolute"; transition.snapshot.style.left = "0"; transition.snapshot.style.top = "0"; transition.snapshot.style.display = "block"; outgoingLayer.appendChild(transition.snapshot);
    }
    const visualLayer = plot.createDiv({ cls: "atomic-clusters-umap-visual-layer" }); if (transition) visualLayer.style.visibility = "hidden";
    const canvas = visualLayer.createEl("canvas", { cls: "atomic-clusters-umap-canvas" }); canvas.setAttribute("role", "img"); canvas.setAttribute("aria-label", "Hierarchical Gaussian cloud visualization. Drag to pan; scroll to zoom; hover a note to preview it."); const hitLayer = visualLayer.createDiv({ cls: "atomic-clusters-umap-hit-layer" });
    const baseSigma = visualizationBaseBandwidth(coordinates); const pointHitRadius = 14; const pointRadius = VISUALIZATION_POINT_RADIUS; const hoverRadius = VISUALIZATION_HOVER_RING_RADIUS; const maxPointHitTargets = 96; const pointHitButtons = new Map<number, HTMLButtonElement>();
    const gestureOverscan = VISUALIZATION_GESTURE_OVERSCAN;
    type VisualizationViewport = [number, number];
    const normalizeViewport = (size: VisualizationViewport): VisualizationViewport => [Math.max(1, Math.round(size[0])), Math.max(1, Math.round(size[1]))];
    const equivalentViewport = (a: VisualizationViewport | null, b: VisualizationViewport | null): boolean => !!a && !!b && Math.abs(a[0] - b[0]) <= VISUALIZATION_VIEWPORT_TOLERANCE && Math.abs(a[1] - b[1]) <= VISUALIZATION_VIEWPORT_TOLERANCE;
    const rectSize = (): VisualizationViewport | null => { const rect = plot.getBoundingClientRect(); const width = rect.width || plot.clientWidth; const height = rect.height || plot.clientHeight; return width > 0 && height > 0 ? [width, height] : null; };
    const providedSize = (size?: VisualizationViewport): VisualizationViewport | null => size && size[0] > 0 && size[1] > 0 ? size : rectSize();
    const measureTextFallback = (context: CanvasRenderingContext2D, text: string): number => typeof context.measureText === "function" ? context.measureText(text).width : text.length * 7;
    let camera: VisualizationCamera | null = null; let points: [number, number][] = []; let width = 0; let height = 0; let renderedCamera: VisualizationCamera | null = null;
    let cachedKey = ""; let cachedBitmap: HTMLCanvasElement | null = null; let cachedClouds: VisualizationSplat[][] = []; let rasterCameraState: VisualizationCameraState | null = null; let densityTimer: number | null = null; let resizeTimer: number | null = null; let pendingResize: VisualizationViewport | null = null; let suppressClick = false;
    let dragging = false; let moved = false; let startX = 0; let startY = 0; let startState: VisualizationCameraState | null = null;

    const bitmapFor = (splats: readonly VisualizationSplat[], opacity: number): HTMLCanvasElement | null => {
      if (typeof document === "undefined") return null;
      const rasterWidth = width + gestureOverscan * 2; const rasterHeight = height + gestureOverscan * 2; const field = accumulateVisualizationDensity(splats, rasterWidth, rasterHeight); const bitmap = document.createElement("canvas"); bitmap.width = rasterWidth; bitmap.height = rasterHeight; const bitmapContext = bitmap.getContext("2d"); if (!bitmapContext) return null;
      const image = bitmapContext.createImageData(rasterWidth, rasterHeight); for (let index = 0; index < field.density.length; index++) { const density = field.density[index]; image.data[index * 4] = density > 0 ? Math.round(field.red[index] / density) : 0; image.data[index * 4 + 1] = density > 0 ? Math.round(field.green[index] / density) : 0; image.data[index * 4 + 2] = density > 0 ? Math.round(field.blue[index] / density) : 0; image.data[index * 4 + 3] = Math.round(visualizationDensityAlpha(density) * opacity * 255); } bitmapContext.putImageData(image, 0, 0); return bitmap;
    };
    const terminalPath = (index: number): string[] => visualizationNoteTerminalPath(root, index, labels, result.hierarchyPlacements); const pointActive = (index: number): boolean => (!this.visualizationSelectedNodeId || terminalPath(index).includes(this.visualizationSelectedNodeId)) && (!searchActive || matchedNotes.has(result.ids[index]));
    const cameraFitOptions = (context: CanvasRenderingContext2D) => ({ pointRadius, hoverPointRadius: hoverRadius, labelBoxes: measureVisualizationClusterLabelBoxes(frontier, displayTitles, (text) => measureTextFallback(context, text)), labelMargin: 8 });
    const selectedCoordinates = (node: VisualizationNode): [number, number][] => node.pointIndices.map((index) => coordinates[index]).filter((point): point is [number, number] => !!point && point.every(Number.isFinite));
    const activatePoint = (index: number): void => { const path = terminalPath(index); const selected = this.visualizationSelectedNodeId; if (selected && path[path.length - 1] === selected) { this.selectNote(result.ids[index]); return; } if (path.length === 1) { if (this.visualizationDepth === 0 && !selected) this.selectNote(result.ids[index]); else { this.visualizationSelectedNodeId = null; this.focusNodeId = null; this.visualizationDepth = 0; this.render(); } return; } const entry = frontier.find((item) => item.pointIndices.includes(index)); if (entry) { this.visualizationSelectedNodeId = entry.node.id; this.focusNodeId = entry.node.id; if (entry.node.children.length) this.visualizationDepth++; this.visualizationNodeId = entry.node.id; this.visualizationCameraState = visualizationFitCameraState(selectedCoordinates(entry.node), width || 1, height || 1); this.render(); return; } const next = path.find((id) => id !== "root"); if (next) { this.visualizationSelectedNodeId = next; this.focusNodeId = next; this.visualizationDepth++; this.visualizationNodeId = next; this.visualizationCameraState = visualizationFitCameraState(coordinates, width || 1, height || 1); this.render(); } };
    const activateCluster = (node: VisualizationNode): void => { this.visualizationSelectedNodeId = node.id; this.focusNodeId = node.id; if (node.children.length) this.visualizationDepth++; this.visualizationNodeId = node.id; this.visualizationCameraState = visualizationFitCameraState(selectedCoordinates(node), width || 1, height || 1); this.render(); };

    const draw = (reraster = true, viewport?: VisualizationViewport): boolean => {
      const rawSize = providedSize(viewport); if (!rawSize) return false; const size = normalizeViewport(rawSize); [width, height] = size; const rasterWidth = width + gestureOverscan * 2; const rasterHeight = height + gestureOverscan * 2; const dpr = Math.max(1, typeof window === "undefined" ? 1 : window.devicePixelRatio || 1);
      canvas.width = rasterWidth * dpr; canvas.height = rasterHeight * dpr; canvas.style.left = `${-gestureOverscan}px`; canvas.style.top = `${-gestureOverscan}px`; canvas.style.width = `${rasterWidth}px`; canvas.style.height = `${rasterHeight}px`; const context = canvas.getContext("2d"); if (!context) return false; context.setTransform(dpr, 0, 0, dpr, 0, 0); context.clearRect(0, 0, rasterWidth, rasterHeight);
      const fitOptions = cameraFitOptions(context); const current = this.visualizationCameraState; const fitted = visualizationFitCameraState(cameraCoordinates, width, height, current?.padding, fitOptions);
      if (!current) this.visualizationCameraState = fitted;
      else if (current.width !== width || current.height !== height || !current.contentBounds || current.padding + 1e-6 < fitted.padding) this.visualizationCameraState = resizeVisualizationCameraState(current, cameraCoordinates, width, height, fitOptions);
      camera = visualizationCameraFromState(this.visualizationCameraState!); renderedCamera = camera; this.visualizationLastCamera = camera;
      points = coordinates.map((point) => visualizationWorldToScreen(camera!, point)); this.visualizationPoints = points; const rasterPoints = points.map(([x, y]) => [x + gestureOverscan, y + gestureOverscan] as [number, number]);
      const key = `${this.visualizationDepth}|${this.visualizationSelectedNodeId || ""}|${camera.scale}|${camera.offsetX}|${camera.offsetY}|${width}x${height}|${this.visualizationKernelScale}|${this.searchQuery}|${[...this.searchFilters].join(",")}`;
      if (reraster && key !== cachedKey) {
        cachedKey = key; cachedBitmap = null; const active: VisualizationSplat[] = []; const inactive: VisualizationSplat[] = []; cachedClouds = [];
        for (const entry of frontier) {
          const value = visualizationCloudColor(entry.node, palette); const color = /^#[0-9a-f]{6}$/i.test(value) ? [parseInt(value.slice(1, 3), 16), parseInt(value.slice(3, 5), 16), parseInt(value.slice(5, 7), 16)] as [number, number, number] : visualizationColorVector([], ordering); const splats: VisualizationSplat[] = [];
          for (const index of entry.pointIndices) { const point = rasterPoints[index]; if (!point) continue; splats.push({ x: point[0], y: point[1], sigma: visualizationScaledStageSigma(baseSigma, entry.remainingDepth, !entry.node.children.length, this.visualizationKernelScale) * camera.scale, color, amplitude: visualizationMembershipAmplitude(memberships[index] || [], visualizationP95RowSum(memberships)) }); }
          cachedClouds.push(splats); const clusterKey = entry.node.sourceId === null ? "root" : String(entry.node.sourceId); (entry.active && (!searchActive || matchedClusters.has(clusterKey)) ? active : inactive).push(...splats);
        }
        const activeBitmap = bitmapFor(active, 1); const inactiveBitmap = bitmapFor(inactive, .2); cachedBitmap = document.createElement("canvas"); cachedBitmap.width = rasterWidth; cachedBitmap.height = rasterHeight; const merged = cachedBitmap.getContext("2d"); if (merged) { if (inactiveBitmap) merged.drawImage(inactiveBitmap, 0, 0); if (activeBitmap) merged.drawImage(activeBitmap, 0, 0); }
      }
      if (cachedBitmap) { context.imageSmoothingEnabled = true; context.drawImage(cachedBitmap, 0, 0); }
      const computedStyle = typeof getComputedStyle === "function" ? getComputedStyle(frame) : null; const background = computedStyle?.getPropertyValue("--background-primary").trim() || "transparent"; const residualDots = new Set(frontier.flatMap((entry) => entry.residualIndices)); const directResidualDots = new Set(frontier.flatMap((entry) => entry.directResidualIndices || []));
      this.visualizationRenderedPointIndices = points.map((_point, index) => index); this.visualizationRenderedPointIndices.forEach((index) => { const point = rasterPoints[index]; if (!point) return; const radius = index === this.hoveredVisualizationPoint ? VISUALIZATION_HOVER_POINT_RADIUS : pointRadius; context.beginPath(); const placement = result.hierarchyPlacements?.[index]; context.fillStyle = directResidualDots.has(index) ? "#6f757b" : labels[index] === -1 || residualDots.has(index) || placement?.kind === "residual" ? VISUALIZATION_NOISE_COLOR : blendVisualizationColor(memberships[index] || [], ordering, palette.leafColors); context.globalAlpha = index === this.hoveredVisualizationPoint ? 1 : pointActive(index) ? 1 : .2; context.arc(point[0], point[1], radius, 0, Math.PI * 2); context.fill(); context.globalAlpha = 1; context.strokeStyle = background; context.stroke(); });
      // Layout in viewport coordinates, then translate into the overscanned
      // raster.  Laying out directly in raster coordinates leaves labels at
      // x=-overscan or y=-overscan when a point is near an edge.
      const labelMargin = Math.min(Math.max(2, this.visualizationCameraState!.padding), Math.min(width, height) / 2); const labelPlacements = layoutVisualizationClusterLabels(frontier, points, displayTitles, palette.nodeColors, width, height, { margin: labelMargin, measureText: (text) => measureTextFallback(context, text) });
      if (typeof context.fillText === "function") for (const label of labelPlacements) {
        const x = label.x + gestureOverscan; const y = label.y + gestureOverscan; const right = x + label.width; const bottom = y + label.height; const radius = Math.min(label.height / 2, 8); context.save(); context.beginPath(); if (typeof context.roundRect === "function") context.roundRect(x, y, label.width, label.height, radius); else { context.moveTo(x + radius, y); context.lineTo(right - radius, y); context.arcTo(right, y, right, y + radius, radius); context.lineTo(right, bottom - radius); context.arcTo(right, bottom, right - radius, bottom, radius); context.lineTo(x + radius, bottom); context.arcTo(x, bottom, x, bottom - radius, radius); context.lineTo(x, y + radius); context.arcTo(x, y, x + radius, y, radius); } const entry = frontier.find((item) => item.node.id === label.id); const clusterKey = entry?.node.sourceId === null ? "root" : entry ? String(entry.node.sourceId) : ""; context.fillStyle = label.contrast.background; context.globalAlpha = entry?.active && (!searchActive || matchedClusters.has(clusterKey)) ? .92 : .2; context.fill(); context.globalAlpha = 1; context.strokeStyle = label.contrast.foreground; context.lineWidth = 2; context.stroke(); context.fillStyle = label.contrast.foreground; context.font = "600 12px system-ui, sans-serif"; context.textAlign = "center"; context.textBaseline = "middle"; context.fillText(label.text, x + label.width / 2, y + label.height / 2); context.restore();
      }
      this.visualizationSpatialIndex = buildVisualizationPointSpatialIndex(points, pointHitRadius * 2); const poolIndices = this.visualizationSpatialIndex.queryRect(-pointHitRadius, -pointHitRadius, width + pointHitRadius, height + pointHitRadius).slice(0, maxPointHitTargets); const poolSet = new Set(poolIndices);
      for (const [index, hit] of pointHitButtons) if (!poolSet.has(index)) { if (hit.parentElement === hitLayer) hitLayer.removeChild(hit); pointHitButtons.delete(index); }
      this.visualizationHitElements = poolIndices.map((index) => { let hit = pointHitButtons.get(index); if (!hit) { hit = hitLayer.createEl("button", { cls: "atomic-clusters-umap-point-hit", attr: { type: "button", "data-point-index": String(index), "aria-label": `Select note ${result.ids[index]}` } }); hit.addEventListener("click", (event) => { event.preventDefault(); this.selectNote(result.ids[index]); }); pointHitButtons.set(index, hit); } hit.style.left = `${points[index][0]}px`; hit.style.top = `${points[index][1]}px`; return hit; });
      if (this.hoveredVisualizationPoint !== null) { const point = points[this.hoveredVisualizationPoint]; if (point) { context.beginPath(); context.strokeStyle = computedStyle?.color || "currentColor"; context.lineWidth = 1.5; context.arc(point[0] + gestureOverscan, point[1] + gestureOverscan, hoverRadius, 0, Math.PI * 2); context.stroke(); } } context.globalAlpha = 1;
      if (reraster) rasterCameraState = { ...this.visualizationCameraState! }; if (!transition) this.visualizationDisplayedCamera = camera; return true;
    };

    const redrawHover = (): void => { if (visualLayer.style.transform && visualLayer.style.transform !== "none") return; draw(false); };
    const clearHover = (): void => this.setHoveredVisualizationPoint(null, null, null, redrawHover);
    const updateInteractionIndex = (): void => { if (!this.visualizationCameraState) return; const interactionCamera = visualizationCameraFromState(this.visualizationCameraState); const interactionPoints = coordinates.map((point) => visualizationWorldToScreen(interactionCamera, point)); this.visualizationPoints = interactionPoints; this.visualizationSpatialIndex = buildVisualizationPointSpatialIndex(interactionPoints, pointHitRadius * 2); };
    const updateLayerTransform = (): void => { if (!rasterCameraState || !this.visualizationCameraState) return; const source = visualizationCameraFromState(rasterCameraState); const target = visualizationCameraFromState(this.visualizationCameraState); const ratio = target.scale / source.scale; const translateX = target.offsetX - ratio * source.offsetX; const translateY = target.offsetY - ratio * source.offsetY; visualLayer.style.transformOrigin = "0 0"; visualLayer.style.transform = `translate(${translateX}px, ${translateY}px) scale(${ratio})`; updateInteractionIndex(); };
    const scheduleDensity = (): void => { if (densityTimer !== null) globalThis.clearTimeout(densityTimer); densityTimer = globalThis.setTimeout(() => { densityTimer = null; visualLayer.style.transform = "none"; cachedKey = ""; cachedBitmap = null; draw(true); }, 100) as unknown as number; };
    const scheduleResize = (size: VisualizationViewport): void => { pendingResize = size; if (densityTimer !== null) { globalThis.clearTimeout(densityTimer); densityTimer = null; } if (resizeTimer !== null) globalThis.clearTimeout(resizeTimer); clearHover(); resizeTimer = globalThis.setTimeout(() => { resizeTimer = null; const nextSize = pendingResize; pendingResize = null; if (!nextSize) return; visualLayer.style.transform = "none"; cachedKey = ""; cachedBitmap = null; cachedClouds = []; draw(true, nextSize); }, 100) as unknown as number; };
    const fit = navigation.createEl("button", { text: "Fit all", attr: { type: "button", "aria-label": "Fit all notes in view" } }); fit.addEventListener("click", () => { this.visualizationCameraState = visualizationFitCameraState(coordinates, width || 1, height || 1); visualLayer.style.transform = "none"; cachedKey = ""; cachedBitmap = null; draw(true); });

    const pointerPosition = (event: MouseEvent | PointerEvent): [number, number] => { const rect = plot.getBoundingClientRect(); return [event.clientX - rect.left, event.clientY - rect.top]; };
    const updateHoverFromPointer = (event: MouseEvent | PointerEvent): void => { if (dragging) return; const [x, y] = pointerPosition(event); const index = this.visualizationSpatialIndex?.queryNearest(x, y, pointHitRadius) ?? null; const element = event.target instanceof HTMLElement ? event.target : null; const target = element?.classList.contains("atomic-clusters-umap-point-hit") ? element : canvas; this.setHoveredVisualizationPoint(index, event, target, redrawHover); };
    const onPointerDown = (event: PointerEvent): void => { if (event.button !== 0) return; dragging = true; moved = false; startX = event.clientX; startY = event.clientY; startState = this.visualizationCameraState ? { ...this.visualizationCameraState } : null; clearHover(); const target = event.currentTarget as (HTMLElement & { setPointerCapture?: (pointerId: number) => void }) | null; target?.setPointerCapture?.(event.pointerId); };
    const onPointerMove = (event: PointerEvent): void => { if (!dragging || !startState) { updateHoverFromPointer(event); return; } const dx = event.clientX - startX; const dy = event.clientY - startY; if (!moved && Math.hypot(dx, dy) < 4) return; moved = true; suppressClick = true; this.visualizationCameraState = panVisualizationCamera(startState, dx, dy); updateLayerTransform(); clearHover(); };
    const onPointerEnd = (event: PointerEvent): void => { const wasDragging = dragging; if (wasDragging && event.type === "pointerup" && moved && startState) { const dx = event.clientX - startX; const dy = event.clientY - startY; this.visualizationCameraState = panVisualizationCamera(startState, dx, dy); updateLayerTransform(); } dragging = false; const shouldSettle = wasDragging && moved; startState = null; clearHover(); if (shouldSettle) scheduleDensity(); };
    const onPointerLeave = (): void => clearHover();
    const onWheel = (event: WheelEvent): void => { event.preventDefault(); clearHover(); const [x, y] = pointerPosition(event); this.visualizationCameraState = zoomVisualizationCameraAt(this.visualizationCameraState || visualizationFitCameraState(cameraCoordinates, width || 1, height || 1), x, y, Math.exp(-event.deltaY * .001)); updateLayerTransform(); scheduleDensity(); };
    const onClick = (event: MouseEvent): void => { if (suppressClick) { suppressClick = false; return; } const pointButton = event.target instanceof HTMLElement && event.target.classList.contains("atomic-clusters-umap-point-hit"); if (pointButton) return; if (event.target !== canvas && event.target !== visualLayer && event.target !== hitLayer) return; const [x, y] = pointerPosition(event); const index = this.visualizationSpatialIndex?.queryNearest(x, y, pointHitRadius) ?? null; if (index !== null) { if (event.target === canvas) this.selectNote(result.ids[index]); else activatePoint(index); return; } const picked = pickVisualizationCloud(cachedClouds, x + gestureOverscan, y + gestureOverscan); if (picked !== null && frontier[picked]) activateCluster(frontier[picked].node); };
    const onMouseMove = (event: MouseEvent): void => updateHoverFromPointer(event);

    visualLayer.addEventListener("pointerdown", onPointerDown); visualLayer.addEventListener("pointermove", onPointerMove); visualLayer.addEventListener("pointerup", onPointerEnd); visualLayer.addEventListener("pointercancel", onPointerEnd); visualLayer.addEventListener("pointerleave", onPointerLeave); visualLayer.addEventListener("wheel", onWheel, { passive: false }); visualLayer.addEventListener("click", onClick); visualLayer.addEventListener("mousemove", onMouseMove); visualLayer.addEventListener("mouseleave", onPointerLeave);
    this.visualizationCleanup = () => { if (densityTimer !== null) globalThis.clearTimeout(densityTimer); if (resizeTimer !== null) globalThis.clearTimeout(resizeTimer); pendingResize = null; visualLayer.removeEventListener("pointerdown", onPointerDown); visualLayer.removeEventListener("pointermove", onPointerMove); visualLayer.removeEventListener("pointerup", onPointerEnd); visualLayer.removeEventListener("pointercancel", onPointerEnd); visualLayer.removeEventListener("pointerleave", onPointerLeave); visualLayer.removeEventListener("wheel", onWheel); visualLayer.removeEventListener("click", onClick); visualLayer.removeEventListener("mousemove", onMouseMove); visualLayer.removeEventListener("mouseleave", onPointerLeave); this.visualizationHitElements = []; this.visualizationRenderedPointIndices = []; this.visualizationSpatialIndex = null; };

    let pendingTransition = transition;
    const clearLayerTransform = (): void => { visualLayer.style.transform = "none"; visualLayer.style.transformOrigin = ""; visualLayer.style.opacity = ""; visualLayer.style.visibility = ""; visualLayer.classList.remove("is-animating"); if (outgoingLayer?.parentElement) outgoingLayer.parentElement.removeChild(outgoingLayer); };
    const maybeStartTransition = (): void => { if (!pendingTransition || !renderedCamera) return; const currentTransition = pendingTransition; pendingTransition = null; if (!equivalentViewport([currentTransition.fromCamera.width, currentTransition.fromCamera.height], [renderedCamera.width, renderedCamera.height])) { this.visualizationDisplayedCamera = renderedCamera; clearLayerTransform(); return; } this.startVisualizationAnimation(outgoingLayer, visualLayer, currentTransition.snapshot, currentTransition.fromCamera, renderedCamera, clearLayerTransform); };
    const initialSize = rectSize(); let lastObservedPlotSize: VisualizationViewport | null = initialSize ? normalizeViewport(initialSize) : null;
    if (typeof ResizeObserver === "function") { this.visualizationResizeObserver = new ResizeObserver((entries) => { const entry = entries?.[0]; const rect = entry?.contentRect; const nextSize = rect && rect.width > 0 && rect.height > 0 ? normalizeViewport([rect.width, rect.height]) : rectSize(); if (!nextSize) return; const changed = !lastObservedPlotSize || Math.abs(nextSize[0] - lastObservedPlotSize[0]) > VISUALIZATION_VIEWPORT_TOLERANCE || Math.abs(nextSize[1] - lastObservedPlotSize[1]) > VISUALIZATION_VIEWPORT_TOLERANCE; lastObservedPlotSize = nextSize; if (!changed) { maybeStartTransition(); return; } if (this.visualizationAnimating) { this.cancelVisualizationAnimation(true); clearLayerTransform(); } scheduleResize(nextSize); }); this.visualizationResizeObserver.observe(plot); }
    draw(true); maybeStartTransition();
  }

  private captureVisualizationSnapshot(sourceCamera?: VisualizationCamera): HTMLCanvasElement | null {
    const source = this.contentEl.querySelector(".atomic-clusters-umap-canvas") as HTMLCanvasElement | null;
    if (!source || typeof document === "undefined") return null;
    const snapshot = document.createElement("canvas");
    snapshot.width = Math.max(1, source.width || 1); snapshot.height = Math.max(1, source.height || 1);
    const sourceRect = source.getBoundingClientRect();
    snapshot.style.width = `${Math.max(1, sourceRect.width || source.clientWidth || sourceCamera?.width || 1)}px`;
    snapshot.style.height = `${Math.max(1, sourceRect.height || source.clientHeight || sourceCamera?.height || 1)}px`;
    const context = snapshot.getContext("2d");
    if (!context) return null;
    context.drawImage(source, 0, 0);
    return snapshot;
  }
  private navigateVisualization(targetId: string, update: () => void): void {
    // Keep navigation locked while the outgoing image is moving. This avoids
    // having to transfer ownership of an in-flight snapshot and its listeners
    // to a second semantic stage; the next click can start once the target is
    // visible and owns the canvas again.
    if (targetId === this.visualizationNodeId || this.visualizationAnimating) return;
    const fromCamera = this.visualizationDisplayedCamera || this.visualizationLastCamera;
    const snapshot = fromCamera ? this.captureVisualizationSnapshot(fromCamera) : null;
    this.cancelVisualizationAnimation(false);
    this.visualizationTransition = fromCamera && snapshot ? { fromCamera: { ...fromCamera, worldRegion: { ...fromCamera.worldRegion } }, snapshot } : null;
    this.visualizationNodeId = targetId; update(); this.render();
  }
  private startVisualizationAnimation(outgoing: HTMLElement | null, target: HTMLElement, snapshot: HTMLCanvasElement, from: VisualizationCamera, to: VisualizationCamera, clear: () => void): void {
    const sameViewport = Math.abs(from.width - to.width) <= VISUALIZATION_VIEWPORT_TOLERANCE && Math.abs(from.height - to.height) <= VISUALIZATION_VIEWPORT_TOLERANCE;
    const reduced = typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!outgoing || snapshot.parentElement !== outgoing || !sameViewport || reduced || (from.scale === to.scale && from.offsetX === to.offsetX && from.offsetY === to.offsetY)) { this.visualizationDisplayedCamera = to; clear(); return; }
    this.cancelVisualizationAnimation(false); this.visualizationAnimating = true; this.visualizationAnimationTarget = to; const token = ++this.visualizationAnimationToken; const started = typeof performance !== "undefined" ? performance.now() : Date.now();
    outgoing.style.transformOrigin = "0 0"; outgoing.style.transform = "translate(0px, 0px) scale(1)"; outgoing.classList.add("is-animating"); outgoing.style.visibility = "visible";
    target.style.visibility = "hidden";
    const animationNow = (): number => typeof performance !== "undefined" ? performance.now() : Date.now();
    const requestFrame = (callback: FrameRequestCallback): number => typeof requestAnimationFrame === "function" ? requestAnimationFrame(callback) : setTimeout(() => callback(animationNow()), 16) as unknown as number;
    const tick = (now: number): void => {
      if (token !== this.visualizationAnimationToken) return;
      const progress = Math.max(0, Math.min(1, (now - started) / VISUALIZATION_CAMERA_TRANSITION_MS)); const transform = visualizationOutgoingLayerTransform(from, to, progress);
      outgoing.style.transformOrigin = "0 0"; outgoing.style.transform = `translate(${transform.translateX}px, ${transform.translateY}px) scale(${transform.scale})`;
      if (progress >= 1) { this.visualizationAnimating = false; this.visualizationAnimationFrame = null; this.visualizationAnimationTarget = null; this.visualizationDisplayedCamera = to; this.visualizationAnimationCleanup = null; clear(); return; }
      this.visualizationAnimationFrame = requestFrame(tick);
    };
    this.visualizationAnimationCleanup = clear;
    this.visualizationAnimationFrame = requestFrame(tick);
  }
  private cancelVisualizationAnimation(snap: boolean): void {
    this.visualizationAnimationToken++;
    if (this.visualizationAnimationFrame !== null) { if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(this.visualizationAnimationFrame); else clearTimeout(this.visualizationAnimationFrame); this.visualizationAnimationFrame = null; }
    if (snap && this.visualizationAnimationTarget) this.visualizationDisplayedCamera = this.visualizationAnimationTarget;
    this.visualizationAnimationTarget = null; this.visualizationAnimating = false; const cleanup = this.visualizationAnimationCleanup; this.visualizationAnimationCleanup = null; cleanup?.();
  }
  private clearVisualizationHoverSummary(): void {
    this.visualizationHoverSummaryToken++;
    if (this.visualizationHoverSummaryTimer !== null) { globalThis.clearTimeout(this.visualizationHoverSummaryTimer); this.visualizationHoverSummaryTimer = null; }
    this.leaf.hoverPopover?.hoverEl.querySelectorAll(".atomic-clusters-hover-membership-summary").forEach((element) => element.remove());
  }
  private scheduleVisualizationHoverSummary(point: number): void {
    const token = this.visualizationHoverSummaryToken; const result = this.result; if (!result) return;
    let summary: HTMLElement | null = null;
    const apply = (attempt: number): void => {
      if (token !== this.visualizationHoverSummaryToken || this.hoveredVisualizationPoint !== point || this.result !== result) return;
      this.visualizationHoverSummaryTimer = null;
      const hoverEl = this.leaf.hoverPopover?.hoverEl;
      if (!hoverEl) { if (attempt < 20) this.visualizationHoverSummaryTimer = globalThis.setTimeout(() => apply(attempt + 1), 50) as unknown as number; return; }
      const content = hoverEl.querySelector(".markdown-preview-view") || hoverEl;
      if (!summary) summary = createVisualizationHoverSummary(hoverEl.ownerDocument || document, result, point, this.effectiveTitles(result));
      const existing = Array.from(hoverEl.querySelectorAll(".atomic-clusters-hover-membership-summary")); existing.filter((element) => element !== summary).forEach((element) => element.remove());
      if (summary.parentElement !== content) content.insertBefore(summary, content.firstChild);
      if (attempt < 20) this.visualizationHoverSummaryTimer = globalThis.setTimeout(() => apply(attempt + 1), 50) as unknown as number;
    };
    this.visualizationHoverSummaryTimer = globalThis.setTimeout(() => apply(0), 0) as unknown as number;
  }
  private setHoveredVisualizationPoint(point: number | null, event: MouseEvent | null, target: HTMLElement | null, draw: () => void): void { if (this.visualizationAnimating) return; if (point === this.hoveredVisualizationPoint && target === this.hoveredVisualizationTarget) return; this.clearVisualizationHoverSummary(); this.hoveredVisualizationPoint = point; this.hoveredVisualizationTarget = target; draw(); if (point !== null && event && this.result?.ids[point]) { const pointButton = this.visualizationHitElements.find((hit) => Number(hit.dataset.pointIndex) === point); this.app.workspace.trigger("hover-link", { event, source: VIEW_TYPE_CLUSTER_EXPLORER, hoverParent: this.leaf, targetEl: target || pointButton || this.visualizationHitElements[0], linktext: this.result.ids[point], sourcePath: "" }); this.scheduleVisualizationHoverSummary(point); } }
  private disposeVisualization(): void { this.cancelVisualizationAnimation(false); this.visualizationCleanup?.(); this.visualizationCleanup = null; this.visualizationResizeObserver?.disconnect(); this.visualizationResizeObserver = null; this.clearVisualizationHoverSummary(); this.visualizationPoints = []; this.visualizationHitElements = []; this.hoveredVisualizationPoint = null; this.hoveredVisualizationTarget = null; }
}
