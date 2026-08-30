import { ClusterResult, HierarchyMerge, HierarchyPlacement, HierarchyTree, VisualizationCoordinate } from "./types";

const TAB20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"] as const;
export const VISUALIZATION_NOISE_COLOR = "#9aa0a6";
export const VISUALIZATION_POINT_PADDING = 18;
export const VISUALIZATION_KERNEL_SCALE_MIN = 0.25;
export const VISUALIZATION_KERNEL_SCALE_MAX = 2;
export const VISUALIZATION_KERNEL_SCALE_STEP = 0.05;
export const VISUALIZATION_KERNEL_SCALE_DEFAULT = 0.65;
const EPSILON = 1e-9;

export interface VisualizationNode { id: string; sourceId: number | null; children: VisualizationNode[]; leafLabels: number[]; pointIndices: number[]; depth: number; nary?: boolean; }
export interface NaryVisualizationHierarchy { leaves: number[]; root: number | null; children?: Record<string, number[]>; nodes?: Array<{ id: number; children: number[]; descendantLeaves?: number[] }>; rootChildren?: number[]; merges?: HierarchyMerge[]; }
export interface VisualizationFrontierEntry { node: VisualizationNode; depth: number; remainingDepth: number; pointIndices: number[]; residualIndices: number[]; directResidualIndices?: number[]; descendantResidualIndices?: number[]; /** true when a leaf was explicitly selected and its actual notes should be shown. */ actualPoints: boolean; }
/** An entry in the global depth cut used by the explorer.  Unlike the legacy
 * frontier, entries from every branch are returned at the same depth. */
export interface VisualizationDepthEntry extends VisualizationFrontierEntry {
  /** True when this entry lies on the currently selected path. */
  active: boolean;
  /** Opacity to use for this entry's cloud and label. */
  opacity: number;
  /** Root-to-entry ids, useful for assigning note emphasis without rescanning. */
  pathIds: string[];
}
export interface VisualizationCameraState { centerX: number; centerY: number; zoom: number; fitScale: number; width: number; height: number; padding: number; }
export interface VisualizationCamera { scale: number; offsetX: number; offsetY: number; width: number; height: number; worldRegion: { minX: number; maxX: number; minY: number; maxY: number }; }
export interface VisualizationCameraLayerTransform { scale: number; translateX: number; translateY: number; }
export interface VisualizationSplat { x: number; y: number; sigma: number; color: [number, number, number]; amplitude: number; }
export interface VisualizationDensityField { width: number; height: number; density: Float32Array; red: Float32Array; green: Float32Array; blue: Float32Array; }
export interface VisualizationHslColor { hue: number; saturation: number; lightness: number; }
export interface VisualizationColorScheme { nodeColors: Map<string, string>; leafColors: Map<number, string>; nodeHsl: Map<string, VisualizationHslColor>; }
export interface VisualizationLabelContrast { foreground: "#000000" | "#ffffff"; background: "#000000" | "#ffffff"; }
export interface VisualizationLabelPlacement { id: string; text: string; x: number; y: number; width: number; height: number; contrast: VisualizationLabelContrast; }
export interface VisualizationMembershipSummary { leafId: number; title: string; value: number; }

/**
 * A deterministic uniform-grid index for screen-space note picking.
 *
 * The canvas is the source of truth for interaction, so this index includes
 * every finite point even when only a bounded accessibility button pool is
 * mounted in the DOM.  Querying the few neighboring cells keeps pointer
 * movement independent of the total number of notes.
 */
export interface VisualizationPointSpatialIndex {
  readonly cellSize: number;
  readonly points: readonly VisualizationCoordinate[];
  queryNearest(x: number, y: number, radius: number): number | null;
  queryRect(minX: number, minY: number, maxX: number, maxY: number): number[];
}

function spatialCellKey(x: number, y: number): string { return `${x},${y}`; }

export function buildVisualizationPointSpatialIndex(points: readonly VisualizationCoordinate[], cellSize = 24): VisualizationPointSpatialIndex {
  const size = Number.isFinite(cellSize) && cellSize > 0 ? cellSize : 24;
  const buckets = new Map<string, number[]>();
  const cell = (value: number): number => Math.floor(value / size);
  for (let index = 0; index < points.length; index++) {
    const point = points[index];
    if (!point || point.length < 2 || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) continue;
    const key = spatialCellKey(cell(point[0]), cell(point[1]));
    const bucket = buckets.get(key);
    if (bucket) bucket.push(index); else buckets.set(key, [index]);
  }
  const queryRect = (minX: number, minY: number, maxX: number, maxY: number): number[] => {
    if (![minX, minY, maxX, maxY].every(Number.isFinite)) return [];
    const left = Math.floor(Math.min(minX, maxX) / size); const right = Math.floor(Math.max(minX, maxX) / size);
    const top = Math.floor(Math.min(minY, maxY) / size); const bottom = Math.floor(Math.max(minY, maxY) / size);
    const found: number[] = [];
    for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) {
      const bucket = buckets.get(spatialCellKey(x, y)); if (bucket) found.push(...bucket);
    }
    return found;
  };
  const queryNearest = (x: number, y: number, radius: number): number | null => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    const boundedRadius = Math.max(0, Number.isFinite(radius) ? radius : 0); const radiusSquared = boundedRadius ** 2;
    let closest: number | null = null; let closestDistance = radiusSquared;
    for (const index of queryRect(x - boundedRadius, y - boundedRadius, x + boundedRadius, y + boundedRadius)) {
      const point = points[index]; if (!point) continue;
      const distance = (point[0] - x) ** 2 + (point[1] - y) ** 2;
      // Match the historical helper's deterministic later-index tie break.
      if (distance <= closestDistance && (distance < closestDistance || closest === null || index >= closest)) { closest = index; closestDistance = distance; }
    }
    return closest;
  };
  return { cellSize: size, points, queryNearest, queryRect };
}

export const createVisualizationPointSpatialIndex = buildVisualizationPointSpatialIndex;

/** Build a generalized children[] tree. Binary left/right knowledge ends here. */
export function buildVisualizationTree(hierarchy: HierarchyTree | NaryVisualizationHierarchy, labels: readonly number[]): VisualizationNode {
  const mergeList = hierarchy.merges || []; const mergeIds = new Set<number>();
  for (const merge of mergeList) { if (!Number.isSafeInteger(merge.id) || mergeIds.has(merge.id)) throw new Error("Hierarchy contains malformed or duplicate node ids"); mergeIds.add(merge.id); }
  if (!Array.isArray(hierarchy.leaves) || hierarchy.leaves.some((label, index) => !Number.isSafeInteger(label) || hierarchy.leaves.indexOf(label) !== index || mergeIds.has(label))) throw new Error("Hierarchy contains malformed or duplicate leaf ids");
  const merges = new Map<number, HierarchyMerge>(mergeList.map((merge) => [merge.id, merge]));
  const naryChildren = new Map<number, number[]>((hierarchy.nodes || []).map((node) => [node.id, node.children]));
  const explicitChildren = "children" in hierarchy ? hierarchy.children : undefined;
  const visiting = new Set<number>(); const built = new Set<number>();
  const make = (sourceId: number, depth: number): VisualizationNode => {
    if (!Number.isSafeInteger(sourceId)) throw new Error("Hierarchy contains a malformed node id"); if (visiting.has(sourceId)) throw new Error("Hierarchy contains a cycle"); if (built.has(sourceId)) throw new Error("Hierarchy contains duplicate node references"); built.add(sourceId); visiting.add(sourceId);
    const merge = merges.get(sourceId); const futureChildren = naryChildren.get(sourceId) ?? explicitChildren?.[String(sourceId)]; if (futureChildren !== undefined && !Array.isArray(futureChildren)) throw new Error("Hierarchy children must be arrays"); const children = (futureChildren || (merge ? [merge.left, merge.right] : [])).map((child: number) => make(child, depth + 1)); visiting.delete(sourceId);
    const leafLabels = children.length ? children.flatMap((child) => child.leafLabels) : [sourceId]; const uniqueLeaves = [...new Set(leafLabels)]; const pointIndices = labels.map((label, index) => label === sourceId || uniqueLeaves.includes(label) ? index : -1).filter((index) => index >= 0); return { id: `node:${sourceId}`, sourceId, children, leafLabels: uniqueLeaves, pointIndices, depth };
  };
  const roots = hierarchy.root === null ? hierarchy.leaves.map((leaf) => make(leaf, 1)) : [make(hierarchy.root, 1)]; return { id: "root", sourceId: null, children: roots, leafLabels: [...new Set(roots.flatMap((node) => node.leafLabels))], pointIndices: labels.map((_label, index) => index), depth: 0, nary: !!(hierarchy.nodes || explicitChildren) };
}

export function childMembershipMass(node: VisualizationNode, row: readonly number[], leafOrdering: readonly number[]): number { const leaves = new Set(node.leafLabels); let mass = 0; for (let column = 0; column < Math.min(leafOrdering.length, row.length); column++) if (leaves.has(leafOrdering[column])) mass += Math.max(0, Number(row[column]) || 0); return mass; }
/** Return the highest soft-membership values for one note in display order. */
export function visualizationTopMemberships(result: Pick<ClusterResult, "memberships" | "softMemberships" | "leafOrder" | "leafOrdering" | "hierarchy" | "titles">, pointIndex: number, limit = 3): VisualizationMembershipSummary[] {
  const memberships = result.memberships ?? result.softMemberships; const ordering = result.leafOrdering ?? result.leafOrder ?? result.hierarchy.leaves; const row = memberships?.[pointIndex];
  if (!row || !Number.isSafeInteger(pointIndex) || pointIndex < 0) return [];
  const count = Number.isSafeInteger(limit) && limit > 0 ? limit : 3;
  return ordering.map((leafId, column) => ({ leafId, title: result.titles?.[String(leafId)]?.trim() || `Cluster ${leafId}`, value: Number(row[column]) }))
    .filter((item) => Number.isFinite(item.value) && item.value >= 0)
    .sort((left, right) => right.value - left.value || left.leafId - right.leafId)
    .slice(0, count);
}
/** Identify notes below 0.5 conditional membership as residual. */
export function residualPointIndices(node: VisualizationNode, _labels: readonly number[], memberships: readonly number[][], leafOrdering: readonly number[]): number[] { if (!node.children.length) return []; return node.pointIndices.filter((index) => { const row = memberships[index] || []; const masses = node.children.map((child) => childMembershipMass(child, row, leafOrdering)); const total = masses.reduce((sum, mass) => sum + mass, 0); return total <= EPSILON || Math.max(...masses) / total <= 0.5; }); }
function visualizationResidualIndices(node: VisualizationNode, memberships: readonly number[][], leafOrdering: readonly number[], strictLegacy = false): number[] { if (!strictLegacy) return residualPointIndices(node, [], memberships, leafOrdering); if (!node.children.length) return []; return node.pointIndices.filter((index) => { const row = memberships[index] || []; const masses = node.children.map((child) => childMembershipMass(child, row, leafOrdering)); const total = masses.reduce((sum, mass) => sum + mass, 0); return total <= EPSILON || Math.max(...masses) / total < 0.5; }); }
function visualizationResidualNodes(root: VisualizationNode, parent: VisualizationNode): VisualizationNode[] {
  const nodes = visualizationPath(root, parent.id); if (!root.nary) return nodes;
  const descendants: VisualizationNode[] = []; const visit = (node: VisualizationNode) => { descendants.push(node); node.children.forEach(visit); }; visit(parent);
  return descendants;
}

/**
 * Return the visible stage for one focused node.
 *
 * The synthetic root is a bookkeeping node, not a cluster that should be
 * rendered.  With no focus (or with the synthetic root focused), the first
 * real hierarchy root is therefore opened one level and its immediate
 * children are shown.  Every subsequent call shows only the focused node's
 * immediate children.  A leaf is the sole exception: selecting it exposes
 * its note points, which keeps notes hidden at every cluster stage.
 *
 * `expandedIds` is retained in the public signature for saved callers, but it
 * now represents the focused node (the last valid id), rather than a set of
 * independently expanded branches.
 */
export function visualizationFrontier(root: VisualizationNode, expandedIds: Iterable<string> = [], memberships?: readonly number[][], leafOrdering?: readonly number[], placements?: readonly HierarchyPlacement[]): VisualizationFrontierEntry[] {
  const ids = [...expandedIds];
  const find = (node: VisualizationNode, id: string): VisualizationNode | null => {
    if (node.id === id) return node;
    for (const child of node.children) { const found = find(child, id); if (found) return found; }
    return null;
  };
  const requested = ids.length ? ids[ids.length - 1] : "root";
  const focused = requested === "root" ? null : find(root, requested);
  const overview = !focused;
  const hierarchyRoot = root.children.length === 1 && root.children[0].children.length ? root.children[0] : root;
  const parent = focused || hierarchyRoot;
  const maxDepth = visualizationTreeDepth(root);
  const placementAware = !!placements && placements.length === root.pointIndices.length;
  const subtreeIds = new Set<number>(); const subtreeLeaves = new Set<number>();
  const collect = (node: VisualizationNode): void => { if (node.sourceId !== null) subtreeIds.add(node.sourceId); if (!node.children.length && node.sourceId !== null) subtreeLeaves.add(node.sourceId); node.children.forEach(collect); };
  collect(parent);
  const inFocusedSubtree = (placement: HierarchyPlacement): boolean => overview
    ? true
    : placement.nodeId !== null && (placement.kind === "leaf" ? subtreeLeaves.has(placement.nodeId) : subtreeIds.has(placement.nodeId));
  const placedResiduals = placementAware ? placements!.map((placement, index) => placement.kind === "residual" && inFocusedSubtree(placement) ? index : -1).filter((index) => index >= 0) : [];
  const directResiduals = placementAware ? placedResiduals.filter((index) => placements![index].nodeId === parent.sourceId) : [];
  const directResidualSet = new Set(directResiduals); const descendantResiduals = placementAware ? placedResiduals.filter((index) => !directResidualSet.has(index)) : [];
  const placedLeafPoints = (node: VisualizationNode): number[] => placementAware
    ? placements!.map((placement, index) => placement.kind === "leaf" && placement.nodeId !== null && node.leafLabels.includes(placement.nodeId) ? index : -1).filter((index) => index >= 0)
    : node.pointIndices.slice();
  if (!parent.children.length) {
    const residual = placementAware ? new Set(placedResiduals) : memberships && leafOrdering
      ? new Set(visualizationResidualNodes(root, parent).filter((node) => node.children.length).flatMap((node) => visualizationResidualIndices(node, memberships, leafOrdering, !root.nary)))
      : new Set<number>();
    return [{ node: parent, depth: parent.depth, remainingDepth: Math.max(0, maxDepth - parent.depth), pointIndices: placedLeafPoints(parent).filter((point) => !residual.has(point)), residualIndices: [...residual], directResidualIndices: directResiduals, descendantResidualIndices: descendantResiduals, actualPoints: true }];
  }
  // Residual/noise is emitted once at the current boundary. Normal cluster
  // members remain in their cloud and are not exposed as note dots until a
  // leaf is selected. Include every internal ancestor so synthetic-root noise
  // and residuals classified at a previous split survive zooming.
  const residual = placementAware ? new Set(placedResiduals) : memberships && leafOrdering
    ? new Set(visualizationResidualNodes(root, parent).filter((node) => node.children.length).flatMap((node) => visualizationResidualIndices(node, memberships, leafOrdering, !root.nary)))
    : new Set<number>();
  const residualIndices = [...residual];
  return parent.children.map((node) => ({
    node,
    depth: node.depth,
    remainingDepth: Math.max(0, maxDepth - node.depth),
    pointIndices: placedLeafPoints(node).filter((point) => !residual.has(point)),
    residualIndices: node === parent.children[0] ? residualIndices : [],
    directResidualIndices: node === parent.children[0] ? directResiduals : [],
    descendantResidualIndices: node === parent.children[0] ? descendantResiduals : [],
    actualPoints: false,
  }));
}

/** Return the real hierarchy roots (the adapter's synthetic root is omitted). */
export function visualizationHierarchyRoots(root: VisualizationNode): VisualizationNode[] {
  return root.children.length === 1 && root.children[0].children.length ? root.children[0].children : root.children;
}

function visualizationDescendsFrom(node: VisualizationNode, ancestorId: string): boolean {
  if (node.id === ancestorId) return true;
  return node.children.some((child) => visualizationDescendsFrom(child, ancestorId));
}

function visualizationPathIds(root: VisualizationNode, targetId: string): string[] {
  return visualizationPath(root, targetId).map((node) => node.id);
}

/**
 * Build a cut through every hierarchy branch at one global depth.  A leaf
 * reached before the requested depth remains in the cut, so an unbalanced
 * hierarchy never makes a note cloud disappear while another branch expands.
 * `depth` is relative to the real hierarchy roots (zero is the first split).
 */
export function visualizationGlobalDepthFrontier(
  root: VisualizationNode,
  depth = 0,
  selectedNodeId: string | null = null,
  memberships?: readonly number[][],
  leafOrdering?: readonly number[],
  placements?: readonly HierarchyPlacement[],
): VisualizationDepthEntry[] {
  const roots = visualizationHierarchyRoots(root);
  const requestedDepth = Math.max(0, Number.isFinite(depth) ? Math.floor(depth) : 0);
  const selectedPath = selectedNodeId ? visualizationPath(root, selectedNodeId) : [];
  const selected = selectedNodeId && visualizationPath(root, selectedNodeId).some((node) => node.id === selectedNodeId)
    ? selectedNodeId : null;
  const selectedNode = selected ? selectedPath[selectedPath.length - 1] : null;
  const placedResiduals = placements ? new Set(placements.map((placement, index) => placement.kind === "residual" ? index : -1).filter((index) => index >= 0)) : new Set<number>();
  const entries: VisualizationDepthEntry[] = [];
  const walk = (node: VisualizationNode, relativeDepth: number, parentPath: string[]): void => {
    const pathIds = [...parentPath, node.id];
    if (!node.children.length || relativeDepth >= requestedDepth) {
      const active = !selected || (visualizationDescendsFrom(node, selected) || (!!selectedNode && visualizationDescendsFrom(selectedNode, node.id)));
      const residualIndices = placements
        ? placements.map((placement, index) => placement.kind === "residual" && visualizationNoteTerminalPath(root, index, undefined, placements).includes(node.id) ? index : -1).filter((index) => index >= 0)
        : memberships && leafOrdering ? residualPointIndices(node, [], memberships, leafOrdering) : [];
      // Leaves are still clouds in the global cut.  Their note dots are
      // rendered at every depth, so selecting a leaf changes emphasis only;
      // it must not remove its title/cloud from the stage.
      const excluded = new Set([...placedResiduals, ...residualIndices]);
      entries.push({ node, depth: relativeDepth, remainingDepth: Math.max(0, visualizationTreeDepth(root) - node.depth), pointIndices: node.pointIndices.filter((index) => !excluded.has(index)), residualIndices, directResidualIndices: residualIndices, descendantResidualIndices: [], actualPoints: false, active, opacity: active ? 1 : 0.2, pathIds });
      return;
    }
    node.children.forEach((child) => walk(child, relativeDepth + 1, pathIds));
  };
  roots.forEach((node) => walk(node, 0, ["root"]));
  return entries;
}
export const globalVisualizationDepthFrontier = visualizationGlobalDepthFrontier;
export const visualizationDepthCut = visualizationGlobalDepthFrontier;
export const visualizationGlobalDepthEntries = visualizationGlobalDepthFrontier;

/** Return the hierarchy path containing a note, including residual locations. */
export function visualizationNoteTerminalPath(
  root: VisualizationNode,
  pointIndex: number,
  labels?: readonly number[],
  placements?: readonly HierarchyPlacement[],
): string[] {
  const placement = placements?.[pointIndex];
  if (placement) {
    if (placement.nodeId === null) return ["root"];
    return visualizationPathIds(root, `node:${placement.nodeId}`);
  }
  const label = labels?.[pointIndex];
  if (typeof label !== "number" || !Number.isSafeInteger(label) || label < 0) return ["root"];
  const leaf = visualizationPath(root, `node:${label}`);
  return leaf.length && leaf.at(-1)?.id === `node:${label}` ? leaf.map((node) => node.id) : ["root"];
}
export const visualizationTerminalPath = visualizationNoteTerminalPath;

/** Return the terminal node id for a note (root denotes root noise). */
export function visualizationNoteTerminalNode(root: VisualizationNode, pointIndex: number, labels?: readonly number[], placements?: readonly HierarchyPlacement[]): string {
  const path = visualizationNoteTerminalPath(root, pointIndex, labels, placements);
  return path[path.length - 1] || "root";
}

export function clampVisualizationZoom(value: number): number {
  if (Number.isNaN(value)) return 1;
  if (value === Number.POSITIVE_INFINITY) return 16;
  if (value === Number.NEGATIVE_INFINITY) return 0.5;
  return Math.max(0.5, Math.min(16, Number.isFinite(value) ? value : 1));
}
export const VISUALIZATION_ZOOM_MIN = 0.5;
export const VISUALIZATION_ZOOM_MAX = 16;

/** Fit all world coordinates and return a pan/zoom state centered on them. */
export function visualizationFitCameraState(coordinates: readonly VisualizationCoordinate[], width: number, height: number, padding = VISUALIZATION_POINT_PADDING): VisualizationCameraState {
  const region = visualizationRegion({ pointIndices: coordinates.map((_point, index) => index) } as VisualizationNode, coordinates);
  const safeWidth = Math.max(1, Number.isFinite(width) ? width : 1), safeHeight = Math.max(1, Number.isFinite(height) ? height : 1);
  const safePadding = Math.max(0, Math.min(Number.isFinite(padding) ? padding : VISUALIZATION_POINT_PADDING, Math.min(safeWidth, safeHeight) / 2));
  const fitScale = Math.min((safeWidth - safePadding * 2) / Math.max(EPSILON, region.maxX - region.minX), (safeHeight - safePadding * 2) / Math.max(EPSILON, region.maxY - region.minY));
  return { centerX: (region.minX + region.maxX) / 2, centerY: (region.minY + region.maxY) / 2, zoom: 1, fitScale: Math.max(EPSILON, fitScale), width: safeWidth, height: safeHeight, padding: safePadding };
}
export const fitVisualizationCamera = visualizationFitCameraState;
export const createVisualizationCameraState = visualizationFitCameraState;
/** Resize a camera while preserving its world center and user zoom. */
export function resizeVisualizationCameraState(state: VisualizationCameraState, coordinates: readonly VisualizationCoordinate[], width: number, height: number): VisualizationCameraState {
  const fitted = visualizationFitCameraState(coordinates, width, height, state.padding);
  return { ...fitted, centerX: state.centerX, centerY: state.centerY, zoom: clampVisualizationZoom(state.zoom) };
}
export const resizeVisualizationCamera = resizeVisualizationCameraState;

/** Convert a pan/zoom state into the camera consumed by world/screen helpers. */
export function visualizationCameraFromState(state: VisualizationCameraState): VisualizationCamera {
  const scale = Math.max(EPSILON, state.fitScale * clampVisualizationZoom(state.zoom));
  return { scale, offsetX: state.width / 2 - state.centerX * scale, offsetY: state.height / 2 + state.centerY * scale, width: state.width, height: state.height, worldRegion: { minX: state.centerX - state.width / (2 * scale), maxX: state.centerX + state.width / (2 * scale), minY: state.centerY - state.height / (2 * scale), maxY: state.centerY + state.height / (2 * scale) } };
}
export const cameraFromVisualizationState = visualizationCameraFromState;
export const visualizationPanZoomCamera = visualizationCameraFromState;

/** Pan by screen pixels (dragging right moves the world right). */
export function panVisualizationCamera(state: VisualizationCameraState, deltaX: number, deltaY: number): VisualizationCameraState {
  const camera = visualizationCameraFromState(state); const dx = Number.isFinite(deltaX) ? deltaX : 0; const dy = Number.isFinite(deltaY) ? deltaY : 0;
  return { ...state, centerX: state.centerX - dx / camera.scale, centerY: state.centerY + dy / camera.scale };
}
export const panCamera = panVisualizationCamera;

/** Zoom around a screen point while preserving the world coordinate beneath it. */
export function zoomVisualizationCameraAt(state: VisualizationCameraState, screenX: number, screenY: number, factor: number): VisualizationCameraState {
  const before = visualizationCameraFromState(state); const world = visualizationScreenToWorld(before, [screenX, screenY]);
  const zoom = clampVisualizationZoom(state.zoom * (Number.isFinite(factor) && factor > 0 ? factor : 1));
  const next = visualizationCameraFromState({ ...state, zoom }); const centerX = world[0] - (screenX - next.width / 2) / next.scale; const centerY = world[1] + (screenY - next.height / 2) / next.scale;
  return { ...state, zoom, centerX, centerY };
}
export const zoomVisualizationCamera = zoomVisualizationCameraAt;
export const zoomCameraAt = zoomVisualizationCameraAt;
export function collapseVisualizationBranch(expandedIds: Iterable<string>, targetId: string): string[] { return [...expandedIds].filter((id) => id !== targetId && !id.startsWith(`${targetId}/`)); }
export function visualizationTreeDepth(root: VisualizationNode): number { return root.children.reduce((depth, child) => Math.max(depth, visualizationTreeDepth(child)), root.depth); }
export function visualizationRemainingDepth(node: VisualizationNode, root: VisualizationNode): number { return Math.max(0, visualizationTreeDepth(root) - node.depth); }
export const computeVisualizationDepth = visualizationRemainingDepth;

function distanceSquared(a: VisualizationCoordinate, b: VisualizationCoordinate): number { return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2; }
function worldDiagonal(coordinates: readonly VisualizationCoordinate[]): number { if (!coordinates.length) return 1; const xs = coordinates.map(([x]) => x).filter(Number.isFinite); const ys = coordinates.map(([, y]) => y).filter(Number.isFinite); if (!xs.length || !ys.length) return 1; return Math.max(Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)), Number.EPSILON); }
/** Robust world-space bandwidth: median third-nearest distance, scaled and clamped. */
export function visualizationBaseBandwidth(coordinates: readonly VisualizationCoordinate[]): number {
  const diagonal = worldDiagonal(coordinates); if (coordinates.length < 4) return diagonal / 50; const thirdNearest: number[] = [];
  for (let i = 0; i < coordinates.length; i++) { let first = Number.POSITIVE_INFINITY, second = Number.POSITIVE_INFINITY, third = Number.POSITIVE_INFINITY; for (let j = 0; j < coordinates.length; j++) { if (i === j) continue; const distance = Math.sqrt(distanceSquared(coordinates[i], coordinates[j])); if (distance < first) { third = second; second = first; first = distance; } else if (distance < second) { third = second; second = distance; } else if (distance < third) third = distance; } thirdNearest.push(Number.isFinite(third) ? third : 0); }
  thirdNearest.sort((a, b) => a - b); const middle = Math.floor(thirdNearest.length / 2); const median = thirdNearest.length % 2 ? thirdNearest[middle] : (thirdNearest[middle - 1] + thirdNearest[middle]) / 2; const raw = Number.isFinite(median) && median > 0 ? median * 0.8 : diagonal / 50; return Math.max(diagonal / 500, Math.min(diagonal / 30, raw));
}
export const computeVisualizationBandwidth = visualizationBaseBandwidth;
export const computeBaseBandwidth = visualizationBaseBandwidth;
/** Internal nodes grow with remaining hierarchy depth; leaves are crisp actual points. */
export function visualizationStageSigma(baseSigma: number, remainingDepth: number, leaf = false): number { const base = Number.isFinite(baseSigma) && baseSigma > 0 ? baseSigma : Number.EPSILON; if (leaf) return base * 0.65; const remaining = Math.max(0, Number.isFinite(remainingDepth) ? remainingDepth : 0); return Math.min(base * 4, base * 2 ** (remaining * 0.35)); }
export const computeVisualizationStageSigma = visualizationStageSigma;
export const computeStageSigma = visualizationStageSigma;
/** Keep the user-controlled global Gaussian scale within the supported UI range. */
export function clampVisualizationKernelScale(value: number): number {
  if (Number.isNaN(value)) return VISUALIZATION_KERNEL_SCALE_DEFAULT;
  if (value === Number.POSITIVE_INFINITY) return VISUALIZATION_KERNEL_SCALE_MAX;
  if (value === Number.NEGATIVE_INFINITY) return VISUALIZATION_KERNEL_SCALE_MIN;
  return Math.max(VISUALIZATION_KERNEL_SCALE_MIN, Math.min(VISUALIZATION_KERNEL_SCALE_MAX, value));
}
/** Depth adaptation remains intact; this multiplier only adjusts the final kernel size. */
export function visualizationScaledStageSigma(baseSigma: number, remainingDepth: number, leaf = false, scale = VISUALIZATION_KERNEL_SCALE_DEFAULT): number {
  return visualizationStageSigma(baseSigma, remainingDepth, leaf) * clampVisualizationKernelScale(scale);
}

function rgb(value: string): [number, number, number] { return [parseInt(value.slice(1, 3), 16), parseInt(value.slice(3, 5), 16), parseInt(value.slice(5, 7), 16)]; }
function hex(values: [number, number, number]): string { return `#${values.map((value) => Math.round(Math.max(0, Math.min(255, value))).toString(16).padStart(2, "0")).join("")}`; }
type HslColor = VisualizationHslColor;

function normalizeHue(value: number): number { return ((value % 360) + 360) % 360; }
function clampColorChannel(value: number, minimum: number, maximum: number): number { return Math.max(minimum, Math.min(maximum, value)); }
function hashString(value: string): number { let hash = 2166136261; for (let index = 0; index < value.length; index++) { hash ^= value.charCodeAt(index); hash = Math.imul(hash, 16777619); } return hash >>> 0; }
function visualizationTreeSignature(node: VisualizationNode): string { return `${node.id}[${node.children.map(visualizationTreeSignature).join(",")}]`; }
function hslToRgb(color: HslColor): [number, number, number] {
  const h = normalizeHue(color.hue) / 360; const s = clampColorChannel(color.saturation, 0, 100) / 100; const l = clampColorChannel(color.lightness, 0, 100) / 100;
  const chroma = (1 - Math.abs(2 * l - 1)) * s; const section = h * 6; const x = chroma * (1 - Math.abs(section % 2 - 1)); const match = l - chroma / 2; let red = 0; let green = 0; let blue = 0;
  if (section < 1) [red, green, blue] = [chroma, x, 0]; else if (section < 2) [red, green, blue] = [x, chroma, 0]; else if (section < 3) [red, green, blue] = [0, chroma, x]; else if (section < 4) [red, green, blue] = [0, x, chroma]; else if (section < 5) [red, green, blue] = [x, 0, chroma]; else [red, green, blue] = [chroma, 0, x];
  return [Math.round((red + match) * 255), Math.round((green + match) * 255), Math.round((blue + match) * 255)];
}
function hslHex(color: HslColor): string { return hex(hslToRgb(color)); }

/**
 * Build one deterministic palette for a hierarchy. Top-level sibling clusters
 * occupy equally spaced points on the hue wheel. Descendants stay close to
 * their parent's hue while depth and sibling order provide small, bounded
 * hue/lightness changes. The rotation is a hash of the complete tree, so it
 * behaves like a random rotation without changing on every render.
 */
export function visualizationColorScheme(root: VisualizationNode): VisualizationColorScheme {
  const nodeColors = new Map<string, string>(); const leafColors = new Map<number, string>(); const nodeHsl = new Map<string, VisualizationHslColor>();
  const signature = visualizationTreeSignature(root); const rotation = hashString(`atomic-clusters-palette:${signature}`) / 0x100000000 * 360;
  const top = root.children.length === 1 && root.children[0].children.length ? root.children[0].children : root.children;
  const base: HslColor = { hue: rotation, saturation: 70, lightness: 52 };
  const assign = (node: VisualizationNode, color: HslColor): void => {
    const bounded: HslColor = { hue: normalizeHue(color.hue), saturation: clampColorChannel(color.saturation, 55, 82), lightness: clampColorChannel(color.lightness, 30, 76) };
    nodeColors.set(node.id, hslHex(bounded)); nodeHsl.set(node.id, { ...bounded }); if (!node.children.length && node.sourceId !== null) leafColors.set(node.sourceId, hslHex(bounded));
    if (!node.children.length) return;
    const count = node.children.length; const center = (count - 1) / 2; const denominator = Math.max(1, center); node.children.forEach((child, index) => {
      const siblingOffset = ((index - center) / denominator) * 16; const childColor: HslColor = { hue: bounded.hue + 3.5 + siblingOffset, saturation: bounded.saturation - 1.5, lightness: bounded.lightness + 5 + ((index - center) / denominator) * 3 };
      assign(child, childColor);
    });
  };
  if (top.length) top.forEach((node, index) => assign(node, { hue: base.hue + index * 360 / top.length, saturation: base.saturation, lightness: base.lightness }));
  // The synthetic root is not a cluster, but keeping a color for it makes
  // callers that render every tree node behave consistently.
  if (!nodeColors.has(root.id)) { nodeColors.set(root.id, VISUALIZATION_NOISE_COLOR); nodeHsl.set(root.id, { hue: 0, saturation: 0, lightness: 62 }); }
  if (root.children.length === 1 && root.children[0].children.length) { nodeColors.set(root.children[0].id, hslHex(base)); nodeHsl.set(root.children[0].id, { ...base }); }
  return { nodeColors, leafColors, nodeHsl };
}
export const visualizationClusterColors = visualizationColorScheme;
/** Return the single color used for every splat in a cluster cloud. */
export function visualizationCloudColor(node: VisualizationNode, palette: Pick<VisualizationColorScheme, "nodeColors">): string { return palette.nodeColors.get(node.id) || VISUALIZATION_NOISE_COLOR; }

/** Normalize membership only for hue; unexplained mass never pulls the hue toward grey. */
export function visualizationColorVector(row: readonly number[], leafOrdering: readonly number[], leafColors?: ReadonlyMap<number, string>): [number, number, number] { let total = 0; let red = 0; let green = 0; let blue = 0; for (let index = 0; index < Math.min(row.length, leafOrdering.length); index++) { const weight = Math.max(0, Number(row[index]) || 0); if (!weight) continue; const [r, g, b] = rgb(leafColors?.get(leafOrdering[index]) || visualizationColor(leafOrdering[index])); red += r * weight; green += g * weight; blue += b * weight; total += weight; } return total <= EPSILON ? rgb(VISUALIZATION_NOISE_COLOR) : [red / total, green / total, blue / total]; }
export function blendVisualizationColor(row: readonly number[], leafOrdering: readonly number[], leafColors?: ReadonlyMap<number, string>): string { return hex(visualizationColorVector(row, leafOrdering, leafColors)); }
export function visualizationColor(label: number): string { return !Number.isSafeInteger(label) || label < 0 ? VISUALIZATION_NOISE_COLOR : TAB20[label % TAB20.length]; }
export function visualizationP95RowSum(memberships: readonly number[][]): number { const values = memberships.map((row) => row.reduce((sum, value) => sum + Math.max(0, Number(value) || 0), 0)).sort((a, b) => a - b); return values.length ? values[Math.min(values.length - 1, Math.ceil(values.length * 0.95) - 1)] || 0 : 0; }
export function visualizationMembershipAmplitude(row: readonly number[], p95RowSum: number): number { const sum = row.reduce((total, value) => total + Math.max(0, Number(value) || 0), 0); const confidence = p95RowSum > EPSILON ? Math.max(0, Math.min(1, sum / p95RowSum)) : 0; return 0.25 + 0.75 * Math.sqrt(confidence); }

export function scaleVisualizationPoints(coordinates: readonly VisualizationCoordinate[], width: number, height: number, padding = VISUALIZATION_POINT_PADDING): VisualizationCoordinate[] { if (!coordinates.length || width <= 0 || height <= 0) return []; const safePadding = Math.max(0, Math.min(padding, Math.min(width, height) / 2)); const minX = Math.min(...coordinates.map(([x]) => x)); const maxX = Math.max(...coordinates.map(([x]) => x)); const minY = Math.min(...coordinates.map(([, y]) => y)); const maxY = Math.max(...coordinates.map(([, y]) => y)); const rangeX = maxX - minX; const rangeY = maxY - minY; const drawableWidth = Math.max(0, width - safePadding * 2); const drawableHeight = Math.max(0, height - safePadding * 2); const scale = Math.min(rangeX > 0 ? drawableWidth / rangeX : Number.POSITIVE_INFINITY, rangeY > 0 ? drawableHeight / rangeY : Number.POSITIVE_INFINITY); const finiteScale = Number.isFinite(scale) ? scale : 0; const contentWidth = rangeX * finiteScale; const contentHeight = rangeY * finiteScale; const offsetX = safePadding + (drawableWidth - contentWidth) / 2; const offsetY = safePadding + (drawableHeight - contentHeight) / 2; return coordinates.map(([x, y]) => [offsetX + (rangeX > 0 ? (x - minX) * finiteScale : 0), offsetY + (rangeY > 0 ? (maxY - y) * finiteScale : 0)]); }
/** Camera maps one selected world region into the canvas; all coordinates remain global. */
export function visualizationCameraTransform(region: { minX: number; maxX: number; minY: number; maxY: number }, width: number, height: number, padding = VISUALIZATION_POINT_PADDING): VisualizationCamera { const safeWidth = Math.max(1, width), safeHeight = Math.max(1, height); const safePadding = Math.max(0, Math.min(padding, Math.min(safeWidth, safeHeight) / 2)); const rangeX = Math.max(EPSILON, region.maxX - region.minX), rangeY = Math.max(EPSILON, region.maxY - region.minY); const scale = Math.min((safeWidth - safePadding * 2) / rangeX, (safeHeight - safePadding * 2) / rangeY); const contentWidth = rangeX * scale, contentHeight = rangeY * scale; return { scale, offsetX: safePadding + (safeWidth - safePadding * 2 - contentWidth) / 2 - region.minX * scale, offsetY: safePadding + (safeHeight - safePadding * 2 - contentHeight) / 2 + region.maxY * scale, width: safeWidth, height: safeHeight, worldRegion: region }; }
/**
 * Quintic ease-in-out used by camera navigation. Its zero velocity at both
 * endpoints avoids the perceptual "snap" that smoothstep can leave when a
 * semantic stage is replaced, while remaining monotonic for interpolation.
 */
export function visualizationEaseInOut(progress: number): number {
  const t = Math.max(0, Math.min(1, Number.isFinite(progress) ? progress : 0));
  return t < 0.5 ? 16 * t ** 5 : 1 - 16 * (1 - t) ** 5;
}
/**
 * Return the uniform CSS transform that maps target-camera pixels into the
 * interpolated source-to-target camera. Density is therefore rasterized only
 * once for the target camera; every frame is a cheap affine transform.
 */
export function visualizationCameraLayerTransform(from: VisualizationCamera, to: VisualizationCamera, progress: number): VisualizationCameraLayerTransform {
  const eased = visualizationEaseInOut(progress);
  if (eased <= 0) {
    const sourceScale = to.scale === 0 ? 1 : from.scale / to.scale;
    return { scale: sourceScale, translateX: from.offsetX - sourceScale * to.offsetX, translateY: from.offsetY - sourceScale * to.offsetY };
  }
  if (eased >= 1) return { scale: 1, translateX: 0, translateY: 0 };
  const sourceScale = to.scale === 0 ? 1 : from.scale / to.scale;
  const sourceX = from.offsetX - sourceScale * to.offsetX;
  const sourceY = from.offsetY - sourceScale * to.offsetY;
  return { scale: sourceScale + (1 - sourceScale) * eased, translateX: sourceX * (1 - eased), translateY: sourceY * (1 - eased) };
}
export const visualizationCameraTransitionTransform = visualizationCameraLayerTransform;
/**
 * Transform an image rendered with the source camera into the destination
 * camera. Unlike visualizationCameraLayerTransform (which maps the already
 * rendered destination stage back to the source at t=0), this starts at the
 * source image's identity transform and ends at the destination mapping.
 * This is used for the outgoing snapshot during semantic-stage navigation.
 */
export function visualizationOutgoingLayerTransform(from: VisualizationCamera, to: VisualizationCamera, progress: number): VisualizationCameraLayerTransform {
  const eased = visualizationEaseInOut(progress);
  const sourceScale = from.scale === 0 ? 1 : to.scale / from.scale;
  const targetX = to.offsetX - sourceScale * from.offsetX;
  const targetY = to.offsetY - sourceScale * from.offsetY;
  return {
    scale: 1 + (sourceScale - 1) * eased,
    translateX: targetX * eased || 0,
    translateY: targetY * eased || 0,
  };
}
export function visualizationWorldToScreen(camera: VisualizationCamera, point: VisualizationCoordinate): VisualizationCoordinate { return [camera.offsetX + point[0] * camera.scale, camera.offsetY - point[1] * camera.scale]; }
export function visualizationScreenToWorld(camera: VisualizationCamera, point: VisualizationCoordinate): VisualizationCoordinate { return [(point[0] - camera.offsetX) / camera.scale, (camera.offsetY - point[1]) / camera.scale]; }
export function visualizationRegion(node: VisualizationNode, coordinates: readonly VisualizationCoordinate[]): { minX: number; maxX: number; minY: number; maxY: number } { const points = node.pointIndices.map((index) => coordinates[index]).filter((point): point is VisualizationCoordinate => !!point && point.every(Number.isFinite)); if (!points.length) return { minX: 0, maxX: 1, minY: 0, maxY: 1 }; const minX = Math.min(...points.map(([x]) => x)); const maxX = Math.max(...points.map(([x]) => x)); const minY = Math.min(...points.map(([, y]) => y)); const maxY = Math.max(...points.map(([, y]) => y)); const padX = Math.max(1e-6, (maxX - minX) * 0.12); const padY = Math.max(1e-6, (maxY - minY) * 0.12); return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }; }
export function visualizationPath(root: VisualizationNode, targetId: string): VisualizationNode[] { const visit = (node: VisualizationNode): VisualizationNode[] | null => { if (node.id === targetId) return [node]; for (const child of node.children) { const path = visit(child); if (path) return [node, ...path]; } return null; }; return visit(root) || [root]; }
export function visualizationParent(root: VisualizationNode, targetId: string): VisualizationNode | null { const path = visualizationPath(root, targetId); return path.length > 1 ? path[path.length - 2] : null; }
export function visualizationLeafOrdering(result: Pick<ClusterResult, "hierarchy" | "leafOrdering" | "visualization">): number[] { return [...(result.leafOrdering || result.visualization?.leafOrdering || result.hierarchy.leaves)].filter((label, index, values) => Number.isSafeInteger(label) && values.indexOf(label) === index); }

export function validateVisualizationData(result: Pick<ClusterResult, "ids" | "schemaVersion" | "memberships" | "visualization" | "hierarchy" | "leafOrdering">): boolean { const visualization = result.visualization; const memberships = result.memberships; const ordering = result.leafOrdering; const allNoise = result.schemaVersion >= 6 && result.hierarchy.leaves.length === 0; const validOrdering = Array.isArray(ordering) && (ordering.length > 0 || allNoise) && ordering.length === result.hierarchy.leaves.length && ordering.every((label, index) => Number.isSafeInteger(label) && label >= 0 && ordering.indexOf(label) === index) && ordering.every((label) => result.hierarchy.leaves.includes(label)); return result.schemaVersion >= 4 && !!visualization && validOrdering && visualization.coordinates.length === result.ids.length && visualization.labels.length === result.ids.length && visualization.coordinates.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)) && visualization.labels.every((label) => Number.isSafeInteger(label) && label >= -1 && (label === -1 || ordering!.includes(label))) && visualization.leafOrdering?.length === ordering!.length && visualization.leafOrdering.every((label, index) => label === ordering![index]) && Array.isArray(memberships) && memberships.length === result.ids.length && memberships.every((row) => Array.isArray(row) && row.length === ordering!.length && row.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) && row.reduce((sum, value) => sum + value, 0) <= 1 + 1e-6); }
export function visualizationCloudGeometry(points: readonly VisualizationCoordinate[], indices: readonly number[], width: number, height: number): { x: number; y: number; radius: number } | null { const selected = indices.filter((index) => !!points[index]); if (!selected.length) return null; const x = selected.reduce((sum, index) => sum + points[index][0], 0) / selected.length; const y = selected.reduce((sum, index) => sum + points[index][1], 0) / selected.length; const variance = selected.reduce((sum, index) => sum + (points[index][0] - x) ** 2 + (points[index][1] - y) ** 2, 0) / selected.length; return { x, y, radius: Math.max(22, Math.min(Math.max(width, height) * .42, Math.sqrt(variance) * 2.5 + 20)) }; }

/** Return the display name for a visible internal cluster, or null for notes/noise. */
export function visualizationClusterLabelText(entry: Pick<VisualizationFrontierEntry, "node" | "actualPoints">, titles?: Readonly<Record<string, string>>): string | null {
  if (entry.actualPoints || entry.node.sourceId === null || !Number.isSafeInteger(entry.node.sourceId)) return null;
  const title = titles?.[String(entry.node.sourceId)]?.trim();
  return title || `Cluster ${entry.node.sourceId}`;
}

function visualizationHexRgb(value: string): [number, number, number] | null {
  const match = /^#([0-9a-f]{6})$/i.exec(value.trim());
  return match ? [parseInt(match[1].slice(0, 2), 16), parseInt(match[1].slice(2, 4), 16), parseInt(match[1].slice(4, 6), 16)] : null;
}
function visualizationRelativeLuminance(value: string): number {
  const rgbValue = visualizationHexRgb(value); if (!rgbValue) return 0.5;
  const channels = rgbValue.map((channel) => channel / 255).map((channel) => channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

/** Use a high-contrast pill: its fill contrasts the cloud and its text contrasts the pill. */
export function visualizationLabelContrast(clusterColor: string): VisualizationLabelContrast {
  const luminance = visualizationRelativeLuminance(clusterColor);
  return luminance > 0.179 ? { foreground: "#000000", background: "#ffffff" } : { foreground: "#ffffff", background: "#000000" };
}

export interface VisualizationLabelLayoutOptions { margin?: number; gap?: number; labelHeight?: number; measureText?: (text: string) => number; }

/**
 * Lay out labels for the current cloud frontier in screen coordinates.  The
 * candidate position is the centroid of the visible points (falling back to
 * all points in a cluster when residual classification removed them). Labels
 * are clamped to the viewport and nudged vertically to avoid simple overlaps.
 */
export function layoutVisualizationClusterLabels(
  frontier: readonly VisualizationFrontierEntry[],
  points: readonly VisualizationCoordinate[],
  titles: Readonly<Record<string, string>> | undefined,
  colors: ReadonlyMap<string, string>,
  width: number,
  height: number,
  options: VisualizationLabelLayoutOptions = {},
): VisualizationLabelPlacement[] {
  const viewportWidth = Math.max(0, Number.isFinite(width) ? width : 0); const viewportHeight = Math.max(0, Number.isFinite(height) ? height : 0);
  const margin = Math.max(2, Number.isFinite(options.margin) ? options.margin! : 8); const gap = Math.max(0, Number.isFinite(options.gap) ? options.gap! : 4); const labelHeight = Math.max(10, Number.isFinite(options.labelHeight) ? options.labelHeight! : 20);
  const measure = options.measureText || ((text: string) => text.length * 7);
  const placements: VisualizationLabelPlacement[] = [];
  for (const entry of frontier) {
    const text = visualizationClusterLabelText(entry, titles); if (!text) continue;
    const indices = (entry.pointIndices.length ? entry.pointIndices : entry.node.pointIndices).filter((index) => !!points[index] && points[index].every(Number.isFinite)); if (!indices.length) continue;
    const centerX = indices.reduce((sum, index) => sum + points[index][0], 0) / indices.length; const centerY = indices.reduce((sum, index) => sum + points[index][1], 0) / indices.length;
    const textWidth = Math.max(1, Number(measure(text)) || text.length * 7); const boxWidth = textWidth + 16; const boxHeight = labelHeight; const minX = margin; const maxX = Math.max(minX, viewportWidth - margin - boxWidth); const minY = margin; const maxY = Math.max(minY, viewportHeight - margin - boxHeight);
    const x = Math.max(minX, Math.min(maxX, centerX - boxWidth / 2)); let y = Math.max(minY, Math.min(maxY, centerY - boxHeight / 2));
    const contrast = visualizationLabelContrast(colors.get(entry.node.id) || VISUALIZATION_NOISE_COLOR); const placement: VisualizationLabelPlacement = { id: entry.node.id, text, x, y, width: Math.min(boxWidth, Math.max(0, viewportWidth - margin * 2)), height: Math.min(boxHeight, Math.max(0, viewportHeight - margin * 2)), contrast };
    // Try a deterministic sequence of vertical positions before accepting an overlap.
    const candidates = [y, y - (boxHeight + gap), y + (boxHeight + gap), y - 2 * (boxHeight + gap), y + 2 * (boxHeight + gap)];
    const overlaps = (left: VisualizationLabelPlacement, top: number): boolean => left.x < placement.x + placement.width + gap && placement.x < left.x + left.width + gap && top < left.y + left.height + gap && left.y < top + placement.height + gap;
    for (const candidate of candidates) { const bounded = Math.max(minY, Math.min(maxY, candidate)); if (!placements.some((placed) => overlaps(placed, bounded))) { y = bounded; break; } }
    placement.y = y; placements.push(placement);
  }
  return placements;
}
export const visualizationClusterLabels = layoutVisualizationClusterLabels;
export function findNearestVisualizationPoint(points: readonly VisualizationCoordinate[], x: number, y: number, radius: number): number | null { const radiusSquared = Math.max(0, radius) ** 2; let closest: number | null = null; let closestDistance = radiusSquared; points.forEach(([pointX, pointY], index) => { const distance = (pointX - x) ** 2 + (pointY - y) ** 2; if (distance <= closestDistance) { closest = index; closestDistance = distance; } }); return closest; }

/** Accumulate weighted RGB and optical density, clipping every splat at three sigma. */
export function accumulateVisualizationDensity(splats: readonly VisualizationSplat[], width: number, height: number, target?: VisualizationDensityField): VisualizationDensityField {
  const w = Math.max(1, Math.floor(width)), h = Math.max(1, Math.floor(height)); const size = w * h;
  const field = target && target.width === w && target.height === h ? target : { width: w, height: h, density: new Float32Array(size), red: new Float32Array(size), green: new Float32Array(size), blue: new Float32Array(size) };
  field.density.fill(0); field.red.fill(0); field.green.fill(0); field.blue.fill(0);
  for (const splat of splats) {
    if (![splat.x, splat.y, splat.sigma, splat.amplitude].every(Number.isFinite) || splat.sigma <= 0 || splat.amplitude <= 0) continue;
    const radius = splat.sigma * 3; const minX = Math.max(0, Math.ceil(splat.x - radius)), maxX = Math.min(w - 1, Math.floor(splat.x + radius)); const minY = Math.max(0, Math.ceil(splat.y - radius)), maxY = Math.min(h - 1, Math.floor(splat.y + radius)); const inverse = 1 / (2 * splat.sigma * splat.sigma);
    for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) { const distanceSquared = (x - splat.x) ** 2 + (y - splat.y) ** 2; if (distanceSquared > 9 * splat.sigma * splat.sigma) continue; const weight = splat.amplitude * Math.exp(-distanceSquared * inverse); const offset = y * w + x; field.density[offset] += weight; field.red[offset] += weight * splat.color[0]; field.green[offset] += weight * splat.color[1]; field.blue[offset] += weight * splat.color[2]; }
  }
  return field;
}
export function visualizationDensityAt(splats: readonly VisualizationSplat[], x: number, y: number): number { if (!Number.isFinite(x) || !Number.isFinite(y)) return 0; return splats.reduce((sum, splat) => { if (![splat.x, splat.y, splat.sigma, splat.amplitude].every(Number.isFinite) || splat.sigma <= 0 || splat.amplitude <= 0) return sum; const distanceSquared = (x - splat.x) ** 2 + (y - splat.y) ** 2; return distanceSquared <= 9 * splat.sigma * splat.sigma ? sum + splat.amplitude * Math.exp(-distanceSquared / (2 * splat.sigma * splat.sigma)) : sum; }, 0); }
export function pickVisualizationCloud(cloudSplats: readonly (readonly VisualizationSplat[])[], x: number, y: number, minimumDensity = 0.02): number | null { let best: number | null = null; let bestDensity = minimumDensity; cloudSplats.forEach((splats, index) => { const density = visualizationDensityAt(splats, x, y); if (density > bestDensity) { best = index; bestDensity = density; } }); return best; }
export const accumulateDensity = accumulateVisualizationDensity;
export const densityAt = visualizationDensityAt;
export function visualizationDensityAlpha(density: number): number { return 1 - Math.exp(-Math.max(0, Number.isFinite(density) ? density : 0)); }
