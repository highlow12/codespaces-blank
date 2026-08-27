import { ClusterResult, HierarchyMerge, HierarchyTree, VisualizationCoordinate } from "./types";

const TAB20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"] as const;
export const VISUALIZATION_NOISE_COLOR = "#9aa0a6";
export const VISUALIZATION_POINT_PADDING = 18;
export const VISUALIZATION_KERNEL_SCALE_MIN = 0.25;
export const VISUALIZATION_KERNEL_SCALE_MAX = 2;
export const VISUALIZATION_KERNEL_SCALE_STEP = 0.05;
export const VISUALIZATION_KERNEL_SCALE_DEFAULT = 0.65;
const EPSILON = 1e-9;

export interface VisualizationNode { id: string; sourceId: number | null; children: VisualizationNode[]; leafLabels: number[]; pointIndices: number[]; depth: number; }
export interface NaryVisualizationHierarchy { leaves: number[]; root: number | null; children: Record<string, number[]>; merges?: HierarchyMerge[]; }
export interface VisualizationFrontierEntry { node: VisualizationNode; depth: number; remainingDepth: number; pointIndices: number[]; residualIndices: number[]; /** true when a leaf was explicitly selected and its actual notes should be shown. */ actualPoints: boolean; }
export interface VisualizationCamera { scale: number; offsetX: number; offsetY: number; width: number; height: number; worldRegion: { minX: number; maxX: number; minY: number; maxY: number }; }
export interface VisualizationCameraLayerTransform { scale: number; translateX: number; translateY: number; }
export interface VisualizationSplat { x: number; y: number; sigma: number; color: [number, number, number]; amplitude: number; }
export interface VisualizationDensityField { width: number; height: number; density: Float32Array; red: Float32Array; green: Float32Array; blue: Float32Array; }

/** Build a generalized children[] tree. Binary left/right knowledge ends here. */
export function buildVisualizationTree(hierarchy: HierarchyTree | NaryVisualizationHierarchy, labels: readonly number[]): VisualizationNode {
  const mergeList = hierarchy.merges || []; const mergeIds = new Set<number>();
  for (const merge of mergeList) { if (!Number.isSafeInteger(merge.id) || mergeIds.has(merge.id)) throw new Error("Hierarchy contains malformed or duplicate node ids"); mergeIds.add(merge.id); }
  if (!Array.isArray(hierarchy.leaves) || hierarchy.leaves.some((label, index) => !Number.isSafeInteger(label) || hierarchy.leaves.indexOf(label) !== index || mergeIds.has(label))) throw new Error("Hierarchy contains malformed or duplicate leaf ids");
  const merges = new Map<number, HierarchyMerge>(mergeList.map((merge) => [merge.id, merge])); const visiting = new Set<number>(); const built = new Set<number>();
  const make = (sourceId: number, depth: number): VisualizationNode => {
    if (!Number.isSafeInteger(sourceId)) throw new Error("Hierarchy contains a malformed node id"); if (visiting.has(sourceId)) throw new Error("Hierarchy contains a cycle"); if (built.has(sourceId)) throw new Error("Hierarchy contains duplicate node references"); built.add(sourceId); visiting.add(sourceId);
    const merge = merges.get(sourceId); const futureChildren = hierarchy.children?.[String(sourceId)]; if (futureChildren !== undefined && !Array.isArray(futureChildren)) throw new Error("Hierarchy children must be arrays"); const children = (futureChildren || (merge ? [merge.left, merge.right] : [])).map((child) => make(child, depth + 1)); visiting.delete(sourceId);
    const leafLabels = children.length ? children.flatMap((child) => child.leafLabels) : [sourceId]; const uniqueLeaves = [...new Set(leafLabels)]; const pointIndices = labels.map((label, index) => label === sourceId || uniqueLeaves.includes(label) ? index : -1).filter((index) => index >= 0); return { id: `node:${sourceId}`, sourceId, children, leafLabels: uniqueLeaves, pointIndices, depth };
  };
  const roots = hierarchy.root === null ? hierarchy.leaves.map((leaf) => make(leaf, 1)) : [make(hierarchy.root, 1)]; return { id: "root", sourceId: null, children: roots, leafLabels: [...new Set(roots.flatMap((node) => node.leafLabels))], pointIndices: labels.map((_label, index) => index), depth: 0 };
}

export function childMembershipMass(node: VisualizationNode, row: readonly number[], leafOrdering: readonly number[]): number { const leaves = new Set(node.leafLabels); let mass = 0; for (let column = 0; column < Math.min(leafOrdering.length, row.length); column++) if (leaves.has(leafOrdering[column])) mass += Math.max(0, Number(row[column]) || 0); return mass; }
/** Notes below 0.5 conditional membership are shown as residual dots. */
export function residualPointIndices(node: VisualizationNode, _labels: readonly number[], memberships: readonly number[][], leafOrdering: readonly number[]): number[] { if (!node.children.length) return node.pointIndices.slice(); return node.pointIndices.filter((index) => { const row = memberships[index] || []; const masses = node.children.map((child) => childMembershipMass(child, row, leafOrdering)); const total = masses.reduce((sum, mass) => sum + mass, 0); return total <= EPSILON || Math.max(...masses) / total < 0.5; }); }

/** Return the deterministic visible frontier. expandedIds contains clouds replaced by their children. */
export function visualizationFrontier(root: VisualizationNode, expandedIds: Iterable<string> = [], memberships?: readonly number[][], leafOrdering?: readonly number[]): VisualizationFrontierEntry[] {
  const expanded = new Set(expandedIds); const maxDepth = visualizationTreeDepth(root); const entries: VisualizationFrontierEntry[] = []; const emittedResidual = new Set<number>();
  const emit = (node: VisualizationNode, inheritedResidual: ReadonlySet<number>): void => {
    const residualIndices = [...inheritedResidual].filter((index) => !emittedResidual.has(index)); residualIndices.forEach((index) => emittedResidual.add(index));
    const residualSet = inheritedResidual; const actualPoints = !node.children.length && expanded.has(node.id);
    entries.push({ node, depth: node.depth, remainingDepth: Math.max(0, maxDepth - node.depth), pointIndices: node.pointIndices.filter((index) => !residualSet.has(index)), residualIndices, actualPoints });
  };
  const childrenWithResidual = (parent: VisualizationNode, children: readonly VisualizationNode[], inheritedResidual: ReadonlySet<number>): void => {
    const parentResidual = memberships && leafOrdering ? residualPointIndices(parent, [], memberships, leafOrdering) : [];
    const nextResidual = new Set(inheritedResidual); parentResidual.forEach((index) => nextResidual.add(index));
    for (const child of children) visit(child, nextResidual);
  };
  const visit = (node: VisualizationNode, inheritedResidual: ReadonlySet<number>): void => {
    if (node.id === "root" && expanded.has(node.id)) {
      // Skip a redundant synthetic root while computing residuals against all notes.
      const actual = node.children.length === 1 ? node.children[0] : null;
      if (actual && actual.children.length && !expanded.has(actual.id)) {
        // Preserve any synthetic-root residuals, but classify the first real split
        // against the actual hierarchy root rather than its one-child wrapper.
        const nextResidual = new Set(inheritedResidual); if (memberships && leafOrdering) residualPointIndices(node, [], memberships, leafOrdering).forEach((index) => nextResidual.add(index));
        childrenWithResidual(actual, actual.children, nextResidual); return;
      }
      if (actual) { const nextResidual = new Set(inheritedResidual); if (memberships && leafOrdering) residualPointIndices(node, [], memberships, leafOrdering).forEach((index) => nextResidual.add(index)); visit(actual, nextResidual); return; }
    }
    if (expanded.has(node.id) && node.children.length) { childrenWithResidual(node, node.children, inheritedResidual); return; }
    emit(node, inheritedResidual);
  };
  if (!expanded.size) entries.push({ node: root, depth: root.depth, remainingDepth: maxDepth, pointIndices: root.pointIndices.slice(), residualIndices: [], actualPoints: false }); else visit(root, new Set()); return entries;
}
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
/** Normalize membership only for hue; unexplained mass never pulls the hue toward grey. */
export function visualizationColorVector(row: readonly number[], leafOrdering: readonly number[]): [number, number, number] { let total = 0; let red = 0; let green = 0; let blue = 0; for (let index = 0; index < Math.min(row.length, leafOrdering.length); index++) { const weight = Math.max(0, Number(row[index]) || 0); if (!weight) continue; const [r, g, b] = rgb(visualizationColor(leafOrdering[index])); red += r * weight; green += g * weight; blue += b * weight; total += weight; } return total <= EPSILON ? rgb(VISUALIZATION_NOISE_COLOR) : [red / total, green / total, blue / total]; }
export function blendVisualizationColor(row: readonly number[], leafOrdering: readonly number[]): string { return hex(visualizationColorVector(row, leafOrdering)); }
export function visualizationColor(label: number): string { return !Number.isSafeInteger(label) || label < 0 ? VISUALIZATION_NOISE_COLOR : TAB20[label % TAB20.length]; }
export function visualizationP95RowSum(memberships: readonly number[][]): number { const values = memberships.map((row) => row.reduce((sum, value) => sum + Math.max(0, Number(value) || 0), 0)).sort((a, b) => a - b); return values.length ? values[Math.min(values.length - 1, Math.ceil(values.length * 0.95) - 1)] || 0 : 0; }
export function visualizationMembershipAmplitude(row: readonly number[], p95RowSum: number): number { const sum = row.reduce((total, value) => total + Math.max(0, Number(value) || 0), 0); const confidence = p95RowSum > EPSILON ? Math.max(0, Math.min(1, sum / p95RowSum)) : 0; return 0.25 + 0.75 * Math.sqrt(confidence); }

export function scaleVisualizationPoints(coordinates: readonly VisualizationCoordinate[], width: number, height: number, padding = VISUALIZATION_POINT_PADDING): VisualizationCoordinate[] { if (!coordinates.length || width <= 0 || height <= 0) return []; const safePadding = Math.max(0, Math.min(padding, Math.min(width, height) / 2)); const minX = Math.min(...coordinates.map(([x]) => x)); const maxX = Math.max(...coordinates.map(([x]) => x)); const minY = Math.min(...coordinates.map(([, y]) => y)); const maxY = Math.max(...coordinates.map(([, y]) => y)); const rangeX = maxX - minX; const rangeY = maxY - minY; const drawableWidth = Math.max(0, width - safePadding * 2); const drawableHeight = Math.max(0, height - safePadding * 2); const scale = Math.min(rangeX > 0 ? drawableWidth / rangeX : Number.POSITIVE_INFINITY, rangeY > 0 ? drawableHeight / rangeY : Number.POSITIVE_INFINITY); const finiteScale = Number.isFinite(scale) ? scale : 0; const contentWidth = rangeX * finiteScale; const contentHeight = rangeY * finiteScale; const offsetX = safePadding + (drawableWidth - contentWidth) / 2; const offsetY = safePadding + (drawableHeight - contentHeight) / 2; return coordinates.map(([x, y]) => [offsetX + (rangeX > 0 ? (x - minX) * finiteScale : 0), offsetY + (rangeY > 0 ? (maxY - y) * finiteScale : 0)]); }
/** Camera maps one selected world region into the canvas; all coordinates remain global. */
export function visualizationCameraTransform(region: { minX: number; maxX: number; minY: number; maxY: number }, width: number, height: number, padding = VISUALIZATION_POINT_PADDING): VisualizationCamera { const safeWidth = Math.max(1, width), safeHeight = Math.max(1, height); const safePadding = Math.max(0, Math.min(padding, Math.min(safeWidth, safeHeight) / 2)); const rangeX = Math.max(EPSILON, region.maxX - region.minX), rangeY = Math.max(EPSILON, region.maxY - region.minY); const scale = Math.min((safeWidth - safePadding * 2) / rangeX, (safeHeight - safePadding * 2) / rangeY); const contentWidth = rangeX * scale, contentHeight = rangeY * scale; return { scale, offsetX: safePadding + (safeWidth - safePadding * 2 - contentWidth) / 2 - region.minX * scale, offsetY: safePadding + (safeHeight - safePadding * 2 - contentHeight) / 2 + region.maxY * scale, width: safeWidth, height: safeHeight, worldRegion: region }; }
/** Smoothstep easing used by camera navigation. */
export function visualizationEaseInOut(progress: number): number { const t = Math.max(0, Math.min(1, Number.isFinite(progress) ? progress : 0)); return t * t * (3 - 2 * t); }
/**
 * Return the uniform CSS transform that maps target-camera pixels into the
 * interpolated source-to-target camera. Density is therefore rasterized only
 * once for the target camera; every frame is a cheap affine transform.
 */
export function visualizationCameraLayerTransform(from: VisualizationCamera, to: VisualizationCamera, progress: number): VisualizationCameraLayerTransform {
  const eased = visualizationEaseInOut(progress);
  const sourceScale = to.scale === 0 ? 1 : from.scale / to.scale;
  const sourceX = from.offsetX - sourceScale * to.offsetX;
  const sourceY = from.offsetY - sourceScale * to.offsetY;
  return { scale: sourceScale + (1 - sourceScale) * eased, translateX: sourceX * (1 - eased), translateY: sourceY * (1 - eased) };
}
export const visualizationCameraTransitionTransform = visualizationCameraLayerTransform;
export function visualizationWorldToScreen(camera: VisualizationCamera, point: VisualizationCoordinate): VisualizationCoordinate { return [camera.offsetX + point[0] * camera.scale, camera.offsetY - point[1] * camera.scale]; }
export function visualizationScreenToWorld(camera: VisualizationCamera, point: VisualizationCoordinate): VisualizationCoordinate { return [(point[0] - camera.offsetX) / camera.scale, (camera.offsetY - point[1]) / camera.scale]; }
export function visualizationRegion(node: VisualizationNode, coordinates: readonly VisualizationCoordinate[]): { minX: number; maxX: number; minY: number; maxY: number } { const points = node.pointIndices.map((index) => coordinates[index]).filter((point): point is VisualizationCoordinate => !!point && point.every(Number.isFinite)); if (!points.length) return { minX: 0, maxX: 1, minY: 0, maxY: 1 }; const minX = Math.min(...points.map(([x]) => x)); const maxX = Math.max(...points.map(([x]) => x)); const minY = Math.min(...points.map(([, y]) => y)); const maxY = Math.max(...points.map(([, y]) => y)); const padX = Math.max(1e-6, (maxX - minX) * 0.12); const padY = Math.max(1e-6, (maxY - minY) * 0.12); return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }; }
export function visualizationPath(root: VisualizationNode, targetId: string): VisualizationNode[] { const visit = (node: VisualizationNode): VisualizationNode[] | null => { if (node.id === targetId) return [node]; for (const child of node.children) { const path = visit(child); if (path) return [node, ...path]; } return null; }; return visit(root) || [root]; }
export function visualizationParent(root: VisualizationNode, targetId: string): VisualizationNode | null { const path = visualizationPath(root, targetId); return path.length > 1 ? path[path.length - 2] : null; }
export function visualizationLeafOrdering(result: Pick<ClusterResult, "hierarchy" | "leafOrdering" | "visualization">): number[] { return [...(result.leafOrdering || result.visualization?.leafOrdering || result.hierarchy.leaves)].filter((label, index, values) => Number.isSafeInteger(label) && values.indexOf(label) === index); }

export function validateVisualizationData(result: Pick<ClusterResult, "ids" | "schemaVersion" | "memberships" | "visualization" | "hierarchy" | "leafOrdering">): boolean { const visualization = result.visualization; const memberships = result.memberships; const ordering = result.leafOrdering; const validOrdering = Array.isArray(ordering) && ordering.length > 0 && ordering.length === result.hierarchy.leaves.length && ordering.every((label, index) => Number.isSafeInteger(label) && label >= 0 && ordering.indexOf(label) === index) && ordering.every((label) => result.hierarchy.leaves.includes(label)); return result.schemaVersion >= 4 && !!visualization && validOrdering && visualization.coordinates.length === result.ids.length && visualization.labels.length === result.ids.length && visualization.coordinates.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)) && visualization.labels.every((label) => Number.isSafeInteger(label) && label >= -1 && (label === -1 || ordering.includes(label))) && visualization.leafOrdering?.length === ordering.length && visualization.leafOrdering.every((label, index) => label === ordering[index]) && Array.isArray(memberships) && memberships.length === result.ids.length && memberships.every((row) => Array.isArray(row) && row.length === ordering.length && row.every((value) => Number.isFinite(value) && value >= 0 && value <= 1) && row.reduce((sum, value) => sum + value, 0) <= 1 + 1e-6); }
export function visualizationCloudGeometry(points: readonly VisualizationCoordinate[], indices: readonly number[], width: number, height: number): { x: number; y: number; radius: number } | null { const selected = indices.filter((index) => !!points[index]); if (!selected.length) return null; const x = selected.reduce((sum, index) => sum + points[index][0], 0) / selected.length; const y = selected.reduce((sum, index) => sum + points[index][1], 0) / selected.length; const variance = selected.reduce((sum, index) => sum + (points[index][0] - x) ** 2 + (points[index][1] - y) ** 2, 0) / selected.length; return { x, y, radius: Math.max(22, Math.min(Math.max(width, height) * .42, Math.sqrt(variance) * 2.5 + 20)) }; }
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
