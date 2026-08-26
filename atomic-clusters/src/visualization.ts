import { VisualizationCoordinate } from "./types";

const TAB20 = [
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
  "#17becf", "#9edae5"
] as const;

export const VISUALIZATION_NOISE_COLOR = "#9aa0a6";

// Leave enough room for the largest rendered marker (including its hover ring
// and the invisible pointer target) at every edge of the canvas.
export const VISUALIZATION_POINT_PADDING = 18;

/** Stable matplotlib-tab20-like color for a leaf label; noise is neutral. */
export function visualizationColor(label: number): string {
  if (!Number.isSafeInteger(label) || label < 0) return VISUALIZATION_NOISE_COLOR;
  return TAB20[label % TAB20.length];
}

/**
 * Map UMAP coordinates into a canvas's logical pixel rectangle while keeping
 * the aspect ratio. Degenerate axes are centered instead of producing NaN.
 */
export function scaleVisualizationPoints(
  coordinates: readonly VisualizationCoordinate[],
  width: number,
  height: number,
  padding = VISUALIZATION_POINT_PADDING
): VisualizationCoordinate[] {
  if (!coordinates.length || width <= 0 || height <= 0) return [];
  const safePadding = Math.max(0, Math.min(padding, Math.min(width, height) / 2));
  const minX = Math.min(...coordinates.map(([x]) => x));
  const maxX = Math.max(...coordinates.map(([x]) => x));
  const minY = Math.min(...coordinates.map(([, y]) => y));
  const maxY = Math.max(...coordinates.map(([, y]) => y));
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  const drawableWidth = Math.max(0, width - safePadding * 2);
  const drawableHeight = Math.max(0, height - safePadding * 2);
  const scale = Math.min(
    rangeX > 0 ? drawableWidth / rangeX : Number.POSITIVE_INFINITY,
    rangeY > 0 ? drawableHeight / rangeY : Number.POSITIVE_INFINITY
  );
  const finiteScale = Number.isFinite(scale) ? scale : 0;
  const contentWidth = rangeX * finiteScale;
  const contentHeight = rangeY * finiteScale;
  const offsetX = safePadding + (drawableWidth - contentWidth) / 2;
  const offsetY = safePadding + (drawableHeight - contentHeight) / 2;
  return coordinates.map(([x, y]) => [
    offsetX + (rangeX > 0 ? (x - minX) * finiteScale : 0),
    offsetY + (rangeY > 0 ? (maxY - y) * finiteScale : 0)
  ]);
}

/** Return the closest point within radius, or null when the pointer misses. */
export function findNearestVisualizationPoint(
  points: readonly VisualizationCoordinate[],
  x: number,
  y: number,
  radius: number
): number | null {
  const radiusSquared = Math.max(0, radius) ** 2;
  let closest: number | null = null;
  let closestDistance = radiusSquared;
  points.forEach(([pointX, pointY], index) => {
    const distance = (pointX - x) ** 2 + (pointY - y) ** 2;
    if (distance <= closestDistance) {
      closest = index;
      closestDistance = distance;
    }
  });
  return closest;
}
