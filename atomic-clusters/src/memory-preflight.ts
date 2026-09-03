/**
 * Renderer-safe large-vault memory preflight.
 *
 * This module deliberately has no Node/Electron imports. A renderer usually
 * cannot observe total available system RAM, so an absent or approximate
 * signal is a warning only. A hard stop is reserved for a finite Chromium
 * JS-heap headroom signal and a conservative estimate that would consume the
 * dangerous share of that headroom.
 */

const BYTES_PER_MIB = 1024 * 1024;
const BYTES_PER_GIB = 1024 * BYTES_PER_MIB;

export const MEMORY_WARNING_RATIO = 0.35;
export const MEMORY_DANGER_RATIO = 0.8;

export interface RendererPerformanceMemory {
  readonly jsHeapSizeLimit?: unknown;
  readonly usedJSHeapSize?: unknown;
  readonly totalJSHeapSize?: unknown;
}

export interface RendererMemoryRuntime {
  readonly performance?: { readonly memory?: RendererPerformanceMemory };
  readonly navigator?: { readonly deviceMemory?: unknown };
}

export type RendererMemorySignalSource = "performance.memory" | "navigator.deviceMemory" | "none";

export interface RendererMemorySignal {
  readonly source: RendererMemorySignalSource;
  readonly trustworthy: boolean;
  /** Available JS heap headroom when the renderer exposes it. */
  readonly availableBytes: number | null;
  readonly usedBytes: number | null;
  readonly limitBytes: number | null;
  /** Approximate physical memory, never treated as available headroom. */
  readonly deviceMemoryBytes: number | null;
  readonly detail: string;
}

export interface RendererMemoryEstimate {
  readonly rowCount: number;
  readonly dimension: number;
  readonly assumptions: {
    readonly scalarBytes: 8;
    readonly workerStructuredCloneIncluded: true;
    readonly pcaCovarianceUpperBoundIncluded: true;
    readonly safetyMarginIncluded: true;
    readonly estimateIsNotAnAllocatorGuarantee: true;
  };
  readonly components: {
    readonly pcaSampleRows: number;
    readonly pcaComponents: number;
    readonly umapComponents: number;
  };
  readonly bytes: {
    readonly vectorMatrix: number;
    readonly workerClone: number;
    readonly normalizedMatrix: number;
    readonly pcaPilotProjection: number;
    readonly pcaProjection: number;
    readonly pcaCovarianceUpperBound: number;
    readonly umapWorkingCopies: number;
    readonly hierarchyEstimate: number;
    readonly lowerBound: number;
    readonly orchestrationWorkingSet: number;
    readonly safetyMargin: number;
    readonly rendererOverhead: number;
    readonly predictedAdditional: number;
  };
}

export type RendererMemoryPreflightStatus = "pass" | "warning" | "blocked" | "unavailable";

export interface RendererMemoryPreflight {
  readonly status: RendererMemoryPreflightStatus;
  readonly canProceed: boolean;
  readonly hardBlock: boolean;
  readonly rowCount: number;
  readonly dimension: number;
  readonly estimate: RendererMemoryEstimate | null;
  readonly signal: RendererMemorySignal;
  readonly estimatedToAvailableRatio: number | null;
  readonly detail: string;
  readonly error?: string;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function safeProduct(...values: number[]): number {
  let product = 1;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0 || product >= Number.MAX_SAFE_INTEGER) return Number.MAX_SAFE_INTEGER;
    product *= value;
  }
  return Math.min(Number.MAX_SAFE_INTEGER, Math.ceil(product));
}

function safeSum(...values: number[]): number {
  let sum = 0;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0 || sum >= Number.MAX_SAFE_INTEGER - value) return Number.MAX_SAFE_INTEGER;
    sum += value;
  }
  return Math.min(Number.MAX_SAFE_INTEGER, Math.ceil(sum));
}

export function formatMemoryBytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "unknown memory";
  if (value >= BYTES_PER_GIB) return `${(value / BYTES_PER_GIB).toFixed(1)} GiB`;
  if (value >= BYTES_PER_MIB) return `${(value / BYTES_PER_MIB).toFixed(1)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${Math.round(value)} B`;
}

export function estimateRendererClusteringMemory(rowCount: number, dimension: number): RendererMemoryEstimate {
  if (!Number.isSafeInteger(rowCount) || rowCount < 1) throw new Error("rowCount must be a positive safe integer");
  if (!Number.isSafeInteger(dimension) || dimension < 1) throw new Error("dimension must be a positive safe integer");

  const pcaSampleRows = Math.min(2000, rowCount);
  const pcaComponents = Math.min(512, dimension, Math.max(1, rowCount - 1));
  const umapComponents = Math.min(20, Math.max(2, rowCount - 1));
  const vectorMatrix = safeProduct(rowCount, dimension, 8);
  const workerClone = vectorMatrix;
  const normalizedMatrix = vectorMatrix;
  const pcaPilotProjection = safeProduct(pcaSampleRows, pcaComponents, 8);
  const pcaProjection = safeProduct(rowCount, pcaComponents, 8);
  const pcaCovarianceUpperBound = safeProduct(dimension, dimension, 8);
  const umapWorkingBase = safeProduct(rowCount, umapComponents, 8);
  const umapWorkingCopies = safeProduct(umapWorkingBase, 3);
  const hierarchyEstimate = safeProduct(rowCount, 256);
  const lowerBound = safeSum(vectorMatrix, normalizedMatrix, pcaProjection, umapWorkingBase);

  // Keep this aligned with the offline large-vault estimator, then add an
  // explicit margin for renderer bookkeeping, JS object overhead, and the
  // allocation pattern difference between renderer and worker/WASM paths.
  const orchestrationWorkingSet = safeSum(
    safeProduct(vectorMatrix, 2.5),
    workerClone,
    normalizedMatrix,
    safeProduct(pcaPilotProjection, 2),
    safeProduct(pcaProjection, 2),
    pcaCovarianceUpperBound,
    umapWorkingCopies,
    hierarchyEstimate
  );
  const safetyMargin = Math.max(BYTES_PER_MIB * 64, safeProduct(orchestrationWorkingSet, 0.5));
  const rendererOverhead = BYTES_PER_MIB * 128;
  const predictedAdditional = safeSum(orchestrationWorkingSet, safetyMargin, rendererOverhead);

  return {
    rowCount,
    dimension,
    assumptions: {
      scalarBytes: 8,
      workerStructuredCloneIncluded: true,
      pcaCovarianceUpperBoundIncluded: true,
      safetyMarginIncluded: true,
      estimateIsNotAnAllocatorGuarantee: true
    },
    components: { pcaSampleRows, pcaComponents, umapComponents },
    bytes: {
      vectorMatrix,
      workerClone,
      normalizedMatrix,
      pcaPilotProjection,
      pcaProjection,
      pcaCovarianceUpperBound,
      umapWorkingCopies,
      hierarchyEstimate,
      lowerBound,
      orchestrationWorkingSet,
      safetyMargin,
      rendererOverhead,
      predictedAdditional
    }
  };
}

export function readRendererMemorySignal(runtime: RendererMemoryRuntime = globalThis as unknown as RendererMemoryRuntime): RendererMemorySignal {
  const performanceMemory = runtime.performance?.memory;
  const limitBytes = finiteNumber(performanceMemory?.jsHeapSizeLimit);
  const usedBytes = finiteNumber(performanceMemory?.usedJSHeapSize);
  if (limitBytes !== null && usedBytes !== null && limitBytes > 0 && usedBytes <= limitBytes) {
    const availableBytes = Math.max(0, limitBytes - usedBytes);
    return {
      source: "performance.memory",
      trustworthy: true,
      availableBytes,
      usedBytes,
      limitBytes,
      deviceMemoryBytes: null,
      detail: `${formatMemoryBytes(availableBytes)} renderer JS-heap headroom reported by performance.memory`
    };
  }

  const deviceMemoryGiB = finiteNumber(runtime.navigator?.deviceMemory);
  if (deviceMemoryGiB !== null && deviceMemoryGiB > 0) {
    const deviceMemoryBytes = safeProduct(deviceMemoryGiB, BYTES_PER_GIB);
    return {
      source: "navigator.deviceMemory",
      trustworthy: false,
      availableBytes: null,
      usedBytes: null,
      limitBytes: null,
      deviceMemoryBytes,
      detail: `navigator.deviceMemory reports approximately ${formatMemoryBytes(deviceMemoryBytes)}, but available headroom is not exposed`
    };
  }

  return {
    source: "none",
    trustworthy: false,
    availableBytes: null,
    usedBytes: null,
    limitBytes: null,
    deviceMemoryBytes: null,
    detail: "renderer available-memory headroom is not exposed"
  };
}

function validVectorShape(vectors: ReadonlyArray<ReadonlyArray<number>>): { rowCount: number; dimension: number } | null {
  if (!Array.isArray(vectors) || vectors.length < 1 || !Array.isArray(vectors[0]) || vectors[0].length < 1) return null;
  const dimension = vectors[0].length;
  if (!Number.isSafeInteger(dimension) || vectors.some((vector) => !Array.isArray(vector) || vector.length !== dimension)) return null;
  return { rowCount: vectors.length, dimension };
}

export function preflightRendererClusteringMemory(vectors: ReadonlyArray<ReadonlyArray<number>>, runtime: RendererMemoryRuntime = globalThis as unknown as RendererMemoryRuntime): RendererMemoryPreflight {
  const shape = validVectorShape(vectors);
  const signal = readRendererMemorySignal(runtime);
  if (!shape) {
    return {
      status: "unavailable",
      canProceed: true,
      hardBlock: false,
      rowCount: Array.isArray(vectors) ? vectors.length : 0,
      dimension: 0,
      estimate: null,
      signal,
      estimatedToAvailableRatio: null,
      detail: "Memory preflight skipped because the embedding matrix shape is unavailable; continuing without a memory hard block."
    };
  }

  const estimate = estimateRendererClusteringMemory(shape.rowCount, shape.dimension);
  const predicted = estimate.bytes.predictedAdditional;
  if (signal.trustworthy && signal.availableBytes !== null) {
    const available = signal.availableBytes;
    const ratio = available > 0 ? predicted / available : Number.POSITIVE_INFINITY;
    if (ratio >= MEMORY_DANGER_RATIO) {
      const detail = `Memory preflight blocked clustering: conservative estimate ${formatMemoryBytes(predicted)} for ${shape.rowCount.toLocaleString()} × ${shape.dimension.toLocaleString()} vectors exceeds the safe headroom threshold; ${signal.detail}.`;
      return {
        status: "blocked",
        canProceed: false,
        hardBlock: true,
        rowCount: shape.rowCount,
        dimension: shape.dimension,
        estimate,
        signal,
        estimatedToAvailableRatio: ratio,
        detail,
        error: `${detail} Close large panes or other memory-heavy applications, reduce the selected vault, and retry.`
      };
    }
    if (ratio >= MEMORY_WARNING_RATIO) {
      return {
        status: "warning",
        canProceed: true,
        hardBlock: false,
        rowCount: shape.rowCount,
        dimension: shape.dimension,
        estimate,
        signal,
        estimatedToAvailableRatio: ratio,
        detail: `Memory estimate ${formatMemoryBytes(predicted)} for ${shape.rowCount.toLocaleString()} × ${shape.dimension.toLocaleString()} vectors; ${signal.detail} (${Math.round(ratio * 100)}% of observed headroom). Continuing with monitoring.`
      };
    }
    return {
      status: "pass",
      canProceed: true,
      hardBlock: false,
      rowCount: shape.rowCount,
      dimension: shape.dimension,
      estimate,
      signal,
      estimatedToAvailableRatio: ratio,
      detail: `Memory estimate ${formatMemoryBytes(predicted)} for ${shape.rowCount.toLocaleString()} × ${shape.dimension.toLocaleString()} vectors; ${signal.detail}.`
    };
  }

  return {
    status: "warning",
    canProceed: true,
    hardBlock: false,
    rowCount: shape.rowCount,
    dimension: shape.dimension,
    estimate,
    signal,
    estimatedToAvailableRatio: null,
    detail: `Memory estimate ${formatMemoryBytes(predicted)} for ${shape.rowCount.toLocaleString()} × ${shape.dimension.toLocaleString()} vectors; ${signal.detail}. Continuing without a hard block because trustworthy available memory is unavailable.`
  };
}
