export type EmbeddingProviderId = "gemini" | "local";
export type LocalExecutionProvider = "auto" | "webgpu" | "wasm";

export interface PluginSettings {
  embeddingProvider: EmbeddingProviderId;
  geminiModel: string;
  geminiSecretRef: string;
  localModel: string;
  /** Prefer WebGPU when available, or force the WASM CPU backend. */
  localExecutionProvider?: LocalExecutionProvider;
  excludedFolders: string[];
  minClusterSize: number;
  minSamples: number;
  umapNeighbors: number;
  umapMinDist: number;
  pcaVarianceTarget: number;
  /** Optional Python reference runtime; WASM remains the desktop default. */
  clusteringRuntime?: "wasm" | "pyodide";
  pyodideUrl?: string;
  /** Generate titles for every leaf and merge node after a successful build. */
  clusterTitlesEnabled?: boolean;
}

export interface NoteRecord {
  path: string;
  title: string;
  content: string;
  mtime: number;
  hash: string;
}

export interface CachedEmbedding {
  path: string;
  hash: string;
  provider: string;
  model: string;
  vector: number[];
}

export type EmbeddingLogStatus = "success" | "failure" | "cached";

/** Persisted diagnostics deliberately exclude note content, vectors, and secrets. */
export interface EmbeddingLogEntry {
  path: string;
  timestamp: string;
  provider: string;
  model: string;
  status: EmbeddingLogStatus;
  durationMs: number;
  error?: string;
}

export interface EmbeddingRunLog {
  version: 1;
  startedAt: string;
  completedAt: string;
  provider: string;
  model: string;
  total: number;
  succeeded: number;
  failed: number;
  cached: number;
  entries: EmbeddingLogEntry[];
  status?: "completed" | "failed" | "cancelled";
  stage?: "preflight" | "embedding" | "clustering";
  runtime?: { backend: "webgpu" | "wasm"; fallbackReason?: string };
  error?: string;
}

export interface ClusteringConfig {
  seed?: number;
  /**
   * Legacy display setting retained for saved configurations. PCA dimension
   * selection uses k-NN preservation rather than a variance cutoff.
   */
  pcaVarianceTarget?: number;
  pcaSampleSize?: number;
  /** Lower bound for the automatically selected PCA dimension. Default: 32. */
  pcaMinComponents?: number;
  pcaMaxComponents?: number;
  /**
   * Width of the single sampled PCA spectrum used to locate the knee.
   * Default: 256, bounded by pcaMaxComponents, rank, and embedding dimension.
   */
  pcaKneeProbeComponents?: number;
  umapComponents?: number;
  umapNeighbors?: number;
  umapMinDist?: number;
  minClusterSize?: number;
  /** HDBSCAN core-distance neighbourhood size. Default: 3. */
  minSamples?: number;
}

export interface PcaPreservationCandidate {
  dimension: number;
  meanNeighborPreservation: number;
  neighborPreservationByK: Record<number, number>;
  neighborPreservationGain: number | null;
}

export interface PcaSelection {
  selected: number;
  /** Fraction of the total centered variance (the covariance trace) retained. */
  explainedVariance: number;
  /** Total centered sample variance used as the denominator for this fraction. */
  totalVariance: number;
  /** PCA prefix widths evaluated against original-space neighborhoods. */
  candidates: number[];
  preservationCandidates?: PcaPreservationCandidate[];
  selectionReason?: "all_gains_meet_minimum_use_maximum_dimension" | "first_below_minimum_gain_use_previous_dimension" | "global_preservation_knee_after_local_plateau" | "small_dataset";
  sampleSize: number;
  varianceTarget: number;
}

export type VisualizationCoordinate = [number, number];

export interface VisualizationConfiguration {
  runtime: "umap-js" | string;
  seed: number;
  nComponents: 2;
  nNeighbors: number;
  minDist: number;
  spread: number;
  targetMetric?: "categorical";
  targetWeight?: number;
}

/** Optional, row-aligned 2D projection used by the cluster explorer. */
export interface ClusterVisualization {
  coordinates: VisualizationCoordinate[];
  labels: number[];
  configuration: VisualizationConfiguration;
  timings?: Record<string, number>;
}

export interface HierarchyMerge {
  id: number;
  left: number;
  right: number;
  distance: number;
  mass: number;
}

export interface HierarchyTree {
  leaves: number[];
  merges: HierarchyMerge[];
  root: number | null;
}

export type ClusterTitleStatus = "generated" | "empty";

/** Metadata persisted with a v3 result. It intentionally contains no note text. */
export interface ClusterTitleGeneration {
  method: "keywords";
  algorithmVersion: string;
  inputFingerprint: string;
  generatedAt?: string;
  statuses: Record<string, ClusterTitleStatus>;
  nodeCount: number;
  durationMs: number;
  scores?: Record<string, Array<{ keyword: string; score: number }>>;
}

export interface ClusterResult {
  schemaVersion: 1 | 2 | 3 | 4 | 5;
  ids: string[];
  leafLabels: number[];
  probabilities: number[];
  outlierProxy: number[];
  pca: PcaSelection;
  hierarchy: HierarchyTree;
  timings: Record<string, number>;
  visualization?: ClusterVisualization;
  /** Complete note × leaf soft-membership matrix, aligned with `ids`. */
  softMemberships?: number[][];
  /** Stable display ordering of leaf labels. */
  leafOrder?: number[];
  /** Node id (leaf label or merge id) to the normalized display title. */
  titles?: Record<string, string>;
  titleGeneration?: ClusterTitleGeneration;
}

export type WorkerRequest =
  | { type: "INIT"; version: 1; pyodideUrl?: string; indexURL?: string }
  | { type: "CLUSTER"; jobId: string; ids: string[]; embeddings: number[][]; config: ClusteringConfig }
  | { type: "CANCEL"; jobId?: string };

export type WorkerResponse =
  | { type: "READY"; version: 1 }
  | { type: "PROGRESS"; jobId: string; phase: string; progress: number }
  | { type: "RESULT"; jobId: string; result: ClusterResult }
  | { type: "ERROR"; jobId?: string; code: string; message: string };

export interface PyodideClusterResult extends ClusterResult {
  /** Raw JSON-friendly result emitted by pyodide_core.cluster_documents. */
  pyodide?: Record<string, unknown>;
}
