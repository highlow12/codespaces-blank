export type EmbeddingProviderId = "gemini" | "local";

export interface PluginSettings {
  embeddingProvider: EmbeddingProviderId;
  geminiModel: string;
  geminiSecretRef: string;
  localModel: string;
  excludedFolders: string[];
  minClusterSize: number;
  minSamples: number;
  umapNeighbors: number;
  umapMinDist: number;
  pcaVarianceTarget: number;
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

export interface ClusterResult {
  schemaVersion: 1;
  ids: string[];
  leafLabels: number[];
  probabilities: number[];
  outlierProxy: number[];
  pca: PcaSelection;
  hierarchy: HierarchyTree;
  timings: Record<string, number>;
}

export type WorkerRequest =
  | { type: "INIT"; version: 1 }
  | { type: "CLUSTER"; jobId: string; ids: string[]; embeddings: number[][]; config: ClusteringConfig }
  | { type: "CANCEL"; jobId?: string };

export type WorkerResponse =
  | { type: "READY"; version: 1 }
  | { type: "PROGRESS"; jobId: string; phase: string; progress: number }
  | { type: "RESULT"; jobId: string; result: ClusterResult }
  | { type: "ERROR"; jobId?: string; code: string; message: string };
