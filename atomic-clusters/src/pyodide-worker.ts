import "atomic-clusters-wasm-bootstrap";
import { discoverPcaFeatures, NumericKernel } from "./clustering";
import { PYODIDE_CORE_SOURCE } from "./pyodide-core-source";
import { ClusteringConfig, PyodideClusterResult } from "./types";
import { loadWasmKernel } from "./wasm-loader";

const DEFAULT_PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js";
type Message =
  | { type: "INIT"; version: 1; pyodideUrl?: string; indexURL?: string }
  | { type: "CLUSTER"; jobId: string; ids: string[]; embeddings: number[][]; config: ClusteringConfig }
  | { type: "CANCEL"; jobId?: string };
type Runtime = {
  FS: { writeFile(path: string, contents: string): void; mkdirTree(path: string): void };
  loadPackage(packages: string | string[]): Promise<void>;
  runPython(code: string, options?: { locals?: unknown }): unknown;
  runPythonAsync(code: string, options?: { locals?: unknown }): Promise<unknown>;
  toPy(value: unknown): any;
  ffi: { create_proxy<T extends (...args: any[]) => any>(callback: T): any };
};
type ProxyValue = { toJs?: (options?: Record<string, unknown>) => any; destroy?: () => void };
type Scope = typeof globalThis & {
  loadPyodide?: (options: { indexURL: string }) => Promise<Runtime>;
  importScripts?: (...urls: string[]) => void;
  __ATOMIC_CLUSTERS_WASM__?: unknown;
};

const scope = globalThis as Scope;
let runtimePromise: Promise<Runtime> | null = null;
let cancelled = false;
let wasmKernel: NumericKernel | undefined;

function post(message: Record<string, unknown>): void {
  (scope as unknown as { postMessage: (value: unknown) => void }).postMessage(message);
}

function toPlain(value: unknown): any {
  const proxy = value as ProxyValue;
  if (proxy && typeof proxy.toJs === "function") return proxy.toJs({ create_proxies: false, dict_converter: Object.fromEntries });
  return value;
}

function destroy(value: unknown): void {
  (value as ProxyValue | undefined)?.destroy?.();
}

async function loadRuntime(pyodideUrl = DEFAULT_PYODIDE_URL, indexURL?: string): Promise<Runtime> {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      if (!scope.loadPyodide) {
        if (!scope.importScripts) throw new Error("Pyodide worker requires importScripts or a loadPyodide global.");
        scope.importScripts(pyodideUrl);
      }
      if (!scope.loadPyodide) throw new Error("The configured Pyodide script did not expose loadPyodide.");
      const runtime = await scope.loadPyodide({ indexURL: indexURL || pyodideUrl.slice(0, pyodideUrl.lastIndexOf("/") + 1) });
      await runtime.loadPackage(["numpy", "scikit-learn"]);
      runtime.FS.mkdirTree("/atomic-clusters-pyodide");
      for (const [relativePath, source] of Object.entries(PYODIDE_CORE_SOURCE)) {
        const directory = relativePath.slice(0, relativePath.lastIndexOf("/"));
        if (directory) runtime.FS.mkdirTree(`/atomic-clusters-pyodide/${directory}`);
        runtime.FS.writeFile(`/atomic-clusters-pyodide/${relativePath}`, source);
      }
      runtime.runPython("import sys; sys.path.insert(0, '/atomic-clusters-pyodide')");
      return runtime;
    })();
  }
  return runtimePromise;
}

function configProxy(runtime: Runtime, config: ClusteringConfig): any {
  return runtime.toPy(config);
}

function pythonResultToPluginResult(raw: Record<string, any>, ids: string[]): PyodideClusterResult {
  const pca = raw.pca || {};
  const discovery = raw.discovery || {};
  const tree = raw.hierarchy?.tree || {};
  const merges = (tree.merges || []).map((merge: Record<string, any>) => ({
    id: Number(merge.node), left: Number(merge.left), right: Number(merge.right),
    distance: Number(merge.distance), mass: Number(merge.mass)
  }));
  const leaves = (tree.leaves || []).map((leaf: Record<string, any>) => Number(leaf.leaf));
  return {
    schemaVersion: 2,
    ids: Array.isArray(raw.ids) ? raw.ids.map(String) : ids,
    leafLabels: (discovery.leaf_labels || []).map(Number),
    probabilities: (discovery.probabilities || []).map(Number),
    outlierProxy: (discovery.outlier_scores || []).map(Number),
    pca: {
      selected: Number(pca.selected_dimension), explainedVariance: Number(pca.candidates?.[pca.candidates.length - 1]?.cumulative_explained_variance || 0),
      totalVariance: 0, candidates: (pca.candidates || []).map((candidate: Record<string, any>) => Number(candidate.dimension)),
      preservationCandidates: (pca.candidates || []).map((candidate: Record<string, any>) => ({
        dimension: Number(candidate.dimension), meanNeighborPreservation: Number(candidate.mean_knn_preservation),
        neighborPreservationByK: Object.fromEntries(Object.entries(candidate.knn_preservation_by_k || {}).map(([key, value]) => [Number(key), Number(value)])),
        neighborPreservationGain: candidate.knn_preservation_gain == null ? null : Number(candidate.knn_preservation_gain)
      })),
      selectionReason: pca.selection_reason === "fixed_dimension" ? "all_gains_meet_minimum_use_maximum_dimension" : "first_below_minimum_gain_use_previous_dimension",
      sampleSize: ids.length, varianceTarget: 0.9
    },
    hierarchy: { leaves, merges, root: merges.length ? merges[merges.length - 1].id : (leaves[0] ?? null) },
    timings: {},
    pyodide: raw
  };
}

async function runCluster(request: Extract<Message, { type: "CLUSTER" }>): Promise<PyodideClusterResult> {
  const runtime = await loadRuntime();
  cancelled = false;
  const options = request.config || {};
  const locals = runtime.toPy({ embeddings: request.embeddings, ids: request.ids, config: options });
  let pcaProxy: any;
  try {
    pcaProxy = await runtime.runPythonAsync(`
from atomic_clustering.pca import fit_pca
_atomic_pca = fit_pca(embeddings, **dict(config.get('pca', {})))
_atomic_pca.features.tolist()
`, { locals });
  } finally { destroy(locals); }
  const pcaFeatures = toPlain(pcaProxy);
  destroy(pcaProxy);
  if (cancelled) throw new Error("Clustering cancelled");
  const discovery = await discoverPcaFeatures(pcaFeatures, options, {
    kernel: wasmKernel,
    signal: { get cancelled() { return cancelled; } },
    onProgress: (phase, progress) => post({ type: "PROGRESS", jobId: request.jobId, phase, progress })
  });
  const clusterCount = Math.max(0, ...discovery.labels) + 1;
  const memberships = discovery.labels.map((label, row) => Array.from({ length: clusterCount }, (_, column) => column === label ? discovery.probabilities[row] : 0));
  const discoveryOutput = runtime.toPy({
    umap_features: discovery.umapFeatures, leaf_labels: discovery.labels, memberships,
    probabilities: discovery.probabilities, outlier_scores: discovery.outlierProxy,
    configuration: { runtime: "js-umap-wasm-hdbscan", seed: options.seed ?? 42 }
  });
  const runner = runtime.ffi.create_proxy(() => discoveryOutput);
  const callLocals = runtime.toPy({ embeddings: request.embeddings, ids: request.ids, config: options, runner });
  let resultProxy: any;
  try {
    resultProxy = await runtime.runPythonAsync(`
from atomic_clustering import cluster_documents
cluster_documents(embeddings, ids=ids, config=config, discovery_runner=runner)
`, { locals: callLocals });
  } finally {
    destroy(callLocals); destroy(runner); destroy(discoveryOutput);
  }
  const result = toPlain(resultProxy);
  destroy(resultProxy);
  return pythonResultToPluginResult(result, request.ids);
}

(scope as unknown as { onmessage: (event: MessageEvent<Message>) => void }).onmessage = (event) => {
  const request = event.data;
  if (request.type === "CANCEL") { cancelled = true; return; }
  if (request.type === "INIT") {
    void loadRuntime(request.pyodideUrl, request.indexURL).then(() => {
      wasmKernel = loadWasmKernel();
      post({ type: "READY", version: 1 });
    }).catch((error) => post({ type: "ERROR", code: "PYODIDE_INIT_FAILED", message: error instanceof Error ? error.message : String(error) }));
    return;
  }
  if (request.type === "CLUSTER") {
    void runCluster(request).then((result) => post({ type: "RESULT", jobId: request.jobId, result })).catch((error) => post({ type: "ERROR", jobId: request.jobId, code: cancelled ? "CANCELLED" : "PYODIDE_CLUSTER_FAILED", message: error instanceof Error ? error.message : String(error) }));
  }
};
