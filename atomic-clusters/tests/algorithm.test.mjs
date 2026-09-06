import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadClustering() {
  const source = await readFile(new URL("../src/clustering.ts", import.meta.url), "utf8");
  // PCA tests exercise orchestration only; this keeps the UMAP dependency out
  // of the data-URL module while preserving the production source unchanged.
  const stubbed = source.replace(
    'import { UMAP } from "umap-js";',
    "const UMAP = class { async fitAsync(points) { return points; } };"
  );
  const result = await transform(stubbed, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

function pcaKernel(spectrum, calls) {
  return {
    normalize(rows) { return rows; },
    pca(rows, components) {
      calls.push(components);
      return { projected: rows.map((row) => row.slice(0, components)), explained: spectrum.slice(0, components) };
    },
    cosineDistances() { return []; },
    exactKnn(rows, k) { return rows.map((_, index) => Array.from({ length: k }, (_, rank) => (index + rank + 1) % rows.length)); },
    mst() { return []; },
    hdbscan(rows) { return { labels: rows.map(() => -1), probabilities: rows.map(() => 0) }; }
  };
}

test("fixture is rectangular and has stable note IDs", async () => {
  const fixture = JSON.parse(await readFile(new URL("./fixtures/embeddings.json", import.meta.url)));
  assert.equal(fixture.ids.length, fixture.embeddings.length);
  assert.equal(new Set(fixture.ids).size, fixture.ids.length);
  assert.ok(fixture.embeddings.every((row) => row.length === fixture.embeddings[0].length));
});

test("worker protocol documents one job at a time", async () => {
  const source = await readFile(new URL("../src/types.ts", import.meta.url), "utf8");
  for (const message of ["INIT", "CLUSTER", "CANCEL", "READY", "PROGRESS", "RESULT", "ERROR"]) assert.match(source, new RegExp(`\\"${message}\\"`));
});

test("offline build has no runtime package installation", async () => {
  const source = await readFile(new URL("../README.md", import.meta.url), "utf8");
  assert.match(source, /offline/i);
  assert.doesNotMatch(source, /micropip|pip install/i);
});

test("Community release embeds the worker and confirms remote transmission", async () => {
  const build = await readFile(new URL("../build.mjs", import.meta.url), "utf8");
  const client = await readFile(new URL("../src/worker-client.ts", import.meta.url), "utf8");
  const embedding = await readFile(new URL("../src/embedding.ts", import.meta.url), "utf8");
  assert.match(build, /embedded-worker/);
  assert.match(client, /eval: true/);
  assert.match(embedding, /confirmTransmission/);
  assert.match(embedding, /outputDimensionality: 768/);
});

test("local provider requires explicit model installation and keeps inference offline", async () => {
  const settings = await readFile(new URL("../src/settings.ts", import.meta.url), "utf8");
  const embedding = await readFile(new URL("../src/embedding.ts", import.meta.url), "utf8");
  assert.match(settings, /explicit consent/);
  assert.match(settings, /Delete/);
  assert.match(embedding, /downloadModel/);
  assert.match(embedding, /embed\(\) never reaches this function/);
  assert.match(embedding, /SHA-256/);
});

test("PCA selection uses sharp, smooth, and flat k-NN preservation curves", async () => {
  const { choosePcaPreservationCandidate } = await loadClustering();
  const diagnostics = (scores) => scores.map(([dimension, score], index) => ({
    dimension, meanNeighborPreservation: score, neighborPreservationByK: { 15: score, 30: score },
    neighborPreservationGain: index ? score - scores[index - 1][1] : null
  }));
  const sharp = choosePcaPreservationCandidate(diagnostics([[32, 0.4], [64, 0.75], [96, 0.78], [128, 0.79]]));
  assert.equal(sharp.selected.dimension, 64);
  assert.equal(sharp.reason, "first_below_minimum_gain_use_previous_dimension");

  const smooth = choosePcaPreservationCandidate(diagnostics([[32, 0.555], [64, 0.658], [96, 0.717], [128, 0.759], [160, 0.8], [192, 0.82], [224, 0.833], [256, 0.84]]));
  assert.equal(smooth.selected.dimension, 160);
  assert.equal(smooth.reason, "global_preservation_knee_after_local_plateau");

  const flat = choosePcaPreservationCandidate(diagnostics([[32, 0.5], [64, 0.5], [96, 0.5]]));
  assert.equal(flat.selected.dimension, 32);
  assert.equal(flat.reason, "first_below_minimum_gain_use_previous_dimension");
});

test("PCA preservation uses one bounded pilot plus final PCA, respects the cap, and is deterministic", async () => {
  const { clusterEmbeddings } = await loadClustering();
  const sharp = [...Array(8).fill(1), ...Array(504).fill(0.01)];
  const largeRows = Array.from({ length: 520 }, (_, row) => Array.from({ length: 512 }, (_, column) => ((row + column) % 7) + 1));
  const firstCalls = [];
  const first = await clusterEmbeddings(largeRows.map((_, index) => `note-${index}`), largeRows, {}, { kernel: pcaKernel(sharp, firstCalls) });
  const secondCalls = [];
  const second = await clusterEmbeddings(largeRows.map((_, index) => `note-${index}`), largeRows, {}, { kernel: pcaKernel(sharp, secondCalls) });
  assert.ok(first.pca.candidates.includes(first.pca.selected));
  assert.deepEqual(first.pca.candidates, [32, 64, 96, 128, 160, 192, 224, 256]);
  assert.deepEqual(firstCalls, [256, first.pca.selected]);
  assert.equal(firstCalls.length, 2);
  assert.ok(firstCalls[0] < 512);
  assert.equal(firstCalls.includes(512), false);
  assert.deepEqual(second.pca, first.pca);
  assert.deepEqual(secondCalls, firstCalls);

  const deferred = await clusterEmbeddings(largeRows.map((_, index) => `deferred-${index}`), largeRows, { deferVisualization: true }, { kernel: pcaKernel(sharp, []) });
  assert.equal(deferred.visualization, undefined);
  assert.equal(deferred.schemaVersion, 6);
  assert.equal(deferred.hierarchyPlacements.length, largeRows.length);

  const cappedRows = Array.from({ length: 80 }, (_, row) => Array.from({ length: 40 }, (_, column) => ((row + column) % 5) + 1));
  const cappedCalls = [];
  const capped = await clusterEmbeddings(cappedRows.map((_, index) => `cap-${index}`), cappedRows, { pcaMaxComponents: 40 }, { kernel: pcaKernel(Array(40).fill(1), cappedCalls) });
  assert.equal(capped.pca.selected, 32);
  assert.deepEqual(capped.pca.candidates, [32]);
  assert.deepEqual(cappedCalls, [40, 32]);
});

test("PCA preservation reuses its prefix row workspace across candidate widths", async () => {
  const { selectPcaByPreservation } = await loadClustering();
  const normalizedInputs = [];
  const kernel = {
    normalize(rows) { normalizedInputs.push(rows); return rows; },
    exactKnn(rows, k) { return rows.map((_, index) => Array.from({ length: k }, (_, rank) => (index + rank + 1) % rows.length)); }
  };
  const rows = Array.from({ length: 8 }, (_, row) => [row + 1, row + 2, row + 3]);
  selectPcaByPreservation(rows, rows.map((row) => row.slice()), [1, 2, 3], rows.length, 0.9, kernel);
  // One call is the original-space reference; all prefix calls share one
  // matrix and only resize/refill its row buffers between candidates.
  assert.equal(normalizedInputs.length, 3);
  assert.equal(new Set(normalizedInputs.slice(0, 1)).size, 1);
  assert.equal(new Set(normalizedInputs.slice(1)).size, 1);
});

test("TypeScript fallback limit is enforced at the fallback boundary only", async () => {
  const { discoverPcaFeatures, ClusteringCapabilityError, TYPESCRIPT_FALLBACK_MAX_ROWS } = await loadClustering();
  const rows = Array.from({ length: TYPESCRIPT_FALLBACK_MAX_ROWS + 1 }, (_, row) => [row % 17, (row * 3) % 19]);

  await assert.rejects(
    () => discoverPcaFeatures(rows, { minClusterSize: 5 }, {}),
    (error) => error instanceof ClusteringCapabilityError && /TypeScript clustering fallback is limited to 511 rows/.test(error.message) && /WASM/.test(error.message)
  );

  const customProvider = {
    fit(input) { return { labels: input.map(() => -1), probabilities: input.map(() => 0) }; }
  };
  const customResult = await discoverPcaFeatures(rows, { minClusterSize: 5 }, { hdbscan: customProvider });
  assert.equal(customResult.labels.length, rows.length);
  assert.equal(customResult.probabilities.length, rows.length);

  const customKernel = {
    normalize(input) { return input; },
    pca(input, components) { return { projected: input.map((row) => row.slice(0, components)), explained: Array(components).fill(1) }; },
    cosineDistances() { return []; },
    exactKnn() { return []; },
    mst() { throw new Error("fallback MST should not be reached"); },
    hdbscan(input) { return { labels: input.map(() => -1), probabilities: input.map(() => 0) }; }
  };
  const wasmLikeResult = await discoverPcaFeatures(rows, { minClusterSize: 5 }, { kernel: customKernel });
  assert.equal(wasmLikeResult.labels.length, rows.length);
});
