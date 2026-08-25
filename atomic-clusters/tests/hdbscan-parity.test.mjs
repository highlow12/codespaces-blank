import test from "node:test";
import assert from "node:assert/strict";
import { build } from "esbuild";

async function load() {
  const result = await build({
    stdin: { contents: `export * from ${JSON.stringify(new URL("../src/clustering.ts", import.meta.url).pathname)}; export * from ${JSON.stringify(new URL("../src/hdbscan-parity.ts", import.meta.url).pathname)};`, loader: "ts", resolveDir: new URL("..", import.meta.url).pathname },
    bundle: true, format: "esm", platform: "node", target: "node20", write: false, logLevel: "silent"
  });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].contents).toString("base64")}`);
}

test("HDBSCAN parity labels are permutation invariant and preserve membership metrics", async () => {
  const { compareHdbscanOutputs } = await load();
  const reference = {
    labels: [0, 0, 1, -1], probabilities: [0.8, 0.9, 0.7, 0], outlierProxy: [0.2, 0.1, 0.3, 1],
    memberships: [[0.8, 0], [0.9, 0], [0, 0.7], [0, 0]]
  };
  const candidate = { labels: [1, 1, 0, -1], probabilities: [0.8, 0.9, 0.7, 0], outlierProxy: [0.2, 0.1, 0.3, 1], memberships: [[0, 0.8], [0, 0.9], [0.7, 0], [0, 0]] };
  const metrics = compareHdbscanOutputs(reference, candidate);
  assert.equal(metrics.labelAgreement, 1);
  assert.equal(metrics.noiseAgreement, 1);
  assert.equal(metrics.probabilityMae, 0);
  assert.equal(metrics.outlierMae, 0);
  assert.equal(metrics.membershipMae, 0);
  assert.deepEqual(metrics.mapping, { "0": 1, "1": 0 });
});

test("HDBSCAN parity reports missing soft cross-cluster mass instead of hiding it", async () => {
  const { compareHdbscanOutputs } = await load();
  const reference = { labels: [0, 1], probabilities: [0.8, 0.8], outlierProxy: [0.2, 0.2], memberships: [[0.8, 0.15], [0.1, 0.8]] };
  const assignedOnly = { labels: [0, 1], probabilities: [0.8, 0.8], outlierProxy: [0.2, 0.2] };
  const metrics = compareHdbscanOutputs(reference, assignedOnly);
  assert.equal(metrics.labelAgreement, 1);
  assert.equal(metrics.probabilityMae, 0);
  assert(metrics.membershipMae > 0);
  assert(metrics.membershipMaxError >= 0.15);
});

test("HDBSCAN parity uses optimal assignment beyond eight clusters", async () => {
  const { compareHdbscanOutputs } = await load();
  const reference = []; const candidate = [];
  const add = (count, candidateLabel, referenceLabel) => { for (let index = 0; index < count; index++) { candidate.push(candidateLabel); reference.push(referenceLabel); } };
  add(10, 0, 0); add(9, 0, 1); add(9, 1, 0);
  for (let cluster = 2; cluster < 9; cluster++) add(1, cluster, cluster);
  const result = compareHdbscanOutputs(
    { labels: reference, probabilities: reference.map(() => 1), outlierProxy: reference.map(() => 0) },
    { labels: candidate, probabilities: candidate.map(() => 1), outlierProxy: candidate.map(() => 0) }
  );
  assert.equal(result.referenceClusters, 9);
  assert.equal(result.candidateClusters, 9);
  assert.equal(result.mapping["0"], 1);
  assert.equal(result.mapping["1"], 0);
  // The optimal overlap is 9 + 9 + 7 = 25 of 35 rows. A greedy choice of
  // candidate 0 -> reference 0 would score only 17 rows.
  assert.equal(result.labelAgreement, 25 / 35);
});

test("external HDBSCAN provider boundary validates rows and probabilities", async () => {
  const { ExternalHdbscanProviderAdapter } = await load();
  const provider = new ExternalHdbscanProviderAdapter({
    id: "fixture-external",
    fit(rows, minClusterSize, minSamples) {
      assert.equal(rows.length, 3); assert.equal(minClusterSize, 2); assert.equal(minSamples, 1);
      return { labels: [0, 0, -1], probabilities: [1, 0.5, 0] };
    }
  });
  assert.deepEqual(provider.fit([[1], [2], [3]], 2, 1), { labels: [0, 0, -1], probabilities: [1, 0.5, 0], outlierProxy: [0, 0.5, 1], memberships: undefined });
  assert.throws(() => new ExternalHdbscanProviderAdapter({ id: "bad", fit: () => ({ labels: [0, 0], probabilities: [2, 0] }) }).fit([[1], [2]], 2, 1), /probability/);
});
