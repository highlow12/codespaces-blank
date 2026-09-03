import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadMemoryPreflight() {
  const source = await readFile(new URL("../src/memory-preflight.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

function vectors(rows, dimension) {
  return Array.from({ length: rows }, () => Array.from({ length: dimension }, () => 0));
}

test("renderer memory estimate uses the actual vector shape and includes safety margin", async () => {
  const { estimateRendererClusteringMemory } = await loadMemoryPreflight();
  const small = estimateRendererClusteringMemory(100, 64);
  const large = estimateRendererClusteringMemory(1000, 3072);
  assert.equal(small.rowCount, 100);
  assert.equal(small.dimension, 64);
  assert.equal(small.assumptions.workerStructuredCloneIncluded, true);
  assert.equal(small.assumptions.safetyMarginIncluded, true);
  assert.ok(small.bytes.predictedAdditional > small.bytes.orchestrationWorkingSet);
  assert.ok(large.bytes.predictedAdditional > small.bytes.predictedAdditional);
});

test("missing or approximate renderer memory signals warn and never hard block", async () => {
  const { preflightRendererClusteringMemory } = await loadMemoryPreflight();
  const noSignal = preflightRendererClusteringMemory(vectors(100, 64), {});
  assert.equal(noSignal.status, "warning");
  assert.equal(noSignal.canProceed, true);
  assert.equal(noSignal.hardBlock, false);
  assert.match(noSignal.detail, /Memory estimate/);
  assert.match(noSignal.detail, /without a hard block/);

  const approximate = preflightRendererClusteringMemory(vectors(100, 64), { navigator: { deviceMemory: 0.5 } });
  assert.equal(approximate.status, "warning");
  assert.equal(approximate.canProceed, true);
  assert.equal(approximate.signal.source, "navigator.deviceMemory");
  assert.equal(approximate.signal.trustworthy, false);
});

test("trusted JS-heap headroom permits a safe run and blocks only dangerous estimates", async () => {
  const { estimateRendererClusteringMemory, preflightRendererClusteringMemory } = await loadMemoryPreflight();
  const estimate = estimateRendererClusteringMemory(100, 64);
  const safeAvailable = estimate.bytes.predictedAdditional * 5;
  const safe = preflightRendererClusteringMemory(vectors(100, 64), {
    performance: { memory: { jsHeapSizeLimit: safeAvailable + 100, usedJSHeapSize: 100 } }
  });
  assert.equal(safe.status, "pass");
  assert.equal(safe.canProceed, true);
  assert.equal(safe.signal.source, "performance.memory");
  assert.equal(safe.signal.trustworthy, true);

  const dangerousAvailable = Math.max(1, Math.floor(estimate.bytes.predictedAdditional / 0.9));
  const blocked = preflightRendererClusteringMemory(vectors(100, 64), {
    performance: { memory: { jsHeapSizeLimit: dangerousAvailable + 100, usedJSHeapSize: 100 } }
  });
  assert.equal(blocked.status, "blocked");
  assert.equal(blocked.canProceed, false);
  assert.equal(blocked.hardBlock, true);
  assert.match(blocked.error, /Close large panes/);
});

test("invalid embedding shape is observable but remains runnable", async () => {
  const { preflightRendererClusteringMemory } = await loadMemoryPreflight();
  const result = preflightRendererClusteringMemory([[1, 2], [3]], {});
  assert.equal(result.status, "unavailable");
  assert.equal(result.canProceed, true);
  assert.equal(result.hardBlock, false);
  assert.match(result.detail, /shape is unavailable/);
});
