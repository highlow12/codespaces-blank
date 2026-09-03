import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadHeartbeat() {
  const source = await readFile(new URL("../src/progress-heartbeat.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

test("heartbeat detail includes elapsed time and becomes explicit after a 30-second silence", async () => {
  const { PROGRESS_HEARTBEAT_INTERVAL_MS, PROGRESS_HEARTBEAT_STALL_MS, buildProgressHeartbeatDetail, formatElapsedMs } = await loadHeartbeat();
  assert.equal(PROGRESS_HEARTBEAT_INTERVAL_MS, 10_000);
  assert.equal(PROGRESS_HEARTBEAT_STALL_MS, 30_000);
  assert.equal(formatElapsedMs(61_000), "1m 01s");
  const active = buildProgressHeartbeatDetail("Clustering 5%", 20_000, 20_000);
  assert.match(active, /20s elapsed/);
  assert.doesNotMatch(active, /Still working/);
  const stalled = buildProgressHeartbeatDetail("Clustering 5%", 30_000, 30_000);
  assert.match(stalled, /30s elapsed/);
  assert.match(stalled, /Still working/);
});

test("progress Notice owns a heartbeat interval and clears operation timers", async () => {
  const progress = await readFile(new URL("../src/progress.ts", import.meta.url), "utf8");
  assert.match(progress, /setInterval\(\(\) => this\.updateHeartbeat\(\)/);
  assert.match(progress, /stopHeartbeat\(\)/);
  assert.match(progress, /clearInterval/);
  assert.match(progress, /clearHideTimer/);
  assert.match(progress, /window\.clearTimeout/);
  assert.match(progress, /buildProgressHeartbeatDetail/);
});
