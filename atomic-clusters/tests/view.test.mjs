import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

test("cluster explorer uses the full pane for UMAP and exposes per-note hover targets", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  const css = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(view, /atomic-clusters-umap-hit-layer/);
  assert.match(view, /data-point-index/);
  assert.match(view, /mouseenter/);
  assert.match(view, /targetEl: target \|\| this\.visualizationHitElements\[point\]/);
  assert.match(view, /openLinkText\(path, "", false\)/);
  assert.match(view, /atomic-clusters-tree-panel/);
  assert.match(css, /\.atomic-clusters-view \{[^}]*height: 100%/s);
  assert.match(css, /\.atomic-clusters-umap \{[^}]*flex: 1 1 auto/s);
  assert.match(css, /\.atomic-clusters-umap-point-hit \{/);
  assert.match(css, /\.atomic-clusters-umap-hit-layer \{[^}]*pointer-events: none/s);
  assert.match(css, /\.atomic-clusters-umap-point-hit \{[^}]*background: transparent !important/s);
  assert.match(css, /\.atomic-clusters-umap-point-hit \{[^}]*pointer-events: auto/s);
  assert.match(view, /scaleVisualizationPoints\(visualization\.coordinates, width, height\)/);
  assert.doesNotMatch(view, /nodeMass/);
  assert.match(view, /amplitude: visualizationMembershipAmplitude\(row, p95\)/);
  assert.match(view, /type: "range"/);
  assert.match(view, /aria-expanded/);
  assert.match(view, /cachedKey = ""; cachedBitmap = null; draw\(\)/);
  assert.match(view, /visualizationScaledStageSigma\(baseSigma, entry\.remainingDepth, isLeaf, kernelScale\)/);
  assert.match(css, /atomic-clusters-umap-controls/);
  assert.match(css, /@media \(max-width: 640px\)/);
});

test("resize observer ignores its unchanged initial notification during camera animation", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  assert.match(view, /lastObservedPlotSize/);
  assert.match(view, /const changed = nextSize\[0\] !== lastObservedPlotSize\[0\] \|\| nextSize\[1\] !== lastObservedPlotSize\[1\]/);
  assert.match(view, /if \(!changed\) return/);
  assert.match(view, /callback\(animationNow\(\)\)/);
});
