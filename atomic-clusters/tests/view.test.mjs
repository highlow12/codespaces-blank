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
});
