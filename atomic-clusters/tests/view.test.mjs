import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { build } from "esbuild";

class MockClassList {
  constructor() { this.values = new Set(); }
  add(...names) { names.forEach((name) => this.values.add(name)); }
  remove(...names) { names.forEach((name) => this.values.delete(name)); }
  contains(name) { return this.values.has(name); }
}

class MockElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase(); this.children = []; this.parentElement = null;
    this.classList = new MockClassList(); this.dataset = {}; this.listeners = new Map();
    this.style = { transform: "", transformOrigin: "", opacity: "", setProperty: (name, value) => { this.style[name] = value; } };
    this.clientWidth = globalThis.__mockViewport?.width ?? 500; this.clientHeight = globalThis.__mockViewport?.height ?? 300; this.disabled = false; this.tabIndex = -1;
  }
  createEl(tag, options = {}) { const element = tag.toLowerCase() === "canvas" ? new globalThis.HTMLCanvasElement() : new MockElement(tag); if (options.cls) element.classList.add(...options.cls.split(" ")); if (options.text) element.textContent = options.text; for (const [name, value] of Object.entries(options.attr || {})) { element.setAttribute(name, value); } this.appendChild(element); return element; }
  createDiv(options = {}) { return this.createEl("div", options); }
  createSpan(options = {}) { return this.createEl("span", options); }
  appendChild(element) { element.parentElement = this; this.children.push(element); return element; }
  removeChild(element) { this.children = this.children.filter((child) => child !== element); element.parentElement = null; return element; }
  get firstChild() { return this.children[0] || null; }
  empty() { this.children = []; }
  addClass(name) { this.classList.add(name); return this; }
  toggleClass(name, force) { if (force === undefined ? !this.classList.contains(name) : force) this.classList.add(name); else this.classList.remove(name); return this; }
  setAttribute(name, value) { if (name === "class") this.classList.add(...String(value).split(" ")); else if (name.startsWith("data-")) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value); else this[name] = String(value); }
  addEventListener(type, callback) { const callbacks = this.listeners.get(type) || []; callbacks.push(callback); this.listeners.set(type, callbacks); }
  removeEventListener(type, callback) { this.listeners.set(type, (this.listeners.get(type) || []).filter((item) => item !== callback)); }
  dispatchEvent(event) { if (!event.target) event.target = this; for (const callback of this.listeners.get(event.type) || []) callback(event); if (event.bubbles !== false && this.parentElement) this.parentElement.dispatchEvent(event); return true; }
  querySelector(selector) { if ((selector.startsWith(".") && this.classList.contains(selector.slice(1))) || this.tagName.toLowerCase() === selector.toLowerCase()) return this; for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; } return null; }
  getBoundingClientRect() { return { left: 0, top: 0, width: globalThis.__mockViewport?.width ?? this.clientWidth, height: globalThis.__mockViewport?.height ?? this.clientHeight }; }
  focus() {}
  setText(text) { this.textContent = text; }
}

class MockCanvas extends MockElement {
  constructor() { super("canvas"); this.width = 500; this.height = 300; }
  getContext() { return { setTransform() {}, clearRect() {}, beginPath() {}, arc() {}, fill() {}, stroke() {}, drawImage() {}, putImageData() {}, createImageData(width, height) { return { data: new Uint8ClampedArray(width * height * 4) }; }, imageSmoothingEnabled: true, globalAlpha: 1 }; }
}

async function loadView() {
  const result = await build({ entryPoints: [new URL("../src/view.ts", import.meta.url).pathname], bundle: true, format: "esm", platform: "browser", write: false, plugins: [{ name: "obsidian-test-stub", setup(plugin) { plugin.onResolve({ filter: /^obsidian$/ }, () => ({ path: "obsidian-test-stub", namespace: "test" })); plugin.onLoad({ filter: /.*/, namespace: "test" }, () => ({ contents: "export class ItemView { constructor() { this.contentEl = new globalThis.MockElement('div'); this.leaf = {}; } }", loader: "js" })); } }] });
  return import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString("base64")}`);
}

test("cluster explorer uses the full pane for UMAP and exposes per-note hover targets", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  const css = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(view, /atomic-clusters-umap-hit-layer/);
  assert.match(view, /data-point-index/);
  assert.match(view, /mouseenter/);
  assert.match(view, /const pointButton = this\.visualizationHitElements\.find\(\(hit\) => Number\(hit\.dataset\.pointIndex\) === point\)/);
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
  assert.match(view, /visualizationFrontier\(this\.visualizationRoot, \[current\.id\]/);
  assert.match(view, /buildVisualizationPointSpatialIndex\(indexedPoints, pointHitRadius \* 2\)/);
  assert.match(view, /queryNearest\(event\.clientX - rect\.left, event\.clientY - rect\.top, pointHitRadius\)/);
  assert.match(view, /const maxPointHitTargets = 96/);
  assert.match(view, /slice\(0, maxPointHitTargets\)/);
  assert.match(view, /this\.visualizationRenderedPointIndices/);
  assert.match(view, /const clusterColor = visualizationCloudColor\(entry\.node, palette\)/);
  assert.doesNotMatch(view, /visualizationExpandedIds/);
  assert.match(view, /this\.navigateVisualization\(target\.node\.id, \(\) => undefined\)/);
  assert.match(view, /const parent = path\[path\.length - 2\]/);
  assert.match(view, /const chunkSize = 80/);
  assert.match(view, /viewport\.addEventListener\("scroll", onScroll\)/);
  assert.doesNotMatch(view, /\.slice\(0, 3\)/);
  assert.match(view, /atomic-clusters-hover-membership-summary/);
  assert.match(view, /visualizationTopMemberships\(result, point, 3\)/);
  assert.match(view, /hoverPopover/);
  assert.match(css, /\.atomic-clusters-note-list \{[^}]*overflow-y: auto/s);
  assert.match(css, /\.atomic-clusters-hover-membership-summary \{/);
  assert.match(css, /atomic-clusters-umap-controls/);
  assert.match(css, /@media \(max-width: 640px\)/);
});

test("restored explorer views can render the persisted result on first open", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  const main = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
  assert.match(view, /initialResult\?: ClusterResult \| null/);
  assert.match(view, /this\.result = initialResult \|\| null/);
  assert.match(main, /this\.latestResult = await sqlite\.getResult\(\)/);
  assert.match(main, /new ClusterExplorerView\(leaf, \(\) => this\.buildClusters\(\), this\.latestResult, /);
  assert.match(main, /deferVisualization: true/);
  assert.match(main, /getPcaCoordinatesMany\(result\.ids, modelHash\)/);
  assert.match(main, /patchResultVisualization\(resultId, visualization\)/);
});

test("v6 results without UMAP schedule one idle visualization preparation and apply a returned visualization", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle], ["requestIdleCallback", globalThis.requestIdleCallback], ["cancelIdleCallback", globalThis.cancelIdleCallback]]);
  const idleCallbacks = []; let nextHandle = 1;
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MockCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MockCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  globalThis.requestIdleCallback = (callback) => { idleCallbacks.push(callback); return nextHandle++; }; globalThis.cancelIdleCallback = () => {};
  try {
    const { ClusterExplorerView } = await loadView(); let calls = 0;
    const result = { schemaVersion: 6, ids: ["one.md", "two.md"], leafLabels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], probabilities: [1, 1], outlierProxy: [0, 0], titles: {}, hierarchy: { leaves: [0, 1], merges: [], root: null, nodes: [], rootChildren: [] }, pca: { selected: 2 }, timings: {} };
    const visualization = { coordinates: [[0, 0], [10, 0]], labels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], configuration: {} };
    const view = new ClusterExplorerView({}, undefined, result, async (input) => { calls++; assert.equal(input, result); return visualization; });
    view.setResult(result);
    assert.equal(calls, 0); assert.equal(idleCallbacks.length, 1); assert.match(view.contentEl.children.at(-1).textContent, /Preparing visualization/);
    idleCallbacks.shift()({ timeRemaining: () => 50 }); await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
    assert.equal(calls, 1); assert.ok(view.contentEl.querySelector(".atomic-clusters-umap"));
  } finally {
    for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; }
  }
});

test("lazy visualization completion is ignored after result replacement and close", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle], ["requestIdleCallback", globalThis.requestIdleCallback], ["cancelIdleCallback", globalThis.cancelIdleCallback]]);
  const idleCallbacks = []; let resolveVisualization;
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MockCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MockCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  globalThis.requestIdleCallback = (callback) => { idleCallbacks.push(callback); return idleCallbacks.length; }; globalThis.cancelIdleCallback = () => {};
  try {
    const { ClusterExplorerView } = await loadView();
    const makeResult = (path) => ({ schemaVersion: 6, ids: [path], leafLabels: [-1], leafOrdering: [], memberships: [[]], probabilities: [0], outlierProxy: [1], titles: {}, hierarchy: { leaves: [], merges: [], root: null, nodes: [], rootChildren: [] }, pca: { selected: 0 }, timings: {} });
    const first = makeResult("first.md"); const replacement = makeResult("replacement.md"); const visualization = { coordinates: [[0, 0]], labels: [-1], leafOrdering: [], memberships: [[]], configuration: {} };
    const view = new ClusterExplorerView({}, undefined, first, async () => new Promise((resolve) => { resolveVisualization = resolve; }));
    view.setResult(first); idleCallbacks.shift()({ timeRemaining: () => 50 }); await Promise.resolve(); view.setResult(replacement); resolveVisualization(visualization); await Promise.resolve(); await Promise.resolve();
    assert.equal(view.contentEl.querySelector(".atomic-clusters-umap"), null); assert.match(view.contentEl.children.at(-1).textContent, /Preparing visualization/);
    idleCallbacks.shift()({ timeRemaining: () => 50 }); await Promise.resolve(); view.onClose(); resolveVisualization(visualization); await Promise.resolve(); await Promise.resolve(); assert.equal(view.contentEl.children.length, 0);
  } finally {
    for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; }
  }
});

test("plugin adds a ribbon button that opens the cluster explorer", async () => {
  const main = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
  assert.match(main, /this\.addRibbonIcon\("scatter-chart", "Open Cluster Explorer", \(\) => void this\.openExplorer\(\)\)/);
});

test("resize observer ignores unchanged notifications and waits for a valid content box", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  assert.match(view, /lastObservedPlotSize/);
  assert.match(view, /const changed = !equivalentViewport\(normalizedNextSize, lastObservedPlotSize\)/);
  assert.match(view, /VISUALIZATION_VIEWPORT_TOLERANCE/);
  assert.match(view, /normalizeViewport/);
  assert.match(view, /if \(!nextSize\) return/);
  assert.match(view, /draw\(nextSize\)/);
  assert.match(view, /callback\(animationNow\(\)\)/);
});

test("gesture camera pans without opening a note, debounces density, and cleans up listeners", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle]]);
  let imageDataCalls = 0; let prevented = false;
  class TrackingCanvas extends MockCanvas {
    getContext() { const context = super.getContext(); context.createImageData = (width, height) => { imageDataCalls++; return { data: new Uint8ClampedArray(width * height * 4) }; }; return context; }
  }
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = TrackingCanvas; globalThis.document = { createElement: (tag) => tag === "canvas" ? new TrackingCanvas() : new MockElement(tag) }; globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  try {
    const { ClusterExplorerView } = await loadView(); const view = new ClusterExplorerView({});
    view.setResult({ schemaVersion: 4, ids: ["one.md", "two.md"], leafLabels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], probabilities: [1, 1], titles: {}, hierarchy: { leaves: [0, 1], merges: [], root: null }, pca: { selected: 2 }, visualization: { coordinates: [[0, 0], [10, 0]], labels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], configuration: {} } });
    const layer = view.contentEl.querySelector(".atomic-clusters-umap-visual-layer"); const canvas = layer.querySelector("canvas"); const initialCalls = imageDataCalls;
    layer.dispatchEvent({ type: "pointerdown", button: 0, clientX: 100, clientY: 100, pointerId: 1, bubbles: true }); layer.dispatchEvent({ type: "pointermove", button: 0, clientX: 102, clientY: 102, pointerId: 1, bubbles: true }); assert.equal(layer.style.transform, ""); layer.dispatchEvent({ type: "pointermove", button: 0, clientX: 120, clientY: 130, pointerId: 1, bubbles: true }); assert.match(layer.style.transform, /translate/); layer.dispatchEvent({ type: "pointerup", button: 0, clientX: 120, clientY: 130, pointerId: 1, bubbles: true }); assert.equal(imageDataCalls, initialCalls);
    await new Promise((resolve) => setTimeout(resolve, 125)); assert.ok(imageDataCalls > initialCalls);
    const afterDrag = imageDataCalls; layer.dispatchEvent({ type: "wheel", deltaY: -120, clientX: 250, clientY: 150, preventDefault: () => { prevented = true; }, bubbles: true }); assert.equal(prevented, true); assert.equal(imageDataCalls, afterDrag); await new Promise((resolve) => setTimeout(resolve, 125)); assert.ok(imageDataCalls > afterDrag);
    const afterWheel = imageDataCalls; await view.onClose(); await new Promise((resolve) => setTimeout(resolve, 125)); assert.equal(imageDataCalls, afterWheel);
  } finally { for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; } }
});

test("clicking a rendered cluster cloud advances the global depth", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle]]);
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MockCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MockCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  try {
    const { ClusterExplorerView } = await loadView(); const view = new ClusterExplorerView({});
    view.setResult({ schemaVersion: 4, ids: ["a.md", "b.md", "c.md"], leafLabels: [0, 1, 2], leafOrdering: [0, 1, 2], memberships: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], probabilities: [1, 1, 1], titles: {}, hierarchy: { leaves: [0, 1, 2], merges: [{ id: 3, left: 0, right: 1, distance: 1, mass: 2 }], root: 9, children: { "9": [3, 2] } }, pca: { selected: 2 }, visualization: { coordinates: [[0, 0], [10, 0], [100, 0]], labels: [0, 1, 2], leafOrdering: [0, 1, 2], memberships: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], configuration: {} } });
    const layer = view.contentEl.querySelector(".atomic-clusters-umap-visual-layer");
    // A click following pointer capture may be retargeted to the gesture
    // layer instead of its canvas child. It must still pick the cloud.
    layer.dispatchEvent({ type: "click", clientX: 80, clientY: 150, bubbles: true });
    const breadcrumb = view.contentEl.querySelector(".atomic-clusters-breadcrumb");
    assert.equal(breadcrumb.children.at(-1).textContent, "Cluster 3");
  } finally { for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; } }
});
