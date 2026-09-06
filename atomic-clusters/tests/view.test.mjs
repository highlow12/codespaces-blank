import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { build } from "esbuild";

class MockClassList {
  constructor() { this.values = new Set(); }
  add(...names) { names.forEach((name) => this.values.add(name)); }
  remove(...names) { names.forEach((name) => this.values.delete(name)); }
  contains(name) { return this.values.has(name); }
  toggle(name, force) { const next = force === undefined ? !this.values.has(name) : force; if (next) this.values.add(name); else this.values.delete(name); return next; }
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
  querySelectorAll(selector) { const matches = []; const visit = (element) => { if ((selector.startsWith(".") && element.classList.contains(selector.slice(1))) || element.tagName.toLowerCase() === selector.toLowerCase()) matches.push(element); for (const child of element.children) visit(child); }; visit(this); return matches; }
  getBoundingClientRect() { return { left: 0, top: 0, width: globalThis.__mockViewport?.width ?? this.clientWidth, height: globalThis.__mockViewport?.height ?? this.clientHeight }; }
  focus(options) { this.focusOptions = options; globalThis.__focusedElement = this; }
  scrollIntoView(options) { this.scrollIntoViewOptions = options; globalThis.__scrolledElement = this; }
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
  assert.match(view, /private openActiveSearchResult\(\): void \{[\s\S]*?if \(path\) \{[\s\S]*?openLinkText\(path, "", false\)/);
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

test("Explorer renders local search, combinable filters, and focus controls", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  const css = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(view, /atomic-clusters-search-input/);
  assert.match(view, /parseSearchQuery|SearchIndex/);
  assert.match(view, /tag|path|cluster/);
  assert.match(view, /Current cluster/);
  assert.match(view, /Manually adjusted/);
  assert.match(view, /Recently changed/);
  assert.match(view, /setFocusNode|focusCluster/);
  assert.match(view, /Previous sibling/);
  assert.match(view, /Exit focus/);
  assert.match(view, /event\.key === "Escape"/);
  assert.match(view, /event\.key === "Enter"/);
  assert.match(view, /ArrowDown/);
  assert.match(view, /event\.key === "\/"/);
  assert.match(view, /event\.ctrlKey/);
  assert.match(view, /searchIndexedResult/);
  assert.match(view, /searchIndexedNotes/);
  assert.match(css, /\.atomic-clusters-search-panel/);
  assert.match(css, /\.atomic-clusters-search-dimmed/);
});

test("Explorer exposes bounded manual correction actions and generated-title provenance", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  const css = await readFile(new URL("../styles.css", import.meta.url), "utf8");
  assert.match(view, /composeEffectiveClusterTitles/);
  assert.match(view, /Rename title/);
  assert.match(view, /Reset title/);
  assert.match(view, /Too broad/);
  assert.match(view, /Create manual group/);
  assert.match(view, /Ungroup/);
  assert.match(view, /preferredClusterCandidates\.slice\(0, 5\)/);
  assert.match(view, /Clear preference/);
  assert.match(view, /Generated:/);
  assert.match(css, /\.atomic-clusters-correction-actions/);
  assert.match(css, /\.atomic-clusters-manual-groups/);
});

test("Explorer focuses a context-menu note and makes its preferred-cluster picker available", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle], ["__focusedElement", globalThis.__focusedElement], ["__scrolledElement", globalThis.__scrolledElement]]);
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MockCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MockCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  try {
    const { ClusterExplorerView } = await loadView();
    const result = { schemaVersion: 4, ids: ["one.md", "two.md"], leafLabels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], probabilities: [1, 1], outlierProxy: [0, 0], titles: { "0": "One cluster", "1": "Two cluster" }, hierarchy: { leaves: [0, 1], merges: [], root: null }, pca: { selected: 2 }, visualization: { coordinates: [[0, 0], [10, 0]], labels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], configuration: {} } };
    const view = new ClusterExplorerView({});
    view.setSearchNotes([{ path: "one.md", title: "One", mtime: 1, hash: "one", content: "one" }, { path: "two.md", title: "Two", mtime: 1, hash: "two", content: "two" }]);
    view.setManualCorrections({ titleOverrides: [], notePreferences: [{ notePath: "two.md", preferredClusterKey: "orphan-preference", createdAt: "2026-09-06T00:00:00.000Z" }], groups: [], feedback: [] });
    assert.equal(view.focusNote("missing.md"), false);
    view.setResult(result);
    assert.equal(view.focusNote("two.md"), true);
    assert.equal(view.contentEl.querySelector(".atomic-clusters-note-detail-path").textContent, "two.md");
    const preference = view.contentEl.querySelector(".atomic-clusters-note-preference");
    const picker = preference.querySelector("select");
    assert.ok(picker);
    assert.equal(picker.children.length, 4, "automatic placement, saved orphan preference, and the two available candidates");
    assert.equal(picker.value, "orphan-preference");
    assert.equal(picker.children.filter((option) => option.value === "orphan-preference").length, 1, "saved preference is represented exactly once");
    const candidateKey = picker.children.find((option) => option.value && option.textContent.includes("automatic")).value;
    view.setManualCorrections({ titleOverrides: [], notePreferences: [{ notePath: "two.md", preferredClusterKey: candidateKey, createdAt: "2026-09-06T00:00:00.000Z" }], groups: [], feedback: [] });
    const candidatePicker = view.contentEl.querySelector(".atomic-clusters-note-preference").querySelector("select");
    assert.equal(candidatePicker.children.length, 3, "a current candidate must not be duplicated");
    assert.equal(candidatePicker.value, candidateKey);
    assert.equal(candidatePicker.children.filter((option) => option.value === candidateKey).length, 1);
    assert.equal(view.focusNote("missing.md"), false);
    assert.equal(view.contentEl.querySelector(".atomic-clusters-note-detail-path").textContent, "two.md");
    await view.onClose();
  } finally {
    for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; }
  }
});

test("manual-group search results reveal the group row for click and Enter without invalid focus", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle], ["__focusedElement", globalThis.__focusedElement], ["__scrolledElement", globalThis.__scrolledElement]]);
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MockCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MockCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  try {
    const { ClusterExplorerView } = await loadView();
    const result = { schemaVersion: 4, ids: ["one.md"], leafLabels: [0], leafOrdering: [0], memberships: [[1]], probabilities: [1], outlierProxy: [0], titles: { "0": "One cluster" }, hierarchy: { leaves: [0], merges: [], root: 0 }, pca: { selected: 2 }, visualization: { coordinates: [[0, 0]], labels: [0], leafOrdering: [0], memberships: [[1]], configuration: {} } };
    const view = new ClusterExplorerView({});
    view.setSearchNotes([{ path: "one.md", title: "One", mtime: 1, hash: "one", content: "one" }]);
    view.setManualCorrections({ titleOverrides: [], notePreferences: [], groups: [{ groupId: "group-1", title: "Research group", childClusterKeys: ["missing-child"], createdAt: "2026-09-06T00:00:00.000Z", updatedAt: "2026-09-06T00:00:00.000Z" }], feedback: [] });
    view.setResult(result);

    let input = view.contentEl.querySelector(".atomic-clusters-search-input");
    input.value = "research group";
    input.dispatchEvent({ type: "input", target: input, bubbles: false });
    await new Promise((resolve) => setTimeout(resolve, 90));
    let button = view.contentEl.querySelector(".atomic-clusters-search-result-cluster");
    const row = view.contentEl.querySelector(".atomic-clusters-manual-group");
    assert.ok(button);
    assert.ok(row);
    button.dispatchEvent({ type: "click", target: button, bubbles: false });
    assert.equal(view.focusNodeId, null, "manual group must not become a visualization node");
    assert.equal(globalThis.__scrolledElement, row);
    assert.equal(globalThis.__focusedElement, row);
    assert.equal(row.classList.contains("is-search-target"), true);

    input = view.contentEl.querySelector(".atomic-clusters-search-input");
    globalThis.__focusedElement = null; globalThis.__scrolledElement = null;
    input.dispatchEvent({ type: "keydown", key: "Enter", target: input, bubbles: false, preventDefault() {}, stopPropagation() {} });
    assert.equal(view.focusNodeId, null, "Enter must use the same manual-group activation path");
    assert.equal(globalThis.__scrolledElement, row);
    assert.equal(globalThis.__focusedElement, row);
    assert.equal(row.classList.contains("is-search-target"), true);
    await view.onClose();
  } finally {
    for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; }
  }
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
  assert.match(main, /resultRevision/);
  assert.match(main, /requestedResult === result && current === result/);
  assert.match(main, /this\.resultRevision !== requestedRevision/);
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

test("gesture raster keeps an overscan margin when the affine layer moves", async () => {
  const view = await readFile(new URL("../src/view.ts", import.meta.url), "utf8");
  assert.match(view, /VISUALIZATION_GESTURE_OVERSCAN = 128/);
  assert.match(view, /const rasterWidth = width \+ gestureOverscan \* 2/);
  assert.match(view, /const rasterPoints = points\.map\(\(\[x, y\]\) => \[x \+ gestureOverscan, y \+ gestureOverscan\]/);
  assert.match(view, /gestureOverscan \* \(1 - ratio\)/);
  assert.match(view, /pickVisualizationCloud\(cachedClouds, x \+ gestureOverscan, y \+ gestureOverscan\)/);
});

test("resize observer defers expensive density work until resize settles", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle], ["ResizeObserver", globalThis.ResizeObserver], ["__mockViewport", globalThis.__mockViewport]]);
  let imageDataCalls = 0; let observerCallback = null;
  class TrackingCanvas extends MockCanvas { getContext() { const context = super.getContext(); context.createImageData = (width, height) => { imageDataCalls++; return { data: new Uint8ClampedArray(width * height * 4) }; }; return context; } }
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = TrackingCanvas; globalThis.document = { createElement: (tag) => tag === "canvas" ? new TrackingCanvas() : new MockElement(tag) }; globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" }); globalThis.__mockViewport = { width: 500, height: 300 }; globalThis.ResizeObserver = class { constructor(callback) { observerCallback = callback; } observe() {} disconnect() {} };
  try {
    const { ClusterExplorerView } = await loadView(); const view = new ClusterExplorerView({});
    view.setResult({ schemaVersion: 4, ids: ["one.md", "two.md"], leafLabels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], probabilities: [1, 1], titles: {}, hierarchy: { leaves: [0, 1], merges: [], root: null }, pca: { selected: 2 }, visualization: { coordinates: [[0, 0], [10, 0]], labels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], configuration: {} } });
    const beforeResize = imageDataCalls; globalThis.__mockViewport = { width: 700, height: 360 }; observerCallback([{ contentRect: { width: 700, height: 360 } }]); assert.equal(imageDataCalls, beforeResize); await new Promise((resolve) => setTimeout(resolve, 125)); assert.ok(imageDataCalls > beforeResize); await view.onClose();
  } finally { for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; } }
});

test("renderer measures labels and recomputes camera padding after resize", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle], ["ResizeObserver", globalThis.ResizeObserver], ["__mockViewport", globalThis.__mockViewport]]);
  let observerCallback = null;
  class MeasuringCanvas extends MockCanvas {
    getContext() {
      const context = super.getContext();
      context.measureText = () => ({ width: 180 });
      return context;
    }
  }
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MeasuringCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MeasuringCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" }); globalThis.__mockViewport = { width: 500, height: 300 };
  globalThis.ResizeObserver = class { constructor(callback) { observerCallback = callback; } observe() {} disconnect() {} };
  try {
    const { ClusterExplorerView } = await loadView(); const view = new ClusterExplorerView({});
    view.setResult({ schemaVersion: 4, ids: ["one.md", "two.md"], leafLabels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], probabilities: [1, 1], titles: { "0": "A long measured title" }, hierarchy: { leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 1, mass: 2 }], root: 2 }, pca: { selected: 2 }, visualization: { coordinates: [[0, 0], [100, 0]], labels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], configuration: {} } });
    assert.ok(view.visualizationCameraState.padding > 18);
    const initialPadding = view.visualizationCameraState.padding;
    globalThis.__mockViewport = { width: 200, height: 160 }; observerCallback([{ contentRect: { width: 200, height: 160 } }]);
    await new Promise((resolve) => setTimeout(resolve, 125));
    assert.equal(view.visualizationCameraState.width, 200); assert.equal(view.visualizationCameraState.height, 160); assert.equal(view.visualizationCameraState.padding, 80);
    globalThis.__mockViewport = { width: 700, height: 420 }; observerCallback([{ contentRect: { width: 700, height: 420 } }]);
    await new Promise((resolve) => setTimeout(resolve, 125));
    assert.equal(view.visualizationCameraState.width, 700); assert.equal(view.visualizationCameraState.height, 420); assert.equal(view.visualizationCameraState.padding, initialPadding);
    await view.onClose();
  } finally { for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; } }
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

test("gesture event handling keeps the camera bounded across 20 rapid drags and zooms", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle]]);
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MockCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MockCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  try {
    const { ClusterExplorerView } = await loadView(); const view = new ClusterExplorerView({});
    view.setResult({ schemaVersion: 4, ids: ["a.md", "b.md", "c.md", "d.md"], leafLabels: [0, 1, 2, 3], leafOrdering: [0, 1, 2, 3], memberships: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], probabilities: [1, 1, 1, 1], titles: {}, hierarchy: { leaves: [0, 1, 2, 3], merges: [], root: null }, pca: { selected: 2 }, visualization: { coordinates: [[-100, -50], [100, -50], [-100, 50], [100, 50]], labels: [0, 1, 2, 3], leafOrdering: [0, 1, 2, 3], memberships: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], configuration: {} } });
    const layer = view.contentEl.querySelector(".atomic-clusters-umap-visual-layer");
    const assertStateIsFinite = () => { const state = view.visualizationCameraState; assert.ok(state); assert.ok(Number.isFinite(state.centerX) && Number.isFinite(state.centerY)); assert.ok(state.zoom >= .5 && state.zoom <= 16); assert.ok(state.width > 0 && state.height > 0); assert.ok(state.padding >= 0 && state.padding <= Math.min(state.width, state.height) / 2); };
    const zoomDeltas = [-700, 420, -1000, 650, -320]; const drags = [[1000, 700], [-1200, -800], [900, -650], [-1100, 550]];
    for (let index = 0; index < 20; index++) {
      const startX = 250 + (index % 3) * 3; const startY = 150 + (index % 2) * 3; const [deltaX, deltaY] = drags[index % drags.length];
      layer.dispatchEvent({ type: "pointerdown", button: 0, clientX: startX, clientY: startY, pointerId: 1, bubbles: true });
      layer.dispatchEvent({ type: "pointermove", button: 0, clientX: startX + deltaX, clientY: startY + deltaY, pointerId: 1, bubbles: true });
      layer.dispatchEvent({ type: "pointerup", button: 0, clientX: startX + deltaX, clientY: startY + deltaY, pointerId: 1, bubbles: true });
      assertStateIsFinite();
      layer.dispatchEvent({ type: "wheel", deltaY: zoomDeltas[index % zoomDeltas.length], clientX: 30 + (index * 97) % 440, clientY: 20 + (index * 61) % 260, preventDefault: () => {}, bubbles: true });
      assertStateIsFinite();
    }
    await view.onClose();
  } finally { for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; } }
});

test("hover clears immediately on empty space, pointerleave, pointercancel, and drag", async () => {
  const previous = new Map([["MockElement", globalThis.MockElement], ["HTMLElement", globalThis.HTMLElement], ["HTMLCanvasElement", globalThis.HTMLCanvasElement], ["document", globalThis.document], ["window", globalThis.window], ["getComputedStyle", globalThis.getComputedStyle]]);
  globalThis.MockElement = MockElement; globalThis.HTMLElement = MockElement; globalThis.HTMLCanvasElement = MockCanvas;
  globalThis.document = { createElement: (tag) => tag === "canvas" ? new MockCanvas() : new MockElement(tag) };
  globalThis.window = { devicePixelRatio: 1, matchMedia: () => ({ matches: false }) }; globalThis.getComputedStyle = () => ({ color: "#fff", getPropertyValue: () => "transparent" });
  try {
    const { ClusterExplorerView } = await loadView();
    const view = new ClusterExplorerView({}); let hoverEvents = 0;
    view.app = { workspace: { trigger: () => { hoverEvents++; } } };
    view.setResult({ schemaVersion: 4, ids: ["one.md", "two.md"], leafLabels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], probabilities: [1, 1], titles: {}, hierarchy: { leaves: [0, 1], merges: [], root: null }, pca: { selected: 2 }, visualization: { coordinates: [[0, 0], [10, 0]], labels: [0, 1], leafOrdering: [0, 1], memberships: [[1, 0], [0, 1]], configuration: {} } });
    const layer = view.contentEl.querySelector(".atomic-clusters-umap-visual-layer"); const point = view.visualizationPoints[0];
    const dispatchAtPoint = (type, x = point[0], y = point[1]) => layer.dispatchEvent({ type, clientX: x, clientY: y, pointerId: 1, bubbles: true });
    dispatchAtPoint("pointermove");
    assert.equal(view.hoveredVisualizationPoint, 0);
    assert.equal(hoverEvents, 1);
    dispatchAtPoint("pointermove", 5, 5);
    assert.equal(view.hoveredVisualizationPoint, null);
    dispatchAtPoint("pointermove"); dispatchAtPoint("pointerleave");
    assert.equal(view.hoveredVisualizationPoint, null);
    dispatchAtPoint("pointermove"); dispatchAtPoint("pointercancel");
    assert.equal(view.hoveredVisualizationPoint, null);
    dispatchAtPoint("pointermove");
    layer.dispatchEvent({ type: "pointerdown", button: 0, clientX: point[0], clientY: point[1], pointerId: 1, bubbles: true });
    const hit = view.visualizationHitElements[0];
    hit.dispatchEvent({ type: "mouseenter", clientX: point[0], clientY: point[1], pointerId: 1, bubbles: false });
    assert.equal(view.hoveredVisualizationPoint, null);
    layer.dispatchEvent({ type: "pointermove", button: 0, clientX: point[0] + 40, clientY: point[1] + 30, pointerId: 1, bubbles: true });
    assert.equal(view.hoveredVisualizationPoint, null);
    assert.match(layer.style.transform, /translate/);
    layer.dispatchEvent({ type: "pointercancel", pointerId: 1, bubbles: true });
    assert.equal(view.hoveredVisualizationPoint, null);
    await view.onClose();
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
    assert.equal(view.visualizationCameraState.centerX, 5);
    assert.ok(view.visualizationCameraState.fitScale > 20);
  } finally { for (const [name, value] of previous) { if (value === undefined) delete globalThis[name]; else globalThis[name] = value; } }
});
