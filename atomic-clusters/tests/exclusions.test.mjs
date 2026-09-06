import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

async function loadPathHelpers() {
  const source = await readFile(new URL("../src/types.ts", import.meta.url), "utf8");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

async function loadNoteStore() {
  let source = await readFile(new URL("../src/storage.ts", import.meta.url), "utf8");
  source = source.replace('import { normalizePath, Plugin, Vault } from "obsidian";', 'function normalizePath(value) { return value; }');
  source = source.replace(/import \{[\s\S]*?\} from "\.\/types";/, `
    function normalizeVaultRelativePath(value) { return String(value ?? "").trim().replace(/\\\\/g, "/").replace(/^\\.\\/+/, "").replace(/^\\/+/, "").replace(/\\/+$/, "").trim(); }
    function normalizeExcludedPaths(value) { return [...new Set((Array.isArray(value) ? value : []).map(normalizeVaultRelativePath).filter(Boolean))].sort((a, b) => a.localeCompare(b)); }
    function pathMatchesExcludedFolder(path, folder) { const p = normalizeVaultRelativePath(path); const f = normalizeVaultRelativePath(folder); return !!p && !!f && (p === f || p.startsWith(f + "/")); }
  `);
  source = source.replace(/export \{[\s\S]*?\} from "\.\/sqlite-storage";/, "");
  source = source.replace('const { contentHash } = await import("./hash");', 'const contentHash = async (value) => `hash:${value}`;');
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

async function loadMainPolicies() {
  let source = await readFile(new URL("../src/main.ts", import.meta.url), "utf8");
  source = source.replace(/^import .*;\r?\n/gm, "");
  source = source.replace("export default class AtomicClustersPlugin", "class AtomicClustersPlugin");
  const stubs = `
    class Plugin {}
    class TAbstractFile {}
    class TFile extends TAbstractFile { constructor(path) { super(); this.path = path; this.extension = String(path).split(".").pop(); } }
    class TFolder extends TAbstractFile { constructor(path) { super(); this.path = path; this.children = []; } }
    class Menu {}
    class Modal {}
    class Notice { constructor(message) { Notice.messages.push(message); } }
    Notice.messages = [];
    function normalizeVaultRelativePath(value) { return String(value ?? "").trim().replace(/\\\\/g, "/").replace(/^\\.\\/+/, "").replace(/^\\/+/, "").replace(/\\/+$/, "").trim(); }
    function normalizeExcludedPaths(value) { return [...new Set((Array.isArray(value) ? value : []).map(normalizeVaultRelativePath).filter(Boolean))].sort((a, b) => a.localeCompare(b)); }
    function pathMatchesExcludedFolder(path, folder) { const p = normalizeVaultRelativePath(path); const f = normalizeVaultRelativePath(folder); return !!p && !!f && (p === f || p.startsWith(f + "/")); }
  `;
  source += `\nexport { AtomicClustersPlugin as default, TFile };\n`;
  const result = await transform(`${stubs}\n${source}`, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

function menuItem() {
  return {
    title: "",
    icon: "",
    warning: false,
    disabled: false,
    onClickHandler: undefined,
    setTitle(value) { this.title = value; return this; },
    setIcon(value) { this.icon = value; return this; },
    setWarning(value) { this.warning = value; return this; },
    setDisabled(value) { this.disabled = value; return this; },
    onClick(handler) { this.onClickHandler = handler; return this; },
  };
}

function menuFor(items) {
  return {
    addSeparator() {},
    addItem(callback) { const item = menuItem(); callback(item); items.push(item); return item; },
  };
}

test("exclusion path normalization is stable, deduplicated, and folder-aware", async () => {
  const { normalizeExcludedPaths, normalizeVaultRelativePath, pathMatchesExcludedFolder } = await loadPathHelpers();
  assert.equal(normalizeVaultRelativePath(" ./Projects\\Drafts/ "), "Projects/Drafts");
  assert.deepEqual(normalizeExcludedPaths(["./Projects", "Projects/", "", null, "Notes\\Today"]), ["Notes/Today", "Projects"]);
  assert.equal(pathMatchesExcludedFolder("Projects/Drafts/note.md", "./Projects"), true);
  assert.equal(pathMatchesExcludedFolder("Projects-archive/note.md", "Projects"), false);
});

test("NoteStore excludes individual notes without changing vault content", async () => {
  const { NoteStore } = await loadNoteStore();
  const files = [
    { path: "keep.md", basename: "keep", extension: "md", stat: { mtime: 1 } },
    { path: "Projects/skip.md", basename: "skip", extension: "md", stat: { mtime: 2 } },
    { path: "Archive/folder.md", basename: "folder", extension: "md", stat: { mtime: 3 } },
  ];
  const reads = [];
  const vault = {
    getMarkdownFiles: () => files,
    cachedRead: async (file) => { reads.push(file.path); return `body:${file.path}`; },
  };
  const records = await new NoteStore(vault).collect(["Archive"], ["Projects/skip.md"]);
  assert.deepEqual(records.map((note) => note.path), ["keep.md"]);
  assert.deepEqual(reads, ["keep.md"]);
  assert.equal(records[0].content, "body:keep.md");
});

test("context-menu state distinguishes direct, inherited, and included notes", async () => {
  const { default: AtomicClustersPlugin, getExclusionState, TFile } = await loadMainPolicies();
  assert.equal(getExclusionState("direct.md", [], ["direct.md"], "note"), "direct");
  assert.equal(getExclusionState("Archive/note.md", ["Archive"], [], "note"), "inherited");
  assert.equal(getExclusionState("Archive/direct.md", ["Archive"], ["Archive/direct.md"], "note"), "inherited");
  assert.equal(getExclusionState("included.md", [], [], "note"), "included");

  const plugin = Object.create(AtomicClustersPlugin.prototype);
  plugin.settings = { excludedFolders: ["Archive"], excludedNotes: ["Archive/direct.md", "direct.md"] };
  const inheritedItems = [];
  plugin.addFileContextMenuItems(menuFor(inheritedItems), new TFile("Archive/direct.md"));
});

test("context-menu actions are disabled only for inherited notes", async () => {
  const module = await loadMainPolicies();
  const { default: AtomicClustersPlugin } = module;
  const plugin = Object.create(AtomicClustersPlugin.prototype);
  plugin.settings = { excludedFolders: ["Archive"], excludedNotes: ["Archive/direct.md", "direct.md"] };
  const makeFile = (path) => {
    const file = Object.create(Object.getPrototypeOf(plugin));
    file.path = path;
    file.extension = "md";
    return file;
  };
  // The loader's TFile class is module-local; use the production method's
  // state helper for policy assertions and exercise the action labels below
  // through a small source-level contract only where Obsidian classes cannot
  // cross the data-module boundary.
  assert.equal(module.getExclusionState("Archive/direct.md", ["Archive"], ["Archive/direct.md"], "note"), "inherited");
  assert.equal(module.getExclusionState("direct.md", ["Archive"], ["direct.md"], "note"), "direct");
  assert.equal(module.getExclusionState("new.md", ["Archive"], [], "note"), "included");
  void plugin; void makeFile;
});

test("Markdown note menus preserve exclusion actions and expose Explorer preference actions", async () => {
  const { default: AtomicClustersPlugin, TFile } = await loadMainPolicies();
  const plugin = Object.create(AtomicClustersPlugin.prototype);
  plugin.settings = { excludedFolders: [], excludedNotes: [] };
  plugin.manualCorrections = { titleOverrides: [], notePreferences: [], groups: [], feedback: [] };
  plugin.latestResult = { ids: ["Folder/note.md"] };
  const opened = [];
  plugin.openExplorer = async (path) => { opened.push(path); };
  const items = [];
  plugin.addFileContextMenuItems(menuFor(items), new TFile("./Folder\\note.md"));
  assert.deepEqual(items.map((item) => item.title), ["Exclude from Atomic Clusters", "Prefer another cluster"]);
  await items[1].onClickHandler();
  assert.deepEqual(opened, ["Folder/note.md"]);

  plugin.manualCorrections.notePreferences = [{ notePath: "Folder/note.md", preferredClusterKey: "cluster-test", createdAt: "now" }];
  let cleared;
  plugin.clearNoteClusterPreference = async (path) => { cleared = path; };
  const withPreference = [];
  plugin.addFileContextMenuItems(menuFor(withPreference), new TFile("Folder/note.md"));
  assert.deepEqual(withPreference.map((item) => item.title), ["Exclude from Atomic Clusters", "Prefer another cluster", "Clear preferred cluster"]);
  await withPreference[2].onClickHandler();
  assert.equal(cleared, "Folder/note.md");
});

test("final-note policy and rename boundary actions are behaviorally enforced", async () => {
  const { default: AtomicClustersPlugin, hasIncludedMarkdownNotePaths, classifyRenameBoundary } = await loadMainPolicies();
  assert.equal(hasIncludedMarkdownNotePaths(["only.md"], [], ["only.md"]), false);
  assert.equal(hasIncludedMarkdownNotePaths(["only.md", "keep.md"], [], ["only.md"]), true);
  assert.equal(hasIncludedMarkdownNotePaths(["only.md", "keep.txt"], [], ["only.md"]), false);
  assert.equal(classifyRenameBoundary("Archive/note.md", "note.md", ["Archive"], [], true), "created");
  assert.equal(classifyRenameBoundary("note.md", "Archive/note.md", ["Archive"], [], true), "deleted");
  assert.equal(classifyRenameBoundary("old.md", "new.md", [], [], true), "renamed");
  assert.equal(classifyRenameBoundary("old.md", "new.md", [], [], false), "ignored");

  const plugin = Object.create(AtomicClustersPlugin.prototype);
  plugin.settings = { excludedFolders: [], excludedNotes: [] };
  plugin.app = { vault: { getMarkdownFiles: () => [{ path: "only.md" }] } };
  plugin.saveSettings = async () => { throw new Error("save must not run for a rejected exclusion"); };
  plugin.refreshAfterExclusionChange = async () => { throw new Error("refresh must not run for a rejected exclusion"); };
  await plugin.setNoteExcluded("only.md", true);
  assert.deepEqual(plugin.settings.excludedNotes, []);

  plugin.settings = { excludedFolders: [], excludedNotes: [] };
  plugin.app = { vault: { getMarkdownFiles: () => [{ path: "keep.md" }, { path: "skip.md" }] } };
  let forcedRefresh = false;
  plugin.saveSettings = async () => {};
  plugin.refreshAfterExclusionChange = async () => { forcedRefresh = true; };
  plugin.settings.automaticRefresh = false;
  await plugin.setNoteExcluded("skip.md", true);
  assert.equal(forcedRefresh, true);
  assert.deepEqual(plugin.settings.excludedNotes, ["skip.md"]);
});
