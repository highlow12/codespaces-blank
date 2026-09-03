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

test("plugin exposes per-note settings, context-menu actions, and rename migration", async () => {
  const [main, settings, types] = await Promise.all([
    readFile(new URL("../src/main.ts", import.meta.url), "utf8"),
    readFile(new URL("../src/settings.ts", import.meta.url), "utf8"),
    readFile(new URL("../src/types.ts", import.meta.url), "utf8"),
  ]);
  assert.match(types, /excludedNotes\?: string\[\]/);
  assert.match(main, /excludedFolders: \[\], excludedNotes: \[\]/);
  assert.match(main, /this\.settings\.excludedNotes = normalizeExcludedPaths/);
  assert.match(main, /syncActiveNotes\(notes\)/);
  assert.match(main, /syncActiveNotes\(prepared\.notes\)/);
  assert.match(main, /this\.app\.workspace\.on\("file-menu"/);
  assert.match(main, /Exclude from Atomic Clusters/);
  assert.match(main, /Exclude folder/);
  assert.match(main, /rewriteExcludedNoteRename/);
  assert.match(main, /this\.settings\.excludedNotes = normalizeExcludedPaths\(current\.map/);
  assert.match(settings, /Excluded notes/);
  assert.match(settings, /Restore all/);
  assert.match(settings, /restoreExcludedNote/);
  assert.match(settings, /onExcludedNotesChange/);
  assert.match(settings, /await this\.onExcludedNotesChange\?\.\(\)/);
  assert.match(main, /testLocalRuntime, \(\) => \{ void this\.configureAutomaticRefresh\(\); \}, \(\) => this\.refreshAfterExclusionChange\(\)/);
  assert.doesNotMatch(main, /frontmatter|front-matter|atomic-clusters: false/i);
});
