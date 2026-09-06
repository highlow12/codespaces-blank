import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import initSqlJs from "sql.js";
import { transform } from "esbuild";

async function loadStorage() {
  let source = await readFile(new URL("../src/sqlite-storage.ts", import.meta.url), "utf8");
  source = source.replace('import { contentHash } from "./hash";', `
    async function contentHash(value) { let h = 2166136261; for (let i = 0; i < value.length; i++) { h ^= value.charCodeAt(i); h = Math.imul(h, 16777619); } return "fnv1a-" + (h >>> 0).toString(16); }
  `);
  source = source.replace(/import \{[\s\S]*?\} from "\.\/types";/, "");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}

class MemoryAdapter {
  files = new Map();
  async exists(path) { return this.files.has(path); }
  async readBinary(path) { return this.files.get(path); }
  async writeBinary(path, data) { this.files.set(path, data); }
  async mkdir() {}
  async rename(from, to) { this.files.set(to, this.files.get(from)); this.files.delete(from); }
  async remove(path) { this.files.delete(path); }
}

function note(path, index = 0) {
  return { path, title: path, mtime: index, content: `content ${path}`, hash: `hash-${path}` };
}

function oneClusterResult(paths, title = "Generated") {
  return {
    schemaVersion: 6,
    ids: paths,
    leafLabels: paths.map(() => 0),
    probabilities: paths.map(() => 1),
    outlierProxy: paths.map(() => 0),
    memberships: paths.map(() => [1]),
    leafOrder: [0],
    pca: { selected: 1 },
    hierarchy: {
      leaves: [0], merges: [], root: 0,
      nodes: [{ id: 0, children: [], descendantLeaves: [0], distance: 0, mass: paths.length }],
      rootChildren: [0], splitMethod: "distance-knee-2-5"
    },
    hierarchyPlacements: paths.map(() => ({ kind: "leaf", nodeId: 0, confidence: 1 })),
    titles: { "0": title },
    timings: {}
  };
}

function twoClusterResult(left, right, titles = {}) {
  const ids = [...left, ...right];
  const labels = [...left.map(() => 0), ...right.map(() => 1)];
  const memberships = labels.map((label) => label === 0 ? [1, 0] : [0, 1]);
  return {
    schemaVersion: 6,
    ids,
    leafLabels: labels,
    probabilities: labels.map(() => 1),
    outlierProxy: labels.map(() => 0),
    memberships,
    leafOrder: [0, 1],
    pca: { selected: 1 },
    hierarchy: {
      leaves: [0, 1],
      merges: [{ id: 2, left: 0, right: 1, distance: 1, mass: ids.length }],
      root: 2,
      nodes: [
        { id: 0, children: [], descendantLeaves: [0], distance: 0, mass: left.length },
        { id: 1, children: [], descendantLeaves: [1], distance: 0, mass: right.length },
        { id: 2, children: [0, 1], descendantLeaves: [0, 1], distance: 1, mass: ids.length }
      ],
      rootChildren: [0, 1], splitMethod: "distance-knee-2-5"
    },
    hierarchyPlacements: labels.map((label) => ({ kind: "leaf", nodeId: label, confidence: 1 })),
    titles: { "0": titles["0"] || "Left", "1": titles["1"] || "Right", "2": titles["2"] || "Root" },
    timings: {}
  };
}

async function openStore(Storage, adapter = new MemoryAdapter(), options = {}) {
  const SQL = await initSqlJs();
  const store = await new Storage.SqliteClusterStore(adapter, SQL, { now: () => "2026-09-04T00:00:00.000Z", ...options }).open();
  return { store, adapter, SQL };
}

async function seed(store, paths) {
  await store.upsertNotes(paths.map((path, index) => note(path, index)));
}

test("cluster fingerprints normalize, sort, and deduplicate member paths", async () => {
  const Storage = await loadStorage();
  assert.deepEqual(Storage.normalizeClusterMemberPaths([" ./b.md", "a\\x.md", "a/x.md", "./b.md"]), ["a/x.md", "b.md"]);
  const first = Storage.stableClusterFingerprint(["b.md", "a.md", "a.md"]);
  assert.equal(first, Storage.clusterFingerprint(["./a.md", "b.md"]));
  assert.equal(first, Storage.computeClusterFingerprint(["b.md", "a.md"]));
  assert.equal(Storage.jaccardOverlap(["a", "b", "b"], ["a", "b", "c"]), 2 / 3);
});

test("renaming a note updates persisted compact result path metadata and survives reopen", async () => {
  const Storage = await loadStorage();
  const { store, adapter, SQL } = await openStore(Storage);
  await seed(store, ["old.md", "keep.md"]);
  const resultId = await store.saveResult({
    ...oneClusterResult(["old.md", "keep.md"]),
    incremental: { mode: "soft", generatedAt: "2026-09-04T00:00:00.000Z", changedPaths: ["old.md", "keep.md"], provisionalPaths: ["old.md"], outOfDistributionPaths: ["old.md"], fullRebuildRecommended: false, knnSupport: { "old.md": 0.8 }, knnDistance: { "old.md": 1.25 } },
  }, { resultId: "rename-result" });
  const compact = JSON.parse(store.query("SELECT result_json AS resultJson FROM results WHERE result_id=?", [resultId])[0].resultJson);
  // The normal v6 writer omits row-aligned ids. Add the same legacy fields a
  // partially migrated result can still carry so this exercises both shapes.
  compact.ids = ["old.md", "keep.md"];
  compact.provisionalPaths = ["old.md"];
  compact.incremental = { ...compact.incremental, changedPaths: ["old.md", "keep.md"], provisionalPaths: ["old.md"], outOfDistributionPaths: ["old.md"], knnSupport: { "old.md": 0.8 }, knnDistance: { "old.md": 1.25 } };
  await store.transaction((db) => db.run("UPDATE results SET result_json=? WHERE result_id=?", [JSON.stringify(compact), resultId]));

  assert.equal(await store.renameNote("./old.md", "./renamed.md"), true);
  const renamed = JSON.parse(store.query("SELECT result_json AS resultJson FROM results WHERE result_id=?", [resultId])[0].resultJson);
  assert.deepEqual(renamed.ids, ["renamed.md", "keep.md"]);
  assert.deepEqual(renamed.provisionalPaths, ["renamed.md"]);
  assert.deepEqual(renamed.incremental.changedPaths, ["renamed.md", "keep.md"]);
  assert.deepEqual(renamed.incremental.provisionalPaths, ["renamed.md"]);
  assert.deepEqual(renamed.incremental.outOfDistributionPaths, ["renamed.md"]);
  assert.deepEqual(renamed.incremental.knnSupport, { "renamed.md": 0.8 });
  assert.deepEqual(renamed.incremental.knnDistance, { "renamed.md": 1.25 });

  store.close();
  const reopened = await new Storage.SqliteClusterStore(adapter, SQL).open();
  assert.deepEqual((await reopened.getResult(resultId)).ids, ["renamed.md", "keep.md"]);
  reopened.close();
});

test("manual title survives generated title replacement and reset restores the latest generated title", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const paths = ["a.md", "b.md"];
  await seed(store, paths);
  const first = oneClusterResult(paths, "Generated one");
  await store.saveResult(first, { resultId: "generated-one" });
  const key = Storage.stableClusterFingerprint(paths);
  await store.saveClusterTitleOverride({ stableClusterKey: key, title: "My title" });
  await store.patchResultTitles("generated-one", { "0": "Generated two" });
  assert.equal((await store.getResult("generated-one")).titles["0"], "Generated two");
  assert.equal(store.getEffectiveClusterTitle(key, "Generated two"), "My title");
  assert.equal(store.getClusterTitleOverride(key).title, "My title");
  assert.equal(await store.resetClusterTitleOverride(key), true);
  assert.equal(store.getClusterTitleOverride(key), undefined);
  assert.equal(store.getEffectiveClusterTitle(key, "Generated two"), "Generated two");
  assert.equal(store.listFeedbackEvents("title-reset").length, 1);
  store.close();
});

test("exact and Jaccard-similar rebuilds migrate overrides and group child references", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const oldLeft = ["a.md", "b.md", "c.md"];
  const oldRight = ["d.md", "e.md", "f.md"];
  await seed(store, [...oldLeft, ...oldRight, "g.md", "h.md"]);
  await store.saveResult(twoClusterResult(oldLeft, oldRight), { resultId: "old" });
  const oldLeftKey = Storage.stableClusterFingerprint(oldLeft);
  const oldRightKey = Storage.stableClusterFingerprint(oldRight);
  await store.saveClusterTitleOverride({ stableClusterKey: oldLeftKey, title: "Pinned left" });
  await store.saveNoteClusterPreference("a.md", oldLeftKey);
  await store.recordTooBroadFeedback(oldLeftKey, "The old left cluster is too broad");
  await store.createManualGroup({ groupId: "g1", title: "Combined", childClusterKeys: [oldLeftKey, oldRightKey] });

  const newLeft = ["a.md", "b.md", "c.md", "g.md"];
  const newRight = ["d.md", "e.md", "f.md", "h.md"];
  await store.saveResult(twoClusterResult(newLeft, newRight), { resultId: "new" });
  const newLeftKey = Storage.stableClusterFingerprint(newLeft);
  const newRightKey = Storage.stableClusterFingerprint(newRight);
  const override = store.listClusterTitleOverrides()[0];
  assert.equal(override.stableClusterKey, newLeftKey, "the unique 3/4-overlap successor is selected");
  assert.equal(override.title, "Pinned left");
  assert.equal(override.orphaned, undefined);
  assert.equal(store.getNoteClusterPreference("a.md").preferredClusterKey, newLeftKey);
  assert.equal(store.listFeedbackEvents("too-broad")[0].stableClusterKey, newLeftKey);
  assert.deepEqual(store.getManualGroup("g1").childClusterKeys, [newLeftKey, newRightKey]);

  const exact = Storage.stableClusterFingerprint(newRight);
  await store.saveClusterTitleOverride({ stableClusterKey: exact, title: "Pinned right" });
  await store.saveResult(twoClusterResult(newLeft, newRight), { resultId: "newer" });
  assert.equal(store.getClusterTitleOverride(exact).title, "Pinned right");
  store.close();
});

test("ambiguous high-overlap candidates leave the title override orphaned", async () => {
  const Storage = await loadStorage();
  const { store } = await openStore(Storage);
  const oldPaths = ["a.md", "b.md", "c.md", "d.md"];
  await seed(store, [...oldPaths, "e.md"]);
  await store.saveResult(oneClusterResult(oldPaths), { resultId: "old" });
  const oldKey = Storage.stableClusterFingerprint(oldPaths);
  await store.saveClusterTitleOverride({ stableClusterKey: oldKey, title: "Do not guess" });
  await store.saveResult(twoClusterResult(["a.md", "b.md", "c.md"], ["d.md", "e.md"]), { resultId: "split" });
  const orphan = store.getClusterTitleOverride(oldKey);
  assert.equal(orphan.title, "Do not guess");
  assert.equal(orphan.orphaned, true);
  assert.equal(store.getEffectiveClusterTitle(oldKey, "Generated replacement"), "Generated replacement");
  store.close();
});

test("preferences, groups, too-broad/general feedback, and rollback are durable across reopen", async () => {
  const Storage = await loadStorage();
  const { store, adapter, SQL } = await openStore(Storage);
  const paths = ["a.md", "b.md", "c.md", "d.md"];
  await seed(store, paths);
  await store.saveResult(twoClusterResult(["a.md", "b.md"], ["c.md", "d.md"]), { resultId: "result" });
  const leftKey = Storage.stableClusterFingerprint(["a.md", "b.md"]);
  const rightKey = Storage.stableClusterFingerprint(["c.md", "d.md"]);
  await store.saveNoteClusterPreference("folder\\note.md", rightKey);
  await store.createManualGroup({ groupId: "group", title: "A group", childClusterKeys: [leftKey, rightKey] });
  await store.recordTooBroadFeedback(leftKey, "The cluster is too broad", { source: "test" });
  await store.recordGeneralFeedback("A general product note", { source: "test" });
  await assert.rejects(store.transaction((db) => { db.run("INSERT INTO manual_groups(group_id,title,created_at,updated_at) VALUES('rollback','x','x','x')"); throw new Error("abort manual transaction"); }), /abort manual transaction/);
  assert.equal(store.getManualGroup("rollback"), undefined);
  assert.deepEqual(store.getNoteClusterPreference("folder/note.md"), { notePath: "folder/note.md", preferredClusterKey: rightKey, createdAt: "2026-09-04T00:00:00.000Z" });
  assert.deepEqual(store.listFeedbackEvents("too-broad").map((event) => event.message), ["The cluster is too broad"]);
  assert.deepEqual(store.listFeedbackEvents("general").map((event) => event.message), ["A general product note"]);
  assert.equal(store.query("SELECT value FROM metadata WHERE key='manual_corrections_schema_version'")[0].value, "1");
  const corrections = store.loadManualCorrections();
  assert.deepEqual(corrections.notePreferences.map((item) => item.notePath), ["folder/note.md"]);
  assert.deepEqual(corrections.groups.map((item) => item.groupId), ["group"]);
  assert.deepEqual(corrections.feedback.map((event) => event.type), ["note-preference-changed", "manual-group-created", "too-broad", "general"]);
  store.close();

  const reopened = await new Storage.SqliteClusterStore(adapter, SQL).open();
  assert.deepEqual(reopened.getNoteClusterPreference("folder/note.md"), { notePath: "folder/note.md", preferredClusterKey: rightKey, createdAt: "2026-09-04T00:00:00.000Z" });
  assert.deepEqual(reopened.getManualGroup("group").childClusterKeys, [leftKey, rightKey]);
  assert.equal(reopened.listFeedbackEvents().filter((event) => event.type === "too-broad").length, 1);
  assert.equal(reopened.query("SELECT name FROM migrations WHERE name='manual-corrections-v1'").length, 1);
  assert.equal(reopened.query("SELECT COUNT(*) AS count FROM results")[0].count, 1);
  reopened.close();
});

test("manual schema marker is added transactionally when opening a pre-foundation database", async () => {
  const Storage = await loadStorage();
  const SQL = await initSqlJs();
  const legacy = new SQL.Database();
  legacy.run(Storage.SQLITE_SCHEMA);
  legacy.run("DROP TABLE feedback_events");
  legacy.run("DROP TABLE manual_group_children");
  legacy.run("DROP TABLE manual_groups");
  legacy.run("DROP TABLE note_cluster_preferences");
  legacy.run("DROP TABLE manual_title_overrides");
  legacy.run("DROP TABLE generated_cluster_snapshots");
  legacy.run("INSERT INTO metadata(key,value) VALUES('schema_version','6')");
  const adapter = new MemoryAdapter();
  adapter.files.set(Storage.SQLITE_PATH, legacy.export().slice().buffer);
  legacy.close();
  const store = await new Storage.SqliteClusterStore(adapter, SQL).open();
  assert.equal(store.query("SELECT value FROM metadata WHERE key='manual_corrections_schema_version'")[0].value, "1");
  assert.equal(store.query("SELECT name FROM migrations WHERE name='manual-corrections-v1'").length, 1);
  store.close();
  const reopened = await new Storage.SqliteClusterStore(adapter, SQL).open();
  assert.equal(reopened.query("SELECT value FROM metadata WHERE key='manual_corrections_schema_version'")[0].value, "1");
  reopened.close();
});
