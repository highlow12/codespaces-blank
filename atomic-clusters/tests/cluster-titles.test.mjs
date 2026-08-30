import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { transform } from "esbuild";

async function loadTitle() {
  let source = await readFile(new URL("../src/title.ts", import.meta.url), "utf8");
  source = source.replace('import { ClusterResult, NoteRecord } from "./types";', "");
  const result = await transform(source, { loader: "ts", format: "esm", target: "es2020" });
  return import(`data:text/javascript;base64,${Buffer.from(result.code).toString("base64")}`);
}
const result = { schemaVersion: 3, ids: ["a.md", "b.md", "c.md"], leafLabels: [0, 0, 1], probabilities: [1, 0.5, 1], outlierProxy: [0, 0, 0], hierarchy: { leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 0.1, mass: 3 }], root: 2 }, pca: {}, timings: {} };
const notes = [
  { path: "a.md", title: "Crochet Bags.md", content: "Crochet bags and yarn patterns", hash: "a" },
  { path: "b.md", title: "Crochet Tote", content: "Crochet bag handles and yarn", hash: "b" },
  { path: "c.md", title: "Muscle Anatomy", content: "Muscle anatomy and movement", hash: "c" }
];

test("keyword normalization removes markdown, particles and conservative plurals", async () => {
  const { cleanText, normalizeKeyword, tokenizeKeywords } = await loadTitle();
  assert.doesNotMatch(cleanText("---\ntags: x\n---\n[link](https://example.com) /tmp/a.md `code`"), /tags:|https?:|tmp\/a/);
  assert.equal(normalizeKeyword("Bags"), "bag");
  assert.equal(normalizeKeyword("가방을"), "가방");
  assert.deepEqual(tokenizeKeywords("The crochet bags 123"), ["crochet", "bag"]);
});

test("leaf and merge titles contain deterministic unique top-three keywords", async () => {
  const { generateKeywordTitles, KEYWORD_TITLE_ALGORITHM_VERSION } = await loadTitle();
  const first = generateKeywordTitles(result, notes); const second = generateKeywordTitles(result, notes);
  assert.equal(first.titles["0"].split(" · ").length, 3);
  assert.match(first.titles["0"], /crochet/);
  assert.match(first.titles["2"], /^crochet(?: · [^·]+){0,2}$/);
  assert.deepEqual(first.titles, second.titles);
  assert.equal(first.schemaVersion, 3);
  assert.equal(first.titleGeneration.method, "keywords");
  assert.equal(first.titleGeneration.algorithmVersion, KEYWORD_TITLE_ALGORITHM_VERSION);
  assert.equal(first.titleGeneration.nodeCount, 3);
  assert.ok(!JSON.stringify(first).includes("Crochet bags and yarn patterns"));
});

test("title generation responds to cancellation without touching the input result", async () => {
  const { generateKeywordTitles } = await loadTitle(); const controller = new AbortController(); controller.abort();
  assert.throws(() => generateKeywordTitles(result, notes, { signal: controller.signal }), /cancelled/);
  assert.equal(result.schemaVersion, 3); assert.equal(result.titles, undefined);
});

test("production title path is dependency-free", async () => {
  const title = await readFile(new URL("../src/title.ts", import.meta.url), "utf8");
  assert.doesNotMatch(title, /pipeline\(|WebGPU|onnx|huggingface/i);
});

test("result storage preserves keyword titles in v3 and strips legacy titles only", async () => {
  const storage = await readFile(new URL("../src/storage.ts", import.meta.url), "utf8");
  assert.match(storage, /if \(result\.schemaVersion === 3\) return result;/);
  assert.match(storage, /schemaVersion: 3 as const, titles: undefined, titleGeneration: undefined/);
});

test("v6 titles use field-aware ngrams, preserve inflected forms, and include residuals in parent scoring", async () => {
  const { generateKeywordTitles, KEYWORD_TITLE_ALGORITHM_VERSION } = await loadTitle();
  const v6 = {
    schemaVersion: 6, ids: ["a.md", "b.md", "c.md"], leafLabels: [0, 1, 1], probabilities: [1, 1, 1], outlierProxy: [0, 0, 0],
    hierarchy: {
      leaves: [0, 1], merges: [{ id: 2, left: 0, right: 1, distance: 1, mass: 3 }], root: 2,
      nodes: [{ id: 0, children: [], descendantLeaves: [0], distance: 0, mass: 1 }, { id: 1, children: [], descendantLeaves: [1], distance: 0, mass: 2 }, { id: 2, children: [0, 1], descendantLeaves: [0, 1], distance: 1, mass: 3 }], rootChildren: [0, 1], splitMethod: "distance-knee-2-5"
    }, hierarchyPlacements: [{ kind: "leaf", nodeId: 0, confidence: 1 }, { kind: "leaf", nodeId: 1, confidence: 1 }, { kind: "residual", nodeId: 2, confidence: 0.4 }], pca: {}, timings: {}
  };
  const v6Notes = [
    { path: "a.md", title: "Crochet Bags", content: "---\ntags: [handmade]\n---\n# Crochet Bags\n[[Yarn|Yarn craft]]\ncrochet bags and handles", hash: "a" },
    { path: "Muscle.md", title: "Muscle", content: "## Anatomy\nmuscle movement", hash: "b" },
    { path: "Noise.md", title: "Residual", content: "handmade yarn", hash: "c" }
  ];
  const titled = generateKeywordTitles(v6, v6Notes);
  assert.equal(titled.titleGeneration.algorithmVersion, KEYWORD_TITLE_ALGORITHM_VERSION);
  assert.equal(titled.titleGeneration.algorithmVersion, "contrastive-keyphrases-v2");
  assert.ok(titled.titles["0"].includes("crochet"));
  assert.ok(titled.titles["2"].length > 0);
  assert.ok(titled.titles["0"].split(" · ").length <= 2);
  assert.deepEqual(v6.titles, undefined);
});
