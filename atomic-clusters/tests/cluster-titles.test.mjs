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
