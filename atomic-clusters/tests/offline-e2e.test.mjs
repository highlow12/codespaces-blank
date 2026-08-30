import test from "node:test";
import assert from "node:assert/strict";
import { sampleRecords } from "../scripts/offline-e2e.mjs";

test("offline full-dataset selection preserves source order", () => {
  const records = Array.from({ length: 5 }, (_, index) => ({ id: `note-${index}` }));
  const selected = sampleRecords(records, 5, 42);
  assert.deepEqual(selected.map((record) => record.id), records.map((record) => record.id));
  assert.notEqual(selected, records);
});

test("offline subset selection remains deterministic", () => {
  const records = Array.from({ length: 10 }, (_, index) => ({ id: `note-${index}` }));
  assert.deepEqual(
    sampleRecords(records, 4, 42).map((record) => record.id),
    sampleRecords(records, 4, 42).map((record) => record.id)
  );
});
