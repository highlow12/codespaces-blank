import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wikipedia_soft_benchmark.embeddings import embed_chunks, input_checksum, l2_normalize


class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        # A tokenizer double that preserves batch shape while accepting the
        # same call contract as AutoTokenizer.
        return {"input_ids": np.zeros((len(texts), 3), dtype=np.int64), "attention_mask": np.ones((len(texts), 3), dtype=np.int64)}


class FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch = len(kwargs["input_ids"])
        self.calls.append(batch)
        hidden = np.arange(batch * 3 * 4, dtype=np.float32).reshape(batch, 3, 4) + 1
        return {"last_hidden_state": hidden}


class WikipediaBgeEmbeddingTests(unittest.TestCase):
    def _rows(self):
        return [
            {"id": "c1", "text": "first passage", "source_id": "doc-b", "split": "test", "top": "t", "parent": "p", "leaf": "l"},
            {"id": "c2", "text": "second passage", "source_id": "doc-a", "split": "discovery", "top": "t", "parent": "p", "leaf": "l"},
            {"id": "c3", "text": "third passage", "source_id": "doc-b", "split": "test", "top": "t", "parent": "p", "leaf": "l"},
        ]

    def test_normalization_and_document_alignment(self):
        self.assertAlmostEqual(float(np.linalg.norm(l2_normalize(np.ones((2, 4))), axis=1)[0]), 1.0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "chunks.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in self._rows()), encoding="utf-8")
            model = FakeModel()
            result = embed_chunks(source, root / "out", batch_size=2, tokenizer=FakeTokenizer(), model=model)
            self.assertEqual(result.chunk_embeddings.shape, (3, 4))
            self.assertEqual(result.document_embeddings.shape, (2, 4))
            self.assertTrue(np.all(np.isfinite(result.chunk_embeddings)))
            np.testing.assert_allclose(np.linalg.norm(result.chunk_embeddings, axis=1), 1.0, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(result.document_embeddings, axis=1), 1.0, atol=1e-6)
            self.assertEqual(model.calls, [1, 2])

    def test_resume_skips_completed_rows_and_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "chunks.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in self._rows()), encoding="utf-8")
            output = root / "out"
            first_model = FakeModel()
            embed_chunks(source, output, batch_size=2, tokenizer=FakeTokenizer(), model=first_model)
            second_model = FakeModel()
            embed_chunks(source, output, batch_size=2, resume=True, tokenizer=FakeTokenizer(), model=second_model)
            self.assertEqual(second_model.calls, [])
            changed = self._rows()
            changed[0]["text"] = "changed"
            source.write_text("".join(json.dumps(row) + "\n" for row in changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                embed_chunks(source, output, resume=True, tokenizer=FakeTokenizer(), model=FakeModel())

    def test_checksum_is_row_order_sensitive(self):
        rows = self._rows()
        self.assertNotEqual(input_checksum(rows), input_checksum(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
