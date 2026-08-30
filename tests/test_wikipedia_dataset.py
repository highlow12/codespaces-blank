import json
import tempfile
import unittest
import urllib.error
import tarfile
from unittest.mock import patch
from pathlib import Path

from wikipedia_soft_benchmark.core import (
    DatasetError,
    Fetcher,
    chunk_paragraphs,
    clean_html,
    gzip_jsonl_write,
    gzip_jsonl_read,
    jsonl_write,
    make_source_id,
    normalize_source_records,
    package_dataset,
    sha256_bytes,
    sha256_text,
    make_chunk_id,
    validate_manifest,
    validate_artifacts,
)
from wikipedia_soft_benchmark.replace_short_sources import (
    _build_used,
    _identity_values,
    _is_unique,
    _replacement_ranks,
    _stage_and_validate,
)


class WordTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": text.split()}


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class WikipediaDatasetTests(unittest.TestCase):
    @staticmethod
    def _mini_row():
        return {
            "page_id": 123,
            "title": "Test Article",
            "revision_id": "456",
            "permanent_url": "https://en.wikipedia.org/w/index.php?oldid=456",
            "top": "natural-science",
            "parent": "physical-science",
            "leaf": "physics",
            "split": "test",
            "approved": True,
        }

    @staticmethod
    def _mini_source(root, row, *, revision=None):
        source_id = make_source_id(row)
        html = "<p>" + " ".join(["article"] + ["word"] * 120) + "</p>"
        payload = {
            "source_id": source_id,
            "page_id": row["page_id"],
            "title": row["title"],
            "revision_id": row["revision_id"] if revision is None else revision,
            "permanent_url": row["permanent_url"],
            "html": html,
            "raw_sha256": sha256_text(html),
        }
        path = root / f"{source_id}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    @staticmethod
    def _mini_chunks(root, row, *, token_count=121):
        source_id = make_source_id(row)
        text = " ".join(["article"] + ["word"] * 120)
        jsonl_write(root, [{
            "id": make_chunk_id(source_id, "", 0, 0, text),
            "text": text,
            "token_count": token_count,
            "source_id": source_id,
            "section": "",
            "paragraph_start": 0,
            "paragraph_end": 0,
            "split": row["split"],
            "top": row["top"],
            "parent": row["parent"],
            "leaf": row["leaf"],
        }])

    def test_clean_html_keeps_paragraphs_and_removes_non_body_elements(self):
        html = "<h2>Intro</h2><p>One &amp; two.</p><table><tr><td>omit</td></tr></table><p class='reference'>bad</p><p>Three <sup class='reference'>[1]</sup> four.</p><ul><li>omit</li></ul>"
        rows = clean_html(html)
        self.assertEqual([row["text"] for row in rows], ["One & two.", "Three four."])
        self.assertEqual(rows[0]["section"], "Intro")

    def test_clean_html_recovers_after_void_elements_in_skipped_region(self):
        html = "<table><link rel='stylesheet'><img src='x'><br><meta charset='utf-8'><p>omit</p></table><p>Keep this paragraph.</p>"
        rows = clean_html(html)
        self.assertEqual([row["text"] for row in rows], ["Keep this paragraph."])

    def test_clean_html_handles_nested_skipped_elements(self):
        html = "<table><div class='infobox'><span><img src='x'>omit</span><ul><li>omit</li></ul></div><table><p>omit</p></table></table><p>Keep this paragraph.</p>"
        rows = clean_html(html)
        self.assertEqual([row["text"] for row in rows], ["Keep this paragraph."])

    def test_clean_html_removes_class_skipped_region_with_nested_void_elements(self):
        html = "<p>Before <span class='reference'><a href='#ref'>ref</a><br><img src='x'>ignored</span> after.</p><p>Keep this paragraph.</p>"
        rows = clean_html(html)
        self.assertEqual([row["text"] for row in rows], ["Before after.", "Keep this paragraph."])

    def test_chunking_is_bounded_and_deterministic(self):
        text = " ".join(["word"] * 330)
        paragraphs = [{"section": "S", "text": text, "paragraph_index": 0}]
        first = chunk_paragraphs(paragraphs, WordTokenizer())
        second = chunk_paragraphs(paragraphs, WordTokenizer())
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 3)
        self.assertTrue(all(100 <= row["token_count"] <= 250 for row in first))
        self.assertTrue(make_chunk_id("page-1-revision-2", "S", 0, 0, first[0]["text"]))

    def test_manifest_rejects_unapproved_and_wrong_revision_url(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.jsonl"
            row = {"page_id": 1, "title": "Physics", "revision_id": "2", "permanent_url": "https://en.wikipedia.org/w/index.php?oldid=3", "top": "natural-science", "parent": "physical-science", "leaf": "physics", "split": "test", "approved": False}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                validate_manifest(path, require_complete=False)

    def test_replacement_uniqueness_is_independent_per_identity_field(self):
        original = {"page_id": 1, "title": "A Page", "revision_id": "10", "permanent_url": "https://en.wikipedia.org/w/index.php?oldid=10"}
        used = _build_used([original])
        for field, value in {
            "page_id": 1,
            "title": "a page",
            "revision_id": "10",
            "permanent_url": original["permanent_url"],
        }.items():
            candidate = dict(original)
            candidate["page_id"] = 2
            candidate["title"] = "Other"
            candidate["revision_id"] = "11"
            candidate["permanent_url"] = "https://en.wikipedia.org/w/index.php?oldid=11"
            candidate[field] = value
            self.assertFalse(_is_unique(candidate, used), field)

    def test_replacement_ranks_are_deterministic_and_unique_within_leaf(self):
        candidates = [
            {"leaf": "geology", "candidate_rank": 60},
            {"leaf": "physics", "candidate_rank": 60},
        ]
        failed = [
            {"page_id": 1, "revision_id": "1", "leaf": "geology"},
            {"page_id": 2, "revision_id": "2", "leaf": "geology"},
            {"page_id": 3, "revision_id": "3", "leaf": "physics"},
        ]
        ranks = _replacement_ranks(failed, candidates)
        self.assertEqual([ranks["page-1-revision-1"], ranks["page-2-revision-2"], ranks["page-3-revision-3"]], [61, 62, 61])

    def test_failed_replacement_staging_does_not_mutate_active_artifacts(self):
        project = Path(__file__).parents[1]
        active_paths = [
            project / "wikipedia_soft_benchmark" / "candidate_manifest.jsonl",
            project / "wikipedia_soft_benchmark" / "source_manifest.jsonl",
            project / "wikipedia_sources" / "fetch_checkpoint.json",
            project / "wikipedia_artifacts" / "chunks.jsonl",
            project / "wikipedia_artifacts" / "validation_report.json",
        ]
        if not all(path.exists() for path in active_paths):
            self.skipTest("generated collection artifacts are not present")
        before = {path: path.read_bytes() for path in active_paths}
        candidates = json.loads("[" + ",".join(line for line in (project / "wikipedia_soft_benchmark" / "candidate_manifest.jsonl").read_text().splitlines() if line) + "]")
        manifest = json.loads("[" + ",".join(line for line in (project / "wikipedia_soft_benchmark" / "source_manifest.jsonl").read_text().splitlines() if line) + "]")
        manifest[1]["page_id"] = manifest[0]["page_id"]
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(DatasetError):
            _stage_and_validate(Path(temp), manifest, candidates)
        self.assertEqual(before, {path: path.read_bytes() for path in active_paths})

    def test_fetcher_fails_revision_drift_without_fallback(self):
        row = {"page_id": 1, "title": "Physics", "revision_id": "2", "permanent_url": "https://en.wikipedia.org/w/index.php?oldid=2"}
        fetcher = Fetcher(opener=lambda *_args, **_kwargs: FakeResponse({"parse": {"revid": 3, "text": "body"}}), min_interval=0, retries=2)
        with self.assertRaises(DatasetError):
            fetcher.fetch(row)

    def test_fetcher_rejects_api_error_missing_body_and_redirect(self):
        row = {"page_id": 1, "title": "Physics", "revision_id": "2", "permanent_url": "https://en.wikipedia.org/w/index.php?oldid=2"}
        cases = [
            {"error": {"code": "missingrev"}},
            {"parse": {"pageid": 1, "title": "Physics", "revid": 2, "text": ""}},
            {"parse": {"pageid": 1, "title": "Physics", "revid": 2, "text": "ok", "redirect": True}},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                fetcher = Fetcher(opener=lambda *_args, payload=payload, **_kwargs: FakeResponse(payload), min_interval=0, retries=0)
                with self.assertRaises(DatasetError):
                    fetcher.fetch(row)

    def test_fetcher_retries_transient_failure_then_reports_exhaustion(self):
        calls = []

        def fail(*_args, **_kwargs):
            calls.append(1)
            raise urllib.error.URLError("offline")

        row = {"page_id": 1, "title": "Physics", "revision_id": "2", "permanent_url": "https://en.wikipedia.org/w/index.php?oldid=2"}
        fetcher = Fetcher(opener=fail, min_interval=0, retries=2)
        with self.assertRaises(DatasetError):
            fetcher.fetch(row)
        self.assertEqual(len(calls), 3)

    def test_gzip_and_chunk_ids_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.gz", Path(temp) / "b.gz"
            rows = [{"id": 2, "text": "same"}, {"id": 1, "text": "stable"}]
            gzip_jsonl_write(first, rows)
            gzip_jsonl_write(second, rows)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(make_chunk_id("s", "x", 1, 2, "text"), make_chunk_id("s", "x", 1, 2, "text"))

    def test_directory_normalization_matches_package_source_records(self):
        row = self._mini_row()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = root / "sources"
            sources.mkdir()
            self._mini_source(sources, row)
            normalized = normalize_source_records(sources)
            chunks = root / "chunks.jsonl"
            self._mini_chunks(chunks, row)
            manifest = root / "manifest.jsonl"
            jsonl_write(manifest, [row])
            package = root / "package"
            with patch("wikipedia_soft_benchmark.core.validate_manifest", return_value=[row]), patch("wikipedia_soft_benchmark.core.validate_artifacts", return_value={}):
                package_dataset(package, manifest=manifest, source_dir=sources, chunks=chunks, archive_path=root / "package.tar.gz", tokenizer=WordTokenizer())
            packaged = gzip_jsonl_read(package / "sources.jsonl.gz")
            self.assertEqual(packaged, normalized)

    def test_validation_detects_missing_source_and_revision_mismatch(self):
        row = self._mini_row()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = root / "sources"
            sources.mkdir()
            self._mini_source(sources, row, revision="999")
            chunks = root / "chunks.jsonl"
            self._mini_chunks(chunks, row)
            report = root / "validation.json"
            with patch("wikipedia_soft_benchmark.core.validate_manifest", return_value=[row]):
                with self.assertRaises(DatasetError) as raised:
                    validate_artifacts(root / "manifest.jsonl", sources, chunks, report, tokenizer=WordTokenizer())
            self.assertIn("revision_id does not match manifest", str(raised.exception))
            sources.joinpath(f"{make_source_id(row)}.json").unlink()
            with patch("wikipedia_soft_benchmark.core.validate_manifest", return_value=[row]):
                with self.assertRaises(DatasetError) as raised:
                    validate_artifacts(root / "manifest.jsonl", sources, chunks, report, tokenizer=WordTokenizer())
            self.assertIn("sources do not exactly match approved manifest", str(raised.exception))

    def test_validation_report_persists_when_validation_fails(self):
        row = self._mini_row()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = root / "sources"
            sources.mkdir()
            self._mini_source(sources, row)
            chunks = root / "chunks.jsonl"
            self._mini_chunks(chunks, row, token_count=99)
            report = root / "nested" / "validation.json"
            with patch("wikipedia_soft_benchmark.core.validate_manifest", return_value=[row]):
                with self.assertRaises(DatasetError):
                    validate_artifacts(root / "manifest.jsonl", sources, chunks, report, tokenizer=WordTokenizer(), near_threshold=0.0)
            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(saved["errors"])

    def test_committed_baseline_validation_report_is_explicitly_blocked(self):
        report_path = Path(__file__).parents[1] / "wikipedia_soft_benchmark" / "validation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["release_blocked"])
        self.assertEqual(report["manifest_articles"], 720)
        self.assertEqual(report["source_count"], 0)
        self.assertEqual(report["chunk_count"], 0)
        self.assertIn("source manifest promoted", report["reason"])
        self.assertEqual(report["tokenizer"]["revision"], "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")

    def test_package_contains_checksums_license_attribution_and_archive(self):
        from wikipedia_soft_benchmark.core import hierarchy_nodes

        class FakeTokenizer(WordTokenizer):
            pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            sources = root / "sources"
            sources.mkdir()
            chunks = root / "chunks.jsonl"
            rows, chunk_rows = [], []
            leaves = [slug for slug, node in hierarchy_nodes().items() if node["level"] == "leaf"]
            page = 1000
            for leaf in leaves:
                node = hierarchy_nodes()[leaf]
                parent = node["parent"]
                top = hierarchy_nodes()[parent]["parent"]
                for offset in range(60):
                    split = "discovery" if offset < 36 else "calibration" if offset < 48 else "test"
                    revision = str(page + 9000)
                    title = f"Test Article {page}"
                    row = {"page_id": page, "title": title, "revision_id": revision, "permanent_url": f"https://en.wikipedia.org/w/index.php?oldid={revision}", "top": top, "parent": parent, "leaf": leaf, "split": split, "approved": True}
                    rows.append(row)
                    source_id = make_source_id(row)
                    text = " ".join([f"article{page}"] + ["word"] * 99)
                    payload = {"source_id": source_id, "page_id": page, "title": title, "revision_id": revision, "permanent_url": row["permanent_url"], "html": f"<p>{text}</p>", "raw_sha256": sha256_text(text)}
                    (sources / f"{source_id}.json").write_text(json.dumps(payload), encoding="utf-8")
                    chunk_rows.append({"id": make_chunk_id(source_id, "", 0, 0, text), "text": text, "token_count": 100, "source_id": source_id, "section": "", "paragraph_start": 0, "paragraph_end": 0, "split": split, "top": top, "parent": parent, "leaf": leaf})
                    page += 1
            jsonl_write(manifest, rows)
            jsonl_write(chunks, chunk_rows)
            package = root / "package"
            archive = package.with_suffix(".tar.gz")
            result = package_dataset(package, manifest=manifest, source_dir=sources, chunks=chunks, archive_path=archive, tokenizer=FakeTokenizer())
            self.assertEqual(result, archive)
            self.assertTrue((package / "config.json").exists())
            self.assertIn("a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", (package / "dataset_card.md").read_text())
            sums = (package / "SHA256SUMS").read_text().splitlines()
            self.assertTrue(any(line.endswith("  LICENSE-CC-BY-SA-4.0.txt") for line in sums))
            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
            self.assertTrue(all(name.startswith("wikipedia-soft-benchmark-v1/") for name in names))
            attribution = __import__("gzip").open(package / "attribution.jsonl.gz", "rt", encoding="utf-8").readline()
            self.assertIn('"history_url"', attribution)
            self.assertIn('"permanent_url"', attribution)
