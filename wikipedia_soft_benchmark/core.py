"""The collection-stage implementation.

Only standard-library modules are needed for fetching, cleaning, chunking, and
packaging.  ``transformers`` is intentionally an optional dependency: it is
required by the production ``chunk``/``validate`` commands, while unit tests
may inject a deterministic tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import math
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
DEFAULT_HIERARCHY = ROOT / "hierarchy.json"
DEFAULT_MANIFEST = ROOT / "source_manifest.jsonl"
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "wikipedia-soft-benchmark/0.1 (dataset research; contact: dataset@example.invalid)"
TOKENIZER_MODEL = "BAAI/bge-base-en-v1.5"
TOKENIZER_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
MIN_TOKENS = 100
MAX_TOKENS = 250
LEAF_COUNT = 12
ARTICLE_COUNT = 720


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the committed collection contract and reject mutable settings."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot load config: {path}") from exc
    tokenizer = config.get("tokenizer", {})
    revision = tokenizer.get("revision")
    if tokenizer.get("model") != TOKENIZER_MODEL or not re.fullmatch(r"[0-9a-f]{40}", str(revision or "")):
        raise DatasetError("config must pin BAAI/bge-base-en-v1.5 to an immutable 40-character revision")
    chunking = config.get("chunking", {})
    if chunking.get("min_tokens") != MIN_TOKENS or chunking.get("max_tokens") != MAX_TOKENS or chunking.get("max_chunks_per_article") != 3:
        raise DatasetError("config chunking bounds do not match the collection contract")
    return config


class DatasetError(ValueError):
    """An input or generated artifact failed a release invariant."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise DatasetError(f"{path}:{line_no}: expected an object")
            rows.append(value)
    return rows


def jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def gzip_jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    import gzip
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    # GzipFile accepts mtime on all supported Python versions; using a text
    # wrapper keeps output bytes reproducible across package runs.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()


def gzip_jsonl_read(path: Path) -> list[dict[str, Any]]:
    import gzip
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_no}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise DatasetError(f"{path}:{line_no}: expected an object")
                rows.append(row)
    return rows


def hierarchy_nodes(path: Path = DEFAULT_HIERARCHY) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    nodes = data.get("nodes")
    if not isinstance(nodes, dict):
        raise DatasetError("hierarchy.nodes must be an object")
    for slug, node in nodes.items():
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            raise DatasetError(f"invalid hierarchy slug: {slug}")
        if node.get("level") not in {"top", "parent", "leaf"}:
            raise DatasetError(f"invalid hierarchy level for {slug}")
        parent = node.get("parent")
        if parent is not None and parent not in nodes:
            raise DatasetError(f"unknown parent {parent!r} for {slug}")
    tops = [s for s, n in nodes.items() if n["level"] == "top"]
    leaves = [s for s, n in nodes.items() if n["level"] == "leaf"]
    if len(tops) != 2 or len(leaves) != LEAF_COUNT:
        raise DatasetError(f"hierarchy must contain 2 tops and {LEAF_COUNT} leaves")
    for leaf in leaves:
        p = nodes[leaf]["parent"]
        if not p or nodes[p]["level"] != "parent":
            raise DatasetError(f"leaf {leaf} must have a parent node")
        top = nodes[p]["parent"]
        if not top or nodes[top]["level"] != "top":
            raise DatasetError(f"parent {p} must have a top node")
    return nodes


MANIFEST_FIELDS = {"page_id", "title", "revision_id", "permanent_url", "top", "parent", "leaf", "split", "approved"}
SPLITS = {"discovery", "calibration", "test"}


def validate_manifest(path: Path = DEFAULT_MANIFEST, hierarchy: Path = DEFAULT_HIERARCHY, *, require_complete: bool = True) -> list[dict[str, Any]]:
    nodes = hierarchy_nodes(hierarchy)
    rows = jsonl_read(path)
    errors: list[str] = []
    if require_complete and len(rows) != ARTICLE_COUNT:
        errors.append(f"expected exactly {ARTICLE_COUNT} manifest rows, got {len(rows)}")
    seen: dict[str, set[Any]] = defaultdict(set)
    counts: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(rows, 1):
        missing = MANIFEST_FIELDS - set(row)
        if missing:
            errors.append(f"row {index}: missing fields {sorted(missing)}")
            continue
        page_id, revision = row["page_id"], row["revision_id"]
        if not isinstance(page_id, int) or page_id <= 0:
            errors.append(f"row {index}: page_id must be positive integer")
        if not isinstance(revision, (str, int)) or not str(revision):
            errors.append(f"row {index}: revision_id must be non-empty")
        if not isinstance(row["title"], str) or not row["title"].strip():
            errors.append(f"row {index}: title must be non-empty")
        if not isinstance(row["permanent_url"], str) or "oldid=" not in row["permanent_url"]:
            errors.append(f"row {index}: permanent_url must contain oldid=")
        else:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(row["permanent_url"]).query)
            if str(query.get("oldid", [""])[0]) != str(revision):
                errors.append(f"row {index}: permanent_url oldid does not match revision_id")
        if row.get("namespace", 0) != 0 or row.get("is_redirect", False) is True or row.get("is_disambiguation", False) is True or row.get("article_type", "article") != "article":
            errors.append(f"row {index}: only non-redirect main-namespace articles are allowed")
        if "review_status" in row and row["review_status"] != "approved":
            errors.append(f"row {index}: review_status must be approved")
        leaf, parent, top, split = row["leaf"], row["parent"], row["top"], row["split"]
        if leaf not in nodes or nodes.get(leaf, {}).get("level") != "leaf":
            errors.append(f"row {index}: unknown/non-leaf leaf {leaf!r}")
        elif nodes[leaf]["parent"] != parent or nodes[parent]["parent"] != top:
            errors.append(f"row {index}: hierarchy path does not match leaf")
        if split not in SPLITS:
            errors.append(f"row {index}: invalid split {split!r}")
        if row["approved"] is not True:
            errors.append(f"row {index}: article is not explicitly approved=true")
        seen["page_id"].add(page_id)
        seen["revision_id"].add(str(revision))
        seen["title"].add(str(row["title"]).casefold())
        seen["permanent_url"].add(row["permanent_url"])
        counts[(str(leaf), str(split))] += 1
    for key, values in seen.items():
        if len(values) != len(rows):
            errors.append(f"duplicate {key} values in manifest")
    leaves = [s for s, n in nodes.items() if n["level"] == "leaf"]
    for leaf in leaves:
        if require_complete:
            for split, expected in (("discovery", 36), ("calibration", 12), ("test", 12)):
                if counts[(leaf, split)] != expected:
                    errors.append(f"{leaf}/{split}: expected {expected}, got {counts[(leaf, split)]}")
        elif sum(counts[(leaf, split)] for split in SPLITS) and sum(counts[(leaf, split)] for split in SPLITS) != 50:
            errors.append(f"{leaf}: partial manifest must contain 50 rows when populated")
    if errors:
        raise DatasetError("manifest validation failed:\n- " + "\n- ".join(errors))
    return rows


class _ArticleParser(HTMLParser):
    SKIP = {"table", "ul", "ol", "dl", "figure", "style", "script", "noscript", "math", "sup"}
    BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
    # HTMLParser reports void elements through handle_starttag(), but they do
    # not have a matching end tag.  They must therefore never be added to the
    # skip-region stack.
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_stack: list[str] = []
        self.block: str | None = None
        self.buffer: list[str] = []
        self.section = ""
        self.paragraphs: list[dict[str, Any]] = []

    @property
    def skip_depth(self) -> int:
        """Return the number of open skipped elements.

        Keep this as a derived value for callers/debuggers that used the old
        depth field, while using tag identity internally so void elements do
        not make the skip state leak.
        """
        return len(self._skip_stack)

    @staticmethod
    def _is_skip(attrs: Mapping[str, str | None]) -> bool:
        values = " ".join(str(attrs.get(k) or "") for k in ("class", "id", "role")).casefold()
        return any(x in values for x in ("infobox", "navbox", "reference", "reflist", "bibliography", "metadata", "sidebar", "hatnote", "caption", "mw-editsection"))

    def _enter_skip(self, tag: str) -> None:
        if tag not in self.VOID:
            self._skip_stack.append(tag)

    def _leave_skip(self, tag: str) -> None:
        if not self._skip_stack or tag not in self._skip_stack:
            return
        # HTML from the MediaWiki API can contain imperfect nesting.  If a
        # containing skipped element closes before an inner one, discard the
        # stale inner entries as well so they cannot suppress the rest of the
        # article.
        matching_index = next(index for index in range(len(self._skip_stack) - 1, -1, -1) if self._skip_stack[index] == tag)
        del self._skip_stack[matching_index:]

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs_list)
        if self.skip_depth:
            if tag in self.SKIP or self._is_skip(attrs):
                self._enter_skip(tag)
            return
        if tag in self.SKIP or self._is_skip(attrs):
            self._enter_skip(tag)
            return
        if tag in self.BLOCK:
            self._finish()
            self.block = tag
            self.buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            self._leave_skip(tag)
            return
        if self.block == tag:
            text = re.sub(r"\s+", " ", "".join(self.buffer)).strip()
            if text:
                if tag.startswith("h"):
                    self.section = text
                elif tag == "p":
                    self.paragraphs.append({"section": self.section, "text": text, "paragraph_index": len(self.paragraphs)})
            self.block = None
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.block and not self.skip_depth:
            self.buffer.append(data)

    def _finish(self) -> None:
        if self.block:
            self.handle_endtag(self.block)

    def result(self) -> list[dict[str, Any]]:
        self._finish()
        return self.paragraphs


def clean_html(document_html: str) -> list[dict[str, Any]]:
    parser = _ArticleParser()
    parser.feed(document_html)
    parser.close()
    return parser.result()


def _token_count(text: str, tokenizer: Any) -> int:
    value = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = value["input_ids"] if isinstance(value, Mapping) else value
    return len(ids)


class BGETokenizer:
    """Load the pinned model tokenizer; no implicit latest revision fallback."""

    def __init__(self, revision: str | None = None, model: str | None = None, *, config_path: Path = DEFAULT_CONFIG) -> None:
        config = load_config(config_path)
        revision = revision or config["tokenizer"]["revision"]
        model = model or config["tokenizer"]["model"]
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for BGE token counts; install it before chunk/validate") from exc
        if not revision or revision in {"main", "master", "latest"}:
            raise ValueError("tokenizer revision must be explicitly pinned")
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        return self.tokenizer(text, **kwargs)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])(?:\s+|$)", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_paragraphs(paragraphs: Sequence[Mapping[str, Any]], tokenizer: Any, *, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS, max_chunks: int = 3) -> list[dict[str, Any]]:
    if min_tokens <= 0 or max_tokens < min_tokens:
        raise ValueError("invalid token bounds")
    units: list[dict[str, Any]] = []
    for paragraph in paragraphs:
        text = str(paragraph["text"]).strip()
        sentences = _sentences(text) or [text]
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and _token_count(candidate, tokenizer) > max_tokens:
                units.append({"section": paragraph.get("section", ""), "text": current, "start": paragraph["paragraph_index"], "end": paragraph["paragraph_index"]})
                current = sentence
            elif not current and _token_count(sentence, tokenizer) > max_tokens:
                words = sentence.split()
                piece: list[str] = []
                for word in words:
                    trial = " ".join(piece + [word])
                    if piece and _token_count(trial, tokenizer) > max_tokens:
                        units.append({"section": paragraph.get("section", ""), "text": " ".join(piece), "start": paragraph["paragraph_index"], "end": paragraph["paragraph_index"]})
                        piece = [word]
                    else:
                        piece.append(word)
                if piece:
                    current = " ".join(piece)
            else:
                current = candidate
        if current:
            units.append({"section": paragraph.get("section", ""), "text": current, "start": paragraph["paragraph_index"], "end": paragraph["paragraph_index"]})
    merged: list[dict[str, Any]] = []
    for unit in units:
        if merged and _token_count(merged[-1]["text"] + " " + unit["text"], tokenizer) <= max_tokens:
            merged[-1]["text"] += " " + unit["text"]
            merged[-1]["end"] = unit["end"]
        else:
            merged.append(dict(unit))
    # A trailing short unit is joined to its predecessor whenever possible.
    if len(merged) > 1 and _token_count(merged[-1]["text"], tokenizer) < min_tokens:
        candidate = merged[-2]["text"] + " " + merged[-1]["text"]
        if _token_count(candidate, tokenizer) <= max_tokens:
            merged[-2]["text"] = candidate
            merged[-2]["end"] = merged[-1]["end"]
            merged.pop()
    eligible = [u for u in merged if min_tokens <= _token_count(u["text"], tokenizer) <= max_tokens]
    if len(eligible) > max_chunks:
        positions = [round(i * (len(eligible) - 1) / (max_chunks - 1)) for i in range(max_chunks)] if max_chunks > 1 else [0]
        eligible = [eligible[i] for i in positions]
    return [{**u, "token_count": _token_count(u["text"], tokenizer)} for u in eligible]


def make_source_id(row: Mapping[str, Any]) -> str:
    return f"page-{row['page_id']}-revision-{row['revision_id']}"


def make_chunk_id(source_id: str, section: str, start: int, end: int, text: str) -> str:
    section_hash = sha256_text(section)[:16]
    text_hash = sha256_text(text)
    return f"{source_id}:section-{section_hash}:paragraph-{start}-{end}:text-{text_hash}"


def normalize_source_record(value: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Convert one fetched page payload into the canonical source record.

    Fetch artifacts retain the raw MediaWiki HTML.  Packages and validation
    consume the cleaned article text, so this function is deliberately the
    only implementation of that conversion.
    """
    location = f" in {path}" if path is not None else ""
    required = ("source_id", "page_id", "title", "revision_id", "permanent_url")
    missing = [field for field in required if field not in value]
    if missing:
        raise DatasetError(f"source payload{location} is missing fields: {missing}")
    html = value.get("html")
    if not isinstance(html, str):
        raise DatasetError(f"source payload{location} has no string html body")
    paragraphs = clean_html(html)
    text = "\n\n".join(str(paragraph["text"]) for paragraph in paragraphs)
    return {
        "source_id": value["source_id"],
        "page_id": value["page_id"],
        "title": value["title"],
        "revision_id": value["revision_id"],
        "permanent_url": value["permanent_url"],
        "text": text,
        "text_sha256": sha256_text(text),
        "raw_sha256": value.get("raw_sha256"),
    }


def normalize_source_records(source_path: Path) -> list[dict[str, Any]]:
    """Load canonical source records from a fetch directory or source JSONL.

    The directory form is the fetch artifact (``page-*.json``); the file form
    remains compatible with existing plain and gzip-compressed source JSONL.
    """
    if source_path.is_dir():
        paths = sorted(source_path.glob("page-*.json"))
        records: list[dict[str, Any]] = []
        for path in paths:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetError(f"cannot read source payload {path}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise DatasetError(f"source payload {path} must be a JSON object")
            records.append(normalize_source_record(value, path=path))
        return records
    if source_path.suffix == ".gz":
        return gzip_jsonl_read(source_path)
    return jsonl_read(source_path)


@dataclass
class Fetcher:
    endpoint: str = DEFAULT_API
    user_agent: str = DEFAULT_USER_AGENT
    retries: int = 4
    timeout: float = 30.0
    min_interval: float = 0.5
    opener: Callable[..., Any] = urllib.request.urlopen

    def fetch(self, row: Mapping[str, Any]) -> dict[str, Any]:
        # MediaWiki rejects combining ``oldid`` and ``pageid``.  The fixed
        # revision is sufficient to identify the page; retain the returned
        # page-id check below as an invariant against an unexpected payload.
        query = {"action": "parse", "oldid": str(row["revision_id"]), "prop": "text|revid|displaytitle|categories", "redirects": "0", "format": "json", "formatversion": "2"}
        request = urllib.request.Request(self.endpoint + "?" + urllib.parse.urlencode(query), headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(min(60.0, 2.0 ** (attempt - 1)))
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    body = response.read()
                payload = json.loads(body.decode("utf-8"))
                if "error" in payload:
                    raise DatasetError(f"MediaWiki error: {payload['error']}")
                parsed = payload.get("parse") or {}
                actual_revision = str(parsed.get("revid", ""))
                if actual_revision != str(row["revision_id"]):
                    raise DatasetError(f"revision drift for page {row['page_id']}: expected {row['revision_id']}, got {actual_revision}")
                actual_page = parsed.get("pageid", parsed.get("page_id"))
                if str(actual_page) != str(row["page_id"]):
                    raise DatasetError(f"page id mismatch for revision {row['revision_id']}: expected {row['page_id']}, got {actual_page}")
                actual_title = str(parsed.get("title", "")).replace("_", " ").strip()
                expected_title = re.sub(r"\s+", " ", str(row["title"])).strip()
                if not actual_title or actual_title.casefold() != expected_title.casefold():
                    raise DatasetError(f"title mismatch for page {row['page_id']}: expected {expected_title!r}, got {actual_title!r}")
                html = parsed.get("text", "")
                categories = " ".join(str(item.get("*", item.get("title", ""))) for item in parsed.get("categories", []) if isinstance(item, Mapping)).casefold()
                if parsed.get("redirect") or expected_title.casefold().startswith("list of ") or "disambiguation" in expected_title.casefold() or "disambiguation" in categories or "wikipedia lists" in categories:
                    raise DatasetError(f"list/disambiguation page is not allowed: {expected_title}")
                if not isinstance(html, str) or not html.strip():
                    raise DatasetError(f"empty body for page {row['page_id']} revision {row['revision_id']}")
                return {"source_id": make_source_id(row), "page_id": row["page_id"], "title": row["title"], "revision_id": str(row["revision_id"]), "permanent_url": row["permanent_url"], "html": html, "raw_sha256": sha256_text(html), "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, DatasetError) as exc:
                last = exc
                if isinstance(exc, DatasetError) and "revision drift" in str(exc):
                    break
                if isinstance(exc, urllib.error.HTTPError) and exc.code in {429, 500, 502, 503, 504}:
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        try:
                            time.sleep(min(60.0, max(0.0, float(retry_after))))
                        except ValueError:
                            pass
        raise DatasetError(f"fetch failed for {make_source_id(row)} after {self.retries + 1} attempts: {last}") from last


def fetch_sources(manifest: Path, output_dir: Path, *, hierarchy: Path = DEFAULT_HIERARCHY, force: bool = False, fetcher: Fetcher | None = None) -> int:
    rows = validate_manifest(manifest, hierarchy)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "fetch_checkpoint.json"
    done = set(json.loads(checkpoint.read_text(encoding="utf-8")).get("completed", [])) if checkpoint.exists() else set()
    fetcher = fetcher or Fetcher()
    for row in rows:
        source_id = make_source_id(row)
        target = output_dir / f"{source_id}.json"
        if source_id in done and target.exists():
            continue
        if target.exists() and not force:
            raise DatasetError(f"refusing to overwrite existing source without --force: {target}")
        payload = fetcher.fetch(row)
        target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        done.add(source_id)
        checkpoint.write_text(json.dumps({"completed": sorted(done)}, indent=2) + "\n", encoding="utf-8")
        if fetcher.min_interval:
            time.sleep(fetcher.min_interval)
    return len(done)


def chunk_sources(source_dir: Path, output: Path, *, manifest: Path = DEFAULT_MANIFEST, hierarchy: Path = DEFAULT_HIERARCHY, tokenizer: Any | None = None, tokenizer_revision: str | None = None, config: Path = DEFAULT_CONFIG) -> int:
    rows = validate_manifest(manifest, hierarchy)
    settings = load_config(config)
    if tokenizer is None:
        tokenizer = BGETokenizer(tokenizer_revision, config_path=config)
    by_source = {make_source_id(row): row for row in rows}
    chunks: list[dict[str, Any]] = []
    for source_id, row in by_source.items():
        source_path = source_dir / f"{source_id}.json"
        if not source_path.exists():
            raise DatasetError(f"missing fetched source: {source_path}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if str(source.get("revision_id")) != str(row["revision_id"]):
            raise DatasetError(f"source revision mismatch: {source_id}")
        paragraphs = clean_html(source.get("html", ""))
        if not paragraphs:
            raise DatasetError(f"empty cleaned body: {source_id}")
        for unit in chunk_paragraphs(paragraphs, tokenizer):
            chunks.append({"id": make_chunk_id(source_id, unit["section"], unit["start"], unit["end"], unit["text"]), "text": unit["text"], "token_count": unit["token_count"], "source_id": source_id, "section": unit["section"], "paragraph_start": unit["start"], "paragraph_end": unit["end"], "split": row["split"], "top": row["top"], "parent": row["parent"], "leaf": row["leaf"]})
    if output.exists():
        raise DatasetError(f"refusing to overwrite existing chunk artifact: {output}")
    jsonl_write(output, chunks)
    return len(chunks)


def _shingles(text: str, n: int = 5) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return {normalized[i:i+n] for i in range(max(0, len(normalized) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _tfidf_vectors(texts: Sequence[str], n: int = 5) -> list[dict[str, float]]:
    """Build sparse TF-IDF vectors over normalized character n-grams."""
    grams = [Counter(_shingles(text, n)) for text in texts]
    document_frequency: Counter[str] = Counter()
    for row in grams:
        document_frequency.update(row.keys())
    total = len(grams)
    vectors: list[dict[str, float]] = []
    for row in grams:
        length = sum(row.values()) or 1
        vectors.append({gram: (count / length) * (math.log((total + 1) / (document_frequency[gram] + 1)) + 1.0) for gram, count in row.items()})
    return vectors


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _validate_artifacts(manifest: Path, sources: Path, chunks: Path, report_path: Path, *, hierarchy: Path = DEFAULT_HIERARCHY, tokenizer: Any | None = None, near_threshold: float = 0.92, config: Path = DEFAULT_CONFIG, exclusions: Path | None = None) -> dict[str, Any]:
    rows = validate_manifest(manifest, hierarchy)
    settings = load_config(config)
    source_rows = normalize_source_records(sources)
    chunk_rows = gzip_jsonl_read(chunks) if chunks.suffix == ".gz" else jsonl_read(chunks)
    errors: list[str] = []
    approved_ids = {make_source_id(r): r for r in rows}
    source_ids = {str(s.get("source_id")) for s in source_rows}
    if source_ids != set(approved_ids):
        errors.append("sources do not exactly match approved manifest")
    if len(source_ids) != len(source_rows):
        errors.append("sources contain duplicate source_id values")
    for source in source_rows:
        expected = approved_ids.get(str(source.get("source_id")))
        if expected is None:
            continue
        for field in ("page_id", "title", "revision_id", "permanent_url"):
            if str(source.get(field)) != str(expected.get(field)):
                errors.append(f"source {source.get('source_id')}: {field} does not match manifest")
    chunks_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    text_hashes: dict[str, str] = {}
    for index, chunk in enumerate(chunk_rows, 1):
        source_id = str(chunk.get("source_id"))
        if source_id not in approved_ids:
            errors.append(f"chunk {index}: unknown source_id")
        expected = approved_ids.get(source_id)
        if expected is not None:
            for field in ("split", "top", "parent", "leaf"):
                if chunk.get(field) != expected.get(field):
                    errors.append(f"chunk {index}: {field} metadata does not match manifest")
        if chunk.get("id") in ids:
            errors.append(f"duplicate chunk id: {chunk.get('id')}")
        ids.add(str(chunk.get("id")))
        text = chunk.get("text")
        try:
            token_value = float(chunk.get("token_count", float("nan")))
        except (TypeError, ValueError):
            token_value = float("nan")
        if not isinstance(text, str) or not text.strip() or not math.isfinite(token_value):
            errors.append(f"chunk {index}: invalid UTF-8 text or token count")
            continue
        if not isinstance(chunk.get("token_count"), int) or isinstance(chunk.get("token_count"), bool):
            errors.append(f"chunk {index}: token_count must be an integer")
            continue
        count = int(chunk["token_count"])
        if not MIN_TOKENS <= count <= MAX_TOKENS:
            errors.append(f"chunk {index}: token count {count} outside {MIN_TOKENS}..{MAX_TOKENS}")
        if tokenizer is not None and _token_count(text, tokenizer) != count:
            errors.append(f"chunk {index}: token count is not reproducible")
        chunks_by_source[source_id].append(chunk)
        digest = sha256_text(text)
        if digest in text_hashes:
            errors.append(f"exact duplicate chunk text: {digest}")
        text_hashes[digest] = source_id
    ngram = int(settings["near_duplicate"]["ngram"])
    vectors = _tfidf_vectors([str(chunk.get("text", "")) for chunk in chunk_rows], ngram)
    for source_id in approved_ids:
        count = len(chunks_by_source[source_id])
        if not 1 <= count <= 3:
            errors.append(f"{source_id}: expected 1..3 chunks, got {count}")
    near_duplicates: list[dict[str, Any]] = []
    chunk_list = list(chunk_rows)
    for i, left in enumerate(chunk_list):
        for j in range(i + 1, len(chunk_list)):
            right = chunk_list[j]
            if left.get("source_id") == right.get("source_id"):
                continue
            score = _cosine(vectors[i], vectors[j])
            if score >= near_threshold:
                item = {"left": left.get("id"), "right": right.get("id"), "score": round(score, 6), "method": "tfidf_char_ngram", "split_boundary": left.get("split") != right.get("split")}
                near_duplicates.append(item)
                if item["split_boundary"]:
                    errors.append(f"near duplicate crosses split boundary: {left.get('id')} / {right.get('id')}")
    counts: dict[str, Counter[str]] = {key: Counter() for key in ("top", "parent", "leaf", "split")}
    for row in rows:
        for key in counts:
            counts[key][row[key]] += 1
    chunk_counts: dict[str, Counter[str]] = {key: Counter() for key in counts}
    for chunk in chunk_rows:
        for key in chunk_counts:
            chunk_counts[key][chunk.get(key)] += 1
    excluded: list[dict[str, Any]] = []
    if exclusions is not None and exclusions.exists():
        for index, item in enumerate(jsonl_read(exclusions), 1):
            reason = str(item.get("reason", "")).strip()
            if not reason:
                errors.append(f"exclusion {index}: reason is required")
            else:
                excluded.append({"page_id": item.get("page_id"), "title": item.get("title"), "reason": reason})
    report = {"schema_version": "1.0", "manifest_articles": len(rows), "source_count": len(source_rows), "chunk_count": len(chunk_rows), "errors": errors, "near_duplicates": near_duplicates, "near_duplicate_method": "tfidf_char_ngram", "article_counts": {k: dict(v) for k, v in counts.items()}, "chunk_counts": {k: dict(v) for k, v in chunk_counts.items()}, "token_distribution": {"min": min((int(c["token_count"]) for c in chunk_rows), default=None), "max": max((int(c["token_count"]) for c in chunk_rows), default=None), "mean": (sum(int(c["token_count"]) for c in chunk_rows) / len(chunk_rows) if chunk_rows else None)}, "excluded": excluded, "tokenizer": settings["tokenizer"], "config_sha256": sha256_bytes(config.read_bytes())}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise DatasetError("artifact validation failed:\n- " + "\n- ".join(errors[:30]))
    return report


def validate_artifacts(manifest: Path, sources: Path, chunks: Path, report_path: Path, *, hierarchy: Path = DEFAULT_HIERARCHY, tokenizer: Any | None = None, near_threshold: float = 0.92, config: Path = DEFAULT_CONFIG, exclusions: Path | None = None) -> dict[str, Any]:
    """Validate artifacts and persist a diagnostic report for every failure."""
    previous_mtime = report_path.stat().st_mtime_ns if report_path.exists() else None
    try:
        return _validate_artifacts(manifest, sources, chunks, report_path, hierarchy=hierarchy, tokenizer=tokenizer, near_threshold=near_threshold, config=config, exclusions=exclusions)
    except Exception as exc:
        current_mtime = report_path.stat().st_mtime_ns if report_path.exists() else None
        if current_mtime == previous_mtime:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "schema_version": "1.0",
                "manifest_articles": None,
                "source_count": None,
                "chunk_count": None,
                "errors": [str(exc)],
                "near_duplicates": [],
                "article_counts": {},
                "chunk_counts": {},
                "token_distribution": {"min": None, "max": None, "mean": None},
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def _deterministic_archive(source_dir: Path, archive_path: Path) -> None:
    import gzip
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name("." + archive_path.name + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    root_name = "wikipedia-soft-benchmark-v1"
                    for path in sorted(source_dir.rglob("*")):
                        relative = path.relative_to(source_dir)
                        info = archive.gettarinfo(str(path), arcname=str(Path(root_name) / relative))
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        if path.is_file():
                            with path.open("rb") as handle:
                                archive.addfile(info, handle)
                        else:
                            archive.addfile(info)
        temporary.replace(archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_dataset_card(settings: Mapping[str, Any]) -> str:
    tokenizer = settings["tokenizer"]
    chunking = settings["chunking"]
    return DATASET_CARD.format(model=tokenizer["model"], revision=tokenizer["revision"], minimum=chunking["min_tokens"], maximum=chunking["max_tokens"])


def package_dataset(output_dir: Path, *, manifest: Path = DEFAULT_MANIFEST, hierarchy: Path = DEFAULT_HIERARCHY, source_dir: Path, chunks: Path, config: Path = DEFAULT_CONFIG, force: bool = False, archive_path: Path | None = None, tokenizer: Any | None = None) -> Path:
    settings = load_config(config)
    validate_manifest(manifest, hierarchy)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise DatasetError(f"refusing to overwrite non-empty package directory: {output_dir}")
    archive_path = archive_path or output_dir.parent / "wikipedia-soft-benchmark-v1.tar.gz"
    if archive_path.exists() and not force:
        raise DatasetError(f"refusing to overwrite existing archive without --force: {archive_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".wikipedia-package-", dir=str(output_dir.parent)))
    committed = False
    try:
        shutil.copy2(hierarchy, staging / "hierarchy.json")
        shutil.copy2(manifest, staging / "source_manifest.jsonl")
        shutil.copy2(config, staging / "config.json")
        source_records = normalize_source_records(source_dir)
        gzip_jsonl_write(staging / "sources.jsonl.gz", source_records)
        chunk_records = jsonl_read(chunks)
        gzip_jsonl_write(staging / "chunks.jsonl.gz", chunk_records)
        attribution = []
        for source in source_records:
            title_url = urllib.parse.quote(str(source["title"]).replace(" ", "_"), safe="()/:,_-!")
            attribution.append({"source_id": source["source_id"], "title": source["title"], "page_url": "https://en.wikipedia.org/wiki/" + title_url, "history_url": "https://en.wikipedia.org/w/index.php?title=" + title_url + "&action=history", "permanent_url": source["permanent_url"], "revision_id": source["revision_id"], "license": "CC BY-SA 4.0", "modified": True})
        gzip_jsonl_write(staging / "attribution.jsonl.gz", attribution)
        (staging / "LICENSE-CC-BY-SA-4.0.txt").write_text("Creative Commons Attribution-ShareAlike 4.0 International\nhttps://creativecommons.org/licenses/by-sa/4.0/\n\nWikipedia text is redistributed under the terms above. See attribution.jsonl.gz for source history URLs and modification notices.\n", encoding="utf-8")
        (staging / "dataset_card.md").write_text(_render_dataset_card(settings), encoding="utf-8")
        report = staging / "validation_report.json"
        validate_artifacts(staging / "source_manifest.jsonl", staging / "sources.jsonl.gz", staging / "chunks.jsonl.gz", report, hierarchy=staging / "hierarchy.json", tokenizer=tokenizer or BGETokenizer(config_path=config), config=staging / "config.json")
        sums: list[str] = []
        for path in sorted(p for p in staging.iterdir() if p.name != "SHA256SUMS" and p.is_file()):
            sums.append(f"{sha256_bytes(path.read_bytes())}  {path.name}")
        (staging / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
        _deterministic_archive(staging, archive_path)
        if output_dir.exists():
            if force or not any(output_dir.iterdir()):
                shutil.rmtree(output_dir)
            else:
                raise DatasetError(f"refusing to overwrite package directory: {output_dir}")
        staging.rename(output_dir)
        committed = True
        return archive_path
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


DATASET_CARD = """# Wikipedia Soft Benchmark v1 (collection stage)

This package contains fixed-revision English Wikipedia articles arranged in a
2 x 3 x 2 hierarchy. It is intended for later source-disjoint mixture and
soft-clustering experiments; no mixtures or embeddings are included here.

Articles are split at article level (`discovery=30`, `calibration=10`,
`test=10` per leaf). HTML tables, lists, infoboxes, captions, references,
bibliographies, and navigation elements are excluded. Titles remain metadata
and are not prepended to chunk text. Chunks use `{model}` at immutable
Hugging Face revision `{revision}`, with `{minimum}--{maximum}` tokens.

Wikipedia content is licensed CC BY-SA 4.0. Each source's page URL, history
URL, permanent revision URL, and modification notice are in
`attribution.jsonl.gz`.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and validate the Wikipedia soft-clustering benchmark")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate-manifest"); p.add_argument("--allow-incomplete", action="store_true")
    p = sub.add_parser("fetch"); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--force", action="store_true"); p.add_argument("--api-endpoint", default=DEFAULT_API); p.add_argument("--user-agent", default=DEFAULT_USER_AGENT); p.add_argument("--min-interval", type=float, default=0.5)
    p = sub.add_parser("chunk"); p.add_argument("--source-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("validate"); p.add_argument("--sources", type=Path, required=True); p.add_argument("--chunks", type=Path, required=True); p.add_argument("--report", type=Path, required=True); p.add_argument("--exclusions", type=Path)
    p = sub.add_parser("package"); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--source-dir", type=Path, required=True); p.add_argument("--chunks", type=Path, required=True); p.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-manifest":
            rows = validate_manifest(args.manifest, args.hierarchy, require_complete=not args.allow_incomplete)
            print(json.dumps({"articles": len(rows), "status": "ok"}, sort_keys=True))
        elif args.command == "fetch":
            count = fetch_sources(args.manifest, args.output_dir, hierarchy=args.hierarchy, force=args.force, fetcher=Fetcher(endpoint=args.api_endpoint, user_agent=args.user_agent, min_interval=args.min_interval))
            print(json.dumps({"sources": count, "status": "ok"}, sort_keys=True))
        elif args.command == "chunk":
            count = chunk_sources(args.source_dir, args.output, manifest=args.manifest, hierarchy=args.hierarchy, config=args.config)
            print(json.dumps({"chunks": count, "status": "ok"}, sort_keys=True))
        elif args.command == "validate":
            report = validate_artifacts(args.manifest, args.sources, args.chunks, args.report, hierarchy=args.hierarchy, tokenizer=BGETokenizer(config_path=args.config), config=args.config, exclusions=args.exclusions)
            print(json.dumps({"chunks": report["chunk_count"], "status": "ok"}, sort_keys=True))
        elif args.command == "package":
            archive = package_dataset(args.output_dir, manifest=args.manifest, hierarchy=args.hierarchy, source_dir=args.source_dir, config=args.config, chunks=args.chunks, force=args.force)
            print(json.dumps({"archive": str(archive), "package": str(args.output_dir), "status": "ok"}, sort_keys=True))
        return 0
    except (DatasetError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
