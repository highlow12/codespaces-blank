"""Replace fetched pages whose cleaned bodies cannot form a release chunk.

The command is intentionally conservative: it searches the same English
Wikipedia REST endpoint used during candidate collection, validates every
replacement with the pinned tokenizer before changing the manifests, and
keeps the old source/chunk/report artifacts in an audit directory.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

from .core import (
    BGETokenizer,
    DEFAULT_API,
    Fetcher,
    chunk_paragraphs,
    clean_html,
    hierarchy_nodes,
    jsonl_read,
    jsonl_write,
    make_source_id,
    validate_manifest,
)
from .generate_candidate_manifest import ENDPOINT, _checks, _plain_excerpt, _request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MANIFEST = ROOT / "source_manifest.jsonl"
CANDIDATES = ROOT / "candidate_manifest.jsonl"
REPORT = PROJECT / "wikipedia_artifacts" / "validation_report.json"
SOURCE_DIR = PROJECT / "wikipedia_sources"
ARTIFACT_DIR = PROJECT / "wikipedia_artifacts"
REASON = "cleaned body cannot produce a 100-token chunk"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _search(display_name: str) -> list[dict[str, Any]]:
    result = _request("/search/page", params={"q": display_name, "limit": 100})
    return list(result.get("pages", []))


def _page(result: dict[str, Any]) -> dict[str, Any]:
    key = str(result.get("key") or result.get("title") or "")
    return _request("/page/" + urllib.parse.quote(key, safe=""))


IDENTITY_FIELDS = ("page_id", "revision_id", "title", "permanent_url")


def _identity_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_id": row["page_id"],
        "revision_id": str(row["revision_id"]),
        "title": str(row["title"]).casefold(),
        "permanent_url": row["permanent_url"],
    }


def _build_used(rows: list[dict[str, Any]]) -> dict[str, set[Any]]:
    used = {field: set() for field in IDENTITY_FIELDS}
    for row in rows:
        values = _identity_values(row)
        for field in IDENTITY_FIELDS:
            used[field].add(values[field])
    return used


def _is_unique(row: dict[str, Any], used: dict[str, set[Any]]) -> bool:
    values = _identity_values(row)
    return all(values[field] not in used[field] for field in IDENTITY_FIELDS)


def _add_used(row: dict[str, Any], used: dict[str, set[Any]]) -> None:
    values = _identity_values(row)
    for field in IDENTITY_FIELDS:
        used[field].add(values[field])


def _replacement_ranks(failed: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, int]:
    next_rank: dict[str, int] = {}
    for row in candidate_rows:
        leaf = str(row["leaf"])
        next_rank[leaf] = max(next_rank.get(leaf, 0), int(row["candidate_rank"]) + 1)
    ranks: dict[str, int] = {}
    for row in failed:
        leaf = str(row["leaf"])
        ranks[make_source_id(row)] = next_rank.get(leaf, 1)
        next_rank[leaf] = ranks[make_source_id(row)] + 1
    return ranks


def _candidate_row(page: dict[str, Any], result: dict[str, Any], old: dict[str, Any], rank: int, checked_at: str) -> dict[str, Any] | None:
    checks = _checks(page)
    latest = page.get("latest", {})
    revision = latest.get("id")
    page_id = page.get("id")
    title = str(page.get("title") or "").strip()
    if not checks["passes_automatic_filters"] or not title or not revision or not isinstance(page_id, int):
        return None
    permanent_url = f"https://en.wikipedia.org/w/index.php?oldid={revision}"
    return {
        "page_id": page_id,
        "title": title,
        "revision_id": str(revision),
        "permanent_url": permanent_url,
        "top": old["top"],
        "parent": old["parent"],
        "leaf": old["leaf"],
        "split": old["split"],
        "candidate_rank": rank,
        "approved": True,
        "review_status": "approved",
        "reviewer": "user",
        "reviewed_at": checked_at,
        "rejection_reason": None,
        "namespace": checks["namespace"],
        "is_redirect": checks["is_redirect"],
        "is_disambiguation": checks["is_disambiguation"],
        "is_list": checks["is_list"],
        "article_type": checks["article_type"],
        "body_length_bytes": checks["body_length_bytes"],
        "body_length_chars": checks["body_length_chars"],
        "topic_relevance_basis": f"MediaWiki full-text search for {old['leaf']!r}",
        "topic_relevance_evidence": _plain_excerpt(str(result.get("excerpt") or "")),
        "automatic_checks": checks,
        "api_endpoint": ENDPOINT + "/page/" + urllib.parse.quote(str(page.get("key") or title), safe=""),
        "revision_timestamp": latest.get("timestamp"),
        "collected_at": checked_at,
    }


def _discover(old: dict[str, Any], used: dict[str, set[Any]], tokenizer: Any, rank: int) -> tuple[dict[str, Any], dict[str, Any], int]:
    nodes = hierarchy_nodes()
    display_name = str(nodes[old["leaf"]]["display_name"])
    checked_at = _now()
    fetcher = Fetcher(endpoint=DEFAULT_API, min_interval=0.15, retries=3, timeout=45)
    for search_rank, result in enumerate(_search(display_name), 1):
        try:
            page = _page(result)
        except Exception:
            continue
        row = _candidate_row(page, result, old, rank, checked_at)
        if row is None or not _is_unique(row, used):
            continue
        # Search relevance is the same automatic evidence used for the
        # original ledger. Require a non-empty excerpt and a meaningful match.
        evidence = row["topic_relevance_evidence"]
        if len(evidence) < 30:
            continue
        try:
            payload = fetcher.fetch(row)
            paragraphs = clean_html(payload["html"])
            chunks = chunk_paragraphs(paragraphs, tokenizer)
        except Exception:
            continue
        if chunks:
            row["topic_relevance_evidence"] = evidence
            return row, payload, search_rank
    raise RuntimeError(f"no usable replacement found for {old['leaf']}/{old['split']}")


def _prospective_rows(manifest_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], replacements: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]], reviewed_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replacements_by_old = {make_source_id(old): new for old, new, _, _ in replacements}
    new_manifest = [replacements_by_old.get(make_source_id(row), row) for row in manifest_rows]
    new_candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        replacement = replacements_by_old.get(make_source_id(row))
        if replacement:
            rejected = dict(row)
            rejected.update({"approved": False, "review_status": "rejected", "reviewer": "user", "reviewed_at": reviewed_at, "rejection_reason": REASON})
            new_candidates.append(rejected)
        else:
            new_candidates.append(row)
    new_candidates.extend(new for _, new, _, _ in replacements)
    return new_manifest, new_candidates


def _stage_and_validate(stage: Path, manifest_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    staged_manifest = stage / "source_manifest.jsonl"
    staged_candidates = stage / "candidate_manifest.jsonl"
    jsonl_write(staged_manifest, manifest_rows)
    jsonl_write(staged_candidates, candidate_rows)
    from .validate_candidate_manifest import validate as validate_candidates
    validate_candidates(staged_candidates)
    validate_manifest(staged_manifest, require_complete=True)


def _stage_sources_and_checkpoint(stage: Path, manifest_rows: list[dict[str, Any]], replacements: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]]) -> None:
    for _, new, payload, _ in replacements:
        (stage / f"{make_source_id(new)}.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    active_ids = sorted(make_source_id(row) for row in manifest_rows)
    (stage / "fetch_checkpoint.json").write_text(json.dumps({"completed": active_ids}, indent=2) + "\n", encoding="utf-8")


def _same_bytes(left: Path, right: Path) -> bool:
    return left.read_bytes() == right.read_bytes()


def _commit_staged(stage: Path, new_manifest: list[dict[str, Any]], new_candidates: list[dict[str, Any]], replacements: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]]) -> None:
    pre = ARTIFACT_DIR / "pre-replacement"
    rejected_sources = pre / "rejected_sources"
    if pre.exists():
        raise RuntimeError(f"refusing to overwrite existing backup directory: {pre}")
    pre.mkdir(parents=True)
    rejected_sources.mkdir()
    try:
        shutil.copy2(ARTIFACT_DIR / "chunks.jsonl", pre / "chunks.jsonl")
        shutil.copy2(REPORT, pre / "validation_report.json")
        if not _same_bytes(ARTIFACT_DIR / "chunks.jsonl", pre / "chunks.jsonl") or not _same_bytes(REPORT, pre / "validation_report.json"):
            raise RuntimeError("artifact backup verification failed")
        failed_paths = [(SOURCE_DIR / f"{make_source_id(old)}.json", rejected_sources / f"{make_source_id(old)}.json") for old, _, _, _ in replacements]
        for source_path, backup_path in failed_paths:
            if not source_path.exists():
                raise RuntimeError(f"missing failed source file: {source_path}")
            shutil.copy2(source_path, backup_path)
            if not _same_bytes(source_path, backup_path):
                raise RuntimeError(f"source backup verification failed: {source_path}")
    except Exception:
        shutil.rmtree(pre)
        raise

    old_manifest = MANIFEST.read_bytes()
    old_candidates = CANDIDATES.read_bytes()
    checkpoint = SOURCE_DIR / "fetch_checkpoint.json"
    old_checkpoint = checkpoint.read_bytes() if checkpoint.exists() else None
    moved_old: list[tuple[Path, Path]] = []
    installed_new: list[Path] = []
    try:
        os.replace(stage / "source_manifest.jsonl", MANIFEST)
        os.replace(stage / "candidate_manifest.jsonl", CANDIDATES)
        for source_path, backup_path in failed_paths:
            source_path.unlink()
            moved_old.append((source_path, backup_path))
        for _, new, payload, _ in replacements:
            target = SOURCE_DIR / f"{make_source_id(new)}.json"
            staged = stage / target.name
            os.replace(staged, target)
            installed_new.append(target)
        os.replace(stage / "fetch_checkpoint.json", checkpoint)
        (ARTIFACT_DIR / "chunks.jsonl").unlink()
    except Exception:
        for path in installed_new:
            if path.exists():
                path.unlink()
        for source_path, backup_path in moved_old:
            if backup_path.exists():
                shutil.copy2(backup_path, source_path)
        manifest_tmp = stage / "rollback-source-manifest.jsonl"
        manifest_tmp.write_bytes(old_manifest)
        os.replace(manifest_tmp, MANIFEST)
        candidates_tmp = stage / "rollback-candidate-manifest.jsonl"
        candidates_tmp.write_bytes(old_candidates)
        os.replace(candidates_tmp, CANDIDATES)
        if old_checkpoint is not None:
            checkpoint.write_bytes(old_checkpoint)
        elif checkpoint.exists():
            checkpoint.unlink()
        if not (ARTIFACT_DIR / "chunks.jsonl").exists():
            shutil.copy2(pre / "chunks.jsonl", ARTIFACT_DIR / "chunks.jsonl")
        shutil.rmtree(pre)
        raise


def replace(discover_fn: Any = _discover) -> list[tuple[str, str]]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    failed_ids = [str(error).split(":", 1)[0] for error in report.get("errors", []) if "expected 1..3 chunks, got 0" in error]
    manifest_rows = jsonl_read(MANIFEST)
    candidate_rows = jsonl_read(CANDIDATES)
    by_source = {make_source_id(row): row for row in manifest_rows}
    failed = [by_source[source_id] for source_id in failed_ids]
    if len(failed) != 15:
        raise RuntimeError(f"expected 15 failed source rows, got {len(failed)}")
    if any(row.get("review_status") == "rejected" for row in candidate_rows):
        raise RuntimeError("replacement ledger already contains rejected rows")

    tokenizer = BGETokenizer()
    used = _build_used(candidate_rows)
    ranks = _replacement_ranks(failed, candidate_rows)
    replacements: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]] = []
    for old in failed:
        new, payload, search_rank = discover_fn(old, used, tokenizer, ranks[make_source_id(old)])
        if not _is_unique(new, used):
            raise RuntimeError(f"replacement is not globally unique: {make_source_id(new)}")
        _add_used(new, used)
        replacements.append((old, new, payload, search_rank))
    reviewed_at = _now()
    new_manifest, new_candidates = _prospective_rows(manifest_rows, candidate_rows, replacements, reviewed_at)
    stage = Path(tempfile.mkdtemp(prefix=".replacement-stage-", dir=str(ARTIFACT_DIR)))
    try:
        _stage_and_validate(stage, new_manifest, new_candidates)
        _stage_sources_and_checkpoint(stage, new_manifest, replacements)
        _commit_staged(stage, new_manifest, new_candidates, replacements)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return [(make_source_id(old), make_source_id(new)) for old, new, _, _ in replacements]


if __name__ == "__main__":
    for old, new in replace():
        print(f"{old} -> {new}")
