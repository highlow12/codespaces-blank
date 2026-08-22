"""Collect the human-review candidate ledger for the Wikipedia benchmark.

This is deliberately a separate collection utility.  It does not alter the
release manifest or any of the public collection commands.  The records it
writes are *not* release records: all of them are explicitly unapproved until
a person reviews them.

The Wikimedia REST API is used for discovery and page metadata.  Its ``latest``
revision is copied into the permanent ``oldid`` URL at collection time, while
the source fields below retain enough evidence to audit the automated filters.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .core import hierarchy_nodes, jsonl_write

ENDPOINT = "https://en.wikipedia.org/w/rest.php/v1"
USER_AGENT = "wikipedia-soft-benchmark/0.1 (research dataset; contact: dataset@example.invalid)"
MIN_BODY_BYTES = 2000
SEARCH_LIMIT = 100
LEAF_TARGET = 60
SPLIT_COUNTS = (("discovery", 36), ("calibration", 12), ("test", 12))


def _request(path: str, *, params: dict[str, Any] | None = None, retries: int = 4) -> dict[str, Any]:
    query = "" if not params else "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        ENDPOINT + path + query,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # respect Wikimedia throttling
            last = exc
            if attempt + 1 < retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = max(2.0, min(30.0, float(retry_after or 5.0)))
                except ValueError:
                    delay = 5.0
                time.sleep(delay)
        except Exception as exc:  # network errors are retried, not hidden
            last = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Wikipedia REST request failed: {path}: {last}")


def _search(display_name: str) -> list[dict[str, Any]]:
    result = _request("/search/page", params={"q": display_name, "limit": SEARCH_LIMIT})
    return list(result.get("pages", []))


def _page(page: dict[str, Any]) -> dict[str, Any]:
    # ``key`` is the MediaWiki-normalized title and is safe to use as the
    # REST path after URL quoting.  The response source is the exact latest
    # revision identified by ``latest.id``.
    key = str(page.get("key") or page.get("title") or "")
    return _request("/page/" + urllib.parse.quote(key, safe=""))


def _plain_excerpt(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value or ""))
    return re.sub(r"\s+", " ", value).strip()


def _checks(page: dict[str, Any]) -> dict[str, Any]:
    source = str(page.get("source") or "")
    title = str(page.get("title") or "")
    lowered = source.casefold()
    is_redirect = bool(re.match(r"^\s*#redirect\b", source, flags=re.IGNORECASE))
    is_disambiguation = bool(
        re.search(r"\{\{\s*(?:disambiguation|disambig|hndis|geodis)\b", lowered)
        or re.search(r"\bdisambiguation\s+pages\b", lowered)
    )
    is_list = bool(
        re.match(r"^(?:list|outline|index|glossary|timeline|comparison|table) of\b", title, re.I)
        or re.search(r"\{\{\s*(?:list|disambiguation)\b", lowered)
    )
    body_length_bytes = len(source.encode("utf-8"))
    namespace = 0  # /page and /search/page are main-namespace endpoints.
    article_type = "article"
    if is_redirect:
        article_type = "redirect"
    elif is_disambiguation:
        article_type = "disambiguation"
    elif is_list:
        article_type = "list"
    elif body_length_bytes < MIN_BODY_BYTES:
        article_type = "short_article"
    return {
        "namespace": namespace,
        "is_redirect": is_redirect,
        "is_disambiguation": is_disambiguation,
        "is_list": is_list,
        "article_type": article_type,
        "body_length_bytes": body_length_bytes,
        "body_length_chars": len(source),
        "passes_automatic_filters": bool(
            namespace == 0
            and not is_redirect
            and not is_disambiguation
            and not is_list
            and body_length_bytes >= MIN_BODY_BYTES
        ),
    }


def collect(output: Path) -> list[dict[str, Any]]:
    nodes = hierarchy_nodes()
    leaves = [slug for slug, node in nodes.items() if node["level"] == "leaf"]
    display = {slug: nodes[slug]["display_name"] for slug in leaves}
    # Search calls are independent and substantially faster in parallel.  The
    # page metadata requests below are also parallel but bounded to be polite.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        search_results = dict(zip(leaves, pool.map(lambda leaf: _search(display[leaf]), leaves)))

    entries: list[tuple[str, dict[str, Any], int]] = []
    for leaf in leaves:
        for rank, result in enumerate(search_results[leaf], 1):
            entries.append((leaf, result, rank))

    # Fetch a generous prefix from every search ranking.  Invalid/duplicate
    # pages are discarded, then the next ranked result fills the reserve.
    fetched: dict[tuple[str, str], dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        future_map = {
            pool.submit(_page, result): (leaf, result, rank)
            for leaf, result, rank in entries
            if result.get("key") or result.get("title")
        }
        for future in concurrent.futures.as_completed(future_map):
            leaf, result, rank = future_map[future]
            try:
                page = future.result()
            except Exception:
                continue
            key = str(page.get("key") or page.get("title") or "").casefold()
            fetched[(leaf, key)] = {"search": result, "page": page, "search_rank": rank}

    used_titles: set[str] = set()
    rows: list[dict[str, Any]] = []
    collected_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    for leaf in leaves:
        node = nodes[leaf]
        parent = node["parent"]
        top = nodes[parent]["parent"]
        accepted: list[dict[str, Any]] = []
        candidates = sorted(
            (entry for (entry_leaf, _), entry in fetched.items() if entry_leaf == leaf),
            key=lambda entry: entry["search_rank"],
        )
        for entry in candidates:
            page = entry["page"]
            title = str(page.get("title") or "").strip()
            title_key = title.casefold()
            checks = _checks(page)
            revision = page.get("latest", {}).get("id")
            page_id = page.get("id")
            if not checks["passes_automatic_filters"] or not title or not revision or not isinstance(page_id, int):
                continue
            if title_key in used_titles:
                continue
            used_titles.add(title_key)
            rank = len(accepted) + 1
            split = "discovery" if rank <= 36 else "calibration" if rank <= 48 else "test"
            permanent_url = f"https://en.wikipedia.org/w/index.php?oldid={revision}"
            evidence = _plain_excerpt(str(entry["search"].get("excerpt") or ""))
            row = {
                "page_id": page_id,
                "title": title,
                "revision_id": str(revision),
                "permanent_url": permanent_url,
                "top": top,
                "parent": parent,
                "leaf": leaf,
                "split": split,
                "candidate_rank": rank,
                "approved": False,
                "review_status": "pending",
                "reviewer": None,
                "reviewed_at": None,
                "rejection_reason": None,
                "namespace": checks["namespace"],
                "is_redirect": checks["is_redirect"],
                "is_disambiguation": checks["is_disambiguation"],
                "is_list": checks["is_list"],
                "article_type": checks["article_type"],
                "body_length_bytes": checks["body_length_bytes"],
                "body_length_chars": checks["body_length_chars"],
                "topic_relevance_basis": f"MediaWiki full-text search for {display[leaf]!r}",
                "topic_relevance_evidence": evidence,
                "automatic_checks": checks,
                "api_endpoint": ENDPOINT + "/page/" + urllib.parse.quote(str(page.get("key") or title), safe=""),
                "revision_timestamp": page.get("latest", {}).get("timestamp"),
                "collected_at": collected_at,
            }
            accepted.append(row)
            if len(accepted) == LEAF_TARGET:
                break
        if len(accepted) != LEAF_TARGET:
            raise RuntimeError(f"{leaf}: found only {len(accepted)} valid unique pages")
        rows.extend(accepted)

    jsonl_write(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("candidate_manifest.jsonl"))
    args = parser.parse_args()
    rows = collect(args.output)
    print(f"wrote {len(rows)} unapproved candidate rows to {args.output}")


if __name__ == "__main__":
    main()
