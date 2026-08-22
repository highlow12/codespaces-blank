"""Validate the Terra candidate ledger and its promoted release rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from .core import DatasetError, hierarchy_nodes, jsonl_read
from .generate_candidate_manifest import ENDPOINT, USER_AGENT, _request

EXPECTED_APPROVED = 720
SPLIT_COUNTS = {"discovery": 36, "calibration": 12, "test": 12}


def validate(path: Path, *, check_api: bool = False) -> list[dict[str, Any]]:
    rows = jsonl_read(path)
    nodes = hierarchy_nodes()
    errors: list[str] = []
    approved_rows = [row for row in rows if row.get("approved") is True and row.get("review_status") == "approved"]
    rejected_rows = [row for row in rows if row.get("approved") is False and row.get("review_status") == "rejected"]
    if len(approved_rows) != EXPECTED_APPROVED:
        errors.append(f"expected {EXPECTED_APPROVED} approved rows, got {len(approved_rows)}")
    if len(rows) < EXPECTED_APPROVED or len(rows) != len(approved_rows) + len(rejected_rows):
        errors.append("ledger rows must have an approved or rejected status")
    required = {
        "page_id", "title", "revision_id", "permanent_url", "top", "parent", "leaf", "split",
        "candidate_rank", "approved", "review_status", "reviewer", "reviewed_at", "rejection_reason",
        "namespace", "is_redirect", "is_disambiguation", "is_list", "article_type",
        "body_length_bytes", "topic_relevance_basis", "topic_relevance_evidence", "automatic_checks",
    }
    seen = {key: set() for key in ("page_id", "revision_id", "title", "permanent_url")}
    counts: Counter[tuple[str, str]] = Counter()
    for number, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            errors.append(f"row {number}: missing {sorted(missing)}")
            continue
        leaf = row["leaf"]
        if leaf not in nodes or nodes[leaf].get("level") != "leaf":
            errors.append(f"row {number}: invalid leaf")
        else:
            parent = nodes[leaf]["parent"]
            top = nodes[parent]["parent"]
            if row["parent"] != parent or row["top"] != top:
                errors.append(f"row {number}: hierarchy mismatch")
        if row["split"] not in SPLIT_COUNTS:
            errors.append(f"row {number}: invalid split")
        counts[(str(leaf), str(row["split"]))] += 1
        revision = str(row["revision_id"])
        if row["approved"] is True and row["review_status"] == "approved":
            if not row["reviewer"] or not row["reviewed_at"] or row["rejection_reason"] is not None:
                errors.append(f"row {number}: promoted review fields are invalid")
        elif row["approved"] is False and row["review_status"] == "rejected":
            if not row["reviewer"] or not row["reviewed_at"] or not isinstance(row["rejection_reason"], str) or not row["rejection_reason"].strip():
                errors.append(f"row {number}: rejected review fields are invalid")
        else:
            errors.append(f"row {number}: invalid approved/review_status combination")
        if row["namespace"] != 0 or row["is_redirect"] or row["is_disambiguation"] or row["is_list"] or row["article_type"] != "article":
            errors.append(f"row {number}: automatic article filter failed")
        if not isinstance(row["body_length_bytes"], int) or row["body_length_bytes"] < 2000:
            errors.append(f"row {number}: article is too short")
        if f"oldid={revision}" not in row["permanent_url"]:
            errors.append(f"row {number}: revision and permanent URL disagree")
        seen["page_id"].add(row["page_id"])
        seen["revision_id"].add(revision)
        seen["title"].add(str(row["title"]).casefold())
        seen["permanent_url"].add(row["permanent_url"])
    for key, values in seen.items():
        if len(values) != len(rows):
            errors.append(f"duplicate {key} values")
    leaves = [slug for slug, node in nodes.items() if node["level"] == "leaf"]
    approved_counts: Counter[tuple[str, str]] = Counter(
        (str(row.get("leaf")), str(row.get("split"))) for row in approved_rows
    )
    for leaf in leaves:
        for split, expected in SPLIT_COUNTS.items():
            if approved_counts[(leaf, split)] != expected:
                errors.append(f"{leaf}/{split}: expected {expected} approved, got {approved_counts[(leaf, split)]}")
    if check_api and not errors:
        def check(row: dict[str, Any]) -> str | None:
            data = _request(f"/revision/{row['revision_id']}")
            page = data.get("page", {})
            if str(data.get("id")) != str(row["revision_id"]):
                return "revision id mismatch"
            if page.get("id") != row["page_id"] or page.get("title") != row["title"]:
                return "page id/title mismatch"
            if data.get("size") != row["body_length_bytes"]:
                return "revision body size mismatch"
            expected_key = str(row["title"]).replace(" ", "_")
            if page.get("key") != expected_key:
                return "normalized title mismatch"
            return None
        # The public REST service throttles bursts; keep verification bounded.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for number, result in enumerate(pool.map(check, rows), 1):
                if result:
                    errors.append(f"row {number}: {result}")
    if errors:
        raise DatasetError("candidate manifest validation failed:\n- " + "\n- ".join(errors))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).with_name("candidate_manifest.jsonl"))
    parser.add_argument("--api", action="store_true", help="recheck every fixed revision through the REST API")
    args = parser.parse_args()
    rows = validate(args.path, check_api=args.api)
    print(f"validated {len(rows)} ledger rows ({len([r for r in rows if r['approved']])} approved)")


if __name__ == "__main__":
    main()
