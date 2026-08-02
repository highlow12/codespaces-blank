"""Build a compact DBpedia Ontology embedding dataset with Gemini.

The generated dataset intentionally keeps only three fields per record:

    class, class_hierarchy, embedding

DBpedia labels/descriptions/URIs are used while collecting and embedding
records, but they are removed from the final JSON.  The hierarchy is a controlled
three-level hierarchy: top topic -> subtopic -> DBpedia Ontology class.

The API key can be entered in GEMINI_API_KEY below or supplied with
``--api-key``.  The script sends one embedding request per record and spaces
all requests so that the configured requests-per-minute limit is not exceeded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np


# ---------------------------------------------------------------------------
# User-editable Gemini settings
# ---------------------------------------------------------------------------

GEMINI_API_KEY = "AIzaSyDqymyAn4Bj6K8W_ahUdXyM43Z0OHjOCMg"
GEMINI_MODEL = "gemini-embedding-001"
GEMINI_TASK_TYPE = "CLUSTERING"
GEMINI_OUTPUT_DIMENSIONALITY = 3072


# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------

DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
DBPEDIA_ONTOLOGY_BASE = "http://dbpedia.org/ontology/"
DBPEDIA_RESOURCE_BASE = "http://dbpedia.org/resource/"
TOP_TOPIC_COUNT = 6
SAMPLES_PER_TOP_TOPIC = 500
DEFAULT_REQUESTS_PER_MINUTE = 99
DEFAULT_DAILY_LIMIT = 1_000
DEFAULT_OUTPUT = "dbpedia_gemini_embeddings.json"
MAX_INPUT_CHARS = 8_000
CHECKPOINT_VERSION = 1
CHECKPOINT_EVERY = 25
CLASS_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class RateLimiter:
    """Space requests by at least 60 / requests_per_minute seconds."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than zero")
        self.interval = 60.0 / requests_per_minute
        self._last_request_at: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            remaining = self.interval - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a neighboring temporary file, then replace atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hierarchy(path: Path) -> dict[str, dict[str, str]]:
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Hierarchy config must be a JSON object")
    if len(config) != TOP_TOPIC_COUNT:
        raise ValueError(
            f"Hierarchy config must contain exactly {TOP_TOPIC_COUNT} top topics; "
            f"found {len(config)}"
        )

    leaves: list[str] = []
    normalized: dict[str, dict[str, str]] = {}
    for top_topic, subtopics in config.items():
        if not isinstance(top_topic, str) or not top_topic.strip():
            raise ValueError("Every top topic must have a non-empty string name")
        if not isinstance(subtopics, dict) or not 3 <= len(subtopics) <= 4:
            raise ValueError(
                f"Top topic {top_topic!r} must contain 3 or 4 subtopics"
            )

        normalized_subtopics: dict[str, str] = {}
        for subtopic, dbpedia_class in subtopics.items():
            if not isinstance(subtopic, str) or not subtopic.strip():
                raise ValueError(f"Invalid subtopic under {top_topic!r}")
            if (
                not isinstance(dbpedia_class, str)
                or not CLASS_NAME_PATTERN.fullmatch(dbpedia_class)
            ):
                raise ValueError(
                    f"Invalid DBpedia class {dbpedia_class!r} under {top_topic!r}"
                )
            normalized_subtopics[subtopic] = dbpedia_class
            leaves.append(dbpedia_class)
        normalized[top_topic] = normalized_subtopics

    duplicate_leaves = sorted(
        {leaf for leaf in leaves if leaves.count(leaf) > 1}
    )
    if duplicate_leaves:
        raise ValueError(
            "Each configured DBpedia leaf class must be unique; duplicates: "
            + ", ".join(duplicate_leaves)
        )
    return normalized


def split_count(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (index < remainder) for index in range(parts)]


def hierarchy_plan(
    hierarchy: dict[str, dict[str, str]],
) -> list[tuple[str, str, str, int]]:
    plan: list[tuple[str, str, str, int]] = []
    for top_topic, subtopics in hierarchy.items():
        targets = split_count(SAMPLES_PER_TOP_TOPIC, len(subtopics))
        for (subtopic, dbpedia_class), target in zip(
            subtopics.items(), targets, strict=True
        ):
            plan.append((top_topic, subtopic, dbpedia_class, target))
    return plan


def request_json(
    request: Request,
    *,
    timeout: float,
    retries: int,
    retry_statuses: set[int],
    operation: str,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"{operation} returned a non-object JSON response")
            return payload
        except HTTPError as error:
            last_error = error
            if error.code not in retry_statuses or attempt >= retries:
                error_body = error.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(
                    f"{operation} failed with HTTP {error.code}: {error_body}"
                ) from error
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= retries:
                raise RuntimeError(f"{operation} failed: {error}") from error

        time.sleep(min(30.0, 2.0**attempt))

    raise RuntimeError(f"{operation} failed: {last_error}")


def build_sparql_query(dbpedia_class: str, limit: int) -> str:
    class_uri = DBPEDIA_ONTOLOGY_BASE + dbpedia_class
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT DISTINCT ?resource ?label ?text WHERE {{
  ?resource rdf:type/rdfs:subClassOf* <{class_uri}> .
  ?resource rdfs:label ?label .
  FILTER (lang(?label) = "en")
  OPTIONAL {{
    ?resource dbo:abstract ?abstract .
    FILTER (lang(?abstract) = "en")
  }}
  OPTIONAL {{
    ?resource dbo:description ?description .
    FILTER (lang(?description) = "en")
  }}
  BIND (COALESCE(?abstract, ?description) AS ?text)
  FILTER (BOUND(?text))
  FILTER (STRSTARTS(STR(?resource), "{DBPEDIA_RESOURCE_BASE}"))
}}
ORDER BY ?resource
LIMIT {limit}
""".strip()


def fetch_dbpedia_candidates(
    dbpedia_class: str,
    *,
    endpoint: str,
    limit: int,
    timeout: float,
) -> list[dict[str, str]]:
    query = build_sparql_query(dbpedia_class, limit)
    query_parameters = urlencode(
        {"query": query, "format": "application/sparql-results+json"}
    )
    request = Request(
        f"{endpoint}?{query_parameters}",
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "dbpedia-gemini-embedding-dataset/1.0",
        },
        method="GET",
    )
    payload = request_json(
        request,
        timeout=timeout,
        retries=3,
        retry_statuses={429, 500, 502, 503, 504},
        operation=f"DBpedia query for {dbpedia_class}",
    )

    bindings = payload.get("results", {}).get("bindings", [])
    candidates: list[dict[str, str]] = []
    seen_resources: set[str] = set()
    for binding in bindings:
        resource = binding.get("resource", {}).get("value")
        label = binding.get("label", {}).get("value")
        text_value = binding.get("text", {}).get("value")
        if not resource or not label or not text_value or resource in seen_resources:
            continue
        seen_resources.add(resource)
        candidates.append(
            {
                "resource": resource,
                "label": " ".join(label.split()),
                "text": " ".join(text_value.split()),
            }
        )
    return candidates


def make_embedding_text(label: str, abstract: str) -> str:
    text = f"{label}. {abstract}".strip()
    if len(text) <= MAX_INPUT_CHARS:
        return text
    truncated = text[:MAX_INPUT_CHARS]
    return truncated.rsplit(" ", 1)[0]


def select_records(
    hierarchy: dict[str, dict[str, str]],
    *,
    endpoint: str,
    timeout: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    used_resources: set[str] = set()
    records: list[dict[str, Any]] = []

    for top_topic, subtopic, dbpedia_class, target in hierarchy_plan(hierarchy):
        candidate_limit = max(500, target * 5)
        candidates = fetch_dbpedia_candidates(
            dbpedia_class,
            endpoint=endpoint,
            limit=candidate_limit,
            timeout=timeout,
        )
        order = rng.permutation(len(candidates))
        selected: list[dict[str, str]] = []
        for index in order:
            candidate = candidates[int(index)]
            if candidate["resource"] in used_resources:
                continue
            selected.append(candidate)
            used_resources.add(candidate["resource"])
            if len(selected) == target:
                break

        if len(selected) < target:
            raise RuntimeError(
                f"DBpedia class {dbpedia_class!r} supplied only "
                f"{len(selected)} unique records for {target} requested. "
                "Change dbpedia_hierarchy.json or increase the query buffer."
            )

        for candidate in selected:
            records.append(
                {
                    "class": dbpedia_class,
                    "class_hierarchy": [top_topic, subtopic, dbpedia_class],
                    "resource": candidate["resource"],
                    "text": make_embedding_text(
                        candidate["label"], candidate["text"]
                    ),
                    "embedding": None,
                }
            )
        print(
            f"Selected {target:3d} records: "
            f"{top_topic} / {subtopic} / {dbpedia_class}"
        )

    expected_count = TOP_TOPIC_COUNT * SAMPLES_PER_TOP_TOPIC
    if len(records) != expected_count:
        raise RuntimeError(
            f"Internal selection error: expected {expected_count} records, "
            f"selected {len(records)}"
        )
    return records


def normalize_embedding(values: Any, expected_dimension: int) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != expected_dimension:
        raise RuntimeError(
            f"Gemini returned dimension {vector.shape}; expected "
            f"({expected_dimension},)"
        )
    if not np.isfinite(vector).all():
        raise RuntimeError("Gemini returned a non-finite embedding")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise RuntimeError("Gemini returned a zero embedding")
    return (vector / norm).astype(np.float32).tolist()


def extract_embedding_values(payload: dict[str, Any]) -> Any:
    embedding = payload.get("embedding")
    if isinstance(embedding, dict) and "values" in embedding:
        return embedding["values"]
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, dict) and "values" in first:
            return first["values"]
    raise RuntimeError("Gemini response did not contain embedding.values")


def is_quota_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "resource_exhausted",
            "quota",
            "daily limit",
            "per day",
            "rate limit",
            "too many requests",
            "http 429",
        )
    )


def embed_text(
    text: str,
    *,
    api_key: str,
    model: str,
    task_type: str,
    output_dimensionality: int,
    limiter: RateLimiter,
    timeout: float,
) -> list[float]:
    model_path = quote(model, safe="-")
    encoded_api_key = quote(api_key, safe="")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model_path}:embedContent?key={encoded_api_key}"
    )
    body = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "embedContentConfig": {
            "taskType": task_type,
            "outputDimensionality": output_dimensionality,
            "autoTruncate": True,
        },
    }
    request = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    # The limiter covers retries as well as successful requests, so retry
    # traffic cannot accidentally exceed the configured requests-per-minute.
    for attempt in range(4):
        limiter.wait()
        try:
            payload = request_json(
                request,
                timeout=timeout,
                retries=0,
                retry_statuses=set(),
                operation="Gemini embedding request",
            )
            return normalize_embedding(
                extract_embedding_values(payload), output_dimensionality
            )
        except RuntimeError as error:
            if is_quota_error(error):
                raise
            if attempt >= 3:
                raise
            time.sleep(min(30.0, 2.0**attempt))
    raise AssertionError("unreachable")


def checkpoint_path_for(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.checkpoint.json")


def hierarchy_signature(hierarchy: dict[str, dict[str, str]]) -> str:
    serialized = json.dumps(hierarchy, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_or_create_checkpoint(
    checkpoint_path: Path,
    hierarchy: dict[str, dict[str, str]],
    *,
    endpoint: str,
    timeout: float,
    seed: int,
    model: str,
    output_dimensionality: int,
) -> list[dict[str, Any]]:
    signature = hierarchy_signature(hierarchy)
    if checkpoint_path.exists():
        checkpoint = read_json(checkpoint_path)
        if (
            checkpoint.get("version") != CHECKPOINT_VERSION
            or checkpoint.get("hierarchy_signature") != signature
        ):
            raise RuntimeError(
                f"Existing checkpoint {checkpoint_path} does not match the current "
                "hierarchy settings. Move it aside before starting over."
            )
        records = checkpoint.get("records")
        if not isinstance(records, list):
            raise RuntimeError(f"Invalid checkpoint: {checkpoint_path}")

        checkpoint_settings_match = (
            checkpoint.get("model") == model
            and checkpoint.get("output_dimensionality") == output_dimensionality
        )
        if not checkpoint_settings_match:
            if any(row.get("embedding") is not None for row in records):
                raise RuntimeError(
                    f"Existing checkpoint {checkpoint_path} contains embeddings from "
                    "different model/dimension settings. Move it aside before "
                    "starting over."
                )
            # The previous run stopped before the first checkpointed embedding.
            # Its selected, de-duplicated records are safe to reuse.
            atomic_write_json(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "hierarchy_signature": signature,
                    "model": model,
                    "output_dimensionality": output_dimensionality,
                    "records": records,
                },
            )
            print("Updated the selection checkpoint for the new embedding dimension")
        print(f"Resuming {sum(row.get('embedding') is not None for row in records)} "
              f"embedded records from {checkpoint_path}")
        return records

    records = select_records(
        hierarchy,
        endpoint=endpoint,
        timeout=timeout,
        seed=seed,
    )
    atomic_write_json(
        checkpoint_path,
        {
            "version": CHECKPOINT_VERSION,
            "hierarchy_signature": signature,
            "model": model,
            "output_dimensionality": output_dimensionality,
            "records": records,
        },
    )
    return records


def final_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for record in records:
        embedding = record.get("embedding")
        if embedding is None:
            raise RuntimeError("Cannot write final output while embeddings are missing")
        compact.append(
            {
                "class": record["class"],
                "class_hierarchy": record["class_hierarchy"],
                "embedding": embedding,
            }
        )
    return compact


def print_dry_run(hierarchy: dict[str, dict[str, str]]) -> None:
    plan = hierarchy_plan(hierarchy)
    print(f"Top topics: {len(hierarchy)}")
    print(f"Records per top topic: {SAMPLES_PER_TOP_TOPIC}")
    print(f"Total records: {len(plan) and sum(item[3] for item in plan)}")
    print("Hierarchy depth: 3 (top topic -> subtopic -> DBpedia class)")
    for top_topic, subtopic, dbpedia_class, target in plan:
        print(f"  {target:3d}  {top_topic} / {subtopic} / {dbpedia_class}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3,000-record DBpedia Ontology Gemini embedding dataset."
    )
    parser.add_argument(
        "--hierarchy-config",
        type=Path,
        default=Path("dbpedia_hierarchy.json"),
        help="JSON config containing exactly 6 top topics and 3-4 subtopics each",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Final compact JSON output path",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key; when omitted, GEMINI_API_KEY above is used",
    )
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--task-type", default=GEMINI_TASK_TYPE)
    parser.add_argument(
        "--output-dimension",
        type=int,
        default=GEMINI_OUTPUT_DIMENSIONALITY,
    )
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Maximum request rate; default is 99",
    )
    parser.add_argument(
        "--daily-limit",
        type=int,
        default=DEFAULT_DAILY_LIMIT,
        help="Maximum new embeddings per run/day; default is 1000",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dbpedia-endpoint", default=DBPEDIA_SPARQL_ENDPOINT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the 6x500 plan without network/API calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing final output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hierarchy = load_hierarchy(args.hierarchy_config)
    if args.output_dimension <= 0:
        raise ValueError("--output-dimension must be greater than zero")
    if args.requests_per_minute <= 0:
        raise ValueError("--requests-per-minute must be greater than zero")
    if args.daily_limit <= 0:
        raise ValueError("--daily-limit must be greater than zero")

    if args.dry_run:
        print_dry_run(hierarchy)
        return

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"{args.output} already exists; choose another --output or pass --force"
        )

    api_key = args.api_key or GEMINI_API_KEY
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "Gemini API key is missing. Set GEMINI_API_KEY in this file or pass "
            "--api-key YOUR_KEY."
        )

    checkpoint_path = checkpoint_path_for(args.output)
    records = load_or_create_checkpoint(
        checkpoint_path,
        hierarchy,
        endpoint=args.dbpedia_endpoint,
        timeout=args.timeout,
        seed=args.seed,
        model=args.model,
        output_dimensionality=args.output_dimension,
    )
    limiter = RateLimiter(args.requests_per_minute)
    embedded_this_run = 0

    for index, record in enumerate(records, start=1):
        if record.get("embedding") is not None:
            continue
        if embedded_this_run >= args.daily_limit:
            atomic_write_json(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "hierarchy_signature": hierarchy_signature(hierarchy),
                    "model": args.model,
                    "output_dimensionality": args.output_dimension,
                    "records": records,
                },
            )
            print(
                f"Daily limit reached after {embedded_this_run} new embeddings. "
                "Run again tomorrow to resume from the checkpoint."
            )
            return
        try:
            record["embedding"] = embed_text(
                record["text"],
                api_key=api_key,
                model=args.model,
                task_type=args.task_type,
                output_dimensionality=args.output_dimension,
                limiter=limiter,
                timeout=args.timeout,
            )
        except RuntimeError as error:
            atomic_write_json(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "hierarchy_signature": hierarchy_signature(hierarchy),
                    "model": args.model,
                    "output_dimensionality": args.output_dimension,
                    "records": records,
                },
            )
            if is_quota_error(error):
                print(
                    "Gemini quota reached; progress was saved. "
                    "Run again tomorrow to resume from the checkpoint."
                )
                return
            raise
        embedded_this_run += 1
        if index % CHECKPOINT_EVERY == 0 or index == len(records):
            atomic_write_json(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "hierarchy_signature": hierarchy_signature(hierarchy),
                    "model": args.model,
                    "output_dimensionality": args.output_dimension,
                    "records": records,
                },
            )
        if index == 1 or index % CHECKPOINT_EVERY == 0 or index == len(records):
            print(f"Embedded {index}/{len(records)} records")

    compact = final_records(records)
    atomic_write_json(args.output, compact)
    checkpoint_path.unlink(missing_ok=True)

    print(f"Wrote {len(compact)} records to {args.output}")
    print("Final fields: class, class_hierarchy, embedding")
    print("All final embeddings were L2-normalized before writing")


if __name__ == "__main__":
    main()
