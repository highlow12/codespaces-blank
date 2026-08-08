"""Embed the unique ground-truth DBpedia class labels separately.

The input document embeddings are not used as model input.  Each unique
``class`` value is sent to Gemini exactly once, and the output keeps the
corresponding hierarchy only as metadata for later alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dbpedia_gemini_embeddings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_OUTPUT_DIMENSIONALITY,
    GEMINI_TASK_TYPE,
    RateLimiter,
    atomic_write_json,
    embed_text,
)


def load_unique_labels(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty JSON array in {input_path}")

    labels: dict[str, list[str]] = {}
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} is not a JSON object")
        label = row.get("class")
        hierarchy = row.get("class_hierarchy")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Record {index} has no valid class label")
        if (
            not isinstance(hierarchy, list)
            or len(hierarchy) != 3
            or not all(isinstance(item, str) and item.strip() for item in hierarchy)
            or hierarchy[-1] != label
        ):
            raise ValueError(f"Record {index} has an invalid class_hierarchy")

        previous = labels.setdefault(label, hierarchy)
        if previous != hierarchy:
            raise ValueError(f"Class {label!r} has inconsistent hierarchies")

    return [
        {"label": label, "class_hierarchy": labels[label]}
        for label in sorted(labels)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one Gemini embedding for each ground-truth DBpedia class label."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dbpedia_gemini_embeddings.json.gz"),
        help="Completed document embedding dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dbpedia_label_embeddings.json"),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=GEMINI_MODEL)
    parser.add_argument("--task-type", default=GEMINI_TASK_TYPE)
    parser.add_argument(
        "--output-dimension",
        type=int,
        default=GEMINI_OUTPUT_DIMENSIONALITY,
    )
    parser.add_argument("--requests-per-minute", type=float, default=99.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"{args.output} already exists; pass --force to replace it"
        )
    if args.output_dimension <= 0:
        raise ValueError("--output-dimension must be greater than zero")

    api_key = args.api_key or GEMINI_API_KEY
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Gemini API key is missing")

    labels = load_unique_labels(args.input)
    limiter = RateLimiter(args.requests_per_minute)
    records: list[dict[str, Any]] = []

    for index, item in enumerate(labels, start=1):
        label = item["label"]
        records.append(
            {
                "label": label,
                "class_hierarchy": item["class_hierarchy"],
                "embedding": embed_text(
                    label,
                    api_key=api_key,
                    model=args.model,
                    task_type=args.task_type,
                    output_dimensionality=args.output_dimension,
                    limiter=limiter,
                    timeout=args.timeout,
                ),
            }
        )
        print(f"Embedded label {index}/{len(labels)}: {label}")

    atomic_write_json(args.output, records)
    print(f"Wrote {len(records)} unique label embeddings to {args.output}")


if __name__ == "__main__":
    main()
