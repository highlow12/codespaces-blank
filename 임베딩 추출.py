from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


DEFAULT_DATASET = "sunhaozhepy/ag_news_llm_keywords_embeddings"
DEFAULT_SPLIT = "train"
DEFAULT_EMBEDDING_FIELD = "keywords_embeddings"
DEFAULT_LABEL_FIELD = "label"
DEFAULT_PER_LABEL = 1000
DEFAULT_LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract only label and embedding from AG News precomputed embeddings.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--embedding-field", type=str, default=DEFAULT_EMBEDDING_FIELD)
    parser.add_argument("--label-field", type=str, default=DEFAULT_LABEL_FIELD)
    parser.add_argument("--per-label", type=int, default=DEFAULT_PER_LABEL)
    parser.add_argument("--label-names", nargs="+", default=DEFAULT_LABEL_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("ag_news_embeddings_4x1000.json"))
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    rng = np.random.default_rng(args.seed)

    grouped_rows: dict[str, list[dict[str, object]]] = {label_name: [] for label_name in args.label_names}
    counts: dict[str, int] = {label_name: 0 for label_name in args.label_names}

    for row in dataset:
        label_value = row[args.label_field]
        if isinstance(label_value, int):
            if label_value < 0 or label_value >= len(args.label_names):
                continue
            label_name = args.label_names[label_value]
        else:
            label_name = str(label_value)
        if label_name not in grouped_rows:
            continue

        counts[label_name] += 1
        candidates = grouped_rows[label_name]
        sample_row = {
            "tag": label_name,
            "embedding": row[args.embedding_field],
        }
        if len(candidates) < args.per_label:
            candidates.append(sample_row)
        else:
            replace_index = rng.integers(0, counts[label_name])
            if replace_index < args.per_label:
                candidates[replace_index] = sample_row

    sampled_rows: list[dict[str, object]] = []
    for label_name in args.label_names:
        candidates = grouped_rows[label_name]
        if len(candidates) < args.per_label:
            raise ValueError(f"Label '{label_name}' has only {len(candidates)} rows, need {args.per_label}")
        sampled_rows.extend(candidates[: args.per_label])

    args.output.write_text(json.dumps(sampled_rows, ensure_ascii=False), encoding="utf-8")
    print(f"완료! {len(sampled_rows)}개 행이 {args.output}에 저장되었습니다.")
    print(f"Labels: {sorted(grouped_rows)}")


if __name__ == "__main__":
    main()