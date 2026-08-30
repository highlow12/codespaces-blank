"""Emit an authoritative Python HDBSCAN fixture for the JS/WASM parity audit.

This is intentionally a feature-level audit: Python owns PCA -> UMAP ->
HDBSCAN, then the Node runner feeds the *same* UMAP coordinates to the WASM
HDBSCAN provider. It therefore measures HDBSCAN extraction/membership parity
without pretending that umap-learn and umap-js are the same implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embedding_data import load_embeddings_from_json, sample_embedding_batch
from hdbscan_membership_comparison import fit_hdbscan_membership_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dataset-sample-size", type=int, default=100)
    parser.add_argument("--dataset-sample-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.input_json.name == "dbpedia_label_embeddings.json":
        raise ValueError(
            "Refusing dbpedia_label_embeddings.json; use dbpedia_gemini_embeddings.json.gz"
        )
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    source_records = len(embeddings)
    if args.dataset_sample_size is not None:
        embeddings, metadata = sample_embedding_batch(
            embeddings,
            metadata,
            sample_size=args.dataset_sample_size,
            seed=args.dataset_sample_seed,
        )
    count = len(embeddings)
    if count < 3:
        raise ValueError("at least three rows are required for the parity audit")
    if args.fast:
        result = fit_hdbscan_membership_comparison(
            embeddings,
            pca_components=min(8, embeddings.shape[1], count - 1),
            umap_components=min(8, count - 1),
            umap_n_neighbors=min(8, count - 1),
            min_cluster_size=min(5, max(2, count // 10)),
            min_samples=min(3, count - 1),
            neighbor_count=min(8, count - 1),
            seed=args.seed,
        )
    else:
        result = fit_hdbscan_membership_comparison(embeddings, seed=args.seed)
    return {
        "schemaVersion": 1,
        "contract": "hdbscan-membership-v1",
        "authority": "python-hdbscan",
        "comparisonScope": "same-python-umap-features",
        "parityClaim": False,
        "dataset": {
            "path": str(args.input_json),
            "sourceRecords": int(source_records),
            "sampledRecords": count,
            "embeddingDimensions": int(embeddings.shape[1]),
            "sampleSeed": int(args.dataset_sample_seed),
        },
        "configuration": result.configuration,
        "features": np.asarray(result.umap_features, dtype=np.float32).tolist(),
        "reference": {
            "labels": result.leaf_labels.tolist(),
            "probabilities": result.probabilities.tolist(),
            "outlierProxy": result.outlier_scores.tolist(),
            "memberships": result.native_memberships.tolist(),
        },
        "ids": [str(value) for value in metadata["id"].tolist()],
    }


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output_json), "rows": len(report["ids"])}, indent=2))


if __name__ == "__main__":
    main()
