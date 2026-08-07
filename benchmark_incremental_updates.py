"""Measure fit and incremental-update costs for the hierarchical engine.

The benchmark intentionally reports both the ordinary update and a forced
membership-refresh update.  It uses the same persisted state path as the CLI
so the state-size measurement includes the compact contribution ledger and
batch replay history.
"""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from embedding_data import load_embeddings_from_json, sample_embedding_batch
from incremental_clustering import (
    fit_incremental_state,
    save_state,
    update_incremental_state,
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  The supported environment is
    # Linux, but keeping the conversion here makes the report less surprising.
    return value * 1024 if value < 10_000_000 else value


def _timed(operation: Any) -> tuple[Any, float, int]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started, _peak_rss_bytes()


def _make_update_batch(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    *,
    update_size: int,
    new_start: int,
    replacement_start: int,
    noise: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    if update_size < 2:
        raise ValueError("update_size must be at least 2")
    new_count = update_size // 2
    replacement_count = update_size - new_count
    if new_start + new_count > len(embeddings):
        raise ValueError("not enough rows for the requested new batch")
    if replacement_start + replacement_count > len(embeddings):
        raise ValueError("not enough rows for the requested replacement batch")
    new_slice = slice(new_start, new_start + new_count)
    replacement_slice = slice(
        replacement_start,
        replacement_start + replacement_count,
    )
    new_embeddings = embeddings[new_slice].copy()
    replacement_embeddings = embeddings[replacement_slice].copy()
    if noise:
        replacement_embeddings[:, 0] += noise
    update_embeddings = np.vstack([new_embeddings, replacement_embeddings])
    new_metadata = metadata.iloc[new_slice].copy()
    replacement_metadata = metadata.iloc[replacement_slice].copy()
    new_metadata["incremental_operation"] = "new"
    replacement_metadata["incremental_operation"] = "modified"
    return (
        update_embeddings,
        pd.concat([new_metadata, replacement_metadata], ignore_index=True),
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    embeddings, metadata = load_embeddings_from_json(args.input_json)
    if args.dataset_sample_size is not None:
        embeddings, metadata = sample_embedding_batch(
            embeddings,
            metadata,
            sample_size=args.dataset_sample_size,
            seed=(
                args.dataset_sample_seed
                if args.dataset_sample_seed is not None
                else args.seed
            ),
        )
    update_size = min(args.update_size, len(embeddings) // 2)
    if update_size < 2:
        raise ValueError("the benchmark needs at least four input rows")

    initial_embeddings = embeddings[update_size:]
    initial_metadata = metadata.iloc[update_size:].reset_index(drop=True)
    regular_batch = _make_update_batch(
        embeddings,
        metadata,
        update_size=update_size,
        new_start=0,
        replacement_start=update_size,
        noise=args.modification_noise,
    )
    refresh_batch = _make_update_batch(
        embeddings,
        metadata,
        update_size=update_size,
        new_start=0,
        replacement_start=update_size + update_size // 2,
        noise=args.modification_noise,
    )

    state, fit_seconds, fit_rss = _timed(
        lambda: fit_incremental_state(
            initial_embeddings,
            initial_metadata,
            max_depth=args.max_depth,
            min_node_size=args.min_node_size,
            min_child_size=args.min_child_size,
            max_clusters=args.max_clusters,
            pca_components=args.pca_components,
            seed=args.seed,
            center_updates_before_membership_refresh=args.refresh_interval,
            fast_mode=args.fast,
            fit_visualization=True,
        )
    )
    regular_result, regular_seconds, regular_rss = _timed(
        lambda: update_incremental_state(
            state,
            *regular_batch,
            batch_id="benchmark-regular",
        )
    )
    updated_state, regular_summary = regular_result
    updated_state.config["center_updates_before_membership_refresh"] = 1
    refresh_result, refresh_seconds, refresh_rss = _timed(
        lambda: update_incremental_state(
            updated_state,
            *refresh_batch,
            batch_id="benchmark-refresh",
        )
    )
    refreshed_state, refresh_summary = refresh_result

    with tempfile.TemporaryDirectory(prefix="incremental-benchmark-") as directory:
        state_path = Path(directory) / "benchmark.state.pkl"
        save_state(refreshed_state, state_path)
        state_size = state_path.stat().st_size

    return {
        "input_samples": int(len(embeddings)),
        "initial_samples": int(len(initial_embeddings)),
        "update_samples": int(update_size),
        "embedding_dimensions": int(embeddings.shape[1]),
        "fit_seconds": fit_seconds,
        "regular_update_seconds": regular_seconds,
        "refresh_update_seconds": refresh_seconds,
        "regular_membership_refresh_samples": int(
            regular_summary.get("membership_refresh_sample_count", 0)
        ),
        "refresh_membership_refresh_samples": int(
            refresh_summary.get("membership_refresh_sample_count", 0)
        ),
        "refresh_membership_refresh_skipped": int(
            refresh_summary.get("membership_refresh_skipped_count", 0)
        ),
        "state_size_bytes": int(state_size),
        "peak_rss_bytes": int(max(fit_rss, regular_rss, refresh_rss)),
        "peak_rss_after_fit_bytes": int(fit_rss),
        "peak_rss_after_regular_update_bytes": int(regular_rss),
        "peak_rss_after_refresh_update_bytes": int(refresh_rss),
        "state_generation": int(refreshed_state.config["state_generation"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--dataset-sample-size", type=int, default=None)
    parser.add_argument("--dataset-sample-seed", type=int, default=None)
    parser.add_argument("--update-size", type=int, default=10)
    parser.add_argument("--modification-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-node-size", type=int, default=30)
    parser.add_argument("--min-child-size", type=int, default=10)
    parser.add_argument("--max-clusters", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--refresh-interval", type=int, default=10)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_benchmark(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
