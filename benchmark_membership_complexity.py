"""Compare the quadratic and linear FCM membership formulas."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import numpy as np

from fuzzy_cmeans import memberships_from_squared_dissimilarities


def _pairwise_ratio_reference(values: np.ndarray, *, m: float) -> np.ndarray:
    exponent = 1.0 / (m - 1.0)
    ratios = (values[:, :, None] / values[:, None, :]) ** exponent
    return 1.0 / ratios.sum(axis=2)


def _median_runtime(
    operation: Callable[[], np.ndarray],
    *,
    repeats: int,
) -> float:
    durations = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - started_at)
    return float(statistics.median(durations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--fuzzifier", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rows < 1 or args.clusters < 1 or args.repeats < 1:
        raise ValueError("rows, clusters, and repeats must be positive")
    if args.fuzzifier <= 1.0:
        raise ValueError("fuzzifier must be greater than 1")

    rng = np.random.default_rng(args.seed)
    values = rng.uniform(
        1e-3,
        4.0,
        size=(args.rows, args.clusters),
    )
    reference = _pairwise_ratio_reference(values, m=args.fuzzifier)
    optimized = memberships_from_squared_dissimilarities(
        values,
        m=args.fuzzifier,
    )
    np.testing.assert_allclose(optimized, reference, rtol=1e-12, atol=1e-12)

    pairwise_seconds = _median_runtime(
        lambda: _pairwise_ratio_reference(values, m=args.fuzzifier),
        repeats=args.repeats,
    )
    linear_seconds = _median_runtime(
        lambda: memberships_from_squared_dissimilarities(
            values,
            m=args.fuzzifier,
        ),
        repeats=args.repeats,
    )
    item_size = int(values.dtype.itemsize)
    print(
        json.dumps(
            {
                "rows": args.rows,
                "clusters": args.clusters,
                "fuzzifier": args.fuzzifier,
                "repeats": args.repeats,
                "pairwise_ratio_median_sec": pairwise_seconds,
                "linear_inverse_power_median_sec": linear_seconds,
                "speedup": pairwise_seconds / linear_seconds,
                "pairwise_ratio_elements": args.rows
                * args.clusters
                * args.clusters,
                "linear_working_elements": args.rows * args.clusters,
                "pairwise_ratio_bytes": args.rows
                * args.clusters
                * args.clusters
                * item_size,
                "linear_working_bytes": args.rows * args.clusters * item_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
