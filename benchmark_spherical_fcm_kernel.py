"""Compare legacy Euclidean and squared-distance spherical FCM kernels."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from fcm_core import _memberships_from_distances, _spherical_fcm_once


def _legacy_euclidean_once(
    samples: np.ndarray,
    *,
    clusters: int,
    fuzzifier: float,
    max_iter: int,
    tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-optimization SFCM inner loop, retained only as a reference."""

    centers, _ = kmeans_plusplus(
        samples,
        n_clusters=clusters,
        random_state=seed,
    )
    centers = normalize(centers, norm="l2")
    memberships = _memberships_from_distances(
        euclidean_distances(samples, centers),
        m=fuzzifier,
    )
    for _ in range(1, max_iter + 1):
        previous = memberships.copy()
        centers = normalize((memberships**fuzzifier).T @ samples, norm="l2")
        memberships = _memberships_from_distances(
            euclidean_distances(samples, centers),
            m=fuzzifier,
        )
        if np.max(np.abs(memberships - previous)) < tolerance:
            break
    return centers, memberships


def _median_seconds(operation: Callable[[], object], repeats: int) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        values.append(time.perf_counter() - started)
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3_000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--fuzzifier", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.rows, args.dimensions, args.clusters, args.max_iter, args.repeats) < 1:
        raise ValueError("rows, dimensions, clusters, max-iter, and repeats must be positive")
    if args.clusters > args.rows:
        raise ValueError("clusters cannot exceed rows")
    if args.fuzzifier <= 1.0 or args.tolerance <= 0.0:
        raise ValueError("fuzzifier must exceed 1 and tolerance must be positive")

    samples = normalize(
        np.random.default_rng(args.seed).normal(
            size=(args.rows, args.dimensions),
        ),
        norm="l2",
    )
    legacy_centers, legacy_memberships = _legacy_euclidean_once(
        samples,
        clusters=args.clusters,
        fuzzifier=args.fuzzifier,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    optimized = _spherical_fcm_once(
        samples,
        args.clusters,
        m=args.fuzzifier,
        max_iter=args.max_iter,
        tol=args.tolerance,
        seed=args.seed,
    )
    np.testing.assert_allclose(
        optimized.centers,
        legacy_centers,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        optimized.memberships,
        legacy_memberships,
        rtol=1e-10,
        atol=1e-10,
    )

    legacy_seconds = _median_seconds(
        lambda: _legacy_euclidean_once(
            samples,
            clusters=args.clusters,
            fuzzifier=args.fuzzifier,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            seed=args.seed,
        ),
        args.repeats,
    )
    optimized_seconds = _median_seconds(
        lambda: _spherical_fcm_once(
            samples,
            args.clusters,
            m=args.fuzzifier,
            max_iter=args.max_iter,
            tol=args.tolerance,
            seed=args.seed,
        ),
        args.repeats,
    )
    print(
        json.dumps(
            {
                "rows": args.rows,
                "dimensions": args.dimensions,
                "clusters": args.clusters,
                "fuzzifier": args.fuzzifier,
                "repeats": args.repeats,
                "legacy_euclidean_median_seconds": legacy_seconds,
                "squared_distance_median_seconds": optimized_seconds,
                "speedup": legacy_seconds / optimized_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
