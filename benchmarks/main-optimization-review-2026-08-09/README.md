# Main optimization review — 2026-08-09

> Historical comparison: “current” in this directory means commit `0c4613a`,
> not the current workspace. See [`../../README.md`](../../README.md) for the
> active configuration and later benchmark results.

This directory contains the raw output of a 64-run comparison between
pre-optimization commit `a10d936` and main commit `0c4613a`.

- `report.json`: configuration, synthetic case definitions, aggregate performance,
  semantic comparisons, and every run
- `runs.csv`: flat per-run timing, RSS, state size, topology, and quality metrics

Headline p50 results:

| rows | baseline JSON | current JSON | current cache |
|---:|---:|---:|---:|
| 1,000 | 30.196 s | 15.501 s | 9.181 s |
| 3,000 | 73.360 s | 34.102 s | 28.422 s |

The deterministic synthetic generators and complete orchestration command live in
`benchmark_main_optimization_review.py`. The Korean analysis and limitations are in
`.mds/MAIN_OPTIMIZATION_REVIEW.md`.
