# Synthetic tag fusion benchmark

> Historical, fixed-`K=10` benchmark (2026-08-09). It is not the production
> clustering configuration; the current content-only default is documented in
> [`../../README.md`](../../README.md).

This directory contains the reproducible content-noise × tag-corruption × tag-weight sweep.

- report: `report.json`
- flat runs: `runs.csv`
- phase diagram: `phase-diagram-membership-cosine.png`

The content-only rows are the baseline. The observed additive rows are compared against that baseline by seed and content-noise cell.
