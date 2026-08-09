"""Run the synthetic content/tag fusion sweep and write reproducible reports.

The default matrix follows SYNTHETIC_TAG_FUSION_EXPERIMENT_PLAN.md.  Use
``--smoke`` for a small end-to-end check before launching the full sweep.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from synthetic_tag_fusion import (
    SyntheticTagConfig,
    cluster_fusion_dataset,
    generate_synthetic_tag_dataset,
    shuffle_tag_embeddings,
)


DEFAULT_CONTENT_NOISE = (0.05, 0.10, 0.20, 0.30, 0.40)
DEFAULT_CORRUPTION_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_TAG_WEIGHTS = (0.25, 0.5, 1.0, 2.0)
DEFAULT_SEEDS = (42, 43, 44)


def _float_list(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("at least one sweep value is required")
    return result


def _run_row(
    dataset: Any,
    *,
    seed: int,
    tag_source: str,
    variant: str,
    tag_embeddings: np.ndarray | None,
    tag_weight: float,
    pca_components: int,
    n_init: int,
    max_iter: int,
) -> dict[str, Any]:
    result = cluster_fusion_dataset(
        dataset,
        variant=variant,
        tag_embeddings=tag_embeddings,
        tag_source=tag_source,
        tag_weight=tag_weight,
        pca_components=pca_components,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
    )
    row: dict[str, Any] = {
        "seed": int(seed),
        "n_samples": int(len(dataset.content_embeddings)),
        "n_roots": int(dataset.true_memberships.shape[1]),
        "embedding_dim": int(dataset.content_embeddings.shape[1]),
        "content_noise": float(dataset.config.content_noise),
        "tag_corruption": float(dataset.config.tag_corruption),
        "tag_source": tag_source,
        "variant": variant,
        "tag_weight": float(tag_weight),
        "pca_components": int(result.pca_components),
        "corrupted_rows": int(dataset.corruption_flags["is_corrupted"].sum()),
        "boundary_count": int(dataset.metadata["is_boundary"].sum()),
    }
    row.update(result.metrics)
    return row


def _add_baseline_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: Any,
    seed: int,
    content_result: dict[str, Any],
    corruption_levels: tuple[float, ...],
) -> None:
    """Reuse a content-only fit for every tag-corruption cell."""

    for corruption in corruption_levels:
        row = dict(content_result)
        row["tag_corruption"] = float(corruption)
        row["corrupted_rows"] = int(
            dataset.corruption_flags["is_corrupted"].sum()
        )
        rows.append(row)


def _phase_delta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame["variant"] == "content_only"].copy()
    baseline = baseline[
        ["seed", "content_noise", "tag_corruption", "membership_cosine"]
    ].rename(columns={"membership_cosine": "baseline_membership_cosine"})
    candidates = frame[
        (frame["variant"] == "additive")
        & (frame["tag_source"] == "observed")
    ].copy()
    merged = candidates.merge(
        baseline,
        on=["seed", "content_noise", "tag_corruption"],
        how="left",
    )
    merged["delta_membership_cosine"] = (
        merged["membership_cosine"] - merged["baseline_membership_cosine"]
    )
    return merged


def _write_phase_plot(frame: pd.DataFrame, output_dir: Path) -> str | None:
    phase = _phase_delta_frame(frame)
    if phase.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    weights = sorted(float(value) for value in phase["tag_weight"].unique())
    figure, axes = plt.subplots(
        1,
        len(weights),
        figsize=(5.0 * len(weights), 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes[0]
    image = None
    for axis, weight in zip(axes_flat, weights, strict=True):
        subset = phase[np.isclose(phase["tag_weight"], weight)]
        pivot = subset.pivot_table(
            index="content_noise",
            columns="tag_corruption",
            values="delta_membership_cosine",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        scale = max(
            abs(float(np.nanmin(pivot))),
            abs(float(np.nanmax(pivot))),
            1e-12,
        )
        image = axis.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
        )
        axis.set_title(f"tag weight={weight:g}")
        axis.set_xlabel("tag corruption multiplier")
        axis.set_ylabel("content noise")
        axis.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    if image is not None:
        figure.colorbar(image, ax=axes_flat.tolist(), label="Δ membership cosine")
    figure.suptitle("Observed additive fusion vs content-only")
    path = output_dir / "phase-diagram-membership-cosine.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def _write_readme(
    output_dir: Path,
    *,
    report_path: Path,
    runs_path: Path,
    plot_path: str | None,
) -> None:
    lines = [
        "# Synthetic tag fusion benchmark",
        "",
        "This directory contains the reproducible content-noise × tag-corruption × tag-weight sweep.",
        "",
        f"- report: `{report_path.name}`",
        f"- flat runs: `{runs_path.name}`",
    ]
    if plot_path is not None:
        lines.append(f"- phase diagram: `{Path(plot_path).name}`")
    lines.extend(
        [
            "",
            "The content-only rows are the baseline. The observed additive rows are compared against that baseline by seed and content-noise cell.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> tuple[Path, Path, str | None]:
    if args.n_samples < 20:
        raise ValueError("--n-samples must be at least 20")
    if args.n_roots < 2:
        raise ValueError("--n-roots must be at least 2")
    if args.embedding_dim < 2:
        raise ValueError("--embedding-dim must be at least 2")
    if args.pca_components < 1:
        raise ValueError("--pca-components must be positive")
    if args.n_init < 1 or args.max_iter < 1:
        raise ValueError("--n-init and --max-iter must be positive")

    if args.smoke:
        n_samples = min(args.n_samples, 120)
        content_noise = (0.10, 0.30)
        corruption_levels = (0.0, 1.0)
        tag_weights = (0.5, 1.0)
        seeds = tuple(args.seeds[:2]) or (42,)
    else:
        n_samples = args.n_samples
        content_noise = _float_list(args.content_noise)
        corruption_levels = _float_list(args.corruption_levels)
        tag_weights = _float_list(args.tag_weights)
        seeds = tuple(int(seed) for seed in args.seeds)
    if any(value < 0.0 for value in content_noise + corruption_levels):
        raise ValueError("content noise and corruption levels must be non-negative")
    if any(value <= 0.0 for value in tag_weights):
        raise ValueError("tag weights must be positive")

    output_dir = args.output_dir or Path(
        "benchmarks"
    ) / f"synthetic-tag-fusion-{date.today().isoformat()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total_cells = len(seeds) * len(content_noise) * len(corruption_levels)
    cell_number = 0
    for seed in seeds:
        for content_noise_value in content_noise:
            baseline_result: dict[str, Any] | None = None
            for corruption_value in corruption_levels:
                cell_number += 1
                print(
                    f"[{cell_number}/{total_cells}] "
                    f"seed={seed} content_noise={content_noise_value:g} "
                    f"tag_corruption={corruption_value:g}",
                    flush=True,
                )
                config = SyntheticTagConfig(
                    n_samples=n_samples,
                    n_roots=args.n_roots,
                    embedding_dim=args.embedding_dim,
                    factor_dim=min(args.factor_dim, args.n_roots),
                    seed=seed,
                    content_noise=content_noise_value,
                    tag_corruption=corruption_value,
                )
                dataset = generate_synthetic_tag_dataset(config)
                if baseline_result is None:
                    baseline_result = _run_row(
                        dataset,
                        seed=seed,
                        tag_source="none",
                        variant="content_only",
                        tag_embeddings=None,
                        tag_weight=0.0,
                        pca_components=args.pca_components,
                        n_init=args.n_init,
                        max_iter=args.max_iter,
                    )
                _add_baseline_rows(
                    rows,
                    dataset=dataset,
                    seed=seed,
                    content_result=baseline_result,
                    corruption_levels=(corruption_value,),
                )
                tag_sources = {
                    "observed": dataset.observed_tag_embeddings,
                    "shuffled": shuffle_tag_embeddings(
                        dataset.observed_tag_embeddings,
                        seed=seed + 10_000,
                    ),
                    "oracle": dataset.clean_tag_embeddings,
                }
                for tag_weight in tag_weights:
                    for tag_source, tag_values in tag_sources.items():
                        variants = ("additive",)
                        if tag_source == "observed":
                            variants = (
                                "additive",
                                "concat",
                                "same_pca_additive",
                            )
                        for variant in variants:
                            rows.append(
                                _run_row(
                                    dataset,
                                    seed=seed,
                                    tag_source=tag_source,
                                    variant=variant,
                                    tag_embeddings=tag_values,
                                    tag_weight=tag_weight,
                                    pca_components=args.pca_components,
                                    n_init=args.n_init,
                                    max_iter=args.max_iter,
                                )
                            )

    frame = pd.DataFrame(rows)
    runs_path = output_dir / "runs.csv"
    frame.to_csv(runs_path, index=False)
    report_path = output_dir / "report.json"
    report = {
        "schema_version": 1,
        "configuration": {
            "n_samples": n_samples,
            "n_roots": args.n_roots,
            "embedding_dim": args.embedding_dim,
            "factor_dim": min(args.factor_dim, args.n_roots),
            "content_noise": list(content_noise),
            "corruption_levels": list(corruption_levels),
            "tag_weights": list(tag_weights),
            "seeds": list(seeds),
            "pca_components": args.pca_components,
            "n_init": args.n_init,
            "max_iter": args.max_iter,
            "smoke": bool(args.smoke),
        },
        "run_count": int(len(frame)),
        "runs": json.loads(frame.to_json(orient="records")),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_path = None if args.skip_plot else _write_phase_plot(frame, output_dir)
    _write_readme(
        output_dir,
        report_path=report_path,
        runs_path=runs_path,
        plot_path=plot_path,
    )
    return report_path, runs_path, plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--n-roots", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--factor-dim", type=int, default=5)
    parser.add_argument(
        "--content-noise",
        type=float,
        nargs="+",
        default=DEFAULT_CONTENT_NOISE,
    )
    parser.add_argument(
        "--corruption-levels",
        type=float,
        nargs="+",
        default=DEFAULT_CORRUPTION_LEVELS,
    )
    parser.add_argument(
        "--tag-weights",
        type=float,
        nargs="+",
        default=DEFAULT_TAG_WEIGHTS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small two-noise × two-corruption × two-weight matrix.",
    )
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    report_path, runs_path, plot_path = run_sweep(parse_args())
    print(f"Report saved: {report_path}")
    print(f"Runs saved: {runs_path}")
    if plot_path is not None:
        print(f"Phase diagram saved: {plot_path}")


if __name__ == "__main__":
    main()
