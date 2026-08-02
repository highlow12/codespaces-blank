from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from umap import UMAP


def load_embeddings(json_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"No records found in {json_path}")

    embeddings = np.vstack([np.asarray(record["embedding"], dtype=np.float64) for record in records])
    metadata = pd.DataFrame(
        {
            "id": [record.get("id", index) for index, record in enumerate(records)],
            "tag": [record.get("tag", f"Document_{index}") for index, record in enumerate(records)],
        }
    )
    return embeddings, metadata


def project_embeddings(
    embeddings: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
) -> np.ndarray:
    normalized = normalize(embeddings)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
        random_state=seed,
    )
    return reducer.fit_transform(normalized)


def compact_umap_presets() -> list[dict[str, object]]:
    return [
        {"name": "dense", "n_neighbors": 8, "min_dist": 0.0, "metric": "cosine", "spread": 0.7, "densmap": True},
        {"name": "compact", "n_neighbors": 12, "min_dist": 0.01, "metric": "cosine", "spread": 0.8, "densmap": True},
        {"name": "balanced", "n_neighbors": 15, "min_dist": 0.02, "metric": "cosine", "spread": 0.85, "densmap": True},
        {"name": "local", "n_neighbors": 20, "min_dist": 0.03, "metric": "cosine", "spread": 0.9, "densmap": False},
    ]


def load_assignments(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"id", "cluster"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return frame[["id", "cluster"]]


def make_cluster_plot(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    seed: int,
    color_by: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
) -> None:
    reduced = project_embeddings(
        embeddings,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
    )

    if color_by == "auto":
        color_by = "cluster"

    if color_by not in {"cluster", "tag"}:
        raise ValueError(f"Unsupported color mode: {color_by}")

    if color_by not in metadata.columns:
        raise ValueError(f"Missing '{color_by}' column for coloring")

    values = metadata[color_by].to_numpy()
    unique_values = sorted(pd.unique(values), key=lambda value: (value == -1, str(value)))
    cmap = plt.get_cmap("tab20", max(len(unique_values), 1))

    plt.figure(figsize=(12, 9))
    for index, value in enumerate(unique_values):
        mask = values == value
        label = "noise" if value == -1 else f"{color_by} {value}"
        color = "#9aa0a6" if value == -1 else cmap(index)
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=18,
            alpha=0.85,
            c=[color],
            label=f"{label} ({int(mask.sum())})",
            edgecolors="none",
        )

    plt.title(f"{title} [UMAP | {color_by}]")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def make_comparison_plot(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    seed: int,
    color_by: str,
) -> None:
    presets = compact_umap_presets()
    rows = int(np.ceil(len(presets) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5.5 * rows), squeeze=False)

    if color_by == "auto":
        color_by = "cluster"

    if color_by not in {"cluster", "tag"}:
        raise ValueError(f"Unsupported color mode: {color_by}")

    if color_by not in metadata.columns:
        raise ValueError(f"Missing '{color_by}' column for coloring")

    values = metadata[color_by].to_numpy()
    unique_values = sorted(pd.unique(values), key=lambda value: (value == -1, str(value)))
    cmap = plt.get_cmap("tab20", max(len(unique_values), 1))

    for axis, preset in zip(axes.flat, presets, strict=True):
        reduced = project_embeddings(
            embeddings,
            seed=seed,
            n_neighbors=int(preset["n_neighbors"]),
            min_dist=float(preset["min_dist"]),
            metric=str(preset["metric"]),
            spread=float(preset["spread"]),
            densmap=bool(preset["densmap"]),
        )
        for index, value in enumerate(unique_values):
            mask = values == value
            color = "#9aa0a6" if value == -1 else cmap(index)
            axis.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                s=12,
                alpha=0.82,
                c=[color],
                edgecolors="none",
            )
        axis.set_title(
            f"{preset['name']} | n={preset['n_neighbors']} d={preset['min_dist']} spread={preset['spread']} dens={preset['densmap']}"
        )
        axis.set_xlabel("UMAP-1")
        axis.set_ylabel("UMAP-2")

    for axis in axes.flat[len(presets):]:
        axis.axis("off")

    fig.suptitle(f"{title} [UMAP comparison | {color_by}]", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize clustered embeddings with a compact 2D UMAP scatter plot.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cluster_scatter.png"))
    parser.add_argument("--title", type=str, default="Real Embeddings Clustering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-by", type=str, default="auto", choices=["auto", "cluster", "tag"])
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.02)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--spread", type=float, default=0.8)
    parser.add_argument("--densmap", action="store_true", default=True)
    parser.add_argument("--no-densmap", action="store_false", dest="densmap")
    parser.add_argument("--compare", action="store_true", help="Save a comparison grid with multiple compact UMAP presets")
    args = parser.parse_args()

    embeddings, metadata = load_embeddings(args.input_json)
    assignments = load_assignments(args.assignments_csv)

    merged = metadata.merge(assignments, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(metadata):
        raise ValueError("Assignments and embeddings do not align by id")

    merged["cluster"] = merged["cluster"].astype(int)

    if args.compare:
        make_comparison_plot(
            embeddings=embeddings,
            metadata=merged,
            output_path=args.output,
            title=args.title,
            seed=args.seed,
            color_by=args.color_by,
        )
    else:
        make_cluster_plot(
            embeddings=embeddings,
            metadata=merged,
            output_path=args.output,
            title=args.title,
            seed=args.seed,
            color_by=args.color_by,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            spread=args.spread,
            densmap=args.densmap,
        )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()