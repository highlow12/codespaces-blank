from __future__ import annotations

import argparse
from pathlib import Path

from cluster_visualization import (
    DEFAULT_CLUSTER_TARGET_WEIGHT,
    load_assignments,
    load_embeddings,
    make_cluster_plot,
    make_comparison_plot,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize flat or hierarchical clustered embeddings with UMAP."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cluster_scatter.png"),
    )
    parser.add_argument("--title", type=str, default="Real Embeddings Clustering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "PCA dimensions before UMAP; omit to auto-select using UMAP "
            "k-NN preservation."
        ),
    )
    parser.add_argument(
        "--cluster-target-weight",
        type=float,
        default=DEFAULT_CLUSTER_TARGET_WEIGHT,
        help=(
            "Weak supervised UMAP weight for cluster membership; 0 disables it "
            "(default: 0.01)."
        ),
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="auto",
        choices=["auto", "cluster"],
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.02)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--spread", type=float, default=0.85)
    parser.add_argument("--densmap", action="store_true", default=True)
    parser.add_argument("--no-densmap", action="store_false", dest="densmap")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Save a comparison grid with multiple compact UMAP presets.",
    )
    args = parser.parse_args()

    embeddings, metadata = load_embeddings(args.input_json)
    assignments = load_assignments(args.assignments_csv)
    merged = metadata.merge(
        assignments,
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(metadata):
        raise ValueError("Assignments and embeddings do not align by id")

    if args.compare:
        make_comparison_plot(
            embeddings=embeddings,
            metadata=merged,
            output_path=args.output,
            title=args.title,
            seed=args.seed,
            pca_components=args.pca_components,
            color_by=args.color_by,
            cluster_target_weight=args.cluster_target_weight,
        )
    else:
        make_cluster_plot(
            embeddings=embeddings,
            metadata=merged,
            output_path=args.output,
            title=args.title,
            seed=args.seed,
            pca_components=args.pca_components,
            color_by=args.color_by,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            spread=args.spread,
            densmap=args.densmap,
            cluster_target_weight=args.cluster_target_weight,
        )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
