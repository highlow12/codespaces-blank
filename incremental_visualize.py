from __future__ import annotations

import argparse
from pathlib import Path

from cluster_visualization import (
    make_fixed_coordinate_plot,
)
from incremental_clustering import load_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an incremental state using its fixed coordinates."
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/incremental_cluster_scatter.png"),
    )
    parser.add_argument("--title", type=str, default="Incremental Clustering")
    parser.add_argument(
        "--color-by",
        choices=["auto", "cluster", "tag"],
        default="auto",
    )
    args = parser.parse_args()

    state = load_state(args.state)
    make_fixed_coordinate_plot(
        state.coordinates,
        state.assignments,
        args.output,
        title=args.title,
        color_by=args.color_by,
    )
    print(f"Saved fixed-coordinate visualization to: {args.output}")


if __name__ == "__main__":
    main()
