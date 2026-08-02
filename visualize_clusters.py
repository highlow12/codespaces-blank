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


def load_assignments(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"id", "cluster"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return frame[["id", "cluster"]]


def make_side_by_side_plot(
    embeddings: np.ndarray,
    truth: np.ndarray,
    clusters: np.ndarray,
    output_path: Path,
    *,
    title: str,
    seed: int,
) -> None:
    reduced = UMAP(n_components=2, n_neighbors=25, min_dist=0.08, random_state=seed).fit_transform(normalize(embeddings))
    truth_labels = sorted(pd.unique(truth))
    pred_labels = sorted(pd.unique(clusters), key=lambda value: (value == -1, value))
    truth_cmap = plt.get_cmap("tab10", max(len(truth_labels), 1))
    pred_cmap = plt.get_cmap("tab20", max(len(pred_labels), 1))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)

    for index, label_value in enumerate(truth_labels):
        mask = truth == label_value
        axes[0].scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=16,
            alpha=0.85,
            c=[truth_cmap(index)],
            label=f"{label_value} ({int(mask.sum())})",
            edgecolors="none",
        )

    for index, cluster_id in enumerate(pred_labels):
        mask = clusters == cluster_id
        label = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        color = "#9aa0a6" if cluster_id == -1 else pred_cmap(index)
        axes[1].scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=16,
            alpha=0.85,
            c=[color],
            label=f"{label} ({int(mask.sum())})",
            edgecolors="none",
        )

    axes[0].set_title("Ground truth tags")
    axes[1].set_title("Predicted clusters")
    for axis in axes:
        axis.set_xlabel("UMAP-1")
        axis.set_ylabel("UMAP-2")
        axis.legend(loc="best", frameon=True, fontsize=9)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize clustered embeddings with a 2D UMAP scatter plot.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cluster_scatter.png"))
    parser.add_argument("--title", type=str, default="Real Embeddings Clustering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embeddings, metadata = load_embeddings(args.input_json)
    assignments = load_assignments(args.assignments_csv)

    merged = metadata.merge(assignments, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(metadata):
        raise ValueError("Assignments and embeddings do not align by id")

    make_side_by_side_plot(
        embeddings=embeddings,
        truth=merged["tag"].to_numpy(),
        clusters=merged["cluster"].to_numpy(),
        output_path=args.output,
        title=args.title,
        seed=args.seed,
    )
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()