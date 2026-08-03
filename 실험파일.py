from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from clustering_pipelines import (
    PIPELINE_NAMES,
    build_soft_assignments,
    build_compact_umap,
    choose_best_pipeline,
    compact_umap_presets,
    evaluate_clustering,
    pipeline_to_filename,
    run_compact_umap_sweep,
    run_pipeline_1,
    run_pipeline_2,
    run_pipeline_2b,
    run_pipeline_3,
    run_pipeline_4,
    run_pipeline_5,
    run_pipeline_6,
    run_pipeline_by_name,
    run_selected_pipelines,
    save_soft_assignments,
    sort_candidate_metrics,
)
from clustering_types import (
    FCMKCandidate,
    FCMResult,
    HierarchicalResult,
    PipelineResult,
)
from embedding_data import load_embeddings_from_json, make_synthetic_embeddings
from fcm_hierarchy import (
    DEFAULT_CLUSTERING_PCA_COMPONENTS,
    fcm_noise_mask,
    fuzzy_silhouette_proxy,
    pca_normalized_features,
    run_hierarchical_pca_fcm,
    select_fcm_cluster_count,
    spherical_fcm,
    spherical_fcm_objective,
    xie_beni_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run selected clustering pipelines on high-dimensional embeddings."
    )
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--cluster-std", type=float, default=1.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument(
        "--assignments-output",
        type=Path,
        default=Path("best_pipeline_assignments.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Also run recursive PCA+FCM and save hierarchical assignments/tree outputs.",
    )
    parser.add_argument("--hierarchical-max-depth", type=int, default=4)
    parser.add_argument("--hierarchical-min-node-size", type=int, default=60)
    parser.add_argument("--hierarchical-min-child-size", type=int, default=20)
    parser.add_argument("--hierarchical-min-clusters", type=int, default=2)
    parser.add_argument("--hierarchical-max-clusters", type=int, default=8)
    parser.add_argument(
        "--hierarchical-k-selection",
        choices=["silhouette", "knee"],
        default="silhouette",
        help="How to choose k independently at each hierarchy node.",
    )
    parser.add_argument("--hierarchical-min-membership", type=float, default=0.40)
    parser.add_argument("--hierarchical-distance-z", type=float, default=3.5)
    parser.add_argument("--hierarchical-min-silhouette", type=float, default=0.05)
    parser.add_argument(
        "--hierarchical-pca-components",
        type=int,
        default=DEFAULT_CLUSTERING_PCA_COMPONENTS,
        help="PCA dimensions used for hierarchical clustering (default: 256).",
    )
    parser.add_argument(
        "--hierarchical-assignments-output",
        type=Path,
        default=Path("hierarchical_pca256_fcm_assignments.csv"),
    )
    parser.add_argument(
        "--hierarchical-tree-output",
        type=Path,
        default=Path("hierarchical_pca256_fcm_tree.json"),
    )
    parser.add_argument(
        "--pipeline",
        nargs="+",
        choices=["all", *PIPELINE_NAMES],
        default=["all"],
        help="Pipeline(s) to run. The default runs all pipelines.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_json is None:
        X, y = make_synthetic_embeddings(
            n_samples=args.samples,
            n_clusters=args.clusters,
            latent_dim=args.latent_dim,
            embedding_dim=args.embedding_dim,
            cluster_std=args.cluster_std,
            seed=args.seed,
        )
        metadata = pd.DataFrame({"id": np.arange(len(X)), "tag": y})
        has_ground_truth = True
    else:
        X, metadata = load_embeddings_from_json(args.input_json)
        if "tag" in metadata.columns:
            y = pd.factorize(metadata["tag"], sort=True)[0]
            has_ground_truth = True
        else:
            y = None
            has_ground_truth = False

    X = normalize(X, norm="l2")

    if "all" in args.pipeline:
        if args.pipeline != ["all"]:
            parser.error("'all' cannot be combined with individual pipeline names")
        selected_pipeline_names = list(PIPELINE_NAMES)
        pipeline_selection_name = "all"
    else:
        selected_pipeline_names = list(dict.fromkeys(args.pipeline))
        pipeline_selection_name = "+".join(selected_pipeline_names)
    pipeline_runs = run_selected_pipelines(
        selected_pipeline_names,
        X,
        y,
        args.clusters,
    )

    results = [run.metrics for run in pipeline_runs.values()]
    frame = pd.DataFrame(results)
    benchmark_columns = [
        "pipeline",
        "umap_preset",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_spread",
        "umap_densmap",
        "runtime_sec",
        "clusters",
        "noise_ratio",
        "nmi",
        "ari",
        "tag_fragmentation",
        "silhouette",
        "xie_beni",
        "fuzzy_silhouette",
        "iterations",
    ]
    for column in benchmark_columns:
        if column not in frame:
            frame[column] = np.nan
    frame = frame[benchmark_columns]

    if pipeline_selection_name == "all":
        benchmark_stem = "four_pipeline"
    elif len(selected_pipeline_names) == 1:
        benchmark_stem = f"pipeline_{pipeline_to_filename(selected_pipeline_names[0])}"
    else:
        benchmark_stem = "selected_pipelines"
    benchmark_csv = args.output_dir / f"{benchmark_stem}_benchmark.csv"
    benchmark_json = args.output_dir / f"{benchmark_stem}_benchmark.json"
    frame.to_csv(benchmark_csv, index=False)

    summary = {
        "data": {
            "pipeline": pipeline_selection_name,
            "pipelines": selected_pipeline_names,
            "input_normalized": True,
            "samples": args.samples,
            "clusters": args.clusters,
            "latent_dim": args.latent_dim,
            "embedding_dim": args.embedding_dim,
            "cluster_std": args.cluster_std,
            "seed": args.seed,
        },
        "results": frame.to_dict(orient="records"),
    }
    benchmark_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    hierarchical_result: HierarchicalResult | None = None
    if args.hierarchical:
        hierarchical_result = run_hierarchical_pca_fcm(
            X,
            metadata,
            max_depth=args.hierarchical_max_depth,
            min_node_size=args.hierarchical_min_node_size,
            min_child_size=args.hierarchical_min_child_size,
            min_clusters=args.hierarchical_min_clusters,
            max_clusters=args.hierarchical_max_clusters,
            min_membership=args.hierarchical_min_membership,
            distance_z=args.hierarchical_distance_z,
            selection_method=args.hierarchical_k_selection,
            min_split_silhouette=args.hierarchical_min_silhouette,
            pca_components=args.hierarchical_pca_components,
            seed=args.seed,
        )
        hierarchical_assignments_path = (
            args.output_dir / args.hierarchical_assignments_output
        )
        hierarchical_tree_path = args.output_dir / args.hierarchical_tree_output
        hierarchical_result.assignments.to_csv(
            hierarchical_assignments_path,
            index=False,
        )
        hierarchical_tree_path.write_text(
            json.dumps(hierarchical_result.tree, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    best_row = choose_best_pipeline(frame, has_ground_truth)
    best_pipeline = str(best_row["pipeline"])
    soft_assignment_paths: list[Path] = []
    for pipeline_name, run in pipeline_runs.items():
        if run.memberships is not None:
            assignments = build_soft_assignments(
                metadata,
                run.labels,
                run.memberships,
            )
        else:
            assignments = metadata.copy()
            assignments["cluster"] = run.labels
        assignments.to_csv(
            args.output_dir / f"assignments_{pipeline_to_filename(pipeline_name)}.csv",
            index=False,
        )
        if run.memberships is not None:
            soft_path = (
                args.output_dir
                / f"soft_assignments_{pipeline_to_filename(pipeline_name)}.csv"
            )
            save_soft_assignments(metadata, run.labels, run.memberships, soft_path)
            soft_assignment_paths.append(soft_path)

    best_run_name = next(
        pipeline_name
        for pipeline_name, run in pipeline_runs.items()
        if pipeline_name == best_pipeline or run.metrics.get("pipeline") == best_pipeline
    )
    best_run = pipeline_runs[best_run_name]
    if best_run.memberships is not None:
        best_assignments = build_soft_assignments(
            metadata,
            best_run.labels,
            best_run.memberships,
        )
    else:
        best_assignments = metadata.copy()
        best_assignments["cluster"] = best_run.labels
    best_assignments.to_csv(args.output_dir / args.assignments_output, index=False)

    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if has_ground_truth:
        print(f"\nBest by NMI/ARI: {best_pipeline}")
    else:
        print(f"\nBest by silhouette/noise/clusters: {best_pipeline}")
    print(f"Benchmark saved to: {benchmark_csv}")
    print(f"Benchmark summary saved to: {benchmark_json}")
    print(f"Cluster assignments saved to: {args.output_dir / args.assignments_output}")
    for soft_path in soft_assignment_paths:
        print(f"Soft assignments saved to: {soft_path}")
    if hierarchical_result is not None:
        print(
            "Hierarchical PCA+FCM: "
            f"{hierarchical_result.summary['levels_reached']} levels, "
            f"{hierarchical_result.summary['leaf_cluster_count']} leaf clusters, "
            f"{hierarchical_result.summary['noise_count']} noise points"
        )
        print(f"Hierarchical assignments saved to: {hierarchical_assignments_path}")
        print(f"Hierarchical tree saved to: {hierarchical_tree_path}")


if __name__ == "__main__":
    main()
