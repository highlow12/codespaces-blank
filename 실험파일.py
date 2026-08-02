from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from umap import UMAP


@dataclass
class FCMResult:
    labels: np.ndarray
    memberships: np.ndarray
    centers: np.ndarray
    iterations: int


def load_embeddings_from_json(json_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"No records found in {json_path}")

    metadata_rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    for index, record in enumerate(records):
        embedding = np.asarray(record["embedding"], dtype=np.float64)
        embeddings.append(embedding)
        metadata = {key: value for key, value in record.items() if key != "embedding"}
        metadata.setdefault("id", index)
        metadata.setdefault("tag", f"Document_{index}")
        metadata_rows.append(metadata)

    lengths = {embedding.shape[0] for embedding in embeddings}
    if len(lengths) != 1:
        raise ValueError(f"Embeddings have inconsistent dimensions: {sorted(lengths)}")

    return np.vstack(embeddings), pd.DataFrame(metadata_rows)


def make_synthetic_embeddings(
    *,
    n_samples: int,
    n_clusters: int,
    latent_dim: int,
    embedding_dim: int,
    cluster_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    latent, labels = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        n_features=latent_dim,
        cluster_std=cluster_std,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(latent_dim, embedding_dim))
    embeddings = latent @ projection
    embeddings += 0.05 * rng.normal(size=embeddings.shape)
    return embeddings.astype(np.float64), labels.astype(int)


def spherical_fcm(
    X: np.ndarray,
    n_clusters: int,
    *,
    m: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> FCMResult:
    X = normalize(X, norm="l2")
    rng = np.random.default_rng(seed)
    memberships = rng.random((X.shape[0], n_clusters))
    memberships /= memberships.sum(axis=1, keepdims=True)

    epsilon = 1e-12
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    for iteration in range(1, max_iter + 1):
        previous = memberships.copy()
        weights = memberships**m
        centers = weights.T @ X
        centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(centers_norm, epsilon)

        distances = cosine_distances(X, centers)
        distances = np.maximum(distances, epsilon)

        exponent = 2.0 / (m - 1.0)
        ratios = (distances[:, :, None] / distances[:, None, :]) ** exponent
        memberships = 1.0 / ratios.sum(axis=2)

        change = np.max(np.abs(memberships - previous))
        if change < tol:
            break

    labels = memberships.argmax(axis=1)
    return FCMResult(labels=labels, memberships=memberships, centers=centers, iterations=iteration)


def xie_beni_index(X: np.ndarray, result: FCMResult) -> float:
    X = normalize(X, norm="l2")
    memberships = result.memberships
    centers = result.centers
    distances = cosine_distances(X, centers)
    numerator = np.sum((memberships**2) * (distances**2))
    center_distances = cosine_distances(centers, centers)
    np.fill_diagonal(center_distances, np.inf)
    denominator = X.shape[0] * np.min(center_distances) ** 2
    return float(numerator / max(denominator, 1e-12))


def fuzzy_silhouette_proxy(X: np.ndarray, result: FCMResult, *, m: float = 2.0) -> float:
    X = normalize(X, norm="l2")
    memberships = result.memberships
    centers = result.centers
    distances = cosine_distances(X, centers)
    weights = memberships**m
    a = np.sum(weights * distances, axis=1) / np.sum(weights, axis=1)
    b = np.partition(distances, 1, axis=1)[:, 1]
    scores = (b - a) / np.maximum(a, b)
    return float(np.mean(scores))


def evaluate_clustering(
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    X_for_silhouette: np.ndarray,
) -> dict[str, Any]:
    labels = np.unique(y_pred)
    non_noise = y_pred != -1
    cluster_count = int(np.sum(labels != -1))
    noise_ratio = float(np.mean(~non_noise))

    nmi = np.nan
    ari = np.nan
    if y_true is not None:
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))

    metrics: dict[str, Any] = {
        "nmi": nmi,
        "ari": ari,
        "clusters": cluster_count,
        "noise_ratio": noise_ratio,
    }

    if cluster_count >= 2 and np.sum(non_noise) >= 3:
        try:
            metrics["silhouette"] = float(silhouette_score(X_for_silhouette[non_noise], y_pred[non_noise]))
        except Exception:
            metrics["silhouette"] = np.nan
    else:
        metrics["silhouette"] = np.nan
    return metrics


def run_pipeline_1(X: np.ndarray, y: np.ndarray | None, n_clusters: int) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    Xn = normalize(X, norm="l2")
    result = spherical_fcm(Xn, n_clusters=n_clusters, seed=42)
    elapsed = time.perf_counter() - start
    metrics = {
        "pipeline": "1_raw_fcm",
        "runtime_sec": elapsed,
        **evaluate_clustering(y, result.labels, Xn),
        "xie_beni": xie_beni_index(Xn, result),
        "fuzzy_silhouette": fuzzy_silhouette_proxy(Xn, result),
        "iterations": result.iterations,
    }
    return metrics, result.labels


def run_pipeline_2(X: np.ndarray, y: np.ndarray | None, n_clusters: int) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    Xp = PCA(n_components=64, random_state=42).fit_transform(X)
    Xn = normalize(Xp, norm="l2")
    result = spherical_fcm(Xn, n_clusters=n_clusters, seed=42)
    elapsed = time.perf_counter() - start
    metrics = {
        "pipeline": "2_pca64_fcm",
        "runtime_sec": elapsed,
        **evaluate_clustering(y, result.labels, Xn),
        "xie_beni": xie_beni_index(Xn, result),
        "fuzzy_silhouette": fuzzy_silhouette_proxy(Xn, result),
        "iterations": result.iterations,
    }
    return metrics, result.labels


def run_pipeline_2b(X: np.ndarray, y: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    Xp = PCA(n_components=64, random_state=42).fit_transform(X)
    Xn = normalize(Xp, norm="l2")
    labels = HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(Xn)
    elapsed = time.perf_counter() - start
    metrics = evaluate_clustering(y, labels, Xn)
    metrics.update({"xie_beni": np.nan, "fuzzy_silhouette": np.nan})
    return {
        "pipeline": "2b_pca64_hdbscan",
        "runtime_sec": elapsed,
        **metrics,
    }, labels


def run_pipeline_3(X: np.ndarray, y: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    Xp = PCA(n_components=50, random_state=42).fit_transform(X)
    Xu = UMAP(n_components=2, n_neighbors=25, min_dist=0.08, random_state=42).fit_transform(Xp)
    labels = HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(Xu)
    elapsed = time.perf_counter() - start
    metrics = evaluate_clustering(y, labels, Xu)
    metrics.update({"xie_beni": np.nan, "fuzzy_silhouette": np.nan})
    return {
        "pipeline": "3_pca50_umap2_hdbscan",
        "runtime_sec": elapsed,
        **metrics,
    }, labels


def run_pipeline_4(X: np.ndarray, y: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray]:
    start = time.perf_counter()
    Xu5 = UMAP(n_components=8, n_neighbors=25, min_dist=0.05, random_state=42).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(Xu5)
    elapsed = time.perf_counter() - start
    metrics = evaluate_clustering(y, labels, Xu5)
    metrics.update({"xie_beni": np.nan, "fuzzy_silhouette": np.nan})
    return {
        "pipeline": "4_umap8_hdbscan",
        "runtime_sec": elapsed,
        **metrics,
    }, labels


def choose_best_pipeline(frame: pd.DataFrame, has_ground_truth: bool) -> pd.Series:
    sortable = frame.copy()
    if has_ground_truth:
        sortable = sortable.sort_values(["nmi", "ari", "silhouette"], ascending=[False, False, False], na_position="last")
    else:
        sortable = sortable.sort_values(["silhouette", "noise_ratio", "runtime_sec"], ascending=[False, True, True], na_position="last")
    return sortable.iloc[0]


def pipeline_to_filename(pipeline: str) -> str:
    return pipeline.replace("/", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four clustering pipelines on synthetic high-dimensional embeddings.")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--cluster-std", type=float, default=1.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--assignments-output", type=Path, default=Path("best_pipeline_assignments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
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

    pipeline_runs: dict[str, tuple[dict[str, Any], np.ndarray]] = {
        "1_raw_fcm": run_pipeline_1(X, y, args.clusters),
        "2_pca64_fcm": run_pipeline_2(X, y, args.clusters),
        "2b_pca64_hdbscan": run_pipeline_2b(X, y),
        "3_pca50_umap2_hdbscan": run_pipeline_3(X, y),
        "4_umap8_hdbscan": run_pipeline_4(X, y),
    }

    results = [run[0] for run in pipeline_runs.values()]

    frame = pd.DataFrame(results)
    frame = frame[
        [
            "pipeline",
            "runtime_sec",
            "clusters",
            "noise_ratio",
            "nmi",
            "ari",
            "silhouette",
            "xie_beni",
            "fuzzy_silhouette",
            "iterations",
        ]
    ]
    frame.to_csv(args.output_dir / "four_pipeline_benchmark.csv", index=False)

    summary = {
        "data": {
            "samples": args.samples,
            "clusters": args.clusters,
            "latent_dim": args.latent_dim,
            "embedding_dim": args.embedding_dim,
            "cluster_std": args.cluster_std,
            "seed": args.seed,
        },
        "results": frame.to_dict(orient="records"),
    }
    (args.output_dir / "four_pipeline_benchmark.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best_row = choose_best_pipeline(frame, has_ground_truth)
    best_pipeline = str(best_row["pipeline"])
    for pipeline_name, (_, labels) in pipeline_runs.items():
        assignments = metadata.copy()
        assignments["cluster"] = labels
        assignments.to_csv(args.output_dir / f"assignments_{pipeline_to_filename(pipeline_name)}.csv", index=False)

    best_labels = pipeline_runs[best_pipeline][1]
    best_assignments = metadata.copy()
    best_assignments["cluster"] = best_labels
    best_assignments.to_csv(args.output_dir / args.assignments_output, index=False)

    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if has_ground_truth:
        print(f"\nBest by NMI/ARI: {best_pipeline}")
    else:
        print(f"\nBest by silhouette: {best_pipeline}")
    print(f"Cluster assignments saved to: {args.output_dir / args.assignments_output}")


if __name__ == "__main__":
    main()
