from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


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
