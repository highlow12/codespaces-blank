from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def load_embeddings_from_json(
    json_path: Path,
    *,
    start: int = 0,
    limit: int | None = None,
    id_offset: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"No embedding records found in {json_path}")
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")

    end = len(records) if limit is None else min(len(records), start + limit)
    selected_records = records[start:end]
    if not selected_records:
        raise ValueError(
            f"No embedding records selected from {json_path} "
            f"(start={start}, limit={limit})"
        )

    metadata_rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    ids: list[Any] = []
    for index, record in enumerate(selected_records, start=start):
        if not isinstance(record, dict) or "embedding" not in record:
            raise ValueError(f"Invalid embedding record at index {index}")
        embedding = np.asarray(record["embedding"], dtype=np.float64)
        if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
            raise ValueError(f"Invalid embedding at index {index}")
        embeddings.append(embedding)
        metadata = {key: value for key, value in record.items() if key != "embedding"}
        if "id" not in metadata:
            metadata["id"] = metadata.get("resource", id_offset + index)
        metadata.setdefault("tag", f"Document_{index}")
        ids.append(metadata["id"])
        metadata_rows.append(metadata)

    lengths = {embedding.shape[0] for embedding in embeddings}
    if len(lengths) != 1:
        raise ValueError(f"Embeddings have inconsistent dimensions: {sorted(lengths)}")
    try:
        if len(set(ids)) != len(ids):
            raise ValueError("Embedding IDs must be unique in each selected batch")
    except TypeError as error:
        raise ValueError("Embedding IDs must be hashable scalar values") from error

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
