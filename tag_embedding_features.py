"""Load DBpedia label embeddings and append them to content embeddings.

The helper keeps document/tag alignment by class name and constructs the same
feature block used in the Gemini tag experiments:

    L2-normalized content || tag_weight * L2-normalized tag

The caller can pass the returned features to an existing unsupervised
clustering pipeline.  No ground-truth labels are needed by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from embedding_data import load_embeddings_from_json


def _normalize_vector(values: Any, *, name: str, row_index: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} at row {row_index} must be a finite vector")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{name} at row {row_index} must not be zero")
    return vector / length


def load_normalized_tag_embeddings(
    input_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str, str]]]:
    """Load one normalized vector and hierarchy tuple for every class label."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty JSON array in {input_path}")

    vectors: dict[str, np.ndarray] = {}
    hierarchies: dict[str, tuple[str, str, str]] = {}
    for row_index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Tag record {row_index} is not a JSON object")
        label = record.get("label")
        hierarchy = record.get("class_hierarchy")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Tag record {row_index} has no valid label")
        if (
            not isinstance(hierarchy, list)
            or len(hierarchy) != 3
            or not all(isinstance(item, str) and item.strip() for item in hierarchy)
            or hierarchy[-1] != label
        ):
            raise ValueError(f"Tag record {row_index} has an invalid class_hierarchy")
        if label in vectors:
            raise ValueError(f"Duplicate tag label: {label!r}")
        vectors[label] = _normalize_vector(
            record.get("embedding"),
            name="tag embedding",
            row_index=row_index,
        )
        hierarchies[label] = (hierarchy[0], hierarchy[1], hierarchy[2])

    dimensions = {vector.shape[0] for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError(f"Tag embeddings have inconsistent dimensions: {dimensions}")
    return vectors, hierarchies


def load_dbpedia_embedding_pair(
    content_path: Path,
    tag_path: Path,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    dict[str, tuple[str, str, str]],
]:
    """Load content vectors and align each row's tag vector by ``class``."""

    content, metadata = load_embeddings_from_json(content_path)
    if "class" not in metadata.columns:
        raise ValueError("Content embeddings must contain a 'class' field")
    if metadata["class"].isna().any():
        raise ValueError("Content embeddings contain missing class labels")

    class_values = metadata["class"].astype(str).to_numpy()
    tag_vectors, hierarchies = load_normalized_tag_embeddings(tag_path)
    content_classes = set(class_values.tolist())
    tag_classes = set(tag_vectors)
    missing_tags = sorted(content_classes - tag_classes)
    unused_tags = sorted(tag_classes - content_classes)
    if missing_tags or unused_tags:
        raise ValueError(
            "Content and tag classes do not match; "
            f"missing_tags={missing_tags}, unused_tags={unused_tags}"
        )

    tag_matrix = np.vstack([tag_vectors[class_name] for class_name in class_values])
    if "class_hierarchy" in metadata.columns:
        for row_index, (class_name, content_hierarchy) in enumerate(
            zip(class_values, metadata["class_hierarchy"], strict=True)
        ):
            expected = list(hierarchies[class_name])
            if content_hierarchy != expected:
                raise ValueError(
                    "Content/tag hierarchy mismatch at row "
                    f"{row_index}: {content_hierarchy!r} != {expected!r}"
                )

    return content, metadata, tag_matrix, hierarchies


def build_tag_augmented_features(
    content_embeddings: np.ndarray,
    normalized_tag_embeddings: np.ndarray,
    *,
    tag_weight: float = 1.0,
) -> np.ndarray:
    """Return normalized-content concatenated with weighted normalized tags."""

    if not np.isfinite(tag_weight) or tag_weight <= 0.0:
        raise ValueError("tag_weight must be a finite value greater than zero")
    content = np.asarray(content_embeddings, dtype=np.float64)
    tags = np.asarray(normalized_tag_embeddings, dtype=np.float64)
    if content.ndim != 2 or tags.ndim != 2 or content.shape[0] != tags.shape[0]:
        raise ValueError("Content and tag embeddings must be aligned 2D matrices")
    if not np.all(np.isfinite(content)) or not np.all(np.isfinite(tags)):
        raise ValueError("Content and tag embeddings must be finite")
    return np.hstack(
        (
            normalize(content, norm="l2"),
            tag_weight * normalize(tags, norm="l2"),
        )
    )


__all__ = [
    "build_tag_augmented_features",
    "load_dbpedia_embedding_pair",
    "load_normalized_tag_embeddings",
]
