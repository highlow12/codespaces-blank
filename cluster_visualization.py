from __future__ import annotations

import colorsys
import os
import re
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "codex-matplotlib"),
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "codex-numba"),
)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from embedding_data import load_embeddings_from_json


_LEVEL_COLUMN_PATTERN = re.compile(r"^level_(\d+)_cluster$")
NOISE_COLOR = "#9aa0a6"


def _load_umap() -> Any:
    """Load UMAP with a writable Numba cache in restricted environments."""

    from umap import UMAP

    return UMAP


def load_embeddings(json_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    return load_embeddings_from_json(json_path)


def project_embeddings(
    embeddings: np.ndarray,
    *,
    seed: int,
    pca_components: int = 32,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
) -> np.ndarray:
    if pca_components < 1:
        raise ValueError("pca_components must be at least 1")
    normalized = normalize(embeddings)
    component_count = min(pca_components, normalized.shape[0], normalized.shape[1])
    pca_features = PCA(
        n_components=component_count,
        random_state=seed,
    ).fit_transform(normalized)
    pca_features = normalize(pca_features)
    reducer = _load_umap()(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
        random_state=seed,
    )
    return reducer.fit_transform(pca_features)


def fit_projection_model(
    embeddings: np.ndarray,
    *,
    seed: int,
    pca_components: int = 32,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
) -> tuple[PCA, Any, np.ndarray]:
    """Fit PCA+UMAP once and return the model for future point transforms."""

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    if pca_components < 1:
        raise ValueError("pca_components must be at least 1")
    normalized = normalize(embeddings)
    component_count = min(pca_components, normalized.shape[0], normalized.shape[1])
    pca = PCA(n_components=component_count, random_state=seed).fit(normalized)
    pca_features = normalize(pca.transform(normalized))
    reducer = _load_umap()(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
        random_state=seed,
    )
    reduced = reducer.fit_transform(pca_features)
    return pca, reducer, reduced


def transform_projection(
    embeddings: np.ndarray,
    *,
    pca: PCA,
    reducer: Any,
) -> np.ndarray:
    """Project a new batch using a previously fitted PCA+UMAP model."""

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    normalized = normalize(embeddings)
    pca_features = normalize(pca.transform(normalized))
    reduced = np.asarray(reducer.transform(pca_features), dtype=np.float64)
    if reduced.ndim != 2 or reduced.shape[1] != 2:
        raise ValueError("UMAP transform must return two-dimensional coordinates")
    if not np.all(np.isfinite(reduced)):
        raise ValueError("UMAP transform returned non-finite coordinates")
    return reduced


def compact_umap_presets() -> list[dict[str, object]]:
    return [
        {
            "name": "dense",
            "n_neighbors": 8,
            "min_dist": 0.0,
            "metric": "cosine",
            "spread": 0.7,
            "densmap": True,
        },
        {
            "name": "compact",
            "n_neighbors": 12,
            "min_dist": 0.01,
            "metric": "cosine",
            "spread": 0.8,
            "densmap": True,
        },
        {
            "name": "balanced",
            "n_neighbors": 15,
            "min_dist": 0.02,
            "metric": "cosine",
            "spread": 0.85,
            "densmap": True,
        },
        {
            "name": "local",
            "n_neighbors": 20,
            "min_dist": 0.03,
            "metric": "cosine",
            "spread": 0.9,
            "densmap": False,
        },
    ]


def _level_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if _LEVEL_COLUMN_PATTERN.match(column)]
    return sorted(columns, key=lambda column: int(_LEVEL_COLUMN_PATTERN.match(column).group(1)))


def load_assignments(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    hierarchy_columns = _level_columns(frame)
    if "id" not in frame.columns:
        raise ValueError(f"Missing columns in {csv_path}: ['id']")
    if "cluster" not in frame.columns and not hierarchy_columns:
        raise ValueError(
            f"{csv_path} needs 'cluster' or level_N_cluster columns"
        )

    columns = ["id"]
    if "cluster" in frame.columns:
        columns.append("cluster")
    columns.extend(hierarchy_columns)
    for optional_column in ("cluster_path", "is_noise", "noise_level", "leaf_level"):
        if optional_column in frame.columns:
            columns.append(optional_column)
    return frame[list(dict.fromkeys(columns))]


def _as_cluster_number(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def prepare_visual_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Add display labels such as ``3-2-4`` to flat or hierarchical outputs."""

    frame = assignments.copy()
    hierarchy_columns = _level_columns(frame)
    display_labels: list[str] = []
    top_labels: list[str] = []
    noise_flags: list[bool] = []

    for _, row in frame.iterrows():
        is_noise = _as_bool(row.get("is_noise", False))
        cluster_path = str(row.get("cluster_path", ""))
        if cluster_path.lower().endswith("noise"):
            is_noise = True

        if hierarchy_columns:
            parts: list[str] = []
            for column in hierarchy_columns:
                cluster_number = _as_cluster_number(row[column])
                if cluster_number is None:
                    break
                # Internal assignment IDs are 0-based; visual labels are 1-based.
                parts.append(str(cluster_number + 1))
            if is_noise:
                display_label = "-".join(parts + ["noise"]) if parts else "noise"
            else:
                display_label = "-".join(parts) if parts else "root"
            top_label = parts[0] if parts else ("noise" if is_noise else "root")
        else:
            cluster_number = _as_cluster_number(row.get("cluster", -1))
            if cluster_number is None:
                display_label = "noise"
                top_label = "noise"
                is_noise = True
            else:
                display_label = str(cluster_number)
                top_label = display_label

        display_labels.append(display_label)
        top_labels.append(top_label)
        noise_flags.append(is_noise)

    frame["display_label"] = display_labels
    frame["display_top_label"] = top_labels
    frame["is_noise"] = noise_flags
    frame["is_hierarchical"] = bool(hierarchy_columns)
    return frame


def _natural_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _is_noise_label(value: Any) -> bool:
    text = str(value).strip().lower()
    return text == "noise" or text.endswith("-noise")


def _ordered_values(values: np.ndarray) -> list[Any]:
    return sorted(
        pd.unique(values),
        key=lambda value: (
            _is_noise_label(value) or str(value).strip() == "-1",
            _natural_key(value),
        ),
    )


def hierarchical_color_map(values: np.ndarray) -> dict[str, Any]:
    """Give each top-level group a distinct hue and children related shades."""

    unique_values = [str(value) for value in _ordered_values(values)]
    non_noise_values = [
        value for value in unique_values if not _is_noise_label(value)
    ]
    top_values = _ordered_values(
        np.asarray(
            sorted(
                {
                    value.split("-")[0]
                    for value in non_noise_values
                    if value
                },
                key=_natural_key,
            ),
            dtype=object,
        )
    )
    top_values = [str(value) for value in top_values]

    if len(top_values) <= 10:
        top_cmap = plt.get_cmap("tab10", max(len(top_values), 1))
    else:
        top_cmap = plt.get_cmap("hsv", max(len(top_values), 1))

    color_map: dict[str, Any] = {
        value: NOISE_COLOR
        for value in unique_values
        if _is_noise_label(value)
    }
    for top_index, top_value in enumerate(top_values):
        base = top_cmap(top_index)[:3]
        hue, _, saturation = colorsys.rgb_to_hls(*base)
        child_values = [
            value
            for value in non_noise_values
            if value.split("-")[0] == top_value
        ]
        child_values = sorted(dict.fromkeys(child_values), key=_natural_key)
        for child_index, child_value in enumerate(child_values):
            if len(child_values) == 1:
                lightness = 0.50
            else:
                lightness = float(
                    np.linspace(0.36, 0.74, len(child_values))[child_index]
                )
            color_map[child_value] = colorsys.hls_to_rgb(
                hue,
                lightness,
                max(0.55, saturation),
            )
    return color_map


def categorical_color_map(values: np.ndarray) -> dict[str, Any]:
    unique_values = [str(value) for value in _ordered_values(values)]
    cmap = plt.get_cmap("tab20", max(len(unique_values), 1))
    return {
        value: (NOISE_COLOR if _is_noise_label(value) else cmap(index))
        for index, value in enumerate(unique_values)
    }


def _plot_groups(
    axis: Any,
    reduced: np.ndarray,
    metadata: pd.DataFrame,
    values: np.ndarray,
    color_map: dict[str, Any],
    *,
    point_size: float,
    alpha: float,
    include_labels: bool,
) -> list[Patch]:
    unique_values = _ordered_values(values)
    handles: list[Patch] = []
    for value in unique_values:
        value_key = str(value)
        mask = values.astype(str) == value_key
        color = color_map.get(value_key, NOISE_COLOR)
        axis.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=point_size,
            alpha=alpha,
            c=[color],
            edgecolors="none",
        )
        if include_labels:
            label = f"cluster {value_key} ({int(mask.sum())})"
            handles.append(Patch(facecolor=color, edgecolor="none", label=label))
    return handles


def _resolve_color_values(
    metadata: pd.DataFrame,
    color_by: str,
) -> tuple[np.ndarray, str, bool]:
    if color_by == "auto":
        color_by = "cluster"
    if color_by not in {"cluster", "tag"}:
        raise ValueError(f"Unsupported color mode: {color_by}")

    if color_by == "cluster":
        column = "display_label"
        hierarchical = bool(metadata["is_hierarchical"].iloc[0])
    else:
        column = "tag"
        hierarchical = False
    if column not in metadata.columns:
        raise ValueError(f"Missing '{column}' column for coloring")
    return metadata[column].astype(str).to_numpy(), color_by, hierarchical


def make_cluster_plot(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    seed: int,
    pca_components: int = 32,
    color_by: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
) -> None:
    metadata = prepare_visual_assignments(metadata)
    reduced = project_embeddings(
        embeddings,
        seed=seed,
        pca_components=pca_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
    )
    values, color_mode, hierarchical = _resolve_color_values(metadata, color_by)
    color_map = (
        hierarchical_color_map(values)
        if hierarchical
        else categorical_color_map(values)
    )

    fig, axis = plt.subplots(figsize=(12, 9))
    handles = _plot_groups(
        axis,
        reduced,
        metadata,
        values,
        color_map,
        point_size=18,
        alpha=0.85,
        include_labels=True,
    )
    axis.set_title(
        f"{title} [PCA-{pca_components} + UMAP | {color_mode}"
        f"{' | hierarchical' if hierarchical else ''}]"
    )
    axis.set_xlabel("UMAP-1")
    axis.set_ylabel("UMAP-2")
    axis.legend(
        handles=handles,
        loc="best",
        frameon=True,
        title="cluster label (count)",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_fixed_coordinate_plot(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color_by: str,
) -> None:
    """Render stored coordinates without refitting or moving existing points."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be an N x 2 array")
    if len(metadata) != coordinates.shape[0]:
        raise ValueError("metadata and coordinates must have the same row count")

    metadata = prepare_visual_assignments(metadata)
    values, color_mode, hierarchical = _resolve_color_values(metadata, color_by)
    color_map = (
        hierarchical_color_map(values)
        if hierarchical
        else categorical_color_map(values)
    )

    fig, axis = plt.subplots(figsize=(12, 9))
    handles = _plot_groups(
        axis,
        coordinates,
        metadata,
        values,
        color_map,
        point_size=18,
        alpha=0.85,
        include_labels=True,
    )
    axis.set_title(
        f"{title} [fixed PCA + UMAP | {color_mode}"
        f"{' | hierarchical' if hierarchical else ''}]"
    )
    axis.set_xlabel("UMAP-1")
    axis.set_ylabel("UMAP-2")
    axis.legend(
        handles=handles,
        loc="best",
        frameon=True,
        title="cluster label (count)",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_comparison_plot(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    seed: int,
    pca_components: int = 32,
    color_by: str,
) -> None:
    metadata = prepare_visual_assignments(metadata)
    values, color_mode, hierarchical = _resolve_color_values(metadata, color_by)
    color_map = (
        hierarchical_color_map(values)
        if hierarchical
        else categorical_color_map(values)
    )
    presets = compact_umap_presets()
    rows = int(np.ceil(len(presets) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5.5 * rows), squeeze=False)
    handles = []

    for axis, preset in zip(axes.flat, presets, strict=True):
        reduced = project_embeddings(
            embeddings,
            seed=seed,
            pca_components=pca_components,
            n_neighbors=int(preset["n_neighbors"]),
            min_dist=float(preset["min_dist"]),
            metric=str(preset["metric"]),
            spread=float(preset["spread"]),
            densmap=bool(preset["densmap"]),
        )
        panel_handles = _plot_groups(
            axis,
            reduced,
            metadata,
            values,
            color_map,
            point_size=12,
            alpha=0.82,
            include_labels=not handles,
        )
        if not handles:
            handles = panel_handles
        axis.set_title(
            f"{preset['name']} | n={preset['n_neighbors']} "
            f"d={preset['min_dist']} spread={preset['spread']} "
            f"dens={preset['densmap']}"
        )
        axis.set_xlabel("UMAP-1")
        axis.set_ylabel("UMAP-2")

    for axis in axes.flat[len(presets) :]:
        axis.axis("off")

    fig.suptitle(
        f"{title} [PCA-{pca_components} + UMAP comparison | {color_mode}"
        f"{' | hierarchical' if hierarchical else ''}]",
        y=0.995,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=True,
        title="cluster label (count)",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
