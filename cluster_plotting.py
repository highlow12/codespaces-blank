"""Cluster color mapping and static visualization rendering."""

from __future__ import annotations

import colorsys
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "codex-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from umap_projection import (
    DEFAULT_CLUSTER_TARGET_WEIGHT,
    project_embeddings,
)
from visual_assignments import build_cluster_supervision, prepare_visual_assignments


NOISE_COLOR = "#9aa0a6"


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
    if color_by != "cluster":
        if color_by == "tag":
            raise ValueError(
                "Category-name coloring is disabled; use --color-by cluster"
            )
        raise ValueError(f"Unsupported color mode: {color_by}")

    column = "display_label"
    hierarchical = bool(metadata["is_hierarchical"].iloc[0])
    if column not in metadata.columns:
        raise ValueError(f"Missing '{column}' column for coloring")
    return metadata[column].astype(str).to_numpy(), color_by, hierarchical


def _prepare_plot_style(
    metadata: pd.DataFrame,
    color_by: str,
) -> tuple[pd.DataFrame, np.ndarray, str, bool, dict[str, Any]]:
    prepared = prepare_visual_assignments(metadata)
    values, color_mode, hierarchical = _resolve_color_values(prepared, color_by)
    color_map = (
        hierarchical_color_map(values)
        if hierarchical
        else categorical_color_map(values)
    )
    return prepared, values, color_mode, hierarchical, color_map


def _save_cluster_scatter(
    coordinates: np.ndarray,
    values: np.ndarray,
    color_map: dict[str, Any],
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 9))
    handles = _plot_groups(
        axis,
        coordinates,
        values,
        color_map,
        point_size=18,
        alpha=0.85,
        include_labels=True,
    )
    axis.set_title(title)
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


def _pca_title_dimension(pca_components: int | None) -> str:
    return "auto" if pca_components is None else str(pca_components)


def make_cluster_plot(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    seed: int,
    pca_components: int | None = None,
    color_by: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    spread: float,
    densmap: bool,
    cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> None:
    metadata, values, color_mode, hierarchical, color_map = _prepare_plot_style(
        metadata,
        color_by,
    )
    cluster_target, cluster_target_metric, target_description = (
        build_cluster_supervision(metadata)
    )
    reduced = project_embeddings(
        embeddings,
        seed=seed,
        pca_components=pca_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        spread=spread,
        densmap=densmap,
        cluster_target=cluster_target,
        cluster_target_metric=cluster_target_metric,
        cluster_target_weight=cluster_target_weight,
    )
    target_suffix = (
        f" | weak target: {target_description}, w={cluster_target_weight:.2f}"
        if cluster_target is not None and cluster_target_weight > 0.0
        else ""
    )
    _save_cluster_scatter(
        reduced,
        values,
        color_map,
        output_path,
        title=(
            f"{title} [PCA-{_pca_title_dimension(pca_components)} -> UMAP-2 | {color_mode}"
            f"{' | hierarchical' if hierarchical else ''}{target_suffix}]"
        ),
    )


def make_fixed_coordinate_plot(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color_by: str,
    pca_components: int | None = None,
    cluster_target_weight: float | None = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> None:
    """Render stored coordinates without refitting or moving existing points."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be an N x 2 array")
    if len(metadata) != coordinates.shape[0]:
        raise ValueError("metadata and coordinates must have the same row count")

    _, values, color_mode, hierarchical, color_map = _prepare_plot_style(
        metadata,
        color_by,
    )
    target_suffix = (
        f" | weak cluster target, w={cluster_target_weight:.2f}"
        if cluster_target_weight is not None and cluster_target_weight > 0.0
        else ""
    )
    _save_cluster_scatter(
        coordinates,
        values,
        color_map,
        output_path,
        title=(
            f"{title} [fixed PCA-{_pca_title_dimension(pca_components)} -> UMAP-2 | {color_mode}"
            f"{' | hierarchical' if hierarchical else ''}{target_suffix}]"
        ),
    )


def make_selected_coordinate_plot(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color_by: str,
    pca_components: int,
    cluster_target_weight: float | None = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> None:
    """Render coordinates produced by an explicitly selected PCA+UMAP fit."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be an N x 2 array")
    if len(metadata) != coordinates.shape[0]:
        raise ValueError("metadata and coordinates must have the same row count")

    _, values, color_mode, hierarchical, color_map = _prepare_plot_style(
        metadata,
        color_by,
    )
    target_suffix = (
        f" | weak cluster target, w={cluster_target_weight:.2f}"
        if cluster_target_weight is not None and cluster_target_weight > 0.0
        else ""
    )
    _save_cluster_scatter(
        coordinates,
        values,
        color_map,
        output_path,
        title=(
            f"{title} [Auto PCA-{pca_components} -> UMAP-2 | {color_mode}"
            f"{' | hierarchical' if hierarchical else ''}{target_suffix}]"
        ),
    )


def make_comparison_plot(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    seed: int,
    pca_components: int | None = None,
    color_by: str,
    cluster_target_weight: float = DEFAULT_CLUSTER_TARGET_WEIGHT,
) -> None:
    metadata, values, color_mode, hierarchical, color_map = _prepare_plot_style(
        metadata,
        color_by,
    )
    cluster_target, cluster_target_metric, target_description = (
        build_cluster_supervision(metadata)
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
            cluster_target=cluster_target,
            cluster_target_metric=cluster_target_metric,
            cluster_target_weight=cluster_target_weight,
        )
        panel_handles = _plot_groups(
            axis,
            reduced,
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

    target_suffix = (
        f" | weak target: {target_description}, w={cluster_target_weight:.2f}"
        if cluster_target is not None and cluster_target_weight > 0.0
        else ""
    )
    fig.suptitle(
        f"{title} [PCA-{_pca_title_dimension(pca_components)} -> UMAP-2 comparison | {color_mode}"
        f"{' | hierarchical' if hierarchical else ''}{target_suffix}]",
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
