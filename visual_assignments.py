"""Interpret hard, soft, and hierarchical assignment tables for visualization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_LEVEL_COLUMN_PATTERN = re.compile(r"^level_(\d+)_cluster$")
_MEMBERSHIP_COLUMN_PATTERN = re.compile(r"^membership_(\d+)$")
_LEVEL_MEMBERSHIP_COLUMN_PATTERN = re.compile(
    r"^level_(\d+)_membership_(\d+)$"
)
_PATH_MEMBERSHIP_COLUMN_PATTERN = re.compile(
    r"^level_(\d+)_path_membership_(\d+(?:_\d+)*)$"
)


def _level_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if _LEVEL_COLUMN_PATTERN.match(column)]
    return sorted(columns, key=lambda column: int(_LEVEL_COLUMN_PATTERN.match(column).group(1)))


def _membership_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if _MEMBERSHIP_COLUMN_PATTERN.match(column)
    ]
    return sorted(
        columns,
        key=lambda column: int(_MEMBERSHIP_COLUMN_PATTERN.match(column).group(1)),
    )


def _level_membership_columns(frame: pd.DataFrame, level: int) -> list[str]:
    columns = []
    for column in frame.columns:
        match = _LEVEL_MEMBERSHIP_COLUMN_PATTERN.match(column)
        if match and int(match.group(1)) == level:
            columns.append(column)
    return sorted(
        columns,
        key=lambda column: int(
            _LEVEL_MEMBERSHIP_COLUMN_PATTERN.match(column).group(2)
        ),
    )


def _path_membership_columns(frame: pd.DataFrame, level: int) -> list[str]:
    columns = []
    for column in frame.columns:
        match = _PATH_MEMBERSHIP_COLUMN_PATTERN.match(column)
        if match and int(match.group(1)) == level:
            columns.append(column)
    return sorted(
        columns,
        key=lambda column: tuple(
            int(part)
            for part in _PATH_MEMBERSHIP_COLUMN_PATTERN.match(column)
            .group(2)
            .split("_")
        ),
    )


def _path_from_membership_column(column: str) -> tuple[int, ...]:
    match = _PATH_MEMBERSHIP_COLUMN_PATTERN.match(column)
    if match is None:
        raise ValueError(f"Invalid path membership column: {column}")
    return tuple(int(part) for part in match.group(2).split("_"))


def load_assignments(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    hierarchy_columns = _level_columns(frame)
    level_membership_columns = [
        column
        for column in frame.columns
        if _LEVEL_MEMBERSHIP_COLUMN_PATTERN.match(column)
    ]
    level_membership_columns.sort(
        key=lambda column: (
            int(_LEVEL_MEMBERSHIP_COLUMN_PATTERN.match(column).group(1)),
            int(_LEVEL_MEMBERSHIP_COLUMN_PATTERN.match(column).group(2)),
        )
    )
    path_membership_columns = [
        column
        for column in frame.columns
        if _PATH_MEMBERSHIP_COLUMN_PATTERN.match(column)
    ]
    path_membership_columns.sort(
        key=lambda column: (
            int(_PATH_MEMBERSHIP_COLUMN_PATTERN.match(column).group(1)),
            _path_from_membership_column(column),
        )
    )
    if "id" not in frame.columns:
        raise ValueError(f"Missing columns in {csv_path}: ['id']")
    if (
        "cluster" not in frame.columns
        and not hierarchy_columns
        and not _membership_columns(frame)
        and not level_membership_columns
        and not path_membership_columns
    ):
        raise ValueError(
            f"{csv_path} needs 'cluster', membership, or level_N columns"
        )

    columns = ["id"]
    if "cluster" in frame.columns:
        columns.append("cluster")
    columns.extend(hierarchy_columns)
    for optional_column in ("cluster_path", "is_noise", "noise_level", "leaf_level"):
        if optional_column in frame.columns:
            columns.append(optional_column)
    columns.extend(_membership_columns(frame))
    if "membership_noise" in frame.columns:
        columns.append("membership_noise")
    columns.extend(level_membership_columns)
    columns.extend(path_membership_columns)
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


def _soft_argmax(
    row: pd.Series,
    columns: list[str],
    pattern: re.Pattern[str],
    *,
    include_noise: bool = False,
) -> tuple[int | None, bool]:
    """Return the largest soft-membership cluster and whether noise wins."""

    values: list[float] = []
    for column in columns:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError, OverflowError):
            value = float("nan")
        values.append(value if np.isfinite(value) else float("-inf"))

    if not values or max(values) == float("-inf"):
        return None, False

    best_position = int(np.argmax(values))
    best_value = values[best_position]
    if include_noise:
        try:
            noise_value = float(row.get("membership_noise", np.nan))
        except (TypeError, ValueError, OverflowError):
            noise_value = float("nan")
        if np.isfinite(noise_value) and noise_value > best_value:
            return None, True

    match = pattern.match(columns[best_position])
    if match is None:
        return None, False
    return int(match.group(match.lastindex)), False


def _path_soft_argmax(
    row: pd.Series,
    columns: list[str],
    parent_path: tuple[int, ...],
) -> tuple[int, ...] | None:
    if not columns:
        return None
    values: list[float] = []
    paths: list[tuple[int, ...]] = []
    for column in columns:
        path = _path_from_membership_column(column)
        if path[: len(parent_path)] != parent_path:
            continue
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError, OverflowError):
            value = float("nan")
        values.append(value if np.isfinite(value) else float("-inf"))
        paths.append(path)
    if not values or max(values) == float("-inf"):
        return None
    return paths[int(np.argmax(values))]


def prepare_visual_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Add display labels such as ``3-2-4`` to flat or hierarchical outputs."""

    frame = assignments.copy()
    hierarchy_columns = _level_columns(frame)
    flat_membership_columns = _membership_columns(frame)
    level_membership_columns = {
        int(_LEVEL_COLUMN_PATTERN.match(column).group(1)): _level_membership_columns(
            frame,
            int(_LEVEL_COLUMN_PATTERN.match(column).group(1)),
        )
        for column in hierarchy_columns
    }
    path_membership_columns = {
        level: _path_membership_columns(frame, level)
        for level in range(1, len(hierarchy_columns) + 1)
    }
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
            selected_path: tuple[int, ...] = ()
            for column in hierarchy_columns:
                level = int(_LEVEL_COLUMN_PATTERN.match(column).group(1))
                path_columns = path_membership_columns.get(level, [])
                if path_columns:
                    candidates = [
                        path_column
                        for path_column in path_columns
                        if _path_from_membership_column(path_column)[
                            : len(selected_path)
                        ]
                        == selected_path
                    ]
                    path = _path_soft_argmax(row, candidates, selected_path)
                    if path is not None:
                        selected_path = path
                        parts = [str(value + 1) for value in selected_path]
                        continue

                soft_columns = level_membership_columns.get(level, [])
                if soft_columns:
                    cluster_number, soft_noise = _soft_argmax(
                        row, soft_columns, _LEVEL_MEMBERSHIP_COLUMN_PATTERN
                    )
                    is_noise = is_noise or soft_noise
                else:
                    cluster_number = _as_cluster_number(row[column])
                if cluster_number is None:
                    break
                # Internal assignment IDs are 0-based; visual labels are 1-based.
                selected_path = (*selected_path, cluster_number)
                parts.append(str(cluster_number + 1))
            if is_noise:
                display_label = "-".join(parts + ["noise"]) if parts else "noise"
            else:
                display_label = "-".join(parts) if parts else "root"
            top_label = parts[0] if parts else ("noise" if is_noise else "root")
        else:
            if flat_membership_columns:
                cluster_number, soft_noise = _soft_argmax(
                    row,
                    flat_membership_columns,
                    _MEMBERSHIP_COLUMN_PATTERN,
                    include_noise=True,
                )
                is_noise = is_noise or soft_noise
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


def build_cluster_supervision(
    assignments: pd.DataFrame,
) -> tuple[np.ndarray | None, str | None, str]:
    """Build a weak UMAP target from soft memberships or cluster labels.

    Soft membership columns are preferred because they preserve ambiguity between
    clusters. When they are unavailable, the hierarchical display path is encoded
    as a categorical target. The returned target is intentionally separate from
    plotting colors so visualization can use assignments without making them the
    sole source of geometry.
    """

    frame = prepare_visual_assignments(assignments)
    path_membership_columns = [
        column
        for column in frame.columns
        if _PATH_MEMBERSHIP_COLUMN_PATTERN.match(column)
    ]
    path_membership_columns.sort(
        key=lambda column: (
            int(_PATH_MEMBERSHIP_COLUMN_PATTERN.match(column).group(1)),
            _path_from_membership_column(column),
        )
    )
    membership_columns = (
        path_membership_columns
        if path_membership_columns
        else _membership_columns(frame)
    )
    if not path_membership_columns and "membership_noise" in frame.columns:
        membership_columns.append("membership_noise")
    if membership_columns:
        membership_frame = frame[membership_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        membership_target = membership_frame.to_numpy(dtype=np.float64)
        row_norms = np.linalg.norm(membership_target, axis=1, keepdims=True)
        if (
            membership_target.ndim == 2
            and membership_target.shape[1] >= 2
            and np.all(np.isfinite(membership_target))
            and np.all(row_norms > 1e-12)
        ):
            membership_target = membership_target / row_norms
            if np.unique(membership_target, axis=0).shape[0] >= 2:
                return (
                    membership_target,
                    "euclidean",
                    f"soft cluster membership ({len(membership_columns)} dims)",
                )

    labels = frame["display_label"].astype(str).to_numpy()
    unique_labels, encoded = np.unique(labels, return_inverse=True)
    if unique_labels.size < 2:
        return None, None, "no varying cluster target"
    return encoded.astype(np.int32), "categorical", f"cluster labels ({len(unique_labels)} groups)"

