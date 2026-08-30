"""Shared mechanics for the Wikipedia benchmark entry points.

Keeping these helpers in one module makes split handling and stage measurement
consistent across quality, scaling, and comparison benchmarks.
"""

from __future__ import annotations

import csv
import os
import resource
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import gzip

import numpy as np


def rss_kib() -> float:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / 1024.0 if value > 10_000 else value


class MeasuredStage:
    """Measure elapsed wall time and peak RSS concurrently."""

    def __init__(self) -> None:
        self.elapsed_sec = 0.0
        self.peak_rss_kib = 0.0
        self.baseline_rss_kib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MeasuredStage":
        self.baseline_rss_kib = rss_kib()
        self.peak_rss_kib = self.baseline_rss_kib
        started = time.perf_counter()

        def sample() -> None:
            while not self._stop.wait(0.01):
                self.peak_rss_kib = max(self.peak_rss_kib, rss_kib())

        self._started = started
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_kib = max(self.peak_rss_kib, rss_kib())
        self.elapsed_sec = time.perf_counter() - self._started

    def as_dict(self) -> dict[str, float]:
        return {"sec": float(self.elapsed_sec), "baseline_rss_kib": float(self.baseline_rss_kib), "peak_rss_kib": float(self.peak_rss_kib), "peak_rss_delta_kib": float(max(0.0, self.peak_rss_kib - self.baseline_rss_kib))}


def split_data(embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, list[dict[str, Any]]]]:
    result: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    for split in ("discovery", "calibration", "test"):
        indices = [i for i, row in enumerate(metadata) if row.get("split") == split]
        if not indices:
            raise ValueError(f"Wikipedia metadata has no {split!r} rows")
        result[split] = (embeddings[indices], [metadata[i] for i in indices])
    return result


def repeat_rows(embeddings: np.ndarray, rows: list[dict[str, Any]], target_size: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if target_size < 1:
        raise ValueError("target split size must be positive")
    indices = np.arange(target_size, dtype=np.int64) % len(rows)
    repeated_embeddings = np.asarray(embeddings[indices], dtype=embeddings.dtype)
    repeated_rows: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(indices):
        row = dict(rows[int(source_index)])
        original_id = row.get("id", row.get("source_id", int(source_index)))
        row["original_id"] = original_id
        row["id"] = f"{original_id}__repeat_{output_index}"
        row["source_id"] = row["id"]
        repeated_rows.append(row)
    return repeated_embeddings, repeated_rows


def scale_splits(split: dict[str, tuple[np.ndarray, list[dict[str, Any]]]], target_size: int | None) -> dict[str, tuple[np.ndarray, list[dict[str, Any]]]]:
    if target_size is None:
        return split
    if target_size < 5:
        raise ValueError("--target-size must be at least 5")
    discovery_size = int(round(target_size * 0.60))
    calibration_size = int(round(target_size * 0.20))
    sizes = {"discovery": discovery_size, "calibration": calibration_size, "test": target_size - discovery_size - calibration_size}
    return {name: repeat_rows(*split[name], sizes[name]) for name in ("discovery", "calibration", "test")}


def write_gzip_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
