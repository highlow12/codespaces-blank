"""Shared mechanics for the project's incremental update engines.

The clustering engines keep different model state, but they need the same
bookkeeping around incoming batches.  This module deliberately contains no
clustering policy: it only handles ID-based row replacement/appending,
deterministic batch identity, bounded replay history, and checked state-file
I/O.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


BATCH_HISTORY_LIMIT = 256
STATE_ENVELOPE_FORMAT = "incremental_state_envelope_v1"


@dataclass(frozen=True)
class RowMergeResult:
    """Result of replacing existing IDs and appending unseen IDs."""

    embeddings: np.ndarray
    metadata: pd.DataFrame
    replaced_ids: list[Any]
    appended_ids: list[Any]
    coordinates: np.ndarray | None = None


def _positions(ids: list[Any], *, label: str) -> dict[Any, int]:
    positions: dict[Any, int] = {}
    for index, identifier in enumerate(ids):
        try:
            if identifier in positions:
                raise ValueError(f"{label} IDs must be unique")
            positions[identifier] = index
        except TypeError as error:
            raise ValueError(f"{label} IDs must be hashable scalar values") from error
    return positions


def _validate_coordinates(
    coordinates: np.ndarray | None,
    expected_rows: int,
    *,
    label: str,
) -> np.ndarray | None:
    if coordinates is None:
        return None
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (expected_rows, 2):
        raise ValueError(f"{label} coordinates must have shape (samples, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} coordinates must contain only finite values")
    return values


def merge_rows_by_id(
    existing_embeddings: np.ndarray,
    existing_metadata: pd.DataFrame,
    incoming_embeddings: np.ndarray,
    incoming_metadata: pd.DataFrame,
    *,
    existing_coordinates: np.ndarray | None = None,
    incoming_coordinates: np.ndarray | None = None,
) -> RowMergeResult:
    """Replace rows by ID and append incoming IDs not in the state.

    Existing row order is stable.  New rows retain their order in the incoming
    batch.  Metadata columns are the union of both schemas, so a batch may add
    columns without invalidating the stored state.
    """

    existing_values = np.asarray(existing_embeddings, dtype=np.float64)
    incoming_values = np.asarray(incoming_embeddings, dtype=np.float64)
    if existing_values.ndim != 2 or incoming_values.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if existing_values.shape[1] != incoming_values.shape[1]:
        raise ValueError("incremental embeddings have a different dimensionality")
    if len(existing_metadata) != len(existing_values):
        raise ValueError("existing metadata must align with embeddings")
    if len(incoming_metadata) != len(incoming_values):
        raise ValueError("incoming metadata must align with embeddings")
    if "id" not in existing_metadata.columns or "id" not in incoming_metadata.columns:
        raise ValueError("metadata must contain an 'id' column")

    existing_frame = existing_metadata.copy().reset_index(drop=True)
    incoming_frame = incoming_metadata.copy().reset_index(drop=True)
    existing_ids = existing_frame["id"].tolist()
    incoming_ids = incoming_frame["id"].tolist()
    existing_positions = _positions(existing_ids, label="existing")
    incoming_positions = _positions(incoming_ids, label="incoming")
    replaced_ids = [
        identifier for identifier in incoming_ids if identifier in existing_positions
    ]
    appended_ids = [
        identifier for identifier in incoming_ids if identifier not in existing_positions
    ]

    merged_embeddings = existing_values.copy()
    existing_coordinate_values = _validate_coordinates(
        existing_coordinates,
        len(existing_values),
        label="existing",
    )
    incoming_coordinate_values = _validate_coordinates(
        incoming_coordinates,
        len(incoming_values),
        label="incoming",
    )
    if (existing_coordinate_values is None) != (incoming_coordinate_values is None):
        raise ValueError(
            "existing and incoming coordinates must either both be provided or both omitted"
        )
    merged_coordinates = (
        None
        if existing_coordinate_values is None and incoming_coordinate_values is None
        else (
            existing_coordinate_values.copy()
            if existing_coordinate_values is not None
            else np.zeros((len(existing_values), 2), dtype=np.float64)
        )
    )

    columns = list(dict.fromkeys([*existing_frame.columns, *incoming_frame.columns]))
    existing_aligned = existing_frame.reindex(columns=columns).astype(object)
    incoming_aligned = incoming_frame.reindex(columns=columns).astype(object)
    merged_metadata = existing_aligned.copy()
    for identifier in replaced_ids:
        existing_index = existing_positions[identifier]
        incoming_index = incoming_positions[identifier]
        merged_embeddings[existing_index] = incoming_values[incoming_index]
        merged_metadata.iloc[existing_index, :] = incoming_aligned.iloc[
            incoming_index
        ].to_numpy()
        if merged_coordinates is not None and incoming_coordinate_values is not None:
            merged_coordinates[existing_index] = incoming_coordinate_values[incoming_index]

    if appended_ids:
        appended_indices = [incoming_positions[identifier] for identifier in appended_ids]
        merged_embeddings = np.vstack(
            [merged_embeddings, incoming_values[appended_indices]]
        )
        merged_metadata = pd.concat(
            [merged_metadata, incoming_aligned.iloc[appended_indices]],
            ignore_index=True,
        )
        if merged_coordinates is not None:
            if incoming_coordinate_values is None:
                raise ValueError(
                    "incoming coordinates are required when state coordinates exist"
                )
            merged_coordinates = np.vstack(
                [merged_coordinates, incoming_coordinate_values[appended_indices]]
            )

    return RowMergeResult(
        embeddings=merged_embeddings,
        metadata=merged_metadata.reset_index(drop=True),
        replaced_ids=replaced_ids,
        appended_ids=appended_ids,
        coordinates=merged_coordinates,
    )


def batch_fingerprint(embeddings: np.ndarray, metadata: pd.DataFrame) -> str:
    """Return a deterministic fingerprint for the complete incoming batch."""

    values = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float64))
    frame = metadata.reset_index(drop=True)
    descriptor = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": len(frame),
        "values": frame.to_json(
            orient="split",
            date_format="iso",
            force_ascii=False,
            default_handler=str,
        ),
    }
    digest = hashlib.sha256()
    digest.update(b"incremental-batch-v1\0")
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes(order="C"))
    digest.update(
        json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def resolve_batch_id(batch_id: str | None, fingerprint: str) -> str:
    """Use an explicit ID or derive an idempotent ID from the batch content."""

    if batch_id is None:
        return f"auto:{fingerprint}"
    value = str(batch_id).strip()
    if not value:
        raise ValueError("batch_id must not be empty")
    if len(value) > 512:
        raise ValueError("batch_id must be at most 512 characters")
    return value


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def find_processed_batch(
    history: dict[str, dict[str, Any]] | None,
    batch_id: str,
    fingerprint: str,
) -> dict[str, Any] | None:
    """Return a replay record or reject an ID reused with different content."""

    if not history:
        return None
    record = history.get(batch_id)
    if record is None:
        return None
    if record.get("fingerprint") != fingerprint:
        raise ValueError(
            f"batch_id {batch_id!r} was already processed with different content"
        )
    return dict(record)


def remember_processed_batch(
    history: dict[str, dict[str, Any]] | None,
    *,
    batch_id: str,
    fingerprint: str,
    summary: dict[str, Any],
    generation: int,
) -> dict[str, dict[str, Any]]:
    """Add a compact replay record and retain only the newest batch IDs."""

    updated = dict(history or {})
    updated[batch_id] = {
        "fingerprint": fingerprint,
        "generation": int(generation),
        "summary": _plain_value(summary),
    }
    while len(updated) > BATCH_HISTORY_LIMIT:
        oldest = next(iter(updated))
        del updated[oldest]
    return updated


def replay_summary(record: dict[str, Any], *, batch_id: str) -> dict[str, Any]:
    summary = dict(record.get("summary", {}))
    summary["batch_id"] = batch_id
    summary["idempotent_replay"] = True
    return summary


def checked_state_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a pickle payload with a checksum while retaining legacy payloads."""

    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "format": STATE_ENVELOPE_FORMAT,
        "checksum_algorithm": "sha256",
        "checksum": hashlib.sha256(body).hexdigest(),
        "payload_bytes": body,
    }


def unwrap_state_envelope(value: Any) -> dict[str, Any]:
    """Validate a new envelope and return its payload, or pass through legacy data."""

    if not isinstance(value, dict) or value.get("format") != STATE_ENVELOPE_FORMAT:
        if isinstance(value, dict):
            return value
        raise ValueError("Invalid incremental state payload")
    expected = value.get("checksum")
    payload_bytes = value.get("payload_bytes")
    if not isinstance(expected, str):
        raise ValueError("Invalid incremental state checksum envelope")
    if isinstance(payload_bytes, bytes):
        body = payload_bytes
        try:
            payload = pickle.loads(body)
        except Exception as error:
            raise ValueError("Invalid incremental state payload bytes") from error
    else:
        # Accept envelopes produced by the first development version of this
        # feature, where the decoded payload was stored directly.
        payload = value.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Invalid incremental state checksum envelope")
        body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    actual = hashlib.sha256(body).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError("Incremental state checksum mismatch")
    if not isinstance(payload, dict):
        raise ValueError("Invalid incremental state payload")
    return payload


@contextmanager
def state_file_lock(path: Path) -> Iterator[None]:
    """Serialize state-file updates using a sibling advisory lock."""

    try:
        import fcntl
    except ImportError:  # pragma: no cover - the supported runtime is POSIX.
        yield
        return

    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_pickle_dump(value: Any, path: Path) -> None:
    """Write a pickle atomically using a process-specific temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
