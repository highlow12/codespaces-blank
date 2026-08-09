"""Build and read row-addressable float32 embedding caches."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import pandas as pd

from embedding_data import (
    EMBEDDING_METADATA_SCHEMA_VERSION,
    normalize_embedding_record,
)


CACHE_FORMAT = "row-addressable-embedding-cache"
CACHE_VERSION = 1
CACHE_MANIFEST_NAME = "manifest.json"
CACHE_DATA_NAME = "embeddings.f32"
CACHE_METADATA_NAME = "metadata.jsonl"
CACHE_METADATA_OFFSETS_NAME = "metadata_offsets.npy"
CACHE_DTYPE = np.dtype("<f4")
JSON_CHUNK_SIZE = 1024 * 1024


class _JsonStream:
    """Decode individual JSON values without loading the whole document."""

    def __init__(self, handle: TextIO, *, chunk_size: int = JSON_CHUNK_SIZE) -> None:
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self) -> None:
        if self.eof:
            return
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def _skip_whitespace(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[
                self.position
            ].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._fill()

    def peek(self) -> str | None:
        self._skip_whitespace()
        if self.position >= len(self.buffer):
            return None
        return self.buffer[self.position]

    def take(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected {expected!r}, found {actual!r}")
        self.position += 1

    def decode(self) -> Any:
        self._skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as error:
                if self.eof:
                    raise ValueError("Invalid or truncated JSON input") from error
                self._fill()
            else:
                self.position = end
                return value


def _iter_json_array(reader: _JsonStream) -> Iterator[Any]:
    reader.take("[")
    first = True
    while True:
        if not first:
            if reader.peek() == "]":
                reader.take("]")
                return
            reader.take(",")
            if reader.peek() == "]":
                raise ValueError("Trailing comma in embedding records")
        if reader.peek() == "]":
            reader.take("]")
            return
        yield reader.decode()
        first = False


def _iter_json_records(handle: TextIO) -> Iterator[Any]:
    """Yield records from a top-level JSON list or an object.records list."""

    reader = _JsonStream(handle)
    first = reader.peek()
    if first == "[":
        yield from _iter_json_array(reader)
        return
    if first != "{":
        raise ValueError("Embedding JSON must contain a list or records object")

    reader.take("{")
    first_field = True
    found_records = False
    while True:
        if not first_field:
            if reader.peek() == "}":
                reader.take("}")
                break
            reader.take(",")
        if reader.peek() == "}":
            reader.take("}")
            break
        key = reader.decode()
        if not isinstance(key, str):
            raise ValueError("Embedding JSON object keys must be strings")
        reader.take(":")
        if key == "records":
            if found_records:
                raise ValueError("Embedding JSON contains duplicate records fields")
            found_records = True
            yield from _iter_json_array(reader)
        else:
            # The known dataset only has records at the top level. Decode and
            # discard any auxiliary fields while keeping the parser generic.
            reader.decode()
        first_field = False
    if not found_records:
        raise ValueError("Embedding JSON object has no records field")


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(cache_path: Path) -> Path:
    path = Path(cache_path)
    if path.is_dir():
        path = path / CACHE_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Cache manifest not found: {path}")
    return path


def read_cache_manifest(cache_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = _manifest_path(cache_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid cache manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Cache manifest must be a JSON object")
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError("Unsupported embedding cache format")
    if int(manifest.get("version", -1)) != CACHE_VERSION:
        raise ValueError("Unsupported embedding cache version")
    return manifest_path, manifest


def cache_record_count(cache_path: Path) -> int:
    _, manifest = read_cache_manifest(cache_path)
    record_count = int(manifest.get("record_count", 0))
    if record_count < 1:
        raise ValueError("Cache manifest must contain at least one record")
    return record_count


def build_embedding_cache(
    input_path: Path,
    output_dir: Path,
    *,
    id_offset: int = 0,
) -> Path:
    """Stream JSON records into a float32 matrix and indexed metadata files."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Embedding input not found: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Cache output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / CACHE_DATA_NAME
    metadata_path = output_dir / CACHE_METADATA_NAME
    offsets_path = output_dir / CACHE_METADATA_OFFSETS_NAME
    manifest_path = output_dir / CACHE_MANIFEST_NAME
    offsets: list[int] = []
    ids: set[Any] = set()
    record_count = 0
    embedding_dimension: int | None = None

    with (
        _open_text(input_path) as source,
        data_path.open("wb") as data_file,
        metadata_path.open("wb") as metadata_file,
    ):
        for index, record in enumerate(_iter_json_records(source)):
            embedding, metadata = normalize_embedding_record(
                record,
                index=index,
                id_offset=id_offset,
            )
            if embedding_dimension is None:
                embedding_dimension = int(embedding.shape[0])
            elif embedding.shape[0] != embedding_dimension:
                raise ValueError(
                    "Embeddings have inconsistent dimensions: "
                    f"record {index} has {embedding.shape[0]}, "
                    f"expected {embedding_dimension}"
                )
            try:
                if metadata["id"] in ids:
                    raise ValueError(
                        f"Embedding IDs are not unique at index {index}"
                    )
                ids.add(metadata["id"])
            except TypeError as error:
                raise ValueError(
                    f"Embedding ID at index {index} is not hashable"
                ) from error

            stored_embedding = np.asarray(embedding, dtype=CACHE_DTYPE)
            if not np.all(np.isfinite(stored_embedding)):
                raise ValueError(
                    f"Embedding at index {index} cannot be represented as float32"
                )
            stored_embedding.tofile(data_file)
            offsets.append(metadata_file.tell())
            metadata_file.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            metadata_file.write(b"\n")
            record_count += 1

    if record_count == 0 or embedding_dimension is None:
        raise ValueError(f"No embedding records found in {input_path}")
    np.save(
        offsets_path,
        np.asarray(offsets, dtype=np.uint64),
        allow_pickle=False,
    )
    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "source": {
            "path": str(input_path.resolve()),
            "sha256": _sha256_file(input_path),
            "size_bytes": input_path.stat().st_size,
        },
        "record_count": record_count,
        "embedding": {
            "path": CACHE_DATA_NAME,
            "dtype": CACHE_DTYPE.str,
            "shape": [record_count, embedding_dimension],
            "order": "C",
        },
        "metadata": {
            "path": CACHE_METADATA_NAME,
            "offsets_path": CACHE_METADATA_OFFSETS_NAME,
            "schema_version": EMBEDDING_METADATA_SCHEMA_VERSION,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _selection_indices(
    record_count: int,
    *,
    start: int,
    limit: int | None,
    sample_size: int | None,
    sample_seed: int,
) -> np.ndarray:
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")
    if start >= record_count:
        raise ValueError(
            f"No embedding records selected (start={start}, limit={limit})"
        )
    end = record_count if limit is None else min(record_count, start + limit)
    indices = np.arange(start, end, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError(
            f"No embedding records selected (start={start}, limit={limit})"
        )
    if sample_size is None:
        return indices
    if sample_size < 1 or sample_size > len(indices):
        raise ValueError(
            "sample_size must be between 1 and the selected number of embeddings"
        )
    rng = np.random.default_rng(sample_seed)
    selected = np.sort(rng.choice(len(indices), size=sample_size, replace=False))
    return indices[selected]


def load_embeddings_from_cache(
    cache_path: Path,
    *,
    start: int = 0,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load only selected rows from a cache without decoding source JSON."""

    manifest_path, manifest = read_cache_manifest(cache_path)
    record_count = int(manifest["record_count"])
    embedding = manifest.get("embedding")
    metadata = manifest.get("metadata")
    if not isinstance(embedding, dict) or not isinstance(metadata, dict):
        raise ValueError("Cache manifest is missing embedding or metadata details")
    shape = embedding.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("Cache manifest has an invalid embedding shape")
    try:
        rows, dimensions = (int(shape[0]), int(shape[1]))
    except (TypeError, ValueError) as error:
        raise ValueError("Cache manifest has an invalid embedding shape") from error
    if rows != record_count or dimensions < 1:
        raise ValueError("Cache manifest has an invalid embedding shape")
    try:
        dtype = np.dtype(embedding.get("dtype", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("Cache manifest has an invalid embedding dtype") from error
    if dtype != CACHE_DTYPE:
        raise ValueError(f"Unsupported cache dtype: {dtype}")

    data_path = manifest_path.parent / str(embedding["path"])
    metadata_path = manifest_path.parent / str(metadata["path"])
    offsets_path = manifest_path.parent / str(metadata["offsets_path"])
    expected_data_bytes = rows * dimensions * dtype.itemsize
    if not data_path.is_file() or data_path.stat().st_size != expected_data_bytes:
        raise ValueError("Cache embedding data file is missing or has wrong size")
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    if offsets.shape != (rows,) or offsets.dtype != np.dtype(np.uint64):
        raise ValueError("Cache metadata offsets have an invalid shape or dtype")
    if not metadata_path.is_file():
        raise ValueError("Cache metadata file is missing")

    indices = _selection_indices(
        rows,
        start=start,
        limit=limit,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    matrix = np.memmap(
        data_path,
        mode="r",
        dtype=dtype,
        shape=(rows, dimensions),
        order="C",
    )
    values = np.array(matrix[indices], dtype=CACHE_DTYPE, copy=True)
    metadata_rows: list[dict[str, Any]] = []
    ids: set[Any] = set()
    with metadata_path.open("rb") as metadata_file:
        for index in indices.tolist():
            metadata_file.seek(int(offsets[index]))
            line = metadata_file.readline()
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid metadata row at cache index {index}"
                ) from error
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError(f"Invalid metadata row at cache index {index}")
            try:
                if row["id"] in ids:
                    raise ValueError(
                        f"Embedding IDs are not unique at cache index {index}"
                    )
                ids.add(row["id"])
            except TypeError as error:
                raise ValueError(
                    f"Embedding ID at cache index {index} is not hashable"
                ) from error
            metadata_rows.append(row)
    return values, pd.DataFrame(metadata_rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a row-addressable float32 embedding cache."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Build a cache directory.")
    build_parser.add_argument("--input-json", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--id-offset", type=int, default=0)
    build_parser.set_defaults(handler=_run_build)
    return parser


def _run_build(args: argparse.Namespace) -> None:
    manifest_path = build_embedding_cache(
        args.input_json,
        args.output_dir,
        id_offset=args.id_offset,
    )
    print(f"Embedding cache saved: {manifest_path}")


def main() -> None:
    args = _build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
