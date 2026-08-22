"""Deterministic BGE embedding production for the Wikipedia benchmark.

The module deliberately keeps the model dependency at call time.  This makes
the release pipeline usable on a CPU-only machine and allows the unit tests to
inject tiny tokenizer/model doubles without downloading model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import (
    DEFAULT_CONFIG,
    DatasetError,
    gzip_jsonl_read,
    jsonl_read,
    load_config,
)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
POOLING = "cls"
DEFAULT_BATCH_SIZE = 16


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def input_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the row-aligned input, including raw text and all metadata."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_json_bytes(dict(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def l2_normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D array, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embedding values must be finite")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embedding rows must have non-zero norm")
    return (matrix / norms).astype(np.float32, copy=False)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _last_hidden_state(output: Any) -> Any:
    if isinstance(output, Mapping):
        return output["last_hidden_state"]
    value = getattr(output, "last_hidden_state", None)
    if value is not None:
        return value
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def encode_batch(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Encode raw passage text using CLS pooling and L2 normalization."""
    if not texts:
        raise ValueError("cannot encode an empty batch")
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if hasattr(encoded, "items"):
        encoded = {
            key: (value.to(device) if hasattr(value, "to") else value)
            for key, value in encoded.items()
        }
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    if torch is not None:
        with torch.no_grad():
            output = model(**encoded)
    else:
        output = model(**encoded)
    hidden = _last_hidden_state(output)
    array = _as_numpy(hidden)
    if array.ndim != 3 or array.shape[0] != len(texts):
        raise ValueError(f"model output must have shape (batch, tokens, dim), got {array.shape}")
    # BGE's documented pooling for this checkpoint is the first (CLS) token.
    return l2_normalize(array[:, 0, :].astype(np.float32, copy=False))


def load_chunk_rows(path: Path) -> list[dict[str, Any]]:
    """Read chunks from JSONL, gzip JSONL, package directory, or package tar."""
    import tarfile

    if path.is_dir():
        for candidate in (path / "chunks.jsonl.gz", path / "chunks.jsonl"):
            if candidate.exists():
                return gzip_jsonl_read(candidate) if candidate.suffix == ".gz" else jsonl_read(candidate)
        raise DatasetError(f"package directory has no chunks artifact: {path}")
    if path.suffix == ".gz":
        return gzip_jsonl_read(path)
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path, "r:gz") as archive:
            names = [name for name in archive.getnames() if name.endswith("chunks.jsonl.gz") or name.endswith("chunks.jsonl")]
            if not names:
                raise DatasetError(f"package archive has no chunks artifact: {path}")
            member = archive.extractfile(sorted(names)[0])
            if member is None:
                raise DatasetError(f"cannot read chunks artifact from {path}")
            import gzip
            if names[0].endswith(".gz"):
                with gzip.open(member, "rt", encoding="utf-8") as handle:
                    return [json.loads(line) for line in handle if line.strip()]
            return [json.loads(line) for line in member.read().decode("utf-8").splitlines() if line.strip()]
    return jsonl_read(path)


def group_document_rows(chunk_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Return stable document metadata and chunk-index groups."""
    groups: dict[str, list[int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(chunk_rows):
        source_id = str(row.get("source_id", ""))
        if not source_id:
            raise DatasetError(f"chunk row {index} has no source_id")
        groups.setdefault(source_id, []).append(index)
        if source_id not in metadata:
            metadata[source_id] = {
                key: row[key] for key in ("source_id", "split", "top", "parent", "leaf") if key in row
            }
            metadata[source_id].setdefault("id", source_id)
        else:
            for key in ("split", "top", "parent", "leaf"):
                if key in row and metadata[source_id].get(key) != row[key]:
                    raise DatasetError(f"document {source_id} has inconsistent {key}")
    ids = sorted(groups)
    return [metadata[source_id] for source_id in ids], [groups[source_id] for source_id in ids]


def document_mean_embeddings(chunk_embeddings: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    values = np.asarray(chunk_embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("chunk_embeddings must be 2D")
    means = np.asarray([np.mean(values[list(indices)], axis=0) for indices in groups], dtype=np.float32)
    return l2_normalize(means)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


@dataclass
class EmbeddingResult:
    chunk_embeddings: np.ndarray
    document_embeddings: np.ndarray
    chunk_metadata: list[dict[str, Any]]
    document_metadata: list[dict[str, Any]]
    manifest: dict[str, Any]


def _load_model(model_name: str, revision: str, device: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to embed BGE passages") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModel.from_pretrained(model_name, revision=revision)
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return tokenizer, model


def embed_chunks(
    input_path: Path,
    output_dir: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
    resume: bool = False,
    tokenizer: Any | None = None,
    model: Any | None = None,
    config_path: Path = DEFAULT_CONFIG,
) -> EmbeddingResult:
    """Embed every chunk and derive one normalized mean vector per document."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = load_config(config_path)
    model_name = str(config["tokenizer"]["model"])
    revision = str(config["tokenizer"]["revision"])
    if model_name != MODEL_NAME or revision != MODEL_REVISION:
        raise DatasetError("embedding config must use the pinned BGE checkpoint")
    rows = load_chunk_rows(input_path)
    if not rows:
        raise DatasetError("chunk input is empty")
    for index, row in enumerate(rows):
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise DatasetError(f"chunk row {index} has empty text")
    checksum = input_checksum(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "embedding_checkpoint.json"
    previous: dict[str, Any] = {}
    if checkpoint_path.exists():
        try:
            previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError("embedding checkpoint is invalid") from exc
        if previous.get("input_sha256") != checksum or previous.get("model") != model_name or previous.get("revision") != revision:
            raise DatasetError("embedding checkpoint input checksum/model mismatch")
        if not resume and previous.get("completed_rows", 0) < len(rows):
            raise DatasetError("incomplete embedding exists; pass resume=True to continue")
        if int(previous.get("completed_rows", 0)) >= len(rows):
            required = [output_dir / name for name in ("chunk_embeddings.npy", "document_embeddings.npy", "chunk_metadata.jsonl", "document_metadata.jsonl", "manifest.json")]
            if all(path.exists() for path in required):
                manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("input_sha256") != checksum:
                    raise DatasetError("embedding manifest input checksum mismatch")
                chunk_array = np.asarray(np.load(output_dir / "chunk_embeddings.npy"), dtype=np.float32)
                document_array = np.asarray(np.load(output_dir / "document_embeddings.npy"), dtype=np.float32)
                chunk_metadata = jsonl_read(output_dir / "chunk_metadata.jsonl")
                document_metadata = jsonl_read(output_dir / "document_metadata.jsonl")
                if len(chunk_metadata) != len(rows) or chunk_array.shape[0] != len(rows):
                    raise DatasetError("completed embedding artifacts are not row aligned")
                return EmbeddingResult(chunk_array, document_array, chunk_metadata, document_metadata, manifest)
    if tokenizer is None or model is None:
        tokenizer, model = _load_model(model_name, revision, device)
    completed = int(previous.get("completed_rows", 0)) if resume else 0
    if completed < 0 or completed > len(rows):
        raise DatasetError("invalid completed_rows in checkpoint")
    partial_path = output_dir / "chunk_embeddings.partial.npy"
    if completed and not partial_path.exists():
        raise DatasetError("checkpoint says rows are complete but partial embedding is missing")
    if completed:
        chunk_vectors = np.load(partial_path, mmap_mode="r+")
        if chunk_vectors.shape[0] != len(rows):
            raise DatasetError("partial embedding row count does not match input")
    else:
        chunk_vectors = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=(len(rows), 0))
        # Probe model dimension using the first batch before allocating the final memmap.
        first = encode_batch([str(rows[0]["text"])], tokenizer, model, device=device)
        chunk_vectors = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=(len(rows), first.shape[1]))
        chunk_vectors[0] = first[0]
        completed = 1
    for start in range(completed, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        chunk_vectors[start:end] = encode_batch([str(row["text"]) for row in rows[start:end]], tokenizer, model, device=device)
        chunk_vectors.flush()
        _atomic_json(checkpoint_path, {"schema_version": 1, "input_sha256": checksum, "model": model_name, "revision": revision, "pooling": POOLING, "dtype": "float32", "dimension": int(chunk_vectors.shape[1]), "total_rows": len(rows), "completed_rows": end})
    chunk_array = np.asarray(chunk_vectors, dtype=np.float32).copy()
    if not np.all(np.isfinite(chunk_array)):
        raise DatasetError("chunk embeddings contain non-finite values")
    np.save(output_dir / "chunk_embeddings.npy", chunk_array)
    document_metadata, groups = group_document_rows(rows)
    document_array = document_mean_embeddings(chunk_array, groups)
    jsonl_path = output_dir / "chunk_metadata.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    with (output_dir / "document_metadata.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in document_metadata:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    np.save(output_dir / "document_embeddings.npy", document_array)
    manifest = {"schema_version": 1, "input": str(input_path), "input_sha256": checksum, "model": model_name, "revision": revision, "pooling": POOLING, "dtype": "float32", "dimension": int(chunk_array.shape[1]), "chunk_rows": len(rows), "document_rows": len(document_array), "completed_rows": len(rows), "chunk_embeddings": "chunk_embeddings.npy", "document_embeddings": "document_embeddings.npy", "chunk_metadata": "chunk_metadata.jsonl", "document_metadata": "document_metadata.jsonl"}
    _atomic_json(output_dir / "manifest.json", manifest)
    _atomic_json(checkpoint_path, {**manifest, "completed_rows": len(rows)})
    return EmbeddingResult(chunk_array, document_array, [dict(row) for row in rows], document_metadata, manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed Wikipedia chunks with pinned BGE CLS embeddings")
    parser.add_argument("--input", "--package", dest="input_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = embed_chunks(args.input_path, args.output_dir, batch_size=args.batch_size, device=args.device, resume=args.resume)
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
