from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.fs_utils import atomic_replace_with_retry


Row = Dict[str, str]

EXTRACTION_NEW = "NEW"
EXTRACTION_EXTRACTED = "EXTRACTED"
EXTRACTION_FAILED = "FAILED"
EXTRACTION_MISSING = "MISSING"

CHUNK_PENDING = "PENDING"
CHUNK_CHUNKED = "CHUNKED"
CHUNK_FAILED = "FAILED"

INDEX_PENDING = "PENDING"
INDEX_INDEXED = "INDEXED"
INDEX_FAILED = "FAILED"

REGISTRY_FIELDS = [
    "source_id",
    "content_hash",
    "filename",
    "ext",
    "nature",
    "origin_path",
    "canonical_path",
    "extracted_path",
    "size_bytes",
    "modified_time",
    "last_seen_time",
    "extraction_status",
    "extraction_error",
    "last_extracted_time",
    "chunk_status",
    "nb_chunks",
    "chunk_config_hash",
    "chunk_error",
    "last_chunk_time",
    "index_status",
    "indexed_chunks",
    "index_config_hash",
    "index_error",
    "last_index_time",
]


def normalize_registry_row(row: Row) -> Row:
    return {field: row.get(field, "") for field in REGISTRY_FIELDS}


def clear_chunk_tracking(row: Row) -> None:
    row["chunk_status"] = ""
    row["nb_chunks"] = ""
    row["last_chunk_time"] = ""
    row["chunk_error"] = ""
    row["chunk_config_hash"] = ""
    clear_index_tracking(row)


def set_chunk_pending(row: Row) -> None:
    row["chunk_status"] = CHUNK_PENDING
    row["nb_chunks"] = ""
    row["last_chunk_time"] = ""
    row["chunk_error"] = ""
    row["chunk_config_hash"] = ""
    clear_index_tracking(row)


def clear_index_tracking(row: Row) -> None:
    row["index_status"] = ""
    row["indexed_chunks"] = ""
    row["last_index_time"] = ""
    row["index_error"] = ""
    row["index_config_hash"] = ""


def set_index_pending(row: Row) -> None:
    row["index_status"] = INDEX_PENDING
    row["indexed_chunks"] = ""
    row["last_index_time"] = ""
    row["index_error"] = ""
    row["index_config_hash"] = ""


def load_registry_rows(path: Path) -> List[Row]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [normalize_registry_row(row) for row in reader]


def write_registry_rows(
    path: Path,
    rows: List[Row],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_replace_with_retry(temp_path, path, logger=logger)
