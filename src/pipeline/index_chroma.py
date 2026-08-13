from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.common_utils import to_int, utc_now_iso
from src.utils.logging_utils import build_script_logger
from src.integrations.llm.openai_utils import build_openai_client, classify_openai_error
from src.core.registry import (
    CHUNK_CHUNKED,
    EXTRACTION_EXTRACTED,
    INDEX_FAILED,
    INDEX_INDEXED,
    Row,
    clear_index_tracking,
    load_registry_rows,
    write_registry_rows,
)
from src.utils.type_hints import ChromaCollection, OpenAIClient
from src.integrations.vector.chroma_utils import build_chroma_collection
from src.utils.paths import INDEX_DIR, LEAF_CHUNKS_DIR, LOGS_DIR, REGISTRY_DIR

load_dotenv()

REGISTRY_PATH = (REGISTRY_DIR / "sources.csv").resolve()
LEAF_DIR = LEAF_CHUNKS_DIR.resolve()
INDEX_DIR = INDEX_DIR.resolve()
LOG_PATH = (LOGS_DIR / "index_chroma.log").resolve()

INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_COLLECTION_NAME = "rag_leaf_chunks"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

LOGGER = build_script_logger("index_chroma", LOG_PATH)


@dataclass(frozen=True)
class UpsertStats:
    total_chunks: int
    newly_embedded: int
    unchanged: int
    stale_deleted: int


def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Index leaf chunks into a local persistent Chroma vector store.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reindex even if index_status is INDEXED and config is unchanged.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Restrict processing to one or more source_id values.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of sources processed (0 = no limit).",
    )
    parser.add_argument(
        "--keep-stale-index",
        action="store_true",
        help="Do not delete vector entries for sources that are no longer CHUNKED.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "").strip(),
        help="OpenAI API key (default: OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=60.0,
        help="OpenAI request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for OpenAI embedding calls (default: 5).",
    )
    parser.add_argument(
        "--retry-base-delay-sec",
        type=float,
        default=1.0,
        help="Base delay for retry backoff (default: 1.0s).",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.request_timeout_sec <= 0:
        raise ValueError("--request-timeout-sec must be > 0")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.retry_base_delay_sec <= 0:
        raise ValueError("--retry-base-delay-sec must be > 0")
    return args


def compute_index_config_hash(collection_name: str, embedding_model: str) -> str:
    payload = {
        "collection_name": collection_name,
        "embedding_model": embedding_model,
    }
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_registry() -> List[Row]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    return load_registry_rows(REGISTRY_PATH)


def write_registry(rows: List[Row]) -> None:
    write_registry_rows(REGISTRY_PATH, rows, logger=LOGGER)


def leaf_path(source_id: str) -> Path:
    return LEAF_DIR / f"{source_id}.jsonl"


def should_process_row(row: Row, config_hash: str, force: bool, source_filter: set[str]) -> bool:
    if row.get("extraction_status") != EXTRACTION_EXTRACTED:
        return False
    if row.get("chunk_status") != CHUNK_CHUNKED:
        return False

    source_id = row.get("source_id", "")
    if not source_id:
        return False
    if source_filter and source_id not in source_filter:
        return False

    if force:
        return True

    if not leaf_path(source_id).exists():
        return True
    if row.get("index_status", "") != INDEX_INDEXED:
        return True
    if row.get("index_config_hash", "") != config_hash:
        return True
    if row.get("indexed_chunks", "") != row.get("nb_chunks", ""):
        return True
    return False


def load_leaf_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Leaf chunk file not found: {path}")

    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                continue
            records.append(payload)
    return records


def flatten_chunk_record(
    record: Dict[str, object],
    source_id: str,
    embedding_model: str,
) -> Tuple[str, str, Dict[str, str | int | float | bool]]:
    chunk_id = str(record.get("chunk_id", "")).strip()
    chunk_text = str(record.get("text", "") or "")
    if not chunk_id:
        raise ValueError("chunk_id missing in leaf record")
    if not chunk_text.strip():
        raise ValueError(f"chunk text empty for chunk_id={chunk_id}")

    meta_raw = record.get("meta")
    meta: Dict[str, object] = dict(meta_raw) if isinstance(meta_raw, dict) else {}

    metadata: Dict[str, str | int | float | bool] = {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "level": str(record.get("level", "leaf")),
        "parent_id": str(record.get("parent_id", "")),
        "prev_id": str(record.get("prev_id", "")),
        "next_id": str(record.get("next_id", "")),
        "token_count": to_int(record.get("token_count"), 0),
        "start_char": to_int(record.get("start_char"), 0),
        "end_char": to_int(record.get("end_char"), 0),
        "unit_type_start": str(record.get("unit_type_start", "")),
        "unit_index_start": to_int(record.get("unit_index_start"), 0),
        "unit_type_end": str(record.get("unit_type_end", "")),
        "unit_index_end": to_int(record.get("unit_index_end"), 0),
        "filename": str(meta.get("filename", "")),
        "nature": str(meta.get("nature", "")),
        "ext": str(meta.get("ext", "")),
        "origin_path": str(meta.get("origin_path", "")),
        "canonical_path": str(meta.get("canonical_path", "")),
        "modified_time": str(meta.get("modified_time", "")),
        "tokenizer": str(meta.get("tokenizer", "")),
        "embedding_model": embedding_model,
    }
    return chunk_id, chunk_text, metadata


def embed_batch(
    openai_client: OpenAIClient,
    texts: Sequence[str],
    embedding_model: str,
    max_retries: int,
    retry_base_delay_sec: float,
) -> List[List[float]]:
    if not texts:
        return []

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = openai_client.embeddings.create(model=embedding_model, input=list(texts))
            embeddings = [item.embedding for item in response.data]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding response size mismatch: expected={len(texts)} got={len(embeddings)}"
                )
            return embeddings
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retryable, retry_after, reason = classify_openai_error(exc)
            if not retryable:
                LOGGER.exception("Non-retryable embedding error (%s): %s", reason, exc)
                raise
            if attempt >= max_retries:
                break
            delay = retry_after if retry_after is not None else retry_base_delay_sec * (2**attempt)
            LOGGER.warning(
                "Retryable embedding error (%s) attempt %d/%d. Retrying in %.2fs | %s",
                reason,
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
            )
            time.sleep(delay)

    if last_error is None:
        raise RuntimeError("Embedding request failed with unknown error.")
    raise last_error


def normalize_collection_ids(raw_ids: object) -> List[str]:
    if not isinstance(raw_ids, list):
        return []

    normalized: List[str] = []
    for item in raw_ids:
        if isinstance(item, str):
            if item:
                normalized.append(item)
            continue
        if isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, str) and sub_item:
                    normalized.append(sub_item)
    return normalized


def fetch_existing_chunk_ids(collection: ChromaCollection, source_id: str) -> set[str]:
    try:
        payload = collection.get(where={"source_id": source_id}, include=[])
    except (TypeError, ValueError):
        payload = collection.get(where={"source_id": source_id})

    if not isinstance(payload, dict):
        return set()
    return set(normalize_collection_ids(payload.get("ids", [])))


def refresh_existing_chunks_metadata(
    collection: ChromaCollection,
    source_id: str,
    ids: List[str],
    docs: List[str],
    metas: List[Dict[str, str | int | float | bool]],
) -> None:
    if not ids:
        return
    if not hasattr(collection, "update"):
        return
    try:
        collection.update(ids=ids, documents=docs, metadatas=metas)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Could not refresh metadata/documents for existing chunks | source=%s | count=%d | %s",
            source_id,
            len(ids),
            exc,
        )


def upsert_source_chunks(
    row: Row,
    collection: ChromaCollection,
    openai_client: OpenAIClient,
    embedding_model: str,
    reembed_all: bool,
    batch_size: int,
    max_retries: int,
    retry_base_delay_sec: float,
) -> UpsertStats:
    source_id = row.get("source_id", "")
    if not source_id:
        raise ValueError("source_id missing in registry row")

    records = load_leaf_records(leaf_path(source_id))
    by_id: Dict[str, Tuple[str, Dict[str, str | int | float | bool]]] = {}
    for record in records:
        chunk_id, chunk_text, metadata = flatten_chunk_record(
            record=record,
            source_id=source_id,
            embedding_model=embedding_model,
        )
        by_id[chunk_id] = (chunk_text, metadata)

    if not by_id:
        raise ValueError("No valid leaf chunks to index")

    desired_ids = set(by_id.keys())
    existing_ids = fetch_existing_chunk_ids(collection=collection, source_id=source_id)

    stale_ids = sorted(existing_ids - desired_ids)
    unchanged_ids = sorted(existing_ids & desired_ids)
    ids_to_embed = sorted(desired_ids) if reembed_all else sorted(desired_ids - existing_ids)

    upsert_ids: List[str] = []
    upsert_docs: List[str] = []
    upsert_metas: List[Dict[str, str | int | float | bool]] = []
    upsert_embeddings: List[List[float]] = []

    for start in range(0, len(ids_to_embed), batch_size):
        end = min(start + batch_size, len(ids_to_embed))
        batch_ids = ids_to_embed[start:end]
        batch_docs = [by_id[chunk_id][0] for chunk_id in batch_ids]
        batch_metas = [by_id[chunk_id][1] for chunk_id in batch_ids]

        avg_chars = sum(len(text) for text in batch_docs) / len(batch_docs)
        max_chars = max(len(text) for text in batch_docs)
        LOGGER.info(
            "[EMBED BATCH] %s | %d-%d/%d | avg_chars=%.1f | max_chars=%d",
            source_id,
            start + 1,
            end,
            len(ids_to_embed),
            avg_chars,
            max_chars,
        )

        batch_embeddings = embed_batch(
            openai_client=openai_client,
            texts=batch_docs,
            embedding_model=embedding_model,
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
        )

        upsert_ids.extend(batch_ids)
        upsert_docs.extend(batch_docs)
        upsert_metas.extend(batch_metas)
        upsert_embeddings.extend(batch_embeddings)

    if upsert_ids:
        collection.upsert(
            ids=upsert_ids,
            embeddings=upsert_embeddings,
            documents=upsert_docs,
            metadatas=upsert_metas,
        )

    if stale_ids:
        collection.delete(ids=stale_ids)

    if unchanged_ids and not reembed_all:
        refresh_existing_chunks_metadata(
            collection=collection,
            source_id=source_id,
            ids=unchanged_ids,
            docs=[by_id[chunk_id][0] for chunk_id in unchanged_ids],
            metas=[by_id[chunk_id][1] for chunk_id in unchanged_ids],
        )

    return UpsertStats(
        total_chunks=len(desired_ids),
        newly_embedded=len(ids_to_embed),
        unchanged=0 if reembed_all else len(unchanged_ids),
        stale_deleted=len(stale_ids),
    )


def purge_stale_index_entries(rows: List[Row], collection: ChromaCollection, keep_stale: bool) -> int:
    if keep_stale:
        return 0

    removed_sources = 0
    for row in rows:
        source_id = row.get("source_id", "")
        if not source_id:
            continue

        is_active = (
            row.get("extraction_status") == EXTRACTION_EXTRACTED
            and row.get("chunk_status") == CHUNK_CHUNKED
            and leaf_path(source_id).exists()
        )
        if is_active:
            continue

        collection.delete(where={"source_id": source_id})
        clear_index_tracking(row)
        removed_sources += 1

    return removed_sources


def index_status_counts(rows: List[Row]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = row.get("index_status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_indexing(
    *,
    force: bool,
    source_ids: Optional[Sequence[str]],
    limit: int,
    keep_stale_index: bool,
    collection_name: str,
    embedding_model: str,
    batch_size: int,
    openai_api_key: str,
    request_timeout_sec: float,
    max_retries: int,
    retry_base_delay_sec: float,
) -> Dict[str, object]:
    config_hash = compute_index_config_hash(
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    rows = load_registry()
    collection = build_chroma_collection(index_dir=INDEX_DIR, collection_name=collection_name)
    openai_client = build_openai_client(
        api_key=openai_api_key,
        timeout_sec=request_timeout_sec,
    )

    stale_removed = purge_stale_index_entries(rows, collection=collection, keep_stale=keep_stale_index)
    if stale_removed:
        LOGGER.info("Purged stale vector entries for %d sources", stale_removed)
        write_registry(rows)

    source_filter = {value.strip() for value in source_ids or [] if value and value.strip()}
    targets = [
        row
        for row in rows
        if should_process_row(
            row=row,
            config_hash=config_hash,
            force=force,
            source_filter=source_filter,
        )
    ]
    if limit > 0:
        targets = targets[:limit]

    LOGGER.info(
        "Indexing started | targets=%d | force=%s | collection=%s | model=%s | config_hash=%s",
        len(targets),
        force,
        collection_name,
        embedding_model,
        config_hash[:12],
    )

    processed = 0
    failed = 0
    for row in tqdm(targets, desc="Indexing"):
        source_id = row.get("source_id", "")
        try:
            reembed_all = force or row.get("index_config_hash", "") != config_hash
            upsert_stats = upsert_source_chunks(
                row=row,
                collection=collection,
                openai_client=openai_client,
                embedding_model=embedding_model,
                reembed_all=reembed_all,
                batch_size=batch_size,
                max_retries=max_retries,
                retry_base_delay_sec=retry_base_delay_sec,
            )
            row["index_status"] = INDEX_INDEXED
            row["indexed_chunks"] = str(upsert_stats.total_chunks)
            row["last_index_time"] = utc_now_iso()
            row["index_error"] = ""
            row["index_config_hash"] = config_hash
            processed += 1
            LOGGER.info(
                "[INDEXED] %s | reembed_all=%s | total=%d | newly_embedded=%d | unchanged=%d | stale_deleted=%d",
                source_id,
                reembed_all,
                upsert_stats.total_chunks,
                upsert_stats.newly_embedded,
                upsert_stats.unchanged,
                upsert_stats.stale_deleted,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            row["index_status"] = INDEX_FAILED
            row["indexed_chunks"] = ""
            row["last_index_time"] = utc_now_iso()
            row["index_error"] = str(exc)
            row["index_config_hash"] = config_hash
            LOGGER.exception("[INDEX FAILED] %s", source_id)

        write_registry(rows)

    counts = index_status_counts(rows)
    LOGGER.info(
        "Indexing completed | processed=%d | failed=%d | index_status_counts=%s",
        processed,
        failed,
        counts,
    )

    return {
        "index_status_counts": counts,
        "processed": processed,
        "failed": failed,
        "index_dir": str(INDEX_DIR),
        "collection": collection_name,
        "embedding_model": embedding_model,
        "config_hash": config_hash,
        "stale_removed": stale_removed,
    }


def main() -> None:  # pragma: no cover
    args = parse_args()
    result = run_indexing(
        force=args.force,
        source_ids=args.source_id,
        limit=args.limit,
        keep_stale_index=args.keep_stale_index,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        openai_api_key=args.openai_api_key,
        request_timeout_sec=args.request_timeout_sec,
        max_retries=args.max_retries,
        retry_base_delay_sec=args.retry_base_delay_sec,
    )
    print("Index status:", result["index_status_counts"])
    print("Processed:", result["processed"])
    print("Failed:", result["failed"])
    print("Index dir:", result["index_dir"])
    print("Collection:", result["collection"])
    print("Embedding model:", result["embedding_model"])


if __name__ == "__main__":  # pragma: no cover
    main()
