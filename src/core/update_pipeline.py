from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

from dotenv import load_dotenv

from src.pipeline.convert_legacy_office_to_modern import (
    DEFAULT_CONVERTED_DIR,
    run_conversion,
)
from src.pipeline.scan_and_extract import run_scan_and_extract
from src.pipeline.chunk_hierarchical import run_chunking
from src.pipeline.index_chroma import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    run_indexing,
)
from src.utils.logging_utils import build_script_logger
from src.utils.paths import LOGS_DIR

load_dotenv()

LOG_PATH = (LOGS_DIR / "update_pipeline.log").resolve()
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOGGER = build_script_logger("update_pipeline", LOG_PATH)


@dataclass(frozen=True)
class ConversionStepConfig:
    enabled: bool = True
    raw_dir: Path | None = None
    converted_dir: Path | None = None
    soffice_path: str = ""
    force: bool = False
    dry_run: bool = False
    timeout_sec: int = 180
    limit: int = 0
    keep_stale_converted: bool = False


@dataclass(frozen=True)
class ScanExtractStepConfig:
    enabled: bool = True
    scan_only: bool = False
    keep_missing_json: bool = False
    reextract_extracted: bool = False


@dataclass(frozen=True)
class ChunkStepConfig:
    enabled: bool = True
    force: bool = False
    source_ids: Sequence[str] = field(default_factory=list)
    limit: int = 0
    keep_stale_chunks: bool = False
    leaf_min: int = 220
    leaf_target: int = 360
    leaf_max: int = 520
    leaf_overlap: int = 70
    parent_min: int = 700
    parent_target: int = 1100
    parent_max: int = 1500
    parent_overlap: int = 180
    min_leaf_output_tokens: int = 0
    tokenizer: str = "tiktoken"
    tiktoken_encoding: str = "cl100k_base"


@dataclass(frozen=True)
class IndexStepConfig:
    enabled: bool = True
    force: bool = False
    source_ids: Sequence[str] = field(default_factory=list)
    limit: int = 0
    keep_stale_index: bool = False
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 64
    openai_api_key: str = ""
    request_timeout_sec: float = 60.0
    max_retries: int = 5
    retry_base_delay_sec: float = 1.0


def resolve_raw_dir(raw_dir: Path | None) -> Path:
    if raw_dir is not None:
        return raw_dir
    raw_value = os.getenv("RAW_DIR", "").strip()
    if not raw_value:
        raise FileNotFoundError("RAW_DIR not set. Define it in .env.")
    return Path(raw_value).expanduser()


def resolve_converted_dir(converted_dir: Path | None) -> Path:
    if converted_dir is not None:
        return converted_dir
    return DEFAULT_CONVERTED_DIR


# noinspection DuplicatedCode
def run_update_pipeline(
    *,
    conversion: ConversionStepConfig | None = None,
    scan_extract: ScanExtractStepConfig | None = None,
    chunk: ChunkStepConfig | None = None,
    index: IndexStepConfig | None = None,
) -> Dict[str, object]:
    conversion = conversion or ConversionStepConfig()
    scan_extract = scan_extract or ScanExtractStepConfig()
    chunk = chunk or ChunkStepConfig()
    index = index or IndexStepConfig()

    results: Dict[str, object] = {}

    if conversion.enabled:
        LOGGER.info("Pipeline: conversion step started")
        result = run_conversion(
            raw_dir=resolve_raw_dir(conversion.raw_dir),
            converted_dir=resolve_converted_dir(conversion.converted_dir),
            soffice_path=conversion.soffice_path,
            force=conversion.force,
            dry_run=conversion.dry_run,
            timeout_sec=conversion.timeout_sec,
            limit=conversion.limit,
            keep_stale_converted=conversion.keep_stale_converted,
        )
        results["convert"] = result

    if scan_extract.enabled:
        LOGGER.info("Pipeline: scan/extract step started")
        result = run_scan_and_extract(
            scan_only=scan_extract.scan_only,
            keep_missing_json=scan_extract.keep_missing_json,
            reextract_extracted=scan_extract.reextract_extracted,
        )
        results["scan_extract"] = result

    if chunk.enabled:
        LOGGER.info("Pipeline: chunking step started")
        result = run_chunking(
            force=chunk.force,
            source_ids=chunk.source_ids,
            limit=chunk.limit,
            keep_stale_chunks=chunk.keep_stale_chunks,
            leaf_min=chunk.leaf_min,
            leaf_target=chunk.leaf_target,
            leaf_max=chunk.leaf_max,
            leaf_overlap=chunk.leaf_overlap,
            parent_min=chunk.parent_min,
            parent_target=chunk.parent_target,
            parent_max=chunk.parent_max,
            parent_overlap=chunk.parent_overlap,
            min_leaf_output_tokens=chunk.min_leaf_output_tokens,
            tokenizer=chunk.tokenizer,
            tiktoken_encoding=chunk.tiktoken_encoding,
        )
        results["chunk"] = result

    if index.enabled:
        LOGGER.info("Pipeline: indexing step started")
        api_key = index.openai_api_key or os.getenv("OPENAI_API_KEY", "").strip()
        result = run_indexing(
            force=index.force,
            source_ids=index.source_ids,
            limit=index.limit,
            keep_stale_index=index.keep_stale_index,
            collection_name=index.collection_name,
            embedding_model=index.embedding_model,
            batch_size=index.batch_size,
            openai_api_key=api_key,
            request_timeout_sec=index.request_timeout_sec,
            max_retries=index.max_retries,
            retry_base_delay_sec=index.retry_base_delay_sec,
        )
        results["index"] = result

    LOGGER.info("Pipeline completed | steps=%s", list(results.keys()))
    return results
