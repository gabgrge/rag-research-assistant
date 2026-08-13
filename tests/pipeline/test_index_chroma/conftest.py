from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.core import registry
from src.pipeline.index_chroma import compute_index_config_hash


@pytest.fixture
def default_collection_name() -> str:
    return "test_leaf_chunks"


@pytest.fixture
def default_embedding_model() -> str:
    return "text-embedding-3-small"


@pytest.fixture
def default_config_hash(default_collection_name: str, default_embedding_model: str) -> str:
    return compute_index_config_hash(
        collection_name=default_collection_name,
        embedding_model=default_embedding_model,
    )


@pytest.fixture
def mock_leaf_record() -> Dict[str, Any]:
    return {
        "chunk_id": "src1_leaf_0",
        "text": "This is a sample leaf chunk text content for embedding.",
        "level": "leaf",
        "parent_id": "src1_parent_0",
        "prev_id": "",
        "next_id": "src1_leaf_1",
        "token_count": 10,
        "start_char": 0,
        "end_char": 55,
        "unit_type_start": "paragraph",
        "unit_index_start": 1,
        "unit_type_end": "paragraph",
        "unit_index_end": 1,
        "meta": {
            "filename": "document.pdf",
            "nature": "report",
            "ext": ".pdf",
            "origin_path": "/data/document.pdf",
            "canonical_path": "/data/document.pdf",
            "modified_time": "2026-01-01T00:00:00Z",
            "tokenizer": "cl100k_base",
        },
    }


@pytest.fixture
def isolate_index_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Redirects registry path, leaf chunks dir, index dir, and log path
    to temporary test directories to keep test execution isolated and clean.
    """
    registry_file = tmp_path / "registry" / "sources.csv"
    leaf_dir = tmp_path / "chunks" / "leaf"
    index_dir = tmp_path / "index"
    logs_dir = tmp_path / "logs"

    registry_file.parent.mkdir(parents=True, exist_ok=True)
    leaf_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.pipeline.index_chroma.REGISTRY_PATH", registry_file)
    monkeypatch.setattr("src.pipeline.index_chroma.LEAF_DIR", leaf_dir)
    monkeypatch.setattr("src.pipeline.index_chroma.INDEX_DIR", index_dir)
    monkeypatch.setattr("src.pipeline.index_chroma.LOG_PATH", logs_dir / "index_chroma.log")

    return {
        "tmp_path": tmp_path,
        "registry_file": registry_file,
        "leaf_dir": leaf_dir,
        "index_dir": index_dir,
    }


@pytest.fixture
def make_registry_entry(isolate_index_environment: Dict[str, Path]):
    """
    Factory fixture to create sample registry entries along with their leaf JSONL chunk files.
    """

    def _create(
            source_id: str,
            extraction_status: str = registry.EXTRACTION_EXTRACTED,
            chunk_status: str = registry.CHUNK_CHUNKED,
            index_status: str = "",
            index_config_hash: str = "",
            nb_chunks: str = "1",
            indexed_chunks: str = "",
            leaf_records: List[Dict[str, Any]] | None = None,
    ) -> registry.Row:
        row: registry.Row = {
            "source_id": source_id,
            "extraction_status": extraction_status,
            "chunk_status": chunk_status,
            "nb_chunks": nb_chunks,
            "index_status": index_status,
            "index_config_hash": index_config_hash,
            "indexed_chunks": indexed_chunks,
            "last_index_time": "",
            "index_error": "",
            "extracted_path": "",
        }

        if leaf_records is not None:
            leaf_path = isolate_index_environment["leaf_dir"] / f"{source_id}.jsonl"
            with leaf_path.open("w", encoding="utf-8") as handle:
                for rec in leaf_records:
                    handle.write(json.dumps(rec) + "\n")

        return row

    return _create


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Creates a mock OpenAI client pre-configured to return dummy vector embeddings."""
    client = MagicMock()

    def _create_embeddings(*args: Any, **kwargs: Any) -> MagicMock:
        inputs = kwargs.get("input", [])
        if not inputs and args:
            inputs = args[0] if len(args) == 1 else args[1]

        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1] * 1536) for _ in inputs]
        return response

    client.embeddings.create.side_effect = _create_embeddings
    return client


@pytest.fixture
def mock_chroma_collection() -> MagicMock:
    """
    Creates a mock Chroma collection supporting get, upsert, update, and delete methods.
    """
    collection = MagicMock()
    collection.get.return_value = {"ids": []}
    return collection
