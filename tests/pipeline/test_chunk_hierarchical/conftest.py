from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pytest

from src.core import registry
from src.pipeline.chunk_hierarchical import WindowConfig, compute_config_hash


@pytest.fixture
def sample_leaf_config() -> WindowConfig:
    return WindowConfig(
        level="leaf",
        min_tokens=10,
        target_tokens=20,
        max_tokens=30,
        overlap_tokens=5,
    )


@pytest.fixture
def sample_parent_config() -> WindowConfig:
    return WindowConfig(
        level="parent",
        min_tokens=30,
        target_tokens=50,
        max_tokens=80,
        overlap_tokens=10,
    )


@pytest.fixture
def default_config_hash(sample_leaf_config: WindowConfig, sample_parent_config: WindowConfig) -> str:
    return compute_config_hash(
        leaf=sample_leaf_config,
        parent=sample_parent_config,
        tokenizer_mode="whitespace",
        tokenizer_encoding="cl100k_base",
        min_leaf_output_tokens=0,
    )


@pytest.fixture
def mock_extracted_payload() -> Dict[str, Any]:
    text = (
        "First sentence in section one. Second sentence giving details. "
        "Third sentence finishing unit one. "
        "Fourth sentence starting section two. Fifth sentence concluding the sample."
    )
    return {
        "text": text,
        "units": [
            {
                "unit_type": "paragraph",
                "unit_index": 1,
                "start_char": 0,
                "end_char": 97,
            },
            {
                "unit_type": "paragraph",
                "unit_index": 2,
                "start_char": 98,
                "end_char": len(text),
            },
        ],
        "meta": {"author": "test_suite", "domain": "unit_test"},
    }


@pytest.fixture
def isolate_chunk_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Redirects registry path, extracted, chunks, leaf, parent, and log paths
    to temporary test directories to keep test execution isolated and clean.
    """
    registry_file = tmp_path / "registry" / "sources.csv"
    extracted_dir = tmp_path / "extracted"
    chunks_root = tmp_path / "chunks"
    leaf_dir = chunks_root / "leaf"
    parent_dir = chunks_root / "parent"
    logs_dir = tmp_path / "logs"

    registry_file.parent.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    leaf_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("src.pipeline.chunk_hierarchical.REGISTRY_PATH", registry_file)
    monkeypatch.setattr("src.pipeline.chunk_hierarchical.EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr("src.pipeline.chunk_hierarchical.CHUNKS_ROOT", chunks_root)
    monkeypatch.setattr("src.pipeline.chunk_hierarchical.LEAF_DIR", leaf_dir)
    monkeypatch.setattr("src.pipeline.chunk_hierarchical.PARENT_DIR", parent_dir)
    monkeypatch.setattr("src.pipeline.chunk_hierarchical.LOG_PATH", logs_dir / "chunk_hierarchical.log")

    return {
        "tmp_path": tmp_path,
        "registry_file": registry_file,
        "extracted_dir": extracted_dir,
        "chunks_root": chunks_root,
        "leaf_dir": leaf_dir,
        "parent_dir": parent_dir,
    }


@pytest.fixture
def make_registry_entry(isolate_chunk_environment: Dict[str, Path]):
    """
    Factory fixture to create sample registry entries along with their extracted JSON files.
    """
    def _create(
        source_id: str,
        extraction_status: str = registry.EXTRACTION_EXTRACTED,
        chunk_status: str = "",
        chunk_config_hash: str = "",
        payload: Dict[str, Any] | None = None,
    ) -> registry.Row:
        row: registry.Row = {
            "source_id": source_id,
            "extraction_status": extraction_status,
            "chunk_status": chunk_status,
            "chunk_config_hash": chunk_config_hash,
            "nb_chunks": "",
            "last_chunk_time": "",
            "chunk_error": "",
            "extracted_path": "",
            "index_status": "",
        }

        if payload is not None:
            extracted_path = isolate_chunk_environment["extracted_dir"] / f"{source_id}.json"
            extracted_path.write_text(json.dumps(payload), encoding="utf-8")
            row["extracted_path"] = str(extracted_path)

        return row

    return _create
