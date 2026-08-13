"""Shared fixtures for scan_and_extract pipeline tests."""

import pytest

from src.pipeline import scan_and_extract as scan_mod


@pytest.fixture
def mock_pipeline_env(tmp_path, monkeypatch):
    """Isolated directory environment monkeypatched into scan_and_extract constants.

    Provides clean, temporary paths for raw, converted, extracted, registry, and log data.
    """
    raw_dir = tmp_path / "raw"
    converted_dir = tmp_path / "converted"
    extracted_dir = tmp_path / "extracted"
    registry_dir = tmp_path / "registry"
    logs_dir = tmp_path / "logs"

    for d in (raw_dir, converted_dir, extracted_dir, registry_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    registry_path = registry_dir / "sources.csv"

    # Patch module constants
    monkeypatch.setattr(scan_mod, "RAW_DIR", raw_dir)
    monkeypatch.setattr(scan_mod, "CONVERTED_DIR", converted_dir)
    monkeypatch.setattr(scan_mod, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(scan_mod, "REGISTRY_PATH", registry_path)

    return {
        "tmp_path": tmp_path,
        "raw_dir": raw_dir,
        "converted_dir": converted_dir,
        "extracted_dir": extracted_dir,
        "registry_dir": registry_dir,
        "registry_path": registry_path,
        "logs_dir": logs_dir,
    }


@pytest.fixture
def sample_registry_row():
    """Factory fixture generating valid registry row dictionaries matching scan_and_extract Row schema."""

    def _create_row(**kwargs):
        default_row = {
            "source_id": "test_source_id_123",
            "origin_path": "/path/to/raw/doc.pdf",
            "canonical_path": "/path/to/raw/doc.pdf",
            "filename": "doc.pdf",
            "nature": "GENERAL",
            "ext": ".pdf",
            "size_bytes": "1024",
            "modified_time": "2026-01-01T12:00:00",
            "content_hash": "a1b2c3d4e5f67890",
            "extraction_status": "NEW",
            "extraction_error": "",
            "extracted_path": "",
            "last_extracted_time": "",
            "last_seen_time": "2026-01-01T12:00:00",
            "chunk_status": "",
            "nb_chunks": "",
            "last_chunk_time": "",
            "chunk_error": "",
            "chunk_config_hash": "",
            "index_status": "",
            "indexed_chunks": "",
            "last_index_time": "",
            "index_error": "",
            "index_config_hash": "",
        }
        default_row.update(kwargs)
        return default_row

    return _create_row
