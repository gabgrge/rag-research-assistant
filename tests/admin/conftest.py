"""Shared fixtures for admin module tests."""

import pytest

from src.admin import backup_index, reset_local_state


@pytest.fixture
def mock_project(monkeypatch, tmp_path):
    """Sets up an isolated, unified mock environment for all admin test suites."""
    project_dir = tmp_path / "project"

    # Define directory tree
    data_dir = project_dir / "data"
    chunks_dir = data_dir / "chunks"
    leaf_chunks_dir = chunks_dir / "leaf"
    parent_chunks_dir = chunks_dir / "parent"
    converted_dir = data_dir / "converted"
    extracted_dir = data_dir / "extracted"
    index_dir = data_dir / "index" / "chroma"

    registry_dir = project_dir / "registry"
    registry_path = registry_dir / "sources.csv"
    logs_dir = project_dir / "logs"
    backups_dir = project_dir / "backups"

    # Create all directories
    all_dirs = [
        leaf_chunks_dir,
        parent_chunks_dir,
        converted_dir,
        extracted_dir,
        index_dir,
        registry_dir,
        logs_dir,
        backups_dir,
    ]
    for directory in all_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # Seed mock files
    (index_dir / "chroma.sqlite3").write_text("sqlite mock data", encoding="utf-8")
    (leaf_chunks_dir / "chunk1.txt").write_text("chunk leaf", encoding="utf-8")
    (parent_chunks_dir / "chunk_p1.txt").write_text("chunk parent", encoding="utf-8")
    (converted_dir / "file.txt").write_text("converted", encoding="utf-8")
    (extracted_dir / "file.txt").write_text("extracted", encoding="utf-8")
    registry_path.write_text("id,source\n1,doc1.pdf", encoding="utf-8")
    (logs_dir / "app.log").write_text("log line", encoding="utf-8")

    # --- Patch backup_index constants ---
    monkeypatch.setattr(backup_index, "PROJECT_ROOT", project_dir)
    monkeypatch.setattr(backup_index, "INDEX_DIR", index_dir)
    monkeypatch.setattr(backup_index, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(backup_index, "BACKUP_ROOT", backups_dir)

    # --- Patch reset_local_state constants ---
    monkeypatch.setattr(reset_local_state, "PROJECT_ROOT", project_dir)
    monkeypatch.setattr(reset_local_state, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(reset_local_state, "LEAF_CHUNKS_DIR", leaf_chunks_dir)
    monkeypatch.setattr(reset_local_state, "PARENT_CHUNKS_DIR", parent_chunks_dir)
    monkeypatch.setattr(reset_local_state, "CONVERTED_DIR", converted_dir)
    monkeypatch.setattr(reset_local_state, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(reset_local_state, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(reset_local_state, "LOGS_DIR", logs_dir)

    # Re-bind module lists for reset_local_state
    monkeypatch.setattr(
        reset_local_state,
        "RESET_DIRS",
        [chunks_dir, converted_dir, extracted_dir, registry_dir, logs_dir],
    )
    monkeypatch.setattr(
        reset_local_state,
        "RECREATE_DIRS",
        [leaf_chunks_dir, parent_chunks_dir, converted_dir, extracted_dir, registry_dir, logs_dir],
    )

    return {
        "root": project_dir,
        "index_dir": index_dir,
        "registry_path": registry_path,
        "registry_dir": registry_dir,
        "backups_dir": backups_dir,
        "chunks_dir": chunks_dir,
        "leaf_chunks_dir": leaf_chunks_dir,
        "parent_chunks_dir": parent_chunks_dir,
        "converted_dir": converted_dir,
        "extracted_dir": extracted_dir,
        "logs_dir": logs_dir,
    }
