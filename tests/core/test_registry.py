"""Tests for document registry state tracking and CSV persistence (src/core/registry.py)."""

import logging
from unittest.mock import patch

from src.core import registry


class TestRegistryRowNormalization:
    """Tests for registry row schema normalization and default values."""

    def test_normalize_registry_row_fills_missing_fields_with_empty_strings(self):
        """Test that partial dictionary row fills missing REGISTRY_FIELDS with ""."""
        # Arrange
        partial_row = {"source_id": "src_123", "filename": "doc.pdf"}

        # Act
        normalized = registry.normalize_registry_row(partial_row)

        # Assert
        assert len(normalized) == len(registry.REGISTRY_FIELDS)
        assert normalized["source_id"] == "src_123"
        assert normalized["filename"] == "doc.pdf"
        assert normalized["extraction_status"] == ""
        assert normalized["chunk_status"] == ""
        assert normalized["index_status"] == ""

    def test_normalize_registry_row_ignores_extra_fields(self):
        """Test that extra key-value pairs not in REGISTRY_FIELDS are omitted."""
        # Arrange
        extra_row = {"source_id": "src_123", "unknown_field": "foo"}

        # Act
        normalized = registry.normalize_registry_row(extra_row)

        # Assert
        assert "unknown_field" not in normalized
        assert normalized["source_id"] == "src_123"


class TestRegistryStateTransitions:
    """Tests for state mutator functions (clearing and resetting status tracking)."""

    def test_set_chunk_pending_resets_chunk_and_index_fields(self):
        """Test that setting chunk status to PENDING clears chunk error/hash and index fields."""
        # Arrange
        row = registry.normalize_registry_row({
            "source_id": "s1",
            "chunk_status": registry.CHUNK_FAILED,
            "nb_chunks": "10",
            "chunk_error": "Timeout",
            "index_status": registry.INDEX_INDEXED,
            "indexed_chunks": "10",
        })

        # Act
        registry.set_chunk_pending(row)

        # Assert
        assert row["chunk_status"] == registry.CHUNK_PENDING
        assert row["nb_chunks"] == ""
        assert row["chunk_error"] == ""
        assert row["index_status"] == ""
        assert row["indexed_chunks"] == ""

    def test_clear_chunk_tracking_clears_chunk_and_index_fields(self):
        """Test clearing chunk tracking wipes chunk state and cascades clear to index tracking."""
        # Arrange
        row = registry.normalize_registry_row({
            "chunk_status": registry.CHUNK_CHUNKED,
            "nb_chunks": "5",
            "index_status": registry.INDEX_INDEXED,
        })

        # Act
        registry.clear_chunk_tracking(row)

        # Assert
        assert row["chunk_status"] == ""
        assert row["nb_chunks"] == ""
        assert row["index_status"] == ""

    def test_set_index_pending_resets_index_fields(self):
        """Test setting index status to PENDING resets indexed_chunks, timestamps, errors, and hashes."""
        # Arrange
        row = registry.normalize_registry_row({
            "index_status": registry.INDEX_FAILED,
            "indexed_chunks": "3",
            "index_error": "Connection error",
        })

        # Act
        registry.set_index_pending(row)

        # Assert
        assert row["index_status"] == registry.INDEX_PENDING
        assert row["indexed_chunks"] == ""
        assert row["index_error"] == ""
        assert row["last_index_time"] == ""


class TestRegistryPersistence:
    """Tests for reading and writing registry CSV files."""

    def test_load_registry_rows_with_non_existent_file_returns_empty_list(self, tmp_path):
        """Test loading from a non-existent file path returns empty list."""
        # Arrange
        missing_file = tmp_path / "missing_registry.csv"

        # Act
        rows = registry.load_registry_rows(missing_file)

        # Assert
        assert rows == []

    def test_write_and_load_registry_rows_roundtrip_succeeds(self, tmp_path):
        """Test writing rows and loading them back maintains structure and normalization."""
        # Arrange
        registry_path = tmp_path / "registry.csv"
        test_rows = [
            registry.normalize_registry_row({
                "source_id": "s1",
                "filename": "file1.txt",
                "extraction_status": registry.EXTRACTION_EXTRACTED,
            }),
            registry.normalize_registry_row({
                "source_id": "s2",
                "filename": "file2.pdf",
                "extraction_status": registry.EXTRACTION_NEW,
            }),
        ]

        # Act
        registry.write_registry_rows(registry_path, test_rows)
        loaded_rows = registry.load_registry_rows(registry_path)

        # Assert
        assert len(loaded_rows) == 2
        assert loaded_rows[0]["source_id"] == "s1"
        assert loaded_rows[0]["filename"] == "file1.txt"
        assert loaded_rows[1]["source_id"] == "s2"
        assert loaded_rows[1]["extraction_status"] == registry.EXTRACTION_NEW

    @patch("src.core.registry.atomic_replace_with_retry")
    def test_write_registry_rows_uses_atomic_replace(
        self, mock_atomic_replace, tmp_path
    ):
        """Test that write_registry_rows creates temp file and invokes atomic_replace_with_retry."""
        # Arrange
        registry_path = tmp_path / "target_registry.csv"
        temp_path = registry_path.with_suffix(".tmp")
        dummy_logger = logging.getLogger("test_logger")
        test_rows = [registry.normalize_registry_row({"source_id": "s1"})]

        # Act
        registry.write_registry_rows(registry_path, test_rows, logger=dummy_logger)

        # Assert
        mock_atomic_replace.assert_called_once_with(
            temp_path, registry_path, logger=dummy_logger
        )
