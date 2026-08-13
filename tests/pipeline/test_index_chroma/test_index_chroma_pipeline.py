"""Integration tests for indexing pipeline orchestration, registry management, stale entry purging, and end-to-end execution."""

from unittest.mock import patch
import pytest

from src.core import registry
from src.pipeline import index_chroma as index_mod


# ============================================================================
# 1. Stale Index Purge Tests
# ============================================================================

class TestPurgeStaleIndexEntries:
    """Tests for purging obsolete vector entries from Chroma store."""

    def test_purge_stale_index_entries_deletes_unextracted_or_unchunked_sources(
        self, make_registry_entry, mock_chroma_collection
    ):
        """Test stale sources (not extracted/chunked or missing leaf JSONL) are purged from collection."""
        # Arrange
        active_row = make_registry_entry("active_src", leaf_records=[{"chunk_id": "c1"}])
        unextracted_row = make_registry_entry("unextracted_src", extraction_status="FAILED")
        missing_leaf_row = make_registry_entry("missing_leaf_src")  # No leaf_records created

        rows = [active_row, unextracted_row, missing_leaf_row]

        # Act
        removed_count = index_mod.purge_stale_index_entries(
            rows=rows,
            collection=mock_chroma_collection,
            keep_stale=False,
        )

        # Assert
        assert removed_count == 2
        mock_chroma_collection.delete.assert_any_call(where={"source_id": "unextracted_src"})
        mock_chroma_collection.delete.assert_any_call(where={"source_id": "missing_leaf_src"})

    def test_purge_stale_index_entries_bypassed_when_keep_stale_is_true(
        self, make_registry_entry, mock_chroma_collection
    ):
        """Test purge operation is skipped when keep_stale is True."""
        # Arrange
        unextracted_row = make_registry_entry("unextracted_src", extraction_status="FAILED")

        # Act
        removed_count = index_mod.purge_stale_index_entries(
            rows=[unextracted_row],
            collection=mock_chroma_collection,
            keep_stale=True,
        )

        # Assert
        assert removed_count == 0
        mock_chroma_collection.delete.assert_not_called()


# ============================================================================
# 2. Row Processing Eligibility Tests
# ============================================================================

class TestShouldProcessRow:
    """Tests for row eligibility conditions in should_process_row."""

    @pytest.mark.parametrize(
        "row_kwargs, force, source_filter, expected_result",
        [
            # Case 1: Skips when extraction status is not extracted/chunked
            ({"extraction_status": "PENDING"}, False, set(), False),
            # Case 2: Skips when source_id is not included in source_filter
            ({"leaf_records": [{"chunk_id": "c1"}]}, False, {"other_source"}, False),
            # Case 3: Re-indexes when config hash mismatches current config
            (
                {
                    "index_status": registry.INDEX_INDEXED,
                    "index_config_hash": "old_outdated_hash",
                    "nb_chunks": "1",
                    "indexed_chunks": "1",
                    "leaf_records": [{"chunk_id": "c1"}],
                },
                False,
                set(),
                True,
            ),
        ],
        ids=["skip_unextracted", "skip_filtered_source", "reindex_config_mismatch"],
    )
    def test_should_process_row_eligibility_rules(
        self, make_registry_entry, default_config_hash, row_kwargs, force, source_filter, expected_result
    ):
        """Test various row attributes and filters correctly dictate should_process_row eligibility."""
        # Arrange
        row = make_registry_entry("src_1", **row_kwargs)

        # Act
        result = index_mod.should_process_row(
            row=row,
            config_hash=default_config_hash,
            force=force,
            source_filter=source_filter,
        )

        # Assert
        assert result is expected_result


# ============================================================================
# 3. End-to-End Execution Pipeline Tests
# ============================================================================

class TestRunIndexingPipeline:
    """End-to-end integration tests for run_indexing execution workflow."""

    @pytest.fixture(autouse=True)
    def _setup_pipeline_mocks(self, mock_chroma_collection, mock_openai_client):
        """Automatically patch client builder dependencies for all test cases in this class."""
        with patch("src.pipeline.index_chroma.build_chroma_collection", return_value=mock_chroma_collection), \
             patch("src.pipeline.index_chroma.build_openai_client", return_value=mock_openai_client):
            yield

    def _execute_indexing(self, default_collection_name, default_embedding_model, force=False):
        """Helper method to encapsulate repeated default run_indexing arguments."""
        return index_mod.run_indexing(
            force=force,
            source_ids=None,
            limit=0,
            keep_stale_index=False,
            collection_name=default_collection_name,
            embedding_model=default_embedding_model,
            batch_size=64,
            openai_api_key="test_key",
            request_timeout_sec=30.0,
            max_retries=2,
            retry_base_delay_sec=0.01,
        )

    def test_run_indexing_successful_execution(
        self,
        isolate_index_environment,
        make_registry_entry,
        mock_leaf_record,
        default_collection_name,
        default_embedding_model,
    ):
        """Test run_indexing processes target rows, calls vector store, and updates registry state."""
        # Arrange
        row1 = make_registry_entry("source_1", leaf_records=[mock_leaf_record])
        registry.write_registry_rows(isolate_index_environment["registry_file"], [row1])

        # Act
        result = self._execute_indexing(default_collection_name, default_embedding_model, force=False)

        # Assert
        assert result["processed"] == 1
        assert result["failed"] == 0

        updated_rows = registry.load_registry_rows(isolate_index_environment["registry_file"])
        assert updated_rows[0]["index_status"] == registry.INDEX_INDEXED
        assert updated_rows[0]["indexed_chunks"] == "1"

    def test_run_indexing_handles_row_failure_gracefully(
        self,
        isolate_index_environment,
        make_registry_entry,
        default_collection_name,
        default_embedding_model,
    ):
        """Test run_indexing catches row indexing errors and records INDEX_FAILED in registry."""
        # Arrange
        row1 = make_registry_entry("failed_source")  # Missing leaf JSONL file triggers error
        registry.write_registry_rows(isolate_index_environment["registry_file"], [row1])

        # Act
        result = self._execute_indexing(default_collection_name, default_embedding_model, force=True)

        # Assert
        assert result["processed"] == 0
        assert result["failed"] == 1

        updated_rows = registry.load_registry_rows(isolate_index_environment["registry_file"])
        assert updated_rows[0]["index_status"] == registry.INDEX_FAILED
        assert "Leaf chunk file not found" in updated_rows[0]["index_error"]
