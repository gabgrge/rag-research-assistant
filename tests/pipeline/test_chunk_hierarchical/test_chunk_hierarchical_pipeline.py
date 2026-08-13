"""Tests for pipeline orchestration, registry management, purging stale files, and end-to-end chunking runs."""

from src.core import registry
from src.pipeline import chunk_hierarchical as chunk_mod


# ============================================================================
# 1. Stale File Purge & Source Removal Tests
# ============================================================================

class TestPurgeStaleChunks:
    """Tests for purging obsolete leaf and parent JSONL files from disk."""

    def test_remove_chunk_files_for_source_unlinks_existing_files(self, isolate_chunk_environment):
        """Test remove_chunk_files_for_source deletes both leaf and parent files when they exist."""
        # Arrange
        source_id = "src_to_delete"
        leaf_file = isolate_chunk_environment["leaf_dir"] / f"{source_id}.jsonl"
        parent_file = isolate_chunk_environment["parent_dir"] / f"{source_id}.jsonl"
        leaf_file.write_text("{}", encoding="utf-8")
        parent_file.write_text("{}", encoding="utf-8")

        # Act
        removed_count = chunk_mod.remove_chunk_files_for_source(source_id)

        # Assert
        assert removed_count == 2
        assert not leaf_file.exists()
        assert not parent_file.exists()

    def test_purge_stale_chunks_cleans_non_extracted_and_orphan_files(
        self, isolate_chunk_environment, make_registry_entry
    ):
        """Test purge_stale_chunks removes files for sources not marked EXTRACTION_EXTRACTED or missing from registry."""
        # Arrange
        leaf_dir = isolate_chunk_environment["leaf_dir"]
        parent_dir = isolate_chunk_environment["parent_dir"]

        # Valid extracted source
        valid_row = make_registry_entry("valid_src", extraction_status=registry.EXTRACTION_EXTRACTED)
        (leaf_dir / "valid_src.jsonl").write_text("{}", encoding="utf-8")

        # Failed extraction source (stale)
        stale_row = make_registry_entry("stale_src", extraction_status="EXTRACTION_FAILED")
        (leaf_dir / "stale_src.jsonl").write_text("{}", encoding="utf-8")

        # Orphan file (source not in registry)
        (parent_dir / "orphan_src.jsonl").write_text("{}", encoding="utf-8")

        rows = [valid_row, stale_row]

        # Act
        removed_count = chunk_mod.purge_stale_chunks(rows, keep_stale=False)

        # Assert
        assert removed_count == 2
        assert (leaf_dir / "valid_src.jsonl").exists()
        assert not (leaf_dir / "stale_src.jsonl").exists()
        assert not (parent_dir / "orphan_src.jsonl").exists()

    def test_purge_stale_chunks_bypassed_when_keep_stale_is_true(
        self, isolate_chunk_environment, make_registry_entry
    ):
        """Test purge_stale_chunks takes no action when keep_stale is True."""
        # Arrange
        stale_row = make_registry_entry("stale_src", extraction_status="EXTRACTION_FAILED")
        stale_file = isolate_chunk_environment["leaf_dir"] / "stale_src.jsonl"
        stale_file.write_text("{}", encoding="utf-8")

        # Act
        removed_count = chunk_mod.purge_stale_chunks([stale_row], keep_stale=True)

        # Assert
        assert removed_count == 0
        assert stale_file.exists()


# ============================================================================
# 2. Row Processing Eligibility Tests
# ============================================================================

class TestShouldProcessRow:
    """Tests for should_process_row eligibility decision logic."""

    def test_should_process_row_skips_non_extracted_sources(
        self, make_registry_entry, default_config_hash
    ):
        """Test sources with extraction_status != EXTRACTION_EXTRACTED are skipped."""
        # Arrange
        row = make_registry_entry("src_1", extraction_status="PENDING")

        # Act
        result = chunk_mod.should_process_row(
            row, config_hash=default_config_hash, force=False, source_filter=set()
        )

        # Assert
        assert result is False

    def test_should_process_row_skips_when_filtered_out(
        self, make_registry_entry, default_config_hash
    ):
        """Test sources not in source_filter set are skipped."""
        # Arrange
        row = make_registry_entry("src_1", extraction_status=registry.EXTRACTION_EXTRACTED)

        # Act
        result = chunk_mod.should_process_row(
            row, config_hash=default_config_hash, force=False, source_filter={"other_src"}
        )

        # Assert
        assert result is False

    def test_should_process_row_reprocesses_when_config_hash_mismatches(
        self, isolate_chunk_environment, make_registry_entry, default_config_hash
    ):
        """Test re-chunking is triggered when the existing config hash differs from the target hash."""
        # Arrange
        source_id = "src_1"
        row = make_registry_entry(
            source_id,
            extraction_status=registry.EXTRACTION_EXTRACTED,
            chunk_status=registry.CHUNK_CHUNKED,
            chunk_config_hash="old_outdated_hash",
        )
        (isolate_chunk_environment["leaf_dir"] / f"{source_id}.jsonl").write_text("{}", encoding="utf-8")
        (isolate_chunk_environment["parent_dir"] / f"{source_id}.jsonl").write_text("{}", encoding="utf-8")

        # Act
        result = chunk_mod.should_process_row(
            row, config_hash=default_config_hash, force=False, source_filter=set()
        )

        # Assert
        assert result is True


# ============================================================================
# 3. End-to-End Execution Pipeline Tests
# ============================================================================

class TestRunChunkingPipeline:
    """End-to-end integration tests for the run_chunking orchestration loop."""

    def test_run_chunking_successful_end_to_end(
        self, isolate_chunk_environment, make_registry_entry, mock_extracted_payload
    ):
        """Test run_chunking processes eligible rows, generates JSONL files, and updates registry state."""
        # Arrange
        row1 = make_registry_entry("source_1", payload=mock_extracted_payload)
        row2 = make_registry_entry("source_2", payload=mock_extracted_payload)
        registry.write_registry_rows(isolate_chunk_environment["registry_file"], [row1, row2])

        # Act
        result = chunk_mod.run_chunking(
            force=False,
            source_ids=None,
            limit=0,
            keep_stale_chunks=False,
            leaf_min=10,
            leaf_target=20,
            leaf_max=30,
            leaf_overlap=5,
            parent_min=30,
            parent_target=50,
            parent_max=80,
            parent_overlap=10,
            min_leaf_output_tokens=0,
            tokenizer="whitespace",
            tiktoken_encoding="cl100k_base",
        )

        # Assert
        assert result["processed"] == 2
        assert result["failed"] == 0

        leaf_file = isolate_chunk_environment["leaf_dir"] / "source_1.jsonl"
        parent_file = isolate_chunk_environment["parent_dir"] / "source_1.jsonl"
        assert leaf_file.exists()
        assert parent_file.exists()

        # Check updated registry state
        updated_rows = registry.load_registry_rows(isolate_chunk_environment["registry_file"])
        assert updated_rows[0]["chunk_status"] == registry.CHUNK_CHUNKED
        assert int(updated_rows[0]["nb_chunks"]) > 0

    def test_run_chunking_handles_source_processing_failure(
        self, isolate_chunk_environment, make_registry_entry
    ):
        """Test run_chunking marks row as CHUNK_FAILED when input payload file is missing or corrupted."""
        # Arrange
        row = make_registry_entry("corrupted_source")
        # Point to a missing file path to force failure
        row["extracted_path"] = str(isolate_chunk_environment["extracted_dir"] / "non_existent.json")
        registry.write_registry_rows(isolate_chunk_environment["registry_file"], [row])

        # Act
        result = chunk_mod.run_chunking(
            force=True,
            source_ids=None,
            limit=0,
            keep_stale_chunks=False,
            leaf_min=10,
            leaf_target=20,
            leaf_max=30,
            leaf_overlap=5,
            parent_min=30,
            parent_target=50,
            parent_max=80,
            parent_overlap=10,
            min_leaf_output_tokens=0,
            tokenizer="whitespace",
            tiktoken_encoding="cl100k_base",
        )

        # Assert
        assert result["processed"] == 0
        assert result["failed"] == 1

        updated_rows = registry.load_registry_rows(isolate_chunk_environment["registry_file"])
        assert updated_rows[0]["chunk_status"] == registry.CHUNK_FAILED
        assert "Extracted JSON not found" in updated_rows[0]["chunk_error"]
