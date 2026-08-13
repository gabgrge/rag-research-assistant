"""Tests for CSV registry synchronization, corpus scanning, and entry updates in src/pipeline/scan_and_extract.py."""

from src.pipeline import scan_and_extract as scan_mod


# ============================================================================
# 1. Registry I/O Tests
# ============================================================================

class TestRegistryIO:
    """Tests for load_registry and write_registry wrapper functions."""

    def test_load_registry_returns_empty_when_file_missing(self, mock_pipeline_env):
        """Test load_registry returns empty list if CSV registry does not exist yet."""
        # Act
        rows = scan_mod.load_registry()

        # Assert
        assert rows == []

    def test_load_and_write_registry_roundtrip(self, mock_pipeline_env, sample_registry_row):
        """Test write_registry writes rows and load_registry reads them back cleanly."""
        # Arrange
        path_1 = str(mock_pipeline_env["raw_dir"] / "file1.pdf")
        path_2 = str(mock_pipeline_env["raw_dir"] / "file2.pdf")

        initial_rows = [
            sample_registry_row(source_id="hash_1", canonical_path=path_1, filename="file1.pdf"),
            sample_registry_row(source_id="hash_2", canonical_path=path_2, filename="file2.pdf"),
        ]

        # Act
        scan_mod.write_registry(initial_rows)
        loaded_rows = scan_mod.load_registry()

        # Assert
        assert len(loaded_rows) == 2
        assert loaded_rows[0]["source_id"] == "hash_1"
        assert loaded_rows[0]["canonical_path"] == path_1
        assert loaded_rows[1]["canonical_path"] == path_2


# ============================================================================
# 2. Corpus Scanning Tests
# ============================================================================

class TestScanCorpus:
    """Tests for scan_corpus scanning behavior across CONVERTED_DIR and RAW_DIR."""

    def test_scan_corpus_discovers_supported_extensions(self, mock_pipeline_env):
        """Test scan_corpus finds .pdf, .docx, and .pptx files in RAW_DIR and CONVERTED_DIR while filtering unsupported files."""
        # Arrange
        raw_dir = mock_pipeline_env["raw_dir"]
        converted_dir = mock_pipeline_env["converted_dir"]

        valid_pdf = raw_dir / "report.pdf"
        valid_docx = converted_dir / "memo.docx"
        valid_pptx = raw_dir / "slides.pptx"
        unsupported = raw_dir / "notes.txt"

        for f in (valid_pdf, valid_docx, valid_pptx, unsupported):
            f.touch()

        # Act
        discovered = scan_mod.scan_corpus()

        # Assert
        discovered_set = set(discovered)
        assert len(discovered) == 3
        assert valid_pdf.resolve() in discovered_set
        assert valid_docx.resolve() in discovered_set
        assert valid_pptx.resolve() in discovered_set

    def test_scan_corpus_deduplicates_identical_paths(self, mock_pipeline_env):
        """Test scan_corpus deduplicates resolved file paths."""
        # Arrange
        raw_file = mock_pipeline_env["raw_dir"] / "document.pdf"
        raw_file.touch()

        # Act
        discovered = scan_mod.scan_corpus()

        # Assert
        assert len(discovered) == 1
        assert discovered[0] == raw_file.resolve()


# ============================================================================
# 3. Entry Updates & Synchronization Tests
# ============================================================================

class TestEnsureRegistryEntries:
    """Tests for ensure_registry_entries registry reconciliation logic."""

    def test_ensure_registry_entries_adds_new_discovered_file(self, mock_pipeline_env):
        """Test newly discovered corpus path creates a new registry row with EXTRACTION_NEW status."""
        # Arrange
        rows = []
        doc_file = mock_pipeline_env["raw_dir"] / "RH" / "policy.pdf"
        doc_file.parent.mkdir(parents=True, exist_ok=True)
        doc_file.write_text("Policy Content")

        # Act
        updated_rows = scan_mod.ensure_registry_entries(rows, [doc_file])

        # Assert
        assert len(updated_rows) == 1
        assert updated_rows[0]["filename"] == "policy.pdf"
        assert updated_rows[0]["nature"] == "RH"
        assert updated_rows[0]["extraction_status"] == scan_mod.EXTRACTION_NEW
        assert updated_rows[0]["content_hash"] == scan_mod.sha256_file(doc_file)

    def test_ensure_registry_entries_marks_unseen_rows_as_missing(self, mock_pipeline_env, sample_registry_row):
        """Test rows not present in current corpus scan are marked as EXTRACTION_MISSING."""
        # Arrange
        missing_canonical = str((mock_pipeline_env["raw_dir"] / "ghost.pdf").resolve())
        existing_row = sample_registry_row(
            source_id="ghost_hash",
            canonical_path=missing_canonical,
            extraction_status=scan_mod.EXTRACTION_EXTRACTED,
        )
        rows = [existing_row]
        corpus_paths = []

        # Act
        updated_rows = scan_mod.ensure_registry_entries(rows, corpus_paths)

        # Assert
        assert len(updated_rows) == 1
        assert updated_rows[0]["extraction_status"] == scan_mod.EXTRACTION_MISSING

    def test_ensure_registry_entries_detects_hash_change_on_modified_file(self, mock_pipeline_env, sample_registry_row):
        """Test modified file with altered SHA256 content hash gets marked EXTRACTION_NEW."""
        # Arrange
        doc_file = mock_pipeline_env["raw_dir"] / "doc.pdf"
        doc_file.write_text("Initial Content")

        doc_resolved = str(doc_file.resolve())
        initial_hash = scan_mod.sha256_file(doc_file)
        size_bytes, modified_time = scan_mod.file_stat_info(doc_file)

        existing_row = sample_registry_row(
            source_id=initial_hash,
            content_hash=initial_hash,
            canonical_path=doc_resolved,
            size_bytes="0",  # Different size to trigger metadata_changed
            modified_time=modified_time,
            extraction_status=scan_mod.EXTRACTION_EXTRACTED,
        )

        # Modify file content on disk
        doc_file.write_text("Modified New Content")

        # Act
        updated_rows = scan_mod.ensure_registry_entries([existing_row], [doc_file])

        # Assert
        assert len(updated_rows) == 1
        assert updated_rows[0]["extraction_status"] == scan_mod.EXTRACTION_NEW
        assert updated_rows[0]["content_hash"] != initial_hash

    def test_ensure_registry_entries_matches_moved_file_by_hash(self, mock_pipeline_env, sample_registry_row):
        """Test file moved to a new path with identical hash updates canonical_path without duplicating entry."""
        # Arrange
        new_file = mock_pipeline_env["raw_dir"] / "NEW_LOC" / "moved.pdf"
        new_file.parent.mkdir(parents=True, exist_ok=True)
        new_file.write_bytes(b"Identical Binary Content")
        file_hash = scan_mod.sha256_file(new_file)

        old_canonical = str((mock_pipeline_env["raw_dir"] / "OLD_LOC" / "moved.pdf").resolve())

        existing_row = sample_registry_row(
            source_id=file_hash,
            content_hash=file_hash,
            canonical_path=old_canonical,
            extraction_status=scan_mod.EXTRACTION_EXTRACTED,
        )

        # Act
        updated_rows = scan_mod.ensure_registry_entries([existing_row], [new_file])

        # Assert
        assert len(updated_rows) == 1
        assert updated_rows[0]["canonical_path"] == str(new_file.resolve())
        assert updated_rows[0]["content_hash"] == file_hash
