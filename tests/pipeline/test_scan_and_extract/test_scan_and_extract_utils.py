"""Tests for utility, path mapping, and text normalization functions in src/pipeline/scan_and_extract.py."""

from pathlib import Path
import pytest

from src.pipeline import scan_and_extract as scan_mod


# ============================================================================
# 1. File & Path Utilities Tests
# ============================================================================

class TestFileAndPathUtils:
    """Tests for sha256_file, file_stat_info, infer_nature, and output path generation."""

    def test_sha256_file_calculates_correct_hash(self, tmp_path):
        """Test sha256_file produces valid SHA256 hex digest."""
        # Arrange
        file_path = tmp_path / "sample.txt"
        file_path.write_bytes(b"hello world")

        # Act
        digest = scan_mod.sha256_file(file_path, chunk_size=4)

        # Assert
        assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_file_stat_info_returns_size_and_iso_modified_time(self, tmp_path):
        """Test file_stat_info returns integer byte size and valid ISO timestamp."""
        # Arrange
        file_path = tmp_path / "info.txt"
        file_path.write_text("12345")

        # Act
        size, modified_time = scan_mod.file_stat_info(file_path)

        # Assert
        assert size == 5
        assert isinstance(modified_time, str)
        assert "T" in modified_time

    @pytest.mark.parametrize(
        ("path_builder", "expected_nature"),
        [
            (lambda env: env["raw_dir"] / "V1_RAG" / "FINANCE" / "doc.pdf", "FINANCE"),
            (lambda env: env["raw_dir"] / "MARKETING" / "campaign.pdf", "MARKETING"),
            (lambda env: Path("/outside/root/file.pdf"), ""),
        ],
        ids=["v1_rag_folder", "relative_to_raw_dir", "outside_raw_dir"],
    )
    def test_infer_nature(self, mock_pipeline_env, path_builder, expected_nature):
        """Test nature inference based on V1_RAG segment, raw relative directory, or outside paths."""
        # Arrange
        target_path = path_builder(mock_pipeline_env)

        # Act
        nature = scan_mod.infer_nature(target_path)

        # Assert
        assert nature == expected_nature

    def test_extracted_output_path_combines_with_extracted_dir(self, mock_pipeline_env):
        """Test extracted_output_path constructs target JSON path using source_id."""
        # Arrange
        source_id = "hash123"

        # Act
        output_path = scan_mod.extracted_output_path(source_id)

        # Assert
        assert output_path == mock_pipeline_env["extracted_dir"] / "hash123.json"


# ============================================================================
# 2. Converted to Origin Mapping Tests
# ============================================================================

class TestConvertedOriginMapping:
    """Tests for is_in_converted and map_converted_to_origin."""

    def test_is_in_converted_returns_true_only_when_under_converted_dir(self, mock_pipeline_env):
        """Test is_in_converted correctly identifies paths within CONVERTED_DIR."""
        # Arrange
        converted_dir = mock_pipeline_env["converted_dir"]
        inside_path = converted_dir / "sub" / "doc.docx"
        outside_path = mock_pipeline_env["tmp_path"] / "doc.docx"

        # Act
        is_inside = scan_mod.is_in_converted(inside_path)
        is_outside = scan_mod.is_in_converted(outside_path)

        # Assert
        assert is_inside is True
        assert is_outside is False

    @pytest.mark.parametrize(
        ("converted_rel", "legacy_suffix", "created_target"),
        [
            ("doc.docx", ".doc", "legacy"),
            ("doc.docx", ".docx", "modern"),
            ("pres.pptx", ".ppt", "legacy"),
            ("pres.pptx", ".pptx", "modern"),
        ],
        ids=["docx_to_doc", "docx_fallback_docx", "pptx_to_ppt", "pptx_fallback_pptx"],
    )
    def test_map_converted_to_origin_successful_resolutions(
        self, mock_pipeline_env, converted_rel, legacy_suffix, created_target
    ):
        """Test converted files correctly map back to legacy or modern origin files in RAW_DIR."""
        # Arrange
        raw_dir = mock_pipeline_env["raw_dir"]
        converted_dir = mock_pipeline_env["converted_dir"]

        converted_file = converted_dir / converted_rel
        converted_file.touch()

        if created_target == "legacy":
            origin_file = raw_dir / Path(converted_rel).with_suffix(legacy_suffix)
        else:
            origin_file = raw_dir / converted_rel

        origin_file.touch()

        # Act
        mapped = scan_mod.map_converted_to_origin(converted_file)

        # Assert
        assert mapped == origin_file.resolve()

    def test_map_converted_to_origin_returns_none_when_outside_converted_dir(self, mock_pipeline_env):
        """Test map_converted_to_origin returns None for paths outside CONVERTED_DIR."""
        # Arrange
        outside_file = mock_pipeline_env["tmp_path"] / "outside.docx"

        # Act
        mapped = scan_mod.map_converted_to_origin(outside_file)

        # Assert
        assert mapped is None

    def test_map_converted_to_origin_returns_none_when_origin_file_missing(self, mock_pipeline_env):
        """Test map_converted_to_origin returns None when in CONVERTED_DIR but origin file is absent."""
        # Arrange
        converted_file = mock_pipeline_env["converted_dir"] / "orphan.docx"
        converted_file.touch()

        # Act
        mapped = scan_mod.map_converted_to_origin(converted_file)

        # Assert
        assert mapped is None


# ============================================================================
# 3. Text Normalization & Materialization Tests
# ============================================================================

class TestTextNormalizationAndMaterialization:
    """Tests for normalize_text_block and materialize_units."""

    def test_normalize_text_block_cleans_whitespace_and_control_chars(self):
        """Test normalize_text_block strips control characters, normalizes newlines, and collapses blank lines."""
        # Arrange
        raw_text = "Line 1\r\n\r\nLine 2\x0bText\u00A0with\xA0nbsps\x07.\n\n\nLine 3"

        # Act
        cleaned = scan_mod.normalize_text_block(raw_text)

        # Assert
        assert cleaned == "Line 1\n\nLine 2\nText with nbsps .\n\nLine 3"

    def test_materialize_units_calculates_char_cursors_and_combines_text(self):
        """Test materialize_units creates aggregated text and correct character offset spans for units."""
        # Arrange
        raw_units = [
            {"unit_type": "page", "unit_index": 1, "text": "First page text"},
            {"unit_type": "page", "unit_index": 2, "text": "   "},
            {"unit_type": "page", "unit_index": 3, "text": "Third page text"},
        ]

        # Act
        full_text, units = scan_mod.materialize_units(raw_units)

        # Assert
        expected_text = "First page text\n\nThird page text"
        assert full_text == expected_text
        assert len(units) == 2

        # Unit 1
        assert units[0]["start_char"] == 0
        assert units[0]["end_char"] == 15
        assert units[0]["text"] == "First page text"

        # Unit 2
        assert units[1]["start_char"] == 17
        assert units[1]["end_char"] == 32
        assert units[1]["text"] == "Third page text"


# ============================================================================
# 4. Artifact Purging Tests
# ============================================================================

class TestPurgeMissingArtifacts:
    """Tests for purge_missing_artifacts."""

    def test_purge_missing_artifacts_removes_json_files_for_missing_rows(
        self, mock_pipeline_env, sample_registry_row
    ):
        """Test purge_missing_artifacts unlinks extracted JSON files of MISSING rows within EXTRACTED_DIR."""
        # Arrange
        extracted_dir = mock_pipeline_env["extracted_dir"]

        missing_json = extracted_dir / "hash_missing.json"
        missing_json.write_text("{}")

        active_json = extracted_dir / "hash_active.json"
        active_json.write_text("{}")

        rows = [
            sample_registry_row(
                extraction_status=scan_mod.EXTRACTION_MISSING,
                source_id="hash_missing",
                extracted_path=str(missing_json),
            ),
            sample_registry_row(
                extraction_status=scan_mod.EXTRACTION_EXTRACTED,
                source_id="hash_active",
                extracted_path=str(active_json),
            ),
        ]

        # Act
        removed_count = scan_mod.purge_missing_artifacts(rows)

        # Assert
        assert removed_count == 1
        assert not missing_json.exists()
        assert active_json.exists()
        assert rows[0]["extracted_path"] == ""

    def test_purge_missing_artifacts_skips_files_outside_extracted_dir(
        self, mock_pipeline_env, sample_registry_row, caplog
    ):
        """Test purge_missing_artifacts refuses to delete files outside EXTRACTED_DIR for safety."""
        # Arrange
        outside_json = mock_pipeline_env["tmp_path"] / "outside.json"
        outside_json.write_text("{}")

        rows = [
            sample_registry_row(
                extraction_status=scan_mod.EXTRACTION_MISSING,
                source_id="hash_outside",
                extracted_path=str(outside_json),
            )
        ]

        # Act
        removed_count = scan_mod.purge_missing_artifacts(rows)

        # Assert
        assert removed_count == 0
        assert outside_json.exists()
        assert "Skipped purge outside extracted dir" in caplog.text
