"""Tests for execution engine, stale purger, and orchestration in src/pipeline/convert_legacy_office_to_modern.py."""

import subprocess
from pathlib import Path
from unittest.mock import patch
import pytest

from src.pipeline import convert_legacy_office_to_modern as convert_mod


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_conversion_plan(tmp_path):
    """Provides a valid ConversionPlan with source and destination files."""
    raw_dir = tmp_path / "raw"
    converted_dir = tmp_path / "converted"
    raw_dir.mkdir()
    converted_dir.mkdir()

    source = raw_dir / "test_doc.doc"
    destination = converted_dir / "test_doc.docx"
    source.write_text("legacy data")

    return convert_mod.ConversionPlan(
        source=source,
        destination=destination,
        output_format="docx",
    )


# ============================================================================
# 1. Single File Conversion Engine Tests
# ============================================================================

class TestConvertOne:
    """Tests for convert_one execution unit."""

    def test_convert_one_skip_up_to_date(self, mock_conversion_plan):
        """Test skipping conversion when destination is already up to date."""
        # Arrange
        mock_conversion_plan.destination.write_text("already converted")

        # Act
        result = convert_mod.convert_one(
            plan=mock_conversion_plan,
            soffice=Path("/bin/soffice"),
            timeout_sec=10,
            force=False,
            dry_run=False,
        )

        # Assert
        assert result.status == "SKIP"
        assert result.detail == "up-to-date"

    def test_convert_one_dry_run(self, mock_conversion_plan):
        """Test returning DRYRUN status without invoking subprocess."""
        # Act
        result = convert_mod.convert_one(
            plan=mock_conversion_plan,
            soffice=Path("/bin/soffice"),
            timeout_sec=10,
            force=False,
            dry_run=True,
        )

        # Assert
        assert result.status == "DRYRUN"
        assert result.detail == "conversion planned"

    @patch("subprocess.run")
    def test_convert_one_success(self, mock_subprocess_run, mock_conversion_plan):
        """Test successful execution creating valid output file."""
        # Arrange
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK", stderr=""
        )

        def create_file(*args, **kwargs):
            mock_conversion_plan.destination.write_text("converted content")
            return mock_subprocess_run.return_value

        mock_subprocess_run.side_effect = create_file

        # Act
        result = convert_mod.convert_one(
            plan=mock_conversion_plan,
            soffice=Path("/bin/soffice"),
            timeout_sec=10,
            force=True,
            dry_run=False,
        )

        # Assert
        assert result.status == "OK"
        assert result.detail == "converted"

    @pytest.mark.parametrize(
        ("side_effect", "completed_process", "expected_detail"),
        [
            (subprocess.TimeoutExpired(cmd="soffice", timeout=10), None, "timeout after 10s"),
            (RuntimeError("Unexpected error"), None, "execution error: Unexpected error"),
            (
                None,
                subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Error message"),
                "Error message",
            ),
        ],
        ids=["timeout_error", "execution_exception", "nonzero_returncode"],
    )
    def test_convert_one_failure_modes(
        self, mock_conversion_plan, side_effect, completed_process, expected_detail
    ):
        """Test failure handling across timeouts, exceptions, and non-zero exit codes."""
        # Arrange
        with patch("subprocess.run") as mock_run:
            if side_effect:
                mock_run.side_effect = side_effect
            else:
                mock_run.return_value = completed_process

            # Act
            result = convert_mod.convert_one(
                plan=mock_conversion_plan,
                soffice=Path("/bin/soffice"),
                timeout_sec=10,
                force=True,
                dry_run=False,
            )

        # Assert
        assert result.status == "FAIL"
        assert expected_detail in result.detail


# ============================================================================
# 2. Stale Purger Tests
# ============================================================================

class TestPurgeStaleConvertedFiles:
    """Tests for purge_stale_converted_files utility."""

    def test_purge_stale_converted_files_removes_orphans(self, tmp_path):
        """Test removing orphan .docx/.pptx files that are not in expected_paths."""
        # Arrange
        converted_dir = tmp_path / "converted"
        converted_dir.mkdir()

        stale_file = converted_dir / "orphan.docx"
        stale_file.touch()

        valid_file = converted_dir / "valid.docx"
        valid_file.touch()

        expected_paths = {str(valid_file.resolve())}

        # Act
        removed_count = convert_mod.purge_stale_converted_files(
            converted_dir=converted_dir,
            expected_paths=expected_paths,
            dry_run=False,
        )

        # Assert
        assert removed_count == 1
        assert not stale_file.exists()
        assert valid_file.exists()

    def test_purge_stale_converted_files_dry_run_leaves_files_intact(self, tmp_path):
        """Test dry-run counts stale files without unlinking them."""
        # Arrange
        converted_dir = tmp_path / "converted"
        converted_dir.mkdir()

        stale_file = converted_dir / "orphan.docx"
        stale_file.touch()

        # Act
        removed_count = convert_mod.purge_stale_converted_files(
            converted_dir=converted_dir,
            expected_paths=set(),
            dry_run=True,
        )

        # Assert
        assert removed_count == 1
        assert stale_file.exists()


# ============================================================================
# 3. Full Pipeline Orchestration Tests
# ============================================================================

class TestRunConversion:
    """Tests for run_conversion main pipeline function."""

    def test_run_conversion_end_to_end_summary(self, tmp_path):
        """Test end-to-end execution returning structured execution stats."""
        # Arrange
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "doc1.doc").write_text("doc content")

        converted_dir = tmp_path / "converted"
        soffice_bin = tmp_path / "soffice"
        soffice_bin.touch()

        mock_result = convert_mod.ConversionResult(
            source=raw_dir / "doc1.doc",
            destination=converted_dir / "doc1.docx",
            status="OK",
            detail="converted",
        )

        # Act
        with patch.object(convert_mod, "resolve_raw_dir", return_value=raw_dir), \
             patch.object(convert_mod, "resolve_converted_dir", return_value=converted_dir), \
             patch.object(convert_mod, "resolve_soffice_path", return_value=soffice_bin), \
             patch.object(convert_mod, "convert_one", return_value=mock_result):

            result = convert_mod.run_conversion(
                raw_dir=raw_dir,
                converted_dir=converted_dir,
                soffice_path=str(soffice_bin),
                force=False,
                dry_run=False,
                timeout_sec=60,
                limit=0,
                keep_stale_converted=True,
            )

        # Assert
        assert result["summary"] == {"OK": 1, "SKIP": 0, "FAIL": 0, "DRYRUN": 0}
        assert result["total_legacy_files"] == 1
        assert result["processed_files"] == 1
        assert result["dry_run"] is False
