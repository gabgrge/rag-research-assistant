"""Tests for path resolution and helper functions in src/pipeline/convert_legacy_office_to_modern.py."""

from pathlib import Path
from unittest.mock import patch
import pytest

from src.pipeline import convert_legacy_office_to_modern as convert_mod


# ============================================================================
# 1. Directory Resolution Tests
# ============================================================================

class TestResolveRawDir:
    """Tests for resolve_raw_dir helper."""

    def test_resolve_raw_dir_explicit_path_success(self, tmp_path):
        """Test returning resolved Path when valid raw_dir is provided explicitly."""
        # Arrange
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Act
        resolved = convert_mod.resolve_raw_dir(raw_dir)

        # Assert
        assert resolved == raw_dir.resolve()

    def test_resolve_raw_dir_env_var_success(self, tmp_path, monkeypatch):
        """Test resolving RAW_DIR from environment variable when arg is None."""
        # Arrange
        raw_dir = tmp_path / "env_raw"
        raw_dir.mkdir()
        monkeypatch.setenv("RAW_DIR", str(raw_dir))

        # Act
        resolved = convert_mod.resolve_raw_dir(None)

        # Assert
        assert resolved == raw_dir.resolve()

    def test_resolve_raw_dir_missing_env_raises_file_not_found(self, monkeypatch):
        """Test raising FileNotFoundError when RAW_DIR is missing from env and arg."""
        # Arrange
        monkeypatch.delenv("RAW_DIR", raising=False)

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="RAW_DIR not set"):
            convert_mod.resolve_raw_dir(None)

    @pytest.mark.parametrize(
        ("dir_builder", "error_type", "error_match"),
        [
            (lambda base: base / "non_existent", FileNotFoundError, "RAW_DIR not found"),
            (lambda base: (base / "file.txt").touch() or (base / "file.txt"), NotADirectoryError, "RAW_DIR is not a directory"),
        ],
        ids=["non_existent_path", "file_not_a_directory"],
    )
    def test_resolve_raw_dir_invalid_paths_raise_exceptions(self, tmp_path, dir_builder, error_type, error_match):
        """Test raising appropriate exceptions when path does not exist or is a file."""
        # Arrange
        target_path = dir_builder(tmp_path)

        # Act & Assert
        with pytest.raises(error_type, match=error_match):
            convert_mod.resolve_raw_dir(target_path)


# ============================================================================
# 2. Executable Resolution Tests
# ============================================================================

class TestResolveSofficePath:
    """Tests for resolve_soffice_path and find_in_path helpers."""

    def test_resolve_soffice_path_explicit_valid_path(self, tmp_path):
        """Test returning explicitly provided valid soffice executable path."""
        # Arrange
        soffice_bin = tmp_path / "soffice.exe"
        soffice_bin.touch()

        # Act
        resolved = convert_mod.resolve_soffice_path(str(soffice_bin))

        # Assert
        assert resolved == soffice_bin.resolve()

    def test_resolve_soffice_path_explicit_invalid_path_raises(self, tmp_path):
        """Test raising FileNotFoundError when explicit path does not exist."""
        # Arrange
        invalid_bin = str(tmp_path / "non_existent_soffice")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="soffice executable not found"):
            convert_mod.resolve_soffice_path(invalid_bin)

    def test_resolve_soffice_path_from_system_path(self, tmp_path, monkeypatch):
        """Test locating soffice executable via PATH environment variable."""
        # Arrange
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        soffice_bin = bin_dir / "soffice"
        soffice_bin.touch()
        monkeypatch.setenv("PATH", str(bin_dir))

        # Act
        resolved = convert_mod.resolve_soffice_path("")

        # Assert
        assert resolved == soffice_bin.resolve()

    def test_resolve_soffice_path_windows_default_fallback(self, tmp_path, monkeypatch):
        """Test fallback to default Windows soffice installation path."""
        # Arrange
        monkeypatch.setenv("PATH", "")
        fake_win_path = tmp_path / "soffice.exe"
        fake_win_path.touch()

        # Act
        with patch.object(convert_mod, "DEFAULT_WINDOWS_SOFFICE", fake_win_path):
            resolved = convert_mod.resolve_soffice_path("")

        # Assert
        assert resolved == fake_win_path.resolve()

    def test_resolve_soffice_path_not_found_raises(self, monkeypatch):
        """Test raising FileNotFoundError when LibreOffice cannot be found anywhere."""
        # Arrange
        monkeypatch.setenv("PATH", "")

        # Act & Assert
        with patch.object(convert_mod, "DEFAULT_WINDOWS_SOFFICE", Path("/non/existent/soffice.exe")), \
             pytest.raises(FileNotFoundError, match="LibreOffice not found"):
            convert_mod.resolve_soffice_path("")


# ============================================================================
# 3. File Discovery and Plan Creation Tests
# ============================================================================

class TestFileDiscoveryAndPlanning:
    """Tests for iter_legacy_files, build_conversion_plan, and is_up_to_date."""

    def test_iter_legacy_files_filters_correctly(self, tmp_path):
        """Test discovery includes valid .doc/.ppt and excludes temp/other files."""
        # Arrange
        (tmp_path / "doc1.doc").touch()
        (tmp_path / "pres.PPT").touch()
        (tmp_path / "~$temp.doc").touch()
        (tmp_path / "already.docx").touch()
        (tmp_path / "other.txt").touch()

        # Act
        discovered = list(convert_mod.iter_legacy_files(tmp_path))

        # Assert
        discovered_names = {p.name for p in discovered}
        assert discovered_names == {"doc1.doc", "pres.PPT"}

    @pytest.mark.parametrize(
        ("suffix", "expected_suffix", "expected_fmt"),
        [
            (".doc", ".docx", "docx"),
            (".ppt", ".pptx", "pptx"),
        ],
    )
    def test_build_conversion_plan(self, tmp_path, suffix, expected_suffix, expected_fmt):
        """Test creating ConversionPlan with correct target path and extension mapping."""
        # Arrange
        raw_dir = tmp_path / "raw"
        converted_dir = tmp_path / "converted"
        source = raw_dir / "subfolder" / f"document{suffix}"

        # Act
        plan = convert_mod.build_conversion_plan(source, raw_dir, converted_dir)

        # Assert
        assert plan.source == source
        assert plan.destination == converted_dir / "subfolder" / f"document{expected_suffix}"
        assert plan.output_format == expected_fmt

    def test_is_up_to_date_cases(self, tmp_path):
        """Test up-to-date check across mtime and empty-file edge cases."""
        # Arrange
        src = tmp_path / "source.doc"
        dst = tmp_path / "dest.docx"
        src.touch()

        # Case 1: Destination does not exist
        assert not convert_mod.is_up_to_date(src, dst)

        # Case 2: Destination exists but is empty (0 bytes)
        dst.touch()
        assert not convert_mod.is_up_to_date(src, dst)

        # Case 3: Destination exists, non-empty, newer than source
        dst.write_text("content")
        assert convert_mod.is_up_to_date(src, dst)
