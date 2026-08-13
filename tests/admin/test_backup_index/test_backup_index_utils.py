"""Tests for utility and cleanup operations of src/admin/backup_index.py."""

import stat
import pytest
from unittest.mock import patch

from src.admin import backup_index


class TestUtilities:
    """Tests hashing, project boundary validation, and naming utilities."""

    def test_sha256_file_calculates_correct_hash(self, tmp_path):
        """Test sha256_file produces accurate SHA-256 hex digest for known data."""
        # Arrange
        sample_file = tmp_path / "data.txt"
        sample_file.write_text("hello world", encoding="utf-8")

        # Act
        digest = backup_index.sha256_file(sample_file)

        # Assert
        assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_collect_file_hashes_and_missing_directory(self, tmp_path):
        """Test collect_file_hashes lists files or returns empty dict for missing folder."""
        # Arrange
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        (sub_dir / "file.txt").write_text("test", encoding="utf-8")

        # Act
        found_hashes = backup_index.collect_file_hashes(tmp_path)
        missing_hashes = backup_index.collect_file_hashes(tmp_path / "non_existent")

        # Assert
        assert "sub/file.txt" in found_hashes
        assert missing_hashes == {}

    @pytest.mark.parametrize(
        ("input_name", "expected_empty_check"),
        [
            ("custom_backup", "custom_backup"),
            ("   spaced_name   ", "spaced_name"),
            ("", None),
        ],
    )
    def test_build_backup_name(self, input_name, expected_empty_check):
        """Test backup naming returns custom stripped string or auto-generated timestamp."""
        # Act
        result = backup_index.build_backup_name(input_name)

        # Assert
        if expected_empty_check:
            assert result == expected_empty_check
        else:
            assert len(result) > 0

    def test_ensure_under_project_boundary_validation(self, mock_project, tmp_path):
        """Test ensure_under_project validates internal paths and rejects external paths."""
        # Arrange
        inside_path = mock_project["root"] / "data" / "sub"
        outside_path = tmp_path / "external_folder"

        # Act & Assert
        backup_index.ensure_under_project([inside_path])

        with pytest.raises(ValueError, match="Refusing to use path outside project"):
            backup_index.ensure_under_project([outside_path])


class TestDirectoryCleanup:
    """Tests for remove_dir_with_retry including error handling, retries, and read-only clearing."""

    def test_remove_dir_with_retry_deletes_folder(self, tmp_path):
        """Test successful removal of an existing directory tree."""
        # Arrange
        target = tmp_path / "folder_to_remove"
        target.mkdir()
        (target / "file.txt").write_text("content", encoding="utf-8")

        # Act
        backup_index.remove_dir_with_retry(target)

        # Assert
        assert not target.exists()

    def test_remove_dir_with_retry_early_exit_if_non_existent(self, tmp_path):
        """Test early return when path does not exist."""
        # Arrange
        missing_path = tmp_path / "does_not_exist"

        # Act
        backup_index.remove_dir_with_retry(missing_path)

        # Assert
        assert not missing_path.exists()

    def test_remove_dir_with_retry_clears_readonly_files(self, tmp_path):
        """Test _clear_readonly callback when encountering read-only permissions."""
        # Arrange
        target = tmp_path / "readonly_folder"
        target.mkdir()
        readonly_file = target / "locked.txt"
        readonly_file.write_text("read only content", encoding="utf-8")
        readonly_file.chmod(stat.S_IREAD)

        # Act
        backup_index.remove_dir_with_retry(target)

        # Assert
        assert not target.exists()

    @patch("shutil.rmtree")
    @patch("time.sleep")
    def test_remove_dir_with_retry_retries_on_permission_error(self, mock_sleep, mock_rmtree, tmp_path):
        """Test retrying on PermissionError and failing if retries are exhausted."""
        # Arrange
        target = tmp_path / "permission_locked"
        target.mkdir()
        mock_rmtree.side_effect = PermissionError("Access denied")

        # Act & Assert
        with pytest.raises(PermissionError, match="Access denied"):
            backup_index.remove_dir_with_retry(target, retries=2, base_delay_sec=0.01)

        assert mock_rmtree.call_count == 3
        assert mock_sleep.call_count == 2
