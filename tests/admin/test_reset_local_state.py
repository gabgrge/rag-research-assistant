"""Tests for src/admin/reset_local_state.py."""

import os
import stat
from unittest.mock import patch
import pytest

from src.admin import reset_local_state


class TestCLIParser:
    """Tests argument parsing for reset_local_state."""

    @pytest.mark.parametrize(
        ("cli_args", "expected_yes"),
        [
            (["reset_local_state.py"], False),
            (["reset_local_state.py", "--yes"], True),
        ],
        ids=["default_no_flag", "with_yes_flag"]
    )
    def test_parse_args(self, cli_args, expected_yes, monkeypatch):
        """Test default and --yes flag parsing."""
        # Arrange
        monkeypatch.setattr("sys.argv", cli_args)

        # Act
        args = reset_local_state.parse_args()

        # Assert
        assert args.yes is expected_yes


class TestEnsureUnderProject:
    """Tests safety check ensure_under_project."""

    def test_ensure_under_project_valid_paths(self, mock_project):
        """Test valid project paths do not raise any exceptions."""
        # Arrange
        valid_paths = [
            mock_project["root"],
            mock_project["chunks_dir"],
            mock_project["logs_dir"],
        ]

        # Act & Assert
        reset_local_state.ensure_under_project(valid_paths)  # Should not raise

    def test_ensure_under_project_external_path_raises(self, tmp_path):
        """Test external path outside PROJECT_ROOT raises RuntimeError."""
        # Arrange
        external_path = tmp_path / "outside_project"
        external_path.mkdir()

        # Act & Assert
        with pytest.raises(RuntimeError, match="Refusing to delete outside project"):
            reset_local_state.ensure_under_project([external_path])


class TestRemovePath:
    """Tests file and directory removal logic and error handlers."""

    def test_remove_path_non_existent(self, tmp_path):
        """Test early return when path does not exist."""
        # Arrange
        missing_path = tmp_path / "does_not_exist"

        # Act
        reset_local_state.remove_path(missing_path)

        # Assert
        assert not missing_path.exists()

    def test_remove_path_single_file(self, tmp_path):
        """Test removing a single file."""
        # Arrange
        test_file = tmp_path / "standalone.txt"
        test_file.write_text("hello", encoding="utf-8")

        # Act
        reset_local_state.remove_path(test_file)

        # Assert
        assert not test_file.exists()

    def test_remove_path_directory_tree(self, tmp_path):
        """Test removing a directory tree containing files."""
        # Arrange
        target_dir = tmp_path / "nested_dir"
        target_dir.mkdir()
        (target_dir / "file.txt").write_text("data", encoding="utf-8")

        # Act
        reset_local_state.remove_path(target_dir)

        # Assert
        assert not target_dir.exists()

    def test_on_rm_error_clears_readonly_and_retries(self, tmp_path):
        """Test _on_rm_error changes file mode permissions to writable and retries removal."""
        # Arrange
        readonly_file = tmp_path / "locked.txt"
        readonly_file.write_text("content", encoding="utf-8")
        readonly_file.chmod(stat.S_IREAD)  # Make read-only

        mock_func = patch("os.unlink").start()

        # Act
        reset_local_state._on_rm_error(mock_func, str(readonly_file), None)

        # Assert
        mock_func.assert_called_once_with(str(readonly_file))
        assert os.access(readonly_file, os.W_OK)

        patch.stopall()


class TestMainWorkflow:
    """Tests execution flow of main()."""

    def test_main_without_yes_flag_aborts(self, mock_project, monkeypatch, capsys):
        """Test main() prints notice and aborts without removing state if --yes is absent."""
        # Arrange
        monkeypatch.setattr("sys.argv", ["reset_local_state.py"])

        # Act
        reset_local_state.main()

        # Assert
        captured = capsys.readouterr()
        assert "Re-run with --yes to confirm." in captured.out
        # Files should still exist
        assert (mock_project["leaf_chunks_dir"] / "chunk1.txt").exists()
        assert (mock_project["registry_dir"] / "sources.csv").exists()

    def test_main_with_yes_flag_resets_and_recreates(self, mock_project, monkeypatch, capsys):
        """Test main() removes RESET_DIRS and recreates RECREATE_DIRS when --yes is supplied."""
        # Arrange
        monkeypatch.setattr("sys.argv", ["reset_local_state.py", "--yes"])

        # Act
        reset_local_state.main()

        # Assert
        captured = capsys.readouterr()
        assert "Project root:" in captured.out
        assert "Will remove:" in captured.out
        assert "Will recreate:" in captured.out

        # Old files inside directories should be deleted
        assert not (mock_project["leaf_chunks_dir"] / "chunk1.txt").exists()
        assert not (mock_project["registry_dir"] / "sources.csv").exists()

        # Target directories in RECREATE_DIRS must be recreated
        assert mock_project["leaf_chunks_dir"].exists()
        assert mock_project["parent_chunks_dir"].exists()
        assert mock_project["converted_dir"].exists()
        assert mock_project["extracted_dir"].exists()
        assert mock_project["registry_dir"].exists()
        assert mock_project["logs_dir"].exists()
