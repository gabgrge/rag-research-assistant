"""Tests for the CLI argument parsing of src/admin/backup_index.py."""

import pytest
from unittest.mock import patch
from pathlib import Path

from src.admin import backup_index


class TestCLIParser:
    """Tests CLI argument parsing for export and restore subcommands."""

    def test_parse_args_export_defaults(self, monkeypatch):
        """Test default arguments for export subcommand."""
        # Arrange
        monkeypatch.setattr("sys.argv", ["backup_index.py", "export"])

        # Act
        args = backup_index.parse_args()

        # Assert
        assert args.command == "export"
        assert args.output_dir == backup_index.BACKUP_ROOT
        assert args.name == ""
        assert args.zip is False
        assert args.no_registry is False

    @pytest.mark.parametrize(
        ("cli_args", "expected_skip_registry", "expected_yes"),
        [
            (["backup_index.py", "restore", "--input", "backup.zip", "--yes"], False, True),
            (["backup_index.py", "restore", "--input", "backup_folder", "--skip-registry", "--yes"], True, True),
        ],
    )
    def test_parse_args_restore_flags(self, cli_args, expected_skip_registry, expected_yes, monkeypatch):
        """Test restore subcommand parameter parsing using parametrization."""
        # Arrange
        monkeypatch.setattr("sys.argv", cli_args)

        # Act
        args = backup_index.parse_args()

        # Assert
        assert args.command == "restore"
        assert args.input == Path(cli_args[3])
        assert args.skip_registry is expected_skip_registry
        assert args.yes is expected_yes

    @patch("src.admin.backup_index.export_backup")
    def test_main_executes_export(self, mock_export, monkeypatch):
        """Test main() dispatches to export_backup when 'export' command is passed."""
        # Arrange
        monkeypatch.setattr("sys.argv", ["backup_index.py", "export", "--name", "test_run", "--zip"])

        # Act
        backup_index.main()

        # Assert
        assert mock_export.call_count == 1
        assert mock_export.call_args.kwargs["name"] == "test_run"
        assert mock_export.call_args.kwargs["zip_archive"] is True
        assert mock_export.call_args.kwargs["include_registry"] is True

    @patch("src.admin.backup_index.restore_backup")
    def test_main_executes_restore(self, mock_restore, monkeypatch):
        """Test main() dispatches to restore_backup when 'restore' command is passed."""
        # Arrange
        monkeypatch.setattr("sys.argv", ["backup_index.py", "restore", "--input", "my_backup", "--yes"])

        # Act
        backup_index.main()

        # Assert
        assert mock_restore.call_count == 1
        assert mock_restore.call_args.kwargs["input_path"] == Path("my_backup")
        assert mock_restore.call_args.kwargs["restore_registry"] is True
        assert mock_restore.call_args.kwargs["confirmed"] is True
