"""Tests for restore operations and ZIP resolution of src/admin/backup_index.py."""

import zipfile
import pytest

from src.admin import backup_index


class TestRestoreOperations:
    """Tests backup restoration from folders or zip files."""

    def test_resolve_backup_root_dir_and_zip(self, tmp_path):
        """Test resolve_backup_root returns direct dir or extracts zip into temp_dir."""
        # Arrange
        backup_dir = tmp_path / "folder_backup"
        backup_dir.mkdir()

        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("manifest.json", "{}")

        temp_extract = tmp_path / "temp_extract"
        temp_extract.mkdir()

        # Act
        resolved_dir = backup_index.resolve_backup_root(backup_dir, temp_dir=None)
        resolved_zip = backup_index.resolve_backup_root(zip_path, temp_dir=temp_extract)

        # Assert
        assert resolved_dir == backup_dir
        assert (resolved_zip / "manifest.json").exists()

    def test_resolve_backup_root_errors(self, tmp_path):
        """Test resolve_backup_root raises ValueError for invalid extension or missing temp_dir."""
        # Arrange
        invalid_file = tmp_path / "file.txt"
        invalid_file.write_text("data", encoding="utf-8")

        zip_file = tmp_path / "archive.zip"
        zip_file.write_text("fake zip", encoding="utf-8")

        # Act & Assert
        with pytest.raises(ValueError, match="must be a backup directory or .zip archive"):
            backup_index.resolve_backup_root(invalid_file, temp_dir=None)

        with pytest.raises(RuntimeError, match="Internal error: temp_dir missing"):
            backup_index.resolve_backup_root(zip_file, temp_dir=None)

    def test_restore_backup_full_flow_with_zip(self, mock_project):
        """Test restoring index and registry from a .zip backup archive (triggers cleanup in finally)."""
        # Arrange
        backup_index.export_backup(mock_project["backups_dir"], "source_backup", zip_archive=True, include_registry=True)
        zip_backup_path = mock_project["backups_dir"] / "source_backup.zip"

        (mock_project["index_dir"] / "chroma.sqlite3").write_text("corrupted", encoding="utf-8")
        mock_project["registry_path"].write_text("corrupted", encoding="utf-8")

        # Act
        backup_index.restore_backup(input_path=zip_backup_path, restore_registry=True, confirmed=True)

        # Assert
        assert mock_project["index_dir"].exists()
        assert (mock_project["index_dir"] / "chroma.sqlite3").read_text(encoding="utf-8") == "sqlite mock data"
        assert mock_project["registry_path"].read_text(encoding="utf-8") == "id,source\n1,doc1.pdf"

    def test_restore_backup_missing_manifest_raises(self, tmp_path):
        """Test restore_backup raises FileNotFoundError if backup folder lacks manifest.json."""
        # Arrange
        bad_backup = tmp_path / "bad_backup"
        bad_backup.mkdir()

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            backup_index.restore_backup(bad_backup, restore_registry=False, confirmed=True)

    def test_restore_backup_missing_chroma_dir_raises(self, tmp_path):
        """Test restore_backup raises FileNotFoundError if chroma/ directory is missing."""
        # Arrange
        bad_backup = tmp_path / "backup_no_chroma"
        bad_backup.mkdir()
        (bad_backup / "manifest.json").write_text("{}", encoding="utf-8")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="chroma/ not found in backup"):
            backup_index.restore_backup(bad_backup, restore_registry=False, confirmed=True)

    def test_restore_backup_skips_registry_when_missing_in_backup(self, mock_project, tmp_path, capsys):
        """Test restore_backup prints skipped message when backup lacks registry folder."""
        # Arrange
        backup_dir = tmp_path / "backup_no_registry"
        (backup_dir / "chroma").mkdir(parents=True)
        (backup_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (backup_dir / "chroma" / "data.bin").write_text("index data", encoding="utf-8")

        # Act
        backup_index.restore_backup(input_path=backup_dir, restore_registry=True, confirmed=True)

        # Assert
        captured = capsys.readouterr()
        assert "Registry restore: skipped" in captured.out
