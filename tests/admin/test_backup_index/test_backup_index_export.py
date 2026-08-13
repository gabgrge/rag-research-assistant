"""Tests for export operations of src/admin/backup_index.py."""

import json
import pytest

from src.admin import backup_index


class TestExportOperations:
    """Tests export_backup, build_manifest, and write_manifest."""

    def test_write_manifest_creates_valid_json(self, mock_project):
        """Test build_manifest and write_manifest produce valid JSON with expected keys."""
        # Arrange
        backup_dir = mock_project["backups_dir"] / "test_backup"
        (backup_dir / "chroma").mkdir(parents=True)
        (backup_dir / "registry").mkdir(parents=True)
        (backup_dir / "registry" / "sources.csv").write_text("registry content", encoding="utf-8")

        # Act
        manifest_path = backup_index.write_manifest(backup_dir=backup_dir, include_registry=True)

        # Assert
        assert manifest_path.exists()
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert content["registry_included"] is True
        assert "registry_sha256" in content

    def test_export_backup_full_flow_with_zip(self, mock_project):
        """Test exporting a backup directory with registry and zip archive creation."""
        # Act
        backup_index.export_backup(
            output_dir=mock_project["backups_dir"],
            name="full_export",
            zip_archive=True,
            include_registry=True,
        )

        # Assert
        exported_folder = mock_project["backups_dir"] / "full_export"
        assert (exported_folder / "chroma" / "chroma.sqlite3").exists()
        assert (exported_folder / "registry" / "sources.csv").exists()
        assert (exported_folder / "manifest.json").exists()
        assert (mock_project["backups_dir"] / "full_export.zip").exists()

    def test_export_backup_raises_if_index_missing(self, monkeypatch, mock_project):
        """Test export_backup raises FileNotFoundError if INDEX_DIR does not exist."""
        # Arrange
        monkeypatch.setattr(backup_index, "INDEX_DIR", mock_project["root"] / "non_existent")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Index directory not found"):
            backup_index.export_backup(mock_project["backups_dir"], "test", False, False)

    def test_export_backup_raises_if_already_exists(self, mock_project):
        """Test export_backup raises FileExistsError if output backup folder exists."""
        # Arrange
        existing = mock_project["backups_dir"] / "duplicate_backup"
        existing.mkdir()

        # Act & Assert
        with pytest.raises(FileExistsError, match="Backup directory already exists"):
            backup_index.export_backup(mock_project["backups_dir"], "duplicate_backup", False, False)
