"""Tests for src/integrations/vector/chroma_utils.py."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest

from src.integrations.vector import chroma_utils


@pytest.fixture
def mock_chroma():
    """Fixture providing a mock chromadb module structure."""
    mock_collection = MagicMock(name="mock_collection")
    mock_client = MagicMock(name="mock_client")
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_persistent_client = MagicMock(name="mock_persistent_client", return_value=mock_client)
    mock_module = MagicMock(name="chromadb", PersistentClient=mock_persistent_client)

    return {
        "module": mock_module,
        "client_cls": mock_persistent_client,
        "client": mock_client,
        "collection": mock_collection,
    }


class TestBuildChromaSettings:
    """Tests for _build_chroma_settings helper."""

    def test_build_chroma_settings_success(self):
        """Test returning Settings object with telemetry disabled when chromadb.config is available."""
        # Arrange
        mock_settings_cls = MagicMock()
        mock_config_module = MagicMock(Settings=mock_settings_cls)

        # Act
        with patch.dict("sys.modules", {"chromadb.config": mock_config_module}):
            settings = chroma_utils._build_chroma_settings()

        # Assert
        assert settings == mock_settings_cls.return_value
        mock_settings_cls.assert_called_once_with(anonymized_telemetry=False)

    def test_build_chroma_settings_import_error_returns_none(self, monkeypatch):
        """Test returning None when chromadb.config cannot be imported."""
        # Arrange
        real_import = __import__

        def custom_import(name, *args, **kwargs):
            if name == "chromadb.config":
                raise ImportError("No module named chromadb.config")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", custom_import)

        # Act
        settings = chroma_utils._build_chroma_settings()

        # Assert
        assert settings is None


class TestBuildChromaCollection:
    """Tests for build_chroma_collection."""

    INDEX_DIR = Path("/tmp/test_index")
    COLLECTION_NAME = "test_collection"

    def test_missing_chromadb_dependency_raises_runtime_error(self, monkeypatch):
        """Test ImportError during chromadb import raises RuntimeError."""
        # Arrange
        monkeypatch.setattr("builtins.__import__", Mock(side_effect=ImportError("No module named chromadb")))

        # Act & Assert
        with pytest.raises(RuntimeError, match="Missing dependency 'chromadb'"):
            chroma_utils.build_chroma_collection(self.INDEX_DIR, self.COLLECTION_NAME)

    @pytest.mark.parametrize(
        ("settings_return", "expected_kwargs"),
        [
            (MagicMock(name="mock_settings"), {"path": "/tmp/test_index", "settings": MagicMock}),
            (None, {"path": "/tmp/test_index"}),
        ],
        ids=["with_settings", "without_settings"],
    )
    def test_build_chroma_collection_client_instantiation(self, mock_chroma, settings_return, expected_kwargs):
        """Test collection creation both with and without Settings fallback using parametrization."""
        # Arrange
        if expected_kwargs.get("settings") is MagicMock:
            expected_kwargs["settings"] = settings_return

        # Act
        with patch.dict("sys.modules", {"chromadb": mock_chroma["module"]}), \
             patch.object(chroma_utils, "_build_chroma_settings", return_value=settings_return):
            collection = chroma_utils.build_chroma_collection(self.INDEX_DIR, self.COLLECTION_NAME)

        # Assert
        assert collection == mock_chroma["collection"]
        mock_chroma["client_cls"].assert_called_once_with(**expected_kwargs)
        mock_chroma["client"].get_or_create_collection.assert_called_once_with(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
