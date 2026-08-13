"""Tests for end-to-end update pipeline orchestrator (src/core/update_pipeline.py)."""

from pathlib import Path
from unittest.mock import patch
import pytest

from src.core import update_pipeline


class TestPathResolvers:
    """Tests for raw and converted directory resolution functions."""

    def test_resolve_raw_dir_returns_explicit_path(self):
        """Test returning explicit Path if raw_dir parameter is provided."""
        # Arrange
        custom_path = Path("/custom/raw/path")

        # Act
        resolved = update_pipeline.resolve_raw_dir(custom_path)

        # Assert
        assert resolved == custom_path

    def test_resolve_raw_dir_uses_env_var_when_none(self, monkeypatch, tmp_path):
        """Test falling back to RAW_DIR environment variable when raw_dir is None."""
        # Arrange
        monkeypatch.setenv("RAW_DIR", str(tmp_path))

        # Act
        resolved = update_pipeline.resolve_raw_dir(None)

        # Assert
        assert resolved == tmp_path

    def test_resolve_raw_dir_raises_file_not_found_when_env_var_missing(self, monkeypatch):
        """Test raising FileNotFoundError when RAW_DIR is empty or not defined."""
        # Arrange
        monkeypatch.delenv("RAW_DIR", raising=False)

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="RAW_DIR not set"):
            update_pipeline.resolve_raw_dir(None)

    def test_resolve_converted_dir_returns_explicit_path(self):
        """Test returning explicit Path if converted_dir is provided."""
        # Arrange
        custom_path = Path("/custom/converted/path")

        # Act
        resolved = update_pipeline.resolve_converted_dir(custom_path)

        # Assert
        assert resolved == custom_path

    def test_resolve_converted_dir_returns_default_when_none(self):
        """Test returning DEFAULT_CONVERTED_DIR when converted_dir is None."""
        # Act
        resolved = update_pipeline.resolve_converted_dir(None)

        # Assert
        assert resolved == update_pipeline.DEFAULT_CONVERTED_DIR


class TestUpdatePipelineExecution:
    """Tests for run_update_pipeline orchestration and step dispatching."""

    @pytest.fixture
    def mock_steps(self):
        """Fixture providing mocks for the 4 underlying pipeline step runners."""
        with patch("src.core.update_pipeline.run_conversion") as mock_conv, \
             patch("src.core.update_pipeline.run_scan_and_extract") as mock_scan, \
             patch("src.core.update_pipeline.run_chunking") as mock_chunk, \
             patch("src.core.update_pipeline.run_indexing") as mock_index:

            mock_conv.return_value = {"converted": 2}
            mock_scan.return_value = {"scanned": 5}
            mock_chunk.return_value = {"chunked": 10}
            mock_index.return_value = {"indexed": 10}

            yield {
                "convert": mock_conv,
                "scan_extract": mock_scan,
                "chunk": mock_chunk,
                "index": mock_index,
            }

    def test_run_update_pipeline_runs_all_steps_by_default(self, mock_steps, monkeypatch, tmp_path):
        """Test that running pipeline with default configs executes all 4 steps."""
        # Arrange
        monkeypatch.setenv("RAW_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        # Act
        results = update_pipeline.run_update_pipeline()

        # Assert
        assert "convert" in results
        assert "scan_extract" in results
        assert "chunk" in results
        assert "index" in results

        mock_steps["convert"].assert_called_once()
        mock_steps["scan_extract"].assert_called_once()
        mock_steps["chunk"].assert_called_once()
        mock_steps["index"].assert_called_once()

    def test_run_update_pipeline_skips_disabled_steps(self, mock_steps, monkeypatch, tmp_path):
        """Test that setting enabled=False on step configs skips execution of those steps."""
        # Arrange
        monkeypatch.setenv("RAW_DIR", str(tmp_path))

        # Act
        results = update_pipeline.run_update_pipeline(
            conversion=update_pipeline.ConversionStepConfig(enabled=False),
            index=update_pipeline.IndexStepConfig(enabled=False),
        )

        # Assert
        assert "convert" not in results
        assert "scan_extract" in results
        assert "chunk" in results
        assert "index" not in results

        mock_steps["convert"].assert_not_called()
        mock_steps["scan_extract"].assert_called_once()
        mock_steps["chunk"].assert_called_once()
        mock_steps["index"].assert_not_called()

    def test_run_update_pipeline_forwards_custom_configs(self, mock_steps, tmp_path):
        """Test that explicit step parameters are accurately passed to runner functions."""
        # Arrange
        raw_path = tmp_path / "raw"
        conv_path = tmp_path / "converted"

        conv_cfg = update_pipeline.ConversionStepConfig(
            raw_dir=raw_path,
            converted_dir=conv_path,
            force=True,
            timeout_sec=60,
        )
        scan_cfg = update_pipeline.ScanExtractStepConfig(scan_only=True)
        chunk_cfg = update_pipeline.ChunkStepConfig(limit=3, leaf_target=200)
        index_cfg = update_pipeline.IndexStepConfig(openai_api_key="custom-key", batch_size=32)

        # Act
        update_pipeline.run_update_pipeline(
            conversion=conv_cfg,
            scan_extract=scan_cfg,
            chunk=chunk_cfg,
            index=index_cfg,
        )

        # Assert
        mock_steps["convert"].assert_called_once_with(
            raw_dir=raw_path,
            converted_dir=conv_path,
            soffice_path="",
            force=True,
            dry_run=False,
            timeout_sec=60,
            limit=0,
            keep_stale_converted=False,
        )

        mock_steps["scan_extract"].assert_called_once_with(
            scan_only=True,
            keep_missing_json=False,
            reextract_extracted=False,
        )

        assert mock_steps["chunk"].call_args.kwargs["limit"] == 3
        assert mock_steps["chunk"].call_args.kwargs["leaf_target"] == 200

        assert mock_steps["index"].call_args.kwargs["openai_api_key"] == "custom-key"
        assert mock_steps["index"].call_args.kwargs["batch_size"] == 32

    def test_index_step_falls_back_to_env_api_key(self, mock_steps, monkeypatch, tmp_path):
        """Test that indexing step retrieves OPENAI_API_KEY from env if dataclass api_key is empty."""
        # Arrange
        monkeypatch.setenv("RAW_DIR", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")

        # Act
        update_pipeline.run_update_pipeline(
            conversion=update_pipeline.ConversionStepConfig(enabled=False),
            scan_extract=update_pipeline.ScanExtractStepConfig(enabled=False),
            chunk=update_pipeline.ChunkStepConfig(enabled=False),
            index=update_pipeline.IndexStepConfig(enabled=True, openai_api_key=""),
        )

        # Assert
        assert mock_steps["index"].call_args.kwargs["openai_api_key"] == "env-api-key"
