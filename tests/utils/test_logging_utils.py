"""Tests for script logger creation and initialization in src/utils/logging_utils.py."""

from pathlib import Path
import logging
import pytest

from src.utils import logging_utils


@pytest.fixture
def temp_log_path(tmp_path: Path) -> Path:
    """Provides a log file path nested in a non-existent subdirectory."""
    return tmp_path / "logs" / "nested" / "test.log"


class TestBuildScriptLogger:
    """Tests for build_script_logger creation, handler attachment, and idempotency."""

    def test_build_script_logger_creates_handlers_and_parent_directories(
        self, temp_log_path: Path
    ):
        """Test logger attaches file/stream handlers and automatically creates missing log directories."""
        # Arrange
        logger_name = "test_logger_unique_1"

        # Act
        logger = logging_utils.build_script_logger(logger_name, temp_log_path)

        # Assert
        assert logger.name == logger_name
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert temp_log_path.parent.exists()

    def test_build_script_logger_is_idempotent(self, temp_log_path: Path):
        """Test calling build_script_logger with an existing logger name returns existing instance without adding duplicate handlers."""
        # Arrange
        logger_name = "test_logger_unique_2"
        first_logger = logging_utils.build_script_logger(logger_name, temp_log_path)
        initial_handler_count = len(first_logger.handlers)

        # Act
        second_logger = logging_utils.build_script_logger(logger_name, temp_log_path)

        # Assert
        assert first_logger is second_logger
        assert len(second_logger.handlers) == initial_handler_count

    @pytest.mark.parametrize(
        "custom_level",
        [logging.DEBUG, logging.WARNING, logging.ERROR],
        ids=["debug_level", "warning_level", "error_level"],
    )
    def test_build_script_logger_respects_custom_level(
        self, temp_log_path: Path, custom_level: int
    ):
        """Test custom log levels are correctly assigned to the newly initialized logger."""
        # Arrange
        logger_name = f"test_logger_level_{custom_level}"

        # Act
        logger = logging_utils.build_script_logger(
            logger_name, temp_log_path, level=custom_level
        )

        # Assert
        assert logger.level == custom_level
