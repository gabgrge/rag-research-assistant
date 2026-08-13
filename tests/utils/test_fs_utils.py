"""Tests for filesystem operations and path hierarchy checks in src/utils/fs_utils.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.utils import fs_utils


# ============================================================================
# 1. Atomic Replace with Retry Tests
# ============================================================================

class TestAtomicReplaceWithRetry:
    """Tests for atomic file replacements, PermissionError retries, and failure escalation."""

    def test_atomic_replace_succeeds_on_first_try(self, tmp_path: Path):
        """Test atomic replacement moves file successfully when no permissions/locks block it."""
        # Arrange
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"
        source.write_text("content", encoding="utf-8")

        # Act
        fs_utils.atomic_replace_with_retry(source, target)

        # Assert
        assert not source.exists()
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "content"

    def test_atomic_replace_retries_on_permission_error_and_recovers(self, tmp_path: Path):
        """Test transient PermissionError triggers retry mechanism with backoff and eventual success."""
        # Arrange
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"
        source.write_text("content", encoding="utf-8")

        mock_logger = MagicMock()

        # Fail once with PermissionError, then succeed on second call
        with patch.object(Path, "replace", side_effect=[PermissionError("File locked"), None]) as mock_replace, \
             patch("time.sleep") as mock_sleep:
            # Act
            fs_utils.atomic_replace_with_retry(
                source,
                target,
                retries=3,
                base_delay_sec=0.1,
                logger=mock_logger,
            )

        # Assert
        assert mock_replace.call_count == 2
        mock_sleep.assert_called_once_with(0.1)
        mock_logger.warning.assert_called_once()

    def test_atomic_replace_raises_permission_error_after_exhausting_retries(
        self, tmp_path: Path
    ):
        """Test persistent PermissionError raises the exception after retries are exhausted."""
        # Arrange
        source = tmp_path / "source.txt"
        target = tmp_path / "target.txt"

        with patch.object(Path, "replace", side_effect=PermissionError("Locked")) as mock_replace, \
             patch("time.sleep") as mock_sleep:
            # Act & Assert
            with pytest.raises(PermissionError, match="Locked"):
                fs_utils.atomic_replace_with_retry(
                    source,
                    target,
                    retries=2,
                    base_delay_sec=0.01,
                )

        # Attempted initial try + 2 retries = 3 calls
        assert mock_replace.call_count == 3
        assert mock_sleep.call_count == 2

    def test_atomic_replace_raises_non_permission_error_immediately(
        self, tmp_path: Path
    ):
        """Test non-PermissionError exceptions (e.g., FileNotFoundError) raise immediately without retrying."""
        # Arrange
        source = tmp_path / "non_existent.txt"
        target = tmp_path / "target.txt"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            fs_utils.atomic_replace_with_retry(
                source,
                target,
                retries=3,
                base_delay_sec=0.1,
            )


# ============================================================================
# 2. Path Hierarchy Check Tests
# ============================================================================

class TestPathIsUnder:
    """Tests for relative path hierarchy validation in path_is_under."""

    @pytest.mark.parametrize(
        "candidate_rel, expected_result",
        [
            ("child.txt", True),
            ("subdir/child.txt", True),
            (".", True),  # Base itself is under base
            ("../outside.txt", False),
            ("../../outside.txt", False),
        ],
        ids=["direct_child", "nested_child", "same_directory", "parent_directory", "distant_relative"],
    )
    def test_path_is_under_relative_hierarchy(
        self, tmp_path: Path, candidate_rel: str, expected_result: bool
    ):
        """Test path_is_under evaluates relative paths against a base directory correctly."""
        # Arrange
        base = tmp_path / "base_dir"
        base.mkdir(parents=True, exist_ok=True)
        candidate = base / candidate_rel

        # Act
        result = fs_utils.path_is_under(base, candidate)

        # Assert
        assert result is expected_result
