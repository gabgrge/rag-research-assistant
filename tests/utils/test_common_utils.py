"""Unit tests for utility functions in src/utils/common_utils.py."""

from datetime import datetime, timezone
from unittest.mock import patch
import pytest

from src.utils import common_utils


# ============================================================================
# 1. UTC ISO Timestamp Tests
# ============================================================================

class TestUtcNowIso:
    """Tests for ISO 8601 formatted UTC timestamp generation."""

    def test_utc_now_iso_returns_valid_iso_string(self):
        """Test utc_now_iso returns a string that parses back into a UTC datetime."""
        # Act
        timestamp_str = common_utils.utc_now_iso()

        # Assert
        parsed_dt = datetime.fromisoformat(timestamp_str)
        assert isinstance(timestamp_str, str)
        assert parsed_dt.tzinfo == timezone.utc

    def test_utc_now_iso_format_correctness(self):
        """Test utc_now_iso returns deterministic ISO string format for mocked time."""
        # Arrange
        fixed_dt = datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc)

        with patch("src.utils.common_utils.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_dt
            mock_datetime.fromisoformat = datetime.fromisoformat

            # Act
            result = common_utils.utc_now_iso()

        # Assert
        assert result == "2026-01-15T12:30:00+00:00"
        mock_datetime.now.assert_called_once_with(timezone.utc)


# ============================================================================
# 2. Integer Conversion Tests
# ============================================================================

class TestToInt:
    """Tests for safe integer casting with default fallbacks."""

    @pytest.mark.parametrize(
        "input_value, expected_output",
        [
            # Int inputs
            (42, 42),
            (-10, -10),
            (0, 0),
            # Float inputs
            (3.14, 3),
            (-5.9, -5),
            # String inputs
            ("123", 123),
            ("-456", -456),
            (" 789 ", 789),
            # Bytes/Bytearray inputs
            (b"42", 42),
            (bytearray(b"100"), 100),
        ],
        ids=[
            "int_positive",
            "int_negative",
            "int_zero",
            "float_positive",
            "float_negative",
            "str_valid",
            "str_negative",
            "str_with_whitespace",
            "bytes_valid",
            "bytearray_valid",
        ],
    )
    def test_to_int_successful_conversions(self, input_value, expected_output):
        """Test valid types convert accurately to integer values."""
        assert common_utils.to_int(input_value) == expected_output

    @pytest.mark.parametrize(
        "invalid_value, default, expected_output",
        [
            ("not_a_number", 0, 0),
            ("not_a_number", 99, 99),
            ("12.34", -1, -1),  # Invalid string integer format
            (None, 0, 0),
            (None, -1, -1),
            ([1, 2, 3], 0, 0),
            ({"key": "val"}, 5, 5),
            (object(), 0, 0),
        ],
        ids=[
            "invalid_str_default_zero",
            "invalid_str_custom_default",
            "float_string_default",
            "none_default_zero",
            "none_custom_default",
            "list_type",
            "dict_type",
            "object_type",
        ],
    )
    def test_to_int_fallback_on_invalid_inputs(self, invalid_value, default, expected_output):
        """Test invalid types or unparseable strings gracefully return the designated default value."""
        assert common_utils.to_int(invalid_value, default=default) == expected_output
