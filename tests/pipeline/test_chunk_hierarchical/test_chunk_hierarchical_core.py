"""Tests for low-level configuration, hashing, tokenization, and unit span helpers in src/pipeline/chunk_hierarchical.py."""

from unittest.mock import MagicMock, patch
import pytest

from src.pipeline import chunk_hierarchical as chunk_mod


# ============================================================================
# 1. WindowConfig Validation Tests
# ============================================================================

class TestWindowConfigValidation:
    """Tests for WindowConfig dataclass validation constraints."""

    def test_validate_window_config_valid_config_passes(self, sample_leaf_config):
        """Test that a well-formed WindowConfig passes validation without errors."""
        # Arrange
        config = sample_leaf_config

        # Act & Assert
        chunk_mod.validate_window_config(config)

    @pytest.mark.parametrize(
        "invalid_kwargs, expected_error",
        [
            ({"min_tokens": 0}, "min_tokens must be > 0"),
            ({"target_tokens": 5, "min_tokens": 10}, "target_tokens must be >= min_tokens"),
            ({"max_tokens": 15, "target_tokens": 20}, "max_tokens must be >= target_tokens"),
            ({"overlap_tokens": -1}, "overlap_tokens must be >= 0"),
            ({"overlap_tokens": 20, "target_tokens": 20}, "overlap_tokens must be < target_tokens"),
        ],
        ids=[
            "min_tokens_zero",
            "target_less_than_min",
            "max_less_than_target",
            "overlap_negative",
            "overlap_not_less_than_target",
        ]
    )
    def test_validate_window_config_raises_value_error(self, sample_leaf_config, invalid_kwargs, expected_error):
        """Test that invalid config parameters trigger explicit ValueErrors."""
        # Arrange
        base_dict = sample_leaf_config.as_dict()
        base_dict.update(invalid_kwargs)
        config = chunk_mod.WindowConfig(**base_dict)

        # Act & Assert
        with pytest.raises(ValueError, match=expected_error):
            chunk_mod.validate_window_config(config)


# ============================================================================
# 2. Config Hashing Tests
# ============================================================================

class TestComputeConfigHash:
    """Tests for deterministic SHA256 config hash generation."""

    def test_compute_config_hash_returns_deterministic_hex(self, sample_leaf_config, sample_parent_config):
        """Test hash generation produces identical 64-char hex string for identical configs."""
        # Arrange & Act
        hash1 = chunk_mod.compute_config_hash(
            leaf=sample_leaf_config,
            parent=sample_parent_config,
            tokenizer_mode="whitespace",
            tokenizer_encoding="cl100k_base",
            min_leaf_output_tokens=0,
        )
        hash2 = chunk_mod.compute_config_hash(
            leaf=sample_leaf_config,
            parent=sample_parent_config,
            tokenizer_mode="whitespace",
            tokenizer_encoding="cl100k_base",
            min_leaf_output_tokens=0,
        )

        # Assert
        assert len(hash1) == 64
        assert hash1 == hash2

    def test_compute_config_hash_changes_on_parameter_mutation(self, sample_leaf_config, sample_parent_config):
        """Test that changing any parameter alters the resulting hash value."""
        # Arrange
        base_hash = chunk_mod.compute_config_hash(
            leaf=sample_leaf_config,
            parent=sample_parent_config,
            tokenizer_mode="whitespace",
            tokenizer_encoding="cl100k_base",
            min_leaf_output_tokens=0,
        )

        # Act
        mutated_hash = chunk_mod.compute_config_hash(
            leaf=sample_leaf_config,
            parent=sample_parent_config,
            tokenizer_mode="tiktoken",  # Mutated parameter
            tokenizer_encoding="cl100k_base",
            min_leaf_output_tokens=0,
        )

        # Assert
        assert base_hash != mutated_hash


# ============================================================================
# 3. Tokenizer Spans Tests
# ============================================================================

class TestTokenizerSpans:
    """Tests for whitespace and tiktoken tokenization with character span offsets."""

    def test_tokenize_whitespace_mode_returns_spans(self):
        """Test whitespace tokenizer returns exact character offset tuples for non-whitespace terms."""
        # Arrange
        text = "Hello world! Test text."

        # Act
        spans, mode = chunk_mod.tokenize_with_spans(text, mode="whitespace", encoding_name="cl100k_base")

        # Assert
        assert mode == "whitespace"
        assert len(spans) == 4
        assert text[spans[0][0]:spans[0][1]] == "Hello"
        assert text[spans[1][0]:spans[1][1]] == "world!"

    def test_tokenize_invalid_mode_raises_value_error(self):
        """Test passing an unhandled mode name raises ValueError."""
        # Arrange
        text = "Sample text"

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported tokenizer mode"):
            chunk_mod.tokenize_with_spans(text, mode="invalid_mode", encoding_name="cl100k_base")

    def test_tokenize_tiktoken_spans_success(self):
        """Test tiktoken tokenizer extracts character spans correctly when tiktoken is installed."""
        # Arrange
        text = "Artificial intelligence"
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [101, 202]
        mock_encoding.decode_single_token_bytes.side_effect = [b"Artificial", b" intelligence"]

        with patch("tiktoken.get_encoding", return_value=mock_encoding):
            # Act
            spans = chunk_mod.tokenize_with_tiktoken_spans(text, "cl100k_base")

        # Assert
        assert spans is not None
        assert len(spans) == 2
        assert text[spans[0][0]:spans[0][1]] == "Artificial"
        assert text[spans[1][0]:spans[1][1]] == " intelligence"

    def test_tokenize_tiktoken_fallback_when_tiktoken_fails(self):
        """Test auto mode falls back to whitespace when tiktoken fails or is missing."""
        # Arrange
        text = "Fallback to whitespace"

        with patch("tiktoken.get_encoding", side_effect=Exception("Encoding load failed")):
            # Act
            spans, mode = chunk_mod.tokenize_with_spans(text, mode="auto", encoding_name="invalid_encoding")

        # Assert
        assert mode == "whitespace"
        assert len(spans) == 3


# ============================================================================
# 4. Unit Span Validation Helpers
# ============================================================================

class TestUnitSpanHelpers:
    """Tests for unit span verification and normalization functions."""

    def test_spans_are_ordered_non_overlapping_returns_true_for_valid_units(self):
        """Test valid non-overlapping sequential units evaluate to True."""
        # Arrange
        units = [
            {"start_char": 0, "end_char": 10},
            {"start_char": 10, "end_char": 25},
            {"start_char": 30, "end_char": 40},
        ]

        # Act
        result = chunk_mod.spans_are_ordered_non_overlapping(units)

        # Assert
        assert result is True

    def test_spans_are_ordered_non_overlapping_returns_false_for_overlaps(self):
        """Test overlapping or misordered unit boundaries evaluate to False."""
        # Arrange
        units = [
            {"start_char": 0, "end_char": 15},
            {"start_char": 10, "end_char": 25},  # Overlaps previous end
        ]

        # Act
        result = chunk_mod.spans_are_ordered_non_overlapping(units)

        # Assert
        assert result is False

    def test_ensure_units_falls_back_on_invalid_or_missing_units(self):
        """Test ensure_units falls back to document-level unit when input units are malformed."""
        # Arrange
        text = "Full document body text."
        payload = {"units": "not_a_list"}

        # Act
        units = chunk_mod.ensure_units(payload, text)

        # Assert
        assert len(units) == 1
        assert units[0]["unit_type"] == "document"
        assert units[0]["text"] == text
        assert units[0]["start_char"] == 0
        assert units[0]["end_char"] == len(text)

    def test_ensure_units_normalizes_valid_raw_units(self):
        """Test ensure_units clips out-of-bound character indexes and filters empty text."""
        # Arrange
        text = "Hello paragraph one."
        payload = {
            "units": [
                {"unit_type": "p", "unit_index": 1, "start_char": -5, "end_char": 5},
                {"unit_type": "p", "unit_index": 2, "start_char": 6, "end_char": 100},
            ]
        }

        # Act
        units = chunk_mod.ensure_units(payload, text)

        # Assert
        assert len(units) == 2
        assert units[0]["start_char"] == 0
        assert units[0]["end_char"] == 5
        assert units[1]["end_char"] == len(text)


# ============================================================================
# 5. Text & Offset Utilities
# ============================================================================

class TestOffsetAndTextUtilities:
    """Tests for character trimming and byte-to-char conversion utilities."""

    def test_byte_to_char_index_calculates_correct_char(self):
        """Test byte offset array translates UTF-8 byte index to character index correctly."""
        # Arrange
        byte_offsets = [0, 1, 2, 5, 6]  # Represents string where 3rd character is 3 bytes (e.g. emoji or accent)

        # Act & Assert
        assert chunk_mod.byte_to_char_index(byte_offsets, 0) == 0
        assert chunk_mod.byte_to_char_index(byte_offsets, 2) == 2
        assert chunk_mod.byte_to_char_index(byte_offsets, 4) == 2  # Inside the 3-byte char
        assert chunk_mod.byte_to_char_index(byte_offsets, 5) == 3

    def test_trim_text_span_strips_leading_trailing_whitespace(self):
        """Test trim_text_span adjusts start/end offsets to exclude surrounding whitespace."""
        # Arrange
        text = "  Sample Content  "

        # Act
        start, end, chunk_text = chunk_mod.trim_text_span(text, 0, len(text))

        # Assert
        assert chunk_text == "Sample Content"
        assert start == 2
        assert end == 16
