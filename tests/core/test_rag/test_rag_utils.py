"""Tests for utility functions: token counting, metadata handling, formatting."""

import pytest
from src.core import rag


class TestTokenCountingUtilities:
    """Tests for token counting functions."""
    
    def test_token_count_with_text_falls_back_to_word_count(self):
        """Test that token counting falls back to word count if tiktoken unavailable."""
        # Arrange
        text = "one two three"
        
        # Act
        count = rag.token_count(text)
        
        # Assert
        assert count == 3
    
    def test_estimate_prompt_overhead_tokens_returns_positive_value(self):
        """Test that prompt overhead estimation returns positive token count."""
        # Arrange & Act
        overhead = rag.estimate_prompt_overhead_tokens("test query")
        
        # Assert
        assert overhead > 0


class TestMetadataHandling:
    """Tests for metadata extraction and formatting from candidates."""
    
    @pytest.mark.parametrize(
        "meta, expected",
        [
            ({"origin_path": "/path/to/file"}, "/path/to/file"),
            ({}, ""),
        ],
        ids=["origin_path_present", "origin_path_missing"]
    )
    def test_resolve_path_returns_expected_for_various_metadata(self, candidate_factory, meta, expected):
        """Test that resolve_path correctly extracts the origin path from candidate metadata."""
        # Arrange
        cand = candidate_factory(meta=meta)

        # Act & Assert
        assert rag.resolve_path(cand.meta) == expected


class TestRetrievalFieldsExtraction:
    """Tests for extracting retrieval-related fields from candidates."""
    
    def test_retrieval_fields_with_seed_candidate_returns_scores(self, candidate_factory):
        """Test that seed candidates return their assigned scores."""
        # Arrange
        cand = candidate_factory(origin="seed", score=0.5, retrieval_score=0.4, distance=0.1)
        
        # Act
        score, rscore, dist = rag.retrieval_fields(cand)
        
        # Assert
        assert isinstance(score, float) and score == pytest.approx(0.5)
        assert isinstance(rscore, float) and rscore == pytest.approx(0.4)
        assert isinstance(dist, float) and dist == pytest.approx(0.1)
    
    def test_retrieval_fields_with_expanded_candidate_returns_none_scores(self, candidate_factory):
        """Test that expanded candidates have None for retrieval scores."""
        # Arrange
        cand = candidate_factory(origin="expanded")
        
        # Act & Assert
        assert rag.retrieval_fields(cand) == (None, None, None)


class TestOptionalMetadataFieldHandling:
    """Tests for adding optional metadata fields to output payloads."""

    def test_add_optional_meta_fields_with_all_present_includes_all_fields(self):
        """Test that all optional metadata fields are added when present."""
        # Arrange
        payload = {}
        meta = {
            "unit_index_end": 5,
            "unit_type_end": "section",
            "char_start": 100,
            "char_end": 200,
        }

        # Act
        rag.add_optional_meta_fields(payload, meta)

        # Assert
        assert payload["unit_index_end"] == 5
        assert payload["unit_type_end"] == "section"
        assert payload["char_start"] == 100
        assert payload["char_end"] == 200

    def test_add_optional_meta_fields_with_missing_fields_ignores_them(self):
        """Test that missing optional fields don't add keys to payload."""
        # Arrange
        payload = {}
        meta = {}

        # Act
        rag.add_optional_meta_fields(payload, meta)

        # Assert
        assert payload == {}

    def test_add_optional_meta_fields_with_empty_strings_ignores_them(self):
        """Test that empty string values are not added to payload."""
        # Arrange
        payload = {}
        meta = {"unit_index_end": "", "char_start": 50}

        # Act
        rag.add_optional_meta_fields(payload, meta)

        # Assert
        assert "unit_index_end" not in payload
        assert payload["char_start"] == 50
