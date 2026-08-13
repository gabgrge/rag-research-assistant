"""Tests for RAG query parameter validation."""

import pytest
from src.core import rag


class TestRAGQueryParameterValidation:
    """Tests for validating RAG query parameters are within acceptable ranges."""

    @pytest.mark.parametrize(
        "param, invalid_value, expected_error_pattern",
        [
            # Non-negative integer constraints (must be >= 0)
            ("top_k", -1, "top_k must be >= 0"),
            ("max_per_source", -1, "max_per_source must be >= 0"),
            ("neighbor_expansion", -1, "neighbor_expansion must be >= 0"),
            ("context_max_tokens", -1, "context_max_tokens must be >= 0"),
            ("max_parents", -1, "max_parents must be >= 0"),
            ("max_retries", -1, "max_retries must be >= 0"),
            # Strictly positive float/int constraints (must be > 0)
            ("request_timeout_sec", 0, "request_timeout_sec must be > 0"),
            ("retry_base_delay_sec", 0, "retry_base_delay_sec must be > 0"),
        ],
        ids=[
            "negative_top_k",
            "negative_max_per_source",
            "negative_neighbor_expansion",
            "negative_context_max_tokens",
            "negative_max_parents",
            "negative_max_retries",
            "zero_request_timeout",
            "zero_retry_delay",
        ],
    )
    def test_run_rag_query_with_invalid_parameter_raises_value_error(
            self,
            param,
            invalid_value,
            expected_error_pattern
    ):
        """Test that invalid numeric bounds for parameters raise ValueError with proper messages."""
        # Arrange
        invalid_kwargs = {"query": "test", param: invalid_value}

        # Act & Assert
        with pytest.raises(ValueError, match=expected_error_pattern):
            rag.run_rag_query(**invalid_kwargs)
