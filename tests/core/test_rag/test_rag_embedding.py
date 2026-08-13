"""Tests for embedding operations using OpenAI client."""

import pytest
from src.core import rag


@pytest.fixture
def default_embed_config():
    """Fixture providing default embedding configuration."""
    return {
        "embedding_model": "text-embedding-3-small",
        "max_retries": 3,
        "retry_base_delay_sec": 1.0,
    }


@pytest.fixture
def embed_query_caller(default_embed_config):
    """Fixture to call embed_query with default parameters and allow overrides."""
    def _call(client, text="test query", **override_kwargs):
        kwargs = {**default_embed_config, **override_kwargs}
        return rag.embed_query(client=client, text=text, **kwargs)

    return _call


class TestQueryEmbedding:
    """Tests for embedding query text using OpenAI API."""
    
    def test_embed_query_with_valid_response_returns_embedding_vector(
            self,
            mock_openai_client,
            make_embedding_response,
            embed_query_caller
    ):
        """Test successful embedding generation with valid API response."""
        # Arrange
        mock_response = make_embedding_response(embedding=[0.1, 0.2, 0.3])
        mock_openai_client.embeddings.create.return_value = mock_response
        
        # Act
        result = embed_query_caller(client=mock_openai_client)
        
        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.parametrize(
        "include_data, include_embedding, expected_error_msg",
        [
            (False, True, "Embedding response missing data"),
            (True, False, "Embedding response missing embedding"),
        ],
        ids=["missing_data_field", "missing_embedding_field"],
    )
    def test_embed_query_with_invalid_response_raises_runtime_error(
            self,
            mock_openai_client,
            make_embedding_response,
            include_data,
            include_embedding,
            expected_error_msg,
            embed_query_caller
    ):
        """Test that malformed embedding responses raise appropriate RuntimeError."""
        # Arrange
        mock_response = make_embedding_response(
            include_data=include_data,
            include_embedding=include_embedding,
        )
        mock_openai_client.embeddings.create.return_value = mock_response

        # Act & Assert
        with pytest.raises(RuntimeError, match=expected_error_msg):
            embed_query_caller(client=mock_openai_client)
