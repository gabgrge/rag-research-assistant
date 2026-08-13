"""Tests for LLM answer generation: schemas, response extraction, generation."""

from types import SimpleNamespace
import pytest
from src.core import rag


class TestLLMGenerationSchema:
    """Tests for generation schema that constrains model output format."""
    
    def test_build_generation_schema_has_required_structure(self):
        """Test that generation schema has required properties and types."""
        # Arrange & Act
        schema = rag.build_generation_schema()
        
        # Assert
        assert schema["type"] == "object"
        assert "answer" in schema["properties"]
        assert "citation_ids" in schema["properties"]
        assert schema["properties"]["citation_ids"]["type"] == "array"
        assert "answer" in schema["required"]
        assert "citation_ids" in schema["required"]
        assert schema["additionalProperties"] is False


class TestLLMResponseTextExtraction:
    """Tests for extracting text content from LLM API responses."""
    
    def test_extract_response_text_with_output_text_field_returns_it(self):
        """Test extraction when response has output_text field."""
        # Arrange
        response = SimpleNamespace(output_text="Response text")
        
        # Act
        result = rag.extract_response_text(response)
        
        # Assert
        assert result == 'Response text'
    
    def test_extract_response_text_from_nested_output_structure_succeeds(self):
        """Test extraction from nested output[0].content[0].text structure."""
        # Arrange
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    content=[
                        SimpleNamespace(text="Extracted text")
                    ]
                )
            ]
        )
        
        # Act
        result = rag.extract_response_text(response)
        
        # Assert
        assert result == 'Extracted text'
    
    def test_extract_response_text_with_multiple_parts_joins_with_newline(self):
        """Test that multiple text parts are joined with newlines."""
        # Arrange
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    content=[
                        SimpleNamespace(text="Part 1"),
                        SimpleNamespace(text="Part 2")
                    ]
                )
            ]
        )
        
        # Act
        result = rag.extract_response_text(response)
        
        # Assert
        assert result == 'Part 1\nPart 2'

    @pytest.mark.parametrize(
        "invalid_response",
        [
            SimpleNamespace(output_text="   ", output=None),
            SimpleNamespace(output=[SimpleNamespace(content=[])]),
        ],
        ids=["empty_whitespace_output_text", "empty_content_parts"],
    )
    def test_extract_response_text_with_invalid_response_raises_runtime_error(self, invalid_response):
        """Test that invalid or missing response content raises RuntimeError."""
        # Act & Assert
        with pytest.raises(RuntimeError, match="Could not extract text"):
            rag.extract_response_text(invalid_response)


class TestLLMAnswerGeneration:
    """Tests for generating answers from LLM with context."""

    @staticmethod
    def _make_llm_response(text: str) -> SimpleNamespace:
        """Helper to create nested response structure expected by extract_response_text."""
        return SimpleNamespace(
            output=[
                SimpleNamespace(content=[SimpleNamespace(text=text)])
            ]
        )
    
    def test_generate_answer_with_context_returns_json_structure(self, mock_openai_client):
        """Test successful answer generation with context blocks."""
        # Arrange
        mock_openai_client.responses.create.return_value = self._make_llm_response(
            '{"answer": "Test answer", "citation_ids": ["C1"]}'
        )
        
        # Act
        result = rag.generate_answer(
            client=mock_openai_client,
            model="gpt-5-mini",
            query="What is RAG?",
            context_blocks=["[C1] Context block 1"],
            max_retries=1,
            retry_base_delay_sec=0.1,
        )
        
        # Assert
        assert result["answer"] == "Test answer"
        assert "C1" in result["citation_ids"]
    
    def test_generate_answer_without_context_returns_no_source_sentinel(self, mock_openai_client):
        """Test that empty context returns NO_SOURCE_SENTINEL without calling API."""
        # Arrange & Act
        result = rag.generate_answer(
            client=mock_openai_client,
            model="gpt-5-mini",
            query="What is RAG?",
            context_blocks=[],
            max_retries=1,
            retry_base_delay_sec=0.1,
        )
        
        # Assert
        assert result["answer"] == rag.NO_SOURCE_SENTINEL
        assert result["citation_ids"] == []
        mock_openai_client.responses.create.assert_not_called()
    
    def test_generate_answer_with_invalid_json_raises_runtime_error(self, mock_openai_client):
        """Test that non-JSON response raises RuntimeError."""
        # Arrange
        mock_openai_client.responses.create.return_value = self._make_llm_response(
            "Not valid JSON"
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="Model did not return valid JSON"):
            rag.generate_answer(
                client=mock_openai_client,
                model="gpt-5-mini",
                query="What is RAG?",
                context_blocks=["[C1] Context block 1"],
                max_retries=1,
                retry_base_delay_sec=0.1,
            )
