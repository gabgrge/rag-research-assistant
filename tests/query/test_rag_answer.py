"""Tests for CLI query entry-point (src/query/rag_answer.py)."""

import json
from unittest.mock import patch
import pytest

from src.query import rag_answer


class TestParseArgs:
    """Tests for CLI argument parsing and validation."""

    def test_parse_args_with_required_query_uses_defaults(self, monkeypatch):
        """Test parsing with minimal required arguments returns expected defaults."""
        monkeypatch.setattr("sys.argv", ["rag_answer.py", "--query", "What is RAG?"])

        args = rag_answer.parse_args()

        assert args.query == "What is RAG?"
        assert args.mode == "recherche"
        assert args.top_k == 0
        assert args.mmr is False
        assert args.include_parent is False
        assert args.no_parent is False
        assert args.filter == []

    def test_parse_args_with_custom_flags_overrides_defaults(self, monkeypatch):
        """Test parsing with explicit CLI flags."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "rag_answer.py",
                "--query", "What is RAG?",
                "--mode", "resume",
                "--top-k", "5",
                "--mmr",
                "--include-parent",
                "--filter", "source=doc1",
                "--filter", "type=pdf",
                "--debug",
            ],
        )

        args = rag_answer.parse_args()

        assert args.query == "What is RAG?"
        assert args.mode == "resume"
        assert args.top_k == 5
        assert args.mmr is True
        assert args.include_parent is True
        assert args.filter == ["source=doc1", "type=pdf"]
        assert args.debug is True

    def test_parse_args_mutually_exclusive_parent_flags_raises_error(self, monkeypatch):
        """Test that supplying both --include-parent and --no-parent fails argument parsing."""
        monkeypatch.setattr(
            "sys.argv",
            ["rag_answer.py", "--query", "q", "--include-parent", "--no-parent"],
        )

        with pytest.raises(SystemExit):
            rag_answer.parse_args()

    @pytest.mark.parametrize(
        "invalid_flag, value, match_msg",
        [
            ("--top-k", "-1", "--top-k must be >= 0"),
            ("--max-per-source", "-1", "--max-per-source must be >= 0"),
            ("--neighbor-expansion", "-1", "--neighbor-expansion must be >= 0"),
            ("--context-max-tokens", "-1", "--context-max-tokens must be >= 0"),
            ("--max-parents", "-1", "--max-parents must be >= 0"),
            ("--request-timeout-sec", "0", "--request-timeout-sec must be > 0"),
            ("--max-retries", "-1", "--max-retries must be >= 0"),
            ("--retry-base-delay-sec", "0", "--retry-base-delay-sec must be > 0"),
        ],
        ids=[
            "negative_top_k",
            "negative_max_per_source",
            "negative_neighbor_expansion",
            "negative_context_max_tokens",
            "negative_max_parents",
            "zero_timeout",
            "negative_max_retries",
            "zero_retry_delay",
        ],
    )
    def test_parse_args_with_invalid_numeric_values_raises_value_error(
        self, monkeypatch, invalid_flag, value, match_msg
    ):
        """Test that out-of-bounds numeric arguments raise ValueError."""
        monkeypatch.setattr(
            "sys.argv", ["rag_answer.py", "--query", "q", invalid_flag, value]
        )

        with pytest.raises(ValueError, match=match_msg):
            rag_answer.parse_args()


class TestMainExecution:
    """Tests for main() execution and output formatting."""

    @patch("src.query.rag_answer.run_rag_query")
    def test_main_calls_run_rag_query_and_outputs_json(
        self, mock_run_rag_query, monkeypatch, capsys
    ):
        """Test that main parses args, passes them to run_rag_query, and prints JSON."""
        # Arrange
        dummy_result = {"query": "What is RAG?", "answer": "RAG is retrieval augmented generation."}
        mock_run_rag_query.return_value = dummy_result

        monkeypatch.setattr(
            "sys.argv",
            [
                "rag_answer.py",
                "--query", "What is RAG?",
                "--mode", "recherche",
                "--top-k", "3",
            ],
        )

        # Act
        rag_answer.main()

        # Assert
        mock_run_rag_query.assert_called_once()
        call_kwargs = mock_run_rag_query.call_args.kwargs
        assert call_kwargs["query"] == "What is RAG?"
        assert call_kwargs["mode"] == "recherche"
        assert call_kwargs["top_k"] == 3

        # Capture printed stdout and verify it's valid JSON matching dummy_result
        captured = capsys.readouterr()
        output_json = json.loads(captured.out)
        assert output_json == dummy_result
