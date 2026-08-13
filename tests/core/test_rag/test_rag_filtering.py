"""Tests for filter parsing and where clause building in RAG module."""

import pytest
from src.core import rag


class TestFilterParsing:
    """Tests for parse_filters function."""
    
    @pytest.mark.parametrize(
        "filters, expected",
        [
            (["a=1", "b=2"], {"a": "1", "b": "2"}),
            (["a=1", "a=2"], {"a": ["1", "2"]}),
            (["  a  =  1  "], {"a": "1"}),
        ],
        ids=["valid_pairs", "duplicate_keys", "trimming"]
    )
    def test_parse_filters_with_valid_inputs_returns_parsed_dict(self, filters, expected):
        """Test parsing of valid filter inputs with various formats."""
        # Arrange & Act
        result = rag.parse_filters(filters)
        
        # Assert
        assert result == expected
    
    @pytest.mark.parametrize(
        "filters, error_pattern",
        [
            (["=value"], "missing key"),
            (["noeq"], "expected key=value"),
        ],
        ids=["missing_key", "no_equals"]
    )
    def test_parse_filters_with_invalid_format_raises_value_error(self, filters, error_pattern):
        """Test that invalid filter formats raise ValueError with correct message."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match=error_pattern):
            rag.parse_filters(filters)


class TestWhereClauseBuildingFromFilters:
    """Tests for build_where_clause function that converts filters to queries."""
    
    @pytest.mark.parametrize(
        "input_dict, expected",
        [
            ({"a": "1"}, {"a": "1"}),
            ({"a": "1", "b": ["x", "y"]}, {"a": "1", "b": {"$in": ["x", "y"]}}),
            ({}, None),
        ],
        ids=["simple_dict", "list_value_converts_to_in", "empty_dict"]
    )
    def test_build_where_clause_converts_filters_to_query(self, input_dict, expected):
        """Test building where clause from filter dict with various input types."""
        # Arrange & Act
        result = rag.build_where_clause(input_dict)
        
        # Assert
        assert result == expected
