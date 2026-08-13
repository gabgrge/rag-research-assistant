"""Tests for context block building and final output generation."""

import pytest
from src.core import rag


@pytest.fixture
def default_final_output_config():
    """Fixture providing default final output configuration."""
    return {
        "query": "test query",
        "mode": "recherche",
        "model": "gpt-5-mini",
    }


@pytest.fixture
def final_output_caller(default_final_output_config):
    """Fixture to call build_final_output with default parameters and allow overrides."""
    def _call(used_blocks, block_labels, model_json, **override_kwargs):
        kwargs = {**default_final_output_config, **override_kwargs}
        return rag.build_final_output(
            used_blocks=used_blocks,
            block_labels=block_labels,
            model_json=model_json,
            **kwargs
        )
    return _call


class TestContextBlockBuilding:
    """Tests for building context blocks from candidates."""
    
    def test_build_context_blocks_with_single_candidate_creates_block(self, candidate_factory):
        """Test that single candidate creates one context block with proper citation label."""
        # Arrange
        cand = candidate_factory(
            chunk_id="a",
            text="hello",
            meta={"source_id": "s", "unit_type_start": "page", "unit_index_start": 1}
        )
        
        # Act
        blocks, used, used_tokens, budget, stopped = rag.build_context_blocks([cand], context_max_tokens=1000)
        
        # Assert
        assert len(blocks) == 1
        assert len(used) == 1
        assert "[C1]" in blocks[0]
    
    def test_build_context_blocks_with_budget_exceeded_stops_early(self, candidate_factory):
        """Test that context blocks stop when token budget is exceeded."""
        # Arrange
        cand = candidate_factory(chunk_id="a", text="x" * 100)
        
        # Act
        blocks, used, used_tokens, budget, stopped = rag.build_context_blocks([cand], context_max_tokens=5)
        
        # Assert
        assert stopped
        assert len(blocks) == 1


class TestFinalOutputBuilding:
    """Tests for building final output with citations and metadata."""
    
    def test_build_final_output_with_citations_includes_citation_data(self, candidate_factory, final_output_caller):
        """Test that final output includes citation metadata with labels."""
        # Arrange
        cand = candidate_factory(chunk_id="c1", meta={"source_id": "s1", "filename": "f.txt"})
        used_blocks = [cand]
        block_labels = {"C1": cand}
        model_json = {"answer": "Answer with citation [C1]", "citation_ids": ["C1"]}
        
        # Act
        out = final_output_caller(used_blocks=used_blocks, block_labels=block_labels, model_json=model_json)
        
        # Assert
        assert out["query"] == "test query"
        assert out["mode"] == "recherche"
        assert len(out["citations"]) == 1
        assert out["citations"][0]["label"] == "C1"
    
    def test_build_final_output_with_no_sources_returns_no_source_message(self, final_output_caller):
        """Test that output with no sources uses the default NO_SOURCE_MESSAGE."""
        # Arrange
        model_json = {"answer": "UNKNOWN", "citation_ids": []}
        
        # Act
        out = final_output_caller(used_blocks=[], block_labels={}, model_json=model_json)
        
        # Assert
        assert out["answer"] == rag.NO_SOURCE_MESSAGE
        assert out["citations"] == []


class TestCandidateExpansionFields:
    """Tests for extracting expansion metadata from candidates."""

    @pytest.mark.parametrize(
        "candidate_kwargs, label_by_id, expected_expanded_from, expected_depth",
        [
            (
                {"origin": "seed"},
                {},
                None,
                None,
            ),
            (
                {"origin": "expanded", "expanded_from": None, "expanded_depth": 2},
                {},
                None,
                2,
            ),
            (
                {"origin": "expanded", "expanded_from": "parent_id", "expanded_depth": 1},
                {"parent_id": "C1"},
                "C1",
                1,
            ),
        ],
        ids=["seed_candidate", "expanded_no_parent_label", "expanded_with_parent_label"],
    )
    def test_expansion_fields_returns_expected_metadata(
            self,
            candidate_factory,
            candidate_kwargs,
            label_by_id,
            expected_expanded_from,
            expected_depth,
    ):
        # Arrange
        cand = candidate_factory(**candidate_kwargs)

        # Act
        expanded_from, depth = rag.expansion_fields(cand, label_by_id)

        # Assert
        assert expanded_from == expected_expanded_from
        assert depth == expected_depth


class TestCandidateOriginFieldHandling:
    """Tests for adding origin-related fields to output based on candidate origin."""
    
    def test_add_origin_fields_with_seed_candidate_adds_scores(self, candidate_factory):
        """Test that seed candidates include retrieval scores in output."""
        # Arrange
        cand = candidate_factory(
            origin="seed",
            score=0.95,
            retrieval_score=0.85,
            distance=0.15,
        )
        payload = {}
        label_by_id = {}
        
        # Act
        rag.add_origin_fields(payload, cand, label_by_id)
        
        # Assert
        assert payload["score"] == 0.95
        assert payload["retrieval_score"] == 0.85
        assert payload["retrieval_distance"] == 0.15
    
    def test_add_origin_fields_with_expanded_candidate_adds_parent_info(self, candidate_factory):
        """Test that expanded candidates include parent label and expansion depth."""
        # Arrange
        cand = candidate_factory(
            origin="expanded",
            expanded_from="parent_id",
            expanded_depth=1,
        )
        payload = {}
        label_by_id = {"parent_id": "C1"}
        
        # Act
        rag.add_origin_fields(payload, cand, label_by_id)
        
        # Assert
        assert payload["expanded_from_label"] == "C1"
        assert payload["expanded_depth"] == 1
