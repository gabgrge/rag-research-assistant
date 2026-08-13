"""Tests for retrieval operations: flattening results, filtering, ranking candidates."""

import pytest
from src.core import rag


class TestQueryResultsFlattening:
    """Tests for flattening ChromaDB query results into Candidate objects."""
    
    def test_flatten_query_results_with_valid_results_creates_candidates(self, valid_chroma_results):
        """Test flattening ChromaDB results with all fields populated."""
        # Arrange & Act
        cands = rag.flatten_query_results(valid_chroma_results)
        
        # Assert
        assert len(cands) == 2
        assert cands[0].chunk_id == "c1"
        assert cands[0].text == "doc1"
        assert cands[0].score == pytest.approx(-0.1)
        assert cands[1].score == pytest.approx(-0.5)
    
    def test_flatten_query_results_with_empty_results_returns_empty_list(self, empty_chroma_results):
        """Test that empty ChromaDB results return empty candidate list."""
        # Arrange & Act
        cands = rag.flatten_query_results(empty_chroma_results)
        
        # Assert
        assert cands == []
    
    def test_flatten_query_results_with_missing_fields_uses_defaults(self):
        """Test that missing optional fields are handled gracefully."""
        # Arrange
        results = {"ids": [["c1"]]}
        
        # Act
        cands = rag.flatten_query_results(results)
        
        # Assert
        assert len(cands) == 1
        assert cands[0].text == ""


class TestCandidateFiltering:
    """Tests for per-source candidate filtering and capping."""
    
    def test_apply_max_per_source_enforces_limit_per_source(self, sample_candidates_multi_source):
        """Test that max_per_source limits candidates from each source."""
        # Arrange & Act
        out = rag.apply_max_per_source(sample_candidates_multi_source, max_per_source=1, top_k=10)
        
        # Assert
        assert len(out) == 2
        sources = {c.meta.get("source_id") for c in out}
        assert sources == {"s1", "s2"}
    
    def test_apply_max_per_source_respects_top_k_limit(self, candidate_factory):
        """Test that top_k limit is applied even with available candidates."""
        # Arrange
        cands = [
            candidate_factory(chunk_id=f"c{i}", meta={"source_id": f"s{i % 2}"})
            for i in range(10)
        ]
        
        # Act
        out = rag.apply_max_per_source(cands, max_per_source=10, top_k=3)
        
        # Assert
        assert len(out) == 3
    
    def test_apply_soft_cap_per_source_maintains_source_diversity_in_primary(self, sample_candidates_multi_source):
        """Test that soft cap places one from each source in primary, rest in overflow."""
        # Arrange & Act
        out = rag.apply_soft_cap_per_source(sample_candidates_multi_source, max_per_source=1)
        
        # Assert
        assert len(out) == 3
        primary = out[:2]
        sources = {c.meta.get("source_id") for c in primary}
        assert sources == {"s1", "s2"}


class TestRedundancyCalculation:
    """Tests for redundancy ratio calculation on candidates."""

    @pytest.mark.parametrize(
        "source_ids, expected_ratio",
        [
            (["s1", "s1"], 1.0),
            (["s1", "s2"], 0.5),
        ],
        ids=["uniform_sources", "diverse_sources"]
    )
    def test_redundancy_ratio_calculates_expected_value(self, candidate_factory, source_ids, expected_ratio):
        """Test redundancy ratio calculation for various source distributions."""
        # Arrange
        cands = [candidate_factory(chunk_id=f"c{i}", meta={"source_id": sid}) for i, sid in enumerate(source_ids)]

        # Act
        ratio = rag.redundancy_ratio(cands)

        # Assert
        assert ratio == pytest.approx(expected_ratio)
    
    def test_redundancy_ratio_with_empty_list_returns_zero(self):
        """Test that empty candidate list yields zero redundancy."""
        # Arrange & Act
        ratio = rag.redundancy_ratio([])
        
        # Assert
        assert ratio == 0.0


class TestCosineSimilarityMetric:
    """Tests for cosine similarity calculation between vectors."""
    
    @pytest.mark.parametrize(
        "vec1, vec2, expected",
        [
            ([1, 0], [1, 0], 1.0),
            ([1, 0], [0, 1], 0.0),
            ([0, 0], [1, 1], 0.0),
        ],
        ids=["identical_vectors", "orthogonal_vectors", "zero_vector"]
    )
    def test_cosine_similarity_returns_expected_value(self, vec1, vec2, expected):
        """Test cosine similarity computation for various vector pairs."""
        # Arrange & Act
        result = rag.cosine_similarity(vec1, vec2)
        
        # Assert
        assert result == pytest.approx(expected)
    
    def test_cosine_similarity_with_mismatched_length_returns_zero(self):
        """Test that vectors of different lengths return 0 similarity."""
        # Arrange & Act
        result = rag.cosine_similarity([1], [1, 2])
        
        # Assert
        assert result == 0.0


class TestRankingAndScoring:
    """Tests for ranking candidates and assigning scores."""
    
    def test_apply_rank_scores_assigns_decreasing_scores(self, sample_candidates_multi_source):
        """Test that ranked candidates receive decreasing scores with first at 1.0."""
        # Arrange & Act
        ranked = rag.apply_rank_scores(sample_candidates_multi_source)
        
        # Assert
        scores = [c.score for c in ranked]
        assert scores[0] > scores[1] > scores[2]
        assert ranked[0].score == pytest.approx(1.0)


class TestCollectionFetching:
    """Tests for fetching candidates from ChromaDB collection by IDs."""

    @pytest.mark.parametrize(
        "requested_ids, expected_count",
        [
            (["x"], 1),
            (["x", "y"], 2),
        ],
        ids=["single_id", "multiple_ids"]
    )
    def test_fetch_by_ids_returns_requested_candidates(self, sample_chroma_collection, requested_ids, expected_count):
        """Test fetching candidates by IDs returns correct number of candidates."""
        # Arrange & Act
        out = rag.fetch_by_ids(sample_chroma_collection, requested_ids)

        # Assert
        assert len(out) == expected_count
        assert out["x"].text == "doc_x"
        assert out["x"].origin == "expanded"

    def test_fetch_by_ids_with_empty_ids_returns_empty_dict(self, fake_chroma_collection):
        """Test that empty ID list returns empty dict."""
        # Arrange
        coll = fake_chroma_collection({})
        
        # Act & Assert
        assert rag.fetch_by_ids(coll, []) == {}


class TestCandidateExpansion:
    """Tests for expanding candidates with neighbor and parent chunks."""
    
    def test_expand_candidates_without_expansion_returns_unchanged(self, candidate_factory, fake_chroma_collection):
        """Test that no expansion parameters return candidates unchanged."""
        # Arrange
        cand = candidate_factory(chunk_id="c1")
        coll = fake_chroma_collection({})
        
        # Act
        expanded = rag.expand_candidates(
            [cand], collection=coll, neighbor_expansion=0, include_parent=False, max_parents=0
        )
        
        # Assert
        assert len(expanded) == 1
    
    def test_expand_candidates_with_neighbor_expansion_adds_adjacent_chunks(self, candidate_factory, sample_chroma_collection):
        """Test that neighbor expansion includes previous and next chunks."""
        # Arrange
        sel = [candidate_factory(chunk_id="c1", meta={"prev_id": "x", "next_id": "y"}, score=1.0)]
        
        # Act
        expanded = rag.expand_candidates(
            sel, collection=sample_chroma_collection, neighbor_expansion=1, include_parent=False, max_parents=0
        )
        
        # Assert
        ids = {c.chunk_id for c in expanded}
        assert "x" in ids and "y" in ids
