"""Tests for Maximal Marginal Relevance (MMR) candidate selection."""

import pytest
from unittest.mock import MagicMock
from src.core import rag


class TestMaximalMarginalRelevance:
    """Tests for MMR algorithm that balances relevance and diversity."""
    
    def test_apply_mmr_with_single_candidate_returns_it(self, candidate_factory, fake_chroma_collection):
        """Test that single candidate is returned without comparison."""
        # Arrange
        cand = candidate_factory(chunk_id="c1")
        candidates = [cand]
        query_embedding = [0.5, 0.5]
        collection = fake_chroma_collection(
            store={"c1": {"embedding": [0.5, 0.5]}}
        )
        
        # Act
        result = rag.apply_mmr(
            candidates=candidates,
            query_embedding=query_embedding,
            top_k=1,
            collection=collection,
        )
        
        # Assert
        assert len(result) == 1
        assert result[0].chunk_id == "c1"
    
    def test_apply_mmr_with_multiple_candidates_selects_diverse_results(self, candidate_factory, fake_chroma_collection):
        """Test that MMR selects diverse candidates balancing relevance and similarity."""
        # Arrange
        cand1 = candidate_factory(chunk_id="c1")
        cand2 = candidate_factory(chunk_id="c2")
        cand3 = candidate_factory(chunk_id="c3")
        candidates = [cand1, cand2, cand3]
        query_embedding = [1.0, 0.0]

        # c1 is identical direction to query [1,0]
        # c2 is nearly identical to c1 (redundant)
        # c3 is orthogonal to c1 (diverse)
        collection = fake_chroma_collection(
            store={
                "c1": {"embedding": [1.0, 0.0]},
                "c2": {"embedding": [0.99, 0.01]},
                "c3": {"embedding": [0.0, 1.0]},
            }
        )
        
        # Act - using lambda_mult=0.2 heavily penalizes redundancy
        result = rag.apply_mmr(
            candidates=candidates,
            query_embedding=query_embedding,
            top_k=2,
            lambda_mult=0.2,
            collection=collection,
        )
        
        # Assert - verifies c1 (relevance) + c3 (diversity) selected over c2 (redundant)
        selected_ids = {cand.chunk_id for cand in result}
        assert selected_ids == {"c1", "c3"}
    
    def test_apply_mmr_with_empty_candidates_returns_empty_list(self, fake_chroma_collection):
        """Test that empty candidate list returns empty result."""
        # Arrange
        collection = fake_chroma_collection()
        
        # Act
        result = rag.apply_mmr(
            candidates=[],
            query_embedding=[0.5, 0.5],
            top_k=1,
            collection=collection,
        )
        
        # Assert
        assert result == []
    
    def test_apply_mmr_without_embeddings_raises_runtime_error(self, candidate_factory):
        """Test that missing embeddings in collection raises RuntimeError."""
        # Arrange
        cand = candidate_factory(chunk_id="c1")
        collection = MagicMock()
        collection.get.return_value = {"embeddings": []}
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="MMR embeddings unavailable"):
            rag.apply_mmr(
                candidates=[cand],
                query_embedding=[0.5, 0.5],
                top_k=1,
                collection=collection,
            )
