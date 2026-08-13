"""RAG module-specific test fixtures."""

import pytest
from typing import Dict, List, Any, Optional
from types import SimpleNamespace
from unittest.mock import MagicMock
from src.core.rag import Candidate


# ========================
# Factory Fixtures
# ========================

@pytest.fixture
def candidate_factory():
    """Factory fixture for creating Candidate instances with customizable defaults."""
    def _factory(
        chunk_id: str = "cid1",
        text: str = "sample text",
        meta: Optional[Dict[str, Any]] = None,
        distance: Optional[float] = 0.0,
        score: float = 0.0,
        origin: str = "seed",
        retrieval_score: Optional[float] = 0.0,
        expanded_from: Optional[str] = None,
        expanded_depth: Optional[int] = None,
    ) -> Candidate:
        return Candidate(
            chunk_id=str(chunk_id),
            text=str(text),
            meta=meta or {},
            distance=distance,
            score=score,
            origin=origin,
            retrieval_score=retrieval_score,
            expanded_from=expanded_from,
            expanded_depth=expanded_depth,
        )
    return _factory


@pytest.fixture
def fake_chroma_collection():
    """Factory fixture for creating mock ChromaCollection instances."""
    def _factory(store: Optional[Dict[str, Dict[str, Any]]] = None) -> MagicMock:
        store = store or {}
        mock_collection = MagicMock()
        
        def mock_get(ids=None, include=None):
            ids = list(ids or [])
            if not include:
                include = []
            
            result = {"ids": ids}
            
            if "embeddings" in include:
                result["embeddings"] = [
                    store.get(cid, {}).get("embedding", []) for cid in ids
                ]
            if "documents" in include:
                result["documents"] = [
                    store.get(cid, {}).get("document", "") for cid in ids
                ]
            if "metadatas" in include:
                result["metadatas"] = [
                    store.get(cid, {}).get("meta", {}) for cid in ids
                ]
            
            return result
        
        mock_collection.get = mock_get
        return mock_collection
    
    return _factory


@pytest.fixture
def make_embedding_response():
    """Factory fixture to easily craft OpenAI embedding response structures."""
    def _make(embedding=None, include_data=True, include_embedding=True):
        if not include_data:
            return SimpleNamespace()

        if not include_embedding:
            return SimpleNamespace(data=[SimpleNamespace()])

        embedding_obj = SimpleNamespace(embedding=embedding or [0.1, 0.2, 0.3])
        return SimpleNamespace(data=[embedding_obj])

    return _make


# ========================
# Mock Fixtures
# ========================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing LLM operations."""
    client = MagicMock()
    client.embeddings = MagicMock()
    client.responses = MagicMock()
    return client


# ========================
# Sample Data Fixtures
# ========================

@pytest.fixture
def sample_candidates_multi_source(candidate_factory) -> List[Candidate]:
    """Provides a ready-to-use list of 3 candidates across 2 sources."""
    return [
        candidate_factory(chunk_id="a", meta={"source_id": "s1"}),
        candidate_factory(chunk_id="b", meta={"source_id": "s1"}),
        candidate_factory(chunk_id="c", meta={"source_id": "s2"}),
    ]


@pytest.fixture
def sample_chroma_collection(fake_chroma_collection):
    """Provides a ready-to-use mock ChromaCollection with predefined documents and metadata."""
    default_store = {
        "x": {"document": "doc_x", "meta": {"source_id": "s1"}},
        "y": {"document": "doc_y", "meta": {"source_id": "s2"}},
    }
    return fake_chroma_collection(default_store)


@pytest.fixture
def valid_chroma_results() -> Dict[str, List[List[Any]]]:
    """Provides a standard populated ChromaDB query result payload."""
    return {
        "ids": [["c1", "c2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"source_id": "s1"}, {"source_id": "s2"}]],
        "distances": [[0.1, 0.5]],
    }


@pytest.fixture
def empty_chroma_results() -> Dict[str, List[List[Any]]]:
    """Provides an empty ChromaDB query result payload structure."""
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
