"""Tests for OpenAI embedding generation, retry logic, Chroma collection interactions, and upsert operations in src/pipeline/index_chroma.py."""

from unittest.mock import MagicMock, patch
import pytest

from src.pipeline import index_chroma as index_mod


# ============================================================================
# 1. OpenAI Batch Embedding & Retry Tests
# ============================================================================

class TestEmbedBatch:
    """Tests for OpenAI API embedding generation, batch processing, and backoff retries."""

    @pytest.fixture
    def embed_kwargs(self, mock_openai_client, default_embedding_model):
        """Default kwargs bundle for embed_batch calls."""
        return {
            "openai_client": mock_openai_client,
            "embedding_model": default_embedding_model,
            "max_retries": 3,
            "retry_base_delay_sec": 0.01,
        }

    def test_embed_batch_returns_empty_list_for_empty_input(self, mock_openai_client, embed_kwargs):
        """Test embed_batch returns an empty list immediately when given no texts."""
        result = index_mod.embed_batch(texts=[], **embed_kwargs)

        assert result == []
        mock_openai_client.embeddings.create.assert_not_called()

    def test_embed_batch_returns_embeddings_on_success(
        self, mock_openai_client, default_embedding_model, embed_kwargs
    ):
        """Test embed_batch returns embedding vectors when client succeeds on first attempt."""
        texts = ["Sample text 1", "Sample text 2"]

        embeddings = index_mod.embed_batch(texts=texts, **embed_kwargs)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        mock_openai_client.embeddings.create.assert_called_once_with(
            model=default_embedding_model,
            input=texts,
        )

    def test_embed_batch_retries_on_retryable_error_and_succeeds(
        self, mock_openai_client, embed_kwargs
    ):
        """Test embed_batch retries when a retryable OpenAI error occurs and recovers."""
        success_response = MagicMock(data=[MagicMock(embedding=[0.2] * 1536)])
        mock_openai_client.embeddings.create.side_effect = [
            RuntimeError("Rate limit reached"),
            success_response,
        ]

        with patch("src.pipeline.index_chroma.classify_openai_error", return_value=(True, 0.01, "rate_limit")), \
             patch("time.sleep") as mock_sleep:
            embeddings = index_mod.embed_batch(texts=["Text to embed"], **embed_kwargs)

        assert len(embeddings) == 1
        assert mock_openai_client.embeddings.create.call_count == 2
        mock_sleep.assert_called_once_with(0.01)

    def test_embed_batch_raises_immediately_on_non_retryable_error(
        self, mock_openai_client, embed_kwargs
    ):
        """Test embed_batch raises error without retrying when classification is non-retryable."""
        mock_openai_client.embeddings.create.side_effect = ValueError("Invalid API Key")

        with patch("src.pipeline.index_chroma.classify_openai_error", return_value=(False, None, "auth_error")):
            with pytest.raises(ValueError, match="Invalid API Key"):
                index_mod.embed_batch(texts=["Text to embed"], **embed_kwargs)

        assert mock_openai_client.embeddings.create.call_count == 1


# ============================================================================
# 2. Chroma Collection Query Tests
# ============================================================================

class TestFetchExistingChunkIds:
    """Tests for fetching existing chunk IDs for a source_id from Chroma vector collection."""

    def test_fetch_existing_chunk_ids_returns_set_of_ids(self, mock_chroma_collection):
        """Test fetch_existing_chunk_ids queries collection and returns set of existing IDs."""
        source_id = "source_1"
        mock_chroma_collection.get.return_value = {"ids": ["chunk_a", "chunk_b"]}

        existing_ids = index_mod.fetch_existing_chunk_ids(
            collection=mock_chroma_collection,
            source_id=source_id,
        )

        assert existing_ids == {"chunk_a", "chunk_b"}
        mock_chroma_collection.get.assert_called_once_with(where={"source_id": source_id}, include=[])

    def test_fetch_existing_chunk_ids_handles_type_error_fallback(self, mock_chroma_collection):
        """Test fallback when Chroma collection.get doesn't accept the 'include' keyword argument."""
        source_id = "source_1"
        mock_chroma_collection.get.side_effect = [
            TypeError("unexpected keyword argument 'include'"),
            {"ids": ["fallback_chunk"]},
        ]

        existing_ids = index_mod.fetch_existing_chunk_ids(
            collection=mock_chroma_collection,
            source_id=source_id,
        )

        assert existing_ids == {"fallback_chunk"}
        assert mock_chroma_collection.get.call_count == 2


# ============================================================================
# 3. Source Chunks Upsert Tests
# ============================================================================

class TestUpsertSourceChunks:
    """Tests for differential upsert mechanics in upsert_source_chunks."""

    @pytest.fixture
    def default_upsert_kwargs(
        self, mock_chroma_collection, mock_openai_client, default_embedding_model
    ):
        """Base arguments passed into upsert_source_chunks."""
        return {
            "collection": mock_chroma_collection,
            "openai_client": mock_openai_client,
            "embedding_model": default_embedding_model,
            "batch_size": 64,
            "max_retries": 3,
            "retry_base_delay_sec": 0.01,
        }

    def _make_chunk(self, mock_leaf_record, source_id: str, suffix: str) -> dict:
        """Helper to quickly mint chunk record payloads."""
        return dict(mock_leaf_record, chunk_id=f"{source_id}_{suffix}")

    @pytest.mark.parametrize(
        "reembed_all, existing_ids, expected_newly_embedded, expected_unchanged, expect_update_called",
        [
            (False, ["src_test_c1"], 1, 1, True),   # Differential mode: c1 exists, c2 new
            (True, ["src_test_c1"], 1, 0, False),  # Re-embed all mode: forces embedding c1 again
        ],
        ids=["differential_upsert", "reembed_all_mode"],
    )
    def test_upsert_source_chunks_reembed_modes(
        self,
        make_registry_entry,
        mock_leaf_record,
        mock_chroma_collection,
        default_upsert_kwargs,
        reembed_all,
        existing_ids,
        expected_newly_embedded,
        expected_unchanged,
        expect_update_called,
    ):
        """Test differential vs re-embed all execution behavior."""
        source_id = "src_test"
        records = [self._make_chunk(mock_leaf_record, source_id, "c1")]
        if not reembed_all:
            records.append(self._make_chunk(mock_leaf_record, source_id, "c2"))

        row = make_registry_entry(source_id, leaf_records=records)
        mock_chroma_collection.get.return_value = {"ids": existing_ids}

        stats = index_mod.upsert_source_chunks(
            row=row,
            reembed_all=reembed_all,
            **default_upsert_kwargs,
        )

        assert stats.total_chunks == len(records)
        assert stats.newly_embedded == expected_newly_embedded
        assert stats.unchanged == expected_unchanged
        mock_chroma_collection.upsert.assert_called_once()
        assert mock_chroma_collection.update.called is expect_update_called

    def test_upsert_source_chunks_deletes_stale_chunks(
        self,
        make_registry_entry,
        mock_leaf_record,
        mock_chroma_collection,
        default_upsert_kwargs,
    ):
        """Test stale chunk IDs present in vector store but absent from leaf records are deleted."""
        source_id = "src_stale"
        rec1 = self._make_chunk(mock_leaf_record, source_id, "c1")
        row = make_registry_entry(source_id, leaf_records=[rec1])

        # c1 is current, c_stale is obsolete in vector store
        mock_chroma_collection.get.return_value = {"ids": [f"{source_id}_c1", f"{source_id}_c_stale"]}

        stats = index_mod.upsert_source_chunks(
            row=row,
            reembed_all=False,
            **default_upsert_kwargs,
        )

        assert stats.stale_deleted == 1
        mock_chroma_collection.delete.assert_called_once_with(ids=[f"{source_id}_c_stale"])
