"""Tests for low-level configuration hashing, record loading, record flattening, and ID normalization in src/pipeline/index_chroma.py."""

import json
import pytest

from src.pipeline import index_chroma as index_mod


# ============================================================================
# 1. Config Hashing Tests
# ============================================================================

class TestComputeIndexConfigHash:
    """Tests for deterministic SHA256 index configuration hash generation."""

    def test_compute_index_config_hash_returns_deterministic_hex(
        self, default_collection_name, default_embedding_model
    ):
        """Test hash generation produces identical 64-character hex string for identical configurations."""
        hash1 = index_mod.compute_index_config_hash(
            collection_name=default_collection_name,
            embedding_model=default_embedding_model,
        )
        hash2 = index_mod.compute_index_config_hash(
            collection_name=default_collection_name,
            embedding_model=default_embedding_model,
        )

        assert len(hash1) == 64
        assert hash1 == hash2

    def test_compute_index_config_hash_changes_on_parameter_mutation(
        self, default_collection_name, default_embedding_model
    ):
        """Test that modifying any input parameter alters the generated hash."""
        base_hash = index_mod.compute_index_config_hash(
            collection_name=default_collection_name,
            embedding_model=default_embedding_model,
        )

        mutated_hash = index_mod.compute_index_config_hash(
            collection_name=default_collection_name,
            embedding_model="text-embedding-3-large",
        )

        assert base_hash != mutated_hash


# ============================================================================
# 2. Leaf Record Loading Tests
# ============================================================================

class TestLoadLeafRecords:
    """Tests for JSONL file parsing and validation of leaf records."""

    def test_load_leaf_records_parses_valid_jsonl_file(self, isolate_index_environment, mock_leaf_record):
        """Test loading valid JSONL records returns a list of dictionaries."""
        leaf_file = isolate_index_environment["leaf_dir"] / "valid_source.jsonl"
        leaf_file.write_text(
            f"{json.dumps(mock_leaf_record)}\n\n{json.dumps(mock_leaf_record)}\n",
            encoding="utf-8",
        )

        records = index_mod.load_leaf_records(leaf_file)

        assert len(records) == 2
        assert records[0]["chunk_id"] == mock_leaf_record["chunk_id"]

    def test_load_leaf_records_raises_file_not_found_error_when_missing(self, tmp_path):
        """Test attempting to load a non-existent file raises FileNotFoundError."""
        missing_file = tmp_path / "missing.jsonl"

        with pytest.raises(FileNotFoundError, match="Leaf chunk file not found"):
            index_mod.load_leaf_records(missing_file)

    def test_load_leaf_records_raises_value_error_on_invalid_json(self, isolate_index_environment):
        """Test encountering invalid JSON within a file raises a ValueError."""
        corrupt_file = isolate_index_environment["leaf_dir"] / "corrupt_source.jsonl"
        corrupt_file.write_text("{\"valid\": True}\n{this is bad json}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSONL at"):
            index_mod.load_leaf_records(corrupt_file)


# ============================================================================
# 3. Chunk Record Flattening Tests
# ============================================================================

class TestFlattenChunkRecord:
    """Tests for flattening nested chunk records into Chroma metadata structures."""

    def test_flatten_chunk_record_maps_all_fields_correctly(self, mock_leaf_record, default_embedding_model):
        """Test flattening extracts ID, text, and populates expected Chroma metadata dictionary."""
        source_id = "test_source_id"

        chunk_id, chunk_text, metadata = index_mod.flatten_chunk_record(
            record=mock_leaf_record,
            source_id=source_id,
            embedding_model=default_embedding_model,
        )

        assert chunk_id == mock_leaf_record["chunk_id"]
        assert chunk_text == mock_leaf_record["text"]
        assert metadata["source_id"] == source_id
        assert metadata["chunk_id"] == mock_leaf_record["chunk_id"]
        assert metadata["level"] == "leaf"
        assert metadata["token_count"] == 10
        assert metadata["embedding_model"] == default_embedding_model
        assert metadata["filename"] == "document.pdf"

    @pytest.mark.parametrize(
        "mutation, match_pattern",
        [
            ({"chunk_id": ""}, "chunk_id missing in leaf record"),
            ({"text": "   "}, "chunk text empty for chunk_id="),
        ],
        ids=["missing_chunk_id", "empty_text_content"],
    )
    def test_flatten_chunk_record_raises_value_error_on_invalid_record(
        self, mock_leaf_record, default_embedding_model, mutation, match_pattern
    ):
        """Test missing chunk_id or blank text throws ValueError with matching message."""
        record = dict(mock_leaf_record, **mutation)

        with pytest.raises(ValueError, match=match_pattern):
            index_mod.flatten_chunk_record(
                record=record,
                source_id="src_1",
                embedding_model=default_embedding_model,
            )


# ============================================================================
# 4. Collection ID Normalization Tests
# ============================================================================

class TestNormalizeCollectionIds:
    """Tests for normalizing raw Chroma ID collection outputs into a flat list of strings."""

    @pytest.mark.parametrize(
        "raw_ids, expected_normalized",
        [
            (["id_1", "", "id_2", "id_3"], ["id_1", "id_2", "id_3"]),
            ([["id_1", "id_2"], ["id_3", ""], "id_4"], ["id_1", "id_2", "id_3", "id_4"]),
            (None, []),
        ],
        ids=["flat_list_with_blanks", "nested_lists", "invalid_none_input"],
    )
    def test_normalize_collection_ids_parses_various_input_shapes(
        self, raw_ids, expected_normalized
    ):
        """Test normalization flattens lists, strips empty entries, and handles non-list inputs."""
        normalized = index_mod.normalize_collection_ids(raw_ids)

        assert normalized == expected_normalized
