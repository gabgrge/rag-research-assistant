"""Tests for sliding window creation, unit boundary alignments, and parent-leaf link generation in src/pipeline/chunk_hierarchical.py."""

from src.pipeline import chunk_hierarchical as chunk_mod


# ============================================================================
# 1. Window Calculation & Unit Boundary Tests
# ============================================================================

class TestWindowCalculation:
    """Tests for token boundary calculations and sliding window construction."""

    def test_unit_boundary_token_positions_finds_correct_token_indexes(self):
        """Test that character offsets of unit boundaries map accurately to token index positions."""
        # Arrange
        # Text: "Word1 Word2 Word3 Word4"
        token_spans = [(0, 5), (6, 11), (12, 17), (18, 23)]
        units = [
            {"end_char": 11},  # Boundary after "Word2" (token 2)
            {"end_char": 23},  # Boundary after "Word4" (token 4 - end of text)
        ]

        # Act
        boundaries = chunk_mod.unit_boundary_token_positions(token_spans, units)

        # Assert
        assert boundaries == [2]

    def test_choose_window_end_prefers_unit_boundary_within_target_range(self, sample_leaf_config):
        """Test window end snaps to nearest unit boundary inside [min_tokens, max_tokens]."""
        # Arrange
        boundary_positions = [18]  # Boundary near target_tokens=20
        start_token = 0
        total_tokens = 50

        # Act
        end = chunk_mod.choose_window_end(start_token, total_tokens, boundary_positions, sample_leaf_config)

        # Assert
        assert end == 18

    def test_choose_window_end_falls_back_to_target_tokens_when_no_boundary_in_range(self, sample_leaf_config):
        """Test window end falls back to target_tokens when no unit boundary exists in range."""
        # Arrange
        boundary_positions = [5, 45]  # Out of range boundaries (min=10, max=30)
        start_token = 0
        total_tokens = 50

        # Act
        end = chunk_mod.choose_window_end(start_token, total_tokens, boundary_positions, sample_leaf_config)

        # Assert
        assert end == sample_leaf_config.target_tokens

    def test_build_token_windows_handles_short_documents(self, sample_leaf_config):
        """Test build_token_windows creates a single window when total tokens <= max_tokens."""
        # Arrange
        token_spans = [(i * 5, (i + 1) * 5) for i in range(15)]  # 15 tokens
        boundary_positions = []

        # Act
        windows = chunk_mod.build_token_windows(token_spans, boundary_positions, sample_leaf_config)

        # Assert
        assert len(windows) == 1
        assert windows[0] == (0, 15)

    def test_build_token_windows_applies_overlap_step(self, sample_leaf_config):
        """Test build_token_windows advances window start by (end - overlap_tokens)."""
        # Arrange
        token_spans = [(i * 5, (i + 1) * 5) for i in range(50)]  # 50 tokens
        boundary_positions = []

        # Act
        windows = chunk_mod.build_token_windows(token_spans, boundary_positions, sample_leaf_config)

        # Assert
        assert len(windows) > 1
        first_start, first_end = windows[0]
        second_start, _ = windows[1]
        assert second_start == first_end - sample_leaf_config.overlap_tokens


# ============================================================================
# 2. Chunk Record Assembly & Linking Tests
# ============================================================================

class TestChunkRecordAssembly:
    """Tests for record building, neighbor link attachments, and parent-leaf assignments."""

    def test_build_chunk_records_creates_valid_schema(self, mock_extracted_payload):
        """Test build_chunk_records produces dictionaries with expected metadata and fields."""
        # Arrange
        text = mock_extracted_payload["text"]
        units = mock_extracted_payload["units"]
        spans, _ = chunk_mod.tokenize_with_spans(text, mode="whitespace", encoding_name="cl100k_base")
        windows = [(0, 10)]

        # Act
        records = chunk_mod.build_chunk_records(
            source_id="src_1",
            text=text,
            units=units,
            token_spans=spans,
            windows=windows,
            level="leaf",
            meta={"domain": "test"},
        )

        # Assert
        assert len(records) == 1
        rec = records[0]
        assert rec["source_id"] == "src_1"
        assert rec["level"] == "leaf"
        assert rec["token_count"] == 10
        assert rec["unit_type_start"] == "paragraph"
        assert rec["meta"]["domain"] == "test"

    def test_attach_neighbor_links_sets_prev_and_next_ids(self):
        """Test attach_neighbor_links correctly wires sequential chunk IDs."""
        # Arrange
        records = [
            {"chunk_id": "c1"},
            {"chunk_id": "c2"},
            {"chunk_id": "c3"},
        ]

        # Act
        chunk_mod.attach_neighbor_links(records)

        # Assert
        assert records[0]["prev_id"] == ""
        assert records[0]["next_id"] == "c2"
        assert records[1]["prev_id"] == "c1"
        assert records[1]["next_id"] == "c3"
        assert records[2]["prev_id"] == "c2"
        assert records[2]["next_id"] == ""

    def test_filter_short_leaf_chunks_retains_longest_if_all_below_min(self):
        """Test filter_short_leaf_chunks retains at least one chunk if all chunks fall below min_output_tokens."""
        # Arrange
        records = [
            {"token_count": 5},
            {"token_count": 12},
            {"token_count": 8},
        ]

        # Act
        filtered = chunk_mod.filter_short_leaf_chunks(records, min_output_tokens=20)

        # Assert
        assert len(filtered) == 1
        assert filtered[0]["token_count"] == 12

    def test_assign_parent_links_maps_leaf_to_max_overlapping_parent(self):
        """Test assign_parent_links links leaf chunk to the parent chunk with highest character overlap."""
        # Arrange
        leaf_records = [
            {"start_char": 10, "end_char": 50},
        ]
        parent_records = [
            {"chunk_id": "parent_1", "start_char": 0, "end_char": 30},
            {"chunk_id": "parent_2", "start_char": 25, "end_char": 100},  # Highest overlap (25 to 50 = 25 chars)
        ]

        # Act
        chunk_mod.assign_parent_links(leaf_records, parent_records)

        # Assert
        assert leaf_records[0]["parent_id"] == "parent_2"


# ============================================================================
# 3. Full Payload Assembly Tests
# ============================================================================

class TestBuildChunksForPayload:
    """Tests for complete payload chunk building pipeline."""

    def test_build_chunks_for_payload_returns_leaf_and_parent_records(
        self, mock_extracted_payload, sample_leaf_config, sample_parent_config
    ):
        """Test build_chunks_for_payload returns structured leaf and parent chunks with parent_id linkage."""
        # Arrange
        source_id = "test_source"

        # Act
        leaf_records, parent_records = chunk_mod.build_chunks_for_payload(
            source_id=source_id,
            payload=mock_extracted_payload,
            leaf_config=sample_leaf_config,
            parent_config=sample_parent_config,
            min_leaf_output_tokens=0,
            tokenizer_mode="whitespace",
            tokenizer_encoding="cl100k_base",
        )

        # Assert
        assert len(leaf_records) > 0
        assert len(parent_records) > 0
        assert "parent_id" in leaf_records[0]
        assert leaf_records[0]["parent_id"] != ""

    def test_build_chunks_for_payload_empty_text_returns_empty_lists(
        self, sample_leaf_config, sample_parent_config
    ):
        """Test build_chunks_for_payload gracefully returns empty lists for empty text payload."""
        # Arrange
        payload = {"text": "   ", "units": []}

        # Act
        leaf_records, parent_records = chunk_mod.build_chunks_for_payload(
            source_id="empty_source",
            payload=payload,
            leaf_config=sample_leaf_config,
            parent_config=sample_parent_config,
            min_leaf_output_tokens=0,
            tokenizer_mode="whitespace",
            tokenizer_encoding="cl100k_base",
        )

        # Assert
        assert leaf_records == []
        assert parent_records == []
