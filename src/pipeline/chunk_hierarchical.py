from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.common_utils import to_int, utc_now_iso
from src.utils.fs_utils import path_is_under
from src.utils.logging_utils import build_script_logger
from src.utils.paths import (
    CHUNKS_DIR,
    EXTRACTED_DIR,
    LEAF_CHUNKS_DIR,
    LOGS_DIR,
    PARENT_CHUNKS_DIR,
    REGISTRY_DIR,
)
from src.core.registry import (
    CHUNK_CHUNKED,
    CHUNK_FAILED,
    EXTRACTION_EXTRACTED,
    Row,
    clear_index_tracking,
    clear_chunk_tracking,
    load_registry_rows,
    set_index_pending,
    write_registry_rows,
)

load_dotenv()

REGISTRY_PATH = (REGISTRY_DIR / "sources.csv").resolve()
EXTRACTED_DIR = EXTRACTED_DIR.resolve()
CHUNKS_ROOT = CHUNKS_DIR.resolve()
LEAF_DIR = LEAF_CHUNKS_DIR.resolve()
PARENT_DIR = PARENT_CHUNKS_DIR.resolve()
LOG_PATH = (LOGS_DIR / "chunk_hierarchical.log").resolve()

LEAF_DIR.mkdir(parents=True, exist_ok=True)
PARENT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"\S+")

LOGGER = build_script_logger("chunk_hierarchical", LOG_PATH)


@dataclass(frozen=True)
class WindowConfig:
    level: str
    min_tokens: int
    target_tokens: int
    max_tokens: int
    overlap_tokens: int

    def as_dict(self) -> Dict[str, int | str]:
        return {
            "level": self.level,
            "min_tokens": self.min_tokens,
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
        }


def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build hierarchical chunks from extracted JSON files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild chunks even if chunk_status is CHUNKED and config is unchanged.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Restrict processing to one or more source_id values.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of sources processed (0 = no limit).",
    )
    parser.add_argument(
        "--keep-stale-chunks",
        action="store_true",
        help="Do not delete chunk files for non-EXTRACTED or orphan sources.",
    )
    parser.add_argument("--leaf-min", type=int, default=220)
    parser.add_argument("--leaf-target", type=int, default=360)
    parser.add_argument("--leaf-max", type=int, default=520)
    parser.add_argument("--leaf-overlap", type=int, default=70)
    parser.add_argument("--parent-min", type=int, default=700)
    parser.add_argument("--parent-target", type=int, default=1100)
    parser.add_argument("--parent-max", type=int, default=1500)
    parser.add_argument("--parent-overlap", type=int, default=180)
    parser.add_argument(
        "--min-leaf-output-tokens",
        type=int,
        default=0,
        help="Drop very short leaf chunks below this threshold (0 disables filtering).",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("auto", "whitespace", "tiktoken"),
        default="tiktoken",
        help="Tokenizer used for windowing (default: tiktoken).",
    )
    parser.add_argument(
        "--tiktoken-encoding",
        type=str,
        default="cl100k_base",
        help="Encoding name for tiktoken (used when tokenizer is auto or tiktoken).",
    )
    return parser.parse_args()


def validate_window_config(config: WindowConfig) -> None:
    if config.min_tokens <= 0:
        raise ValueError(f"{config.level}: min_tokens must be > 0")
    if config.target_tokens < config.min_tokens:
        raise ValueError(f"{config.level}: target_tokens must be >= min_tokens")
    if config.max_tokens < config.target_tokens:
        raise ValueError(f"{config.level}: max_tokens must be >= target_tokens")
    if config.overlap_tokens < 0:
        raise ValueError(f"{config.level}: overlap_tokens must be >= 0")
    if config.overlap_tokens >= config.target_tokens:
        raise ValueError(f"{config.level}: overlap_tokens must be < target_tokens")


def compute_config_hash(
    leaf: WindowConfig,
    parent: WindowConfig,
    tokenizer_mode: str,
    tokenizer_encoding: str,
    min_leaf_output_tokens: int,
) -> str:
    payload = {
        "leaf": leaf.as_dict(),
        "parent": parent.as_dict(),
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_encoding": tokenizer_encoding,
        "min_leaf_output_tokens": min_leaf_output_tokens,
    }
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_registry() -> List[Row]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    return load_registry_rows(REGISTRY_PATH)


def write_registry(rows: List[Row]) -> None:
    write_registry_rows(REGISTRY_PATH, rows, logger=LOGGER)


def extracted_path_for_row(row: Row) -> Path:
    existing = row.get("extracted_path", "").strip()
    if existing:
        return Path(existing).resolve()
    return (EXTRACTED_DIR / f"{row['source_id']}.json").resolve()


def leaf_path(source_id: str) -> Path:
    return LEAF_DIR / f"{source_id}.jsonl"


def parent_path(source_id: str) -> Path:
    return PARENT_DIR / f"{source_id}.jsonl"


def remove_chunk_files_for_source(source_id: str) -> int:
    removed = 0
    for candidate in (leaf_path(source_id), parent_path(source_id)):
        resolved = candidate.resolve()
        if not path_is_under(CHUNKS_ROOT, resolved):
            continue
        if resolved.exists():
            resolved.unlink()
            removed += 1
    return removed


def purge_stale_chunks(rows: List[Row], keep_stale: bool) -> int:
    if keep_stale:
        return 0

    removed = 0
    all_source_ids = {row.get("source_id", "") for row in rows if row.get("source_id", "")}

    for row in rows:
        if row.get("extraction_status") == EXTRACTION_EXTRACTED:
            continue
        source_id = row.get("source_id", "")
        if not source_id:
            continue
        removed += remove_chunk_files_for_source(source_id)
        clear_chunk_tracking(row)

    for directory in (LEAF_DIR, PARENT_DIR):
        for path in directory.glob("*.jsonl"):
            if path.stem in all_source_ids:
                continue
            if path.exists():
                path.unlink()
                removed += 1

    return removed


def should_process_row(row: Row, config_hash: str, force: bool, source_filter: set[str]) -> bool:
    if row.get("extraction_status") != EXTRACTION_EXTRACTED:
        return False
    if source_filter and row.get("source_id", "") not in source_filter:
        return False

    source_id = row.get("source_id", "")
    if not source_id:
        return False

    if force:
        return True

    row_chunk_status = row.get("chunk_status", "")
    row_config_hash = row.get("chunk_config_hash", "")
    leaf_exists = leaf_path(source_id).exists()
    parent_exists = parent_path(source_id).exists()

    if row_chunk_status != CHUNK_CHUNKED:
        return True
    if row_config_hash != config_hash:
        return True
    if not leaf_exists or not parent_exists:
        return True
    return False


def byte_to_char_index(byte_offsets: Sequence[int], byte_pos: int) -> int:
    return bisect.bisect_right(byte_offsets, byte_pos) - 1


def tokenize_with_tiktoken_spans(text: str, encoding_name: str) -> Optional[List[Tuple[int, int]]]:
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Invalid tiktoken encoding '%s': %s", encoding_name, exc)
        return None

    token_ids = encoding.encode(text, disallowed_special=())
    if not token_ids:
        return []

    text_bytes = text.encode("utf-8")
    byte_offsets: List[int] = [0]
    for char in text:
        byte_offsets.append(byte_offsets[-1] + len(char.encode("utf-8")))

    spans: List[Tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        token_bytes = encoding.decode_single_token_bytes(token_id)
        start_byte = cursor
        cursor += len(token_bytes)
        end_byte = cursor

        if end_byte > len(text_bytes):
            LOGGER.warning("tiktoken byte cursor out of bounds; fallback to whitespace tokenizer.")
            return None

        start_char = byte_to_char_index(byte_offsets, start_byte)
        end_char = byte_to_char_index(byte_offsets, end_byte)
        if end_char > start_char:
            spans.append((start_char, end_char))

    if cursor != len(text_bytes):
        LOGGER.warning("tiktoken byte cursor mismatch; fallback to whitespace tokenizer.")
        return None

    return spans


def tokenize_with_spans(text: str, mode: str, encoding_name: str) -> Tuple[List[Tuple[int, int]], str]:
    if mode not in {"auto", "whitespace", "tiktoken"}:
        raise ValueError(f"Unsupported tokenizer mode: {mode}")

    if mode in {"auto", "tiktoken"}:
        spans = tokenize_with_tiktoken_spans(text, encoding_name)
        if spans is not None:
            return spans, "tiktoken"
        if mode == "tiktoken":
            raise RuntimeError("tiktoken tokenizer requested but unavailable.")

    return [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)], "whitespace"


def fallback_document_unit(text: str) -> List[Dict[str, object]]:
    return [{
        "unit_type": "document",
        "unit_index": 1,
        "text": text,
        "start_char": 0,
        "end_char": len(text),
    }]


def spans_are_ordered_non_overlapping(units: List[Dict[str, object]]) -> bool:
    previous_end = 0
    for unit in units:
        start_char = to_int(unit.get("start_char"), 0)
        end_char = to_int(unit.get("end_char"), start_char)
        if start_char < previous_end:
            return False
        if end_char < start_char:
            return False
        previous_end = end_char
    return True


def ensure_units(payload: Dict[str, object], text: str) -> List[Dict[str, object]]:
    units_raw = payload.get("units")
    if not isinstance(units_raw, list):
        return fallback_document_unit(text)

    units: List[Dict[str, object]] = []
    for idx, raw in enumerate(units_raw, start=1):
        if not isinstance(raw, dict):
            continue

        try:
            start_char = int(raw.get("start_char", 0))
            end_char = int(raw.get("end_char", 0))
        except (TypeError, ValueError):
            continue

        if start_char < 0:
            start_char = 0
        if end_char < start_char:
            end_char = start_char
        if end_char > len(text):
            end_char = len(text)
        if start_char > len(text):
            start_char = len(text)

        # Units are enforced as spans over the global extracted text.
        unit_text = text[start_char:end_char]
        if not unit_text.strip():
            continue

        units.append({
            "unit_type": str(raw.get("unit_type", "unit")),
            "unit_index": int(raw.get("unit_index", idx)),
            "text": unit_text,
            "start_char": start_char,
            "end_char": end_char,
        })

    if not units:
        return fallback_document_unit(text)

    ordered = sorted(units, key=lambda u: (int(u["start_char"]), int(u["end_char"])))
    if not spans_are_ordered_non_overlapping(ordered):
        LOGGER.warning("Invalid unit spans detected; fallback to document-level unit.")
        return fallback_document_unit(text)
    return ordered


def unit_boundary_token_positions(token_spans: List[Tuple[int, int]], units: List[Dict[str, object]]) -> List[int]:
    if not token_spans:
        return []

    token_end_chars = [end for _, end in token_spans]
    boundaries: set[int] = set()
    for unit in units:
        end_char = to_int(unit.get("end_char"), 0)
        token_pos = bisect.bisect_right(token_end_chars, end_char)
        if 0 < token_pos < len(token_spans):
            boundaries.add(token_pos)
    return sorted(boundaries)


def choose_window_end(
    start_token: int,
    total_tokens: int,
    boundary_positions: Sequence[int],
    config: WindowConfig,
) -> int:
    min_end = min(start_token + config.min_tokens, total_tokens)
    ideal_end = min(start_token + config.target_tokens, total_tokens)
    max_end = min(start_token + config.max_tokens, total_tokens)

    if min_end >= total_tokens:
        return total_tokens

    left = bisect.bisect_left(boundary_positions, min_end)
    right = bisect.bisect_right(boundary_positions, max_end)
    candidates = boundary_positions[left:right]
    if candidates:
        return min(candidates, key=lambda pos: abs(pos - ideal_end))
    return ideal_end


def build_token_windows(
    token_spans: List[Tuple[int, int]],
    boundary_positions: Sequence[int],
    config: WindowConfig,
) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    total_tokens = len(token_spans)
    if total_tokens == 0:
        return windows

    start = 0
    while start < total_tokens:
        remaining = total_tokens - start
        if remaining <= config.max_tokens:
            if windows and remaining < config.min_tokens:
                prev_start, _ = windows[-1]
                windows[-1] = (prev_start, total_tokens)
                break
            windows.append((start, total_tokens))
            break

        end = choose_window_end(start, total_tokens, boundary_positions, config)
        if end <= start:
            end = min(start + config.target_tokens, total_tokens)
        if end <= start:
            break

        windows.append((start, end))
        next_start = end - config.overlap_tokens
        if next_start <= start:
            next_start = end
        start = next_start

    deduped: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for window in windows:
        if window in seen:
            continue
        seen.add(window)
        deduped.append(window)
    return deduped


def trim_text_span(text: str, start_char: int, end_char: int) -> Tuple[int, int, str]:
    raw = text[start_char:end_char]
    if not raw:
        return start_char, end_char, ""
    left = len(raw) - len(raw.lstrip())
    right = len(raw) - len(raw.rstrip())
    start = start_char + left
    end = end_char - right
    if end < start:
        end = start
    return start, end, text[start:end]


def locate_unit_span(
    units: List[Dict[str, object]],
    start_char: int,
    end_char: int,
) -> Tuple[int, int]:
    unit_starts = [to_int(u.get("start_char"), 0) for u in units]
    unit_ends = [to_int(u.get("end_char"), 0) for u in units]

    start_idx = bisect.bisect_right(unit_ends, start_char)
    if start_idx >= len(units):
        start_idx = max(len(units) - 1, 0)

    end_idx = bisect.bisect_left(unit_starts, end_char) - 1
    if end_idx < start_idx:
        end_idx = start_idx
    if end_idx >= len(units):
        end_idx = len(units) - 1

    return start_idx, end_idx


def chunk_id(source_id: str, level: str, start_char: int, end_char: int, text: str) -> str:
    payload = f"{source_id}:{level}:{start_char}:{end_char}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_chunk_records(
    source_id: str,
    text: str,
    units: List[Dict[str, object]],
    token_spans: List[Tuple[int, int]],
    windows: List[Tuple[int, int]],
    level: str,
    meta: Dict[str, object],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for start_token, end_token in windows:
        start_char = token_spans[start_token][0]
        end_char = token_spans[end_token - 1][1]
        start_char, end_char, chunk_text = trim_text_span(text, start_char, end_char)
        if not chunk_text:
            continue

        unit_start_idx, unit_end_idx = locate_unit_span(units, start_char, end_char)
        unit_start = units[unit_start_idx]
        unit_end = units[unit_end_idx]
        record = {
            "chunk_id": chunk_id(source_id, level, start_char, end_char, chunk_text),
            "source_id": source_id,
            "level": level,
            "text": chunk_text,
            "token_count": end_token - start_token,
            "start_char": start_char,
            "end_char": end_char,
            "unit_type_start": unit_start.get("unit_type"),
            "unit_index_start": unit_start.get("unit_index"),
            "unit_type_end": unit_end.get("unit_type"),
            "unit_index_end": unit_end.get("unit_index"),
            "meta": meta,
        }
        records.append(record)

    return records


def attach_neighbor_links(records: List[Dict[str, object]]) -> None:
    for index, record in enumerate(records):
        record["prev_id"] = records[index - 1]["chunk_id"] if index > 0 else ""
        record["next_id"] = records[index + 1]["chunk_id"] if index + 1 < len(records) else ""


def filter_short_leaf_chunks(
    records: List[Dict[str, object]],
    min_output_tokens: int,
) -> List[Dict[str, object]]:
    if min_output_tokens <= 0 or len(records) <= 1:
        return records

    kept = [record for record in records if to_int(record.get("token_count"), 0) >= min_output_tokens]
    if kept:
        return kept

    # Keep at least one chunk per source to avoid losing short documents.
    return [max(records, key=lambda record: to_int(record.get("token_count"), 0))]


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    left = max(a_start, b_start)
    right = min(a_end, b_end)
    return max(0, right - left)


def assign_parent_links(
    leaf_records: List[Dict[str, object]],
    parent_records: List[Dict[str, object]],
) -> None:
    if not leaf_records:
        return
    if not parent_records:
        for leaf in leaf_records:
            leaf["parent_id"] = ""
        return

    parent_idx = 0
    for leaf in leaf_records:
        leaf_start = to_int(leaf.get("start_char"), 0)
        leaf_end = to_int(leaf.get("end_char"), leaf_start)

        while (
            parent_idx + 1 < len(parent_records)
            and to_int(parent_records[parent_idx].get("end_char"), 0) <= leaf_start
        ):
            parent_idx += 1

        candidates = [parent_records[parent_idx]]
        if parent_idx + 1 < len(parent_records):
            candidates.append(parent_records[parent_idx + 1])
        if parent_idx - 1 >= 0:
            candidates.append(parent_records[parent_idx - 1])

        best = max(
            candidates,
            key=lambda p, s=leaf_start, e=leaf_end: overlap_len(
                s,
                e,
                to_int(p.get("start_char"), 0),
                to_int(p.get("end_char"), 0),
            ),
        )
        leaf["parent_id"] = best["chunk_id"]


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    temp_path.replace(path)


def build_chunks_for_payload(
    source_id: str,
    payload: Dict[str, object],
    leaf_config: WindowConfig,
    parent_config: WindowConfig,
    min_leaf_output_tokens: int,
    tokenizer_mode: str,
    tokenizer_encoding: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    text = str(payload.get("text", "") or "")
    if not text.strip():
        return [], []

    units = ensure_units(payload, text)
    token_spans, resolved_tokenizer = tokenize_with_spans(text, mode=tokenizer_mode, encoding_name=tokenizer_encoding)
    if not token_spans:
        return [], []

    boundaries = unit_boundary_token_positions(token_spans, units)
    meta_raw = payload.get("meta")
    meta: Dict[str, object] = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    meta["tokenizer"] = resolved_tokenizer

    leaf_windows = build_token_windows(token_spans, boundaries, leaf_config)
    parent_windows = build_token_windows(token_spans, boundaries, parent_config)

    leaf_records = build_chunk_records(
        source_id=source_id,
        text=text,
        units=units,
        token_spans=token_spans,
        windows=leaf_windows,
        level="leaf",
        meta=meta,
    )
    parent_records = build_chunk_records(
        source_id=source_id,
        text=text,
        units=units,
        token_spans=token_spans,
        windows=parent_windows,
        level="parent",
        meta=meta,
    )

    leaf_records = filter_short_leaf_chunks(leaf_records, min_output_tokens=min_leaf_output_tokens)
    attach_neighbor_links(leaf_records)
    attach_neighbor_links(parent_records)
    assign_parent_links(leaf_records, parent_records)
    return leaf_records, parent_records


def status_counts(rows: List[Row]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = row.get("chunk_status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_chunking(
    *,
    force: bool,
    source_ids: Optional[Sequence[str]],
    limit: int,
    keep_stale_chunks: bool,
    leaf_min: int,
    leaf_target: int,
    leaf_max: int,
    leaf_overlap: int,
    parent_min: int,
    parent_target: int,
    parent_max: int,
    parent_overlap: int,
    min_leaf_output_tokens: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> Dict[str, object]:
    leaf_config = WindowConfig(
        level="leaf",
        min_tokens=leaf_min,
        target_tokens=leaf_target,
        max_tokens=leaf_max,
        overlap_tokens=leaf_overlap,
    )
    parent_config = WindowConfig(
        level="parent",
        min_tokens=parent_min,
        target_tokens=parent_target,
        max_tokens=parent_max,
        overlap_tokens=parent_overlap,
    )
    validate_window_config(leaf_config)
    validate_window_config(parent_config)
    if min_leaf_output_tokens < 0:
        raise ValueError("min_leaf_output_tokens must be >= 0")

    config_hash = compute_config_hash(
        leaf=leaf_config,
        parent=parent_config,
        tokenizer_mode=tokenizer,
        tokenizer_encoding=tiktoken_encoding,
        min_leaf_output_tokens=min_leaf_output_tokens,
    )

    rows = load_registry()
    stale_removed = purge_stale_chunks(rows, keep_stale=keep_stale_chunks)
    if stale_removed:
        LOGGER.info("Removed %d stale chunk artifacts", stale_removed)

    source_filter = {value.strip() for value in source_ids or [] if value and value.strip()}
    targets = [
        row
        for row in rows
        if should_process_row(row, config_hash=config_hash, force=force, source_filter=source_filter)
    ]
    if limit > 0:
        targets = targets[:limit]

    LOGGER.info(
        "Chunking started | targets=%d | force=%s | config_hash=%s | tokenizer=%s",
        len(targets),
        force,
        config_hash[:12],
        tokenizer,
    )

    processed = 0
    failed = 0
    for row in tqdm(targets, desc="Chunking"):
        source_id = row.get("source_id", "")
        try:
            payload_path = extracted_path_for_row(row)
            if not payload_path.exists():
                raise FileNotFoundError(f"Extracted JSON not found: {payload_path}")

            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            leaf_records, parent_records = build_chunks_for_payload(
                source_id=source_id,
                payload=payload,
                leaf_config=leaf_config,
                parent_config=parent_config,
                min_leaf_output_tokens=min_leaf_output_tokens,
                tokenizer_mode=tokenizer,
                tokenizer_encoding=tiktoken_encoding,
            )

            if not leaf_records:
                raise ValueError("No leaf chunks generated")

            write_jsonl(leaf_path(source_id), leaf_records)
            write_jsonl(parent_path(source_id), parent_records)

            row["chunk_status"] = CHUNK_CHUNKED
            row["nb_chunks"] = str(len(leaf_records))
            row["last_chunk_time"] = utc_now_iso()
            row["chunk_error"] = ""
            row["chunk_config_hash"] = config_hash
            set_index_pending(row)
            processed += 1
            LOGGER.info(
                "[CHUNKED] %s | leaf=%d | parent=%d",
                source_id,
                len(leaf_records),
                len(parent_records),
            )

        except Exception as exc:  # noqa: BLE001
            failed += 1
            row["chunk_status"] = CHUNK_FAILED
            row["nb_chunks"] = ""
            row["last_chunk_time"] = utc_now_iso()
            row["chunk_error"] = str(exc)
            row["chunk_config_hash"] = config_hash
            clear_index_tracking(row)
            remove_chunk_files_for_source(source_id)
            LOGGER.exception("[CHUNK FAILED] %s", source_id)

        write_registry(rows)

    counts = status_counts(rows)
    LOGGER.info(
        "Chunking completed | processed=%d | failed=%d | chunk_status_counts=%s",
        processed,
        failed,
        counts,
    )

    return {
        "chunk_status_counts": counts,
        "processed": processed,
        "failed": failed,
        "leaf_dir": str(LEAF_DIR),
        "parent_dir": str(PARENT_DIR),
        "config_hash": config_hash,
        "stale_removed": stale_removed,
    }


def main() -> None:  # pragma: no cover
    args = parse_args()
    result = run_chunking(
        force=args.force,
        source_ids=args.source_id,
        limit=args.limit,
        keep_stale_chunks=args.keep_stale_chunks,
        leaf_min=args.leaf_min,
        leaf_target=args.leaf_target,
        leaf_max=args.leaf_max,
        leaf_overlap=args.leaf_overlap,
        parent_min=args.parent_min,
        parent_target=args.parent_target,
        parent_max=args.parent_max,
        parent_overlap=args.parent_overlap,
        min_leaf_output_tokens=args.min_leaf_output_tokens,
        tokenizer=args.tokenizer,
        tiktoken_encoding=args.tiktoken_encoding,
    )
    print("Chunk status:", result["chunk_status_counts"])
    print("Processed:", result["processed"])
    print("Failed:", result["failed"])
    print("Leaf dir:", result["leaf_dir"])
    print("Parent dir:", result["parent_dir"])


if __name__ == "__main__":  # pragma: no cover
    main()
