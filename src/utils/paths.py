from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
LOGS_DIR = RUNTIME_DIR / "logs"
REGISTRY_DIR = RUNTIME_DIR / "registry"
BACKUPS_DIR = RUNTIME_DIR / "backups"
DATA_DIR = RUNTIME_DIR / "data"
CONVERTED_DIR = DATA_DIR / "converted"
EXTRACTED_DIR = DATA_DIR / "extracted"
INDEX_DIR = DATA_DIR / "index" / "chroma"
CHUNKS_DIR = DATA_DIR / "chunks"
LEAF_CHUNKS_DIR = CHUNKS_DIR / "leaf"
PARENT_CHUNKS_DIR = CHUNKS_DIR / "parent"
