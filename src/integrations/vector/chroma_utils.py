from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

from src.utils.type_hints import ChromaCollection


def build_chroma_collection(index_dir: Path, collection_name: str) -> ChromaCollection:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'chromadb'. Install with: pip install chromadb") from exc

    settings = _build_chroma_settings()
    if settings is None:
        client = chromadb.PersistentClient(path=str(index_dir))
    else:
        client = chromadb.PersistentClient(path=str(index_dir), settings=settings)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return cast(ChromaCollection, cast(object, collection))


def _build_chroma_settings() -> Optional[object]:
    try:
        from chromadb.config import Settings
    except ImportError:
        return None
    return Settings(anonymized_telemetry=False)
