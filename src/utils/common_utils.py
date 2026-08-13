from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str, bytes, bytearray)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default
