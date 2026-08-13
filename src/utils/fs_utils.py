from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional


def atomic_replace_with_retry(
    source: Path,
    target: Path,
    *,
    retries: int = 8,
    base_delay_sec: float = 0.15,
    logger: Optional[logging.Logger] = None,
) -> None:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = base_delay_sec * (2**attempt)
            if logger is not None:
                logger.warning(
                    "File lock during replace (%s -> %s), retry %d/%d in %.2fs",
                    source,
                    target,
                    attempt + 1,
                    retries + 1,
                    delay,
                )
            time.sleep(delay)

    if last_error is not None:
        raise last_error


def path_is_under(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False
