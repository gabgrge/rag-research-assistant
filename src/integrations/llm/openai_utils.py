from __future__ import annotations

import logging
import time
from typing import Callable, Mapping, Optional, Tuple, TypeVar, cast

from src.utils.type_hints import OpenAIClient

T = TypeVar("T")


def build_openai_client(api_key: str, timeout_sec: float) -> OpenAIClient:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'openai'. Install with: pip install openai") from exc

    if not api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or pass --openai-api-key.")

    try:
        return cast(OpenAIClient, cast(object, OpenAI(api_key=api_key, timeout=timeout_sec)))
    except TypeError:
        return cast(OpenAIClient, cast(object, OpenAI(api_key=api_key)))


def extract_http_status_code(exc: Exception) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    if response is None:
        return None

    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw_value: Optional[object] = None
    if isinstance(headers, Mapping):
        raw_value = headers.get("retry-after")
        if raw_value is None:
            raw_value = headers.get("Retry-After")
    if raw_value is None:
        return None

    try:
        delay = float(str(raw_value).strip())
    except ValueError:
        return None
    if delay < 0:
        return 0.0
    return delay


def classify_openai_error(exc: Exception) -> Tuple[bool, Optional[float], str]:
    status = extract_http_status_code(exc)
    if status is not None:
        if status == 429:
            return True, extract_retry_after_seconds(exc), f"http_{status}"
        if status in (408, 409):
            return True, None, f"http_{status}"
        if 500 <= status <= 599:
            return True, None, f"http_{status}"
        if 400 <= status <= 499:
            return False, None, f"http_{status}"

    name = exc.__class__.__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return True, None, name
    return False, None, name


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    retry_base_delay_sec: float,
    label: str,
    logger: Optional[logging.Logger] = None,
) -> T:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retryable, retry_after, reason = classify_openai_error(exc)
            if not retryable:
                if logger is not None:
                    logger.error("Non-retryable %s error (%s): %s", label, reason, exc)
                raise
            if attempt >= max_retries:
                break
            delay = retry_after if retry_after is not None else retry_base_delay_sec * (2**attempt)
            if logger is not None:
                logger.warning(
                    "Retryable %s error (%s) attempt %d/%d. Retrying in %.2fs | %s",
                    label,
                    reason,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    exc,
                )
            time.sleep(delay)
    if last_error is None:
        raise RuntimeError(f"{label} failed with unknown error.")
    raise last_error
