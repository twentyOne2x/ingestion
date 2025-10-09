from __future__ import annotations

from typing import Any, Optional


def snip_text(value: Optional[str], max_length: int) -> str:
    """
    Trim `value` to at most `max_length` characters, appending an ellipsis if needed.
    """
    text = (value or "").strip()
    if len(text) <= max_length:
        return text
    clipped = max(0, max_length - 1)
    return text[:clipped] + "…"


def ascii_safe(value: Optional[str], max_length: int = 400) -> str:
    """
    Best-effort ASCII representation of a string, primarily for logging.
    """
    snippet = snip_text(value, max_length)
    try:
        return snippet.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return snippet


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def ms_to_seconds(value: Any) -> Optional[float]:
    milliseconds = safe_float(value)
    if milliseconds is None:
        return None
    return milliseconds / 1000.0
