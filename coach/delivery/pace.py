"""Words per minute from ASR word timestamps (T1.5)."""
from __future__ import annotations

from coach.asr.base import Word


def words_per_minute(words: list[Word]) -> float | None:
    """Over the span from first to last word (includes pauses, like a listener hears it)."""
    if len(words) < 2:
        return None
    span = words[-1].end - words[0].start
    return round(60 * len(words) / span, 1) if span > 0 else None
