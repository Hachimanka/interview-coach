"""Long pauses from gaps between VAD segments, and the delay before the answer starts (T1.5)."""
from __future__ import annotations


def long_pauses(spans: list[tuple[float, float]], threshold_s: float) -> list[dict]:
    """spans = (start, end) of speech segments in seconds, in order."""
    out = []
    for (_, end), (start, _) in zip(spans, spans[1:]):
        gap = start - end
        if gap >= threshold_s:
            out.append({"at_s": round(end, 1), "duration_s": round(gap, 1)})
    return out


def response_latency(spans: list[tuple[float, float]], answer_start: float) -> float | None:
    """Seconds from the moment the student could answer until they started speaking."""
    return round(max(spans[0][0] - answer_start, 0.0), 1) if spans else None
