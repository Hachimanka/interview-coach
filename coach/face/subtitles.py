"""Split text into subtitle chunks (<= 2 lines x ~42 chars) timed by character share (T1.10)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

MAX_CHARS = 42
MAX_LINES = 2


@dataclass
class Cue:
    text: str      # may contain one "\n" line break
    start: float   # seconds from audio start
    end: float


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def make_cues(text: str, duration_s: float) -> list[Cue]:
    lines = _wrap(text, MAX_CHARS)
    chunks = ["\n".join(lines[i:i + MAX_LINES]) for i in range(0, len(lines), MAX_LINES)]
    total = sum(len(c) for c in chunks) or 1
    cues, t = [], 0.0
    for c in chunks:
        d = duration_s * len(c) / total
        cues.append(Cue(c, round(t, 3), round(t + d, 3)))
        t += d
    return cues


def cues_to_json(cues: list[Cue]) -> list[dict]:
    return [asdict(c) for c in cues]
