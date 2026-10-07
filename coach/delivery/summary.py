"""Combine pace, pauses, fillers and eye contact into one delivery record + plain-language tips (T1.5)."""
from __future__ import annotations

from coach.asr.base import Word
from coach.delivery.fillers import count_fillers
from coach.delivery.pace import words_per_minute
from coach.delivery.pauses import long_pauses, response_latency


def analyze(text: str, words: list[Word], spans: list[tuple[float, float]], answer_start: float,
            cfg, eye_contact_pct: float | None = None) -> dict:
    d, a = cfg.delivery, cfg.audio
    wpm = words_per_minute(words)
    fill = count_fillers(text)
    minutes = max((words[-1].end - words[0].start) / 60, 1 / 60) if len(words) > 1 else None
    per_min = round(fill["total"] / minutes, 1) if minutes else None
    pauses = long_pauses(spans, a.long_pause_s)
    latency = response_latency(spans, answer_start)

    tips = []
    if wpm is not None and wpm < d.wpm_slow:
        tips.append(f"You spoke at about {wpm:.0f} words per minute. Try a slightly faster, steadier pace (about 120 to 160).")
    if wpm is not None and wpm > d.wpm_fast:
        tips.append(f"You spoke at about {wpm:.0f} words per minute. Slow down a little so the interviewer can follow (about 120 to 160).")
    if per_min is not None and per_min > d.fillers_per_min_max:
        top = ", ".join(f'"{k}"' for k, _ in sorted(fill["detail"].items(), key=lambda kv: -kv[1])[:3])
        tips.append(f"You used filler words about {per_min:.0f} times per minute ({top}). Pause silently instead.")
    if pauses:
        tips.append(f"You had {len(pauses)} long pause(s). If you need time, say \"Let me think for a moment.\"")
    if latency is not None and latency > d.latency_slow_s:
        tips.append(f"You took {latency:.0f} seconds to start. A short opening line helps you start sooner.")
    if eye_contact_pct is not None and eye_contact_pct < 60:
        tips.append(f"You looked at the interviewer {eye_contact_pct:.0f}% of the time. Try to look up more while you speak.")

    return {"wpm": wpm, "fillers": fill, "fillers_per_min": per_min, "long_pauses": pauses,
            "response_latency_s": latency, "eye_contact_pct": eye_contact_pct, "tips": tips}
