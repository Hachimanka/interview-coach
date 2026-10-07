"""Filler words from the transcript (T1.5). Heuristic v0; T2.4 measures it, T3.3 replaces it if needed.

Two kinds are counted separately:
- hesitations: um, uh, erm, hmm, eh ... (always fillers)
- discourse markers: "ano", "parang", "bale", "kasi", "like", "you know", "I mean". These are also
  real words, so they count only when set off by a pause mark (comma/ellipsis) or repeated.
"""
from __future__ import annotations

import re
from collections import Counter

HESITATIONS = {"um", "umm", "uhm", "uh", "uhh", "erm", "er", "ah", "ahh", "hmm", "hm", "mm", "eh", "ehh"}
MARKERS = {"ano", "parang", "bale", "kasi", "like", "basically"}
PHRASE_MARKERS = ("you know", "i mean")

_TOKEN = re.compile(r"[\w'-]+|[.,!?…]+")


def count_fillers(text: str) -> dict:
    tokens = _TOKEN.findall(text.lower())
    hes, mark = Counter(), Counter()
    for i, tok in enumerate(tokens):
        if tok in HESITATIONS:
            hes[tok] += 1
        elif tok in MARKERS:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            prv = tokens[i - 1] if i > 0 else ""
            if nxt[:1] in {",", ".", "…"} or nxt == tok or prv == tok:
                mark[tok] += 1
    flat = " ".join(t for t in tokens if t[0].isalnum())
    for phrase in PHRASE_MARKERS:
        n = len(re.findall(rf"\b{phrase}\b", flat))
        if n:
            mark[phrase] += n
    return {"hesitations": sum(hes.values()), "markers": sum(mark.values()),
            "total": sum(hes.values()) + sum(mark.values()),
            "detail": dict(hes + mark)}
