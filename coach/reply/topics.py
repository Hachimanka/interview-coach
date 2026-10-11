"""What the student talked about (approved glossary) and how they feel (phrase lists), plus the
approved reply templates. Only canonical glossary names are ever spoken back, never raw ASR text."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from coach.profile import PROJECT_ROOT

SAMPLES = PROJECT_ROOT / "samples"
CATEGORY_ORDER = ("hardware", "software", "tool", "concept", "activity")   # most specific first
FUZZY = 0.86


@dataclass(frozen=True)
class Topic:
    name: str        # canonical name, safe to speak
    category: str
    position: int    # where it was first mentioned (word index)


def _norm(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+]+", text.lower().replace("-", " "))


@lru_cache(maxsize=1)
def glossary() -> list[dict]:
    return json.loads((SAMPLES / "glossary.json").read_text(encoding="utf-8"))["terms"]


@lru_cache(maxsize=1)
def templates() -> dict:
    return json.loads((SAMPLES / "replies.json").read_text(encoding="utf-8"))


def find_topics(text: str) -> list[Topic]:
    """Glossary terms mentioned in the text, most specific category first, then by first mention."""
    words = _norm(text)
    grams = {}
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            grams.setdefault(" ".join(words[i:i + n]), i)
    found: dict[str, Topic] = {}
    for term in glossary():
        for alias in term["aliases"]:
            a = " ".join(_norm(alias))
            pos = grams.get(a)
            if pos is None and len(a) >= 6:   # fuzzy only for longer aliases ("ardwino", "kicad")
                pos = next((p for g, p in grams.items()
                            if abs(len(g) - len(a)) <= 2 and SequenceMatcher(None, g, a).ratio() >= FUZZY), None)
            if pos is not None:
                old = found.get(term["name"])
                if old is None or pos < old.position:
                    found[term["name"]] = Topic(term["name"], term["category"], pos)
    # a generic term ("sensors") is dropped when a specific one ("ultrasonic sensor") was also said
    names = set(found)
    specific = [t for t in found.values()
                if not any(t.name != o and t.name.rstrip("s") in o for o in names)]
    return sorted(specific, key=lambda t: (CATEGORY_ORDER.index(t.category), t.position))


@lru_cache(maxsize=1)
def _misheard() -> list[tuple[re.Pattern, str]]:
    out = []
    for term in glossary():
        for m in term.get("misheard", []):
            words = [re.escape(w) for w in m.split()]
            out.append((re.compile(r"(?<![A-Za-z0-9])" + r"[\s\-]*".join(words) + r"(?![A-Za-z0-9])", re.I), term["name"]))
    return sorted(out, key=lambda x: -len(x[0].pattern))


def canonicalize(text: str) -> str:
    """Replace known speech-recognition mistakes with the glossary name ("ESB 32" -> "ESP32"), so a
    mishearing is never repeated to the student by the AI or kept in the session memory."""
    for pattern, name in _misheard():
        text = pattern.sub(name, text)
    return text


def feeling(text: str) -> str | None:
    """'nervous' / 'stuck' if the student said so (English or Filipino), else None."""
    low = " " + " ".join(_norm(text.replace("'", ""))) + " "
    for kind, langs in templates()["feelings"].items():
        for phrases in langs.values():
            for p in phrases:
                if f" {' '.join(_norm(p.replace(chr(39), '')))} " in low:
                    return kind
    return None


def topic_follow_up(topic: Topic) -> str:
    return templates()["topic_follow_up"][topic.category]["en"].format(topic=topic.name)


def topic_ack(topic: Topic) -> str:
    return templates()["topic_ack"]["en"].format(topic=topic.name)
