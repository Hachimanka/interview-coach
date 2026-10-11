"""Session memory (G5a): what the robot remembers about the student, from Q1 to the closing.

Everything lives in RAM for one session only. forget() clears it; the session calls it when the
student deletes the report (and a new Session starts with an empty memory anyway). Nothing here is
ever written to disk or used for training.

Facts come from two places:
- rules (instant): topics from the approved glossary, and feelings ("I'm nervous");
- the local AI (in the background): short facts in the student's own words. An AI fact is kept only
  if almost all its words were really said (grounded()), so a made-up detail can't enter memory and
  then be "remembered" later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from coach.reply.topics import Topic, feeling, find_topics

_WORD = re.compile(r"[a-z0-9']+")
STOP = {"the", "and", "for", "you", "are", "was", "did", "does", "how", "why", "who", "can", "its", "it",
        "a", "an", "of", "to", "in", "on", "is", "be", "so", "or", "as", "at", "by", "we", "i", "my", "me",
        "our", "us", "do", "if", "but", "not", "no", "yes", "all", "any", "some", "one", "had", "has",
        "they", "them", "their", "student", "he", "she", "his", "her", "with", "that", "this", "from",
        "about", "also", "were", "have", "been", "into", "when", "what", "which", "who", "very", "really"}

NOT_NAMES = {"yes", "no", "okay", "ok", "hello", "hi", "um", "uh", "ah", "the", "a", "my", "name", "is", "i", "im",
             "am", "po", "ako", "si", "ang", "pangalan", "ko", "call", "me", "you", "can", "just", "it", "its",
             "sorry", "what", "good", "morning", "afternoon", "evening", "thank", "thanks", "start", "simulan",
             "oo", "hindi", "sige", "opo", "ma'am", "sir", "maam", "student", "computer", "engineering", "and"}
NAME_PATTERNS = [r"\bmy name is ([a-z][a-z'\-]+)", r"\bmy name's ([a-z][a-z'\-]+)", r"\bcall me ([a-z][a-z'\-]+)",
                 r"\bi am ([a-z][a-z'\-]+)", r"\bi'm ([a-z][a-z'\-]+)", r"\bim ([a-z][a-z'\-]+)",
                 r"\bako si ([a-z][a-z'\-]+)", r"\bpangalan ko ay ([a-z][a-z'\-]+)",
                 r"\bpangalan ko (?:po )?(?:ay )?([a-z][a-z'\-]+)", r"\bit's ([a-z][a-z'\-]+)",
                 r"\bthis is ([a-z][a-z'\-]+)"]
COMPOUND_FIRST = {"john", "jhon", "mary", "ma", "maria", "ann", "anne", "jean", "mark", "jose", "juan", "kim"}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def extract_name(text: str) -> str | None:
    """First name from "My name is Leonard" / "I'm Leonard" / "Ako si Leonard po" / just "Leonard".
    Two-word first names ("John Paul", "Mary Ann") are kept together."""
    low = " ".join(_words(text.replace("’", "'")))
    if not low:
        return None
    tokens: list[str] = []
    for pat in NAME_PATTERNS:
        m = re.search(pat, low)
        if m and m.group(1) not in NOT_NAMES:
            tokens = low[m.start(1):].split()
            break
    if not tokens:
        rest = [w for w in low.split() if w not in NOT_NAMES]
        if not rest or len(rest) > 3:      # a long sentence with no "my name is": not a name
            return None
        tokens = rest
    first = tokens[0]
    if first in NOT_NAMES or len(first) < 2 or first.isdigit():
        return None
    name = [first]
    if first in COMPOUND_FIRST and len(tokens) > 1 and tokens[1] not in NOT_NAMES and len(tokens[1]) > 1:
        name.append(tokens[1])
    return " ".join(w.capitalize() for w in name)


@dataclass
class Fact:
    text: str
    kind: str          # topic | feeling | ai | said
    q: int             # question number (1-based) it came from
    source: str = "rules"


@dataclass
class SessionMemory:
    name: str | None = None
    name_every: int = 2                       # use the name at most every N robot turns
    facts: list[Fact] = field(default_factory=list)
    said: dict[int, str] = field(default_factory=dict)          # question number -> what the student said
    slots: dict[int, str] = field(default_factory=dict)         # question number -> competency
    used: set[str] = field(default_factory=set)                 # facts already referred to
    turn: int = 0
    last_name_turn: int = -99

    # ---------- writing ----------
    def add_answer(self, q: int, competency: str, text: str) -> list[Fact]:
        """Instant rule facts from one answer (topics, feelings)."""
        self.slots[q] = competency
        self.said[q] = (self.said.get(q, "") + " " + text).strip()
        new = []
        for t in find_topics(text):
            if not any(f.kind == "topic" and f.text == t.name for f in self.facts):
                new.append(Fact(t.name, "topic", q))
        if feeling(text) == "nervous" and not any(f.kind == "feeling" for f in self.facts):
            new.append(Fact(f"said they were nervous in question {q}", "feeling", q))
        self.facts += new
        return new

    def add_ai_facts(self, q: int, lines: list[str]) -> list[Fact]:
        """Facts written by the local AI; only grounded ones are kept."""
        said = self.said.get(q, "")
        new = []
        for line in lines:
            text = line.strip().lstrip("-•* ").strip().rstrip(".")
            if not text or text.lower() in ("none", "no facts") or len(text.split()) > 18:
                continue
            if grounded(text, said) and not any(f.text.lower() == text.lower() for f in self.facts):
                new.append(Fact(text, "ai", q, "ai"))
        self.facts += new
        return new

    def forget(self) -> None:
        self.name = None
        self.facts.clear()
        self.said.clear()
        self.slots.clear()
        self.used.clear()
        self.turn, self.last_name_turn = 0, -99

    # ---------- reading ----------
    def topics(self) -> list[str]:
        return [f.text for f in self.facts if f.kind == "topic"]

    def notes(self, max_words: int = 160) -> str:
        """Short notes for the AI prompt: one line per earlier question (AI facts, else the start of
        what was said), plus feelings. Kept short: a long prompt is slow on the Pi."""
        lines = []
        for q in sorted(self.said):
            facts = [f.text for f in self.facts if f.q == q and f.kind == "ai"]
            body = "; ".join(facts) if facts else " ".join(self.said[q].split()[:22])
            lines.append(f"- Q{q} ({self.slots.get(q, '')}): {body}")
        feel = [f.text for f in self.facts if f.kind == "feeling"]
        if feel:
            lines.append(f"- feelings: {feel[0]}")
        out, n = [], 0
        for line in reversed(lines):           # the most recent questions matter most
            n += len(line.split())
            if n > max_words:
                break
            out.insert(0, line)
        return "\n".join(out) or "- (nothing yet)"

    def grounding(self) -> str:
        """Everything the student said and every remembered fact: what a generated sentence may refer to."""
        return " ".join([*self.said.values(), *(f.text for f in self.facts), self.name or ""])

    def callback_candidates(self, before_q: int) -> list[Fact]:
        """Facts from earlier questions that have not been referred to yet (AI facts first, then topics)."""
        facts = [f for f in self.facts if f.q < before_q and f.kind in ("ai", "topic") and f.text not in self.used]
        return sorted(facts, key=lambda f: (f.kind != "ai", f.q))

    def may_use_name(self) -> bool:
        return bool(self.name) and self.turn - self.last_name_turn >= self.name_every

    def next_turn(self, used_name: bool) -> None:
        self.turn += 1
        if used_name:
            self.last_name_turn = self.turn


def grounded(text: str, said: str, allow_new: int = 1) -> bool:
    """True if every content word of `text` (but `allow_new`) was said (same first 5 letters)."""
    known = {w[:5] for w in _words(said)}
    content = [w for w in _words(text) if len(w) > 2 and w not in STOP]
    new = [w for w in content if w[:5] not in known]
    return bool(content) and len(new) <= allow_new


def topic_names(text: str) -> set[str]:
    return {t.name for t in find_topics(text)}


__all__ = ["Fact", "SessionMemory", "extract_name", "grounded", "topic_names", "Topic"]
