"""Checks every AI-written reply before it is spoken. Anything that fails is replaced by the approved line.

A reply passes only if it: is short, keeps the approved question (if any), never judges the answer,
adds no numbers, and adds almost no words that are in neither the approved line nor what the student said.
"""
from __future__ import annotations

import re

JUDGING = {"great", "good", "excellent", "nice", "impressive", "amazing", "awesome", "perfect", "wonderful",
           "fantastic", "brilliant", "outstanding", "wrong", "incorrect", "correct", "right", "bad", "poor",
           "weak", "strong", "mistake", "fail", "failed", "score", "grade", "well done", "good job"}

# Words a natural rephrasing may add without changing meaning.
ALLOWED = {"thank", "thanks", "sharing", "share", "mentioned", "mention", "telling", "tell", "said", "saying",
           "explain", "explained", "describe", "could", "would", "please", "about", "your", "that", "this",
           "work", "worked", "working", "project", "projects", "experience", "using", "used", "choose", "chose",
           "chosen", "decide", "decided", "part", "role", "more", "also", "really", "sounds", "hear", "heard",
           "understand", "interesting", "specifically", "particular", "exactly", "what", "which", "when",
           "where", "there", "their", "them", "they", "with", "from", "into", "have", "were", "what's", "it's",
           "you're", "that's", "let's", "okay", "alright", "sure"}
STOP = {"the", "and", "for", "you", "are", "was", "did", "does", "how", "why", "who", "can", "its", "it",
        "a", "an", "of", "to", "in", "on", "is", "be", "so", "or", "as", "at", "by", "we", "i", "my", "me",
        "our", "us", "do", "if", "but", "not", "no", "yes", "all", "any", "some", "one", "two", "had", "has"}
MAX_NEW_WORDS = 3


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _stem(w: str) -> str:
    return w[:5]


def check(reply: str, approved: str, said: str, topic: str | None = None, max_words: int = 25) -> tuple[bool, str]:
    """(ok, reason). reason explains a rejection for the report."""
    if not reply or not reply.strip():
        return False, "empty"
    words = _words(reply)
    if len(words) > max_words + 5:
        return False, f"too long ({len(words)} words)"
    lowered = " " + " ".join(words) + " "
    for j in JUDGING:
        if f" {j} " in lowered:
            return False, f"judging word '{j}'"
    # The robot must not speak as if it were the student ("In our group project...").
    for w in ("we", "our", "us", "my", "ours", "ourselves", "myself"):
        if w in words:
            return False, f"speaks as the student ('{w}')"
    if approved.rstrip().endswith("?") and "?" not in reply:
        return False, "lost the approved question"
    known_numbers = set(re.findall(r"\d+", approved + " " + said + " " + (topic or "")))
    for n in re.findall(r"\d+", reply):
        if n not in known_numbers:
            return False, f"new number '{n}'"
    if re.search(r"[^\x00-\x7F’—–]", reply):
        return False, "non-English characters"
    known = {_stem(w) for w in _words(approved + " " + said + " " + (topic or ""))}
    new = [w for w in words if len(w) > 3 and w not in STOP and w not in ALLOWED and _stem(w) not in known]
    if len(new) > MAX_NEW_WORDS:
        return False, f"adds words not said or approved: {', '.join(new[:5])}"
    if topic and topic.lower() not in reply.lower():
        return False, f"dropped the topic '{topic}'"
    return True, "ok"
