"""Guard v2 (G5): checks every sentence the generative interviewer writes before it is spoken.

A failing sentence is never spoken: the session falls back to the approved rubric question or
approved reply instead. The same checks filter the synthetic training data (training/interviewer).

A sentence passes only if it:
- has the right shape for its task (a question asks one or two questions; an acknowledgment asks none),
- never judges the answer (praise is added separately, only for what the scorer found),
- never speaks as the student, never asks about private topics (age, religion, family, ...),
- adds no numbers, names or technical terms that are not in what the student said, the session memory,
  the competency goal, the example question or the knowledge passages it was given,
- adds only a few other new words (interview vocabulary is allowed).
"""
from __future__ import annotations

import re

from coach.reply.guard import JUDGING
from coach.reply.topics import find_topics

SENSITIVE = {"age", "old", "religion", "religious", "church", "god", "pray", "politics", "political", "election", "married", "marriage", "wife", "husband", "boyfriend", "girlfriend", "pregnant",
             "children", "kids", "parents", "family", "income", "salary", "money", "rich", "poor", "health",
             "sick", "illness", "disease", "disability", "medicine", "address", "ethnicity", "race", "gender",
             "sexual", "weight", "tribe", "province"}
AS_STUDENT = {"we", "our", "us", "ours", "ourselves", "my", "myself", "mine"}

# Words any interviewer may use without them being "new facts".
INTERVIEW = set("""
thank thanks sharing share shared mentioned mention mentioning telling tell told said saying say explain explained
explaining describe described walk walked through example examples specific specifically particular exactly detail
details challenge challenges challenging problem problems solve solved solving handle handled handling approach
approached step steps process method learn learned learning lesson earlier before again moment situation team
teammates teammate member members group result results outcome test tested testing decide decided decision difficult
hardest part parts think thought feel felt next now let talk talking about kind sort would could should what why how
which when where there their them they with from into have were what's it's you're that's let's okay alright sure
interesting sounds hear heard understand understood also really work worked working project projects experience
experiences using used use choose chose chosen choice pick picked role your yours yourself that this these those
more most other another first then after later during time times today going like just still even something
anything someone things thing make made making build built building create created design designed help helped
know knew improve improving improvement plan planned goal goals reason reasons achieve achieved fix fixed fixing
find found finding check checked happen happened happens start started finish finished done keep kept give gave
best better able question questions answer answers practice interview course year school class subject training
job company field area skills skill strength strengths weakness weaknesses apply applying responsible
responsibility contribute contributed lead led manage managed mind ready glad great nice meet
""".split()) - JUDGING

STOP = {"the", "and", "for", "you", "are", "was", "did", "does", "how", "why", "who", "can", "its", "it", "a",
        "an", "of", "to", "in", "on", "is", "be", "so", "or", "as", "at", "by", "i", "me", "do", "if", "but",
        "not", "no", "yes", "all", "any", "some", "one", "two", "had", "has", "will", "been", "being", "than",
        "very", "much", "many", "each", "own", "too", "may", "might", "must", "same", "such", "only", "over",
        "out", "up", "down", "off", "again", "here", "there", "while", "get", "got", "go", "went", "come"}
CAPS_OK = {"I", "Computer", "Engineering", "Electronics", "CpE", "ECE", "OJT"}

MAX_WORDS = {"question": 42, "follow_up": 36, "ack": 22, "coach": 70}
MAX_NEW = {"question": 6, "follow_up": 5, "ack": 3, "coach": 6}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _stem(w: str) -> str:
    return w[:5]


REQUEST = re.compile(r"(?:^|[.!]\s+)(please\s+)?(tell me|describe|walk me through|introduce yourself|explain|share|"
                     r"talk about|give me an example)", re.I)
CLAIM = re.compile(r"\b(you mentioned|you said|you told me|you talked about|earlier you|you shared|you described|"
                   r"you explained)\b|^\s*(so,? you|you're|you are)\b", re.I)


def check_turn(text: str, kind: str, context: str, name: str | None = None,
               allow_praise: bool = False, memory: str | None = None) -> tuple[bool, str]:
    """(ok, reason). kind: question | follow_up | ack | coach.
    context: everything the sentence may draw on (student's words, memory, goal, example, passages).
    memory: only what the student really said (+ remembered facts). A sentence that claims "you mentioned
    ..." must be about something in here, not something from the goal or the example question."""
    if not text or not text.strip():
        return False, "empty"
    words = _words(text)
    if len(words) > MAX_WORDS[kind]:
        return False, f"too long ({len(words)} words)"
    n_q = text.count("?")
    request = REQUEST.search(text)          # "Please introduce yourself", "Tell me about ..." also ask
    if kind in ("question", "follow_up") and n_q == 0 and not request:
        return False, "does not ask a question"
    if kind in ("question", "follow_up") and n_q > 2:
        return False, f"asks {n_q} questions at once"
    if kind in ("ack", "coach") and n_q:
        return False, "asks a question"
    # greetings and tag questions are not judgments: "Good morning", "..., right?"
    plain = re.sub(r"\bgood (morning|afternoon|evening|day)\b|,?\s*right\s*\?", " ", text, flags=re.I)
    lowered = " " + " ".join(_words(plain)) + " "
    if not allow_praise:
        for j in JUDGING:
            if f" {j} " in lowered:
                return False, f"judging word '{j}'"
    else:
        for j in ("wrong", "incorrect", "bad", "poor", "mistake", "fail", "failed", "terrible", "weak"):
            if f" {j} " in lowered:
                return False, f"negative word '{j}'"
    for w in words:
        if w in AS_STUDENT:
            return False, f"speaks as the student ('{w}')"
        if w in SENSITIVE:
            return False, f"private topic ('{w}')"
    if re.search(r"[^\x00-\x7F’‘“”—–…]", text):
        return False, "non-English characters"
    ctx_words = set(_words(context))
    ctx_numbers = set(re.findall(r"\d+", context))
    for n in re.findall(r"\d+", text):
        if n not in ctx_numbers:
            return False, f"new number '{n}'"
    if memory is not None:
        said = {_stem(w) for w in _words(memory)}
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            m = CLAIM.search(sentence)
            if not m:
                continue
            claim = sentence[m.end():].split("?")[0]
            claim = re.split(r"\b(how|why|what|which|when|could|can|did|do|would|tell)\b", claim, flags=re.I)[0]
            content = [w for w in _words(claim) if len(w) > 3 and w not in STOP and w not in INTERVIEW]
            unsaid = [w for w in content if _stem(w) not in said]
            if len(unsaid) > 1:
                return False, f"claims the student said something they did not: {', '.join(unsaid[:4])}"
    # technical terms must come from the context (no invented components or tools)
    known_topics = {t.name for t in find_topics(context)}
    low_ctx = context.lower()
    for t in find_topics(text):
        if t.category == "activity":          # "group project", "thesis": not technical claims
            continue
        if t.name not in known_topics and t.name.lower().rstrip("s") not in low_ctx:
            return False, f"new technical term '{t.name}'"
    # names: a capitalized word mid-sentence that is not known (an invented name or place)
    name_parts = set((name or "").split())
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+)\b", text.strip()):
        w = m.group(1)
        if w in CAPS_OK or w in name_parts or w.lower() in ctx_words or w.lower() in INTERVIEW | STOP:
            continue
        return False, f"unknown name '{w}'"
    if name:
        if text.count(name) > 1:
            return False, "repeats the name"
    known = {_stem(w) for w in ctx_words} | {_stem(w) for w in INTERVIEW}
    new = [w for w in words if len(w) > 3 and w not in STOP and _stem(w) not in known]
    if len(new) > MAX_NEW[kind]:
        return False, f"adds words not said or given: {', '.join(new[:6])}"
    return True, "ok"
