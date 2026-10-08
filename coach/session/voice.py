"""Voice commands: every on-screen choice can be spoken, in English or Filipino (no touch screen needed).

A spoken command becomes the same choice a tap would make, so the session flow does not care how
the student answered. Only the options on screen (plus "stop the interview") are valid at any moment.

Matching is plain text matching on the Whisper transcript: no extra model, works offline.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from coach.audio.vad import Segmenter

FUZZY = 0.84          # similarity needed for a near-miss spelling ("simula an" -> "simulan")
FUZZY_MIN_CHARS = 6   # short words ("oo", "no", "yes") must match exactly
MAX_CHOICE_WORDS = 8  # longer utterances on a choice screen are chatter, not a command


def normalize(text: str) -> str:
    """Lowercase, drop accents and punctuation: "That's my answer." -> "thats my answer"."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"['’`]", "", text)
    return " ".join(re.sub(r"[^a-z0-9ñ ]+", " ", text).split())


def _p(lang: str | None, *phrases: str, whole: bool = False) -> list[tuple[str, str | None, bool]]:
    return [(normalize(p), lang, whole) for p in phrases]


# command -> [(phrase, language, whole)]
# language None = the phrase says nothing about which language the student speaks.
# whole=True = counts only when it is the entire utterance (used for risky short words in answers).
COMMANDS: dict[str, list[tuple[str, str | None, bool]]] = {
    "start": _p("en", "start", "begin", "lets start", "let us start", "im ready", "i am ready", "ready")
             + _p(None, "hello", "hi")
             + _p("fil", "simulan", "simulan na", "simulan na natin", "magsimula", "magsimula na tayo",
                  "kumusta", "handa na ako", "handa na po ako", "tara"),
    "yes": _p("en", "yes", "yeah", "yep", "yes i agree", "i agree", "sure")
           + _p(None, "okay", "ok")
           + _p("fil", "oo", "opo", "oo po", "sige", "sige po", "pumapayag ako", "pumapayag po ako", "payag ako"),
    "no": _p("en", "no", "nope", "i do not agree", "i dont agree", "i disagree", "no thanks", "no thank you")
          + _p("fil", "hindi", "hindi po", "ayoko", "ayoko po", "ayaw ko", "huwag", "huwag na",
               "hindi ako pumapayag", "hindi po ako pumapayag"),
    # Letters ("E E", "E C E", "C P E") sound alike, so the screen asks for the full word or a number.
    "CpE": _p(None, "computer engineering", "computer", "kompyuter", "cpe", "c p e"),
    "EE": _p(None, "electrical engineering", "electrical", "elektrikal", "ee", "e e"),
    "ECE": _p(None, "electronics engineering", "electronics", "electronic", "elektroniks", "ece", "e c e"),
    "keep": _p("en", "keep it", "keep", "save it", "save") + _p("fil", "itago", "itago mo", "i save", "isave"),
    "delete": _p("en", "delete it", "delete", "dont keep it", "do not keep it", "dont save it")
              + _p("fil", "burahin", "burahin mo", "huwag itago", "huwag i save"),
    # End of an answer: must come at the END of what was said, followed by a pause.
    "done": _p("en", "thats my answer", "that is my answer", "that was my answer", "thats all", "that is all",
               "im done", "i am done", "im finished", "i am finished")
            + _p("en", "done", "finished", "next question", whole=True)
            + _p("fil", "iyon na po ang sagot ko", "iyon na ang sagot ko", "yun na po ang sagot ko",
                 "yun na ang sagot ko", "iyon lang po", "yun lang po", "tapos na po ako", "tapos na ako")
            + _p("fil", "tapos na", "tapos na po", "iyon lang", "yun lang", whole=True),
    # Two words on purpose, so a "stop" inside an answer does not end the session.
    "stop": _p("en", "stop the interview", "end the interview", "stop interview", "stop the session")
            + _p("fil", "itigil ang interview", "ihinto ang interview", "itigil na ang interview", "tigil na ang interview"),
}

# What the screen tells the student to say for each command.
HINTS: dict[str, dict[str, str]] = {
    "start": {"en": "start", "fil": "simulan"},
    "yes": {"en": "yes", "fil": "oo"},
    "no": {"en": "no", "fil": "hindi"},
    "CpE": {"en": "computer", "fil": "computer"},
    "EE": {"en": "electrical", "fil": "electrical"},
    "ECE": {"en": "electronics", "fil": "electronics"},
    "keep": {"en": "keep it", "fil": "itago"},
    "delete": {"en": "delete it", "fil": "burahin"},
    "done": {"en": "that's my answer", "fil": "iyon na po ang sagot ko"},
    "stop": {"en": "stop the interview", "fil": "itigil ang interview"},
}

NUMBERS = [("one", "1", "isa", "uno", "una", "first"), ("two", "2", "dalawa", "dos", "pangalawa", "second"),
           ("three", "3", "tatlo", "tres", "pangatlo", "third"), ("four", "4", "apat", "kwatro", "pang apat", "fourth")]
_NUMBER_FILLER = {"number", "numero", "option", "po", "the", "ang"}


def hint(command: str, lang: str | None) -> str:
    """Text for a button: “yes” in the known language, “yes” / “oo” while the language is unknown."""
    h = HINTS.get(command) or {"en": command.lower(), "fil": command.lower()}
    if lang in h:
        return f"“{h[lang]}”"
    return f"“{h['en']}”" if h["en"] == h["fil"] else f"“{h['en']}” / “{h['fil']}”"


@dataclass
class Match:
    command: str
    lang: str | None      # language of the matched phrase, if it tells us one
    n_words: int          # words the phrase covers (for cutting it out of a transcript)
    exact: bool = True    # False = matched by similar spelling


def _same(words: list[str], phrase: str) -> int:
    """2 = same words, 1 = similar spelling, 0 = different."""
    said = " ".join(words)
    if said == phrase:
        return 2
    return int(len(phrase) >= FUZZY_MIN_CHARS and SequenceMatcher(None, said, phrase).ratio() >= FUZZY)


def match(text: str, commands, tail: bool = False, numbered: bool = False) -> Match | None:
    """Find one of `commands` in a transcript.

    tail=False (choice screens): a short utterance that contains the phrase.
    tail=True (inside an answer): the phrase must be the last thing said.
    numbered=True: "one" / "isa" / "2" also pick by position in `commands`.
    Two different commands matching equally well is treated as unclear (returns None).
    """
    tokens = normalize(text).split()
    if not tokens or (not tail and len(tokens) > MAX_CHOICE_WORDS):
        return None
    found: list[Match] = []
    for cmd in commands:
        for phrase, lang, whole in COMMANDS.get(cmd) or _p(None, cmd):
            n = len(phrase.split())
            if n > len(tokens) or (whole and tail and n != len(tokens)):
                continue
            spans = [tokens[-n:]] if tail else [tokens[i:i + n] for i in range(len(tokens) - n + 1)]
            how = max((_same(s, phrase) for s in spans), default=0)
            if how:
                found.append(Match(cmd, lang, n, how == 2))
    if numbered and not tail and len(commands) > 1:
        rest = [t for t in tokens if t not in _NUMBER_FILLER]
        for i, names in enumerate(NUMBERS[: len(commands)]):
            if len(rest) == 1 and rest[0] in names or " ".join(rest) in names:
                found.append(Match(list(commands)[i], None, len(tokens)))
    if not found:
        return None
    best = max((m.exact, m.n_words) for m in found)     # exact wording first, then the longest phrase
    top = [m for m in found if (m.exact, m.n_words) == best]
    if len({m.command for m in top}) > 1:
        return None
    return next((m for m in top if m.lang), top[0])


def strip_tail(text: str, words: list, commands=("done", "stop")) -> tuple[str, list]:
    """Cut a spoken end phrase off the end of a transcript, so it is not scored or counted as speech."""
    m = match(text, commands, tail=True)
    if not m:
        return text, words

    def cut(items: list, as_text) -> list:
        items, left = list(items), m.n_words
        while items and left > 0:
            left -= len(normalize(as_text(items.pop())).split())
        return items

    rest, rest_words = cut(text.split(), str), cut(words, lambda w: w.text)
    if rest and normalize(rest[-1]) in {"and", "so", "at", "kaya"}:      # "..., and that's my answer"
        rest.pop()
        rest_words = rest_words[:-1]
    return " ".join(rest).rstrip(" ,;:-"), rest_words


@dataclass
class Heard:
    text: str
    command: str | None = None   # None = speech was heard but it matched nothing on screen
    lang: str | None = None


class VoiceListener:
    """Hears one short utterance at a time from the mic and matches it to the allowed commands."""

    def __init__(self, cfg, mic, vad, asr, face=None):
        v = cfg.get("voice") or {}
        self.mic, self.asr, self.face = mic, asr, face
        self.segmenter = Segmenter(vad, cfg.audio.min_speech_ms, v.get("command_silence_ms", 500),
                                   v.get("command_max_s", 6))
        self.no_speech_max = v.get("no_speech_prob_max", 0.6)
        self.logprob_min = v.get("avg_logprob_min", -1.2)

    def open(self) -> None:
        """Start listening now: drops what the mic heard before (the robot's own voice)."""
        self.mic.clear()
        self.segmenter.reset()

    def poll(self, commands, numbered: bool = False, timeout: float = 0.1) -> Heard | None:
        """Read one mic block. Returns a Heard when an utterance just ended, else None."""
        block = self.mic.read(timeout=timeout)
        if block is None:
            return None
        if self.face:
            self.face.hear(block)
        for seg in self.segmenter.process(block):
            heard = self.recognize(seg.audio, commands, numbered)
            if heard:
                return heard
        return None

    def recognize(self, audio, commands, numbered: bool = False) -> Heard | None:
        """Transcribe with language auto-detect; if nothing matches, try the other language once.
        No vocabulary prompt is used: a prompt full of command words makes Whisper invent them in noise."""
        first = None
        for lang in (None, "other"):
            if lang == "other":
                lang = "en" if first.language == "tl" else "tl"
            t = self.asr.transcribe(audio, language=lang, prompt="")
            if t.no_speech_prob > self.no_speech_max or t.avg_logprob < self.logprob_min or not normalize(t.text):
                if first is None:
                    return None            # noise, not speech: say nothing, keep listening
                continue
            first = first or t
            m = match(t.text, commands, numbered=numbered)
            if m:
                return Heard(t.text, m.command, m.lang)
        return Heard(first.text)


@dataclass
class Choice:
    command: str | None
    lang: str | None = None
    how: str = "touch"           # touch | voice | timeout | stop | tick


def wait_choice(face, listener: VoiceListener | None, kind: str, options: list[tuple[str, str]], *,
                aliases: dict[str, str] | None = None, also: tuple[str, ...] = (), timeout: float | None = None,
                retries: int = 2, tick=None, on_retry=None, on_lang=None) -> Choice:
    """Wait until the student picks one of `options` [(command, label)] by voice, tap, or key.

    aliases: extra spoken commands mapped onto an option, e.g. {"yes": "keep"}.
    also: other page events that pick the first option (Enter = "done" on the start screen).
    tick(): called every loop; returning True ends the wait (how="tick").
    on_retry(): called after an unclear utterance, up to `retries` times (the robot asks again).
    """
    server, aliases = face.server, aliases or {}
    names = [c for c, _ in options]
    by_label = {label: c for c, label in options}
    spoken = names + list(aliases) + ["stop"]
    end = None if timeout is None else time.monotonic() + timeout
    misses = 0

    def listen() -> None:
        if listener:
            listener.open()
            face.listen(kind)

    listen()
    try:
        while True:
            ev = server.next_event(timeout=0 if listener else 0.1)
            if ev:
                name, value = ev["name"], ev.get("value")
                if name == "stop":
                    return Choice(None, how="stop")
                if name == "lang" and on_lang:
                    on_lang(value)
                elif name in also:
                    return Choice(names[0])
                elif name == kind:
                    cmd = by_label.get(value) or next((c for c in names if str(value).lower() == c.lower()), None)
                    if cmd:
                        return Choice(cmd)
            if listener:
                heard = listener.poll(spoken, numbered=True)
                if heard and heard.command == "stop":
                    return Choice(None, heard.lang, "stop")
                if heard and heard.command:
                    cmd = aliases.get(heard.command, heard.command)
                    face.heard(heard.text, next(label for c, label in options if c == cmd))
                    if face.play_audio:
                        time.sleep(0.7)        # let the student see which button was picked
                    return Choice(cmd, heard.lang, "voice")
                if heard:
                    misses += 1
                    face.heard(heard.text, None)
                    if on_retry and misses <= retries:
                        on_retry()
                        listen()
            if tick and tick():
                return Choice(None, how="tick")
            if end is not None and time.monotonic() >= end:
                return Choice(None, how="timeout")
    finally:
        if listener:
            face.listen("none")
