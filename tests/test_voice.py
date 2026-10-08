"""Voice commands and automatic language: matcher rules, the choice loop, and full spoken sessions."""
import queue

import numpy as np
import pytest

from coach.asr.base import Transcript, Word
from coach.face.server import FaceServer
from coach.session import voice
from coach.session.state_machine import Answer, Session
from coach.session.voice import Choice, Heard, match, strip_tail, wait_choice


# ---------- matcher (no models needed) ----------

@pytest.mark.parametrize("said, options, command, lang", [
    ("Yes.", ["yes", "no"], "yes", "en"),
    ("Oo po.", ["yes", "no"], "yes", "fil"),
    ("Okay", ["yes", "no"], "yes", None),
    ("Hindi po ako pumapayag.", ["yes", "no"], "no", "fil"),
    ("I do not agree", ["yes", "no"], "no", "en"),
    ("Simulan na natin!", ["start"], "start", "fil"),
    ("Electronics engineering", ["CpE", "EE", "ECE"], "ECE", None),
    ("Electrical.", ["CpE", "EE", "ECE"], "EE", None),
    ("Number two", ["CpE", "EE", "ECE"], "EE", None),
    ("Isa po", ["CpE", "EE", "ECE"], "CpE", None),
    ("Burahin mo", ["keep", "delete"], "delete", "fil"),
])
def test_choice_commands(said, options, command, lang):
    m = match(said, options, numbered=True)
    assert m and (m.command, m.lang) == (command, lang)


@pytest.mark.parametrize("said, options", [
    ("yes no", ["yes", "no"]),                                   # two answers at once = unclear
    ("I know the answer", ["yes", "no"]),                        # "know" is not "no"
    ("well I was thinking about what you said before yes", ["yes", "no"]),   # chatter, not a command
    ("Yes", ["CpE", "EE", "ECE"]),                               # only what is on screen counts
    ("", ["yes", "no"]),
])
def test_unclear_speech_is_not_a_choice(said, options):
    assert match(said, options, numbered=True) is None


@pytest.mark.parametrize("said, command", [
    ("and after the fix it worked. That's my answer.", "done"),
    ("Iyon na po ang sagot ko.", "done"),
    ("Tapos na po.", "done"),
    ("Done.", "done"),
    ("I do not want to continue, stop the interview", "stop"),
])
def test_end_phrases_at_the_end_of_an_answer(said, command):
    assert match(said, ("done", "stop"), tail=True).command == command


@pytest.mark.parametrize("said", [
    "so I fixed it and the job was done",          # "done" alone counts only as the whole utterance
    "That's my answer to the first part, and then we tested it",   # not at the end
    "I told the motor to stop",                    # stop needs "stop the interview"
    "yes",
])
def test_normal_answer_text_does_not_end_the_answer(said):
    assert match(said, ("done", "stop"), tail=True) is None


def test_end_phrase_is_cut_from_the_transcript():
    said = "We tested both sensors, and that's my answer."
    words = [Word(w, i, i + 1) for i, w in enumerate(said.split())]
    text, kept = strip_tail(said, words)
    assert text == "We tested both sensors"
    assert [w.text for w in kept] == ["We", "tested", "both", "sensors,"]
    assert strip_tail("We tested both sensors.", words[:4])[0] == "We tested both sensors."


def test_every_command_has_a_screen_hint_in_both_languages():
    for cmd in voice.COMMANDS:
        assert set(voice.HINTS[cmd]) == {"en", "fil"}
        for lang in ("en", "fil"):      # what the screen says to say must itself be understood
            assert match(voice.HINTS[cmd][lang], [cmd], tail=cmd in ("done", "stop")).command == cmd


# ---------- the choice loop, with a scripted listener (no models needed) ----------

class StubServer:
    def __init__(self):
        self.events = queue.Queue()

    def next_event(self, timeout=None):
        try:
            return self.events.get(timeout=timeout or 0)
        except queue.Empty:
            return None


class StubFace:
    play_audio = False

    def __init__(self):
        self.server, self.log = StubServer(), []

    def listen(self, kind):
        self.log.append(("listen", kind))

    def heard(self, text, picked):
        self.log.append(("heard", text, picked))


class StubListener:
    def __init__(self, *heard):
        self.heard, self.opened = list(heard), 0

    def open(self):
        self.opened += 1

    def poll(self, commands, numbered=False, timeout=0.1):
        return self.heard.pop(0) if self.heard else None


YES_NO = [("yes", "Yes / Oo"), ("no", "No / Hindi")]


def test_spoken_choice_picks_the_button_and_reports_language():
    face = StubFace()
    c = wait_choice(face, StubListener(Heard("Oo po", "yes", "fil")), "consent", YES_NO)
    assert (c.command, c.lang, c.how) == ("yes", "fil", "voice")
    assert ("heard", "Oo po", "Yes / Oo") in face.log and face.log[-1] == ("listen", "none")


def test_unclear_speech_asks_again_then_accepts():
    face, retried = StubFace(), []
    listener = StubListener(Heard("uh what"), Heard("mmm"), Heard("hmm"), Heard("No", "no", "en"))
    c = wait_choice(face, listener, "consent", YES_NO, retries=2, on_retry=lambda: retried.append(1))
    assert c.command == "no" and len(retried) == 2          # asked again twice, then just kept listening
    assert listener.opened == 3                            # mic reopened after each spoken retry


def test_nobody_answers_is_a_timeout_never_a_yes():
    c = wait_choice(StubFace(), StubListener(), "consent", YES_NO, timeout=0.05)
    assert c.command is None and c.how == "timeout"


def test_tap_key_and_spoken_stop_still_work():
    face = StubFace()
    face.server.events.put({"type": "event", "name": "consent", "value": "Yes / Oo"})
    assert wait_choice(face, StubListener(), "consent", YES_NO).command == "yes"
    face.server.events.put({"type": "event", "name": "stop"})
    assert wait_choice(face, StubListener(), "consent", YES_NO).how == "stop"
    assert wait_choice(face, StubListener(Heard("stop the interview", "stop", "en")), "consent", YES_NO).how == "stop"
    c = wait_choice(face, StubListener(Heard("yes", "yes", "en")), "consent",
                    [("keep", "Keep it"), ("delete", "Delete it")], aliases={"yes": "keep", "no": "delete"})
    assert c.command == "keep"


# ---------- language follows the student (no models needed) ----------

class StubVad:
    sample_rate, threshold = 16000, 0.5

    def reset(self):
        pass


def _session(cfg):
    return Session(cfg, StubFace(), None, StubVad(), None, None, None, None)


def _answer(lang, prob):
    a = Answer(None)
    a.language, a.language_prob = lang, prob
    a.segments.append((None, Transcript("x", lang)))
    return a


def test_first_command_sets_the_language(cfg):
    s = _session(cfg)
    assert s.lang is None and s._l == "en"
    s._lang_from_command("fil")
    assert s.lang == "fil"
    s._lang_from_command("en")            # "yes" / "start" are also said by Filipino speakers
    assert s.lang == "fil"
    s2 = _session(cfg)
    s2._lang_from_command("en")
    assert s2.lang == "en"
    s2._lang_from_command("fil")          # "oo" / "simulan" always mean Filipino
    assert s2.lang == "fil"


def test_language_follows_each_answer_only_when_sure(cfg):
    s = _session(cfg)
    s._lang_from_answer(_answer("tl", 0.4))     # nothing known yet: take the best guess
    assert s.lang == "fil"
    s._lang_from_answer(_answer("en", 0.5))     # not sure enough to switch
    assert s.lang == "fil"
    s._lang_from_answer(_answer("en", 0.9))     # the student switched to English
    assert s.lang == "en"
    s._lang_from_answer(_answer("tl", 0.95))    # and back (Taglish follows the dominant language)
    assert s.lang == "fil"


def test_operator_key_locks_the_language(cfg):
    s = _session(cfg)
    s._set_lang("en", "operator")
    s._lang_from_answer(_answer("tl", 0.99))
    s._lang_from_command("fil")
    assert s.lang == "en"


def test_filipino_falls_back_to_english_when_clips_are_missing(cfg):
    class NoFilipino:
        def available(self, key, lang, text):
            return lang == "en"

        def question(self, key, lang, text):
            return (key, lang)

    s = _session(cfg)
    s.speech, s.lang = NoFilipino(), "fil"
    speech, text, lang = s._clip("line.next", {"en": "Next question.", "fil": "Susunod na tanong."})
    assert (lang, text, speech) == ("en", "Next question.", ("line.next", "en"))


def _ended_answer(s, reason, said, command):
    class IdleWorker:
        error = None

    s._worker = IdleWorker()
    a = Answer(None)
    a.end_reason, a.queued = reason, 1
    if command == "quick":            # found by the quick check of the last few seconds
        a.spotted = (1, match(said, ("done", "stop"), tail=True))
    else:                             # found in the full transcript
        a.segments.append((None, Transcript(said, "en")))
    return a


@pytest.mark.parametrize("how", ["quick", "full"])
def test_last_words_after_a_silence_ending(cfg, how):
    from coach.session.state_machine import SessionStopped

    s = _session(cfg)
    a = _ended_answer(s, "silence", "and it worked. That's my answer.", how)
    s._last_words(a, wait_s=0.1)
    assert a.end_reason == "voice_done"
    a = _ended_answer(s, "silence", "and then it worked.", how)
    s._last_words(a, wait_s=0.1)
    assert a.end_reason == "silence"
    with pytest.raises(SessionStopped):
        s._last_words(_ended_answer(s, "time_limit", "I am nervous, stop the interview.", how), wait_s=0.1)


# ---------- full spoken sessions (need the models) ----------

@pytest.fixture(scope="module")
def env(cfg, speak):
    from eval.simulate_session import load_models

    server = FaceServer(ws_port=8865).start()
    yield cfg, load_models(cfg), server
    server.stop()


def test_whole_session_by_voice_no_touch(env, speak):
    from eval.simulate_session import ANSWERS, run_simulated

    cfg, models, server = env
    answers = [speak(a + " That's my answer.") for a in ANSWERS]
    spoken = {"choice": speak("Start."), "consent": speak("Yes."), "program": speak("Electronics."),
              "keep": speak("Keep it.")}
    result = run_simulated(cfg, answers, server, models, spoken=spoken, wait=True, keep=True)
    meta = result["meta"]
    assert meta["consented"] and not meta["ended_early"] and meta["program"] == "ECE" and meta["kept"]
    assert meta["final_language"] == "en" and meta["language"] == "en"
    assert len(result["answers"]) == 5
    # The simulation runs faster than real time, so an answer can hit the silence rule before the
    # worker has transcribed its last words; test_last_words_* cover the end phrase itself.
    for a in result["answers"]:
        assert a.end_reason in ("voice_done", "silence"), a.end_reason
        assert "my answer" not in a.result["transcript"].lower()      # the command is never scored
    assert "attendance" in result["answers"][0].result["transcript"].lower()


def test_spoken_no_declines_consent(env, speak):
    from eval.simulate_session import run_simulated

    cfg, models, server = env
    result = run_simulated(cfg, [], server, models, spoken={"consent": speak("No.")})
    assert not result["meta"]["consented"] and result["answers"] == []


def test_spoken_stop_ends_the_session(env, speak):
    from eval.simulate_session import run_simulated

    cfg, models, server = env
    answers = [speak("I am a fourth year student. Stop the interview.")]
    result = run_simulated(cfg, answers, server, models, spoken={"consent": speak("Yes."), "program": speak("Computer.")})
    assert result["meta"]["ended_early"] and result["answers"] == []


def test_asr_detects_only_english_or_tagalog(cfg, speak):
    from coach.profile import make_asr

    t = make_asr(cfg).transcribe(speak("Good morning, I am ready to start the interview."))
    assert t.language == "en" and t.language_probability > 0.8 and t.no_speech_prob < 0.6
