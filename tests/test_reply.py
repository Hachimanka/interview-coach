"""Replies to what the student said: topics, feelings, laughter, local-AI rephrasing and its guard."""
import time

import pytest

from coach.reply.guard import check
from coach.reply.topics import feeling, find_topics
from coach.scoring.rubric import load_bank
from coach.session.lines import LINES
from coach.session.state_machine import Answer

from .test_wait_for_human import Speech, _session


# ---------- understanding what was said ----------

def test_topics_use_canonical_names_even_when_misheard():
    names = [t.name for t in find_topics("For our capstone I built an attendance system with an ESB 32.")]
    assert names[0] == "ESP32" and "capstone project" in names       # hardware first, then the activity


def test_generic_topic_dropped_when_a_specific_one_was_said():
    names = [t.name for t in find_topics("I used an ultrasonic sensor and the sensors were noisy.")]
    assert "ultrasonic sensor" in names and "sensors" not in names


@pytest.mark.parametrize("said, kind", [
    ("Sorry, I am nervous.", "nervous"), ("I don't know.", "stuck"),
    ("Kinakabahan po ako.", "nervous"), ("Hindi ko po alam.", "stuck"),
    ("I told the motor to stop.", None), ("We tested both sensors.", None),
])
def test_feelings(said, kind):
    assert feeling(said) == kind


# ---------- the guard on AI-written replies ----------

APPROVED = "You mentioned the Arduino. Why did you choose it for that work?"
SAID = "We made a water monitor with Arduino and some sensors."


def test_guard_accepts_a_faithful_rephrase():
    ok, why = check("You mentioned using an Arduino for a water monitor. Why did you choose it?", APPROVED, SAID, "Arduino")
    assert ok, why


@pytest.mark.parametrize("reply, problem", [
    ("Great job with the Arduino water monitor! Why did you choose it?", "judging"),
    ("In our project we used Arduino. Why did you choose it?", "speaks as the student"),
    ("You mentioned the Arduino for a water monitor.", "lost the approved question"),
    ("You used 3 Arduino boards for the water monitor. Why?", "new number"),
    ("You used the Arduino to measure voltage resistance humidity pressure. Why did you choose it?", "adds words"),
    ("You mentioned a water monitor. Why did you choose that board?", "dropped the topic"),
])
def test_guard_rejects(reply, problem):
    ok, why = check(reply, APPROVED, SAID, "Arduino")
    assert not ok and problem in why, why


# ---------- choosing the reply ----------

def _question(cfg, qid="cpe.project"):
    return next(q for q in load_bank(cfg, "CpE").questions if q.id == qid)


def _probed(q, text, expected):
    a = Answer(q)
    a.probe = {"words": len(text.split()), "text": text, "score": {"too_short": False, "criteria": [
        {"criterion_id": c.id, "criterion": c.name, "expected": expected} for c in q.criteria]}}
    a.probe_done.set()
    return a


class LiveSpeech(Speech):
    def live(self, text, lang="en"):
        return ("live", text)


def _reply_session(cfg, rephraser=None):
    s = _session(cfg)
    s.speech, s.rephraser = LiveSpeech(), rephraser
    return s


def test_topic_follow_up_when_the_answer_is_not_yet_strong(cfg):
    s, q = _reply_session(cfg), _question(cfg)
    plan = s._plan_reply(q, _probed(q, "We made a water monitor with Arduino and some sensors and a website.", 3.4))
    assert plan["kind"] == "topic_follow_up" and plan["topic"].name == "Arduino" and plan["asks"]
    assert plan["approved"] == "You mentioned the Arduino. Why did you choose it for that work?"


def test_strong_answer_with_a_topic_only_gets_an_acknowledgment(cfg):
    s, q = _reply_session(cfg), _question(cfg)
    text = ("For our capstone I wrote the ESP32 firmware that reads the flow sensor and sends data over MQTT, "
            "and I added a filter because the readings were noisy.")
    plan = s._plan_reply(q, _probed(q, text, 4.8))
    assert plan["kind"] == "topic_ack" and not plan["asks"]


def test_same_topic_is_never_mentioned_twice(cfg):
    s, q = _reply_session(cfg), _question(cfg)
    text = "We made a water monitor with Arduino and some sensors and a website."
    assert s._plan_reply(q, _probed(q, text, 3.4))["topic"].name == "Arduino"
    second = s._plan_reply(q, _probed(q, text, 3.4))["topic"]
    assert second is None or second.name != "Arduino"


def test_filipino_session_uses_approved_replies_without_topics(cfg):
    s, q = _reply_session(cfg), _question(cfg)
    s._lang_from_command("fil")
    plan = s._plan_reply(q, _probed(q, "Gumawa kami ng water monitor gamit ang Arduino.", 2.0))
    assert plan["topic"] is None and plan["key"]              # a pre-recorded approved clip


def test_nervous_student_is_reassured(cfg):
    s, q = _reply_session(cfg), _question(cfg)
    plan = s._plan_reply(q, _probed(q, "Sorry, I am nervous. We made a water monitor with Arduino.", 3.0))
    assert plan["feeling"] == "nervous" and LINES["reassure"]["en"]


# ---------- AI rephrasing with fallback ----------

class FakeAI:
    def __init__(self, out, delay=0.0):
        self.out, self.delay = out, delay

    def rephrase(self, approved, said, topic=None):
        time.sleep(self.delay)
        return self.out


def _plan(s, cfg):
    q = _question(cfg)
    ans = _probed(q, "We made a water monitor with Arduino and some sensors and a website.", 3.4)
    return s._plan_reply(q, ans), ans


def test_ai_reply_is_used_when_it_passes_the_guard(cfg):
    s = _reply_session(cfg, FakeAI("You mentioned using an Arduino for a water monitor. Why did you choose it?"))
    plan, ans = _plan(s, cfg)
    text, speech, source, note = s._realize(plan, ans)
    assert source == "ai" and "water monitor" in text and speech == ("live", text)


def test_judging_ai_reply_falls_back_to_the_approved_line(cfg):
    s = _reply_session(cfg, FakeAI("Great project! Why did you choose the Arduino?"))
    plan, ans = _plan(s, cfg)
    text, _, source, note = s._realize(plan, ans)
    assert source == "template" and text == plan["approved"] and "judging" in note


def test_slow_ai_falls_back_in_time(cfg):
    s = _reply_session(cfg, FakeAI("You mentioned the Arduino. Why did you choose it?", delay=2.0))
    s.cfg = type(s.cfg)(dict(s.cfg, reply=dict(s.cfg["reply"], llm_timeout_s=0.2)))
    plan, ans = _plan(s, cfg)
    t0 = time.monotonic()
    text, _, source, note = s._realize(plan, ans)
    assert source == "template" and note == "ai too slow" and time.monotonic() - t0 < 1.0


# ---------- laughter: mirror only, once per session ----------

class FakeLaughter:
    threshold = 0.3

    def probability(self, audio):
        return 0.9


def test_laughter_is_mirrored_only_once(cfg):
    s = _reply_session(cfg)
    s.laughter = FakeLaughter()
    a = Answer(None)
    a.laugh_prob = 0.9
    assert s._student_laughed(a)
    s._laughed = True                       # set by _maybe_follow_up after the robot laughed
    assert not s._student_laughed(a)


def test_no_laugh_without_the_student_laughing(cfg):
    s = _reply_session(cfg)
    s.laughter = FakeLaughter()
    assert not s._student_laughed(Answer(None))      # laugh_prob 0.0


def test_amused_face_exists_and_is_positive():
    from pathlib import Path

    from coach.face import controller as ctl

    page = (Path(ctl.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    assert "amused" in ctl.STATES and "amused:" in page
    assert not {"sad", "angry", "disappointed", "frown", "mocking"} & set(ctl.STATES)


# ---------- the real local model (skipped if not downloaded) ----------

def test_real_local_ai_rephrases_within_the_time_limit(cfg):
    from coach.profile import make_rephraser, models_dir

    path = models_dir(cfg) / "llm" / cfg.registry["llm"][cfg.reply.llm_model]["file"]
    if not path.exists():
        pytest.skip("local AI model not downloaded")
    ai = make_rephraser(cfg)
    out = ai.rephrase(APPROVED, SAID, "Arduino")
    ok, why = check(out, APPROVED, SAID, "Arduino")
    assert ok, f"{why}: {out}"
    assert ai.last_seconds < cfg.reply.llm_timeout_s
