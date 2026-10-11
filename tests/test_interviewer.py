"""Generative interviewer: names, session memory, guard v2, AI-written questions with fallback, praise,
coaching summary, and a full simulated interview with the real local model (skipped if not downloaded)."""
import time

import pytest

from coach.interviewer.guard import check_turn
from coach.interviewer.memory import SessionMemory, extract_name, grounded
from coach.scoring.rubric import load_bank
from coach.session.state_machine import Answer

from .test_reply import LiveSpeech, _probed
from .test_wait_for_human import _session


# ---------- the student's name ----------

@pytest.mark.parametrize("said, name", [
    ("My name is Leonard.", "Leonard"), ("I'm Leonard po.", "Leonard"), ("Leonard.", "Leonard"),
    ("Ako si Leonard po.", "Leonard"), ("Call me Leo.", "Leo"), ("Hi, I am John Paul.", "John Paul"),
    ("Pangalan ko po ay Maria.", "Maria"), ("Yes.", None), ("Um.", None),
    ("I think I would like to start with my project first today.", None),
])
def test_first_name_is_extracted(said, name):
    assert extract_name(said) == name


# ---------- session memory ----------

INTRO = ("Good morning. I am a fourth year Computer Engineering student. For our capstone I built an attendance "
         "system with an ESP32, and I am applying because your team builds embedded products.")


def test_memory_keeps_topics_feelings_and_only_grounded_ai_facts():
    m = SessionMemory()
    m.add_answer(1, "Self-introduction", "Sorry, I'm nervous. " + INTRO)
    assert "ESP32" in m.topics() and any(f.kind == "feeling" for f in m.facts)
    kept = m.add_ai_facts(1, ["- built an attendance system with an ESP32 for capstone",
                              "- won first place in a national robotics competition"])    # never said
    assert [f.text for f in kept] == ["built an attendance system with an ESP32 for capstone"]
    assert "attendance system" in m.notes() and "robotics competition" not in m.notes()


def test_memory_offers_each_earlier_fact_once_and_spaces_out_the_name():
    m = SessionMemory(name="Leonard", name_every=2)
    m.add_answer(1, "Self-introduction", INTRO)
    m.add_ai_facts(1, ["- built an attendance system with an ESP32 for capstone"])
    first = m.callback_candidates(2)[0]
    assert first.kind == "ai"
    m.used.add(first.text)
    assert first not in m.callback_candidates(3)
    assert m.may_use_name()
    m.next_turn(used_name=True)
    assert not m.may_use_name()          # not again on the very next turn
    m.next_turn(used_name=False)
    m.next_turn(used_name=False)
    assert m.may_use_name()


def test_forget_clears_everything():
    m = SessionMemory(name="Leonard")
    m.add_answer(1, "intro", INTRO)
    m.forget()
    assert m.name is None and not m.facts and not m.said and m.notes() == "- (nothing yet)"


def test_misheard_terms_are_replaced_before_the_ai_sees_them():
    from coach.reply.topics import canonicalize

    assert canonicalize("I built it with an ESB 32 and an esb32.") == "I built it with an ESP32 and an ESP32."
    assert canonicalize("We used Key Cad and the R F I D reader.") == "We used KiCad and the RFID reader."
    assert canonicalize("the sensor and the espresso") == "the sensor and the espresso"


def test_grounded():
    assert grounded("built an attendance system", INTRO)
    assert not grounded("designed a drone with GPS", INTRO)


# ---------- guard v2 ----------

MEM = "fourth year Computer Engineering student built an attendance system with an ESP32 for capstone"
GOAL = "a real time they worked in a team and faced a disagreement: what happened, what they did, and the result"
CTX = MEM + " " + GOAL + " Tell me about a time you worked in a team and faced a disagreement. What did you do?"


def test_guard_accepts_a_personalized_question():
    ok, why = check_turn("Thanks, Leonard. Earlier you mentioned your ESP32 attendance system for capstone. Did your "
                         "team ever disagree while building it, and what did you do?", "question", CTX, "Leonard",
                         memory=MEM)
    assert ok, why


@pytest.mark.parametrize("text, problem", [
    ("That's an excellent project. Did your team ever disagree?", "judging"),
    ("Earlier you mentioned winning a robotics competition. Did your team disagree?", "claims the student said"),
    ("Did Maria disagree with you while building it?", "unknown name"),
    ("Did your team disagree about the Raspberry Pi you used?", "new technical term"),
    ("How old are you, and did your team ever disagree?", "private topic"),
    ("In our capstone we disagreed. What did you do?", "speaks as the student"),
    ("Earlier you mentioned your ESP32 attendance system.", "does not ask a question"),
    ("Did your team disagree in 2019?", "new number"),
])
def test_guard_rejects(text, problem):
    ok, why = check_turn(text, "question", CTX, None, memory=MEM)
    assert not ok and problem in why, why


def test_name_is_rejected_when_not_allowed_this_turn():
    ok, why = check_turn("Thanks, Leonard. Did your team ever disagree?", "question", CTX, None, memory=MEM)
    assert not ok and "unknown name 'Leonard'" in why


def test_coach_may_praise_only_without_negative_words():
    ctx = "Leonard Technical depth: you explained the technology you used Say which part you built"
    assert check_turn("Leonard, you explained the technology you used very clearly. Next, say which part you built.",
                      "coach", ctx, "Leonard", allow_praise=True)[0]
    ok, why = check_turn("Leonard, your answer was poor. Say which part you built.", "coach", ctx, "Leonard",
                         allow_praise=True)
    assert not ok and "negative" in why


# ---------- AI-written questions in the session (fake AI) ----------

class FakeInterviewer:
    def __init__(self, question="", follow_up="", delay=0.0, coach="", facts=()):
        self.q, self.fu, self.delay, self.c, self.f = question, follow_up, delay, coach, list(facts)
        self.last_seconds = 0.0
        self.calls = []

    def question(self, **kw):
        self.calls.append(("question", kw))
        time.sleep(self.delay)
        return self.q.format(**{k: v or "" for k, v in kw.items() if isinstance(v, (str, type(None)))})

    def follow_up(self, **kw):
        self.calls.append(("follow_up", kw))
        return self.fu

    def facts(self, said):
        return self.f

    def coach(self, **kw):
        return self.c


def _gen_session(cfg, ai, name=None):
    s = _session(cfg)
    s.speech, s.interviewer, s.knowledge = LiveSpeech(), ai, None
    s.icfg = dict(s.icfg, mode="generative", question_timeout_s=0.5)
    s.memory.name = name
    return s


def _q(cfg, qid):
    bank = load_bank(cfg, "CpE")
    return next(q for q in bank.questions if q.id == qid)


def test_question_written_by_ai_keeps_the_rubric_and_uses_the_callback(cfg):
    ai = FakeInterviewer("Thanks, Leonard. Earlier you mentioned that you {callback}. Did your team ever disagree "
                         "while building it, and what did you do?")
    s = _gen_session(cfg, ai, name="Leonard")
    s.memory.add_answer(1, "Self-introduction", INTRO)
    s.memory.add_ai_facts(1, ["- built an attendance system with an ESP32 for capstone"])
    q = _q(cfg, "common.teamwork")
    asked, speech, text, info = s._generated_question(2, q)
    assert info["source"] == "ai", info
    assert "built an attendance system with an ESP32 for capstone" in text and speech == ("live", text)
    assert asked.id == q.id and asked.criteria == q.criteria and asked.text["en"] == text
    assert info["callback"] in s.memory.used          # never brought up again
    kw = ai.calls[0][1]
    assert kw["name"] == "Leonard" and kw["goal"].startswith("a real time")


def test_bad_ai_question_falls_back_to_the_rubric_question(cfg):
    s = _gen_session(cfg, FakeInterviewer("Great answer! How old are you?"))
    q = _q(cfg, "common.teamwork")
    asked, speech, text, info = s._generated_question(2, q)
    assert info["source"] == "rubric" and text == q.text["en"] and "judging" in info["check"]


def test_slow_ai_says_let_me_think_then_falls_back(cfg):
    s = _gen_session(cfg, FakeInterviewer("Did your team ever disagree?", delay=2.0))
    q = _q(cfg, "common.teamwork")
    t0 = time.monotonic()
    asked, speech, text, info = s._generated_question(2, q)
    assert info["source"] == "rubric" and info["check"] == "ai too slow" and time.monotonic() - t0 < 2.0
    assert ("gen.think_1", "en") in s.face.said


def test_praise_only_when_the_scorer_rated_a_criterion_high(cfg):
    q = _q(cfg, "cpe.project")
    s = _gen_session(cfg, FakeInterviewer(), name="Leonard")
    assert s._maybe_praise(_probed(q, "We built an ESP32 water meter that sends data over MQTT. " * 3, 3.5)) is None
    p = s._maybe_praise(_probed(q, "We built an ESP32 water meter that sends data over MQTT. " * 3, 4.6))
    assert p and "Leonard" in p["said"] and p["probe_score"] == 4.6
    assert ("live", p["said"]) in s.face.said


def test_ai_follow_up_used_when_it_passes_otherwise_the_approved_line(cfg):
    q = _q(cfg, "cpe.project")
    text = "We made a water monitor with Arduino and some sensors and a website."
    good = FakeInterviewer(follow_up="So you made a water monitor with an Arduino. What made you pick the Arduino?")
    s = _gen_session(cfg, good)
    ans = _probed(q, text, 3.4)
    plan = s._plan_reply(q, ans)
    out, speech, source, note = s._generative_follow_up(plan, q, ans)
    assert source == "ai" and "water monitor" in out
    bad = FakeInterviewer(follow_up="So you won a national award with the Arduino. Why did you pick it?")
    s = _gen_session(cfg, bad)
    ans = _probed(q, text, 3.4)
    plan = s._plan_reply(q, ans)
    out, speech, source, note = s._generative_follow_up(plan, q, ans)
    assert source != "ai" and out == plan["approved"] and "rejected" in note


def test_closing_uses_the_name_and_a_remembered_topic(cfg):
    s = _gen_session(cfg, FakeInterviewer(), name="Leonard")
    s.memory.add_answer(1, "intro", INTRO)
    assert s._closing_text() == "That was the last question. Thank you, Leonard! I enjoyed hearing about the ESP32."


def test_coach_summary_falls_back_when_the_ai_text_fails(cfg):
    q = _q(cfg, "cpe.project")
    a = Answer(q)
    a.result = {"score": {"score": 3.0, "criteria": [
        {"criterion_id": "technical_depth", "criterion": "Technical depth", "expected": 4.4},
        {"criterion_id": "role", "criterion": "Own role and result", "expected": 2.1}],
        "correction": {"text": "Say exactly which part you built."}}, "delivery": {"tips": []}}
    s = _gen_session(cfg, FakeInterviewer(coach="Your answer was poor and wrong."), name="Leonard")
    out = s._coach_summary([a])
    assert out["source"] == "template" and out["strength"] == "Technical depth"
    assert out["text"].startswith("Leonard, your strongest area was technical depth.")
    s = _gen_session(cfg, FakeInterviewer(coach="Leonard, you explained the technology you used and why you chose "
                                                "it. Next, say exactly which part you built."), name="Leonard")
    assert s._coach_summary([a])["source"] == "ai"


# ---------- full simulated interview with the real local model ----------

@pytest.fixture(scope="module")
def gen_env(cfg):
    from coach.profile import models_dir

    path = models_dir(cfg) / "llm" / cfg.registry["llm"][cfg.reply.llm_model]["file"]
    if not path.exists():
        pytest.skip("local AI model not downloaded")
    from coach.face.server import FaceServer
    from eval.simulate_session import load_models

    server = FaceServer(ws_port=8899).start()
    yield load_models(cfg, reply=True, interviewer=True), server
    server.stop()


def test_full_generative_interview_remembers_the_student(cfg, gen_env, speak):
    from coach.session import report as rpt
    from eval.simulate_session import ANSWERS, run_simulated

    models, server = gen_env
    answers = [speak(a) for a in ANSWERS]
    spoken = {"consent": speak("Yes."), "name_ok": speak("Yes."), "keep": speak("Delete it.")}
    result = run_simulated(cfg, answers, server, models, spoken=spoken, keep=False,
                           name=speak("My name is Leonard."), followups=[speak(
                               "I programmed the ESP32 to read the RFID cards and send them to a database.")] * 5)
    meta, memory = result["meta"], result["memory"]
    assert meta["mode"] == "generative" and meta.get("name_confirmed") is True
    assert memory.name == "Leonard" and "ESP32" in memory.topics()
    assert len(result["answers"]) == 5 and all(a.result and "score" in a.result for a in result["answers"])
    infos = [a.question_info for a in result["answers"]]
    assert all(i is not None for i in infos)
    assert sum(i["source"] == "ai" for i in infos) >= 3, [(i["source"], i.get("check")) for i in infos]
    # every AI-written question passed the guard against what the student really said
    for a in result["answers"]:
        if a.question_info["source"] == "ai":
            assert a.question.text["en"].endswith(("?", "."))
            assert "ESB" not in a.question.text["en"]          # a mishearing is never repeated
    later = " ".join(a.question.text["en"] for a in result["answers"][1:])
    assert "Leonard" in later or any(i.get("callback") for i in infos[1:]), later
    assert meta["coach_summary"]["text"]
    report = rpt.build_report(result)
    html = rpt.render_html(report)
    assert "Your coach says" in html and "Leonard" in html
    memory.forget()
    assert memory.name is None and not memory.facts


# ---------- training data (G3): same format and same guard as the robot ----------

def test_synthetic_records_use_the_runtime_format_and_guard():
    from coach.interviewer.interviewer import build_messages, question_prompt
    from training.interviewer.dataset import make_record, validate

    user = question_prompt(goal=GOAL, example="Tell me about a time you worked in a team and faced a disagreement.",
                           notes="- Q1 (intro): " + MEM, just_said="I want to learn embedded products.",
                           callback="built an attendance system with an ESP32 for capstone", name="Leonard")
    good = make_record(user, "Thanks, Leonard. Earlier you mentioned your ESP32 attendance system for capstone. "
                             "Did your team ever disagree while building it, and what did you do?", "strong")
    assert good["messages"][:-1] == build_messages(user) and validate(good) == (True, "ok")
    bad = make_record(user, "Great job, Leonard! In our capstone we used a Raspberry Pi. Why?", "strong")
    assert not validate(bad)[0]
