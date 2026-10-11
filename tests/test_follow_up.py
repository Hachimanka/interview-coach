"""The robot replies to the answer with approved lines only: an acknowledgment and at most one follow-up."""
import pytest

from coach.face.server import FaceServer
from coach.scoring.rubric import load_bank
from coach.session.lines import LINES
from coach.session.state_machine import Answer

from .test_wait_for_human import _session


def _question(cfg, qid="cpe.project"):
    return next(q for q in load_bank(cfg, "CpE").questions if q.id == qid)


def _probed(q, words, scores):
    """An answer whose preliminary score is ready: scores = {criterion_id: expected}."""
    a = Answer(q)
    a.probe = {"words": words, "score": {"too_short": False, "criteria": [
        {"criterion_id": cid, "criterion": cid, "expected": v} for cid, v in scores.items()]}}
    a.probe_done.set()
    return a


def test_short_answer_gets_tell_me_more(cfg):
    s, q = _session(cfg), _question(cfg)
    key, texts, reason = s._pick_follow_up(q, _probed(q, 6, {"technical_depth": 2.0, "role": 2.0}))
    assert key == "line.tell_more" and texts == LINES["tell_more"] and "6 words" in reason


def test_weak_criterion_gets_its_approved_follow_up(cfg):
    s, q = _session(cfg), _question(cfg)
    key, texts, reason = s._pick_follow_up(q, _probed(q, 40, {"technical_depth": 4.6, "role": 2.4}))
    role = next(c for c in q.criteria if c.id == "role")
    assert key == "cpe.project.fu.role" and texts == role.follow_up
    assert "Own role" in reason and "2.4" in reason             # the report can say why it was asked


def test_good_answer_gets_no_follow_up(cfg):
    s, q = _session(cfg), _question(cfg)
    assert s._pick_follow_up(q, _probed(q, 50, {"technical_depth": 4.5, "role": 4.1})) is None


def test_no_follow_up_when_nothing_was_recognized_or_score_is_late(cfg):
    s, q = _session(cfg), _question(cfg)
    assert s._pick_follow_up(q, _probed(q, 0, {"technical_depth": 1.0, "role": 1.0})) is None
    late = Answer(q)                                           # probe never finished
    from coach.profile import Config

    s.cfg = Config(dict(s.cfg, session=dict(s.cfg["session"], follow_up_wait_s=0.05)))  # copy: cfg is shared
    assert s._pick_follow_up(q, late) is None


def test_every_rubric_criterion_has_an_approved_follow_up_in_both_languages(cfg):
    for program in ("CpE", "ECE"):
        for q in load_bank(cfg, program).questions:
            for c in q.criteria:
                assert c.follow_up and set(c.follow_up) >= {"en", "fil"}, (q.id, c.id)


def test_acknowledgments_are_neutral():
    for k in ("ack_1", "ack_2", "ack_3"):
        text = LINES[k]["en"].lower()
        assert not any(w in text for w in ("great", "excellent", "wrong", "bad", "good answer", "perfect"))


# ---------- full simulated session ----------

@pytest.fixture(scope="module")
def env(cfg, speak):
    from eval.simulate_session import load_models

    server = FaceServer(ws_port=8873).start()
    yield cfg, load_models(cfg), server
    server.stop()


def test_short_answer_triggers_one_follow_up_scored_with_the_answer(env, speak):
    from eval.simulate_session import ANSWERS, run_simulated

    cfg, models, server = env
    answers = [speak(a) for a in ANSWERS]          # ANSWERS[3] = "We made a project, it was okay." (short)
    extra = speak("I was the firmware lead. I wrote the ESP32 code and tested the sensors.")
    result = run_simulated(cfg, answers, server, models, followups=[extra])
    by_id = {a.question.id: a for a in result["answers"]}
    short = by_id["cpe.project"]
    fu = short.result["follow_up"]
    assert fu["key"] == "line.tell_more" and "short answer" in fu["reason"]
    assert "firmware" in fu["transcript"].lower()
    assert "firmware" not in short.result["transcript"].lower()          # main transcript kept separate
    assert len(result["answers"]) == 5                                    # a follow-up is not a new question
    asked = [a for a in result["answers"] if a.follow_up_info]
    assert len(asked) <= len(result["answers"])                           # at most one per question
    assert all(a.followup is not None for a in asked)
