"""Scope: CpE and ECE only (EE removed), and the robot never asks the student's course."""
import random

from coach.scoring.rubric import all_questions, load_session_bank, pick_questions, programs
from coach.session.lines import LINES, TEXT
from coach.session.voice import COMMANDS, HINTS


def test_only_cpe_and_ece_are_in_scope(cfg, cfg_pi5):
    for c in (cfg, cfg_pi5):
        assert programs(c) == ["CpE", "ECE"]
    banks = {q.bank for q in all_questions(cfg)}
    assert banks == {"common", "CpE", "ECE"}
    assert not any(q.id.startswith("ee.") for q in all_questions(cfg))
    assert "EE" not in COMMANDS and "EE" not in HINTS


def test_the_robot_has_no_line_asking_for_the_course():
    assert "program" not in LINES and "program_title" not in TEXT


def test_every_session_mixes_both_courses(cfg):
    bank = load_session_bank(cfg)
    fixed = [q.id for q in pick_questions(bank, 5, vary=False)]
    assert fixed == ["common.intro", "common.teamwork", "common.weakness", "cpe.project", "ece.signal"]
    for seed in range(20):                       # with rotation: always one CpE and one ECE question
        qs = pick_questions(bank, 5, vary=True, rng=random.Random(seed))
        assert len(qs) == 5 and {q.bank for q in qs} == {"common", "CpE", "ECE"}


def test_rotation_varies_the_technical_questions(cfg):
    bank = load_session_bank(cfg)
    seen = {tuple(q.id for q in pick_questions(bank, 5, vary=True, rng=random.Random(s)) if q.bank != "common")
            for s in range(20)}
    assert len(seen) > 1                          # not the same technical pair every session


def test_simulated_session_never_shows_a_course_screen(cfg, speak):
    from coach.face.server import FaceServer
    from eval.simulate_session import ANSWERS, load_models, run_simulated

    server = FaceServer(ws_port=8871).start()
    panels = []
    send = server.send
    server.send = lambda m: (panels.append(m["kind"]) if m["type"] == "panel" else None, send(m))[1]
    try:
        result = run_simulated(cfg, [speak(a) for a in ANSWERS], server, load_models(cfg))
    finally:
        server.stop()
    assert "program" not in panels
    assert result["meta"]["program"] == "CpE & ECE"
    assert result["meta"]["questions"][-2:] == ["cpe.project", "ece.signal"]
