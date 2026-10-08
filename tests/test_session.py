"""T1.7: a full simulated session end to end, plus declining consent and pressing Stop."""
import pytest

from coach.face.server import FaceServer
from coach.session import report as rpt
from eval.simulate_session import load_models, make_answers, run_simulated


@pytest.fixture(scope="module")
def env(cfg, speak):
    server = FaceServer(ws_port=8885).start()
    yield cfg, load_models(cfg), make_answers(cfg), server
    server.stop()


def test_full_session_produces_report(env, tmp_path):
    cfg, models, answers, server = env
    result = run_simulated(cfg, answers, server, models)
    rep = rpt.build_report(result)
    assert result["meta"]["consented"] and not result["meta"]["ended_early"]
    assert rep["summary"]["answers"] == 5
    assert all("score" in a for a in rep["answers"]), [a.get("error") for a in rep["answers"]]
    assert rep["summary"]["top_corrections"]                       # at least one specific correction
    by_id = {a["question_id"]: a for a in rep["answers"]}
    assert by_id["cpe.project"]["score"]["score"] < by_id["common.intro"]["score"]["score"]  # short answer is weakest
    assert "attendance" in by_id["common.intro"]["transcript"].lower()
    assert by_id["common.intro"]["end_reason"] == "silence"
    html = rpt.save(rep, tmp_path)
    assert html.exists() and "unapproved sample rubric" in html.read_text(encoding="utf-8")


def test_declining_consent_ends_without_answers(env):
    cfg, models, answers, server = env
    result = run_simulated(cfg, answers, server, models, consent="No")
    assert not result["meta"]["consented"] and result["answers"] == []


def test_stop_ends_session_early(env):
    cfg, models, answers, server = env
    result = run_simulated(cfg, answers, server, models, stop_after_panels=1)  # stop at program choice
    assert result["meta"]["ended_early"] and result["answers"] == []
