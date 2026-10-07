"""T1.6: scoring v0 orders answers correctly and every score is traceable to a criterion and anchor."""
import pytest

from coach.profile import make_embedder
from coach.scoring.rubric import load_bank
from coach.scoring.score import Scorer

ANSWERS = {
    "cpe.project": {
        "good": "For our capstone I built a greenhouse monitor. I wrote the ESP32 firmware that reads soil moisture and temperature sensors, sends the readings over WiFi using MQTT to a Raspberry Pi server, and I added a median filter because the moisture sensor was noisy. I was the firmware lead and the system cut water use by about twenty percent in our test.",
        "mid": "We made a monitoring system for plants using Arduino and some sensors, and it shows the data on a website. I did some of the coding.",
        "poor": "We did a project in school about plants, it was okay, we used a computer and it worked I think.",
    },
    "common.teamwork": {
        "good": "During our thesis two teammates disagreed on using Python or C for the firmware. I proposed that each write a small prototype in a day and we compared speed and memory. C was faster on the microcontroller, so we chose it and both agreed. I learned that testing settles disagreements better than arguing.",
        "mid": "In a group project we had a disagreement about the design. We discussed it and in the end we agreed on one design and finished on time.",
        "poor": "I never had a disagreement because I always get along with everyone in my group, so no problems.",
    },
}


@pytest.fixture(scope="module")
def setup(cfg):
    bank = load_bank(cfg, "CpE")
    return Scorer.from_config(cfg, make_embedder(cfg)), {q.id: q for q in bank.questions}, bank


@pytest.mark.parametrize("qid", list(ANSWERS))
def test_good_beats_mid_beats_poor(setup, qid):
    scorer, qs, _ = setup
    s = {k: scorer.score(qs[qid], v)["score"] for k, v in ANSWERS[qid].items()}
    assert s["good"] > s["mid"] > s["poor"], s


def test_score_is_traceable(setup):
    scorer, qs, _ = setup
    r = scorer.score(qs["cpe.project"], ANSWERS["cpe.project"]["mid"])
    for c in r["criteria"]:
        assert c["criterion_id"] and c["matched_anchor"] and c["matched_level"] in (1, 3, 5)
        assert 1 <= c["score"] <= 5 and set(c["level_similarity"]) == {"1", "3", "5"}
    assert r["correction"]["text"] and r["correction"]["criterion_id"]


def test_short_answer_scores_one(setup):
    scorer, qs, _ = setup
    r = scorer.score(qs["cpe.project"], "I don't know.")
    assert r["score"] == 1.0 and r["too_short"]


def test_taglish_answer_ranks_above_poor(setup):
    scorer, qs, _ = setup
    taglish = ("Sa capstone namin, gumawa ako ng greenhouse monitor gamit ang ESP32. Ako yung nag-program ng "
               "firmware para basahin yung soil moisture sensor at i-send sa Raspberry Pi through MQTT, "
               "tapos naglagay ako ng filter kasi maingay yung sensor.")
    good = scorer.score(qs["cpe.project"], taglish)["score"]
    poor = scorer.score(qs["cpe.project"], ANSWERS["cpe.project"]["poor"])["score"]
    assert good > poor


def test_sample_bank_is_flagged(setup):
    _, qs, bank = setup
    assert bank.is_sample and len(bank.questions) == 5
    assert bank.questions[0].bank == "common"
