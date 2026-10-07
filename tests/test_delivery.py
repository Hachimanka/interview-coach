"""T1.5: pace, pauses, fillers and tips."""
from coach.asr.base import Word
from coach.delivery.fillers import count_fillers
from coach.delivery.pace import words_per_minute
from coach.delivery.pauses import long_pauses, response_latency
from coach.delivery.summary import analyze


def _words(n, span):
    step = span / n
    return [Word("w", i * step, i * step + step * 0.8) for i in range(n)]


def test_wpm():
    assert 118 < words_per_minute(_words(60, 30.0)) < 122


def test_long_pauses_and_latency():
    spans = [(2.0, 8.0), (8.5, 12.0), (16.0, 20.0)]
    assert long_pauses(spans, 3.0) == [{"at_s": 12.0, "duration_s": 4.0}]
    assert response_latency(spans, 0.5) == 1.5


def test_hesitations_always_count():
    r = count_fillers("Um, I think, uh, the answer is, hmm, yes.")
    assert r["hesitations"] == 3


def test_markers_only_with_pause_or_repeat():
    assert count_fillers("Ano ang ginawa mo?")["markers"] == 0          # real question word
    assert count_fillers("Ginawa ko yung, ano, circuit.")["markers"] == 1
    assert count_fillers("It was like a timer.")["markers"] == 0
    assert count_fillers("It was, like, really hard, you know.")["markers"] == 2


def test_analyze_gives_tips(cfg):
    words = _words(100, 30.0)     # 200 wpm: too fast
    d = analyze("um uh um uh um uh", words, [(1.0, 31.0)], 0.0, cfg, eye_contact_pct=40.0)
    assert d["wpm"] > cfg.delivery.wpm_fast
    joined = " ".join(d["tips"])
    assert "Slow down" in joined and "filler" in joined and "looked at" in joined


def test_fillers_survive_whisper(cfg, speak):
    """Whisper tends to drop um/uh; the filler-primed prompt should keep some of them."""
    from coach.profile import make_asr

    t = make_asr(cfg).transcribe(speak("Um, I think, uh, the main problem was, um, the power supply."))
    assert count_fillers(t.text)["hesitations"] >= 1, t.text
