"""T1.3 + T1.10: question clips load fast and carry subtitle cues and lip-sync levels."""
import time

import pytest

from coach.audio.util import resample
from coach.face.subtitles import MAX_CHARS, make_cues
from coach.profile import make_asr, resolve
from coach.scoring.rubric import all_questions
from coach.tts.cache import SpeechCache


@pytest.fixture(scope="module")
def cache(cfg):
    c = SpeechCache(resolve(cfg, cfg.paths.cache_dir))
    if not c.has("common.intro", "en"):
        pytest.skip("cache empty: run python -m offline.synth_questions")
    return c


def test_every_question_cached_in_both_languages(cfg, cache):
    missing = [f"{q.id}.{l}" for q in all_questions(cfg) for l in ("en", "fil") if not cache.has(q.id, l)]
    assert not missing


def test_clip_loads_under_200ms(cache):
    t0 = time.perf_counter()
    cache.load("cpe.project", "en")
    assert time.perf_counter() - t0 < 0.2


def test_cues_and_mouth_cover_the_clip(cache):
    s = cache.load("cpe.project", "fil")
    assert s.cues[0].start == 0 and abs(s.cues[-1].end - s.duration) < 0.05
    assert all(len(line) <= MAX_CHARS for c in s.cues for line in c.text.split("\n"))
    assert abs(len(s.mouth) * s.frame_s - s.duration) < 0.1
    assert max(s.mouth) == 3 and min(s.mouth) == 0     # mouth opens and closes


def test_cues_split_long_text():
    cues = make_cues("word " * 40, 10.0)
    assert len(cues) > 1 and all(c.text.count("\n") <= 1 for c in cues)


def test_filipino_clip_is_intelligible(cfg, cache):
    """Round-trip: MMS Tagalog audio -> Whisper should recognise key words."""
    s = cache.load("common.intro", "fil")
    t = make_asr(cfg).transcribe(resample(s.audio, s.sample_rate), language="tl")
    text = t.text.lower()
    assert sum(w in text for w in ("pakilala", "sarili", "training", "apply")) >= 2, t.text
