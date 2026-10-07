"""T1.1 + T1.2: VAD segmentation and ASR on Piper-generated speech with a known pause."""
import re

import numpy as np

from coach.audio.util import blocks
from coach.audio.vad import Segmenter
from coach.profile import make_asr, make_vad

from .conftest import silence

S1 = "I designed a temperature monitoring system using an ESP32 microcontroller."
S2 = "The main challenge was reducing power consumption."


def _segments(cfg, audio):
    a = cfg.audio
    seg = Segmenter(make_vad(cfg), a.min_speech_ms, a.segment_silence_ms, a.max_segment_s)
    out = []
    for b in blocks(audio):
        out += seg.process(b)
    return out + seg.flush()


def test_vad_finds_two_segments_and_the_pause(cfg, speak):
    audio = np.concatenate([silence(1.0), speak(S1), silence(4.0), speak(S2), silence(1.0)])
    segs = _segments(cfg, audio)
    assert len(segs) == 2
    gap = segs[1].start - segs[0].end
    assert 3.0 < gap < 4.8, f"pause measured as {gap:.2f}s"


def test_vad_ignores_silence(cfg):
    assert _segments(cfg, silence(3.0)) == []


def test_asr_transcribes_with_word_times(cfg, speak):
    t = make_asr(cfg).transcribe(speak(S1))
    norm = re.sub(r"[^a-z0-9]", "", t.text.lower())   # "ESP-32" and "ESP 32" both count
    for word in ("temperature", "monitoring", "esp32", "microcontroller"):
        assert word in norm, t.text
    assert t.language == "en"
    assert t.words and all(w.end >= w.start for w in t.words)


def test_pi5_profile_asr_runs_on_pc(cfg_pi5, speak):
    """Rule: the pi5 profile (base, int8, beam 1, 2 threads) must work on the PC CPU."""
    t = make_asr(cfg_pi5).transcribe(speak(S2))
    assert "power" in t.text.lower()
