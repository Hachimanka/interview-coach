"""Speech clips with subtitle cues + lip-sync levels; pre-made question WAVs in cache/ (T1.3).

Layout: cache/questions/<question_id>.<lang>.wav  +  .json (cues, mouth levels)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from coach.face.lipsync import FRAME_S, mouth_levels
from coach.face.subtitles import Cue, cues_to_json, make_cues


@dataclass
class Speech:
    audio: np.ndarray
    sample_rate: int
    cues: list[Cue]
    mouth: list[int]
    frame_s: float = FRAME_S

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def mouth_at(self, t: float) -> int:
        i = int(t / self.frame_s)
        return self.mouth[i] if 0 <= i < len(self.mouth) else 0

    def cue_at(self, t: float) -> str:
        for c in self.cues:
            if c.start <= t < c.end:
                return c.text
        return self.cues[-1].text if self.cues and t >= self.cues[-1].start else ""


def build_speech(text: str, audio: np.ndarray, sample_rate: int) -> Speech:
    return Speech(audio.astype(np.float32), sample_rate,
                  make_cues(text, len(audio) / sample_rate), mouth_levels(audio, sample_rate))


class SpeechCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir) / "questions"

    def _paths(self, key: str, lang: str) -> tuple[Path, Path]:
        base = self.dir / f"{key}.{lang}"
        return base.with_suffix(f".{lang}.wav"), base.with_suffix(f".{lang}.json")

    def has(self, key: str, lang: str) -> bool:
        return all(p.exists() for p in self._paths(key, lang))

    def save(self, key: str, lang: str, text: str, speech: Speech) -> None:
        wav, meta = self._paths(key, lang)
        self.dir.mkdir(parents=True, exist_ok=True)
        sf.write(wav, speech.audio, speech.sample_rate)
        meta.write_text(json.dumps({"text": text, "cues": cues_to_json(speech.cues),
                                    "mouth": speech.mouth, "frame_s": speech.frame_s},
                                   ensure_ascii=False), encoding="utf-8")

    def load(self, key: str, lang: str) -> Speech:
        wav, meta = self._paths(key, lang)
        audio, sr = sf.read(wav, dtype="float32")
        m = json.loads(meta.read_text(encoding="utf-8"))
        return Speech(audio, sr, [Cue(**c) for c in m["cues"]], m["mouth"], m["frame_s"])


class SpeechSource:
    """Questions come from the cache when present; anything else is synthesized live with Piper."""

    def __init__(self, cache: SpeechCache, tts=None):
        self.cache = cache
        self.tts = tts

    def question(self, question_id: str, lang: str, text: str) -> Speech:
        if self.cache.has(question_id, lang):
            return self.cache.load(question_id, lang)
        return self.live(text, lang)

    def live(self, text: str, lang: str = "en") -> Speech:
        if lang != "en":
            raise RuntimeError(f"Filipino clip not cached: run python -m offline.synth_questions ({text[:40]}...)")
        if self.tts is None:
            raise RuntimeError("No live TTS configured and clip is not cached; run offline.synth_questions")
        audio, sr = self.tts.synthesize(text, "en")
        return build_speech(text, audio, sr)
