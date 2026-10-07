"""Piper implementation of TTSBackend (English questions + live report summary) (T1.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from piper import PiperVoice

from coach.tts.base import TTSBackend


class PiperBackend(TTSBackend):
    def __init__(self, model_path: str | Path):
        self.voice = PiperVoice.load(str(model_path))

    def synthesize(self, text: str, language: str = "en") -> tuple[np.ndarray, int]:
        if language != "en":
            raise ValueError("Piper is used for English only; Filipino audio is pre-synthesized (MMS-TTS)")
        chunks = list(self.voice.synthesize(text))
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, chunks[0].sample_rate
