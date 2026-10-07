"""faster-whisper (CTranslate2) implementation of ASRBackend (T1.2). Multilingual models only."""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from coach.asr.base import ASRBackend, Transcript, Word
from coach.profile import Config, models_dir


class FasterWhisperBackend(ASRBackend):
    def __init__(self, cfg: Config):
        a = cfg.asr
        local = models_dir(cfg) / "whisper" / a.model
        source = str(local) if (local / "model.bin").exists() else cfg.registry["asr"][a.model]["repo"]
        self.model = WhisperModel(source, device=a.device, compute_type=a.compute_type,
                                  cpu_threads=a.cpu_threads,
                                  download_root=str(models_dir(cfg) / "whisper"))
        self.beam_size = a.beam_size
        self.default_language = a.language
        self.initial_prompt = a.initial_prompt

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: str | None = None) -> Transcript:
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        segments, info = self.model.transcribe(
            audio.astype(np.float32),
            language=language or self.default_language,
            beam_size=self.beam_size,
            word_timestamps=True,
            initial_prompt=self.initial_prompt,
            vad_filter=False,               # our Silero segmenter already did this
            condition_on_previous_text=False,
        )
        words: list[Word] = []
        texts: list[str] = []
        for seg in segments:
            texts.append(seg.text.strip())
            for w in seg.words or []:
                words.append(Word(w.word.strip(), w.start, w.end, w.probability))
        return Transcript(" ".join(t for t in texts if t), info.language, words)
