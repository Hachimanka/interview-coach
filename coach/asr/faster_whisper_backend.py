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
        self.languages = list(a.get("languages") or [])   # auto-detect picks only among these

    def _run(self, audio: np.ndarray, language: str | None, prompt: str | None):
        return self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            initial_prompt=self.initial_prompt if prompt is None else (prompt or None),
            vad_filter=False,               # our Silero segmenter already did this
            condition_on_previous_text=False,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: str | None = None, prompt: str | None = None) -> Transcript:
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        audio = audio.astype(np.float32)
        language = language or self.default_language
        segments, info = self._run(audio, language, prompt)
        lang, prob = info.language, info.language_probability
        if language is None and self.languages:
            # Whisper may call Tagalog "Indonesian" or "Malay": keep the best of the allowed languages.
            probs = dict(info.all_language_probs or [])
            allowed = {code: probs.get(code, 0.0) for code in self.languages}
            best = max(allowed, key=allowed.get)
            prob = allowed[best] / (sum(allowed.values()) or 1.0)
            if best != lang:
                lang = best
                segments, info = self._run(audio, lang, prompt)   # the first pass was never decoded
        words: list[Word] = []
        texts: list[str] = []
        no_speech, logprobs = 0.0, []
        for seg in segments:
            texts.append(seg.text.strip())
            no_speech = max(no_speech, seg.no_speech_prob)
            logprobs.append(seg.avg_logprob)
            for w in seg.words or []:
                words.append(Word(w.word.strip(), w.start, w.end, w.probability))
        return Transcript(" ".join(t for t in texts if t), lang, words, float(prob), float(no_speech),
                          float(np.mean(logprobs)) if logprobs else 0.0)
