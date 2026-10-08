"""ASR backend interface (T1.2). Every backend returns the same Transcript shape."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Word:
    text: str
    start: float  # seconds from segment start
    end: float
    probability: float = 1.0


@dataclass
class Transcript:
    text: str
    language: str
    words: list[Word] = field(default_factory=list)
    language_probability: float = 1.0   # confidence of the detected language (among the allowed ones)
    no_speech_prob: float = 0.0         # high = probably noise, not speech
    avg_logprob: float = 0.0            # low = the model was guessing


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000,
                   language: str | None = None, prompt: str | None = None) -> Transcript:
        """Transcribe one VAD segment. language=None means auto-detect (EN/FIL).
        prompt=None uses the configured initial prompt; "" uses none (voice commands)."""
