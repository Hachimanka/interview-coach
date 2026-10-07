"""TTS backend interface (T1.3)."""
from abc import ABC, abstractmethod

import numpy as np


class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str = "en") -> tuple[np.ndarray, int]:
        """Return (audio samples, sample_rate)."""
