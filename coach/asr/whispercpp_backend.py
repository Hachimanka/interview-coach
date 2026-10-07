"""whisper.cpp implementation of ASRBackend; Pi fallback (lever 9).

TODO(T4.2): only needed if faster-whisper int8 is too slow on the Pi 5 (decided in T4.3).
"""
from coach.asr.base import ASRBackend


class WhisperCppBackend(ASRBackend):
    def __init__(self, cfg):
        raise NotImplementedError("whisper.cpp backend is planned for T4.2; use asr.backend: faster_whisper")

    def transcribe(self, audio, sample_rate=16000, language=None):
        raise NotImplementedError
