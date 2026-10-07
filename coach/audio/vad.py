"""Silero VAD on onnxruntime + a segmenter that turns a mic stream into speech segments (T1.1).

Numpy port of silero_vad.utils_vad.OnnxWrapper, so the Pi needs no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

WINDOW = 512   # samples per call at 16 kHz
CONTEXT = 64   # samples of previous window the model expects in front


class SileroVAD:
    def __init__(self, model_path: str | Path, sample_rate: int = 16000, threshold: float = 0.5):
        if sample_rate != 16000:
            raise ValueError("Pipeline runs at 16 kHz")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        """Speech probability for one 512-sample window."""
        x = np.concatenate([self._context, window.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self.session.run(None, {
            "input": x, "state": self._state, "sr": np.array(self.sample_rate, dtype=np.int64)})
        self._context = x[:, -CONTEXT:]
        return float(out[0][0])


@dataclass
class Segment:
    audio: np.ndarray   # float32 16 kHz, includes a short pre-roll
    start: float        # seconds since the segmenter started
    end: float


class Segmenter:
    """Feed 512-sample blocks; get finished speech segments back.

    A segment closes after `segment_silence_ms` of silence, or is force-split at
    `max_segment_s` so streaming ASR never waits on one huge chunk.
    """

    def __init__(self, vad: SileroVAD, min_speech_ms: int = 250, segment_silence_ms: int = 600,
                 max_segment_s: float = 15.0, pre_roll_ms: int = 200):
        self.vad = vad
        sr = vad.sample_rate
        self.min_speech = int(min_speech_ms * sr / 1000)
        self.silence_close = int(segment_silence_ms * sr / 1000)
        self.max_segment = int(max_segment_s * sr)
        self.pre_roll = int(pre_roll_ms * sr / 1000)
        self.on, self.off = vad.threshold, max(vad.threshold - 0.15, 0.01)  # hysteresis, as in Silero
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.t = 0                    # samples processed
        self.speaking = False
        self.buf: list[np.ndarray] = []
        self.history: list[np.ndarray] = []   # recent blocks, for pre-roll
        self.seg_start = 0
        self.silence = 0              # samples of silence inside the current segment
        self.last_speech_end = 0      # sample index where speech last stopped
        self.heard_speech = False

    @property
    def trailing_silence_s(self) -> float:
        """Seconds since the student last spoke (0 while speaking)."""
        if self.speaking:
            return 0.0
        return (self.t - self.last_speech_end) / self.vad.sample_rate

    def _close(self, end: int, trailing_silence: int = 0) -> Segment | None:
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        if trailing_silence > self.pre_roll:
            # Whisper tends to hallucinate words ("You", "Thank you") in trailing silence: cut it,
            # keeping a short tail so the last word is not clipped.
            audio = audio[: max(len(audio) - trailing_silence + self.pre_roll, 0)]
        self.buf, self.silence = [], 0
        if len(audio) < self.min_speech + self.pre_roll:
            return None
        sr = self.vad.sample_rate
        return Segment(audio, max(self.seg_start - self.pre_roll, 0) / sr, end / sr)

    def process(self, block: np.ndarray) -> list[Segment]:
        out: list[Segment] = []
        prob = self.vad(block)
        n = len(block)
        if not self.speaking:
            self.history = (self.history + [block])[-(self.pre_roll // n + 1):]
            if prob >= self.on:
                self.speaking, self.heard_speech = True, True
                self.seg_start = self.t
                self.buf = list(self.history)
                self.silence = 0
        else:
            self.buf.append(block)
            self.silence = self.silence + n if prob < self.off else 0
            seg_len = self.t + n - self.seg_start
            if self.silence >= self.silence_close:
                self.speaking = False
                self.last_speech_end = self.t + n - self.silence
                seg = self._close(self.last_speech_end, self.silence)
                if seg:
                    out.append(seg)
                self.history = []
            elif seg_len >= self.max_segment:
                seg = self._close(self.t + n)
                if seg:
                    out.append(seg)
                self.seg_start = self.t + n
        self.t += n
        return out

    def flush(self) -> list[Segment]:
        """Close any open segment (call when the answer ends)."""
        if self.speaking:
            self.speaking = False
            self.last_speech_end = self.t
            seg = self._close(self.t)
            return [seg] if seg else []
        return []
