"""Microphone stream (sounddevice), 16 kHz mono float32 blocks (T1.1)."""
from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd

BLOCK = 512  # samples per block = Silero VAD window at 16 kHz (32 ms)


class MicStream:
    """Context manager that yields fixed-size float32 blocks from the microphone."""

    def __init__(self, sample_rate: int = 16000, device=None, block: int = BLOCK):
        self.sample_rate = sample_rate
        self.block = block
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                      blocksize=block, device=device, callback=self._callback)

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata[:, 0].copy())

    def __enter__(self) -> "MicStream":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next block, or None if nothing arrived within timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Drop buffered audio (e.g. the robot's own voice after it finishes speaking)."""
        while not self._q.empty():
            self._q.get_nowait()


def list_input_devices() -> list[tuple[int, str]]:
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
