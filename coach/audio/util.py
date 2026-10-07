"""Small audio helpers shared by modules."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample(audio: np.ndarray, src_rate: int, dst_rate: int = 16000) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32)
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def blocks(audio: np.ndarray, size: int = 512):
    """Split audio into fixed-size blocks (last one zero-padded), like the mic stream."""
    for i in range(0, len(audio), size):
        b = audio[i:i + size]
        if len(b) < size:
            b = np.pad(b, (0, size - len(b)))
        yield b
