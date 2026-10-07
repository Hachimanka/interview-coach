"""WAV loudness envelope (~30 ms frames) -> mouth-open levels 0..3 (T1.10)."""
from __future__ import annotations

import numpy as np

FRAME_S = 0.03
LEVELS = 3   # 0 = closed ... 3 = wide open


def mouth_levels(audio: np.ndarray, sample_rate: int, frame_s: float = FRAME_S) -> list[int]:
    n = max(int(sample_rate * frame_s), 1)
    frames = len(audio) // n
    if frames == 0:
        return []
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1))
    ref = np.percentile(rms, 95) or 1.0
    norm = np.clip(rms / ref, 0, 1)
    levels = np.round(norm * LEVELS).astype(int)
    levels[norm < 0.08] = 0          # treat near-silence as closed
    return levels.tolist()
