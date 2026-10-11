"""Did the student laugh? YAMNet (MediaPipe audio classifier, 521 sound classes, 4 MB, offline).

The threshold in config (reply.laughter_threshold) must be calibrated with real recordings: a
synthetic voice reading "ha ha" is not real laughter, so tests only check the wiring.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LAUGH_CLASSES = ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh", "Snicker"]


class LaughterDetector:
    def __init__(self, model_path: str | Path, threshold: float = 0.3):
        from mediapipe.tasks.python import BaseOptions, audio

        from coach.vision.face import quiet_native_logs

        opts = audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=audio.RunningMode.AUDIO_CLIPS, max_results=-1,
            category_allowlist=LAUGH_CLASSES)
        with quiet_native_logs():
            self.clf = audio.AudioClassifier.create_from_options(opts)
        self.threshold = threshold

    def probability(self, audio16k: np.ndarray) -> float:
        """Highest laughter score over ~1 s windows of the clip."""
        from mediapipe.tasks.python.components.containers import audio_data

        if len(audio16k) < 1600:
            return 0.0
        results = self.clf.classify(audio_data.AudioData.create_from_array(audio16k.astype(np.float32), 16000))
        return max((c.score for r in results for c in r.classifications[0].categories), default=0.0)

    def laughed(self, audio16k: np.ndarray) -> bool:
        return self.probability(audio16k) >= self.threshold
