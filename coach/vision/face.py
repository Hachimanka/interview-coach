"""Face presence, turn angle and gaze from one MediaPipe Face Landmarker pass per frame (T1.4).

The landmarker finds the face itself, so no separate face detector runs (lighter on the Pi).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from coach.vision.gaze import Gaze, estimate


@dataclass
class FaceObservation:
    present: bool
    center: tuple[float, float] = (0.5, 0.5)   # normalized image coords of the face
    turn_deg: float = 0.0                      # + = student is to the camera's right
    gaze: Gaze | None = None


class FaceTracker:
    def __init__(self, model_path: str | Path, hfov_deg: float = 60, yaw_max: float = 15,
                 pitch_max: float = 15, iris_max: float = 0.3, video: bool = True):
        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO if video else RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(opts)
        self.video = video
        self.hfov = hfov_deg
        self.limits = (yaw_max, pitch_max, iris_max)

    @classmethod
    def from_config(cls, cfg, video: bool = True) -> "FaceTracker":
        from coach.profile import models_dir

        v = cfg.vision
        return cls(models_dir(cfg) / cfg.registry["vision"]["face_landmarker"]["file"], v.hfov_deg,
                   v.gaze_yaw_max_deg, v.gaze_pitch_max_deg, v.iris_offset_max, video)

    def process(self, rgb: np.ndarray, timestamp_ms: int = 0) -> FaceObservation:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = (self.landmarker.detect_for_video(image, timestamp_ms) if self.video
               else self.landmarker.detect(image))
        if not res.face_landmarks:
            return FaceObservation(False)
        pts = np.array([(p.x, p.y, p.z) for p in res.face_landmarks[0]], dtype=np.float32)
        cx, cy = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2
        gaze = estimate(pts, res.facial_transformation_matrixes[0], *self.limits)
        return FaceObservation(True, (float(cx), float(cy)), float((cx - 0.5) * self.hfov), gaze)

    def close(self) -> None:
        self.landmarker.close()
