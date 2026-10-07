"""Eye contact per frame from Face Landmarker output: head pose + iris offsets (T1.4).

v0 = angle thresholds from the profile. T3.2 replaces `is_looking` with a trained
classifier if gate G2 shows accuracy < 85%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Face mesh indices (subject's right eye = image left when not mirrored)
R_OUTER, R_INNER, L_INNER, L_OUTER = 33, 133, 362, 263
R_IRIS, L_IRIS = 468, 473


@dataclass
class Gaze:
    yaw: float           # degrees, + = head turned to the subject's left
    pitch: float         # degrees, + = head tilted up
    iris_offset: float   # 0 = irises centered, 1 = at eye corners
    looking: bool


def head_pose(matrix: np.ndarray) -> tuple[float, float]:
    """Yaw and pitch (degrees) from MediaPipe's 4x4 facial transformation matrix."""
    pitch, yaw, _roll = Rotation.from_matrix(np.asarray(matrix)[:3, :3]).as_euler("xyz", degrees=True)
    return float(yaw), float(pitch)


def iris_offset(pts: np.ndarray) -> float:
    """Mean horizontal offset of both irises from their eye centers, normalized by half eye width."""
    offs = []
    for outer, inner, iris in ((R_OUTER, R_INNER, R_IRIS), (L_OUTER, L_INNER, L_IRIS)):
        a, b = pts[outer, :2], pts[inner, :2]
        half = np.linalg.norm(b - a) / 2
        if half > 1e-6:
            offs.append(abs(np.dot(pts[iris, :2] - (a + b) / 2, (b - a) / (2 * half))) / half)
    return float(np.mean(offs)) if offs else 0.0


def estimate(pts: np.ndarray, matrix: np.ndarray, yaw_max: float, pitch_max: float,
             iris_max: float) -> Gaze:
    yaw, pitch = head_pose(matrix)
    off = iris_offset(pts)
    return Gaze(yaw, pitch, off, abs(yaw) <= yaw_max and abs(pitch) <= pitch_max and off <= iris_max)
