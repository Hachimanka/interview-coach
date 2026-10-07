"""T1.4: face presence, turn angle direction and frontal gaze on MediaPipe's public sample portrait."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from coach.profile import models_dir
from coach.vision.face import FaceTracker

from .conftest import require

PORTRAIT = Path(__file__).parent.parent / "samples" / "images" / "portrait.jpg"


@pytest.fixture(scope="module")
def tracker(cfg):
    require(models_dir(cfg) / "face_landmarker.task")
    if not PORTRAIT.exists():
        pytest.skip("samples/images/portrait.jpg missing")
    t = FaceTracker.from_config(cfg, video=False)
    yield t
    t.close()


@pytest.fixture(scope="module")
def img():
    return cv2.cvtColor(cv2.imread(str(PORTRAIT)), cv2.COLOR_BGR2RGB)


def test_frontal_face_is_present_and_looking(tracker, img):
    o = tracker.process(img)
    assert o.present and o.gaze.looking
    assert abs(o.turn_deg) < 3


def test_turn_angle_follows_face_position(tracker, img):
    w = img.shape[1]
    left = tracker.process(np.ascontiguousarray(img[:, w // 4:]))     # face moves to image left
    right = tracker.process(np.ascontiguousarray(img[:, : 3 * w // 4]))
    assert left.turn_deg < -5 and right.turn_deg > 5


def test_mirroring_flips_yaw_sign(tracker, img):
    a = tracker.process(img).gaze.yaw
    b = tracker.process(np.ascontiguousarray(img[:, ::-1])).gaze.yaw
    assert np.sign(a) != np.sign(b)


def test_no_face_in_blank_image(tracker):
    assert not tracker.process(np.full((480, 640, 3), 128, np.uint8)).present
