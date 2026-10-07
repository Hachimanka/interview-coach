"""Background camera thread: latest face observation + eye-contact % over an answer window (T1.4)."""
from __future__ import annotations

import sys
import threading
import time

import cv2

from coach.vision.face import FaceObservation, FaceTracker


class VisionLoop:
    def __init__(self, cfg):
        v = cfg.vision
        self.cfg = cfg
        self.index, self.size, self.fps = v.camera_index, (v.width, v.height), v.fps
        self.latest = FaceObservation(False)
        self.fps_measured = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._window: list[bool] | None = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> "VisionLoop":
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    # ---- eye-contact window (one per answer) ----
    def start_window(self) -> None:
        with self._lock:
            self._window = []

    def stop_window(self) -> dict:
        """Eye contact over frames where a face was visible, plus how many frames had no face."""
        with self._lock:
            w, self._window = self._window or [], None
        looked = sum(w)
        return {"eye_contact_pct": round(100 * looked / len(w), 1) if w else None,
                "frames": len(w)}

    def _run(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            self.error = f"camera {self.index} could not be opened"
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        tracker = FaceTracker.from_config(self.cfg, video=True)
        period, t0, n, tick = 1.0 / self.fps, time.monotonic(), 0, time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                now = time.monotonic()
                obs = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int((now - t0) * 1000))
                with self._lock:
                    self.latest = obs
                    if self._window is not None and obs.present and obs.gaze:
                        self._window.append(obs.gaze.looking)
                n += 1
                if now - tick >= 2.0:
                    self.fps_measured, n, tick = n / (now - tick), 0, now
                time.sleep(max(0.0, period - (time.monotonic() - now)))  # cap at profile fps
        finally:
            tracker.close()
            cap.release()
