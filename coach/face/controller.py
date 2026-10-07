"""Expression rules + speech playback with lip-sync and subtitles (T1.9, T1.10).

States (plan section 2.1): idle, greeting, talking, listening, waiting, thinking, closing.
Design rule: expressions stay neutral or positive. There is no "disappointed" state, so the face
can never look like it is judging an answer before the rubric score exists.
"""
from __future__ import annotations

import time

import sounddevice as sd

from coach.face.server import FaceServer
from coach.tts.cache import Speech

STATES = ("idle", "greeting", "talking", "listening", "waiting", "thinking", "closing")
UPDATE_HZ = 30


class FaceController:
    def __init__(self, server: FaceServer, nod=None, play_audio: bool = True):
        self.server = server
        self.play_audio = play_audio   # False = silent, instant playback (simulated test sessions)
        self.state = None
        self.nod_hook = nod            # T4.5: callable that also nods the arm
        self._last_nod = 0.0
        self.set_state("idle")

    def set_state(self, state: str) -> None:
        if state not in STATES:
            raise ValueError(state)
        if state != self.state:
            self.state = state
            self.server.send({"type": "state", "state": state})

    def subtitle(self, text: str, persist: bool = False) -> None:
        self.server.send({"type": "subtitle", "text": text, "persist": persist})

    def hud(self, timer: str = "", question: str = "") -> None:
        self.server.send({"type": "hud", "timer": timer, "q": question})

    def panel(self, kind: str, options: list[str] | None = None, title: str = "") -> None:
        """kind: none | consent | program | choice | answer (shows Done). Taps come back as events named after kind."""
        self.server.send({"type": "panel", "kind": kind, "options": options or [], "title": title})

    def look_at(self, obs) -> None:
        """Eyes follow the student. Camera and screen face the same way, so a student on the
        camera image's left is on the screen's right as the student sees it: x is flipped."""
        if obs is None or not obs.present:
            self.server.send({"type": "gaze", "x": 0.0, "y": 0.0, "present": False})
            return
        x = max(-1.0, min(1.0, (0.5 - obs.center[0]) * 2))
        y = max(-1.0, min(1.0, (obs.center[1] - 0.45) * 2))
        self.server.send({"type": "gaze", "x": round(x, 2), "y": round(y, 2), "present": True})

    def maybe_nod(self, every_s: float = 8.0) -> None:
        """While listening: a slow nod every few seconds (face + arm via nod_hook)."""
        now = time.monotonic()
        if self.state == "listening" and now - self._last_nod >= every_s:
            self._last_nod = now
            self.server.send({"type": "nod"})
            if self.nod_hook:
                self.nod_hook()

    def say(self, speech: Speech, keep_subtitle: str | None = None, stop_check=None) -> bool:
        """Play speech with lip-sync and timed subtitles. Blocks until done.
        Returns False if stop_check() became true mid-sentence (playback is cut)."""
        prev = self.state
        self.set_state("talking")
        if not self.play_audio:
            for c in speech.cues:
                self.subtitle(c.text)
            self.subtitle(keep_subtitle or "", persist=bool(keep_subtitle))
            self.set_state(prev if prev != "talking" else "idle")
            return True
        sd.play(speech.audio, speech.sample_rate)
        t0 = time.monotonic()
        last_cue, last_mouth, completed = None, None, True
        while (t := time.monotonic() - t0) < speech.duration:
            if stop_check and stop_check():
                sd.stop()
                completed = False
                break
            cue, mouth = speech.cue_at(t), speech.mouth_at(t)
            if cue != last_cue:
                self.subtitle(cue)
                last_cue = cue
            if mouth != last_mouth:
                self.server.send({"type": "mouth", "level": mouth})
                last_mouth = mouth
            time.sleep(1 / UPDATE_HZ)
        sd.wait()
        self.server.send({"type": "mouth", "level": 0})
        self.subtitle(keep_subtitle or "", persist=bool(keep_subtitle))
        self.set_state(prev if prev != "talking" else "idle")
        return completed
