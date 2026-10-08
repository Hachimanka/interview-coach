"""T1.9 + T1.10: face server replays state, receives touch events, and speech drives mouth + subtitles."""
import json
import time
import urllib.request

import numpy as np
import pytest
from websockets.sync.client import connect

from coach.face import controller as ctl
from coach.face.controller import FaceController
from coach.face.server import FaceServer
from coach.tts.cache import build_speech


@pytest.fixture
def server():
    s = FaceServer(ws_port=8875).start()
    time.sleep(0.2)
    yield s
    s.stop()


def _drain(ws, seconds=0.3):
    msgs, end = [], time.time() + seconds
    while time.time() < end:
        try:
            msgs.append(json.loads(ws.recv(timeout=0.05)))
        except TimeoutError:
            pass
    return msgs


def test_page_is_served(server):
    html = urllib.request.urlopen(server.url, timeout=2).read().decode()
    assert "Interviewer face" in html


def test_state_replayed_to_new_client(server):
    face = FaceController(server)
    face.set_state("listening")
    face.subtitle("Question text", persist=True)
    with connect(f"ws://127.0.0.1:{server.ws_port}") as ws:
        msgs = _drain(ws)
    assert {"type": "state", "state": "listening"} in msgs
    assert any(m["type"] == "subtitle" and m["persist"] for m in msgs)


def test_touch_events_reach_session(server):
    with connect(f"ws://127.0.0.1:{server.ws_port}") as ws:
        ws.send(json.dumps({"type": "event", "name": "program", "value": "EE"}))
        ev = server.next_event(timeout=2)
    assert ev["name"] == "program" and ev["value"] == "EE"


def test_say_sends_mouth_and_cues(server, monkeypatch):
    monkeypatch.setattr(ctl.sd, "play", lambda *a, **k: None)
    monkeypatch.setattr(ctl.sd, "wait", lambda: None)
    sr = 16000
    t = np.arange(int(sr * 0.6)) / sr
    audio = (0.5 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    speech = build_speech("Hello there, this is a short test sentence.", audio, sr)
    face = FaceController(server)
    with connect(f"ws://127.0.0.1:{server.ws_port}") as ws:
        _drain(ws, 0.1)
        assert face.say(speech, keep_subtitle="Hello there")
        msgs = _drain(ws)
    levels = {m["level"] for m in msgs if m["type"] == "mouth"}
    assert len(levels) >= 2 and 0 in levels          # mouth moved and closed at the end
    assert any(m["type"] == "state" and m["state"] == "talking" for m in msgs)
    assert msgs[-1] == {"type": "state", "state": "idle"} or face.state == "idle"


def test_mic_level_reaches_page_and_is_not_replayed(server):
    face = FaceController(server)
    with connect(f"ws://127.0.0.1:{server.ws_port}") as ws:
        _drain(ws, 0.1)
        face.hear(np.zeros(512, dtype=np.float32))
        time.sleep(0.1)
        face.hear(np.full(512, 0.2, dtype=np.float32))
        levels = [m["value"] for m in _drain(ws) if m["type"] == "level"]
    assert levels == [0.0, 1.0]
    with connect(f"ws://127.0.0.1:{server.ws_port}") as ws:
        assert not [m for m in _drain(ws) if m["type"] == "level"]


def test_page_options_follow_profile():
    assert FaceServer(ws_port=1, fps=24, effects=False).url.endswith("?ws=1&fps=24&fx=0")
    assert FaceServer(ws_port=1).url.endswith("?ws=1")


def test_no_negative_expressions():
    assert not {"sad", "angry", "disappointed", "frown"} & set(ctl.STATES)
