"""Simulated student for end-to-end session tests and gate G1 (T1.7).

FakeMic plays Piper-spoken answers (+ trailing silence) each time the session starts listening;
ScriptedStudent taps the touch panels. Everything runs silently and faster than real time.

    python -m eval.simulate_session --sessions 3 --profile pc
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from coach.audio.util import blocks, resample
from coach.face.controller import FaceController
from coach.face.server import FaceServer
from coach.profile import load_profile, make_asr, make_embedder, make_tts, make_vad, resolve
from coach.scoring.rubric import load_bank
from coach.scoring.score import Scorer
from coach.session import report as rpt
from coach.session.state_machine import Session
from coach.tts.cache import SpeechCache, SpeechSource

ANSWERS = [
    "Good morning. I am a fourth year Computer Engineering student. For our capstone I built an attendance "
    "system with an ESP32, and I am applying here because your team builds embedded products.",
    "Um, in our microprocessors project two members wanted different sensors. I suggested we test both for "
    "one day and compare the accuracy. The ultrasonic sensor won and everyone agreed.",
    "I used to underestimate how long debugging takes. Now I add extra time to every estimate and track my "
    "actual hours, and this semester I submitted every lab on time.",
    "We made a project, it was okay.",
    "First I reproduce the bug. Then I test one module at a time with serial prints or a logic analyzer, "
    "and after the fix I retest the whole system.",
]


class FakeMic:
    def __init__(self, answers: list[np.ndarray], tail_s: float = 5.0):
        self.answers, self.tail = list(answers), np.zeros(int(tail_s * 16000), np.float32)
        self._blocks: list[np.ndarray] = []

    def clear(self) -> None:          # session calls this when listening starts
        audio = self.answers.pop(0) if self.answers else np.zeros(16000, np.float32)
        self._blocks = list(blocks(np.concatenate([np.zeros(8000, np.float32), audio, self.tail])))

    def read(self, timeout: float = 1.0):
        return self._blocks.pop(0) if self._blocks else np.zeros(512, np.float32)


class ScriptedStudent:
    """Intercepts panel messages and taps an answer, like a student on the touch screen."""

    def __init__(self, server: FaceServer, language="English", consent="Yes", program="CpE",
                 stop_after_panels: int | None = None):
        self.server, self.choices = server, {"choice": None, "consent": consent, "program": program}
        self.language, self.stop_after, self.panels = language, stop_after_panels, 0
        self._send = server.send
        server.send = self._intercept

    def _intercept(self, msg: dict) -> None:
        self._send(msg)
        if msg["type"] != "panel" or msg["kind"] in ("none", "answer"):
            return
        self.panels += 1
        if self.stop_after is not None and self.panels > self.stop_after:
            self.server.events.put({"type": "event", "name": "stop"})
            return
        kind = msg["kind"]
        value = (self.language if "English" in msg["options"] else msg["options"][0]) if kind == "choice" \
            else self.choices[kind]
        self.server.events.put({"type": "event", "name": kind, "value": value})


def make_answers(cfg, texts=ANSWERS) -> list[np.ndarray]:
    tts = make_tts(cfg)
    out = []
    for t in texts:
        a, sr = tts.synthesize(t)
        out.append(resample(a, sr))
    return out


def run_simulated(cfg, answers, server, models, **student) -> dict:
    vad, asr, scorer, speech = models
    ScriptedStudent(server, **student)
    face = FaceController(server, play_audio=False)
    s = Session(cfg, face, FakeMic(answers), vad, asr, scorer, lambda p: load_bank(cfg, p), speech)
    result = s.run()
    server.send = server.__class__.send.__get__(server)   # remove interceptor
    return result


def load_models(cfg):
    return (make_vad(cfg), make_asr(cfg), Scorer.from_config(cfg, make_embedder(cfg)),
            SpeechSource(SpeechCache(resolve(cfg, cfg.paths.cache_dir)), make_tts(cfg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pc")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="save reports under eval/results/sim/")
    ap.add_argument("--scoring-backend", help="override scoring.backend (pi5 needs the T4.2 ONNX export)")
    args = ap.parse_args()
    cfg = load_profile(args.profile)
    if args.scoring_backend:
        cfg["scoring"]["backend"] = args.scoring_backend
        print(f"note: scoring.backend overridden to {args.scoring_backend}")
    models = load_models(cfg)
    answers = make_answers(cfg)
    server = FaceServer(ws_port=8895).start()
    ok = 0
    try:
        for i in range(1, args.sessions + 1):
            t0 = time.perf_counter()
            result = run_simulated(cfg, answers, server, models)
            rep = rpt.build_report(result)
            s = rep["summary"]
            scores = [a["score"]["score"] for a in rep["answers"] if "score" in a]
            good = s["answers"] == 5 and len(scores) == 5 and "worker_error" not in rep["meta"]
            ok += good
            print(f"session {i}: {'OK ' if good else 'FAIL'} {time.perf_counter() - t0:5.1f}s  "
                  f"scores={scores}  avg={s.get('average_score')}  corrections={len(s.get('top_corrections', []))}")
            if args.save:
                rpt.save(rep, resolve(cfg, "eval/results/sim") / f"{cfg.profile}-{i}")
    finally:
        server.stop()
    print(f"G1 ({cfg.profile}): {ok}/{args.sessions} complete sessions without crash")


if __name__ == "__main__":
    main()
