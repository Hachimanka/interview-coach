"""Simulated student for end-to-end session tests and gate G1 (T1.7).

The simulated student reacts to what the face page is told: when the robot starts listening
("listen" message) the student speaks into the FakeMic, a Piper-spoken answer or command.
ScriptedStudent taps the buttons; VoiceStudent says the commands instead (no touch screen).
Everything runs silently and faster than real time.

    python -m eval.simulate_session --sessions 3 --profile pc
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from coach.audio.util import blocks, resample
from coach.face.controller import FaceController
from coach.face.server import FaceServer
from coach.profile import (load_profile, make_asr, make_command_asr, make_embedder, make_interviewer, make_knowledge,
                           make_laughter, make_llm, make_rephraser, make_tts, make_vad, resolve)
from coach.scoring.rubric import load_session_bank
from coach.scoring.score import Scorer
from coach.session import report as rpt
from coach.session.lines import TEXT
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
    """Silence, except for what the simulated student says."""

    def __init__(self, tail_s: float = 5.0):
        self.tail = np.zeros(int(tail_s * 16000), np.float32)
        self._blocks: list[np.ndarray] = []

    def clear(self) -> None:
        self._blocks = []

    def say(self, audio: np.ndarray) -> None:
        self._blocks = list(blocks(np.concatenate([np.zeros(8000, np.float32), audio, self.tail])))

    def read(self, timeout: float = 1.0):
        return self._blocks.pop(0) if self._blocks else np.zeros(512, np.float32)


class ScriptedStudent:
    """Watches the messages to the face page: taps the buttons like a student with a mouse,
    and answers out loud when the robot starts listening for an answer."""

    def __init__(self, server: FaceServer, mic: FakeMic, answers: list[np.ndarray], consent="Yes", program="CpE",
                 language: str | None = None, stop_after_panels: int | None = None,
                 followups: list[np.ndarray] | None = None, name: np.ndarray | None = None):
        self.server, self.mic, self.answers = server, mic, list(answers)
        self.name = name                         # what to say when the robot asks for a first name
        self.title = ""                          # title of the last panel shown (tells consent screens apart)
        self.followups = list(followups or [])   # what to say when the robot asks a follow-up
        self.followups_heard = 0
        self.choices = {"choice": None, "consent": consent, "program": program}
        self.stop_after, self.panels = stop_after_panels, 0
        self._send = server.send
        server.send = self._intercept
        if language:                          # like the operator pressing E / F
            server.events.put({"type": "event", "name": "lang", "value": language})

    def _intercept(self, msg: dict) -> None:
        self._send(msg)
        if msg["type"] == "listen" and msg["kind"] != "none":
            self.on_listen(msg["kind"])
        if msg["type"] == "panel":
            self.title = msg.get("title", "")
        if msg["type"] == "panel" and msg["kind"] not in ("none", "answer"):
            self.panels += 1
            if self.stop_after is not None and self.panels > self.stop_after:
                self.server.events.put({"type": "event", "name": "stop"})
            else:
                self.on_panel(msg["kind"], msg["options"])

    def on_listen(self, kind: str) -> None:
        if kind == "answer":
            self.mic.say(self.answers.pop(0) if self.answers else np.zeros(16000, np.float32))
        elif kind == "followup":
            self.followups_heard += 1
            if self.followups:
                self.mic.say(self.followups.pop(0))
        elif kind == "name" and self.name is not None:
            self.mic.say(self.name)

    def on_panel(self, kind: str, options: list[str]) -> None:
        self.server.events.put({"type": "event", "name": kind, "value": self.choices[kind] or options[0]})


class VoiceStudent(ScriptedStudent):
    """Never touches the screen: says every choice. `spoken` maps a screen to the audio to say there,
    e.g. {"consent": <"yes">}; "keep" is the keep-report question, "name_ok" the "did I get your name right?"."""

    def __init__(self, server, mic, answers, spoken: dict[str, np.ndarray], **kw):
        super().__init__(server, mic, answers, **kw)
        self.spoken, self.consent_screens = spoken, 0

    def on_listen(self, kind: str) -> None:
        if kind in ("answer", "followup", "name"):
            return super().on_listen(kind)
        if kind == "consent":                 # yes/no screens: told apart by their title
            self.consent_screens += 1
            if self.title in TEXT["keep_title"].values():
                kind = "keep"
            elif self.title in TEXT["name_title"].values():
                kind = "name_ok"
            else:
                kind = "consent"
        if kind in self.spoken:
            self.mic.say(self.spoken[kind])

    def on_panel(self, kind: str, options: list[str]) -> None:
        pass


def make_answers(cfg, texts=ANSWERS) -> list[np.ndarray]:
    tts = make_tts(cfg)
    out = []
    for t in texts:
        a, sr = tts.synthesize(t)
        out.append(resample(a, sr))
    return out


def run_simulated(cfg, answers, server, models, spoken: dict | None = None, wait: bool = False,
                  keep: bool = False, **student) -> dict:
    """spoken: use a VoiceStudent that says these commands instead of tapping.
    wait: also run the idle "say start" screen. keep: also ask the keep-report question."""
    vad, asr, scorer, speech, *rest = models
    cmd_asr, rephraser, laughter, interviewer, knowledge = (list(rest) + [None] * 5)[:5]
    mic = FakeMic()
    server.clear_events()
    if spoken is None:
        ScriptedStudent(server, mic, answers, **student)
    else:
        VoiceStudent(server, mic, answers, spoken, **student)
    face = FaceController(server, play_audio=False)
    s = Session(cfg, face, mic, vad, asr, scorer, lambda: load_session_bank(cfg), speech, cmd_asr=cmd_asr,
                rephraser=rephraser, laughter=laughter, interviewer=interviewer, knowledge=knowledge)
    try:
        if wait:
            s.wait_for_student()
        result = s.run()
        if keep:
            result["meta"]["kept"] = s.ask_keep()
        result["meta"]["final_language"] = s.lang
        result["memory"] = s.memory               # for tests: what the robot remembered
    finally:
        server.send = server.__class__.send.__get__(server)   # remove interceptor
    return result


def load_models(cfg, reply: bool = True, interviewer: bool = False):
    """reply=False skips the local AI and laughter models (faster tests of other parts).
    interviewer=True adds the generative interviewer + knowledge library (otherwise: scripted mode)."""
    scorer = Scorer.from_config(cfg, make_embedder(cfg))
    llm = make_llm(cfg) if reply or interviewer else None
    return (make_vad(cfg), make_asr(cfg), scorer,
            SpeechSource(SpeechCache(resolve(cfg, cfg.paths.cache_dir)), make_tts(cfg)),
            make_command_asr(cfg),
            make_rephraser(cfg, llm) if reply and not interviewer else None,
            make_laughter(cfg) if reply else None,
            make_interviewer(cfg, llm) if interviewer else None,
            make_knowledge(cfg, scorer.emb) if interviewer else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pc")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--save", action="store_true", help="save reports under eval/results/sim/")
    ap.add_argument("--scoring-backend", help="override scoring.backend (pi5 needs the T4.2 ONNX export)")
    ap.add_argument("--generative", action="store_true", help="AI-written questions and replies (says a name too)")
    args = ap.parse_args()
    cfg = load_profile(args.profile)
    if args.scoring_backend:
        cfg["scoring"]["backend"] = args.scoring_backend
        print(f"note: scoring.backend overridden to {args.scoring_backend}")
    models = load_models(cfg, interviewer=args.generative)
    answers = make_answers(cfg)
    extra = {}
    if args.generative:     # the student says a name, and "yes" when the robot checks it
        name, yes = make_answers(cfg, ["My name is Leonard.", "Yes."])
        extra = {"spoken": {"consent": yes, "name_ok": yes}, "name": name}
    server = FaceServer(ws_port=8895).start()
    ok = 0
    try:
        for i in range(1, args.sessions + 1):
            t0 = time.perf_counter()
            result = run_simulated(cfg, answers, server, models, **extra)
            rep = rpt.build_report(result)
            for a in rep["answers"] if args.generative else []:
                print(f"   Q ({(a.get('question_info') or {}).get('source', '-')}): {a['question']['en']}")
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
