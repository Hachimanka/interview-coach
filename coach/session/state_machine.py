"""Session flow (T1.7): wait -> greeting -> consent -> language -> program -> questions -> report.

Slow work stays off the critical path (principle P6): each speech segment is transcribed by a
background worker while the student is still talking, and scoring runs on the same worker after
the answer ends. Only "answer ends -> next line plays" happens in the foreground.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from coach.audio.vad import Segment, Segmenter
from coach.delivery.summary import analyze
from coach.scoring.rubric import Bank, Question
from coach.session.lines import LINES
from coach.tts.cache import SpeechSource

PROGRAMS = ["CpE", "EE", "ECE"]
LANGS = {"English": "en", "Filipino": "fil"}


class SessionStopped(Exception):
    """The student pressed Stop."""


@dataclass
class Answer:
    question: Question
    segments: list[tuple[Segment, object]] = field(default_factory=list)  # (segment, Transcript)
    end_reason: str = ""
    duration_s: float = 0.0
    eye: dict = field(default_factory=dict)
    audio: list[np.ndarray] = field(default_factory=list)
    language: str | None = None
    result: dict | None = None
    done: threading.Event = field(default_factory=threading.Event)


class Worker(threading.Thread):
    """Runs ASR per segment and scoring per answer, in order, off the main thread."""

    def __init__(self, asr, scorer, cfg):
        super().__init__(name="asr-score", daemon=True)
        self.asr, self.scorer, self.cfg = asr, scorer, cfg
        self.jobs: queue.Queue = queue.Queue()
        self.error: Exception | None = None

    def run(self) -> None:
        while (job := self.jobs.get()) is not None:
            kind, ans, payload = job
            try:
                if kind == "seg":
                    # lock the language for the rest of the answer after the first segment
                    t = self.asr.transcribe(payload.audio, language=ans.language)
                    ans.language = ans.language or t.language
                    ans.segments.append((payload, t))
                else:
                    ans.result = finalize(ans, self.scorer, self.cfg)
                    ans.done.set()
            except Exception as e:  # keep the session alive; the report shows the failure
                self.error = e
                if kind == "final":
                    ans.result = {"error": repr(e)}
                    ans.done.set()

    def stop(self) -> None:
        self.jobs.put(None)


def finalize(ans: Answer, scorer, cfg) -> dict:
    words, texts, spans = [], [], []
    for seg, t in ans.segments:
        spans.append((seg.start, seg.end))
        texts.append(t.text)
        words += [type(w)(w.text, w.start + seg.start, w.end + seg.start, w.probability) for w in t.words]
    text = " ".join(x for x in texts if x).strip()
    delivery = analyze(text, words, spans, 0.0, cfg, ans.eye.get("eye_contact_pct"))
    score = scorer.score(ans.question, text)
    return {"transcript": text, "language": ans.language, "score": score, "delivery": delivery}


class Session:
    def __init__(self, cfg, face, mic, vad, asr, scorer, bank_loader, speech: SpeechSource,
                 vision=None, record_audio: bool = False):
        self.cfg, self.face, self.mic, self.vision = cfg, face, mic, vision
        a = cfg.audio
        self.segmenter = Segmenter(vad, a.min_speech_ms, a.segment_silence_ms, a.max_segment_s)
        self.asr, self.scorer, self.bank_loader, self.speech = asr, scorer, bank_loader, speech
        self.record_audio = record_audio
        self.lang = "en"
        self.answers: list[Answer] = []
        self.meta: dict = {}

    # ---------- helpers ----------
    def _event(self, names: set[str], timeout: float | None = None) -> dict | None:
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if end is None else max(end - time.monotonic(), 0)
            ev = self.face.server.next_event(timeout=0.1 if left is None else min(left, 0.1))
            if ev and ev["name"] == "stop":
                raise SessionStopped()
            if ev and ev["name"] in names:
                return ev
            if self.vision:
                self.face.look_at(self.vision.latest)
            if end is not None and time.monotonic() >= end:
                return None

    def _stop_pressed(self) -> bool:
        ev = self.face.server.next_event(timeout=0)
        if ev and ev["name"] == "stop":
            raise SessionStopped()
        return False

    def _line(self, key: str) -> None:
        self.face.say(self.speech.question(f"line.{key}", self.lang, LINES[key][self.lang]),
                      stop_check=self._stop_pressed)

    def _choose(self, kind: str, options: list[str], title: str) -> str:
        self.face.panel(kind, options, title)
        ev = self._event({kind})
        self.face.panel("none")
        return ev["value"]

    # ---------- flow ----------
    def wait_for_student(self) -> None:
        """Start when a face is seen for ~1 s, or on any tap/Enter."""
        self.face.set_state("idle")
        self.face.subtitle("Tap the screen or sit in front of me to start practicing.")
        self.face.panel("choice", ["Start"], "")
        seen_since = None
        while True:
            ev = self.face.server.next_event(timeout=0.1)
            if ev and ev["name"] in {"choice", "done"}:
                break
            if self.vision:
                obs = self.vision.latest
                self.face.look_at(obs)
                seen_since = (seen_since or time.monotonic()) if obs.present else None
                if seen_since and time.monotonic() - seen_since > 1.0:
                    break
        self.face.panel("none")
        self.face.subtitle("")

    def run(self) -> dict:
        worker = Worker(self.asr, self.scorer, self.cfg)
        worker.start()
        self._worker = worker
        self._n_questions = 0
        self.meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": self.cfg.profile,
                     "ended_early": False, "consented": False}
        try:
            self.face.set_state("greeting")
            lang_label = self._choose("choice", list(LANGS), "Choose a language / Pumili ng wika")
            self.lang = LANGS[lang_label]
            self._line("greeting")
            self.face.set_state("waiting")
            self.face.panel("consent", ["Yes", "No"] if self.lang == "en" else ["Oo", "Hindi"],
                            "Do you agree to continue?" if self.lang == "en" else "Pumapayag ka ba?")
            self._line("consent")
            ev = self._event({"consent"})
            self.face.panel("none")
            if ev["value"] not in ("Yes", "Oo"):
                self._line("declined")
                return self._finish(worker, saved=False)
            self.meta["consented"] = True

            program = self._choose("program", PROGRAMS,
                                   "Choose your program" if self.lang == "en" else "Piliin ang iyong programa")
            bank: Bank = self.bank_loader(program)
            questions = bank.questions[: self.cfg.session.questions_per_session]
            self._n_questions = len(questions)
            self.meta.update(program=program, language=self.lang, sample_rubric=bank.is_sample,
                             rubric_source=bank.source)
            self._line("instructions")

            for i, q in enumerate(questions, 1):
                text = q.text.get(self.lang) or q.text["en"]
                self.face.hud("", f"Q{i} / {len(questions)}")
                self.face.say(self.speech.question(q.id, self.lang, text), keep_subtitle=text,
                              stop_check=self._stop_pressed)
                ans = self._listen(q, text)
                for seg in self.segmenter.flush():
                    worker.jobs.put(("seg", ans, seg))
                worker.jobs.put(("final", ans, None))
                self.face.set_state("thinking")
                if i < len(questions):
                    self._line("next")
            self.face.set_state("closing")
            self._line("closing")
        except SessionStopped:
            self.meta["ended_early"] = True
        return self._finish(worker, saved=True)

    def _listen(self, q: Question, text: str) -> Answer:
        ans = Answer(q)
        self.answers.append(ans)
        self.face.set_state("listening")
        self.face.panel("answer")
        self.face.server.clear_events()
        self.mic.clear()                       # drop the robot's own voice
        self.segmenter.reset()
        if self.vision:
            self.vision.start_window()
        a, s = self.cfg.audio, self.cfg.session
        sr = a.sample_rate
        limit, last_hud, waiting = s.answer_time_limit_s, -1, False
        while True:
            block = self.mic.read(timeout=0.5)
            if block is not None:
                if self.record_audio:
                    ans.audio.append(block)
                for seg in self.segmenter.process(block):
                    self._worker_put(("seg", ans, seg))
            elapsed = self.segmenter.t / sr
            if int(elapsed) != last_hud:
                last_hud = int(elapsed)
                left = max(limit - last_hud, 0)
                self.face.hud(f"{left // 60:02d}:{left % 60:02d}", f"Q{len(self.answers)} / {self._n_questions}")
            # face reactions
            if self.vision:
                self.face.look_at(self.vision.latest)
            silence = self.segmenter.trailing_silence_s
            if self.segmenter.heard_speech and silence > 4.0 and not waiting:
                waiting = True
                self.face.set_state("waiting")
                self.face.subtitle(LINES["take_your_time"][self.lang])
            elif waiting and self.segmenter.speaking:
                waiting = False
                self.face.set_state("listening")
                self.face.subtitle(text, persist=True)
            if self.segmenter.speaking:
                self.face.maybe_nod()
            # end conditions
            ev = self.face.server.next_event(timeout=0)
            if ev and ev["name"] == "stop":
                self._end_answer(ans, "stopped", elapsed)
                raise SessionStopped()
            if ev and ev["name"] == "done":
                return self._end_answer(ans, "done", elapsed)
            if elapsed >= limit:
                return self._end_answer(ans, "time_limit", elapsed)
            if self.segmenter.heard_speech and silence >= a.end_of_answer_silence_s:
                return self._end_answer(ans, "silence", elapsed)

    def _worker_put(self, job) -> None:
        self._worker.jobs.put(job)

    def _end_answer(self, ans: Answer, reason: str, elapsed: float) -> Answer:
        ans.end_reason, ans.duration_s = reason, round(elapsed, 1)
        if self.vision:
            ans.eye = self.vision.stop_window()
        self.face.panel("none")
        return ans

    def _finish(self, worker: Worker, saved: bool) -> dict:
        self.face.set_state("thinking")
        self.face.panel("none")
        worker.stop()
        worker.join()
        self.meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if worker.error:
            self.meta["worker_error"] = repr(worker.error)
        return {"meta": self.meta, "answers": [a for a in self.answers if a.done.is_set()]}
