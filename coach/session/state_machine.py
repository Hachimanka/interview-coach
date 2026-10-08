"""Session flow (T1.7): wait -> greeting -> consent -> program -> questions -> report.

Every choice can be spoken or tapped (coach/session/voice.py). There is no language screen: the
robot follows the language the student speaks (English or Filipino), from the first command and
then from each answer.

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
from coach.session import voice
from coach.session.lines import LINES, label, text as screen_text
from coach.tts.cache import SpeechSource

PROGRAMS = ["CpE", "EE", "ECE"]
WHISPER_LANG = {"en": "en", "tl": "fil"}     # Whisper language code -> session language
TAIL_S = 3.5    # seconds at the end of a segment checked first for "that's my answer" / "stop the interview"
BILINGUAL = {"greeting", "consent", "consent_touch", "program", "retry", "keep"}  # said in both while unknown


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
    language: str | None = None        # Whisper code of what the student spoke ("en" / "tl")
    language_prob: float = 0.0
    asked_lang: str = "en"             # language the robot asked the question in
    queued: int = 0                    # segments sent to ASR
    tails: int = 0                     # quick end-phrase checks sent / finished
    tails_done: int = 0
    spotted: tuple | None = None       # (segment number, voice.Match) from the latest quick check
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
                    if ans.language is None:
                        ans.language, ans.language_prob = t.language, t.language_probability
                    ans.segments.append((payload, t))
                elif kind == "tail":
                    # Before the whole segment (slow when long): did it END with a spoken command?
                    number, audio = payload
                    try:
                        t = self.asr.transcribe(audio, language=ans.language, prompt="")
                        ans.spotted = (number, voice.match(t.text, ("done", "stop"), tail=True))
                    finally:
                        ans.tails_done += 1
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
    spoken = [i for i, (_, t) in enumerate(ans.segments) if t.text.strip()]
    for i, (seg, t) in enumerate(ans.segments):
        seg_text, seg_words = t.text, t.words
        if spoken and i == spoken[-1] and (cfg.get("voice") or {}).get("enabled"):
            seg_text, seg_words = voice.strip_tail(seg_text, seg_words)   # "that's my answer" is not part of it
        spans.append((seg.start, seg.end))
        texts.append(seg_text)
        words += [type(w)(w.text, w.start + seg.start, w.end + seg.start, w.probability) for w in seg_words]
    text = " ".join(x for x in texts if x).strip()
    delivery = analyze(text, words, spans, 0.0, cfg, ans.eye.get("eye_contact_pct"))
    score = scorer.score(ans.question, text)
    return {"transcript": text, "language": ans.language, "asked_in": ans.asked_lang,
            "score": score, "delivery": delivery}


class Session:
    def __init__(self, cfg, face, mic, vad, asr, scorer, bank_loader, speech: SpeechSource,
                 vision=None, record_audio: bool = False):
        self.cfg, self.face, self.mic, self.vision = cfg, face, mic, vision
        a = cfg.audio
        self.segmenter = Segmenter(vad, a.min_speech_ms, a.segment_silence_ms, a.max_segment_s)
        self.asr, self.scorer, self.bank_loader, self.speech = asr, scorer, bank_loader, speech
        self.record_audio = record_audio
        self.vcfg = cfg.get("voice") or {}
        self.listener = voice.VoiceListener(cfg, mic, vad, asr, face) if self.vcfg.get("enabled") else None
        self.lang: str | None = None      # None = not known yet; the robot uses both languages until it is
        self.lang_locked = False          # an operator pressed E / F: stop following the student
        self.answers: list[Answer] = []
        self.meta: dict = {}
        self._worker: Worker | None = None
        self._n_questions = 0
        self._warned: set[str] = set()

    # ---------- language ----------
    @property
    def _l(self) -> str:
        return self.lang or "en"

    def _set_lang(self, lang: str | None, source: str) -> None:
        if lang not in ("en", "fil"):
            return
        if source == "operator":
            self.lang, self.lang_locked = lang, True
        elif not self.lang_locked and self.vcfg.get("auto_language", True) and lang != self.lang:
            self.lang = lang
            print(f"[coach] language -> {lang} (from {source})")

    def _lang_from_command(self, lang: str | None) -> None:
        """Filipino command words always mean Filipino. English ones ("yes", "start") are also used
        by Filipino speakers, so they only set the language while it is still unknown."""
        if lang == "fil" or self.lang is None:
            self._set_lang(lang, "command")

    def _lang_from_answer(self, ans: Answer, wait_s: float = 2.0) -> None:
        """The next question is asked in the language of the answer just given."""
        end = time.monotonic() + wait_s          # the first segment is usually transcribed already
        while ans.language is None and ans.segments == [] and self.segmenter.heard_speech \
                and time.monotonic() < end:
            time.sleep(0.05)
        lang = WHISPER_LANG.get(ans.language)
        sure = ans.language_prob >= self.vcfg.get("language_switch_confidence", 0.6)
        if lang and (self.lang is None or sure):
            self._set_lang(lang, "answer")

    # ---------- helpers ----------
    def _clip(self, key: str, texts: dict[str, str], lang: str | None = None):
        """Speech + text for a line in the session language; English if there is no Filipino clip."""
        lang = lang or self._l
        text = texts.get(lang)
        if lang != "en" and not (text and self.speech.available(key, lang, text)):
            if "fil" not in self._warned:
                self._warned.add("fil")
                print("[coach] no Filipino clip (run: python -m offline.synth_questions); speaking English")
            lang, text = "en", texts["en"]
        return self.speech.question(key, lang, text), text, lang

    def _stop_pressed(self) -> bool:
        ev = self.face.server.next_event(timeout=0)
        if ev and ev["name"] == "stop":
            raise SessionStopped()
        if ev and ev["name"] == "lang":
            self._set_lang(ev.get("value"), "operator")
        elif ev:
            self.face.server.events.put(ev)      # a tap made while the robot is speaking still counts
        return False

    def _line(self, key: str) -> None:
        """Say a fixed line. While the language is unknown, short prompts are said in both."""
        langs = ["en", "fil"] if self.lang is None and key in BILINGUAL else [self._l]
        said = set()
        for lang in langs:
            speech, _, used = self._clip(f"line.{key}", LINES[key], lang)
            if used not in said:
                said.add(used)
                self.face.say(speech, stop_check=self._stop_pressed)

    def _tick(self) -> bool:
        if self.vision:
            self._ticks = getattr(self, "_ticks", 0) + 1
            if self._ticks % 3 == 0:
                self.face.look_at(self.vision.latest)
        return False

    def _choose(self, kind: str, commands: list[str], title_key: str, line: str | None = None,
                aliases: dict[str, str] | None = None) -> str | None:
        """Show buttons, optionally say a line, then wait for a spoken or tapped choice.
        Returns the chosen command, or None if nobody answered in time."""
        options = [(c, label(c, self.lang)) for c in commands]
        hints = [voice.hint(c, self.lang) for c in commands] if self.listener else []
        self.face.panel(kind, [lab for _, lab in options], screen_text(title_key, self.lang), hints)
        if line:
            self._line(line)
        choice = voice.wait_choice(
            self.face, self.listener, kind, options, aliases=aliases,
            timeout=self.vcfg.get("choice_timeout_s") if self.listener else None,
            retries=self.vcfg.get("retries", 2), tick=self._tick,
            on_retry=lambda: self._line("retry"), on_lang=lambda v: self._set_lang(v, "operator"))
        self.face.panel("none")
        if choice.how == "stop":
            raise SessionStopped()
        if choice.how == "voice":
            self._lang_from_command(choice.lang)
        return choice.command

    # ---------- flow ----------
    def wait_for_student(self) -> None:
        """Start on "start" / "simulan", when a face is seen for ~1 s, or on any tap/Enter."""
        self.face.set_state("idle")
        self.face.hud("", "")
        self.face.subtitle(screen_text("start_voice" if self.listener else "start_touch", None))
        self.face.panel("choice", [label("start", None)], "", [voice.hint("start", None)] if self.listener else [])
        seen_since = None

        def tick() -> bool:
            nonlocal seen_since
            if not self.vision:
                return False
            obs = self.vision.latest
            self.face.look_at(obs)
            seen_since = (seen_since or time.monotonic()) if obs.present else None
            return bool(seen_since and time.monotonic() - seen_since > 1.0)

        while True:
            choice = voice.wait_choice(self.face, self.listener, "choice", [("start", label("start", None))],
                                       also=("done",), retries=0, tick=tick,
                                       on_lang=lambda v: self._set_lang(v, "operator"))
            if choice.how != "stop":          # "stop" means nothing while idle
                break
        if choice.how == "voice":
            self._lang_from_command(choice.lang)
        self.face.panel("none")
        self.face.subtitle("")

    def run(self) -> dict:
        worker = Worker(self.asr, self.scorer, self.cfg)
        worker.start()
        self._worker = worker
        self._n_questions = 0
        self.meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": self.cfg.profile,
                     "ended_early": False, "consented": False}
        touch = "" if self.listener else "_touch"
        try:
            self.face.set_state("greeting")
            self._line("greeting")
            self.face.set_state("waiting")
            # Consent is never assumed: only a clear yes counts; no answer in time is a no.
            if self._choose("consent", ["yes", "no"], "consent_title", line="consent" + touch) != "yes":
                self._line("declined")
                return self._finish(worker, saved=False)
            self.meta["consented"] = True

            program = self._choose("program", PROGRAMS, "program_title", line="program" if self.listener else None)
            if program is None:
                raise SessionStopped()
            bank: Bank = self.bank_loader(program)
            questions = bank.questions[: self.cfg.session.questions_per_session]
            self._n_questions = len(questions)
            self.meta.update(program=program, language=self._l, sample_rubric=bank.is_sample,
                             rubric_source=bank.source)
            self._line("instructions" + touch)

            for i, q in enumerate(questions, 1):
                speech, text, lang = self._clip(q.id, q.text)
                self.face.hud("", f"Q{i} / {len(questions)}")
                self.face.say(speech, keep_subtitle=text, stop_check=self._stop_pressed)
                ans = self._listen(q, text, lang)
                for seg in self.segmenter.flush():
                    worker.jobs.put(("seg", ans, seg))
                    ans.queued += 1
                self.face.set_state("thinking")
                self._last_words(ans)
                worker.jobs.put(("final", ans, None))
                self._lang_from_answer(ans)
                if i < len(questions):
                    self._line("next")
            self.face.set_state("closing")
            self._line("closing")
        except SessionStopped:
            self.meta["ended_early"] = True
        return self._finish(worker, saved=True)

    def _listen(self, q: Question, text: str, lang: str = "en") -> Answer:
        ans = Answer(q, asked_lang=lang)
        self.answers.append(ans)
        self.face.set_state("listening")
        self.face.panel("answer", hints=[voice.hint("done", self.lang)] if self.listener else [])
        self.face.server.clear_events()
        self.mic.clear()                       # drop the robot's own voice
        self.segmenter.reset()
        self.face.listen("answer")
        if self.vision:
            self.vision.start_window()
        a, s = self.cfg.audio, self.cfg.session
        sr = a.sample_rate
        limit, last_hud, waiting = s.answer_time_limit_s, -1, False
        checked = 0                            # segments already checked for a spoken command
        while True:
            block = self.mic.read(timeout=0.5)
            self.face.hear(block)
            if block is not None:
                if self.record_audio:
                    ans.audio.append(block)
                for seg in self.segmenter.process(block):
                    ans.queued += 1
                    if self.listener and not self.segmenter.speaking:      # closed by a pause
                        ans.tails += 1
                        self._worker_put(("tail", ans, (ans.queued, seg.audio[-int(TAIL_S * sr):])))
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
                self.face.subtitle(LINES["take_your_time"][lang])
            elif waiting and self.segmenter.speaking:
                waiting = False
                self.face.set_state("listening")
                self.face.subtitle(text, persist=True)
            if self.segmenter.speaking:
                self.face.maybe_nod()
            # spoken "that's my answer" / "stop the interview": only as the last thing said, then a pause
            spoken = None
            if ans.spotted and ans.spotted[0] == ans.queued and not self.segmenter.speaking:
                spoken = ans.spotted[1]
            while self.listener and checked < len(ans.segments):
                t = ans.segments[checked][1]
                checked += 1
                m = voice.match(t.text, ("done", "stop"), tail=True)
                if m and checked == ans.queued and not self.segmenter.speaking:
                    spoken = m
            # end conditions
            ev = self.face.server.next_event(timeout=0)
            if ev and ev["name"] == "lang":
                self._set_lang(ev.get("value"), "operator")
            if (ev and ev["name"] == "stop") or (spoken and spoken.command == "stop"):
                self._end_answer(ans, "stopped", elapsed)
                raise SessionStopped()
            if ev and ev["name"] == "done":
                return self._end_answer(ans, "done", elapsed)
            if spoken:
                self._lang_from_command(spoken.lang if spoken.lang == "fil" else None)
                return self._end_answer(ans, "voice_done", elapsed)
            if elapsed >= limit:
                return self._end_answer(ans, "time_limit", elapsed)
            if self.segmenter.heard_speech and silence >= a.end_of_answer_silence_s:
                return self._end_answer(ans, "silence", elapsed)

    def _last_words(self, ans: Answer, wait_s: float = 4.0) -> None:
        """The answer ended by silence or time before its last words were transcribed (slow ASR):
        if those words were "stop the interview", stop now; if "that's my answer", record that."""
        if not self.listener or ans.end_reason not in ("silence", "time_limit"):
            return
        end = time.monotonic() + wait_s
        while ans.tails_done < ans.tails and not self._worker.error and time.monotonic() < end:
            time.sleep(0.02)
        m = ans.spotted[1] if ans.spotted and ans.spotted[0] == ans.queued else None
        if not m and len(ans.segments) == ans.queued:
            said = [t.text for _, t in ans.segments if t.text.strip()]
            m = voice.match(said[-1], ("done", "stop"), tail=True) if said else None
        if m and m.command == "stop":
            ans.end_reason = "stopped"
            raise SessionStopped()
        if m:
            ans.end_reason = "voice_done"

    def _worker_put(self, job) -> None:
        self._worker.jobs.put(job)

    def _end_answer(self, ans: Answer, reason: str, elapsed: float) -> Answer:
        ans.end_reason, ans.duration_s = reason, round(elapsed, 1)
        if self.vision:
            ans.eye = self.vision.stop_window()
        self.face.listen("none")
        self.face.panel("none")
        return ans

    def ask_keep(self) -> bool:
        """After the session: keep the report on this device? Anything but a clear "keep" deletes it."""
        self.face.set_state("closing")
        try:
            return self._choose("consent", ["keep", "delete"], "keep_title", line="keep" if self.listener else None,
                                aliases={"yes": "keep", "no": "delete"}) == "keep"
        except SessionStopped:
            return False

    def _finish(self, worker: Worker, saved: bool) -> dict:
        self.face.set_state("thinking")
        self.face.listen("none")
        self.face.panel("none")
        worker.stop()
        worker.join()
        self.meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if worker.error:
            self.meta["worker_error"] = repr(worker.error)
        answers = [a for a in self.answers if a.done.is_set()]
        if answers:      # the report is written in the language most questions were asked in
            asked = [a.asked_lang for a in answers]
            self.meta["language"] = max(set(asked), key=asked.count)
            self.meta["languages_spoken"] = sorted({a.language for a in answers if a.language})
        return {"meta": self.meta, "answers": answers}
