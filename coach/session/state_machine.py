"""Session flow (T1.7): wait -> greeting -> consent -> questions -> report.
The student's course is never asked: questions mix CpE and ECE (session.programs).

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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field

import numpy as np

from coach.audio.vad import Segment, Segmenter
from coach.delivery.summary import analyze
from coach.interviewer.memory import SessionMemory
from coach.reply.guard import check
from coach.reply.topics import feeling, find_topics, topic_ack, topic_follow_up
from coach.scoring.rubric import Bank, Question, pick_questions, programs
from coach.session import voice
from coach.session.generative import GenerativeMixin
from coach.session.lines import LINES, label, text as screen_text
from coach.tts.cache import SpeechSource

TAIL_S = 3.5    # seconds at the end of a segment checked first for "that's my answer" / "stop the interview"


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
    finalizing: bool = False           # sent for scoring; done is set when the score exists
    seg_audio: list = field(default_factory=list)   # audio of each speech segment, for the quick probe
    probe: dict | None = None          # preliminary score, used only to choose a follow-up
    probe_done: threading.Event = field(default_factory=threading.Event)
    followup: "Answer | None" = None   # the student's answer to the one follow-up question
    laugh_prob: float = 0.0            # highest YAMNet laughter score over the answer's segments
    laughs_sent: int = 0
    laughs_done: int = 0
    reply_info: dict | None = None     # what the robot replied and why (for the report)
    follow_up_info: dict | None = None  # {"key", "question", "reason"}: what was asked and why
    question_info: dict | None = None  # generative mode: how the question was written (slot, callback, check)
    praise: dict | None = None         # generative mode: what was praised and the score behind it
    result: dict | None = None
    done: threading.Event = field(default_factory=threading.Event)


class Worker(threading.Thread):
    """Background work, on two threads so nothing urgent waits behind something slow:

    - main thread: accurate transcription of each segment ("seg") and the real score ("final"), in order.
    - quick thread: decisions the student is waiting for, with the FAST command model: did the answer
      end with "that's my answer" / "stop the interview" ("tail"), and which follow-up to ask ("probe").
      A running job cannot be interrupted, so with one thread these waited for a slow "seg" to finish.
    """

    QUICK = ("tail", "probe", "laugh")
    laughter = None                        # LaughterDetector, set by the session when enabled

    def __init__(self, asr, scorer, cfg, cmd_asr=None):
        super().__init__(name="asr-score", daemon=True)
        self.asr, self.scorer, self.cfg = asr, scorer, cfg
        self.cmd_asr = cmd_asr or asr      # quick jobs use the fast command model
        self.jobs: queue.Queue = queue.Queue()
        self.quick: queue.Queue = queue.Queue()
        self._quick_thread = threading.Thread(target=self._run_quick, name="asr-quick", daemon=True)
        self.error: Exception | None = None
        self.tail_cost: float | None = None   # seconds the last quick check took on this machine

    def start(self) -> None:
        super().start()
        self._quick_thread.start()

    def put(self, job) -> None:
        if job is None:                    # stop: both threads finish what is queued, then exit
            self.quick.put(None)
            self.jobs.put(None)
        elif job[0] in self.QUICK:
            self.quick.put(job)
        else:
            self.jobs.put(job)

    def join(self, timeout: float | None = None) -> None:
        super().join(timeout)
        self._quick_thread.join(timeout)

    def _run_quick(self) -> None:
        while (job := self.quick.get()) is not None:
            self._handle(*job)

    def tail_wait(self, floor: float = 4.0) -> float:
        """How long to wait for a pending quick check: about 2.5x what it took last time. Whisper
        always encodes a 30 s window, so on a slow CPU even a 3.5 s clip can take several seconds."""
        if self.tail_cost is None:
            return 15.0
        return min(max(floor, 2.5 * self.tail_cost + 1.0), 20.0)

    def run(self) -> None:
        while (job := self.jobs.get()) is not None:
            self._handle(*job)

    def _handle(self, kind: str, ans: "Answer", payload) -> None:
        try:
            if kind == "seg":
                # lock the language for the rest of the answer after the first segment
                t = self.asr.transcribe(payload.audio, language=ans.language)
                if ans.language is None:
                    ans.language, ans.language_prob = t.language, t.language_probability
                ans.segments.append((payload, t))
            elif kind == "tail":
                # Before the whole segment (slow when long): did it END with a spoken command?
                number, audio, lang = payload
                t0 = time.monotonic()
                try:
                    t = self.cmd_asr.transcribe(audio, language=ans.language or lang, prompt="")
                    ans.spotted = (number, voice.match(t.text, ("done", "stop"), tail=True))
                finally:
                    self.tail_cost = time.monotonic() - t0
                    ans.tails_done += 1
            elif kind == "laugh":
                try:
                    if self.laughter is not None:
                        ans.laugh_prob = max(ans.laugh_prob, self.laughter.probability(payload))
                finally:
                    ans.laughs_done += 1
            elif kind == "probe":
                # Rough transcript with the FAST model (~1 s), only to choose a follow-up. Waiting for
                # the accurate transcript took longer than the student would wait on the PC profile.
                audio = np.concatenate(ans.seg_audio) if ans.seg_audio else np.zeros(0, np.float32)
                text = ""
                if len(audio):
                    t = self.cmd_asr.transcribe(audio, language=ans.language or payload, prompt="")
                    text = voice.strip_tail(t.text, list(t.words))[0] if t.text.strip() else ""
                ans.probe = {"words": len(text.split()), "text": text,
                             "score": self.scorer.score(ans.question, text)}
                ans.probe_done.set()
            else:
                ans.result = finalize(ans, self.scorer, self.cfg)
                ans.done.set()
        except Exception as e:  # keep the session alive; the report shows the failure
            self.error = e
            if kind == "probe":
                ans.probe_done.set()
            if kind == "final":
                ans.result = {"error": repr(e)}
                ans.done.set()

    def stop(self) -> None:
        self.put(None)


def _collect(ans: Answer, cfg, offset: float = 0.0) -> tuple[list, str, list]:
    """(words with absolute times, text, speech spans) of one answer part, end phrase removed."""
    words, texts, spans = [], [], []
    spoken = [i for i, (_, t) in enumerate(ans.segments) if t.text.strip()]
    for i, (seg, t) in enumerate(ans.segments):
        seg_text, seg_words = t.text, t.words
        if spoken and i == spoken[-1] and (cfg.get("voice") or {}).get("enabled"):
            seg_text, seg_words = voice.strip_tail(seg_text, seg_words)   # "that's my answer" is not part of it
        start = seg.start + offset
        spans.append((start, seg.end + offset))
        texts.append(seg_text)
        words += [type(w)(w.text, w.start + start, w.end + start, w.probability) for w in seg_words]
    return words, " ".join(x for x in texts if x).strip(), spans


def finalize(ans: Answer, scorer, cfg) -> dict:
    """Score the answer. If a follow-up was asked, its answer is scored together with the main one;
    its timeline is placed 1 s after the main answer so the gap is not counted as a long pause."""
    words, text, spans = _collect(ans, cfg)
    fu_text = ""
    if ans.followup and ans.followup.segments:
        main_end = spans[-1][1] if spans else 0.0
        offset = main_end + 1.0 - ans.followup.segments[0][0].start
        fu_words, fu_text, fu_spans = _collect(ans.followup, cfg, offset)
        words, spans = words + fu_words, spans + fu_spans
    full = " ".join(x for x in (text, fu_text) if x)
    delivery = analyze(full, words, spans, 0.0, cfg, ans.eye.get("eye_contact_pct"))
    score = scorer.score(ans.question, full)
    result = {"transcript": text, "language": ans.language, "asked_in": ans.asked_lang,
              "score": score, "delivery": delivery}
    if ans.follow_up_info:
        result["follow_up"] = dict(ans.follow_up_info, transcript=fu_text,
                                   end_reason=ans.followup.end_reason if ans.followup else "")
    if ans.reply_info:
        result["reply"] = dict(ans.reply_info)   # what the robot said back and why (traceable)
    if ans.question_info:
        result["question_info"] = dict(ans.question_info)
    if ans.praise:
        result["praise"] = dict(ans.praise)
    return result


class Session(GenerativeMixin):
    def __init__(self, cfg, face, mic, vad, asr, scorer, bank_loader, speech: SpeechSource,
                 vision=None, record_audio: bool = False, cmd_asr=None, rephraser=None, laughter=None,
                 interviewer=None, knowledge=None):
        self.cfg, self.face, self.mic, self.vision = cfg, face, mic, vision
        a = cfg.audio
        self.segmenter = Segmenter(vad, a.min_speech_ms, a.segment_silence_ms, a.max_segment_s)
        self.asr, self.scorer, self.bank_loader, self.speech = asr, scorer, bank_loader, speech
        self.cmd_asr = cmd_asr or asr          # fast model for spoken commands (make_command_asr)
        self.rephraser, self.laughter = rephraser, laughter   # local AI + YAMNet (None = off)
        self._laughed = False                  # mirrored laughter already used this session
        self._used_topics: set[str] = set()
        self._ai_pool = None
        # Generative mode (coach/session/generative.py): the AI writes questions and replies, and remembers
        # the whole session. RAM only; forgotten when the session ends (forget() / a new Session).
        self.interviewer, self.knowledge = interviewer, knowledge
        self.icfg = cfg.get("interviewer") or {}
        self.memory = SessionMemory(name_every=self.icfg.get("name_every", 2))
        self._next_q = None                    # future of the next AI-written question
        self.record_audio = record_audio
        self.vcfg = cfg.get("voice") or {}
        self.listener = voice.VoiceListener(cfg, mic, vad, self.cmd_asr, face) if self.vcfg.get("enabled") else None
        # One language per session, never mixed: the robot starts in English, the student's FIRST spoken
        # reply decides the language once ("simulan" / "oo" = Filipino), and it is locked from the first
        # question to the end. Answers, end phrases and later replies never change it.
        self.lang: str = self.vcfg.get("start_language") or "en"
        self._sync_listener()
        self.lang_decided = False         # set by the first spoken reply, or when the interview begins
        self.lang_locked = False          # an operator pressed E / F
        self._fil_ready: bool | None = None
        self.answers: list[Answer] = []
        self.meta: dict = {}
        self._worker: Worker | None = None
        self._n_questions = 0
        self._warned: set[str] = set()

    # ---------- language ----------
    @property
    def _l(self) -> str:
        return self.lang or "en"

    @property
    def _whisper_lang(self) -> str:
        return "tl" if self.lang == "fil" else "en"

    def _sync_listener(self) -> None:
        if self.listener:
            self.listener.prefer = self._whisper_lang   # commands are decoded in this language first

    def filipino_ready(self) -> bool:
        """Filipino is used only if EVERY line and question has a Filipino clip; otherwise the whole
        session stays English (a missing clip must never make one line English mid-session)."""
        if self._fil_ready is None:
            available = getattr(self.speech, "available", None)
            if available is None:
                self._fil_ready = True
            else:
                items = [(f"line.{k}", v) for k, v in LINES.items()]
                if callable(self.bank_loader):
                    for q in self.bank_loader().questions:
                        items.append((q.id, q.text))
                        items += [(f"{q.id}.fu.{c.id}", c.follow_up) for c in q.criteria if c.follow_up]
                missing = [k for k, t in items if not (t.get("fil") and available(k, "fil", t["fil"]))]
                self._fil_ready = not missing
                if missing:
                    print(f"[coach] {len(missing)} Filipino clip(s) missing (e.g. {missing[0]}); this session "
                          f"stays in English. Run: python -m offline.synth_questions")
        return self._fil_ready

    def _set_lang(self, lang: str | None, source: str) -> None:
        if lang not in ("en", "fil"):
            return
        if lang == "fil" and not self.filipino_ready():
            lang = "en"
        if source == "operator":                 # staff pressed E / F: always obeyed
            self.lang, self.lang_locked, self.lang_decided = lang, True, True
        elif not (self.lang_locked or self.lang_decided):
            if self.vcfg.get("auto_language", True) and lang != self.lang:
                self.lang = lang
                print(f"[coach] language -> {lang} (from the student's first reply; locked for this session)")
            self.lang_decided = True
        self._sync_listener()

    def _lang_from_command(self, lang: str | None) -> None:
        """Only the student's FIRST spoken reply that says something about language decides it
        ("start"/"yes" = English, "simulan"/"oo" = Filipino). Neutral words ("okay", "computer") don't."""
        self._set_lang(lang, "command")

    def _lock_language(self) -> None:
        """The interview is starting: from here to the end, one language only."""
        self.lang_decided = True
        self.meta["language"] = self._l
        print(f"[coach] interview language: {self._l}")

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
        """Say a fixed line once, in the current session language."""
        speech, _, _ = self._clip(f"line.{key}", LINES[key])
        self.face.say(speech, stop_check=self._stop_pressed)

    def _tick(self) -> bool:
        if self.vision:
            self._ticks = getattr(self, "_ticks", 0) + 1
            if self._ticks % 3 == 0:
                self.face.look_at(self.vision.latest)
        return False

    def _choose(self, kind: str, commands: list[str], title_key: str, line: str | None = None,
                aliases: dict[str, str] | None = None) -> str | None:
        """Show buttons, optionally say a line, then wait for the student's spoken or tapped choice.
        Silence never moves the session on by itself: after choice_timeout_s the robot asks again
        (choice_reprompts times). Returns the chosen command, or None if nobody ever answered."""
        options = [(c, label(c, self.lang)) for c in commands]
        hints = [voice.hint(c, self.lang) for c in commands] if self.listener else []
        self.face.panel(kind, [lab for _, lab in options], screen_text(title_key, self.lang), hints)
        if line:
            self._line(line)
        reprompts = self.vcfg.get("choice_reprompts", 1) if self.listener else 0
        while True:
            choice = voice.wait_choice(
                self.face, self.listener, kind, options, aliases=aliases,
                timeout=self.vcfg.get("choice_timeout_s") if self.listener else None,
                retries=self.vcfg.get("retries", 2), tick=self._tick,
                on_retry=lambda: self._line("retry"), on_lang=lambda v: self._set_lang(v, "operator"))
            if choice.how != "timeout" or reprompts <= 0:
                break
            reprompts -= 1
            self._line(line or "retry")          # still nobody: ask the same thing again
        self.face.panel("none")
        if choice.how == "stop":
            raise SessionStopped()
        if choice.how == "voice":
            self._lang_from_command(choice.lang)
        return choice.command

    # ---------- flow ----------
    def wait_for_student(self) -> None:
        """Wait for the student: "start" / "simulan", a tap, or Enter. A face only makes the eyes
        follow, unless session.start_on_face is true (then ~1 s in front of the camera also starts)."""
        self.face.set_state("idle")
        self.face.hud("", "")
        self.face.subtitle(screen_text("start_voice" if self.listener else "start_touch", self.lang))
        self.face.panel("choice", [label("start", self.lang)], "",
                        [voice.hint("start", self.lang)] if self.listener else [])
        start_on_face = bool(self.cfg.session.get("start_on_face", False))
        seen_since = None

        def tick() -> bool:
            nonlocal seen_since
            if not self.vision:
                return False
            obs = self.vision.latest
            self.face.look_at(obs)
            seen_since = (seen_since or time.monotonic()) if obs.present else None
            return start_on_face and bool(seen_since and time.monotonic() - seen_since > 1.0)

        while True:
            choice = voice.wait_choice(self.face, self.listener, "choice", [("start", label("start", self.lang))],
                                       also=("done",), retries=0, tick=tick,
                                       on_lang=lambda v: self._set_lang(v, "operator"))
            if choice.how != "stop":          # "stop" means nothing while idle
                break
        if choice.how == "voice":
            self._lang_from_command(choice.lang)
        self.face.panel("none")
        self.face.subtitle("")

    def run(self) -> dict:
        worker = Worker(self.asr, self.scorer, self.cfg, self.cmd_asr)
        worker.laughter = self.laughter
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

            # The robot never asks the student's course: every session mixes common questions with
            # technical questions from each program in scope (CpE and ECE), see pick_questions().
            s = self.cfg.session
            bank: Bank = self.bank_loader()
            questions = pick_questions(bank, s.questions_per_session, s.get("vary_questions", True))
            self._n_questions = len(questions)
            self.meta.update(program=" & ".join(programs(self.cfg)), questions=[q.id for q in questions],
                             sample_rubric=bank.is_sample, rubric_source=bank.source)
            self._lock_language()
            gen = self.generative                  # decided once, with the language
            self.meta["mode"] = "generative" if gen else "scripted"
            if gen:
                self._ask_name()
                self._next_q = self._prefetch_question(1, questions[0])   # written while instructions play
            self._line("instructions" + touch)

            for i, q in enumerate(questions, 1):
                nxt = questions[i] if i < len(questions) else None
                info = None
                if gen:
                    q, speech, text, info = self._generated_question(i, q)
                    lang = "en"
                else:
                    speech, text, lang = self._clip(q.id, q.text)
                self.face.hud("", f"Q{i} / {len(questions)}")
                self.face.say(speech, keep_subtitle=text, stop_check=self._stop_pressed)
                ans = self._listen(q, text, lang)
                ans.question_info = info
                self._flush_into(ans)
                self.face.set_state("thinking")
                self._last_words(ans)
                self._maybe_follow_up(q, ans, lang, i=i, nxt=nxt if gen else None)
                worker.put(("final", ans, None))
                ans.finalizing = True
                if gen and nxt is not None and self._next_q is None:   # no follow-up was asked
                    said = self._remember(i, q, ans)
                    self._next_q = self._prefetch_question(i + 1, nxt, self._answer_text(ans))
                    self._remember_ai(i, said)
                if i < len(questions) and not gen:
                    self._line("next")
            self.face.set_state("closing")
            if gen:
                self._say_text(self._closing_text())
            else:
                self._line("closing")
        except SessionStopped:
            self.meta["ended_early"] = True
        return self._finish(worker, saved=True)

    # ---------- replying to the answer ----------
    def _flush_into(self, ans: Answer) -> None:
        for seg in self.segmenter.flush():
            ans.seg_audio.append(seg.audio)
            self._send_laugh(ans, seg.audio)
            self._worker.put(("seg", ans, seg))
            ans.queued += 1

    def _send_laugh(self, ans: Answer, audio) -> None:
        if self.laughter is not None:
            ans.laughs_sent += 1
            self._worker.put(("laugh", ans, audio))

    def _student_laughed(self, ans: Answer, wait_s: float = 1.0) -> bool:
        """Mirror laughter only if the student laughed, and at most once per session."""
        if self.laughter is None or self._laughed:
            return False
        end = time.monotonic() + wait_s
        while ans.laughs_done < ans.laughs_sent and time.monotonic() < end:
            time.sleep(0.02)
        return ans.laugh_prob >= self.laughter.threshold

    def _maybe_follow_up(self, q: Question, ans: Answer, lang: str, i: int | None = None, nxt=None) -> None:
        """Reply to what the student said:
        1. mirror their laughter (once per session) or say a neutral acknowledgment (covers the wait);
        2. reassure them if they said they were nervous or stuck;
        3. mention their topic and/or ask at most ONE follow-up. The meaning always comes from an approved
           line (topic template, rubric follow-up, "tell me more"); in English sessions the local AI
           rephrases it to refer to their words, and guard.py checks the result before it is spoken."""
        s = self.cfg.session
        if not s.get("follow_ups") or ans.end_reason not in ("silence", "voice_done", "done", "time_limit"):
            return
        self._worker.put(("probe", ans, self._whisper_lang))   # preliminary score, ready while the robot speaks
        info: dict = {"laughed": False}
        ans.reply_info = info
        if self._student_laughed(ans):
            self._laughed = info["laughed"] = True
            speech, _, _ = self._clip("line.laugh", LINES["laugh"])
            self.face.say(speech, stop_check=self._stop_pressed, expression="amused")
        else:
            self._ack_n = getattr(self, "_ack_n", 0) + 1
            self._line(f"ack_{(self._ack_n - 1) % 3 + 1}")
        plan = self._plan_reply(q, ans)
        gen = self.generative and i is not None
        if gen:
            ans.praise = self._maybe_praise(ans)     # only for a criterion the scorer rated high
            if ans.praise:
                info["praise"] = ans.praise["said"]
        if not plan:
            return
        info.update({k: plan[k] for k in ("kind", "reason", "feeling")}, topic=plan["topic"] and plan["topic"].name)
        if plan["feeling"]:
            self._line("reassure")
        if not plan["kind"] or (gen and ans.praise and not plan["asks"]):
            return                                   # praise already acknowledged the answer
        if gen and plan["asks"]:
            text, speech, source, check_note = self._generative_follow_up(plan, q, ans)
        else:
            text, speech, source, check_note = self._realize(plan, ans)
        info.update(said=text, source=source, approved=plan["approved"], check=check_note)
        print(f"[coach] reply ({plan['reason']}, {source}): {text}")
        if not plan["asks"]:
            self.face.say(speech, stop_check=self._stop_pressed)
            return
        fu_text = text
        ans.follow_up_info = {"key": plan["key"] or plan["kind"], "question": fu_text, "reason": plan["reason"],
                              "source": source}
        self.face.say(speech, keep_subtitle=fu_text, stop_check=self._stop_pressed)
        if gen and nxt is not None:
            # Write the next question while the student answers the follow-up. It has no "reaction" part:
            # the follow-up answer is acknowledged by the robot's short ack line instead.
            said = self._remember(i, q, ans)
            self._next_q = self._prefetch_question(i + 1, nxt)
            self._remember_ai(i, said)
        try:
            fu = self._listen(q, fu_text, lang, record=False, kind="followup")
        except SessionStopped:
            self._worker.put(("final", ans, None))  # keep the main answer even if Stop is pressed now
            ans.finalizing = True
            raise
        self._flush_into(fu)
        self.face.set_state("thinking")
        self._last_words(fu)
        ans.followup = fu
        if gen and fu.end_reason != "no_answer":
            self._ack_n = getattr(self, "_ack_n", 0) + 1
            self._line(f"ack_{(self._ack_n - 1) % 3 + 1}")     # acknowledges the follow-up answer
            said = self._answer_text(fu)
            if said:     # rules only (topics, feelings); the next question is already being written
                self.memory.add_answer(i, self.memory.slots.get(i, q.bank), said)

    def _plan_reply(self, q: Question, ans: Answer) -> dict | None:
        """Choose what to reply (meaning only; wording comes later). Order:
        topic follow-up (they mentioned a glossary topic and the answer was short or not yet strong)
        > rubric follow-up / "tell me more" (_pick_follow_up) > topic acknowledgment > nothing.
        Topic replies are English-only (no live Filipino voice); Filipino sessions use approved clips."""
        s, r = self.cfg.session, (self.cfg.get("reply") or {})
        base = self._pick_follow_up(q, ans)          # also waits for the preliminary score
        if not ans.probe:
            return None
        said, words = ans.probe.get("text", ""), ans.probe["words"]
        feel = feeling(said) if said else None
        topic = None
        if r.get("topic_replies") and said and self._l == "en":
            topic = next((t for t in find_topics(said) if t.name not in self._used_topics), None)
        crits = ans.probe["score"].get("criteria") or []
        weakest = min((c["expected"] for c in crits), default=5.0)
        plan = {"feeling": feel, "topic": topic, "kind": None, "key": None, "texts": None, "asks": False,
                "reason": "", "approved": ""}
        not_strong = words < s.get("follow_up_min_words", 15) or weakest <= s.get("follow_up_below", 3.0) + 1.0
        if topic and words and not_strong:
            plan.update(kind="topic_follow_up", texts={"en": topic_follow_up(topic)}, asks=True,
                        reason=f"mentioned '{topic.name}' (answer not yet strong: weakest {weakest})")
        elif base:
            key, texts, reason = base
            plan.update(kind="tell_more" if key == "line.tell_more" else "rubric_follow_up", key=key,
                        texts=texts, asks=True, reason=reason)
        elif topic and words:
            plan.update(kind="topic_ack", texts={"en": topic_ack(topic)}, reason=f"mentioned '{topic.name}'")
        if not plan["kind"] and not feel:
            return None
        if plan["kind"] in ("topic_follow_up", "topic_ack"):
            self._used_topics.add(topic.name)        # never "You mentioned the ESP32" twice in a session
        if plan["texts"]:
            plan["approved"] = plan["texts"].get(self._l) or plan["texts"]["en"]
        return plan

    def _realize(self, plan: dict, ans: Answer):
        """(text, speech, source, check note). English: the local AI rephrases the approved line to refer
        to the student's words; guard.check() must pass, else the approved line is spoken as-is."""
        r = self.cfg.get("reply") or {}
        approved = plan["approved"]
        topic = plan["topic"].name if plan["topic"] else None
        note = "ai off"
        if self._l == "en" and self.rephraser is not None:
            said = ans.probe.get("text", "") if ans.probe else ""
            fut = self._ai().submit(self.rephraser.rephrase, approved, said, topic)
            try:
                out = fut.result(timeout=r.get("llm_timeout_s", 5))
                ok, note = check(out, approved, said, topic, r.get("llm_max_words", 25))
                if ok:
                    return out, self.speech.live(out, "en"), "ai", "ok"
                note = f"rejected: {note}"
            except FuturesTimeout:
                note = "ai too slow"
            except Exception as e:   # the AI must never break the interview
                note = f"ai error: {e!r}"
        if plan["key"]:                              # approved clip, pre-recorded in both languages
            speech, text, _ = self._clip(plan["key"], plan["texts"])
            return text, speech, "approved", note
        return approved, self.speech.live(approved, "en"), "template", note

    def _ai(self):
        if self._ai_pool is None:
            self._ai_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reply-ai")
        return self._ai_pool

    def _pick_follow_up(self, q: Question, ans: Answer):
        """(clip key, texts, reason) of the one follow-up to ask, or None. Every choice is traceable:
        the reason is stored in the report."""
        s = self.cfg.session
        if not ans.probe_done.wait(timeout=s.get("follow_up_wait_s", 8)) or not ans.probe:
            return None                              # score not ready in time: just move on
        words, score = ans.probe["words"], ans.probe["score"]
        if words == 0:
            return None                              # nothing was recognized: do not interrogate
        if words < s.get("follow_up_min_words", 15):
            return "line.tell_more", LINES["tell_more"], f"short answer ({words} words)"
        weakest = min(score["criteria"], key=lambda c: c["expected"])
        if weakest["expected"] <= s.get("follow_up_below", 3.0):
            crit = next((c for c in q.criteria if c.id == weakest["criterion_id"]), None)
            if crit and crit.follow_up:
                return (f"{q.id}.fu.{crit.id}", crit.follow_up,
                        f"'{crit.name}' scored {weakest['expected']} (follow-up approved in the rubric)")
        return None

    def _listen(self, q: Question, text: str, lang: str = "en", record: bool = True,
                kind: str = "answer") -> Answer:
        ans = Answer(q, asked_lang=lang)
        if record:                             # follow-up answers belong to their main answer
            self.answers.append(ans)
        self.face.set_state("listening")
        self.face.panel("answer", hints=[voice.hint("done", self.lang)] if self.listener else [])
        self.face.server.clear_events()
        self.mic.clear()                       # drop the robot's own voice
        self.segmenter.reset()
        self.face.listen(kind)
        if self.vision:
            self.vision.start_window()
        a, s = self.cfg.audio, self.cfg.session
        sr = a.sample_rate
        limit, last_hud, waiting = s.answer_time_limit_s, -1, False
        checked = 0                            # segments already checked for a spoken command
        # The speaker and the room keep sounding briefly after the robot's audio call returns; if the
        # mic heard that, the robot would treat its own echo as the start of the answer and move on.
        guard = int(a.get("echo_guard_ms", 400) / 1000 * sr / 512)
        while True:
            block = self.mic.read(timeout=0.5)
            if block is not None and guard > 0:
                guard -= 1
                continue
            self.face.hear(block)
            if block is not None:
                if self.record_audio:
                    ans.audio.append(block)
                for seg in self.segmenter.process(block):
                    ans.queued += 1
                    ans.seg_audio.append(seg.audio)
                    self._send_laugh(ans, seg.audio)
                    if self.listener and not self.segmenter.speaking:      # closed by a pause
                        ans.tails += 1
                        self._worker_put(("tail", ans, (ans.queued, seg.audio[-int(TAIL_S * sr):],
                                                        self._whisper_lang)))
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
            # "Spoke" = at least one real speech segment (>= min_speech_ms). A click or an echo blip
            # is too short to become a segment, so it can neither start nor end the answer.
            spoke = ans.queued > 0
            # Not started yet after take_your_time_s: encourage. Once the student has spoken,
            # end_of_answer_silence_s of silence means the answer is finished (no extra prompt).
            if not spoke and silence > a.get("take_your_time_s", 5.0) and not waiting:
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
            if spoken:     # "tapos na po" in an English interview ends the answer but changes nothing else
                return self._end_answer(ans, "voice_done", elapsed)
            if elapsed >= limit:
                return self._end_answer(ans, "time_limit", elapsed)
            if spoke and silence >= a.end_of_answer_silence_s:
                return self._end_answer(ans, "silence", elapsed)
            if not spoke and not self.segmenter.speaking and silence >= a.get("no_answer_timeout_s", 15.0):
                return self._end_answer(ans, "no_answer", elapsed)

    def _last_words(self, ans: Answer, wait_s: float | None = None) -> None:
        """The answer ended by silence or time before its last words were transcribed (slow ASR):
        if those words were "stop the interview", stop now; if "that's my answer", record that.
        Waits only until the pending quick check finishes, capped by how long checks take here."""
        if not self.listener or ans.end_reason not in ("silence", "time_limit"):
            return
        if wait_s is None:
            tail_wait = getattr(self._worker, "tail_wait", None)
            wait_s = tail_wait() if tail_wait else 4.0
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
        self._worker.put(job)

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

    def _say_quietly(self, key: str) -> None:
        """Say a closing line; Stop at this point only skips the line (results are still processed)."""
        try:
            self._line(key)
        except SessionStopped:
            pass

    def _finish(self, worker: Worker, saved: bool) -> dict:
        """Wait for every sent answer to be scored. The robot says it is processing while that is still
        running, and says the results are ready only once they really are."""
        self.face.set_state("thinking")
        self.face.listen("none")
        self.face.panel("none")
        pending = [a for a in self.answers if a.finalizing]
        if pending and not all(a.done.is_set() for a in pending):
            self._say_quietly("processing")
            self.face.set_state("thinking")
            last = None
            while not all(a.done.is_set() for a in pending) and worker.is_alive() and not worker.error:
                n = sum(a.done.is_set() for a in pending)
                if n != last:
                    last = n
                    self.face.subtitle(screen_text("processing", self._l).format(done=n, total=len(pending)),
                                       persist=True)
                time.sleep(0.1)
        worker.stop()
        worker.join()
        if self.meta.get("mode") == "generative" and any(a.done.is_set() for a in pending):
            self.face.set_state("thinking")
            self.meta["coach_summary"] = self._coach_summary([a for a in pending if a.done.is_set()])
            if self.memory.name:
                self.meta["student_name"] = self.memory.name     # saved only if the student keeps the report
        if pending and any(a.done.is_set() for a in pending):
            self.face.subtitle("")
            self.face.set_state("closing")
            self._say_quietly("ready")
        self.meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if worker.error:
            self.meta["worker_error"] = repr(worker.error)
        answers = [a for a in self.answers if a.done.is_set()]
        if answers:      # the report is written in the language most questions were asked in
            asked = [a.asked_lang for a in answers]
            self.meta["language"] = max(set(asked), key=asked.count)
            self.meta["languages_spoken"] = sorted({a.language for a in answers if a.language})
        return {"meta": self.meta, "answers": answers}
