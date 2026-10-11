"""The robot starts in English, says each line once, and waits for the human before moving on."""
import numpy as np
import pytest

from coach.audio.util import blocks
from coach.profile import make_vad, models_dir
from coach.session.state_machine import Session

from .conftest import require
from .test_voice import StubFace, StubListener


class SayFace(StubFace):
    def __init__(self):
        super().__init__()
        self.said = []

    def say(self, speech, keep_subtitle=None, stop_check=None):
        self.said.append(speech)
        return True

    def panel(self, *a, **k):
        pass

    def set_state(self, *a):
        pass

    def subtitle(self, *a, **k):
        pass

    def hud(self, *a):
        pass

    def hear(self, *a):
        pass

    def maybe_nod(self):
        pass


class Speech:
    """Stub speech source: every line exists in both languages; returns (key, lang)."""

    def available(self, key, lang, text):
        return True

    def question(self, key, lang, text):
        return key, lang


def _session(cfg, face=None, mic=None, vad=None):
    s = Session(cfg, face or SayFace(), mic, vad or _Vad(), None, None, None, Speech())
    return s


class _Vad:
    sample_rate, threshold = 16000, 0.5

    def reset(self):
        pass


def test_each_line_is_said_once_in_english_then_in_filipino(cfg):
    s = _session(cfg)
    s._line("consent")
    assert s.face.said == [("line.consent", "en")]        # one language only, starting with English
    s._lang_from_command("fil")
    s._line("consent")
    assert s.face.said[-1] == ("line.consent", "fil") and len(s.face.said) == 2


def test_silence_on_a_choice_asks_again_instead_of_moving_on(cfg):
    s = _session(cfg)
    s.listener = StubListener()                            # nobody ever says anything
    s.vcfg = dict(s.vcfg, choice_timeout_s=0.05, choice_reprompts=1)
    assert s._choose("consent", ["yes", "no"], "consent_title", line="consent") is None
    assert s.face.said == [("line.consent", "en")] * 2     # asked, then asked again, then gave up


def test_a_late_answer_after_the_reprompt_still_counts(cfg):
    from coach.session.voice import Heard

    class LateListener(StubListener):
        def poll(self, commands, numbered=False, timeout=0.1):
            return Heard("Yes", "yes", "en") if self.opened >= 2 else None

    s = _session(cfg)
    s.listener = LateListener()
    s.vcfg = dict(s.vcfg, choice_timeout_s=0.05, choice_reprompts=1)
    assert s._choose("consent", ["yes", "no"], "consent_title", line="consent") == "yes"


# ---------- the answer loop with the real VAD ----------

class ScriptMic:
    """When the session starts listening, plays `audio` placed `at_s` seconds in, then silence."""

    def __init__(self, audio, at_s):
        self.audio, self.at = audio, at_s
        self._blocks = []

    def clear(self):
        lead = np.zeros(int(self.at * 16000), np.float32)
        self._blocks = list(blocks(np.concatenate([lead, self.audio, np.zeros(16000 * 12, np.float32)])))

    def read(self, timeout=1.0):
        return self._blocks.pop(0) if self._blocks else np.zeros(512, np.float32)


class _Worker:
    error = None

    def __init__(self):
        self.jobs = []

    def put(self, job):
        self.jobs.append(job)


class _Q:
    id, text, criteria = "q", {"en": "Question?"}, []


def _listen(cfg, audio, at_s):
    require(models_dir(cfg) / cfg.registry["vad"]["silero"]["file"])
    cfg = dict(cfg)
    cfg["session"] = dict(cfg["session"], answer_time_limit_s=20)
    from coach.profile import Config

    cfg = Config(cfg)
    face = SayFace()
    face.look_at = lambda *a: None
    face.server.clear_events = lambda: None
    s = _session(cfg, face, ScriptMic(audio, at_s), make_vad(cfg))
    s._worker, s._n_questions = _Worker(), 1
    return s._listen(_Q(), "Question?")


def test_robot_echo_right_after_speaking_does_not_start_or_end_the_answer(cfg, speak):
    echo = speak("Next question.")[: int(0.35 * 16000)]   # the tail of the robot's own voice
    ans = _listen(cfg, echo, at_s=0.0)
    assert ans.queued == 0 and ans.end_reason == "no_answer"     # echo ignored: treated as no answer yet
    assert ans.duration_s >= cfg.audio.no_answer_timeout_s        # it waited the full time for the student


def test_answer_is_done_after_5s_of_silence(cfg, speak):
    speech = speak("I built a temperature monitor with an ESP32.")
    ans = _listen(cfg, speech, at_s=1.0)
    assert ans.queued >= 1 and ans.end_reason == "silence"
    spoken_until = 1.0 + len(speech) / 16000
    assert cfg.audio.end_of_answer_silence_s == 5.0
    assert ans.duration_s - spoken_until == pytest.approx(5.0, abs=0.8)   # ~5 s after the last word


def test_profiles_timing(cfg, cfg_pi5):
    for c in (cfg, cfg_pi5):
        assert c.audio.end_of_answer_silence_s == 5.0
        assert c.audio.no_answer_timeout_s > c.audio.take_your_time_s
        assert c.voice.start_language == "en" and c.session.start_on_face is False
        assert c.voice.command_model == "base"


# ---------- end of session: "processing" first, "ready" only when it is true ----------

class _SlowScorer:
    """Stands in for the worker: scores each answer 0.15 s apart in the background."""
    error = None

    def __init__(self, answers, delay=0.15):
        import threading

        self.answers, self.ready_at = answers, None

        def work():
            import time
            for a in answers:
                time.sleep(delay)
                a.done.set()
            self.ready_at = time.monotonic()

        self.t = threading.Thread(target=work, daemon=True)
        self.t.start()

    def is_alive(self):
        return self.t.is_alive()

    def stop(self):
        pass

    def join(self):
        self.t.join()


def _finished_session(cfg, n_answers, already_scored=False):
    import time
    from coach.session.state_machine import Answer

    s = _session(cfg)
    s.face.subtitles = []
    s.face.subtitle = lambda text, persist=False: s.face.subtitles.append(text)
    said_at = []
    s.face.say = lambda speech, **k: (s.face.said.append(speech), said_at.append(time.monotonic()))
    for _ in range(n_answers):
        a = Answer(_Q())
        a.finalizing = True
        if already_scored:
            a.done.set()
        s.answers.append(a)
    worker = _SlowScorer([] if already_scored else s.answers)
    s.meta = {}
    s._finish(worker, saved=True)
    return s, said_at, worker


def test_robot_says_processing_then_ready_only_after_scoring(cfg):
    s, said_at, worker = _finished_session(cfg, 3)
    assert s.face.said == [("line.processing", "en"), ("line.ready", "en")]
    assert said_at[1] >= worker.ready_at                      # "ready" never before the scores exist
    assert any("Processing your results" in t and "of 3" in t for t in s.face.subtitles)


def test_no_processing_line_when_results_are_already_done(cfg):
    s, _, _ = _finished_session(cfg, 2, already_scored=True)
    assert s.face.said == [("line.ready", "en")]


def test_nothing_about_results_when_nothing_was_answered(cfg):
    s, _, _ = _finished_session(cfg, 0)
    assert s.face.said == []


# ---------- robot speech is not cut off ----------

def test_speech_is_padded_so_the_last_word_is_not_cut(monkeypatch):
    from coach.face import controller as ctl
    from coach.face.controller import FaceController
    from coach.tts.cache import build_speech

    played = {}
    monkeypatch.setattr(ctl.sd, "play", lambda audio, sr: played.update(n=len(audio), sr=sr))
    monkeypatch.setattr(ctl.sd, "wait", lambda: None)

    class Server:
        def send(self, msg):
            pass

    sr = 16000
    speech = build_speech("Hello.", np.full(int(0.2 * sr), 0.1, np.float32), sr)
    FaceController(Server()).say(speech)
    assert played["n"] - len(speech.audio) == int(ctl.TAIL_PAD_S * sr) and ctl.TAIL_PAD_S >= 0.3


# ---------- spoken commands are recognized fast ----------

@pytest.fixture(scope="module")
def listener(cfg):
    require(models_dir(cfg) / "whisper" / cfg.voice.command_model / "model.bin")
    from coach.profile import make_command_asr
    from coach.session.voice import VoiceListener

    return VoiceListener(cfg, None, _Vad(), make_command_asr(cfg))


@pytest.mark.parametrize("said, commands, expected", [
    ("Yes.", ["yes", "no"], "yes"),
    ("and it worked. That's my answer.", ["done", "stop"], "done"),
    ("Electronics.", ["CpE", "ECE"], "ECE"),
])
def test_commands_are_recognized_in_about_a_second(listener, speak, said, commands, expected):
    import time

    audio = speak(said)
    listener.recognize(audio, commands)                    # warm-up
    t0 = time.perf_counter()
    heard = listener.recognize(audio, commands)
    took = time.perf_counter() - t0
    assert heard and heard.command == expected
    assert took < 2.5, f"{took:.2f}s (was ~7.5 s with the answer model)"

