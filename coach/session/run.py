"""Run the interview coach (T1.7).

    python -m coach.session.run                  # PC profile, opens the face page in the browser
    python -m coach.session.run --profile pi5    # light profile (test it on the PC too)
    python -m coach.session.run --no-camera      # without vision
    python -m coach.session.run --record         # pilot data collection (T0.5): also saves answer audio

On the Pi, Chromium kiosk opens the printed URL instead of --open (T4.6).
"""
from __future__ import annotations

import os

# Set before any model library is imported. setdefault: set the variable yourself to override.
# Offline: models come only from models/ (python -m offline.download_models fetches them). Without this,
# sentence-transformers asks the Hugging Face Hub for updates at every start ("unauthenticated requests").
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# (MediaPipe's startup log lines are silenced in coach/vision/face.py; it ignores logging env vars.)

import argparse  # noqa: E402
import time  # noqa: E402
import webbrowser  # noqa: E402

from coach.audio.capture import MicStream
from coach.face.controller import FaceController
from coach.face.server import FaceServer
from coach.profile import (interviewer_mode, load_profile, make_asr, make_command_asr, make_embedder,
                           make_interviewer, make_knowledge, make_laughter, make_llm, make_rephraser, make_tts,
                           make_vad, resolve)
from coach.scoring.rubric import load_session_bank
from coach.scoring.score import Scorer
from coach.session import report as rpt
from coach.session.lines import text as screen_text
from coach.session.state_machine import Session
from coach.tts.cache import SpeechCache, SpeechSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pc")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser (kiosk opens it)")
    ap.add_argument("--record", action="store_true", help="save answer audio (pilot data collection)")
    ap.add_argument("--once", action="store_true", help="run one session and exit")
    args = ap.parse_args()

    cfg = load_profile(args.profile)
    print(f"[coach] profile={cfg.profile}  loading models...")
    t0 = time.perf_counter()
    vad, asr, cmd_asr = make_vad(cfg), make_asr(cfg), make_command_asr(cfg)
    scorer = Scorer.from_config(cfg, make_embedder(cfg))
    llm, laughter = make_llm(cfg), make_laughter(cfg)       # one local AI model, shared
    if interviewer_mode(cfg) == "generative":
        # the AI writes questions and replies; the CpE/ECE library uses the scoring embedder
        interviewer, knowledge, rephraser = make_interviewer(cfg, llm), make_knowledge(cfg, scorer.emb), None
        print(f"[coach] interviewer: generative ({'ready' if interviewer else 'model missing: scripted'})")
    else:
        interviewer, knowledge, rephraser = None, None, make_rephraser(cfg, llm)
    speech = SpeechSource(SpeechCache(resolve(cfg, cfg.paths.cache_dir)), make_tts(cfg))
    print(f"[coach] models ready in {time.perf_counter() - t0:.1f}s")

    server = FaceServer(cfg.face.host, cfg.face.websocket_port, cfg.face.fps, cfg.face.effects).start()
    face = FaceController(server)
    print(f"[coach] face page: {server.url}")
    if not args.no_open:
        webbrowser.open(server.url)

    vision = None
    if not args.no_camera:
        from coach.vision.loop import VisionLoop

        vision = VisionLoop(cfg).start()

    sessions_dir = resolve(cfg, cfg.paths.sessions_dir)
    try:
        with MicStream(cfg.audio.sample_rate, cfg.audio.input_device) as mic:
            while True:
                session = Session(cfg, face, mic, vad, asr, scorer, lambda: load_session_bank(cfg), speech,
                                  vision=vision, record_audio=args.record, cmd_asr=cmd_asr,
                                  rephraser=rephraser, laughter=laughter, interviewer=interviewer,
                                  knowledge=knowledge)
                session.wait_for_student()
                result = session.run()
                report = rpt.build_report(result)
                if result["answers"]:
                    if args.record or session.ask_keep():
                        out = sessions_dir / time.strftime("%Y%m%d-%H%M%S")
                        audio = {a.question.id: a.audio for a in result["answers"]} if args.record else None
                        path = rpt.save(report, out, audio)
                        print(f"[coach] report saved: {path}")
                        face.subtitle(screen_text("saved", result["meta"].get("language", "en")))
                    else:
                        print("[coach] student chose not to keep the report; nothing saved")
                session.memory.forget()        # the robot forgets the student (name, what they said)
                s = report["summary"]
                print(f"[coach] session done: {s.get('answers', 0)} answers, average {s.get('average_score', '-')}")
                time.sleep(3)
                if args.once:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if vision:
            vision.stop()
        server.stop()


if __name__ == "__main__":
    main()
