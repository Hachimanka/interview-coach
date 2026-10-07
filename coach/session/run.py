"""Run the interview coach (T1.7).

    python -m coach.session.run                  # PC profile, opens the face page in the browser
    python -m coach.session.run --profile pi5    # light profile (test it on the PC too)
    python -m coach.session.run --no-camera      # without vision
    python -m coach.session.run --record         # pilot data collection (T0.5): also saves answer audio

On the Pi, Chromium kiosk opens the printed URL instead of --open (T4.6).
"""
from __future__ import annotations

import argparse
import time
import webbrowser

from coach.audio.capture import MicStream
from coach.face.controller import FaceController
from coach.face.server import FaceServer
from coach.profile import load_profile, make_asr, make_embedder, make_tts, make_vad, resolve
from coach.scoring.rubric import load_bank
from coach.scoring.score import Scorer
from coach.session import report as rpt
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
    vad, asr = make_vad(cfg), make_asr(cfg)
    scorer = Scorer.from_config(cfg, make_embedder(cfg))
    speech = SpeechSource(SpeechCache(resolve(cfg, cfg.paths.cache_dir)), make_tts(cfg))
    print(f"[coach] models ready in {time.perf_counter() - t0:.1f}s")

    server = FaceServer(cfg.face.host, cfg.face.websocket_port).start()
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
                session = Session(cfg, face, mic, vad, asr, scorer, lambda p: load_bank(cfg, p), speech,
                                  vision=vision, record_audio=args.record)
                session.wait_for_student()
                result = session.run()
                report = rpt.build_report(result)
                if result["answers"]:
                    keep = args.record or _ask_keep(face, server, result["meta"].get("language", "en"))
                    if keep:
                        out = sessions_dir / time.strftime("%Y%m%d-%H%M%S")
                        audio = {a.question.id: a.audio for a in result["answers"]} if args.record else None
                        path = rpt.save(report, out, audio)
                        print(f"[coach] report saved: {path}")
                        face.subtitle("Your report is saved." if result["meta"].get("language") != "fil"
                                      else "Na-save na ang iyong report.")
                    else:
                        print("[coach] student chose not to keep the report; nothing saved")
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


def _ask_keep(face, server, lang: str) -> bool:
    face.set_state("closing")
    yes, no = ("Keep it", "Delete it") if lang != "fil" else ("Itago", "Burahin")
    face.panel("consent", [yes, no], "Keep your report on this device?" if lang != "fil"
               else "Itago ang report sa device na ito?")
    server.clear_events()
    while True:
        ev = server.next_event(timeout=60)
        if ev is None or ev["name"] == "stop":
            face.panel("none")
            return False
        if ev["name"] == "consent":
            face.panel("none")
            return ev["value"] == yes


if __name__ == "__main__":
    main()
