"""Download every model a profile needs into models/ (T1.8).

Usage:
    python -m offline.download_models --profile pc
    python -m offline.download_models --profile pi5 --all-asr   # also fetch tiny/base/small for T2.1
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from coach.profile import load_profile, models_dir


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok      {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def silero(entry: dict, out: Path) -> None:
    dest = out / entry["file"]
    if not dest.exists():
        try:  # PC: reuse the copy shipped inside the silero-vad package
            import silero_vad

            src = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
            if src.exists():
                shutil.copy(src, dest)
        except ImportError:
            pass
    fetch(entry["url"], dest)


def whisper(size: str, repo: str, out: Path) -> None:
    from faster_whisper import download_model

    target = out / "whisper" / size
    print(f"  whisper {size} -> {target.relative_to(out.parent)}")
    download_model(repo, output_dir=str(target))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="pc")
    ap.add_argument("--all-asr", action="store_true", help="fetch tiny, base and small")
    ap.add_argument("--all-llm", action="store_true", help="fetch both reply models (PC 1.5B and Pi 0.5B)")
    args = ap.parse_args()

    cfg = load_profile(args.profile)
    reg = cfg.registry
    out = models_dir(cfg)
    out.mkdir(exist_ok=True)

    print("VAD"); silero(reg["vad"]["silero"], out)

    print("Vision")
    for entry in reg["vision"].values():
        fetch(entry["url"], out / entry["file"])

    print("TTS")
    for name, url in reg["tts"][cfg.tts.english_voice]["files"].items():
        fetch(url, out / name)

    print("ASR")
    cmd_model = (cfg.get("voice") or {}).get("command_model")
    sizes = list(reg["asr"]) if args.all_asr else list(dict.fromkeys([cfg.asr.model, cmd_model] if cmd_model else [cfg.asr.model]))
    for size in sizes:
        whisper(size, reg["asr"][size]["repo"], out)

    print("Audio events (laughter)")
    for entry in reg.get("audio_events", {}).values():
        fetch(entry["url"], out / entry["file"])

    reply = cfg.get("reply") or {}
    if reply.get("llm"):
        print("Local reply model (llama.cpp)")
        names = list(reg["llm"]) if args.all_llm else [reply["llm_model"]]
        for name in names:
            entry = reg["llm"][name]
            if not entry.get("url"):       # fine-tuned on Colab (training/interviewer): copy the file in yourself
                state = "ok" if (out / "llm" / entry["file"]).exists() else "MISSING: copy it from Colab (G4)"
                print(f"  {state}  {entry['file']}")
                continue
            fetch(entry["url"], out / "llm" / entry["file"])

    if cfg.scoring.backend == "sentence_transformers":
        print("Scoring embedder")
        from coach.profile import make_embedder

        make_embedder(cfg).encode(["warm-up"])  # downloads into models/embedders
        print("  ok")
    else:
        onnx_dir = out / reg["scoring"][cfg.scoring.embedder]["onnx_dir"]
        state = "ok" if onnx_dir.exists() else "MISSING: export it on the PC (T4.2)"
        print(f"Scoring embedder (onnx)\n  {state}  {onnx_dir.name}")


if __name__ == "__main__":
    main()
