"""Pre-synthesize every approved question (T1.3) with subtitle cues + lip-sync levels (T1.10).

English: Piper. Filipino: Meta MMS-TTS tgl (PC only; uses torch + transformers).
Also caches the fixed robot lines in coach/session/lines.py.

Usage:
    python -m offline.synth_questions            # EN + FIL
    python -m offline.synth_questions --lang en  # English only
"""
from __future__ import annotations

import argparse

import numpy as np

from coach.profile import load_profile, make_tts, models_dir, resolve
from coach.scoring.rubric import all_questions
from coach.session.lines import LINES
from coach.tts.cache import SpeechCache, build_speech


class MMSTagalog:
    def __init__(self, cache_dir):
        import torch
        from transformers import AutoTokenizer, VitsModel

        repo = "facebook/mms-tts-tgl"
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir)
        self.model = VitsModel.from_pretrained(repo, cache_dir=cache_dir).eval()
        self.sample_rate = self.model.config.sampling_rate

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        with self.torch.no_grad():
            wav = self.model(**self.tok(text, return_tensors="pt")).waveform[0].numpy()
        return wav.astype(np.float32), self.sample_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "fil", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="re-synthesize existing clips")
    args = ap.parse_args()

    cfg = load_profile("pc")
    cache = SpeechCache(resolve(cfg, cfg.paths.cache_dir))
    langs = ["en", "fil"] if args.lang == "all" else [args.lang]
    engines = {}
    if "en" in langs:
        engines["en"] = lambda t, tts=make_tts(cfg): tts.synthesize(t, "en")
    if "fil" in langs:
        mms = MMSTagalog(models_dir(cfg) / "mms")
        engines["fil"] = mms.synthesize

    items = [(q.id, q.text) for q in all_questions(cfg)] + [(f"line.{k}", v) for k, v in LINES.items()]
    made = 0
    for key, texts in items:
        for lang in langs:
            text = texts.get(lang)
            if not text or (cache.text(key, lang) == text and not args.force):
                continue
            audio, sr = engines[lang](text)
            cache.save(key, lang, text, build_speech(text, audio, sr))
            made += 1
            print(f"  {key}.{lang}  {len(audio) / sr:.1f}s")
    print(f"done: {made} new clips in {cache.dir}")


if __name__ == "__main__":
    main()
