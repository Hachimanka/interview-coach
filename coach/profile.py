"""Load config/pc.yaml or config/pi5.yaml and build the matching backends (T1.8).

Switching profile changes models with no code edits (principles P2/P3). Backends are
imported lazily inside the factories, so importing this module never pulls in torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class Config(dict):
    """dict with attribute access: cfg.asr.model == cfg["asr"]["model"]."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(value) if isinstance(value, dict) else value


def load_profile(name: str = "pc") -> Config:
    """Load a profile by name ("pc", "pi5") or by path to a YAML file."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    with open(CONFIG_DIR / "models.yaml", encoding="utf-8") as f:
        cfg["registry"] = yaml.safe_load(f)
    return cfg


def resolve(cfg: Config, relative: str) -> Path:
    """Paths in the config are relative to the project root."""
    return PROJECT_ROOT / relative


def models_dir(cfg: Config) -> Path:
    return resolve(cfg, cfg.paths.models_dir)


# ---------- factories ----------

def make_vad(cfg: Config):
    from coach.audio.vad import SileroVAD

    return SileroVAD(models_dir(cfg) / cfg.registry["vad"]["silero"]["file"],
                     sample_rate=cfg.audio.sample_rate, threshold=cfg.audio.vad_threshold)


def make_asr(cfg: Config):
    backend = cfg.asr.backend
    if backend == "faster_whisper":
        from coach.asr.faster_whisper_backend import FasterWhisperBackend

        return FasterWhisperBackend(cfg)
    if backend == "whispercpp":
        from coach.asr.whispercpp_backend import WhisperCppBackend

        return WhisperCppBackend(cfg)
    raise ValueError(f"Unknown asr.backend: {backend}")


def make_command_asr(cfg: Config):
    """Small, fast Whisper for short spoken commands ("yes", "that's my answer", "stop the interview").
    The answer model (accurate, slower) keeps transcribing answers for scoring."""
    v = cfg.get("voice") or {}
    if not v.get("command_model"):
        return make_asr(cfg)
    from coach.asr.faster_whisper_backend import FasterWhisperBackend

    return FasterWhisperBackend(cfg, model=v["command_model"], compute_type=v.get("command_compute_type", "int8"),
                                beam_size=1, fast=True)


_LLMS: dict = {}


def make_llm(cfg: Config):
    """The local LLM (llama.cpp), loaded once per process and shared by the rephraser and the
    generative interviewer. None if disabled or the model file is missing."""
    r = cfg.get("reply") or {}
    if not r.get("llm"):
        return None
    path = models_dir(cfg) / "llm" / cfg.registry["llm"][r["llm_model"]]["file"]
    if not path.exists():
        print(f"[coach] local AI model missing ({path.name}); the robot uses approved lines only. "
              f"Run: python -m offline.download_models")
        return None
    key = (str(path), r.get("llm_threads", 4), r.get("llm_ctx", 4096))
    if key not in _LLMS:
        from coach.reply.llm import LocalLLM

        _LLMS[key] = LocalLLM(path, threads=key[1], n_ctx=key[2])
    return _LLMS[key]


def make_rephraser(cfg: Config, llm=None):
    """Local AI that rephrases approved replies (None if disabled or the model is missing)."""
    llm = llm or make_llm(cfg)
    if llm is None:
        return None
    from coach.reply.llm import Rephraser

    return Rephraser(llm=llm, max_words=(cfg.get("reply") or {}).get("llm_max_words", 25))


def interviewer_mode(cfg: Config) -> str:
    return (cfg.get("interviewer") or {}).get("mode", "scripted")


def make_interviewer(cfg: Config, llm=None):
    """Generative interviewer (interviewer.mode: generative). None in scripted mode or without the model."""
    if interviewer_mode(cfg) != "generative":
        return None
    llm = llm or make_llm(cfg)
    if llm is None:
        return None
    from coach.interviewer.interviewer import Interviewer

    return Interviewer(llm, max_words=(cfg.get("interviewer") or {}).get("max_words", 40))


def make_knowledge(cfg: Config, embedder):
    """CpE/ECE knowledge library, embedded with the scoring embedder (None if turned off)."""
    if not (cfg.get("interviewer") or {}).get("knowledge", True):
        return None
    from coach.interviewer.knowledge import Knowledge

    return Knowledge(embedder, cfg)


def make_laughter(cfg: Config):
    r = cfg.get("reply") or {}
    path = models_dir(cfg) / cfg.registry["audio_events"]["yamnet"]["file"]
    if not r.get("laughter") or not path.exists():
        return None
    from coach.reply.laughter import LaughterDetector

    return LaughterDetector(path, r.get("laughter_threshold", 0.3))


def make_tts(cfg: Config):
    from coach.tts.piper_backend import PiperBackend

    return PiperBackend(models_dir(cfg) / f"{cfg.tts.english_voice}.onnx")


def make_embedder(cfg: Config):
    entry = cfg.registry["scoring"][cfg.scoring.embedder]
    if cfg.scoring.backend == "sentence_transformers":
        from coach.scoring.st_backend import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(entry["repo"], prefix=entry.get("prefix", ""),
                                           cache_dir=models_dir(cfg) / "embedders")
    if cfg.scoring.backend == "onnx_int8":
        from coach.scoring.onnx_backend import OnnxEmbedder

        return OnnxEmbedder(models_dir(cfg) / entry["onnx_dir"], prefix=entry.get("prefix", ""))
    raise ValueError(f"Unknown scoring.backend: {cfg.scoring.backend}")
