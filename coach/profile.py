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
