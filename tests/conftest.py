"""Shared fixtures. Model-dependent tests skip when models/ is empty
(run: python -m offline.download_models --profile pc --all-asr)."""
import numpy as np
import pytest

from coach.profile import load_profile, make_tts, models_dir


@pytest.fixture(scope="session")
def cfg():
    return load_profile("pc")


@pytest.fixture(scope="session")
def cfg_pi5():
    return load_profile("pi5")


def require(path):
    if not path.exists():
        pytest.skip(f"model missing: {path.name} (run offline.download_models)")


@pytest.fixture(scope="session")
def speak(cfg):
    """speak("text") -> 16 kHz float32 speech made with Piper (test audio with known words)."""
    require(models_dir(cfg) / f"{cfg.tts.english_voice}.onnx")
    from coach.audio.util import resample

    tts = make_tts(cfg)

    def _speak(text: str) -> np.ndarray:
        audio, sr = tts.synthesize(text)
        return resample(audio, sr, 16000)

    return _speak


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)
