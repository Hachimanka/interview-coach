"""Scaffold guards: the runtime package imports cleanly and never pulls in torch."""
import importlib
import pkgutil
import sys

import coach


def test_coach_imports_without_torch():
    for mod in pkgutil.walk_packages(coach.__path__, prefix="coach."):
        importlib.import_module(mod.name)
    assert "torch" not in sys.modules, "coach/ must not import torch (runs on Pi 5)"


def test_profiles_have_same_keys():
    import yaml

    with open("config/pc.yaml") as f:
        pc = yaml.safe_load(f)
    with open("config/pi5.yaml") as f:
        pi = yaml.safe_load(f)
    for section in pc:
        if isinstance(pc[section], dict):
            assert set(pc[section]) == set(pi[section]), f"key mismatch in '{section}'"
