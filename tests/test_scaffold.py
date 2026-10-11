"""Scaffold guards: the runtime package imports cleanly and never pulls in torch."""
import importlib
import pkgutil
import sys

import coach


def test_coach_imports_without_torch():
    """Checked in a fresh interpreter: other tests in this process may load the PC embedder (torch)."""
    import subprocess

    code = ("import importlib, pkgutil, sys, coach\n"
            "for m in pkgutil.walk_packages(coach.__path__, prefix='coach.'):\n"
            "    importlib.import_module(m.name)\n"
            "sys.exit(1 if 'torch' in sys.modules else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, "coach/ must not import torch (runs on Pi 5)\n" + r.stderr[-2000:]


def test_profiles_have_same_keys():
    import yaml

    with open("config/pc.yaml") as f:
        pc = yaml.safe_load(f)
    with open("config/pi5.yaml") as f:
        pi = yaml.safe_load(f)
    for section in pc:
        if isinstance(pc[section], dict):
            assert set(pc[section]) == set(pi[section]), f"key mismatch in '{section}'"
