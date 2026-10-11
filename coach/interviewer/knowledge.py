"""CpE/ECE knowledge library and competency blueprint (G1).

Passages are embedded once with the scoring embedder (no extra model). The interviewer looks up the
passages closest to what the student said, only to choose a relevant question; the guard also uses
them as the list of technical words a generated question may contain.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from coach.profile import PROJECT_ROOT, Config, resolve

SAMPLES = PROJECT_ROOT / "samples"


def _first_existing(cfg: Config | None, approved: str, sample):
    if cfg is not None:
        path = resolve(cfg, approved)
        if path.exists():
            return path, False
    return sample, True


@lru_cache(maxsize=4)
def _load_blueprint(path: str) -> dict:
    return json.loads(open(path, encoding="utf-8").read())["slots"]


def blueprint(cfg: Config | None = None) -> dict[str, dict]:
    """Rubric question id -> {competency, goal, callback}."""
    path, _ = _first_existing(cfg, "data/rubric/blueprint.json", SAMPLES / "blueprint.json")
    return _load_blueprint(str(path))


def slot(cfg: Config | None, question) -> dict:
    """The slot of a rubric question; questions without one get a goal made from their own text."""
    s = blueprint(cfg).get(question.id)
    if s:
        return s
    return {"competency": question.bank, "goal": "the same thing as: " + question.text["en"], "callback": True}


class Knowledge:
    def __init__(self, embedder, cfg: Config | None = None):
        approved = resolve(cfg, "data/knowledge") if cfg is not None else None
        files = sorted(approved.glob("*.json")) if approved is not None and approved.exists() else []
        self.is_sample = not files
        files = files or sorted((SAMPLES / "knowledge").glob("*.json"))
        self.passages: list[dict] = []
        for f in files:
            self.passages += json.loads(f.read_text(encoding="utf-8"))["passages"]
        self.emb = embedder
        self.vecs = embedder.encode([p["text"] for p in self.passages]) if self.passages else np.zeros((0, 1))

    def search(self, text: str, k: int = 2, min_sim: float = 0.80) -> list[dict]:
        if not text.strip() or not self.passages:
            return []
        sims = self.vecs @ self.emb.encode([text])[0]
        order = np.argsort(-sims)[:k]
        return [dict(self.passages[i], similarity=round(float(sims[i]), 3)) for i in order if sims[i] >= min_sim]

    def similarity(self, a: str, b: str) -> float:
        v = self.emb.encode([a, b])
        return float(v[0] @ v[1])

    def best_match(self, query: str, candidates: list[str]) -> tuple[int, float]:
        """(index, similarity) of the candidate closest to the query; (-1, 0) if none."""
        if not candidates:
            return -1, 0.0
        v = self.emb.encode([query] + candidates)
        sims = v[1:] @ v[0]
        i = int(np.argmax(sims))
        return i, float(sims[i])
