"""Scoring v0 (T1.6): pretrained embeddings only, no training.

Per criterion: cosine similarity of the answer to each score level's anchor answers, softmax over
levels -> expected score on the 1-5 scale. Every result records the matched criterion, level and
anchor so a faculty member can trace the score. T3.1 replaces the level step with a trained
ordinal classifier over these same features (see features.py).
"""
from __future__ import annotations

import re

import numpy as np

from coach.scoring.explain import SHORT_ANSWER, correction_for
from coach.scoring.rubric import Question

MIN_WORDS = 8


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\s*,\s*(?=and |but |so |then )", text) if p.strip()]
    return parts or [text]


class Scorer:
    def __init__(self, embedder, temperature: float = 0.02, key_point_threshold: float = 0.85,
                 key_points_in_feedback: bool = False):
        self.emb = embedder
        self.temperature = temperature
        self.kp_threshold = key_point_threshold
        self.kp_feedback = key_points_in_feedback
        self._cache: dict[str, dict] = {}

    @classmethod
    def from_config(cls, cfg, embedder) -> "Scorer":
        s = cfg.scoring
        return cls(embedder, s.level_temperature, s.key_point_threshold, s.key_points_in_feedback)

    def _prepare(self, q: Question) -> dict:
        """Embed anchors and key points once per question."""
        if q.id not in self._cache:
            prep = {}
            for c in q.criteria:
                levels = sorted(c.anchors)
                texts = [a for lv in levels for a in c.anchors[lv]]
                owners = [lv for lv in levels for _ in c.anchors[lv]]
                prep[c.id] = {"levels": levels, "anchor_texts": texts, "anchor_levels": owners,
                              "anchors": self.emb.encode(texts),
                              "kps": self.emb.encode(c.key_points) if c.key_points else None}
            self._cache[q.id] = prep
        return self._cache[q.id]

    def score(self, q: Question, answer: str) -> dict:
        n_words = len(answer.split())
        if n_words < MIN_WORDS:
            crits = [{"criterion_id": c.id, "criterion": c.name, "score": 1, "expected": 1.0,
                      "matched_level": 1, "matched_anchor": None, "level_similarity": {},
                      "key_points": [{"text": k, "covered": False, "similarity": 0.0} for k in c.key_points]}
                     for c in q.criteria]
            return {"question_id": q.id, "score": 1.0, "criteria": crits, "too_short": True,
                    "correction": {"criterion_id": q.criteria[0].id, "criterion": q.criteria[0].name,
                                   "text": SHORT_ANSWER}}

        prep = self._prepare(q)
        sentences = split_sentences(answer)
        vecs = self.emb.encode([answer] + sentences)
        a_vec, s_vecs = vecs[0], vecs[1:]

        crits = []
        for c in q.criteria:
            p = prep[c.id]
            sims = p["anchors"] @ a_vec
            level_sim = {lv: float(max(s for s, o in zip(sims, p["anchor_levels"]) if o == lv))
                         for lv in p["levels"]}
            lv_arr = np.array(p["levels"], dtype=float)
            ls = np.array([level_sim[lv] for lv in p["levels"]])
            w = np.exp((ls - ls.max()) / self.temperature)
            expected = float((w / w.sum()) @ lv_arr)
            best = int(np.argmax(sims))
            kp = []
            if p["kps"] is not None:
                kp_sims = (p["kps"] @ s_vecs.T).max(axis=1)
                kp = [{"text": t, "covered": bool(s >= self.kp_threshold), "similarity": round(float(s), 3)}
                      for t, s in zip(c.key_points, kp_sims)]
            crits.append({"criterion_id": c.id, "criterion": c.name,
                          "score": int(round(expected)), "expected": round(expected, 2),
                          "matched_level": p["anchor_levels"][best],
                          "matched_anchor": p["anchor_texts"][best],
                          "level_similarity": {str(k): round(v, 3) for k, v in level_sim.items()},
                          "key_points": kp})
        overall = round(float(np.mean([c["expected"] for c in crits])), 1)
        return {"question_id": q.id, "score": overall, "criteria": crits, "too_short": False,
                "correction": correction_for(q, crits, self.kp_feedback)}
