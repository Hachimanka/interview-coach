"""Feature vector per (answer, criterion) for the trained scorer (T3.1).

Built from the same numbers scoring v0 already records, so training in T3.1 can start from
saved session JSON without re-running the embedder.
"""
from __future__ import annotations

import numpy as np


def criterion_features(crit_result: dict, n_words: int) -> np.ndarray:
    sims = crit_result["level_similarity"]
    levels = sorted(sims, key=int)
    s = np.array([sims[k] for k in levels])
    kp = crit_result["key_points"]
    coverage = np.mean([k["covered"] for k in kp]) if kp else 0.0
    kp_mean = np.mean([k["similarity"] for k in kp]) if kp else 0.0
    return np.concatenate([s, s - s.max(), [coverage, kp_mean, np.log1p(n_words)]])

# TODO(T3.1): classifier.py trains an ordinal model on these features against human ratings.
