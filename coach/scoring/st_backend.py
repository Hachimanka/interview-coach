"""sentence-transformers Embedder (PC profile only; pulls in torch when constructed) (T1.6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from coach.scoring.embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, repo: str, prefix: str = "", cache_dir: str | Path | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: keeps torch out of imports

        self.model = SentenceTransformer(repo, device="cpu",
                                         cache_folder=str(cache_dir) if cache_dir else None)
        self.prefix = prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.prefix + t for t in texts], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
