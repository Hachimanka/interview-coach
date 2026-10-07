"""Sentence-embedding interface (T1.6).

PC: sentence-transformers backend (imported lazily, never at module import).
Pi: ONNX int8 backend via onnxruntime + tokenizers (T4.2).
"""
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
