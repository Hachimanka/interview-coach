"""ONNX Runtime Embedder for the Pi 5 (no torch): tokenizers + mean pooling (T1.6 runtime, T4.2 export).

Expects a folder produced by training/export/export_embedder_onnx.py containing
tokenizer.json and model_quantized.onnx (or model.onnx).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from coach.scoring.embedder import Embedder


class OnnxEmbedder(Embedder):
    def __init__(self, model_dir: str | Path, prefix: str = "", max_length: int = 512, threads: int = 2):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} not found. Export the embedder on the PC first (T4.2).")
        onnx_file = next((model_dir / n for n in ("model_quantized.onnx", "model.onnx")
                          if (model_dir / n).exists()), None)
        if onnx_file is None:
            raise FileNotFoundError(f"No model_quantized.onnx or model.onnx in {model_dir}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_file), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.prefix = prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer.encode_batch([self.prefix + t for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]                   # (batch, tokens, dim)
        m = mask[..., None].astype(np.float32)
        emb = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)
