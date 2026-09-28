"""
embed/onnx_encoder.py - MiniLM-L6-v2 inference on ONNX Runtime (no PyTorch).

Why this exists
---------------
The deployed app runs on Render's 512 MB free tier. Importing PyTorch to run
the embedding model costs ~560 MB peak RSS and ~10-60 s of import time on a
cold instance, which OOM-kills the process mid-question and makes the first
answer feel like an eternal spinner. The same weights exported to ONNX run
under onnxruntime (~1 s import, ~80-100 MB), which ChromaDB already pulls in.

This module is a drop-in subset of the sentence-transformers ``encode`` API
that the rest of the pipeline calls (see embed/index.load_model): it returns
float32 [N, 384] numpy vectors, exposes ``get_embedding_dimension`` /
``get_sentence_embedding_dimension`` / ``max_seq_length``, and can be swapped
for the torch-backed model without touching any caller.

Graph contract (produced by tools/export_onnx.py)::

    input_ids        int64  [B, T]
    attention_mask   int64  [B, T]
    last_hidden_state float32 [B, T, 384]

The sentence embedding is the attention-masked mean of last_hidden_state,
then L2-normalised - exactly the "Mean pooling" + "Normalize" modules that
ship with the repo-local copy (models/all-MiniLM-L6-v2/1_Pooling/config.json
has pooling_mode_mean_tokens=true). tools/export_onnx.py verifies cosine >=
0.999 against sentence-transformers on probe queries at export time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


class OnnxMiniLM:
    """Embed MiniLM-L6-v2 text with onnxruntime, matching ST-style output."""

    dimension = 384

    def __init__(self, model_dir: str | Path, max_seq_length: int = 256) -> None:
        import onnxruntime as ort  # noqa: PLC0415
        from tokenizers import Tokenizer  # noqa: PLC0415

        d = Path(model_dir)
        self._session = ort.InferenceSession(
            str(d / "model.onnx"), providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_seq_length)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------ API
    def get_embedding_dimension(self) -> int:
        return self.dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: Iterable[str] | str,
        *,
        batch_size: int | None = None,          # accepted for API parity
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,          # accepted for API parity
        show_progress_bar: bool = False,        # accepted for API parity
        convert_to_tensor: bool = False,        # accepted for API parity; never
                                                # emitted by this backend
        **_: Any,
    ) -> np.ndarray:
        """Return float32 [N, 384] vectors for the given texts."""
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype="int64")            # [B, T]
        mask = np.array([e.attention_mask for e in encodings], dtype="int64")
        (hidden,) = self._session.run(
            ["last_hidden_state"], {"input_ids": ids, "attention_mask": mask}
        )                                                                    # [B, T, H]

        mask_f = mask[..., None].astype("float32")
        pooled = (hidden * mask_f).sum(axis=1) / mask_f.sum(axis=1).clip(min=1.0)
        if normalize_embeddings:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return np.asarray(pooled, dtype="float32").reshape(-1, self.dimension)