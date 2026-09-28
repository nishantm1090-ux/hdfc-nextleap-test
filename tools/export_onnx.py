"""Export the repo-local MiniLM checkpoint to ONNX for memory-light runtime use.

Why this exists
---------------
The deployed app runs on Render's 512 MB free tier. Importing PyTorch to run
the embedding model costs ~560 MB peak RSS and ~10-60 s of import time on a
cold instance, which OOM-kills the process mid-question and makes the first
answer feel like an eternal spinner. The same MiniLM weights exported to ONNX
run under onnxruntime (~1 s import, ~50 MB) - and onnxruntime is already in
the image as a ChromaDB dependency.

What it produces
----------------
    models/all-MiniLM-L6-v2/model.onnx      (ONNX graph)
    models/all-MiniLM-L6-v2/model.onnx.data (external weights, ~87 MB)

The ONNX graph is ONLY the transformer (a BertModel): input_ids + attention_mask
-> last_hidden_state. Mean-pooling over the attention mask + L2 normalisation
(the sentence-transformers "Mean pooling" + "Normalize" modules, as shipped in
this repo under models/all-MiniLM-L6-v2/1_Pooling/config.json and the pooler in
embed/onnx_encoder.OnnxMiniLM) are applied in numpy on the client side.

Run (requires torch + sentence-transformers, i.e. the DEV environment, not the
deployed image):

    python tools/export_onnx.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
ONNX_PATH = MODEL_DIR / "model.onnx"

QUERIES = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "Is there a lock-in period on HDFC ELSS Tax Saver?",
    "What is the minimum SIP for HDFC Flexi Cap Direct Growth?",
    "HDFC Small Cap Fund NAV as of last week, Direct plan growth option",
    "Who manages the HDFC Balanced Advantage Fund and what is its AUM?",
    "How much did HDFC ELSS return since inception across the last five years?",
]


def main() -> int:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not (MODEL_DIR / "model.safetensors").is_file():
        print(f"fatal: {MODEL_DIR}/model.safetensors is missing; nothing to export")
        return 1

    print(f"loading sentence-transformers model from {MODEL_DIR}...")
    t0 = time.perf_counter()
    st = SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True)
    print(f"  loaded in {time.perf_counter()-t0:.2f}s (dim={st.get_sentence_embedding_dimension()}, "
          f"max_seq_length={st.max_seq_length})")

    bert = st[0].auto_model  # a transformers.BertModel
    tok = st[0].tokenizer

    # Trace the transformer with batch + sequence axes dynamic.
    example = tok(["export probe"], padding="max_length", truncation=True,
                  max_length=16, return_tensors="pt")
    print(f"exporting BertModel -> {ONNX_PATH.name} ...")
    t0 = time.perf_counter()
    torch.onnx.export(
        bert,
        (example["input_ids"], example["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"  exported in {time.perf_counter()-t0:.2f}s ({size_mb:.1f} MB)")

    # --- End-to-end parity: ST (torch) vs ONNX on the same queries ---------
    from tokenizers import Tokenizer  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    tk = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tk.enable_truncation(max_length=st.max_seq_length)
    tk.enable_padding(pad_id=0, pad_token="[PAD]")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    st_vecs = st.encode(QUERIES, normalize_embeddings=True, convert_to_numpy=True)
    enc = tk.encode_batch(QUERIES)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    hidden = sess.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
    m = mask[..., None].astype(np.float32)
    pooled = (hidden * m).sum(1) / m.sum(1).clip(min=1.0)
    pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
    onnx_vecs = np.asarray(pooled, dtype="float32")

    worst = 1.0
    for q, a, b in zip(QUERIES, st_vecs, onnx_vecs):
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst = min(worst, cos)
        print(f"  cosine {cos:.6f}  <- {q[:60]}")
    print(f"worst cosine: {worst:.6f}")
    if worst < 0.9990:
        print("  FAIL: ONNX drift beyond tolerance; not committing this export")
        return 1
    print("  OK: cosine >= 0.9990 for all probe queries — commit this export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())