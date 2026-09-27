"""
The tools package: human-readable views of the pipeline's output.

Kept out of the numbered stages on purpose. Nothing in here is part of the RAG
pipeline - it reads `data/` and writes text files, and nothing in `ingest/`,
`embed/`, `store/`, or `rag/` imports it. If a dump ever starts failing, the
build did not break; only the window onto it did.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DUMPS", "dump_chunks", "dump_corpus", "dump_embeddings", "dump_store"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import dump  # noqa: PLC0415

        return getattr(dump, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
