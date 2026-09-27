"""
The store package: STAGE 4.

Re-exported lazily (PEP 562) rather than eagerly, so that
`python -m store.chroma_store` executes the module exactly once. An eager
`from .chroma_store import ...` here would put the module in sys.modules before
runpy reaches it and produce a double-initialisation warning.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FILTERABLE_KEYS",
    "STORED_KEYS",
    "collection_name",
    "coerce_metadata",
    "coverage_gaps",
    "get_client",
    "get_collection",
    "load_join_index",
    "query",
    "rebuild",
    "report",
    "stats",
    "sync",
    "upsert_chunks",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import chroma_store  # noqa: PLC0415

        return getattr(chroma_store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
