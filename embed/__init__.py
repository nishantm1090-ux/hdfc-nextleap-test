"""
The embed package: STAGE 3.

The public names are re-exported LAZILY (PEP 562) rather than with a top-level
`from .index import ...`. An eager import puts `embed.index` in sys.modules
before `python -m embed.index` gets to execute it, which makes runpy warn and can
produce a double-initialised module. A lazy `__getattr__` keeps `from embed import
embed_all` working while letting `-m` run the module exactly once.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EmbedStats",
    "build_embed_text",
    "embed_all",
    "embed_chunks",
    "embed_hash",
    "load_cache",
    "load_model",
    "save_vector",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import index  # noqa: PLC0415

        return getattr(index, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
