"""
The rag package: STAGE 5 (retrieval) and STAGE 6 (guardrails + generation).

Re-exported lazily (PEP 562), matching `embed/` and `store/`: an eager
`from .retrieve import ...` would put the module in sys.modules before runpy
reaches it, which makes `python -m rag.retrieve` warn about double initialisation.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BM25Index",
    "OutOfCorpus",
    "RetrievedChunk",
    "detect_scheme",
    "expand_query",
    "explain",
    "mmr_diversify",
    "normalise_query",
    "rrf_fuse",
    "tokenize",
]

_SUBMODULE = __name__ + ".retrieve"

# NOTE: the `retrieve` FUNCTION is deliberately not re-exported here. This
# package contains a module called `rag.retrieve` that defines a function called
# `retrieve`, and a package cannot have both meanings for one name - whichever
# wins, the other becomes unreachable and the surprise lands on a reader. The
# module keeps the name (standard Python: `rag.retrieve` is the module) and
# callers reach the function as `rag.retrieve.retrieve(...)`.


def __getattr__(name: str) -> Any:
    if name in __all__:
        # importlib, not `from . import ...`: the `from` form re-enters this
        # __getattr__ for the submodule name and recurses.
        value = getattr(importlib.import_module(_SUBMODULE), name)
        globals()[name] = value  # cache so later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
