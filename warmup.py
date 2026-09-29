"""warmup.py - one-time background warm-up for the Streamlit UI.

The page must paint instantly, but the first question would otherwise pay a
~1.5s ONNX model load plus the Chroma client open inside the
"Retrieving and checking…" spinner. `start()` kicks both off on a daemon thread
right after the page renders, so by the time the user types, the heavy steps
are done and the first answer is ~0.1s instead.

Failures are swallowed on purpose: `ask()` re-attempts each load lazily and
reports its own errors, so a failed warm-up changes nothing about behaviour -
only about latency. This module only reads build artefacts; it never writes,
matching FR-8.9.
"""

from __future__ import annotations

import threading
from typing import Any


def _load() -> None:
    # Deliberately deferred imports: importing this module must cost nothing.
    try:
        from embed.index import load_model
        from store.chroma_store import get_client

        load_model()
        get_client()
    except Exception:
        pass  # ask() re-tries both lazily and reports failures itself


def start() -> Any:
    """Spawn the daemon warm-up thread. Never joins, never blocks paint."""
    threading.Thread(target=_load, name="ui-warmup", daemon=True).start()