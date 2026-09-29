"""
tests/conftest.py - shared fixtures.

The single most valuable thing here is `model`: it is session-scoped, so the
test suite loads MiniLM once instead of once per test. A 121-chunk corpus
re-encoded 20+ times turned a 10-second suite into a 9-minute one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os  # noqa: E402

import common  # noqa: E402

# The suite must be deterministic and offline no matter what the developer's
# `.env` or shell says. `rag.answer._generate` honours `LLM_PROVIDER` over the
# YAML precisely so a developer can point the running app at a real model - but
# a test that made paid network calls, or that asserted extractive answers
# against an LLM's prose, would be neither reproducible nor free. Pin it before
# any test module imports `rag.answer`.
os.environ["LLM_PROVIDER"] = "stub"
for _leaked in ("OPENAI_API_KEY", "GROQ_API_KEY"):
    os.environ.pop(_leaked, None)


@pytest.fixture(scope="session")
def app_cfg() -> dict:
    return common.load_config()


# ---------------------------------------------------------------------------
# Scheme URL helper
# ---------------------------------------------------------------------------

#: Internal slug -> the public URL config/sources.yaml actually declares.
#:
#: The tests used to assert that a citation "ends with the scheme slug", which
#: was true only while every source was a Groww page, where the URL path IS the
#: slug (`.../hdfc-large-cap-fund-direct-growth`). The corpus is now the AMC's
#: own site, whose canonical paths are shorter and which renamed one scheme
#: without renaming its URL: the Flexi Cap fund lives at
#: `.../hdfc-flexi-cap-fund/direct` while its internal slug is still
#: `hdfc-equity-fund-direct-growth`. Asserting the slug is a second, redundant
#: copy of a fact the config already states, and it disagreed with the truth.
#: Reading the URL out of the config instead means one source of truth: if the
#: declared URL changes, these tests follow it.
_SCHEME_URLS: dict[str, str] = {
    s["scheme_slug"]: s["url"] for s in common.load_schemes()
}


def scheme_url(slug: str) -> str:
    """The configured public URL for a scheme's internal slug."""
    try:
        return _SCHEME_URLS[slug]
    except KeyError:  # pragma: no cover - a typo in a test, fail loudly
        raise AssertionError(
            f"unknown scheme slug {slug!r}; known: {sorted(_SCHEME_URLS)}"
        ) from None


@pytest.fixture(scope="session")
def model():
    """MiniLM, loaded once. Skips the test - not the suite - if unavailable."""
    from embed.index import load_model

    try:
        return load_model()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"all-MiniLM-L6-v2 unavailable ({type(exc).__name__}: {exc}); "
                    f"run once online to populate the Hugging Face cache")
