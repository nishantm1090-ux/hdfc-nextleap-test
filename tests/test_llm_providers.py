"""Regressions for the two bugs that only appeared once a real LLM was wired in.

Both are in `tests/test_golden_regressions.py`'s spirit - found by running the
thing, not by reading it - and both are about the seam between Stage 6's
generator and Stage 6's validator, which no test had crossed with more than one
chunk in play.

1. **Grounding was checked against one chunk instead of the whole context.**
   `generation.context_chunks` is 5, so a real LLM reads 5 chunks and may quote
   a figure from any of them. Validating against `[used]` alone reported a
   correct answer as "states the figure '2214.57', which is not in any
   retrieved source" and refused the question. The stub never showed it, because
   the stub quotes exactly one chunk - so for the stub, one chunk *is* the whole
   context and the narrow check was accidentally right.

2. **The model list is per-account.** `llama-3.1-8b-instant` is in most Groq
   documentation and was not offered on the key this was built against, so a
   hard-coded default 404s. The available set has to be discoverable.
"""

from __future__ import annotations

import inspect
import os

import pytest

import common


# ---------------------------------------------------------------------------
# 1. The grounding context must cover everything the generator was shown
# ---------------------------------------------------------------------------


def _chunks_with_the_figure(figure: str):
    """Two chunks: the first lacks the figure, the second holds it.

    This is the real shape of the failure. Retrieval ranks a chunk that matches
    the question's *words* first and the chunk holding the *figure* second, so
    the two are almost never the same chunk for a table-shaped fact.
    """
    from rag.retrieve import RetrievedChunk

    def mk(cid: str, text: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=cid, text=text,
            metadata={"source_url": "https://groww.in/mutual-funds/x",
                      "fact_key": "nav"},
            dense_score=0.5, bm25_score=3.0, rrf_score=0.03, final_score=0.03,
            coverage=1.0, matched_terms=["nav"],
        )

    return [
        mk("first-no-figure", "nav: 25 sep '26  (heading and label only)"),
        mk("second-has-figure", f"nav: 25 sep '26 {figure}"),
    ]


def test_grounding_with_real_corpus_chunks():
    """Grounding was checked against one chunk instead of the whole context.

    Before the fix (context_for_validation = [used]), a correct answer that
    quoted a figure from chunk 2 (or any chunk besides the first) was refused
    as a hallucination because the validator only looked at chunk 1. The stub
    never showed this because the stub quotes exactly one chunk, so one chunk
    IS the whole context for the stub and the narrow check was accidentally
    right.

    After the fix (context_for_validation = list(chunks)), the validator checks
    every chunk the generator was shown, so a correct answer from any chunk
    passes grounding. This test pins the correct behavior using real corpus
    chunks with real corpus URLs, so validate_answer's corpus URL check passes.
    """
    from rag import guardrails as G
    from common import read_jsonl
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    corpus_chunks = [
        c for c in read_jsonl(ROOT / "data" / "chunks.jsonl")
        if c.get("fact_key")
    ]
    assert len(corpus_chunks) >= 2, "need at least 2 fact chunks"

    # Use the first two corpus chunks - they have real corpus URLs and real
    # fact_key figures. The first chunk may have a simpler figure, the second
    # may have a compound one.
    chunk1 = corpus_chunks[0]
    chunk2 = corpus_chunks[1]

    # Take the actual text from each chunk, strip any inline source
    text1 = common.strip_inline_source(chunk1["text"]).strip()
    text2 = common.strip_inline_source(chunk2["text"]).strip()

    url1 = chunk1.get("source_url", "https://groww.in/mutual-funds/x")
    url2 = chunk2.get("source_url", "https://groww.in/mutual-funds/x")

    # Answer text from chunk2's figure, citing chunk2's URL
    text = text2
    answer = {"text": text,
              "sources": [{"url": url2}]}

    # With ONE chunk only - the one WITH the figure - should pass grounding.
    ok_one, reasons_one = G.validate_answer(answer, context=[chunk2])
    assert ok_one, (
        f"One chunk WITH the figure should pass grounding but failed. "
        f"reasons={reasons_one}")

    # With TWO chunks - the one WITH and the one WITHOUT the figure - should
    # also pass because the figure IS in the context set.
    ok_two, reasons_two = G.validate_answer(answer, context=[chunk1, chunk2])
    assert ok_two, (
        f"Two chunks including one with the figure should pass grounding. "
        f"reasons={reasons_two}")


def test_ask_validates_against_the_whole_retrieved_set():
    """`ask()` must not narrow the context to a single chunk.

    Read from the source rather than from a live call, so the assertion holds
    without an API key and without the model warm.
    """
    src = inspect.getsource(__import__("rag.answer", fromlist=["ask"]).ask)
    assert "context_for_validation = list(chunks)" in src, (
        "ask() no longer validates the draft against every retrieved chunk; "
        "a figure read from chunk 2+ will be rejected as ungrounded")
    assert "context_for_validation = [used]" not in src, (
        "ask() is validating against one chunk again - this is the bug that "
        "made 0/3 benchmark models answer the NAV question")


def test_the_fallback_candidate_is_still_validated_against_its_own_chunk():
    """The extractive fallback is quoted from one chunk, so it gets one chunk.

    Widening this to `chunks` would be a *different* bug: a fallback sentence
    containing a figure from a different chunk would pass a check it should
    fail. Both directions are asserted, because only pinning the fix invites
    over-correction.
    """
    src = inspect.getsource(__import__("rag.answer", fromlist=["ask"]).ask)
    assert "context=[alt]" in src, (
        "the fallback candidate must be validated against [alt] alone")


def test_context_chunks_is_greater_than_one_or_the_bug_returns():
    """The bug is only reachable when the model sees more than one chunk.

    If someone sets `context_chunks: 1` to "make it consistent", the narrow check
    becomes correct again - but they would have thrown away the multi-chunk
    context to work around a validator bug, which is the wrong trade.
    """
    n = int(common.load_config()["generation"].get("context_chunks", 5))
    assert n > 1, (
        "generation.context_chunks is 1, so the grounding-context fix cannot "
        "be exercised; multi-chunk context is the point")


# ---------------------------------------------------------------------------
# 2. The model list is per-account
# ---------------------------------------------------------------------------


def test_the_default_groq_model_is_not_hard_coded_to_a_model_that_may_not_exist():
    """`llama-3.1-8b-instant` is documented by Groq and was not on our key.

    A hard-coded default that 404s on a given account reads as "the API is
    broken" and costs an hour. The default stays as a last resort, but the
    `.env` value is what actually selects the model, and it is documented as
    discovered rather than assumed.
    """
    from rag import answer as A

    src = inspect.getsource(A.generate_groq)
    assert "GROQ_MODEL" in src, "the env var must be consulted"
    assert "cfg.get(\"groq_model\")" in src, "config must take precedence"


def test_groq_sends_a_user_agent_because_cloudflare_403s_the_default_one():
    """A 403 with a plain-text body and no JSON is Cloudflare, not a bad key.

    httpx sends `User-Agent: python-httpx/x.y`, which Groq's edge rejects with
    `error code: 1010`. Since 1010 is not a 401, the key is fine and the
    request is the problem - and the two are easy to confuse because both are
    "the call failed". Pinned so the header is not removed as noise.
    """
    from rag import answer as A

    assert "User-Agent" in A._HTTP_HEADERS, (
        "groq calls will be 403'd by Cloudflare without a browser User-Agent")
    assert "Mozilla" in A._HTTP_HEADERS["User-Agent"]
    src = inspect.getsource(A.generate_groq)
    assert "_HTTP_HEADERS" in src, "generate_groq must actually send them"


def test_groq_can_list_the_models_a_key_can_actually_call():
    """The function exists so a model can be checked before spending a request."""
    from rag import answer as A

    assert callable(A.groq_available_models)
    src = inspect.getsource(A.groq_available_models)
    assert "/models" in src
    # It must be a GET on the models endpoint, not a chat call.
    assert "chat/completions" not in src


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"),
                    reason="GROQ_API_KEY not set")
def test_the_configured_groq_model_is_callable():
    """Live check: is the model in `.env` actually offered to this key?

    Skipped rather than failed when no key is set, because the whole point of
    the offline stub default is that the suite runs with no credentials. When a
    key IS present, a model that 404s is a genuine misconfiguration and should
    fail loudly instead of falling back to the stub and looking fine.
    """
    import httpx

    from rag import answer as A

    common.load_dotenv_if_present()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        pytest.skip("no key after loading .env")
    model = os.environ.get("GROQ_MODEL", "").strip()
    assert model, "GROQ_MODEL is not set"
    live = A.groq_available_models(key)
    assert model in live, (
        f"GROQ_MODEL={model!r} is not callable on this key. "
        f"Available: {live}")
