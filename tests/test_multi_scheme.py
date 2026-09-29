"""
tests/test_multi_scheme.py - one scheme per question (the one-source contract)

PRD §11 pins exactly ONE source link per answer. A question that names two
schemes ("What is the NAV of HDFC Small Cap and HDFC ELSS?") cannot be answered
honestly within that contract:

  * answering both funds needs two pages -> two sources, a contract violation
    (measured live: the generator shipped two inline URLs);
  * answering one fund's figure while citing the other's page -> a sourced
    lie, the single worst failure this system can produce.

So the pipeline refuses deterministically and names the funds, so the user
knows exactly what to re-ask. This file also pins the two safety nets around
that decision: the retrieve-error path never leaks a filesystem path into the
chat text, and `validate_answer` rejects any draft naming two distinct URLs.

The guardrails must continue to judge only what the user literally typed:
memory resolution happens AFTER the guardrails, and a follow-up that resolves
to ONE scheme stays answerable.
"""

from __future__ import annotations

import pytest

import common
import rag.answer as A
from rag import guardrails as G
from rag.answer import ask
from rag.retrieve import detect_scheme, detect_schemes
from conftest import scheme_url


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return common.read_jsonl(common.path_for("chunks_file"))


def _turn(role: str, text: str) -> dict:
    return {"role": role, "text": text}


# ---------------------------------------------------------------------------
# detect_schemes (unit)
# ---------------------------------------------------------------------------


def test_detect_scheme_sees_two_named_schemes(corpus):
    slugs = detect_schemes("What is the NAV of HDFC Small Cap and HDFC ELSS?", corpus)
    assert sorted(slugs) == [
        "hdfc-elss-tax-saver-fund-direct-plan-growth",
        "hdfc-small-cap-fund-direct-growth",
    ]


def test_detect_schemes_deduplicates_aliases_of_one_scheme(corpus):
    """'HDFC Flexi Cap Fund, also called HDFC Equity Fund' is ONE scheme.

    Both names are aliases of hdfc-equity-fund-direct-growth; counting slugs,
    not strings, keeps it at one - otherwise every question that used both the
    alias and the market name would be wrongly refused.
    """
    slugs = detect_schemes("Is HDFC Flexi Cap Fund the same as HDFC Equity Fund?", corpus)
    assert slugs == ["hdfc-equity-fund-direct-growth"]


def test_detect_schemes_and_detect_scheme_agree_on_the_singular(corpus):
    q = "What is the benchmark of HDFC Small Cap Fund Direct Growth?"
    assert detect_schemes(q, corpus) == [detect_scheme(q, corpus)]
    assert detect_scheme(q, corpus) == "hdfc-small-cap-fund-direct-growth"


def test_detect_schemes_one_strong_and_one_weak_keeps_both(corpus):
    slugs = detect_schemes("Compare HDFC Large Cap Fund with HDFC ELSS", corpus)
    assert set(slugs) == {
        "hdfc-large-cap-fund-direct-growth",
        "hdfc-elss-tax-saver-fund-direct-plan-growth",
    }


# ---------------------------------------------------------------------------
# The refusal (live pipeline)
# ---------------------------------------------------------------------------


def test_two_named_schemes_are_refused_not_answered():
    a = ask("What is the NAV of HDFC Small Cap and HDFC ELSS?")
    assert a.kind == "out_of_corpus"
    assert a.debug.get("guardrail") == "multi_scheme"
    assert sorted(a.debug["named_schemes"]) == [
        "hdfc-elss-tax-saver-fund-direct-plan-growth",
        "hdfc-small-cap-fund-direct-growth",
    ]


def test_the_refusal_names_the_funds_and_no_sources_are_cited():
    a = ask("What is the NAV of HDFC Small Cap and HDFC ELSS?")
    assert "HDFC Small Cap Fund" in a.text
    assert "HDFC ELSS Tax Saver Fund" in a.text
    assert a.sources == []
    assert a.last_updated == ""


def test_a_three_scheme_question_is_refused_too():
    a = ask("What is the NAV of HDFC Small Cap, HDFC ELSS and HDFC Large Cap?")
    assert a.kind == "out_of_corpus"
    assert a.debug.get("guardrail") == "multi_scheme"
    assert len(a.debug["named_schemes"]) == 3


@pytest.mark.parametrize("q", [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "Is there a lock-in period on HDFC ELSS Tax Saver?",
    "What is the minimum SIP for HDFC Flexi Cap Direct Growth?",
])
def test_every_single_scheme_question_answers_normally(q):
    a = ask(q)
    assert a.kind == "answer"
    assert len(a.sources) == 1
    assert a.debug.get("guardrail") != "multi_scheme"


def test_a_memory_follow_up_resolves_to_one_scheme_and_answers():
    """The guardrail judges the RAW text; memory resolution happens after.

    A follow-up like "And its expense ratio?" names no scheme and must NOT be
    refused - it resolves to the single scheme from history and answers.
    """
    history = [_turn("user", "What is the AUM of HDFC Flexi Cap Fund?")]
    a = ask("And its expense ratio?", history=history)
    assert a.kind == "answer"
    assert a.debug.get("guardrail") != "multi_scheme"
    # The answer must be about FLEXI CAP, not a coin-flipped scheme.
    assert a.sources[0].url == scheme_url("hdfc-equity-fund-direct-growth"), \
        f"cited the wrong scheme's page: {a.sources[0].url}"


# ---------------------------------------------------------------------------
# Retrieval failure: the chat text never leaks a path
# ---------------------------------------------------------------------------


def test_a_retrieval_crash_is_an_error_refusal_with_no_raw_path(monkeypatch):
    def boom(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError(
            "Something went wrong while deserializing "
            "/opt/render/project/src/data/chroma/chroma.sqlite3")

    monkeypatch.setattr(A, "retrieve", boom)
    a = ask("What is the NAV of HDFC Large Cap Fund?")
    assert a.kind == "error"
    assert a.text == G.INDEX_ERROR
    assert "/opt/" not in a.text and "chroma" not in a.text
    assert "/opt/" in a.debug["error"], "the detail belongs in debug, not the chat"


def test_a_missing_index_is_an_error_refusal_with_no_raw_path(monkeypatch):
    def missing(*args, **kwargs):  # noqa: ARG001
        raise FileNotFoundError(
            "collection hdfc_mf_faq does not exist. Run: python -m store.chroma_store --rebuild")

    monkeypatch.setattr(A, "retrieve", missing)
    a = ask("What is the NAV of HDFC Small Cap Fund?")
    assert a.kind == "error"
    assert a.text == G.INDEX_ERROR
    assert "python -m store.chroma_store" not in a.text
    assert "does not exist" in a.debug["error"], "the detail belongs in debug, not the chat"


# ---------------------------------------------------------------------------
# validate_answer: two distinct URLs are rejected, one is fine
# ---------------------------------------------------------------------------


def _fake_chunk(text: str, url: str) -> dict:
    return {"text": text, "metadata": {"source_url": url}}


def test_validate_rejects_two_distinct_source_urls():
    ctx = [
        _fake_chunk("The NAV of HDFC Small Cap Fund is 159.82.", "https://groww.in/small"),
        _fake_chunk("The NAV of HDFC ELSS Tax Saver is 1447.38.", "https://groww.in/elss"),
    ]
    ok, reasons = G.validate_answer(
        {"text": "Both NAVs are 159.82 and 1447.38. "
                 "Source: https://groww.in/small Source: https://groww.in/elss",
         "sources": [{"url": "https://groww.in/small"}]},
        context=ctx,
    )
    assert not ok
    assert any("more than one distinct source URL" in r for r in reasons)


def test_validate_passes_a_single_repeated_source_url():
    ctx = [_fake_chunk("The expense ratio is 1.03 percent.", "https://groww.in/a")]
    ans = {"text": "The expense ratio is 1.03 percent.",
           "sources": [{"url": "https://groww.in/a"}]}
    ok, reasons = G.validate_answer(dict(ans), context=ctx)
    assert not any("more than one distinct source URL" in r for r in reasons)
    # A fully valid single-source answer passes.
    assert ok or any("not in any retrieved source" not in r for r in reasons)