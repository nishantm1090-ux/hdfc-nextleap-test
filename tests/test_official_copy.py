"""
tests/test_official_copy.py - the 12-point brief's source-copy and guardrail upgrades.

This file pins the behaviours added in the official-sources pass:

  * no user-visible Groww copy: the disclaimer, the performance refusal and the
    missing-information response all read as official-only;
  * scheme filtering happens in the STORE (`where={"scheme_slug": ...}`) before
    similarity search, and "Why this answer?" shows only the named scheme;
  * a scheme-less question about a fact that differs by scheme ASKS which scheme
    instead of guessing by retrieval order;
  * capital-gains / statement questions get a directed refusal naming the
    official place to go;
  * future-speculation questions ("who will manage ... in 2030") and
    "which of these funds is the best" are refused.

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_official_copy.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from rag import answer as A  # noqa: E402
from rag import guardrails as G  # noqa: E402

LARGE_CAP = "hdfc-large-cap-fund-direct-growth"


# ---------------------------------------------------------------------------
# User-visible copy: no Groww framing
# ---------------------------------------------------------------------------


def test_the_disclaimer_is_official_only():
    assert "Sources: HDFC Mutual Fund, SEBI and AMFI." in G.DISCLAIMER
    assert "groww" not in G.DISCLAIMER.lower()
    assert "HDFC AMC, Groww" not in G.DISCLAIMER


def test_the_performance_refusal_points_to_official_publishers():
    assert "groww" not in G.PERFORMANCE_REFUSAL.lower()
    assert "https://www.hdfcfund.com/" in G.PERFORMANCE_REFUSAL
    assert "https://www.amfiindia.com/" in G.PERFORMANCE_REFUSAL


def test_the_missing_information_response_is_the_required_sentence():
    assert G.OUT_OF_CORPUS.startswith(
        "I couldn't verify that from the available official sources.")


def test_the_account_document_refusal_names_official_places():
    assert "groww" not in G.STATEMENT_REFUSAL.lower()
    for host in ("hdfcfund.com", "sebi.gov.in", "amfiindia.com"):
        assert host in G.STATEMENT_REFUSAL


# ---------------------------------------------------------------------------
# Scheme filter happens BEFORE similarity search (the `where` argument)
# ---------------------------------------------------------------------------


def test_a_named_scheme_is_sent_as_a_store_where_filter(monkeypatch):
    """The metadata filter reaches `retrieve` as `where`, so Chroma only scores
    that scheme's vectors - this is the brief's "filter BEFORE retrieval"."""
    real = A.retrieve
    seen: dict[str, object] = {}

    def spy(question, **kwargs):
        seen["where"] = kwargs.get("where")
        return real(question, **kwargs)

    monkeypatch.setattr(A, "retrieve", spy)

    a = A.ask("What is the expense ratio of HDFC Small Cap Fund?")
    assert a.kind == "answer"
    assert seen["where"] == {"scheme_slug": "hdfc-small-cap-fund-direct-growth"}

    a = A.ask("What is the exit load?")
    assert a.kind == "out_of_corpus"
    assert seen["where"] is None, "a scheme-less question must not be filtered"


def test_why_this_answer_shows_only_the_named_scheme():
    a = A.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    assert a.kind == "answer"
    rows = a.debug["retrieved"]
    assert rows, "the trace is empty for an answer that retrieved chunks"
    assert {r["scheme_slug"] for r in rows} == {LARGE_CAP}


# ---------------------------------------------------------------------------
# Ask-which-scheme instead of guessing
# ---------------------------------------------------------------------------


def test_a_tied_scheme_less_fact_asks_which_scheme():
    a = A.ask("What is the exit load?")
    assert a.kind == "out_of_corpus"
    assert a.debug.get("guardrail") == "ask_scheme"
    assert "scheme" in a.text.lower()
    # The full universe is named so the user knows exactly what to re-ask.
    assert "HDFC Large Cap" in a.text
    assert "HDFC Small Cap" in a.text


def test_a_tie_does_not_fire_when_the_question_names_a_scheme():
    a = A.ask("What is the exit load on HDFC Small Cap Fund?")
    assert a.kind == "answer"
    assert a.debug.get("guardrail") != "ask_scheme"


def test_a_lock_in_question_is_genuinely_ambiguous_so_it_asks():
    """Every AMC scheme page carries a "Lock-in period" row, so this ties.

    Four state "NA" and HDFC ELSS states "3 years". The old corpus had the fact
    on one page only and this question answered directly; on the AMC corpus the
    same words cover five funds with two different answers, which is exactly the
    case the ask-which-scheme guardrail exists for. The user is told the
    universe rather than handed one fund's row.
    """
    a = A.ask("Is there a lock-in period?")
    assert a.kind == "out_of_corpus"
    assert a.debug.get("guardrail") == "ask_scheme"


def test_the_ask_scheme_list_never_names_a_document():
    """The list of funds is the declared scheme universe, not chunk scope keys.

    It was derived from chunk `scheme_slug`, and a chunk's slug is a SCOPE key:
    the AMC's Consolidated Account Statement page is scoped under
    `hdfc-consolidated-account-statement`, which the refusal then printed to the
    user as though it were a sixth mutual fund.
    """
    a = A.ask("Is there a lock-in period?")
    said = a.text
    assert "hdfc-consolidated-account-statement" not in said, said
    # Independently derived from config/sources.yaml rather than through the
    # display-name helper, so the test cannot agree with the implementation by
    # construction.
    declared = [re.split(r"\s+[-\u2013]\s+", s["scheme_name"])[0]
                for s in common.load_schemes()]
    assert len(declared) == 5, declared
    for name in declared:
        assert name in said, f"{name} is missing from the fund list: {said}"


# ---------------------------------------------------------------------------
# Capital gains / statements - directed, sourced refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "How do I download my capital gains statement?",
    "Where can I get my capital gains statement for HDFC Small Cap?",
])
def test_a_statement_request_is_a_directed_official_refusal(question):
    a = A.ask(question)
    assert a.kind == "out_of_corpus"
    assert a.debug.get("guardrail") == "statement"
    assert "https://" in a.text
    assert "hdfcfund.com" in a.text


def test_the_statement_predicate_is_narrow():
    assert G.is_statement_question("How do I download my capital gains statement?")
    assert G.is_statement_question("where can I view my consolidated account statement")
    assert not G.is_statement_question("What is a capital gains statement?")
    assert not G.is_statement_question("What is the expense ratio of HDFC Large Cap Fund?")


# ---------------------------------------------------------------------------
# Future speculation
# ---------------------------------------------------------------------------


def test_who_will_manage_in_2030_is_refused_as_unverifiable():
    a = A.ask("Who will manage HDFC Large Cap Fund in 2030?")
    assert a.kind == "out_of_corpus"
    assert a.debug.get("guardrail") == "future_speculation"


def test_the_future_predicate_is_narrow():
    assert G.is_future_speculation("Who will manage HDFC Large Cap Fund in 2030?")
    assert G.is_future_speculation("What will the NAV of HDFC Large Cap be in 2040?")
    assert not G.is_future_speculation("Who is the fund manager of HDFC Large Cap Fund?")
    assert not G.is_future_speculation("What is the expense ratio of HDFC Large Cap Fund?")
    assert not G.is_future_speculation("What was the NAV of HDFC Large Cap in 2022?")


# ---------------------------------------------------------------------------
# "Which of these funds is the best?" is advice
# ---------------------------------------------------------------------------


def test_which_of_these_funds_is_best_is_advice():
    a = A.ask("Which of these funds is the best?")
    assert a.kind == "refusal_advice"
    assert G.is_advice("Which of these funds is the best?")
    assert G.is_advice("which of those schemes has the lowest expense ratio")
    assert not G.is_advice("Which index does HDFC Flexi Cap Direct Growth track?")