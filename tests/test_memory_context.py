"""
tests/test_memory_context.py - retrieval memory (context window of N messages)

A user can follow up with a question that omits the scheme it is about:
    Q1: "What is the AUM of HDFC Flexi Cap Fund?"
    Q2: "And its expense ratio?"          <- names no scheme
Without memory, Q2 retrieves the strongest generic "expense ratio" chunk,
which is any scheme's - the answer is a coin flip between five funds. With the
context window, Q2 resolves against the last N transcript messages and becomes
"…its expense ratio? HDFC Flexi Cap Fund" for retrieval, ranking, generation.

The feature must be strictly additive:
  * no history            -> question passes through byte-for-byte
  * question names scheme -> question passes through (the named scheme wins)
  * guardrails still judge the RAW text ("should I invest more?" stays advice)
"""

from __future__ import annotations

import pytest

import common
import rag.answer as A
from rag.retrieve import with_memory_context

SLUGS = {
    "hdfc-large-cap-fund-direct-growth": "HDFC Large Cap Fund",
    "hdfc-equity-fund-direct-growth": "HDFC Flexi Cap Fund",
    "hdfc-elss-tax-saver-fund-direct-plan-growth": "HDFC ELSS Tax Saver Fund",
    "hdfc-small-cap-fund-direct-growth": "HDFC Small Cap Fund",
    "hdfc-balanced-advantage-fund-direct-growth": "HDFC Balanced Advantage Fund",
}


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return common.read_jsonl(common.path_for("chunks_file"))


def _turn(role: str, text: str) -> dict:
    return {"role": role, "text": text}


# ---------------------------------------------------------------------------
# with_memory_context (unit)
# ---------------------------------------------------------------------------


def test_no_history_passes_the_question_through(corpus):
    q, memo = with_memory_context("what is the exit load", None, corpus)
    assert q == "what is the exit load"
    assert memo == {}


def test_empty_history_passes_the_question_through(corpus):
    q, memo = with_memory_context("what is the exit load", [], corpus)
    assert q == "what is the exit load"
    assert memo == {}


def test_a_question_that_names_a_scheme_needs_no_memory(corpus):
    q = "what is the expense ratio of HDFC Small Cap Fund?"
    resolved, memo = with_memory_context(q, [_turn("user", "tell me about HDFC Large Cap")],
                                         corpus)
    assert resolved == q
    assert memo == {}


def test_a_terse_follow_up_resolves_to_the_last_named_scheme(corpus):
    history = [
        _turn("user", "What is the NAV of HDFC Flexi Cap Fund?"),
        _turn("assistant", "The NAV is ₹2,214.57 as of 27 Sep 2026."),
    ]
    resolved, memo = with_memory_context("and its expense ratio?", history, corpus)
    assert memo["resolved_scheme"] == "hdfc-equity-fund-direct-growth"
    assert memo["from_history"] is True
    assert "hdfc flexi cap" in resolved.lower()


def test_newest_scheme_wins_the_scan(corpus):
    history = [
        _turn("user", "What is the AUM of HDFC Large Cap Fund?"),
        _turn("assistant", "₹39,933.37 cr."),
        _turn("user", "And of HDFC Balanced Advantage?"),
        _turn("assistant", "₹1,07,295.79 cr."),
    ]
    resolved, memo = with_memory_context("its rating?", history, corpus)
    assert memo["resolved_scheme"] == "hdfc-balanced-advantage-fund-direct-growth"


def test_the_window_is_limited_to_the_last_10_messages(corpus):
    older = [_turn("user", f"question number {i}") for i in range(5)]
    older.append(_turn("user", "tell me about HDFC Small Cap Fund"))
    recent = [_turn("user", f"unrelated filler {i}") for i in range(10)]
    history = older + recent
    resolved, memo = with_memory_context("and its exit load?", history, corpus)
    assert resolved == "and its exit load?"
    assert memo == {}


def test_no_scheme_anywhere_means_no_change(corpus):
    history = [_turn("user", "hello"), _turn("user", "what funds do you cover?")]
    resolved, memo = with_memory_context("and the expense ratio?", history, corpus)
    assert resolved == "and the expense ratio?"
    assert memo == {}


def test_a_dict_that_holds_an_answer_object_still_resolves(corpus):
    a = A.ask("What is the NAV of HDFC Small Cap Fund?")
    history = [{"role": "assistant", "answer": a}]
    resolved, memo = with_memory_context("its rating?", history, corpus)
    assert memo["resolved_scheme"] == "hdfc-small-cap-fund-direct-growth"


# ---------------------------------------------------------------------------
# ask() end-to-end
# ---------------------------------------------------------------------------


def test_follow_up_answers_with_the_scheme_from_history(corpus):
    history = [
        _turn("user", "What is the AUM of HDFC Flexi Cap Fund?"),
        _turn("assistant", "The AUM is ₹1,13,606.47 cr."),
    ]
    a = A.ask("And its expense ratio?", history=history)
    assert a.kind == "answer", f"refused instead of answered: {a.refusal_reason}"
    assert len(a.sources) == 1, f"expected exactly one source, got {len(a.sources)}"
    assert a.sources[0].url.endswith("hdfc-equity-fund-direct-growth"), \
        f"cited the wrong scheme's page: {a.sources[0].url}"
    assert a.debug.get("memory_context", {}).get("resolved_scheme") \
        == "hdfc-equity-fund-direct-growth"
    assert "0.77%" in a.text.lower().replace(",", ""), \
        f"expected the flexi cap expense ratio in: {a.text!r}"


def test_follow_up_with_a_proper_scheme_name_ignores_history(corpus):
    history = [_turn("user", "What is the NAV of HDFC Large Cap Fund?")]
    a = A.ask("What is the expense ratio of HDFC Small Cap Fund?", history=history)
    assert a.kind == "answer"
    assert a.sources[0].url.endswith("hdfc-small-cap-fund-direct-growth")
    assert "memory_context" not in a.debug


def test_follow_up_about_the_small_cap_one(corpus):
    history = [_turn("user", "How do I stop my HDFC ELSS SIP?"),
               _turn("assistant", "You can stop it from the Groww app."),
               _turn("user", "What is the rating of HDFC Small Cap Fund?")]
    a = A.ask("And the exit load?", history=history)
    assert a.kind == "answer", f"refused: {a.refusal_reason}"
    assert a.sources[0].url.endswith("hdfc-small-cap-fund-direct-growth")


def test_guardrails_judge_the_raw_text_not_the_resolved_memory(corpus):
    """A memory resolution must never turn an advice question into an answer."""
    history = [_turn("user", "What is the NAV of HDFC Large Cap Fund?")]
    a = A.ask("Should I invest more in it?", history=history)
    assert a.kind == "refusal_advice"