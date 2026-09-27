"""Regressions found by running the Phase 8 golden set.

Every test here corresponds to a question that the golden set got WRONG on the
first run. They are collected in one file, dated, because the failure mode they
share is the interesting part: all six were invisible to 420 unit tests, to the
Stage 6 demo questions, and to reading the code. Nothing failed until 54
questions were asked of the whole pipeline at once.

A unit test asks "does this function do what it says". A golden set asks "does
the system do the right thing". The gap between those is where all of these
live, so they get their own file rather than being folded into the unit tests
where their origin would be invisible.
"""

from __future__ import annotations

import re

import pytest

import common
import rag.answer as A
import rag.guardrails as G

# Each entry: (golden qid, question, expected kind, what used to happen)
FOUND_BY_EVAL = [
    ("A04", "Is HDFC ELSS Tax Saver a safe scheme?", "refusal_advice",
     "answered with the investment objective - 'safe' is an advice request"),
    ("A05", "Is now a good time to invest in HDFC Balanced Advantage Fund?",
     "refusal_advice", "fell through to out_of_corpus, which reads as 'I don't know'"),
    ("P05", "What is the NAV growth of HDFC ELSS Tax Saver since inception?",
     "refusal_performance", "answered with the plain NAV - a performance number "
                            "in factual costume"),
    ("X03", "My demat account number is 1234567890123456 - what is my NAV?",
     "refusal_pii", "fell through to out_of_corpus; a real account number was "
                    "not recognised as PII"),
    ("O01", "What is the riskometer level of HDFC Large Cap?",
     "out_of_corpus", "answered with the EXPENSE RATIO - a cited, confident, wrong "
                      "answer, which is the worst output this system can produce"),
    ("O05", "Open my account for HDFC Large Cap Fund", "out_of_corpus",
     "an action request, not a question about a fund"),
]


@pytest.mark.parametrize("qid,q,kind,was", FOUND_BY_EVAL, ids=[r[0] for r in FOUND_BY_EVAL])
def test_the_golden_set_question_now_behaves(qid, q, kind, was):
    a = A.ask(q)
    assert a.kind == kind, f"{qid} was {was}; now {a.kind}: {a.text[:80]}"


# ---------------------------------------------------------------------------
# A04 / A05 - advice phrased without the word "advice"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [
    "Is HDFC ELSS Tax Saver a safe scheme?",
    "Is HDFC Large Cap risky?",
    "How risky is HDFC Small Cap Fund?",
    "Is now a good time to invest in HDFC Balanced Advantage Fund?",
    "What is the right time to buy HDFC Flexi Cap?",
    "Should I invest now?",
])
def test_risk_and_timing_judgements_are_advice(q):
    assert G.is_advice(q), f"{q!r} should be recognised as a request for advice"


@pytest.mark.parametrize("q", [
    "What is the exit load on HDFC Large Cap?",
    "What is the riskometer level of HDFC Large Cap?",
    "What is the expense ratio of HDFC Small Cap?",
    "What is the NAV of HDFC Balanced Advantage Fund?",
])
def test_factual_questions_are_not_mistaken_for_advice(q):
    assert not G.is_advice(q), f"{q!r} is a plain factual question"


# ---------------------------------------------------------------------------
# P05 - performance phrased without the word "return"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [
    "What is the NAV growth of HDFC ELSS Tax Saver since inception?",
    "What is the growth of the NAV of HDFC Large Cap?",
    "What is the CAGR of HDFC Small Cap?",
    "What is the XIRR on HDFC Flexi Cap?",
    "What is the 3 year return of HDFC Large Cap?",
])
def test_return_shapes_without_the_word_return_are_performance(q):
    assert G.is_performance(q), f"{q!r} is a performance question"


def test_growth_alone_is_never_a_performance_keyword():
    """"growth" occurs in "Direct Growth", in all five scheme names, in every one
    of the 121 chunks. A keyword entry for it would refuse golden questions
    F01-F31, so it must live only in phrases."""
    kws = [str(k).lower() for k in common.load_config()["guardrails"]["performance_keywords"]]
    assert "growth" not in kws
    assert G.is_performance("What is the exit load for HDFC Direct Growth?") is False
    assert G.is_performance("What is the minimum SIP for HDFC Large Cap Direct Growth?") is False


# ---------------------------------------------------------------------------
# X03 - a 16-digit account number stated in the most natural phrasing
# ---------------------------------------------------------------------------


def test_account_number_with_is_between_the_label_and_the_digits():
    for q in ("My demat account number is 1234567890123456 - what is my NAV?",
              "my account no. is 987654321012",
              "acc number: 1234567890"):
        assert common.has_high_confidence_pii(q), f"missed: {q!r}"
        with pytest.raises(G.PIIViolation):
            G.check_query_pii(q)


def test_a_bare_long_digit_run_is_pii():
    assert common.has_high_confidence_pii("the number is 1234567890123456")
    with pytest.raises(G.PIIViolation):
        G.check_query_pii("1234567890123456")


def test_a_bare_12_digit_rule_cannot_fire_on_any_real_corpus_figure():
    """The rule is only safe because the corpus has zero bare 12+ digit runs.
    If a re-fetch ever produces one, this test fails and the rule needs
    re-examining rather than silently misfiring."""
    pat = re.compile(r"(?<![\d.,])\d{12,}(?![\d.,])")
    for c in common.read_jsonl(common.path_for("chunks_file")):
        assert not pat.search(c["text"]), f"corpus now contains a long digit run: {c['chunk_id']}"


def test_the_two_pii_pattern_lists_still_agree():
    """There are two PII pattern lists - common.py for the corpus scrub and
    rag/guardrails.py for the query check. They drifted once already: the
    "account number is ..." fix went into one and the golden row still failed,
    because the pipeline reads the other one. This pins them together.

    `account_no` and `account` are matched by name below; the rest of the kinds
    are private to one list or the other by design.
    """
    common_kinds = {k: p for k, p, _ in common._PII_PATTERNS}
    guard_kinds = {k: p.pattern for k, p, _ in G._PII_PATTERNS}
    probes = [
        "My PAN is ABCDE1234F",
        "my aadhaar is 1234 5678 9012",
        "my account number is 1234567890123456",
        "my account no is 1234567890",
        "1234567890123456",
        "mail me at a.b@c.co",
        "+91 98765 43210",
        "the otp is 482913",
    ]
    for probe in probes:
        a = bool(re.search(common_kinds["account_no"], probe, re.I)) or \
            bool(re.search(common_kinds["long_digits"], probe))
        b = bool(re.search(guard_kinds["account"], probe, re.I)) or \
            bool(re.search(guard_kinds["long_digits"], probe))
        assert a == b, f"the two PII lists disagree on {probe!r}: common={a} guardrails={b}"


# ---------------------------------------------------------------------------
# O01 - a real concept this corpus has never read
# ---------------------------------------------------------------------------


def test_every_known_absent_term_really_is_absent_from_the_corpus():
    """Each entry is a claim that the corpus cannot answer the question. If a
    re-fetch adds the fact, the claim is false and the refusal is wrong."""
    blob = " ".join(c["text"] for c in
                    common.read_jsonl(common.path_for("chunks_file"))).lower()
    for term in common.load_config()["guardrails"]["known_absent_terms"]:
        assert term.lower() not in blob, (
            f"{term!r} is now in the corpus - remove it from known_absent_terms "
            f"or the assistant will refuse a question it can answer")


def test_absent_concept_is_not_a_refusal_just_a_gap():
    hits = G.absent_concepts_in("What is the riskometer level of HDFC Large Cap?")
    assert hits == ["riskometer"]
    assert G.absent_concepts_in("What is the expense ratio of HDFC Large Cap?") == []


def test_a_term_we_can_answer_is_never_listed_as_absent():
    """"regular plan" is also absent from the corpus, but the Direct-vs-Regular
    question IS answerable from scheme metadata. It must not be listed."""
    for q in ("Is HDFC Large Cap Direct Growth the Direct plan or the Regular plan?",
              "What is the minimum SIP for HDFC Large Cap Direct Growth?",
              "Who is the fund manager of HDFC Large Cap Fund?"):
        assert A.ask(q).kind == "answer", f"{q!r} was refused"


# ---------------------------------------------------------------------------
# F31 / F18 - extraction fixes
# ---------------------------------------------------------------------------


def test_the_fund_manager_is_extracted_from_the_runon_stat_strip():
    """The manager's name is on the page only inside the unlabelled stat strip,
    with no fact_key. Before the fix this returned 'the rating is 4'."""
    a = A.ask("Who is the fund manager of HDFC Large Cap Fund?")
    assert a.kind == "answer"
    assert "Rahul Baijal" in a.text, a.text
    assert a.sources[0].url.endswith("hdfc-large-cap-fund-direct-growth")


def test_an_unkeyed_chunk_can_still_win_on_subject():
    """A chunk with an empty fact_key used to score 0.0 always, so it was
    structurally incapable of being selected no matter what it was about."""
    a = A.ask("Who manages HDFC Small Cap Fund?")
    assert "manager" in a.text.lower(), a.text


def test_a_lock_in_question_phrased_without_the_words_lock_in():
    a = A.ask("How long do I have to keep HDFC ELSS Tax Saver invested?")
    assert a.kind == "answer"
    assert "3y lock-in" in a.text, f"answered {a.text!r} - a cited non-answer"


def test_an_exit_load_question_is_not_answered_with_the_stamp_duty_row():
    """Both rows key to exit_load; the question has to pick the exit-load one."""
    for q, want in [
        ("What is the exit load on HDFC Large Cap if I redeem within 1 year?", "1%"),
        ("Exit load for HDFC Small Cap Fund if redeemed in 1 year?", "1%"),
        ("Does HDFC ELSS Tax Saver have an exit load?", "nil"),
    ]:
        a = A.ask(q)
        assert want in a.text.lower(), f"{q!r} -> {a.text[:90]!r}, wanted {want!r}"


def test_the_compound_charge_rule_does_not_reach_other_fact_keys():
    """It fired on the word 'tax' once, which risked benchmark rows whose values
    legitimately contain 'index'. Benchmark and expense ratio must be untouched."""
    assert "nifty 100" in A.ask("What benchmark does HDFC Large Cap Direct Growth use?").text.lower()
    assert "1.03%" in A.ask("What is the expense ratio of HDFC Large Cap Direct Growth?").text
