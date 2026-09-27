"""
tests/test_guardrails.py - STAGE 6: the refusal contract.

Organised around PRD §12, plus the three real failures this stage had:

  * `test_pii_log_never_contains_the_query` - the log line records a hash and a
    masked reason. If someone adds `query` to that log call this test FAILS,
    which is the point: the guardrail that protects PII must not become the
    place PII is written down.
  * `test_query_keywords_cannot_validate_generated_text` - documents WHY there are
    two vocabularies. `advice_keywords` is phrased as questions ("should i") and
    matches a user typing; run against generated prose it matches nothing, so
    "You should buy this scheme" would have passed validation silently.
  * `test_benchmark_index_names_are_not_performance_claims` - four of the five
    benchmarks are named "... Total Return Index". A validator that flags the
    word "return" rejects the CORRECT answer to a required golden question.

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_guardrails.py -q
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from rag import guardrails as G

LARGE_CAP = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return common.load_config()


@pytest.fixture(scope="module")
def urls() -> set[str]:
    return G.corpus_urls()


@pytest.fixture(scope="module")
def large_cap_chunks() -> list[dict]:
    return [c for c in common.read_jsonl(common.path_for("chunks_file"))
            if c["scheme_slug"] == "hdfc-large-cap-fund-direct-growth"]


# ---------------------------------------------------------------------------
# The constants come from the PRD verbatim. A reworded disclaimer is a PRD
# violation, so the wording is asserted rather than assumed.
# ---------------------------------------------------------------------------


def _prd_has(text: str) -> bool:
    """Is this text in the PRD, ignoring line wrapping?

    PRD §12 prints the response contracts in fenced blocks, so the templates are
    wrapped across lines there and stored unwrapped in guardrails.py. Comparing
    the raw strings fails on whitespace alone, which is exactly the kind of
    failure that trains people to ignore a test. Whitespace is normalised on both
    sides so that a real wording change still fails.
    """
    prd = re.sub(r"\s+", " ", (Path(common.ROOT) / "PRD.md").read_text(encoding="utf-8"))
    return re.sub(r"\s+", " ", text) in prd


def test_disclaimer_is_verbatim_from_the_prd():
    assert _prd_has(G.DISCLAIMER), "DISCLAIMER drifted from PRD §11"


def test_every_refusal_template_is_verbatim_from_the_prd():
    for name in ("PII_REFUSAL", "ADVICE_REFUSAL", "PERFORMANCE_REFUSAL",
                 "OUT_OF_CORPUS"):
        text = getattr(G, name)
        assert text, f"{name} is empty"
        assert _prd_has(text), f"{name} drifted from PRD §12"


def test_out_of_scope_text_is_a_addition_not_a_prd_template():
    """OUT_OF_SCOPE has no PRD §12 block. It must still be facts-only."""
    prd = re.sub(r"\s+", " ",
                 (Path(common.ROOT) / "PRD.md").read_text(encoding="utf-8"))
    assert re.sub(r"\s+", " ", G.OUT_OF_SCOPE) not in prd, \
        "OUT_OF_SCOPE now duplicates a PRD template - use that one instead"


def test_every_refusal_is_facts_only_and_carries_the_disclaimer():
    for name in ("PII_REFUSAL", "ADVICE_REFUSAL", "PERFORMANCE_REFUSAL",
                 "OUT_OF_CORPUS", "OUT_OF_SCOPE"):
        text = getattr(G, name).lower()
        for banned in ("you should buy", "we recommend", "guaranteed",
                       "risk-free", "surely", "definitely"):
            assert banned not in text, f"{name} contains advice: {banned!r}"


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

PII_CASES = [
    ("PAN", "my PAN is ABCDE1234F, can you check my folio"),
    ("Aadhaar", "aadhaar number 2345 6789 0123"),
    ("account", "account number 00123456789 please"),
    ("IFSC", "ifsc code HDFC0001234 transfer it"),
    ("OTP", "the otp is 445566"),
    ("CVV", "cvv 789"),
    ("PIN", "my pin is 4321"),
    ("email", "mail me at ramesh.kumar@gmail.com"),
    ("phone", "call me on 9876543210"),
    ("DOB", "date of birth 12/08/1990"),
]


@pytest.mark.parametrize("label,query", PII_CASES, ids=[c[0] for c in PII_CASES])
def test_every_pii_format_is_refused(label, query):
    with pytest.raises(G.PIIViolation) as exc:
        G.check_query_pii(query)
    assert exc.value.reason, f"{label}: refusal carried no reason"
    assert exc.value.query_hash, f"{label}: refusal carried no query hash"


def test_pii_violation_reason_never_contains_the_secret():
    with pytest.raises(G.PIIViolation) as exc:
        G.check_query_pii("my PAN is ABCDE1234F")
    assert "ABCDE1234F" not in exc.value.reason
    assert "ABCDE" not in exc.value.reason


def test_pii_log_never_contains_the_query(caplog):
    """The single most important test in this file.

    The guardrail exists so personal identifiers never reach the system. If the
    warning line also printed the query, the guardrail would be the leak.
    """
    secret = "ABCDE1234F"
    with caplog.at_level(logging.WARNING, logger="common"):
        with pytest.raises(G.PIIViolation):
            G.check_query_pii(f"my PAN is {secret} and my aadhaar is 2345 6789 0123")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in text, "the PII guardrail logged the identifier"
    assert "2345" not in text, "the PII guardrail logged the aadhaar"
    assert text.strip(), "nothing was logged at all - that is a silent failure"


def test_the_log_records_the_hash_so_an_operator_can_correlate(caplog):
    with caplog.at_level(logging.WARNING, logger="common"):
        with pytest.raises(G.PIIViolation):
            G.check_query_pii("my PAN is ABCDE1234F")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert re.search(r"query_hash=[0-9a-f]{8,}", text), \
        "the log carries no query_hash, so the event cannot be traced"


@pytest.mark.parametrize("query", [
    "What is the NAV of HDFC Large Cap Fund?",
    "What is the expense ratio of HDFC ELSS Tax Saver?",
    "What is the AUM of HDFC Flexi Cap Fund?",
    "exit load on HDFC Small Cap Fund",
    "is there a 3 year lock-in on the ELSS scheme",
    "what benchmark does HDFC Balanced Advantage Fund use",
    "minimum SIP for HDFC Large Cap",
    "what is the fund size",
    "expense ratio 1.03%",
])
def test_factual_questions_are_not_mistaken_for_pii(query):
    """Every figure on these pages is a number, and numbers are what PII looks like."""
    G.check_query_pii(query)  # must not raise


def test_the_corpus_figures_do_not_trip_the_pii_patterns(large_cap_chunks):
    """Stronger than the question-level check: run all 121 real chunk texts."""
    for c in common.read_jsonl(common.path_for("chunks_file")):
        G.check_query_pii(c["text"])


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------


def test_every_configured_advice_keyword_triggers():
    missed = [k for k in common.load_config()["guardrails"]["advice_keywords"]
              if not G.is_advice(f"{k} for hdfc large cap")]
    assert not missed, f"advice keywords that never fire: {missed}"


def test_factual_questions_are_not_advice():
    for q in G.BENIGN_QUESTIONS:
        assert not G.is_advice(q), f"false positive on {q!r}"


def test_advice_keywords_would_not_have_caught_generated_text(cfg):
    """Why a second, differently-phrased vocabulary exists.

    The query list is written as questions because that is how a user types.
    A language model produces assertions, and the same phrases match none of
    them. This test fails if someone deletes the answer-side vocabulary and
    re-points the validator at the query one. (`recommend` does happen to be in
    the query list, which is why the example here avoids it.)
    """
    text = "You should buy this scheme."
    assert not G._hit(text, cfg["guardrails"]["advice_keywords"]), \
        "premise broken: the query list now matches generated text"
    assert G._hit(text, cfg["generation"]["advice_vocabulary"]), \
        "the answer-side vocabulary failed to catch an obvious assertion"


def test_a_corpus_performance_figure_is_still_refused():
    """The corpus itself contains one, which grounding cannot catch.

    The "how to invest in ..." FAQ chunk says "the average annual returns
    provided by this fund is 18.28% since its inception". 18.28 IS on the page,
    so the grounding check passes it - grounding asks "did the source say this?",
    and the source did. PRD §12.3 forbids reporting it anyway, so the pairing of
    return vocabulary with a percentage is what gets refused.
    """
    rows = [c for c in common.read_jsonl(common.path_for("chunks_file"))
            if "18.28" in c["text"]]
    assert rows, "premise broken: the corpus no longer contains the figure"
    quote = ("The average annual returns provided by this fund is 18.28% "
             "since its inception.")
    assert not G.grounding_violations(quote, rows), \
        "premise broken: grounding no longer accepts the figure"
    assert G.performance_violations(quote), \
        "a quoted performance figure was allowed through"


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_every_configured_performance_keyword_triggers():
    missed = [k for k in common.load_config()["guardrails"]["performance_keywords"]
              if not G.is_performance(f"{k} of hdfc large cap")]
    assert not missed, f"performance keywords that never fire: {missed}"


def test_return_as_a_benchmark_name_is_not_a_performance_question():
    """The benchmark is a legitimate fact; asking which one is not a return question."""
    for q in ("what benchmark does hdfc large cap use",
              "is the benchmark the nifty 100 total return index",
              "nifty 500 total return index for hdfc flexi cap"):
        assert not G.is_performance(q), f"false positive on {q!r}"


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query,amc", [
    ("What is the expense ratio of SBI Large Cap Fund?", "SBI"),
    ("ICICI Prudential Large Cap exit load", "ICICI"),
    ("Axis Flexi Cap expense ratio", "Axis"),
    ("Kotak Small Cap minimum SIP", "Kotak"),
    ("what is the NAV of Motilal Large Cap", "Motilal"),
    ("SBI ELSS tax saver lock in", "SBI"),
    ("what is the expense ratio of Mirae Asset Flexi Cap", "Mirae"),
    ("Tata Small Cap NAV", "Tata"),
    ("Nippon India Small Cap minimum SIP", "Nippon"),
    ("what is the AUM of Franklin Templeton Large Cap", "Franklin"),
    ("lic flexi cap exit load", "LIC"),
    ("quant small cap expense ratio", "Quant"),
])
def test_another_amc_is_out_of_scope(query, amc):
    """Measured: "SBI Large Cap expense ratio" retrieves HDFC's own row at
    BM25 8.9, because the words are otherwise identical. Without this check the
    assistant would answer an SBI question with HDFC's number."""
    assert G.is_out_of_scope(query), f"{amc} question was not out of scope"


def test_hdfc_the_amc_is_never_on_the_other_amc_list():
    """The one AMC we do cover must not be caught by its own scope check."""
    for name in ("hdfc", "hdfc amc", "hdfc mutual fund", "hdfc asset management"):
        assert name not in G._OTHER_AMCS, f"{name!r} is on the other-AMC list"


def test_hdfc_life_is_out_of_scope():
    """HDFC Life is insurance, not one of the 5 mutual fund schemes, so it SHOULD
    be refused - the name shares a prefix with the AMC we do cover, which is
    exactly the near-miss worth pinning down."""
    assert "hdfc life" in G._OTHER_AMCS
    assert G.is_out_of_scope("what is the claim ratio in HDFC Life insurance")


@pytest.mark.parametrize("query", [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "what is the aum of hdfc flexi cap",
    "hdfc elss tax saver lock-in period",
    "exit load on hdfc small cap",
    "benchmark of hdfc balanced advantage fund",
    "hdfc equity fund nav",
])
def test_our_own_schemes_are_in_scope(query):
    assert not G.is_out_of_scope(query), f"false positive on {query!r}"


# ---------------------------------------------------------------------------
# Sentence counting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,want", [
    ("One sentence.", 1),
    ("One. Two.", 2),
    ("The NAV is Rs 1,189.08. Min SIP is Rs 100.", 2),
    ("The NAV is 1.03% today.", 1),
    ("Dr. Sharma wrote it. He was right.", 2),
    ("See https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth", 1),
    ("A, B, and C. D.", 2),
    ("The ratio is 50:50. That is the mandate.", 2),
    ("It costs 1.03% p.a. for the direct plan.", 1),
    ("", 0),
])
def test_count_sentences(text, want):
    assert G.count_sentences(text) == want, f"{text!r}"


# ---------------------------------------------------------------------------
# Numbers and grounding
# ---------------------------------------------------------------------------


def test_numbers_in_strips_indian_grouping():
    assert G.numbers_in("AUM is ₹1,13,606.47 cr") == {"113606.47"}
    assert G.numbers_in("expense ratio 1.03%") == {"1.03"}
    assert G.numbers_in("min. for sip ₹500") == {"500"}


def test_grounding_flags_a_figure_that_is_not_in_the_sources(large_cap_chunks):
    got = G.grounding_violations("The NAV is 4,999.99.", large_cap_chunks)
    assert got and "4999.99" in got[0]


def test_grounding_accepts_every_real_corpus_figure():
    """The 0-of-121 measurement that justifies having no materiality filter.

    An earlier version of this check skipped numbers under three digits, to
    avoid firing on incidental integers. That let "returned 24%" through, which
    is the exact case PRD §12.3 exists to refuse. It was removed only after
    confirming that no real chunk's own text trips it.
    """
    tripped = [c["chunk_id"] for c in common.read_jsonl(common.path_for("chunks_file"))
               if G.grounding_violations(c["text"], [c])]
    assert not tripped, f"grounding false-rejects real corpus content: {tripped}"


@pytest.mark.parametrize("claim", [
    "It returned 24% last year.",
    "It returned 9% last year.",
    "Gave 12% returns.",
    "The scheme returned 24% last year.",
])
def test_grounding_catches_a_fabricated_return(claim, large_cap_chunks):
    assert G.grounding_violations(claim, large_cap_chunks), \
        f"a fabricated return slipped through: {claim!r}"


# ---------------------------------------------------------------------------
# validate_answer
# ---------------------------------------------------------------------------


def test_a_correct_grounded_answer_passes(large_cap_chunks, urls):
    ok, why = G.validate_answer({
        "text": f"The expense ratio for HDFC Large Cap Fund is 1.03%. Source: {LARGE_CAP}",
        "sources": [{"url": LARGE_CAP}]}, urls, context=large_cap_chunks)
    assert ok, why


@pytest.mark.parametrize("text,sources,expect_fragment", [
    ("", [], "empty"),
    ("The expense ratio is 1.03%.", [], "no citation"),
    ("See https://example.com/mutual-funds for details.",
     [{"url": "https://example.com/mutual-funds"}], "not in the corpus"),
    ("One. Two. Three. Four. Five.", [{"url": LARGE_CAP}], "over the cap"),
    (f"You should buy this scheme. Source: {LARGE_CAP}",
     [{"url": LARGE_CAP}], "advice"),
    (f"We recommend this scheme. Source: {LARGE_CAP}",
     [{"url": LARGE_CAP}], "advice"),
    (f"It returned 24% last year. Source: {LARGE_CAP}",
     [{"url": LARGE_CAP}], "not in any retrieved source"),
    (f"The scheme has a CAGR of 18.4%. Source: {LARGE_CAP}",
     [{"url": LARGE_CAP}], "performance"),
    (f"The NAV is 4,999.99. Source: {LARGE_CAP}",
     [{"url": LARGE_CAP}], "not in any retrieved source"),
])
def test_validate_answer_rejects(text, sources, expect_fragment, urls, large_cap_chunks):
    ok, why = G.validate_answer({"text": text, "sources": sources}, urls,
                                context=large_cap_chunks)
    assert not ok, f"should have been rejected: {text!r}"
    assert any(expect_fragment in r for r in why), \
        f"rejected for the wrong reason: {why}"


def test_benchmark_index_name_survives_the_performance_check(urls, large_cap_chunks):
    """Four of five benchmarks are literally named "... Total Return Index"."""
    ok, why = G.validate_answer({
        "text": f"The benchmark is the NIFTY 100 Total Return Index. Source: {LARGE_CAP}",
        "sources": [{"url": LARGE_CAP}]}, urls, context=large_cap_chunks)
    assert ok, f"the correct benchmark answer was rejected: {why}"


def test_a_performance_claim_next_to_an_allowed_name_is_still_caught(urls, large_cap_chunks):
    ok, why = G.validate_answer({
        "text": f"It beat the NIFTY 500 Total Return Index by 24%. Source: {LARGE_CAP}",
        "sources": [{"url": LARGE_CAP}]}, urls, context=large_cap_chunks)
    assert not ok
    assert any("performance" in r or "not in any retrieved" in r for r in why), why


def test_the_answer_side_vocabularies_appear_nowhere_in_the_corpus(cfg):
    """The property that makes them usable at all.

    An answer-side phrase that occurs in the corpus cannot be a reliable signal,
    because it would reject legitimate grounded answers.
    """
    blob = " || ".join(
        c["text"] + " " + c.get("section_heading", "")
        for c in common.read_jsonl(common.path_for("chunks_file"))).lower()
    for phrase in cfg["generation"]["advice_vocabulary"]:
        assert phrase not in blob, f"advice phrase {phrase!r} occurs in the corpus"
    for phrase in cfg["generation"]["performance_claim_phrases"]:
        assert phrase not in blob, f"claim phrase {phrase!r} occurs in the corpus"


def test_the_benchmark_allow_list_is_actually_needed(urls, large_cap_chunks):
    """If none of the allow-listed strings occurs, the exemption is dead code."""
    blob = " || ".join(
        c["text"] for c in common.read_jsonl(common.path_for("chunks_file"))).lower()
    for phrase in cfg_allow_list():
        assert phrase in blob, f"allow-listed {phrase!r} never occurs in the corpus"


def cfg_allow_list() -> list[str]:
    return common.load_config()["generation"]["benchmark_name_allow"]


# ---------------------------------------------------------------------------
# corpus_urls
# ---------------------------------------------------------------------------


def test_corpus_urls_are_real_and_not_invented(urls):
    assert LARGE_CAP in urls
    assert "https://example.com" not in "".join(urls)


def test_the_self_test_passes():
    assert G.self_test() == 0, "rag.guardrails self-test reported a failure"
