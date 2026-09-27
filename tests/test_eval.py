"""Phase 8 evaluation harness tests.

Two jobs. First, pin the harness itself: a scoring bug that flatters the
numbers is worse than no harness. Second, pin the golden set against the
corpus, because every expected value in it is a claim about what the pages
actually say, and a hand-typed figure that drifts from the corpus turns the
whole report into fiction.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

import common
from eval import run_eval as R

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_set.csv"
REPORT = ROOT / "eval" / "report.md"
RESULTS = ROOT / "eval" / "results.json"

#: The PRD §15.1 split, plus one documented deviation: F31.
#:
#: The PRD's out-of-corpus examples are riskometer, capital-gains statement,
#: gold price and "open my account" - NOT "fund manager". The manager's name
#: is in fact in the corpus ("rahul baijal is the current fund manager of
#: hdfc large cap fund direct growth fund"), so classifying that question as
#: out-of-corpus would have been falsifying the golden set to make the score
#: look better. It is a real answerable row instead.
EXPECT_ANSWERABLE = 31
EXPECT_REFUSE = 23


# ---------------------------------------------------------------------------
# The golden set's own integrity
# ---------------------------------------------------------------------------


def test_every_row_has_the_five_required_columns_populated():
    """An answerable row must name the page it should cite.

    A refusal row must NOT: PRD §11 gives a refusal zero sources, so an empty
    URL there is the contract being asserted, not a gap in the data.
    """
    rows = R.load_rows()
    assert rows
    for r in rows:
        for col in ("qid", "category", "question", "expect_kind"):
            assert (r.get(col) or "").strip(), f"{r.get('qid')} is missing {col}"
        if r["expect_kind"] == "answer":
            assert (r.get("expected_source_url") or "").strip(), \
                f"{r['qid']} is answerable but names no expected_source_url"
        else:
            assert not (r.get("expected_source_url") or "").strip(), \
                f"{r['qid']} must refuse, so it must not name a source"
            assert (r.get("expected_value") or "").strip() == "", \
                f"{r['qid']} must refuse, so it must not assert a figure"


def test_no_row_has_a_shifted_column():
    """The bug this catches is a comma inside an unquoted field.

    A03 read "Which is better, HDFC Large Cap or HDFC Flexi Cap?" and the
    comma split it: `expect_kind` came out empty and the row silently stopped
    testing anything. The behaviour was right; the test was a no-op.
    """
    raw = GOLDEN.read_text(encoding="utf-8")
    for r in csv.DictReader(raw.splitlines()):
        assert None not in r, f"{r.get('qid')} has more fields than the header"
        assert r.get("expect_kind") in {
            "answer", "refusal_advice", "refusal_performance",
            "refusal_pii", "out_of_corpus", "error"}, r.get("expect_kind")


def test_qids_are_unique():
    qids = [r["qid"] for r in R.load_rows()]
    assert len(qids) == len(set(qids))


def test_the_split_matches_the_documented_composition():
    rows = R.load_rows()
    ans = [r for r in rows if r["expect_kind"] == "answer"]
    ref = [r for r in rows if r["expect_kind"] != "answer"]
    assert len(ans) == EXPECT_ANSWERABLE, sorted(r["qid"] for r in ans)
    assert len(ref) == EXPECT_REFUSE, sorted(r["qid"] for r in ref)


def test_the_prd_thresholds_are_met():
    """PRD §15.2. Read from the committed report, so this is a regression gate
    on the last recorded run rather than a fresh 53-question execution."""
    if not REPORT.exists():
        pytest.skip("no eval/report.md yet - run `python -m eval.run_eval`")
    text = REPORT.read_text(encoding="utf-8")
    s = json.loads(RESULTS.read_text(encoding="utf-8"))["summary"]
    assert s["answerable"]["citation_present"] == 100.0, "citation present must be 100%"
    assert s["answerable"]["citation_ok"] >= 90.0, "citation correct must be >= 90%"
    assert s["answerable"]["grounded"] >= 90.0, "groundedness must be >= 90%"
    assert s["must_refuse"]["kind_ok"] == 100.0, "refusal accuracy must be 100%"
    assert "Failures" in text


# ---------------------------------------------------------------------------
# PII safety of the harness output
# ---------------------------------------------------------------------------

#: The exact identifiers in the golden set's PII rows. If any of these reaches
#: the report, the JSON, or stdout, the harness has copied a user's data into a
#: file that gets handed to a class.
PII_LITERALS = [
    "ABCDE1234F",
    "1234567890123456",
    "ramesh.kumar@example.com",
]


def test_redact_blanks_every_identifier_shape():
    for lit in PII_LITERALS:
        assert lit not in R.redact(f"My PAN is {lit}, what is the TER?"), lit
    out = R.redact("email me at a.b@c.co and my aadhaar is 1234 5678 9012")
    assert "@" not in out, "email survived redaction"
    assert "1234 5678 9012" not in out, "grouped aadhaar survived redaction"
    assert "redacted" in out


def test_redact_leaves_ordinary_questions_alone():
    q = "What is the expense ratio of HDFC Large Cap Direct Growth?"
    assert R.redact(q) == q


@pytest.mark.skipif(not REPORT.exists(), reason="needs a recorded run")
def test_no_identifier_reaches_the_committed_artifacts():
    for path in (REPORT, RESULTS):
        blob = path.read_text(encoding="utf-8")
        for lit in PII_LITERALS:
            assert lit not in blob, f"{lit} leaked into {path.name}"


def test_no_identifier_reaches_stdout(capsys):
    rows = [r for r in R.load_rows() if r["category"] == "pii"]
    assert rows, "the golden set lost its PII rows"
    for r in rows:
        rec = R.score_row(r, _fake_answer(r["expect_kind"]), 0.01)
        printed = json.dumps(rec, default=str)
        for lit in PII_LITERALS:
            assert lit not in printed, f"{lit} survived into a scored record"


def _fake_answer(kind: str):
    from rag.answer import Answer
    from rag import guardrails as G

    return Answer(text="refused", sources=[], kind=kind, refusal_reason="r",
                  last_updated="", disclaimer=G.DISCLAIMER, debug={})


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def test_normalise_folds_currency_and_grouping():
    assert R.normalise("₹1,13,606.47 cr") == "113606.47 crore"
    assert R.normalise("1.03%") == "1.03"
    assert R.normalise("BSE 250 SmallCap TRI") == "bse 250 smallcap tri"


@pytest.mark.parametrize("expected,text,want", [
    ("1.03%", "The expense ratio is 1.03%.", True),
    ("1,189.08", "nav is ₹1,189.08", True),
    ("159.82", "nav is ₹159.82", True),
    ("0.78%", "the expense ratio is 1.03%", False),
    ("", "anything", None),
])
def test_grounded(expected, text, want):
    assert R.grounded(expected, text) is want


def test_short_figures_are_flagged_as_weak_grounding():
    """'100' is a substring of '1,100' and '100.5', so a substring check on a
    three-character figure proves much less than on a nine-character one."""
    assert R.significant("100") < R.STRICT_MIN_SIGNIFICANT
    assert R.significant("1,189.08") >= R.STRICT_MIN_SIGNIFICANT


def test_a_refusal_only_passes_on_the_exact_kind():
    """Refusing a performance question as PII is not a pass."""
    good = R.score_row({"qid": "P1", "category": "performance", "question": "q?",
                        "expect_kind": "refusal_performance", "expected_source_url": "",
                        "expected_value": ""},
                       _fake_answer("refusal_performance"), 0.0)
    wrong = R.score_row({"qid": "P1", "category": "performance", "question": "q?",
                         "expect_kind": "refusal_performance", "expected_source_url": "",
                         "expected_value": ""},
                        _fake_answer("refusal_pii"), 0.0)
    assert good["kind_ok"] is True
    assert wrong["kind_ok"] is False


def test_summarise_keeps_the_two_buckets_apart():
    recs = [
        {"qid": "F1", "category": "expense_ratio", "expect_kind": "answer",
         "kind_ok": True, "citation_present": True, "citation_ok": True,
         "grounded": True, "grounded_strict": True, "elapsed_ms": 10.0},
        {"qid": "A1", "category": "advice", "expect_kind": "refusal_advice",
         "kind_ok": True, "citation_present": False, "citation_ok": None,
         "grounded": None, "grounded_strict": None, "elapsed_ms": 1.0},
        {"qid": "A2", "category": "advice", "expect_kind": "refusal_advice",
         "kind_ok": False, "citation_present": False, "citation_ok": None,
         "grounded": None, "grounded_strict": None, "elapsed_ms": 1.0},
    ]
    s = R.summarise(recs)
    assert s["answerable"]["n"] == 1 and s["answerable"]["kind_ok"] == 100.0
    assert s["must_refuse"]["n"] == 2 and s["must_refuse"]["kind_ok"] == 50.0
    assert s["must_refuse"]["wrong_reason"] == ["A2"]


def test_a_crash_is_recorded_as_an_error_row_not_an_exception():
    row = {"qid": "F1", "category": "c", "question": "q?", "expect_kind": "answer",
           "expected_source_url": "https://x/", "expected_value": ""}
    a = R._crash("F1", RuntimeError("provider exploded"))
    rec = R.score_row(row, a, 0.0)
    assert rec["kind_ok"] is False and rec["got_kind"] == "error"
    assert "provider exploded" in rec["answer"]


# ---------------------------------------------------------------------------
# The golden set against the corpus
# ---------------------------------------------------------------------------


def test_every_expected_url_is_a_real_corpus_source():
    corpus_urls = {c.get("source_url") for c in common.read_jsonl(common.path_for("chunks_file"))}
    for r in R.load_rows():
        if r["expect_kind"] == "answer":
            assert r["expected_source_url"] in corpus_urls, \
                f"{r['qid']} cites a URL that is not in the corpus"


def test_every_expected_figure_really_appears_in_the_corpus():
    """Every hand-typed expected value is a claim about the pages.

    A figure that drifts from the corpus makes groundedness meaningless: the
    pipeline would be marked wrong for being right.
    """
    blob = " ".join(c["text"] for c in common.read_jsonl(common.path_for("chunks_file"))).lower()
    blob_ns = re.sub(r"[\s,]+", " ", blob)
    for r in R.load_rows():
        val = (r.get("expected_value") or "").strip()
        if not val or r["expect_kind"] != "answer":
            continue
        needle = re.sub(r"[\s,]+", " ", val.lower())
        assert needle in blob_ns, f"{r['qid']} expects {val!r}, which is not in the corpus"


def test_the_five_schemes_are_all_exercised():
    urls = {r["expected_source_url"] for r in R.load_rows() if r["expect_kind"] == "answer"}
    assert len(urls) == 5, sorted(urls)


def test_every_category_in_the_prd_table_is_present():
    cats = {r["category"] for r in R.load_rows()}
    for required in ("expense_ratio", "exit_load", "min_sip", "elss_lockin",
                     "benchmark", "nav_aum", "objective_pe_pb", "plan_nuance",
                     "advice", "performance", "pii", "out_of_corpus"):
        assert required in cats, f"category {required} is missing from the golden set"
