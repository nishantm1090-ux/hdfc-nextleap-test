"""
tests/test_common.py - Phase 1 (bootstrap) test suite.

Covers the guarantees the rest of the pipeline relies on:
  * config + source list load and are well-formed
  * hashing / slugs are deterministic  (idempotency, PRD FR-2.4)
  * token counting works with AND without tiktoken
  * PII detection: true positives and, critically, no false positives on the
    financial figures this corpus is full of  (PRD FR-1.5 / FR-7.1)
  * JSONL round-trip
  * scrubbing never leaks the raw value

Run:  python -m pytest tests/test_common.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import common
from common import (
    content_hash,
    count_tokens,
    load_config,
    load_schemes,
    load_supplementary,
    read_jsonl,
    scrub_pii,
    scan_pii,
    short_hash,
    slugify,
    truncate_to_tokens,
    utc_date,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Config / sources
# ---------------------------------------------------------------------------


def test_app_yaml_loads():
    cfg = load_config()
    assert cfg["embedding"]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg["embedding"]["dimension"] == 384
    assert cfg["vector_store"]["backend"] == "chromadb"
    assert cfg["chunking"]["strategy"] == "structure_first_hybrid"


def test_exactly_five_schemes_all_direct_growth():
    schemes = load_schemes()
    assert len(schemes) == 5
    for s in schemes:
        assert s["plan"] == "Direct Growth"
        assert s["url"].startswith("https://groww.in/mutual-funds/")
        assert s["scheme_slug"] and s["category"] and s["short_name"]


def test_scheme_categories_are_the_five_required():
    categories = {s["category"] for s in load_schemes()}
    assert categories == {
        "Large Cap",
        "Flexi Cap",
        "ELSS",
        "Small Cap",
        "Balanced Advantage",
    }


def test_scheme_slugs_unique():
    slugs = [s["scheme_slug"] for s in load_schemes()]
    assert len(slugs) == len(set(slugs))


def test_supplementary_sources_have_publishers():
    """Allowlist of publishers we are willing to cite.

    The point is to keep third-party blogs, forums and content farms out
    (PRD S4.3), not to insist that only regulators may appear. Groww is here
    because it is the publisher of the 5 mandated scheme pages and of
    groww.in/help; SEBI/AMFI/HDFC AMC are here because they are the official
    bodies the brief asks for.
    """
    allowed = {"SEBI", "AMFI", "HDFC AMC", "Groww"}
    for d in load_supplementary():
        assert d["publisher"] in allowed, \
            f"{d['doc_id']} cites {d['publisher']!r}, which is not an approved publisher"


# ---------------------------------------------------------------------------
# Determinism / idempotency
# ---------------------------------------------------------------------------


def test_content_hash_is_whitespace_insensitive():
    a = "Expense ratio  is  1.03% p.a."
    b = "Expense ratio is 1.03% p.a."
    assert content_hash(a) == content_hash(b)


def test_content_hash_differs_on_real_change():
    # Large Cap (1.03%) vs ELSS (1.21%) - two real, sourceable values.
    assert content_hash("Expense ratio 1.03%") != content_hash("Expense ratio 1.21%")


def test_short_hash_stable_and_correct_length():
    assert short_hash("hdfc", 6) == short_hash("hdfc", 6)
    assert len(short_hash("hdfc", 6)) == 6


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HDFC Large Cap Fund", "hdfc_large_cap_fund"),
        ("Fees & charges (Direct)", "fees_charges_direct"),
        ("  3-year lock-in!  ", "3_year_lock_in"),
        ("", "unknown"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_count_tokens_positive_and_scales():
    short = count_tokens("Expense ratio 1.03% p.a.")
    longer = count_tokens("Expense ratio 1.03% p.a. " * 20)
    assert short > 0
    assert longer > short * 5


def test_truncate_respects_budget():
    text = "word " * 600
    out = truncate_to_tokens(text, 100)
    assert count_tokens(out) <= 100
    # A string already inside budget is returned untouched.
    assert truncate_to_tokens("short text", 100) == "short text"


def test_fallback_token_counter_used_when_tiktoken_absent(monkeypatch):
    """The offline fallback must stay within ~25% of tiktoken for budgeting."""
    monkeypatch.setattr(common, "_ENCODER", None)
    monkeypatch.setattr(common, "_ENCODER_TRIED", True)
    sample = "HDFC Large Cap Fund Direct Growth has an expense ratio of 1.03% per annum."
    approx = count_tokens(sample)
    assert 5 <= approx <= 40
    assert truncate_to_tokens(sample * 40, 60).split()


# ---------------------------------------------------------------------------
# PII - true positives
# ---------------------------------------------------------------------------

PII_SAMPLES = [
    ("My PAN is ABCDE1234F, what is the TER?", "pan"),
    ("call me on +91 98765 43210", "phone"),
    ("my mobile is 9876543210", "phone"),
    ("aadhaar 4567 8901 2345", "aadhaar"),
    ("email me at ravi.sharma@example.com", "email"),
    ("acct no 30123456789012", "account_no"),
    ("IFSC is HDFC0001234", "ifsc"),
    ("DP ID 12345678", "demat"),
]


@pytest.mark.parametrize("text,expected_kind", PII_SAMPLES)
def test_pii_detected(text, expected_kind):
    kinds = {m.kind for m in scan_pii(text) if m.confidence == "high"}
    assert expected_kind in kinds, f"expected {expected_kind} in {kinds}"


# ---------------------------------------------------------------------------
# PII - the important direction: no false positives on scheme facts
# ---------------------------------------------------------------------------

# Realistic answer-shaped strings for the PII scanner to run against. The
# figures are the LIVE values read off the five Groww pages on 2026-09-27
# (Large Cap TER 1.03%, min SIP Rs 100, AUM Rs 39,933.37 Cr, NAV Rs 1,189.08),
# deliberately not invented ones: a fixture in this repo asserting a number
# nobody can source is exactly the habit the project exists to discourage.
# The one exception is marked below.
FINANCIAL_FACTS = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "The TER is 1.03% p.a. and exit load is 1.00% under 12 months.",
    "Exit load: 1.00% if redeemed within 12 months, nil thereafter.",
    "The lock-in period is 3 years.",
    "Minimum SIP amount is Rs 100.",
    "Minimum SIP amount for HDFC ELSS is Rs 500.",
    "Minimum lump sum investment is Rs 5000.",
    "AUM is Rs 39,933.37 crore.",
    "The benchmark is NIFTY 100 Total Return Index.",
    "NAV is Rs 1,189.08.",
    "Expense ratio 1.03% p.a., charged as a daily percentage of the NAV.",
    "Direct plan growth option, 1000 minimum units.",
]


@pytest.mark.parametrize("text", FINANCIAL_FACTS)
def test_no_false_positive_on_financial_facts(text):
    high = [m for m in scan_pii(text) if m.confidence == "high"]
    assert high == [], f"false positive {[(m.kind, m.masked) for m in high]} in {text!r}"


def test_low_confidence_otp_never_treated_as_pii():
    """Bare 4-6 digit runs may match, but must never be 'high' confidence."""
    matches = scan_pii("AUM is Rs 50000 crore")
    assert all(m.confidence == "low" for m in matches)


def test_overlapping_patterns_do_not_double_report():
    """A phone number must not also be reported as a possible_otp substring."""
    matches = scan_pii("call me on +91 98765 43210")
    high = [m for m in matches if m.confidence == "high"]
    assert len(high) == 1
    assert high[0].kind == "phone"


# ---------------------------------------------------------------------------
# PII - masking / scrubbing
# ---------------------------------------------------------------------------


def test_scan_never_returns_raw_values():
    secret = "ABCDE1234F"
    for m in scan_pii(f"my pan is {secret}"):
        assert secret not in m.masked


def test_scrub_redacts_and_reports():
    text = "PAN ABCDE1234F and email a.b@x.com"
    clean, matches = scrub_pii(text)
    assert "ABCDE1234F" not in clean
    assert "a.b@x.com" not in clean
    assert clean.count("[REDACTED]") == 2
    assert {m.kind for m in matches} == {"pan", "email"}


def test_scrub_is_a_noop_on_clean_text():
    text = "Expense ratio is 1.03% per annum with a 3 year lock-in."
    clean, matches = scrub_pii(text)
    assert clean == text
    assert matches == []


def test_scrub_handles_multiple_spans_correctly():
    text = "pan1 ABCDE1234F pan2 ZXCVB9876K"
    clean, _ = scrub_pii(text)
    assert "ABCDE1234F" not in clean and "ZXCVB9876K" not in clean
    # Surrounding text must survive intact.
    assert "pan1" in clean and "pan2" in clean


# ---------------------------------------------------------------------------
# JSONL + misc
# ---------------------------------------------------------------------------


def test_jsonl_roundtrip(tmp_path: Path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "nested" / "out.jsonl"
    assert write_jsonl(p, rows) == 2
    assert read_jsonl(p) == rows


def test_read_jsonl_missing_file_is_empty(tmp_path: Path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_jsonl_rejects_corrupt_line(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(p)


def test_ensure_dirs_is_idempotent():
    common.ensure_dirs()
    common.ensure_dirs()
    assert Path(common.path_for("chunks_file")).parent.exists()


def test_path_for_unknown_key_raises():
    with pytest.raises(KeyError):
        common.path_for("no_such_path")


def test_utc_date_is_iso():
    d = utc_date()
    assert len(d) == 10 and d[4] == "-" and d[7] == "-"


def test_pii_report_is_log_safe():
    """The log summary must not contain raw PII - logs are committed evidence."""
    report = common.pii_report(scan_pii("pan ABCDE1234F mail a@b.com"))
    assert "ABCDE1234F" not in report
    assert "a@b.com" not in report
    assert "pan" in report and "email" in report


def test_pii_report_of_empty_is_clean():
    assert common.pii_report([]) == "clean"


# ---------------------------------------------------------------------------
# strip_inline_source - PRD FR-8.4, the one source link
# ---------------------------------------------------------------------------


def test_strip_inline_source_removes_a_single_trailing_source():
    text = "The expense ratio is 1.03%. Source: https://groww.in/f"
    assert common.strip_inline_source(text) == "The expense ratio is 1.03%."


def test_strip_inline_source_removes_both_trailing_copies_when_generator_inlines():
    """A generator that inlines the URL AND step 9 appends the citation leaves
    two trailing `Source:` tokens. Only one may ship - the citation block.
    """
    text = ("The expense ratio is 1.03%. Source: https://groww.in/f "
            "Source: https://groww.in/f")
    assert common.strip_inline_source(text) == "The expense ratio is 1.03%."


def test_strip_inline_source_keeps_prose_that_mentions_source_mid_sentence():
    text = "The Source: https://x link explains the scheme's benchmark 1.03%."
    assert common.strip_inline_source(text) == text
