"""
tests/test_ingest.py - STAGE 1: loading + cleaning.

These tests run against the REAL cached corpus in data/, because the point of
Stage 1 is "these five public pages yield these facts". Mocking the HTML would
assert only that the parser agrees with itself.

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_ingest.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from common import read_json, read_jsonl, scan_pii

RAW = Path(common.path_for("raw_dir"))
CORPUS = Path(common.path_for("corpus_file"))
SNAPSHOT = Path(common.path_for("snapshot_file"))


# ---------------------------------------------------------------------------
# Config gate: a URL must be declared before it can be fetched
# ---------------------------------------------------------------------------


def test_every_configured_url_is_a_cited_public_source():
    """Nothing outside config/sources.yaml may enter the corpus (PRD S4.2)."""
    allowed = set(common.load_schemes_urls()) if hasattr(common, "load_schemes_urls") else set()
    src = read_json(SNAPSHOT, {}) or {}
    for entry in src.get("sources", src if isinstance(src, list) else []):
        url = entry.get("url", "")
        assert url.startswith("https://"), f"non-https source: {url}"
        if allowed:
            assert url in allowed, f"{url} is not declared in config/sources.yaml"


def test_no_banned_publishers_in_corpus():
    """Third-party blogs / aggregators are out of scope (PRD S4.3)."""
    banned = ("medium.com", "wikipedia.org", "reddit.com", "youtube.com",
              "substack.com", "blogspot", "quora.com", "linkedin.com",
              "twitter.com", "x.com", "facebook.com", "instagram.com")
    for path in RAW.glob("*.html"):
        for b in banned:
            assert b not in path.name.lower(), f"{path.name} looks like a banned publisher"


# ---------------------------------------------------------------------------
# Stage 1a: fetch
# ---------------------------------------------------------------------------


def test_all_schemes_fetched():
    schemes = common.load_schemes()
    assert len(schemes) == 5, f"expected 5 schemes, config has {len(schemes)}"
    for s in schemes:
        assert (RAW / f"{s['scheme_slug']}.html").exists(), \
            f"no cached HTML for {s['scheme_slug']} - run python -m ingest.fetch --all"


def test_no_undeclared_files_sitting_in_raw():
    """data/raw/ must contain exactly the declared corpus - no more, no less.

    A leftover probe download is how an unvetted page quietly enters an index
    later, when some future code path globs the directory instead of reading
    config/sources.yaml. The config is the gate; the directory is a cache of it.
    """
    raw = Path(common.path_for("raw_dir"))
    declared = ({s["scheme_slug"] for s in common.load_schemes()}
                | {d["doc_id"] for d in common.load_fetchable_supplementary()})
    have = {p.stem for p in raw.glob("*.html")}
    assert not (have - declared), (
        f"undeclared HTML in data/raw/: {sorted(have - declared)}. "
        f"Either declare it in config/sources.yaml or delete the file."
    )
    assert not (declared - have), f"declared but not fetched: {sorted(declared - have)}"


def test_no_scheme_page_is_thin():
    """Every scheme page carried real content, so PRD risk R1 never fired.

    Measured on VISIBLE text (the same signal fetch.py uses), not raw HTML bytes:
    a page can be 800 KB of markup around 44 visible characters.
    """
    from ingest.fetch import THIN_PAGE_CHARS, _visible_chars

    for s in common.load_schemes():
        path = Path(common.path_for("raw_dir")) / f"{s['scheme_slug']}.html"
        vis = _visible_chars(path.read_text(encoding="utf-8", errors="replace"))
        assert vis >= THIN_PAGE_CHARS, (
            f"{path.name} has only {vis} visible chars (< {THIN_PAGE_CHARS}) - "
            f"it is a JS shell and should not be in the corpus"
        )


def test_ingestion_log_records_every_attempt():
    log = read_jsonl(common.path_for("ingestion_log"))
    assert log, "ingestion log is empty - every fetch must leave a record"
    for row in log:
        assert {"url", "status", "bytes", "fetched_at", "sha256"} <= set(row), \
            f"incomplete log row: {sorted(row)}"


# ---------------------------------------------------------------------------
# Stage 1b: clean
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> dict:
    c = read_json(CORPUS, None)
    assert c, "no corpus - run python -m ingest.fetch --all && python -m ingest.clean"
    return c


def test_corpus_documents(corpus):
    assert len(corpus["documents"]) == 7, "5 schemes + 2 supplementary pages"
    slugs = {d.get("scheme_slug") for d in corpus["documents"] if d.get("scheme_slug")}
    assert len(slugs) == 5, f"expected 5 scheme slugs, got {slugs}"


def test_every_document_is_citable(corpus):
    """A chunk with no URL could never back a citation (architecture S4 Stage 2)."""
    for d in corpus["documents"]:
        assert d.get("source_url", "").startswith("https://"), \
            f"{d['doc_id']} has no source_url"
        assert d.get("page_title"), f"{d['doc_id']} has no page_title"
        assert d.get("source_fetched_at"), f"{d['doc_id']} has no fetch timestamp"


def test_jsonld_faq_extracted_before_script_stripping(corpus):
    """Groww's FAQ lives in a JSON-LD <script>; the 8 Q&As must survive."""
    for d in corpus["documents"]:
        if not d.get("scheme_slug"):
            continue
        accordions = [n for n in d["nodes"] if n.get("type") == "accordion"]
        assert len(accordions) == 8, \
            f"{d['doc_id']} has {len(accordions)} FAQ nodes, expected 8"


def test_elss_lock_in_pill_survives_header_stripping(corpus):
    """The ELSS 3-year lock-in pill lives inside <header>, which we strip.

    Losing it would silently make the single most distinctive ELSS fact
    unanswerable, so it gets its own test.
    """
    elss = next(d for d in corpus["documents"]
                if d.get("scheme_slug") == "hdfc-elss-tax-saver-fund-direct-plan-growth")
    text = " ".join(n.get("text", "") for n in elss["nodes"]).lower()
    assert "lock" in text, "ELSS lock-in pill was lost during cleaning"
    assert "3y" in text or "3 y" in text or "3 year" in text, \
        "ELSS lock-in period (3 years) missing from the cleaned nodes"


@pytest.mark.parametrize("slug,expect", [
    ("hdfc-large-cap-fund-direct-growth", {"expense_ratio": "1.03", "aum": "39,933.37"}),
    ("hdfc-elss-tax-saver-fund-direct-plan-growth", {"expense_ratio": "1.21", "aum": "15,991.78"}),
    ("hdfc-equity-fund-direct-growth", {"expense_ratio": "0.77", "aum": "1,13,606.47"}),
    ("hdfc-small-cap-fund-direct-growth", {"expense_ratio": "0.78", "aum": "41,890.86"}),
    ("hdfc-balanced-advantage-fund-direct-growth", {"expense_ratio": "0.78", "aum": "1,07,295.79"}),
])
def test_live_fact_values_are_present(corpus, slug, expect):
    """Guard against inventing figures.

    These are the values the live pages showed on 2026-09-27. If a re-fetch ever
    changes them, this test FAILS ON PURPOSE - a changed source number must be
    re-verified against the page, never silently absorbed. The test message says
    so, because a "fix" that edits these numbers to match a new response is
    exactly how fabricated figures enter an RAG demo.
    """
    doc = next(d for d in corpus["documents"] if d.get("scheme_slug") == slug)
    text = " ".join(n.get("text", "") for n in doc["nodes"])
    for what, value in expect.items():
        assert value in text, (
            f"{slug}: {what} = {value!r} not found in cleaned nodes.\n"
            f"If the source page genuinely changed, re-open the number against the "
            f"live page and update sources.yaml's snapshot date - do not just edit "
            f"this test to match whatever came back."
        )


def test_minimum_sip_values(corpus):
    """Four schemes at Rs 100, ELSS at Rs 500 (live values, 2026-09-27)."""
    want = {
        "hdfc-large-cap-fund-direct-growth": "100",
        "hdfc-elss-tax-saver-fund-direct-plan-growth": "500",
        "hdfc-equity-fund-direct-growth": "100",
        "hdfc-small-cap-fund-direct-growth": "100",
        "hdfc-balanced-advantage-fund-direct-growth": "100",
    }
    for slug, sip in want.items():
        doc = next(d for d in corpus["documents"] if d.get("scheme_slug") == slug)
        pairs = {n.get("label", "").lower(): n.get("value", "")
                 for n in doc["nodes"] if n.get("label")}
        mips = [v for k, v in pairs.items() if "min" in k and "sip" in k]
        assert mips, f"{slug}: no 'min. for sip' node"
        assert any(f"₹{sip}" in v for v in mips), \
            f"{slug}: expected min SIP ₹{sip}, found {mips}"


# ---------------------------------------------------------------------------
# Banned content: returns must never reach the index
# ---------------------------------------------------------------------------

_RETURN_TABLE_MARKERS = (
    "annualised return", "annualized return", "absolute return",
    "1 year return", "since inception return", "return calculator",
)


def test_no_return_figures_survive_cleaning(corpus):
    """We must never state a return (PRD FR-3.3).

    Any <table> on these pages is a returns table or a holdings list, so both are
    excluded at clean time. This asserts the outcome, not the mechanism.
    """
    for d in corpus["documents"]:
        for n in d["nodes"]:
            t = (n.get("text") or "").lower()
            for marker in _RETURN_TABLE_MARKERS:
                assert marker not in t, \
                    f"{d['doc_id']} node {n.get('type')} still contains {marker!r}: {t[:120]!r}"


def test_no_holdings_listings(corpus):
    """A holdings table is ~850 stock names - pure retrieval noise."""
    for d in corpus["documents"]:
        blob = " ".join(n.get("text", "") for n in d["nodes"])
        assert "holdings" not in blob.lower() or len(blob) < 200000
        # No node may look like a ticker list.
        for n in d["nodes"]:
            if n.get("type") != "table_row":
                continue
            assert len(n.get("value", "")) < 400, \
                f"{d['doc_id']}: suspiciously long table value, maybe a holdings row"


# ---------------------------------------------------------------------------
# PII (PRD FR-1.5 / FR-7.1)
# ---------------------------------------------------------------------------


def test_corpus_is_pii_clean(corpus):
    """Nothing high-confidence may reach the index."""
    for d in corpus["documents"]:
        for n in d["nodes"]:
            hits = scan_pii(n.get("text", ""))
            high = [h for h in hits if h.confidence == "high"]
            assert not high, f"{d['doc_id']}: PII {high} in node {n.get('type')!r}"


def test_pii_scanner_flags_the_things_the_brief_forbids():
    """Positive control - a scanner that never fires is worse than none."""
    cases = {
        "pan": "My PAN is ABCDE1234F.",
        "aadhaar": "Aadhaar number 2345 6789 0123",
        "email": "write to investor.relations@hdfcmf.com please",
        "phone": "call 9876543210 for support",
        "otp": "your OTP is 482913",
    }
    for kind, text in cases.items():
        hits = [h for h in scan_pii(text) if h.confidence == "high"]
        assert hits, f"PII scanner missed {kind}: {text!r}"


def test_pii_scanner_does_not_flag_financial_figures():
    """A scanner that flags Rs 1,189.08 as an account number is useless."""
    clean = [
        "nav: ₹1,189.08 as on 25 sep '26",
        "expense ratio: 1.03% p.a.",
        "fund size (aum): ₹39,933.37 cr",
        "min. for sip: ₹500",
        "exit load of 1% if redeemed within 1 year",
        "stamp duty on investment: 0.005%",
        "pe 22.4 pb 3.1",
    ]
    for text in clean:
        high = [h for h in scan_pii(text) if h.confidence == "high"]
        assert not high, f"false positive {high} on {text!r}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_clean_is_idempotent_on_cached_html(corpus):
    """Re-cleaning the same cached HTML must give the same node text.

    Only ingested_at-style metadata may move; node text and order may not.
    """
    from ingest.clean import clean_all

    before = {d["doc_id"]: [n.get("text") for n in d["nodes"]]
              for d in corpus["documents"]}
    docs, _quarantined = clean_all(force=True)
    after = {d.doc_id: [n.text for n in d.nodes] for d in docs}
    assert before == after, "re-running ingest.clean changed the node text"


def test_no_node_text_is_empty(corpus):
    for d in corpus["documents"]:
        for n in d["nodes"]:
            assert n.get("text", "").strip(), \
                f"{d['doc_id']} order={n.get('order')} has an empty text node"
