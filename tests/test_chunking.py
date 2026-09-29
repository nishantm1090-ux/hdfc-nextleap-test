"""
tests/test_chunking.py - STAGE 2: the ADR-001 acceptance suite.

The headline assertion is `test_fact_label_value_never_separated`, which is the
whole reason ADR-001 exists. Everything else here is either a contract check
(FR-2.1 .. FR-2.5) or a guard against a regression that would silently produce a
wrong answer.

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_chunking.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from common import count_tokens, read_json, read_jsonl
from ingest.chunk import (
    REQUIRED_CHUNK_FIELDS,
    build_chunk_id,
    canonical_fact_key,
    chunk_all,
    group_sections,
    split_facts,
    split_prose,
    validate_chunk,
)


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    cs = read_jsonl(common.path_for("chunks_file"))
    assert cs, "no chunks - run python -m ingest.chunk"
    return cs


@pytest.fixture(scope="module")
def ck() -> dict:
    return common.load_config()["chunking"]


@pytest.fixture(scope="module")
def stats() -> dict:
    s = read_json(common.path_for("chunk_stats_file"), None)
    assert s, "no chunk stats - run python -m ingest.chunk"
    return s


# ---------------------------------------------------------------------------
# The ADR-001 acceptance test
# ---------------------------------------------------------------------------


def test_fact_label_value_never_separated(chunks):
    """0.0% of facts may lose their value. This is the whole point of ADR-001.

    A chunk that says "Expense ratio" without saying "1.03%" is not a smaller
    fact, it is a different and wrong fact, and the generator will happily write
    "the expense ratio is low" around it.
    """
    facts = [c for c in chunks if c["doc_type"] in ("fact", "faq")]
    assert facts, "no fact chunks at all - the fact family is missing"

    separated = []
    for c in facts:
        value = (c.get("fact_value") or "").strip().lower()
        if not value:
            continue
        if value not in c["text"].lower():
            separated.append((c["chunk_id"], c["fact_key"], c["text"][:120]))

    pct = 100.0 * len(separated) / len(facts)
    assert not separated, (
        f"{len(separated)}/{len(facts)} facts ({pct:.2f}%) lost their value - "
        f"re-open ADR-001:\n" + "\n".join(f"    {s[2]!r}" for s in separated[:5])
    )


def test_fact_chunks_respect_the_fact_cap(chunks, ck):
    """A fact is one label + one value; it must fit the fact cap or be reclassified."""
    cap = int(ck["fact_max_tokens"])
    for c in chunks:
        if c["doc_type"] in ("fact", "faq"):
            assert c["token_count"] <= cap, (
                f"{c['chunk_id']} is a {c['doc_type']} of {c['token_count']} tokens "
                f"(cap {cap}) - it should have been declined and routed to prose"
            )


# ---------------------------------------------------------------------------
# FR-2.1 both families exist
# ---------------------------------------------------------------------------


def test_both_chunk_families_present(chunks):
    kinds = {c["doc_type"] for c in chunks}
    assert "fact" in kinds, "no atomic fact chunks (FR-2.1)"
    assert "prose" in kinds, "no prose chunks (FR-2.1)"


def test_fact_chunks_carry_a_canonical_fact_key(chunks):
    for c in chunks:
        if c["doc_type"] in ("fact", "faq"):
            assert c["fact_key"], f"{c['chunk_id']} has no fact_key"
            assert c["fact_key"] in FACT_KEYS, f"unknown fact_key {c['fact_key']!r}"


FACT_KEYS = {
    "expense_ratio", "exit_load", "entry_load", "minimum_sip",
    "minimum_investment", "lock_in_period", "riskometer_level", "benchmark",
    "aum", "nav", "statement_download", "stamp_duty", "fund_objective",
    "fund_category", "fund_managers", "taxation", "risk_profile", "rating",
    "fund_launch", "contact", "how_to_invest", "how_to_redeem",
    "sip_vs_lumpsum", "pe_pb_ratio",
}


# ---------------------------------------------------------------------------
# FR-2.2 / FR-2.3 budgets and overlap
# ---------------------------------------------------------------------------


def test_no_chunk_exceeds_the_token_cap(chunks, ck):
    cap = int(ck["prose_max_tokens"])
    over = [c for c in chunks if c["token_count"] > cap]
    assert not over, f"{len(over)} chunks over the {cap}-token cap: " \
                     f"{[(c['chunk_id'], c['token_count']) for c in over[:3]]}"


def test_prose_chunks_meet_the_minimum_size(chunks, ck):
    """FR-2.2: prose is chunked to a 1000-token target, not to stray sentences.

    Fact chunks are exempt by design - a fact IS short. This test therefore
    measures prose only, which is the only place the 40-token floor means anything.
    """
    floor = int(ck["prose_min_tokens"])
    short = [c for c in chunks if c["doc_type"] == "prose" and c["token_count"] < floor]
    assert not short, f"{len(short)} prose chunks under {floor} tokens: " \
                      f"{[(c['chunk_id'], c['token_count']) for c in short[:3]]}"


def test_overlap_only_on_prose(chunks):
    """FR-2.3. Facts must never carry duplicated neighbours' text."""
    for c in chunks:
        if c["doc_type"] in ("fact", "faq"):
            assert "returning" not in c["text"][:60], \
                "a fact chunk appears to begin with overlap text"


# ---------------------------------------------------------------------------
# FR-2.4 determinism
# ---------------------------------------------------------------------------


def test_chunk_ids_are_unique(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), \
        f"{len(ids) - len(set(ids))} duplicate chunk_ids"


def test_chunk_id_is_content_addressed(chunks):
    """The id must be a function of the text alone, not of position or time."""
    for c in chunks:
        assert build_chunk_id(c["scheme_slug"], c["section_heading"], c["text"]) == c["chunk_id"], \
            f"{c['chunk_id']} is not reproducible from its own fields"


def test_rechunking_reproduces_the_same_ids():
    """The Chroma cache and the embedding cache both depend on this."""
    before = [c["chunk_id"] for c in read_jsonl(common.path_for("chunks_file"))]
    chunk_all(force=True)
    after = [c["chunk_id"] for c in read_jsonl(common.path_for("chunks_file"))]
    assert before == after, "re-chunking produced different chunk_ids (FR-2.4)"


# ---------------------------------------------------------------------------
# FR-2.5 stats
# ---------------------------------------------------------------------------


def test_chunk_stats_file_is_complete(stats, chunks):
    assert stats["strategy"] == "structure_first_hybrid"
    assert stats["totals"]["chunks"] == len(chunks)
    for slug in [s["scheme_slug"] for s in common.load_schemes()]:
        assert slug in stats["per_scheme"], f"no per-scheme stats for {slug}"
        assert stats["per_scheme"][slug]["chunks"] > 0, f"{slug} produced no chunks"
    for key in ("mean", "median", "min", "max", "cap"):
        assert key in stats["tokens"], f"stats.tokens missing {key}"
    assert "quarantined" in stats["totals"]


def test_adr001_evidence_is_zero(stats):
    assert stats["adr001_evidence"]["fact_label_value_separated_pct"] == 0.0


# ---------------------------------------------------------------------------
# The metadata contract - a chunk that cannot be cited is a bug
# ---------------------------------------------------------------------------


def test_every_chunk_carries_the_full_metadata_schema(chunks):
    for c in chunks:
        missing = [f for f in REQUIRED_CHUNK_FIELDS if f not in c]
        assert not missing, f"{c['chunk_id']} missing {missing}"


def test_every_chunk_has_a_citable_url(chunks):
    for c in chunks:
        assert c["source_url"].startswith("https://"), \
            f"{c['chunk_id']} has an unusable source_url {c['source_url']!r}"


def test_every_chunk_states_its_source_and_date(chunks):
    """PRD requires a 'Last updated from sources' stamp in every answer, which
    means the date has to survive all the way from the fetch to the chunk."""
    for c in chunks:
        assert c["page_title"], f"{c['chunk_id']} has no page_title"
        assert c["publisher"], f"{c['chunk_id']} has no publisher"
        assert c["source_fetched_at"], f"{c['chunk_id']} has no source_fetched_at"
        assert c["scheme_slug"], f"{c['chunk_id']} has no scheme_slug"


def test_token_count_matches_the_text(chunks):
    for c in chunks:
        assert count_tokens(c["text"]) == c["token_count"], \
            f"{c['chunk_id']} token_count is stale"


# ---------------------------------------------------------------------------
# Scope guards baked into the chunker
# ---------------------------------------------------------------------------


def test_no_chunk_contains_return_data(chunks):
    """We are forbidden to state returns (PRD FR-3.3)."""
    banned = ("annualised return", "annualized return", "absolute return",
              "1 year return", "since inception return", "return calculator",
              "cagr", "xirr")
    for c in chunks:
        low = c["text"].lower()
        for b in banned:
            assert b not in low, f"{c['chunk_id']} contains {b!r}: {c['text'][:120]!r}"


def test_no_chunk_contains_holdings_data(chunks):
    """Portfolio holdings must stay out of the index.

    Asserted in the negative direction only, because on this corpus there is no
    positive case to keep: every AMC scheme page renders "Portfolio Allocation
    & Top Holdings" as a section title plus the single line "As on 31 Aug 2026",
    and the actual holdings table is drawn client-side, so it never reaches the
    captured HTML at all.

    That section title is the real regression risk. It was briefly indexed as
    the fact "portfolio allocation & top holdings: as on 31 aug 2026" - a fact
    with no content in it - which then anchored a prose chunk and surfaced in
    answers. So it is asserted as a positive too: the title may exist as prose
    context, but never as a stored value.
    """
    import re

    listing = re.compile(r"holdings?\s*(?:\(|\[|[:\-/])\s*[\d.]")
    for c in chunks:
        assert not listing.search(c["text"].lower()), \
            f"{c['chunk_id']} contains a holdings listing"

    dated = [c for c in chunks
             if c["fact_key"] and re.match(r"(?i)^as on \d", c.get("fact_value") or "")]
    assert not dated, f"a section's as-on date became a stored value: {dated[:1]}"


def test_no_chunk_contains_pii(chunks):
    from common import scan_pii

    for c in chunks:
        high = [m for m in scan_pii(c["text"]) if m.confidence == "high"]
        assert not high, f"{c['chunk_id']} contains PII: {[m.kind for m in high]}"


# ---------------------------------------------------------------------------
# Corpus quality gate
# ---------------------------------------------------------------------------


def test_chrome_pages_are_rejected_not_silently_indexed(stats):
    """A page that yields nothing answerable must be REPORTED as rejected.

    A corpus that quietly drops an input looks identical to a corpus that
    quietly lost it, so the rejection is part of the published stats rather than
    a side effect. On this corpus exactly one fetched page is rejected: the AMFI
    basics page, which is a navigation shell.
    """
    rejected = stats.get("rejected_documents", {})
    assert "amfi-mutual-fund-basics" in rejected, \
        f"the AMFI basics page should still be rejected; got {sorted(rejected)}"
    for doc_id, why in rejected.items():
        assert why, f"{doc_id} was rejected without a stated reason"
    for c in read_jsonl(common.path_for("chunks_file")):
        assert c["scheme_slug"] not in rejected, \
            f"{c['chunk_id']} came from a rejected document"


# ---------------------------------------------------------------------------
# Unit-level behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,expected", [
    ("Expense ratio", "expense_ratio"),
    ("Exit load", "exit_load"),
    ("Min. for SIP", "minimum_sip"),
    ("Fund size (AUM)", "aum"),
    ("NAV", "nav"),
    ("Benchmark", "benchmark"),
    ("Stamp duty", "stamp_duty"),
    ("Lock-in period", "lock_in_period"),
    ("Riskometer", "riskometer_level"),
    ("How to invest", "how_to_invest"),
])
def test_canonical_fact_key(label, expected):
    assert canonical_fact_key(label) == expected


@pytest.mark.parametrize("label", ["", "   ", "Frequently asked questions",
                                  "About the fund", "Total investments"])
def test_non_facts_get_no_fact_key(label):
    """Returning a key for a non-fact would invent a classification."""
    assert canonical_fact_key(label) is None


def test_split_facts_declines_an_oversized_fact(ck):
    """A 600-token FAQ answer is a section, not a fact - it must go to prose.

    Truncating it at the 120-token fact cap is the exact ADR-001 failure: the
    question survives, the answer is cut mid-sentence.
    """
    doc = {"doc_id": "d", "source_url": "https://x", "scheme_slug": "d"}
    node = {"type": "accordion", "order": 0,
            "label": "How to invest in the fund?",
            "value": "You can invest. " * 200,
            "text": "How to invest in the fund?: You can invest. " * 200}
    section = type("S", (), {"heading": "FAQ", "level": 2, "nodes": [node]})()
    chunks, consumed = split_facts(section, doc, ck, 0)
    assert chunks == [], "an oversized 'fact' was emitted instead of declined"
    assert not consumed, "a declined node was marked consumed, hiding it from prose"


def test_prose_does_not_repeat_a_fact_chunk(ck):
    """A fact is either its own chunk or part of the narrative, never both."""
    doc = {"doc_id": "d", "source_url": "https://x", "scheme_slug": "d"}
    fact = {"type": "table_row", "order": 7, "label": "Expense ratio",
            "value": "1.03%", "text": "Expense ratio: 1.03%"}
    other = {"type": "paragraph", "order": 8, "text": "The fund follows a value strategy."}
    section = type("S", (), {"heading": "Overview", "level": 2,
                             "nodes": [fact, other]})()

    f_chunks, consumed = split_facts(section, doc, ck, 0)
    p_chunks = split_prose(section, doc, ck, len(f_chunks), skip_orders=consumed)

    assert len(f_chunks) == 1
    assert "1.03%" not in " ".join(c["text"] for c in p_chunks), \
        "the fact's text leaked into a prose chunk as well"
    assert "value strategy" in " ".join(c["text"] for c in p_chunks), \
        "the paragraph was lost along with the skipped fact"


def test_group_sections_puts_nodes_under_their_heading():
    """A heading starts a section; nodes before the first heading fall under the
    page title, so nothing that precedes the first h2 is silently dropped."""
    nodes = [
        {"type": "heading", "text": "Overview", "level": 2},
        {"type": "paragraph", "text": "body one", "order": 1},
        {"type": "heading", "text": "Fees", "level": 3},
        {"type": "paragraph", "text": "body two", "order": 3},
    ]
    sections = {s.heading: [n["text"] for n in s.nodes]
                for s in group_sections(nodes, "HDFC Large Cap")}
    assert sections["Overview"] == ["body one"]
    assert sections["Fees"] == ["body two"]
    assert "HDFC Large Cap" not in sections, \
        "empty sections must be filtered out, not emitted blank"


def test_group_sections_keeps_content_before_the_first_heading():
    """Real pages open with the fund-name pill block before any heading."""
    nodes = [
        {"type": "paragraph", "text": "HDFC Large Cap Fund - Direct Growth", "order": 0},
        {"type": "heading", "text": "Fees", "level": 2},
        {"type": "paragraph", "text": "expense ratio 1.03%", "order": 2},
    ]
    sections = group_sections(nodes, "HDFC Large Cap")
    lead = [s for s in sections if s.heading == "HDFC Large Cap"]
    assert lead and [n["text"] for n in lead[0].nodes] == \
        ["HDFC Large Cap Fund - Direct Growth"], "leading content was dropped"


def test_headings_keep_their_case_for_the_citation_line(chunks):
    """The section heading is shown to the reader, so it must not be lower-cased."""
    for c in chunks:
        h = c["section_heading"]
        assert h, f"{c['chunk_id']} has no section heading"
        if h[:1].isalpha():
            assert h[:1].isupper() or h[:1].isdigit() or h[:1] in "₹(", \
                f"{c['chunk_id']} heading is lower-cased: {h!r}"


def test_validate_chunk_rejects_an_uncitable_chunk():
    bad = read_jsonl(common.path_for("chunks_file"))[0]
    bad["source_url"] = ""
    with pytest.raises(ValueError, match="source_url"):
        validate_chunk(bad)


def test_validate_chunk_rejects_a_stale_token_count(chunks):
    bad = dict(chunks[0])
    bad["token_count"] = bad["token_count"] + 50
    with pytest.raises(ValueError, match="token_count"):
        validate_chunk(bad)
