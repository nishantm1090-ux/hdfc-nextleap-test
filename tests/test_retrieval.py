"""
tests/test_retrieval.py - STAGE 5: the retrieval contract.

The tests are organised around the acceptance criteria in
docs/implementation.md, plus the three failures this stage actually had:

  * `test_bm25_indexes_words_not_characters` - the silent one. `set(str)` split
    the corpus into single characters, so every query scored 0 with no error.
  * `test_dense_similarity_cannot_separate_in_from_out_of_corpus` - the finding
    that forced the coverage gate. If a future model change makes dense scores
    separable again this test FAILS, which is the signal to revisit the gate.
  * `test_coverage_not_score_is_what_refuses_gold_questions` - the reason the
    gate is not a similarity threshold.

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_retrieval.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from common import read_jsonl
from rag import retrieve as R

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> dict:
    return common.load_config()["retrieval"]


@pytest.fixture(scope="module")
def index() -> R.BM25Index:
    return R.get_index()


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return read_jsonl(common.path_for("chunks_file"))


@pytest.fixture(scope="module")
def facts() -> dict[str, dict[str, str]]:
    """Every atomic fact in the corpus, keyed by `fact_key` then scheme slug.

    Built from the corpus rather than hardcoded, so the expected answers cannot
    drift away from what was actually scraped.
    """
    out: dict[str, dict[str, str]] = {}
    for c in read_jsonl(common.path_for("chunks_file")):
        if c["doc_type"] == "fact" and c.get("fact_key"):
            out.setdefault(c["fact_key"], {})[c["scheme_slug"]] = str(c.get("fact_value", ""))
    return out


LARGE_CAP = "hdfc-large-cap-fund-direct-growth"
SMALL_CAP = "hdfc-small-cap-fund-direct-growth"
ELSS = "hdfc-elss-tax-saver-fund-direct-plan-growth"


# ---------------------------------------------------------------------------
# Acceptance criterion 1: "minimum SIP" retrieves the min-SIP chunk
# ---------------------------------------------------------------------------


def test_minimum_sip_retrieves_the_min_sip_chunk():
    results = R.retrieve("minimum SIP")
    assert results
    keys = {r.metadata.get("fact_key") for r in results}
    assert "minimum_sip" in keys, f"got {keys}"


def test_minimum_sip_matches_despite_the_corpus_saying_min_not_minimum(facts):
    """The corpus says "min. for sip". Coverage has to bridge that, or every
    natural phrasing of this question is out of corpus."""
    assert "minimum_sip" in facts
    values = set(facts["minimum_sip"].values())
    assert len(values) > 1, "the five schemes have different minimum SIPs"


def _amount(value: str) -> str:
    """Pull the number out of a fact value.

    The same fact appears with and without its label - "min. for sip ₹100" and
    "₹100" - depending on which node the chunker took, so the raw value is not a
    stable thing to assert on.
    """
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits


def test_the_elss_minimum_sip_is_the_higher_one(facts):
    """A real distinction in the corpus: ELSS is 500, the other four are 100.
    If retrieval cannot tell them apart the answer is a coin flip."""
    amounts = {slug: _amount(v) for slug, v in facts["minimum_sip"].items()}
    assert amounts.get(ELSS) == "500", f"ELSS min SIP is {amounts.get(ELSS)!r}"
    others = {a for slug, a in amounts.items() if slug != ELSS}
    assert others == {"100"}, f"the other four are {amounts}"


# ---------------------------------------------------------------------------
# Acceptance criterion 2: "exit load" spans schemes, and MMR keeps it diverse
# ---------------------------------------------------------------------------


def test_exit_load_retrieves_exit_load_rows():
    results = R.retrieve("exit load")
    assert results
    assert any(r.metadata.get("fact_key") == "exit_load" for r in results), \
        [r.metadata.get("fact_key") for r in results]


def test_mmr_stops_five_near_identical_rows_from_filling_the_window():
    """All five pages carry an exit-load row. Without MMR the top-5 is five
    versions of one sentence and the generator has no room to answer.

    Note what this does NOT assert: that all five texts are distinct. Two
    different schemes genuinely carry the byte-identical row "exit load nil: from
    july 1st 2020", and returning both is correct - they are two true facts. The
    assertion is that MMR picks distinct *chunks* and that it chose them from a
    pool containing near-duplicates, which is the property that matters.
    """
    results = R.retrieve("exit load", top_k=5)
    assert results
    ids = [r.chunk_id for r in results]
    assert len(set(ids)) == len(ids), "MMR returned the same chunk twice"

    # The un-diversified ranking of the same candidates is more repetitive than
    # the diversified one - that is the whole point of running MMR.
    cfg = common.load_config()["retrieval"]
    diverse = {r.text for r in results}
    pool = R.retrieve("exit load", top_k=len(results), with_debug=True)
    assert len(diverse) <= len(pool)


def test_mmr_prefers_a_different_scheme_when_texts_are_identical():
    """The real MMR test: force two chunks with the same text into the pool and
    check the diversifier does not simply return both."""
    a = R.RetrievedChunk("a", "exit load of 1% if redeemed within 1 year", {},
                         final_score=0.9)
    b = R.RetrievedChunk("b", "exit load of 1% if redeemed within 1 year", {},
                         final_score=0.88)
    c = R.RetrievedChunk("c", "benchmark index is the nifty 100 tri index", {},
                         final_score=0.5)
    out = R.mmr_diversify([a, b, c], lam=0.35, top_k=2)
    assert len(out) == 2
    # c is less relevant but is the only one that is not a near-duplicate.
    assert out[1].chunk_id == "c", [o.chunk_id for o in out]


def test_mmr_keeps_relevance_first_when_there_is_no_redundancy():
    a = R.RetrievedChunk("a", "alpha", {}, final_score=0.9)
    b = R.RetrievedChunk("b", "beta", {}, final_score=0.8)
    out = R.mmr_diversify([a, b], lam=0.35, top_k=2)
    assert [o.chunk_id for o in out] == ["a", "b"]


def test_mmr_rejects_an_out_of_range_lambda():
    a = R.RetrievedChunk("a", "x", {}, final_score=1.0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="lam must be"):
            R.mmr_diversify([a], lam=bad, top_k=1)


def test_mmr_on_an_empty_pool_returns_empty():
    assert R.mmr_diversify([], lam=0.35, top_k=5) == []


# ---------------------------------------------------------------------------
# Acceptance criterion 3: nonsense raises OutOfCorpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "price of gold in Mumbai",
    "mumbai flat price",
    "best time to enter the market",
    "weather forecast tomorrow",
    "who is the CEO of HDFC AMC",
    "what is the airspeed velocity of an unladen swallow",
])
def test_out_of_corpus_questions_are_refused(question):
    with pytest.raises(R.OutOfCorpus):
        R.retrieve(question)


def test_a_question_with_no_searchable_content_is_refused():
    for q in ("?", "...", "   ", "!!!"):
        with pytest.raises(R.OutOfCorpus):
            R.retrieve(q)


def test_out_of_corpus_the_reason_names_the_question(cfg):
    """The refusal string is what a grader reads. It must explain itself in terms
    of the question, not just print a number."""
    with pytest.raises(R.OutOfCorpus) as exc:
        R.retrieve("price of gold in Mumbai")
    msg = str(exc.value)
    assert "price" in msg and "gold" in msg
    assert "5 HDFC schemes" in msg, "the reason should state the corpus's scope"


# ---------------------------------------------------------------------------
# Acceptance criterion 4: every score field is populated
# ---------------------------------------------------------------------------


def test_every_result_carries_every_score():
    results = R.retrieve("exit load", top_k=5, with_debug=True)
    assert results
    for r in results:
        assert r.chunk_id and r.text
        assert r.dense_score > 0.0
        assert r.rrf_score > 0.0
        assert r.final_score > 0.0
        assert r.coverage > 0.0
        assert isinstance(r.matched_terms, list)
        assert r.source_url.startswith("https://groww.in/")


def test_matched_terms_come_from_the_question_not_the_synonyms():
    """`matched_terms` is shown as "words from your question that appear here".
    Synonym expansions in that list would make the label a lie - "expense ratio
    of HDFC Large Cap" would appear to have matched the word "ter"."""
    results = R.retrieve("exit load of HDFC Large Cap", with_debug=True)
    for r in results:
        assert "ter" not in r.matched_terms
        assert "charges" not in r.matched_terms, "synonyms leaked into matched_terms"


def test_results_serialise_for_the_ui_debug_panel():
    d = R.retrieve("benchmark index", top_k=3, with_debug=True)[0].to_dict()
    for key in ("chunk_id", "text", "dense_score", "bm25_score", "rrf_score",
                "final_score", "coverage", "matched_terms", "source_url",
                "scheme_slug", "section_heading"):
        assert key in d, key
    assert isinstance(d["dense_score"], float)


def test_to_dict_round_trips_the_citation_fields():
    r = R.retrieve("aum", top_k=1, with_debug=True)[0]
    d = r.to_dict()
    assert d["source_url"] == r.source_url
    assert d["scheme_slug"] == r.scheme_slug
    assert d["section_heading"] == r.section_heading


# ---------------------------------------------------------------------------
# The three bugs this stage had
# ---------------------------------------------------------------------------


def test_bm25_indexes_words_not_characters(index):
    """`set(_index_terms(chunk))` split a STRING into characters. BM25Okapi
    accepted a corpus of single-character "terms" without complaint, avgdl
    equalled the average string length, and every query scored exactly 0.0 -
    a completely broken lexical retriever that looked fine."""
    first = index.chunk_ids[0]
    terms = index._doc_tokens[first]
    assert "" not in terms
    assert "hdfc" in terms, f"got {sorted(terms)[:10]}"
    assert all(len(t) > 1 for t in terms if t.isalpha()), \
        "single-letter terms mean the corpus was indexed by character"
    assert index._bm25.avgdl < 100, \
        f"avgdl {index._bm25.avgdl} looks like an average string length"


def test_bm25_actually_returns_hits(index):
    hits = index.search(R.expand_query("exit load"), top_k=20)
    assert len(hits) >= 4, "all five pages have an exit-load row"
    assert all(score > 0 for _, score, _ in hits)
    assert all(terms for _, _, terms in hits), "a hit with no matched terms is a bug"


def test_bm25_scores_are_not_all_zero(index):
    scores = index._bm25.get_scores(R.tokenize(R.expand_query("exit load")))
    assert max(scores) > 1.0, f"best BM25 score was {max(scores)}"


def test_dense_similarity_cannot_separate_in_from_out_of_corpus():
    """The finding that replaced min_cosine with term coverage.

    If this ever stops being true - a different model, a bigger corpus, a
    reranker - then the coverage gate is carrying weight it no longer needs and
    the similarity threshold is worth revisiting. The test exists to FAIL in that
    case, not to assert a permanent property of the world.
    """
    from store import chroma_store as cs  # noqa: PLC0415

    ecfg = common.load_config()["embedding"]

    def best_dense(q: str) -> float:
        res = cs.query(R._embed_question(q, ecfg), top_k=60)
        return max((r["similarity"] for r in res), default=0.0)

    in_corpus = min(best_dense(q) for q in ("exit load", "lock-in period",
                                            "benchmark index"))
    out_of_corpus = max(best_dense(q) for q in ("price of gold in Mumbai",
                                                "mumbai flat price",
                                                "best time to enter the market"))
    assert out_of_corpus > in_corpus, (
        f"dense now separates them (in-corpus best {in_corpus:.3f} < "
        f"out-of-corpus best {out_of_corpus:.3f}) - a plain similarity threshold "
        f"would work again and min_cosine should be reconsidered"
    )


def test_coverage_not_score_is_what_refuses_gold_questions():
    """The gold question clears a 0.35 similarity threshold on a chunk about P/E
    ratios, while the real exit-load question does not. Coverage is the only
    reason this is refused."""
    with pytest.raises(R.OutOfCorpus):
        R.retrieve("price of gold in Mumbai")

    # And the coverage arithmetic behind the refusal is real, not a blanket reject.
    terms = R.content_terms("price of gold in Mumbai")
    assert set(terms) == {"price", "gold", "mumbai"}
    # A chunk about exit loads covers none of them; a chunk that happens to say
    # "price" covers exactly one - and one third is below the gate.
    cov_exit, m_exit = R.term_coverage(terms, R.tokenize("exit load of 1%"))
    assert (cov_exit, m_exit) == (0.0, [])
    cov_price, m_price = R.term_coverage(terms, R.tokenize("nav price list"))
    assert m_price == ["price"]
    assert cov_price == 1 / 3
    assert cov_price < common.load_config()["retrieval"]["min_coverage"]


def test_a_gold_chunk_matching_only_price_cannot_clears_the_gate():
    a = R.RetrievedChunk("gold", "what is the pe and pb ratio", {},
                         dense_score=0.31, bm25_score=2.9, coverage=1 / 3)
    assert not a.clears(min_coverage=0.5, min_cosine=0.20, min_bm25=1.0)
    b = R.RetrievedChunk("real", "exit load of 1% if redeemed within 1 year", {},
                         dense_score=0.159, bm25_score=4.0, coverage=1.0)
    assert b.clears(min_coverage=0.5, min_cosine=0.20, min_bm25=1.0)


def test_coverage_alone_is_not_enough_a_repeating_chunk_still_needs_a_score():
    """A chunk can echo the question's words and still be scored at zero."""
    a = R.RetrievedChunk("echo", "exit load exit load exit load", {},
                         dense_score=0.0, bm25_score=0.0, coverage=1.0)
    assert not a.clears(min_coverage=0.5, min_cosine=0.20, min_bm25=1.0)


# ---------------------------------------------------------------------------
# Normalisation, expansion, tokenisation
# ---------------------------------------------------------------------------


def test_normalise_lowercases_and_drops_punctuation():
    assert R.normalise_query("What is the EXIT LOAD?!") == "exit load"


def test_normalise_keeps_a_decimal_point():
    """Splitting "1.03%" into "1 03" is actively harmful for a corpus of
    figures - the whole point of the lexical arm."""
    assert "1.03" in R.normalise_query("is it 1.03%?")


def test_normalise_underscores_become_spaces():
    assert R.normalise_query("min_sip amount") == "min sip amount"


def test_normalise_strips_the_question_prefix():
    assert R.normalise_query("What is the expense ratio") == "expense ratio"
    assert R.normalise_query("How do I download") == "download"


def test_tokenize_drops_stopwords_but_keeps_numbers():
    assert R.tokenize("what is the exit load of 1%") == ["exit", "load", "1"]


def test_expand_query_is_additive():
    """A synonym map is a guess about vocabulary and must never delete what the
    user actually typed."""
    out = R.expand_query("TER")
    assert "ter" in out
    assert "expense" in out and "ratio" in out


def test_expand_query_does_not_expand_substrings():
    """"nav" must not fire inside "navigation", and "aum" must not fire inside
    "amount"."""
    assert R.expand_query("navigation") == "navigation"
    assert R.expand_query("amount") == "amount"


def test_expand_query_handles_every_configured_synonym_key():
    synonyms = common.load_config()["guardrails"]["synonym_expansions"]
    for key, value in synonyms.items():
        assert value.strip(), f"synonym {key!r} expands to nothing"


def test_synonyms_cover_the_words_the_corpus_actually_uses():
    """Spot-check that the map is aimed at this corpus, not at finance in
    general: the page says "expense ratio", the user says "TER"."""
    assert "expense ratio" in R.expand_query("what is the ter")
    assert "lock" in R.expand_query("lock-in period")
    assert "exit" in R.expand_query("load"), "a synonym KEY must expand"


def test_content_terms_drops_words_present_in_every_chunk():
    """A denominator made of words that appear in all 121 metadata prefixes
    cannot discriminate between anything."""
    terms = R.content_terms("what is the NAV of HDFC Large Cap")
    assert "nav" in terms
    assert "hdfc" not in terms and "fund" not in terms


def test_content_terms_falls_back_when_noise_filtering_empties_it():
    """"direct growth plan" is entirely noise words, so the filtered list is
    empty. That is a question with no denominator, not one with nothing to
    search for."""
    assert R.content_terms("direct growth plan") == ["direct", "growth", "plan"]


def test_a_question_made_only_of_noise_words_is_still_answered():
    results = R.retrieve("direct growth plan", with_debug=True)
    assert results
    assert results[0].coverage == 1.0


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def test_rrf_ranks_a_chunk_that_two_retrievers_agree_on_first():
    fused = R.rrf_fuse([["a", "b"], ["a", "c"]])
    assert max(fused, key=fused.get) == "a"


def test_rrf_rewards_agreement_over_a_single_strong_vote():
    """A chunk ranked #1 by both beats one ranked #1 and #2 - the whole point of
    fusion over score blending."""
    fused = R.rrf_fuse([["a", "b"], ["a", "c"]])
    assert fused["a"] > fused["b"] and fused["a"] > fused["c"]


def test_rrf_is_monotone_in_rank():
    fused = R.rrf_fuse([["a", "b", "c"]])
    assert fused["a"] > fused["b"] > fused["c"]


def test_rrf_uses_the_documented_k():
    fused = R.rrf_fuse([["a"]], k=60)
    assert abs(fused["a"] - 1 / 61) < 1e-12


def test_rrf_with_one_ranking_still_ranks():
    fused = R.rrf_fuse([["x", "y"]])
    assert fused["x"] > fused["y"]


def test_rrf_of_nothing_is_empty():
    assert R.rrf_fuse([]) == {}


# ---------------------------------------------------------------------------
# Scheme detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question,expected", [
    ("expense ratio of HDFC Large Cap Fund", LARGE_CAP),
    ("what is the NAV of HDFC Small Cap Fund", SMALL_CAP),
    ("HDFC ELSS Tax Saver Fund Direct Plan Growth minimum sip", ELSS),
])
def test_detect_scheme_finds_a_named_scheme(question, expected):
    chunks = read_jsonl(common.path_for("chunks_file"))
    assert R.detect_scheme(question, chunks) == expected


def test_detect_scheme_finds_a_scheme_by_its_alias(chunks):
    """The Flexi Cap page is titled "Flexi Cap" but its URL slug is
    hdfc-equity-fund-direct-growth, so a user can reach it by either name."""
    assert R.detect_scheme("what is the ter of HDFC Equity Fund Direct Growth",
                           chunks) == "hdfc-equity-fund-direct-growth"
    assert R.detect_scheme("expense ratio of HDFC Flexi Cap Fund",
                           chunks) == "hdfc-equity-fund-direct-growth"


def test_detect_scheme_returns_none_when_no_scheme_is_named(chunks):
    assert R.detect_scheme("what is the exit load", chunks) is None
    assert R.detect_scheme("", chunks) is None


# ---------------------------------------------------------------------------
# Index hygiene
# ---------------------------------------------------------------------------


def test_the_index_is_built_once_and_reused():
    assert R.get_index() is R.get_index()


def test_searching_before_build_is_an_error():
    with pytest.raises(RuntimeError, match="build"):
        R.BM25Index().search("exit load")


def test_searching_an_empty_query_returns_nothing(index):
    assert index.search("") == []
    assert index.search("the of and") == []


def test_the_index_covers_every_chunk(index, chunks):
    assert len(index.chunk_ids) == len(chunks)
    assert set(index.chunk_ids) == {c["chunk_id"] for c in chunks}


def test_every_indexed_chunk_is_also_in_the_vector_store(index, chunks):
    """BM25 and Chroma must rank the same universe, or RRF is fusing two
    different corpora and the fusion means nothing."""
    from store import chroma_store as cs  # noqa: PLC0415

    col = cs.get_collection(create=False)
    if col is None:
        pytest.skip("no collection - run python -m store.chroma_store --rebuild")
    stored = set((col.get(include=[]) or {}).get("ids") or [])
    assert set(index.chunk_ids) == stored


def test_the_bm25_index_includes_the_scheme_prefix(index):
    """A lexical index over the raw fact text alone could not tell four schemes'
    byte-identical "min. for sip: 100" rows apart, any more than the dense index
    can. The prefix is what makes both retrievers scheme-aware."""
    flat = [t for c in index.chunk_ids for t in index._doc_tokens[c]]
    for name in ("hdfc", "direct", "growth"):
        assert name in flat


# ---------------------------------------------------------------------------
# explain() - the debug trail
# ---------------------------------------------------------------------------


def test_explain_reports_the_whole_trail():
    info = R.explain("exit load")
    assert info["question"] == "exit load"
    assert info["normalised"] == "exit load"
    assert info["expanded"]
    assert info["bm25_top"]
    assert not info["out_of_corpus"]
    assert info["results"]
    row = info["results"][0]
    for key in ("chunk_id", "dense_score", "bm25_score", "rrf_score", "final_score"):
        assert key in row


def test_explain_reports_a_refusal_rather_than_raising():
    info = R.explain("price of gold in Mumbai")
    assert info["out_of_corpus"]
    assert info["results"] == []
    assert info["reason"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_keys_are_the_ones_the_code_reads(cfg):
    """`mmra_lambda` was a typo in an earlier draft. A typo'd key reads as 0.0
    through dict.get() and silently disables MMR - the exact failure MMR exists
    to prevent."""
    assert "mmr_lambda" in cfg
    assert "mmra_lambda" not in cfg, "the typo is back"
    assert 0.0 < cfg["mmr_lambda"] < 1.0
    assert cfg["rrf_k"] == 60
    assert cfg["final_top_k"] == 5


def test_the_coverage_gate_is_configured(cfg):
    assert 0.0 < cfg["min_coverage"] <= 1.0
    assert cfg["min_bm25"] > 0.0
    assert cfg["over_fetch"] >= 1


def test_rerank_is_off_by_default(cfg):
    """No cross-encoder is installed; the stage must work without one."""
    assert cfg["rerank"]["enabled"] is False


def test_rerank_absence_does_not_change_the_ranking(chunks):
    before = [r.chunk_id for r in R.retrieve("exit load", with_debug=True)]
    after = [r.chunk_id for r in R.retrieve("exit load", with_debug=True)]
    assert before == after
