"""
tests/test_embedding.py - STAGE 3: the embedding contract.

Two kinds of test live here:

  * cheap, always-on  - the lead-window text, the cache key, the index file.
    These catch the bug this stage actually had (see test_cache_key_must_not_be
    the raw content hash).
  * model-dependent   - semantic sanity checks that MiniLM really did learn
    something. They skip cleanly when the model cannot be downloaded, so a
    fresh clone on a plane still gets a green suite instead of a stack trace.

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_embedding.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from common import read_json, read_jsonl
from embed.index import (
    EmbedStats,
    build_embed_text,
    embed_all,
    embed_chunks,
    embed_hash,
    load_cache,
    load_model,
    model_is_cached,
)


@pytest.fixture(scope="module")
def cfg() -> dict:
    return common.load_config()["embedding"]


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    cs = read_jsonl(common.path_for("chunks_file"))
    assert cs, "no chunks - run python -m ingest.chunk"
    return cs


@pytest.fixture(scope="module")
def index_rows() -> list[dict]:
    rows = read_jsonl(common.path_for("embed_index_file"))
    assert rows, "no embed index - run python -m embed.index"
    return rows


@pytest.fixture(scope="module")
def vectors() -> dict[str, np.ndarray]:
    return load_cache()


# ---------------------------------------------------------------------------
# The lead-window text
# ---------------------------------------------------------------------------


def test_embed_text_is_the_prefix_plus_the_lead(chunks, cfg):
    lead = int(cfg["embed_lead_tokens"])
    for c in chunks[:20]:
        text = build_embed_text(c, cfg)
        assert c["scheme_name"] in text, \
            f"{c['chunk_id']} lost its scheme name - the prefix is what stops " \
            f"'min. for sip: 100' being identical across all five schemes"
        assert common.count_tokens(text) <= lead


def test_embed_text_respects_the_lead_budget(chunks, cfg):
    """A 325-token prose chunk must embed ~200 tokens, not all 325.

    If this regresses, the vector silently starts describing more than the
    model's 256-token window can hold and the tail is dropped by the encoder
    instead of by us, where we cannot see it happening.
    """
    lead = int(cfg["embed_lead_tokens"])
    long_chunks = [c for c in chunks if c["token_count"] > lead]
    assert long_chunks, "no chunk exceeds the lead budget, so this proves nothing"
    for c in long_chunks:
        embedded = build_embed_text(c, cfg)
        assert common.count_tokens(embedded) <= lead, \
            f"{c['chunk_id']} embedded {common.count_tokens(embedded)} tokens, over {lead}"


def test_embed_text_never_returns_something_empty(chunks, cfg):
    for c in chunks:
        assert build_embed_text(c, cfg).strip(), f"{c['chunk_id']} embeds to nothing"


def test_embed_text_survives_a_literal_brace_in_a_heading(cfg):
    """A page heading with a brace must not crash the whole run."""
    c = {"scheme_name": "HDFC Large Cap", "category": "Large Cap",
         "section_heading": "Fees {2024-25}", "text": "expense ratio: 1.03%"}
    assert "expense ratio" in build_embed_text(c, cfg)


# ---------------------------------------------------------------------------
# The cache key - the bug this stage actually had
# ---------------------------------------------------------------------------


def test_cache_key_must_not_be_the_raw_content_hash(chunks, cfg):
    """`content_hash` addresses the raw text; the vector is of `prefix + text`.

    "Min. for SIP: Rs 100" is byte-identical in four of the five schemes but must
    embed to four DIFFERENT vectors. Caching on content_hash would hand three of
    those four the first scheme's vector, and a question about HDFC Flexi Cap
    would be answered with HDFC Large Cap's row.
    """
    by_raw: dict[str, list[dict]] = {}
    for c in chunks:
        by_raw.setdefault(c["content_hash"], []).append(c)
    collisions = {h: v for h, v in by_raw.items() if len(v) > 1}
    assert collisions, \
        "no content_hash collisions exist, so this test cannot prove the key is right"

    for h, group in collisions.items():
        embed_hashes = {embed_hash(c, cfg) for c in group}
        assert len(embed_hashes) == len(group), (
            f"{len(group)} chunks share raw text {group[0]['text'][:50]!r} but only "
            f"{len(embed_hashes)} distinct embed hashes - they will share a vector "
            f"and become indistinguishable to dense retrieval"
        )


def test_embed_index_has_one_row_per_chunk(chunks, index_rows):
    assert len(index_rows) == len(chunks)
    assert {r["chunk_id"] for r in index_rows} == {c["chunk_id"] for c in chunks}


def test_embed_index_rows_point_at_real_files(index_rows):
    d = Path(common.path_for("embeddings_dir"))
    for r in index_rows:
        assert (d / r["vector_file"]).exists(), f"missing vector file for {r['chunk_id']}"
        assert r["dimension"] == 384, f"{r['chunk_id']} recorded dim {r['dimension']}"


def test_chunks_carry_no_vectors_inline(chunks):
    """FR-3: vectors must not bloat chunks.jsonl; Stage 4 joins on the index."""
    banned = {"embedding", "vector", "dense_vector", "embeddings"}
    for c in chunks:
        assert not (banned & set(c)), f"{c['chunk_id']} carries vector data inline"


# ---------------------------------------------------------------------------
# Vector properties
# ---------------------------------------------------------------------------


def test_every_chunk_has_a_vector(chunks, vectors, cfg):
    missing = [c["chunk_id"] for c in chunks if embed_hash(c, cfg) not in vectors]
    assert not missing, f"{len(missing)} chunk(s) have no vector: {missing[:3]}"


def test_vectors_are_384_dim(vectors):
    for h, v in vectors.items():
        assert v.shape == (384,), f"{h} has shape {v.shape}, expected (384,)"


def test_vectors_are_unit_norm(vectors):
    for h, v in vectors.items():
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3, \
            f"{h} has norm {float(np.linalg.norm(v)):.4f}; cosine maths in Stage 5 " \
            f"assumes unit vectors"


def test_vectors_are_finite(vectors):
    for h, v in vectors.items():
        assert np.isfinite(v).all(), f"{h} contains NaN or inf"


# ---------------------------------------------------------------------------
# The cache contract
# ---------------------------------------------------------------------------


def test_second_run_computes_nothing(tmp_path, model):
    """Re-running must cost nothing. This is the whole point of the .npy cache."""
    sample = read_jsonl(common.path_for("chunks_file"))[:6]
    cache_dir = tmp_path / "emb"

    _v1, first = embed_chunks(sample, model=model, cache_dir=cache_dir, show_progress=False)
    assert first.computed == 6 and first.cached == 0

    _v2, second = embed_chunks(sample, model=model, cache_dir=cache_dir, show_progress=False)
    assert second.computed == 0, f"re-embed computed {second.computed}, expected 0"
    assert second.cached == 6
    assert second.cache_hit_pct == 100.0


def test_force_re_embeds_everything(tmp_path, model):
    sample = read_jsonl(common.path_for("chunks_file"))[:4]
    cache_dir = tmp_path / "emb"
    embed_chunks(sample, model=model, cache_dir=cache_dir, show_progress=False)
    _v, forced = embed_chunks(sample, model=model, cache_dir=cache_dir,
                              force=True, show_progress=False)
    assert forced.computed == 4 and forced.cached == 0


def test_a_corrupt_cache_file_is_discarded_not_trusted(tmp_path):
    """A half-written .npy must be dropped, not returned as a valid vector."""
    cache_dir = tmp_path / "emb"
    cache_dir.mkdir()
    (cache_dir / "deadbeef.npy").write_bytes(b"not a numpy file at all")
    assert load_cache(cache_dir) == {}
    assert not (cache_dir / "deadbeef.npy").exists(), \
        "the corrupt file was left in place and will be retried forever"


def test_a_wrong_shaped_cache_file_is_discarded(tmp_path):
    cache_dir = tmp_path / "emb"
    cache_dir.mkdir()
    np.save(cache_dir / "wrong.npy", np.zeros((2, 384), dtype="float32"))
    assert load_cache(cache_dir) == {}


def test_embeddings_are_deterministic(tmp_path, model):
    """Same text twice must give the same vector, or the cache is meaningless."""
    sample = read_jsonl(common.path_for("chunks_file"))[:5]
    a, _ = embed_chunks(sample, model=model, cache_dir=tmp_path / "a", show_progress=False)
    b, _ = embed_chunks(sample, model=model, cache_dir=tmp_path / "b", show_progress=False)
    cfg = common.load_config()["embedding"]
    for c in sample:
        h = embed_hash(c, cfg)
        assert np.allclose(a[h], b[h], atol=1e-6), f"{c['chunk_id']} embedded differently"


def test_embedding_works_with_no_network(tmp_path, model):
    """A fully-cached model must load offline.

    Otherwise a flaky connection turns a stage whose vectors are all already on
    disk into a hard failure, and the class demo dies in front of the class.
    """
    from embed.index import model_is_cached

    name = common.load_config()["embedding"]["model_name"]
    assert model_is_cached(name), \
        f"{name} is not in the Hugging Face cache, so the offline path is untested"
    reloaded = load_model(name, local_files_only=True)   # must not touch the network
    dim = (reloaded.get_embedding_dimension() if hasattr(reloaded, "get_embedding_dimension")
           else reloaded.get_sentence_embedding_dimension())
    assert dim == 384


# ---------------------------------------------------------------------------
# Semantic sanity - does the model actually distinguish the schemes?
# ---------------------------------------------------------------------------


def _rank(query: str, candidates: list[tuple[str, np.ndarray]],
          model: Any = None) -> list[tuple[str, float]]:
    model = model if model is not None else load_model()
    qv = np.asarray(model.encode([query], normalize_embeddings=True,
                                 show_progress_bar=False)[0], dtype="float32")
    scored = [(cid, float(np.dot(qv, vec))) for cid, vec in candidates]
    return sorted(scored, key=lambda kv: -kv[1])


@pytest.mark.parametrize("slug", [
    "hdfc-large-cap-fund-direct-growth",
    "hdfc-equity-fund-direct-growth",
    "hdfc-small-cap-fund-direct-growth",
    "hdfc-balanced-advantage-fund-direct-growth",
])
def _page_fund_name(chunk: dict) -> str:
    """The fund name as the corpus itself carries it.

    Read from the chunk's `scheme_name` rather than parsed out of `page_title`.
    The AMC titles its scheme pages with the whole SEO string - "HDFC Flexi Cap
    Fund - Direct Plan | NAV, Returns & SIP | HDFC Mutual Fund" - and splitting
    on " - " returned the entire title, which is not a fund name and made the
    query read as keyword soup. `scheme_name` is the name the chunk is indexed
    and answered under, so it is the name a user actually types.
    """
    return chunk["scheme_name"].strip()


@pytest.mark.parametrize("slug", [
    "hdfc-large-cap-fund-direct-growth",
    "hdfc-equity-fund-direct-growth",
    "hdfc-small-cap-fund-direct-growth",
    "hdfc-balanced-advantage-fund-direct-growth",
    "hdfc-elss-tax-saver-fund-direct-plan-growth",
])
def test_expense_ratio_query_finds_its_own_scheme(chunks, vectors, cfg, slug, model):
    """The acceptance test that justifies the whole prefix.

    The AMC's fact card states the expense ratio as a bare "TER: 0.78", so two
    of the five chunks are byte-identical and the third differs only in a
    digit. Without the scheme prefix in the embedded text, a query naming a
    scheme can retrieve the wrong row and cite the wrong fund.
    """
    cands = [(c["chunk_id"], vectors[embed_hash(c, cfg)]) for c in chunks
             if c["fact_key"] == "expense_ratio"]
    # Every scheme states its own expense ratio exactly once.
    by_slug: dict[str, int] = {}
    for c in chunks:
        if c["fact_key"] == "expense_ratio":
            by_slug[c["scheme_slug"]] = by_slug.get(c["scheme_slug"], 0) + 1
    assert set(by_slug.values()) == {1}, \
        f"expected exactly 1 expense-ratio chunk per scheme, got {by_slug}"
    assert len(by_slug) == 5, f"expected 5 schemes, got {sorted(by_slug)}"

    mine = [c for c in chunks if c["scheme_slug"] == slug and c["fact_key"] == "expense_ratio"]
    query = f"expense ratio of {_page_fund_name(mine[0])}"
    ranked = _rank(query, cands, model)
    by_id = {c["chunk_id"]: c for c in chunks}

    best_mine = max(s for cid, s in ranked if by_id[cid]["scheme_slug"] == slug)
    best_rival = max(s for cid, s in ranked if by_id[cid]["scheme_slug"] != slug)
    scores = ", ".join(f"{by_id[cid]['scheme_slug'][:26]}={s:.4f}" for cid, s in ranked)
    assert best_mine > best_rival, (
        f"query {query!r}\n  the right scheme scored {best_mine:.4f} but a rival "
        f"scored {best_rival:.4f}\n  all scores: {scores}"
    )


def test_dense_search_blurs_names_that_only_exist_in_metadata(chunks, vectors, cfg, model):
    """A DOCUMENTED LIMITATION, asserted so it cannot be forgotten.

    HDFC's Flexi Cap page is titled "HDFC Flexi Cap Direct Plan Growth" but its
    canonical URL is hdfc-equity-fund-direct-growth, so people ask about it
    under both names. The alias is in the chunk's embed prefix, but the chunk's
    BODY says only "Flexi Cap", and within a 200-token lead the body wins. A
    query built from the URL name therefore ties across all five schemes'
    expense-ratio FAQs (margin < 0.01) instead of separating them.

    This is not fixed here, and should not be: it is the reason Stage 5 fuses
    BM25 with dense retrieval. "Equity Fund" appears literally in the chunk
    metadata, so lexical matching separates it immediately. If a future change
    DOES make dense-only work here, this test fails and the hybrid can be
    re-evaluated - which is the point of writing the limitation down.
    """
    equity = [c for c in chunks
              if c["scheme_slug"] == "hdfc-equity-fund-direct-growth"
              and c["fact_key"] == "expense_ratio"]
    assert equity, "the Flexi Cap scheme is missing from the corpus"
    assert equity[0]["scheme_aliases"], "the alias was dropped somewhere in Stage 1/2"

    cands = [(c["chunk_id"], vectors[embed_hash(c, cfg)]) for c in chunks
             if c["fact_key"] == "expense_ratio"]
    ranked = _rank("expense ratio of HDFC Equity Fund - Direct Growth", cands, model)
    by_id = {c["chunk_id"]: c for c in chunks}
    best_mine = max(s for cid, s in ranked
                    if by_id[cid]["scheme_slug"] == "hdfc-equity-fund-direct-growth")
    best_rival = max(s for cid, s in ranked
                     if by_id[cid]["scheme_slug"] != "hdfc-equity-fund-direct-growth")

    assert best_mine <= best_rival, (
        "dense-only retrieval now separates the URL-slug name on its own.\n"
        "Good - but Stage 5's BM25 fusion was justified by this gap, so re-check "
        "whether the hybrid is still earning its keep, and update this test and "
        "docs/architecture.md together."
    )
    # The page name must work, though - that is the name the corpus uses.
    page_ranked = _rank(f"expense ratio of {_page_fund_name(equity[0])}", cands, model)
    page_best_mine = max(s for cid, s in page_ranked
                         if by_id[cid]["scheme_slug"] == "hdfc-equity-fund-direct-growth")
    page_best_rival = max(s for cid, s in page_ranked
                          if by_id[cid]["scheme_slug"] != "hdfc-equity-fund-direct-growth")
    assert page_best_mine > page_best_rival, (
        f"the page name must work: {_page_fund_name(equity[0])!r} scored "
        f"{page_best_mine:.4f} against a rival's {page_best_rival:.4f}"
    )


def test_lock_in_query_finds_the_elss_chunk(chunks, vectors, cfg, model):
    lock_chunks = [c for c in chunks if c["fact_key"] == "lock_in_period"]
    assert lock_chunks, "no lock-in chunk - the ELSS pill was lost in Stage 1/2"
    by_id = {c["chunk_id"]: c for c in chunks}
    ranked = _rank("lock-in period", [(c["chunk_id"], vectors[embed_hash(c, cfg)])
                                      for c in lock_chunks], model)
    assert by_id[ranked[0][0]]["scheme_slug"] == \
        "hdfc-elss-tax-saver-fund-direct-plan-growth"


def test_aliases_reach_the_chunks(chunks):
    """The Flexi Cap alias must survive config -> clean -> chunk.

    If it is dropped anywhere along that chain, the URL-slug name becomes
    unsearchable and the documented Stage 5 mitigation quietly stops applying.
    """
    flexi = [c for c in chunks if c["scheme_slug"] == "hdfc-equity-fund-direct-growth"]
    assert flexi, "the Flexi Cap scheme is missing from the corpus"
    for c in flexi:
        assert c["scheme_aliases"], f"{c['chunk_id']} lost its aliases"
        assert any("Flexi Cap" in a for a in c["scheme_aliases"]), \
            f"{c['chunk_id']} aliases do not mention the page's own name: {c['scheme_aliases']}"
    for c in chunks:
        if c["scheme_slug"] != "hdfc-equity-fund-direct-growth":
            assert not c["scheme_aliases"], \
                f"{c['chunk_id']} invented aliases for a scheme that has none"


def test_two_schemes_with_the_same_value_stay_distinguishable(chunks, vectors, cfg):
    """Balanced Advantage and Small Cap both publish TER 0.78.

    The AMC's fact card states it as a bare "TER: 0.78", so without the scheme
    prefix these two chunks would be byte-identical and a user asking about one
    would be shown the other's page as the source.
    """
    same = [c for c in chunks if c["text"] == "TER: 0.78"]
    assert len(same) == 2, \
        f"expected 2 chunks stating TER 0.78, found {[c['text'] for c in same]}"
    hs = {embed_hash(c, cfg) for c in same}
    assert len(hs) == 2, "identical raw text produced one shared vector"
    v1, v2 = vectors[embed_hash(same[0], cfg)], vectors[embed_hash(same[1], cfg)]
    assert float(np.dot(v1, v2)) < 0.99, (
        "the two schemes' vectors are near-identical, so the prefix is not doing "
        "its job and a citation could point at the wrong fund"
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_embed_stats_file_is_complete():
    s = read_json(common.path_for("embed_stats_file"), None)
    assert s, "no embed_stats.json - run python -m embed.index"
    assert s["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert s["dimension"] == 384
    assert s["max_seq_length"] == 256, \
        "the 256-token window is the justification for the lead-window trick; " \
        "record it so the claim is checkable"
    assert s["failed"] == 0
    assert abs(s["norm_mean"] - 1.0) < 1e-3
    assert s["embed_lead_tokens"] == 200


def test_no_chunk_failed_to_embed():
    rows = read_jsonl(common.path_for("embed_index_file"))
    assert all(r["norm"] is not None for r in rows), \
        "some chunk has no vector - Stage 4 would silently skip it"
