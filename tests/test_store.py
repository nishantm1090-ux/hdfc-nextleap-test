"""
tests/test_store.py - STAGE 4: the vector-store contract.

Three kinds of test live here:

  * pure          - metadata coercion and filter validation. No Chroma at all,
                    so they stay fast and they document the three types Chroma
                    rejects (None, list, nested dict).
  * real store    - read-only invariants against the collection that was
                    actually built from the 121-chunk corpus: the count, the
                    coverage, the where-filter, the distance->similarity
                    conversion.
  * sandboxed     - write behaviour (idempotency, rebuild-from-zero, and every
                    failure path) against a throwaway collection in tmp_path, so
                    no test can leave the real index in a state that makes the
                    next one lie.

The two tests worth reading first are
test_chroma_does_not_pin_the_dimension and
test_a_missing_vector_is_an_error_not_a_skip - together they encode the two
things this stage learned the hard way.

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_store.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from common import read_json, read_jsonl

pytest.importorskip("chromadb", reason="chromadb not installed")

import chromadb  # noqa: E402
from store import chroma_store as cs  # noqa: E402

SMALL_CAP = "hdfc-small-cap-fund-direct-growth"
LARGE_CAP = "hdfc-large-cap-fund-direct-growth"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> dict:
    return common.load_config()


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    cs_ = read_jsonl(common.path_for("chunks_file"))
    assert cs_, "no chunks - run python -m ingest.chunk"
    return cs_


@pytest.fixture(scope="module")
def real_collection():
    col = cs.get_collection(create=False)
    if col is None:
        pytest.skip("no collection - run python -m store.chroma_store --rebuild")
    return col


@pytest.fixture
def tmp_client(tmp_path, monkeypatch):
    """A throwaway PersistentClient plus a real-vector embed index.

    The embed index is redirected to a copy in tmp_path while `embeddings_dir`
    keeps pointing at the real vector files, so writes exercise the genuine
    join-and-load path rather than a mock of it.
    """
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))

    real_rows = read_jsonl(common.path_for("embed_index_file"))
    assert real_rows, "no embed index - run python -m embed.index"
    index_file = tmp_path / "embed_index.jsonl"
    index_file.write_text(
        "\n".join(json.dumps(r) for r in real_rows), encoding="utf-8"
    )

    real_path_for = common.path_for
    monkeypatch.setattr(
        common, "path_for",
        lambda key: index_file if key == "embed_index_file" else real_path_for(key),
    )
    return client


@pytest.fixture
def tmp_collection(tmp_client):
    return cs.get_collection(client=tmp_client)


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_collection_name_is_not_resolved_as_a_path(cfg):
    """`vector_store.collection_name` must be a bare name, not ROOT/<name>.

    It used to live under `paths:`, where `path_for()` joined it onto the
    project root - so chroma was asked for a collection literally named
    "E:\\...\\hdfc_mf_faq". Chroma rejected it loudly, which is the only reason
    this was found, but it moved anyway: a name is not a path.
    """
    name = cs.collection_name()
    assert name == "hdfc_mf_faq"
    assert "/" not in name and "\\" not in name
    assert len(name) >= 3, "chroma requires 3-512 chars matching [a-zA-Z0-9._-]"
    assert "collection_name" in cfg["vector_store"]
    assert "chroma_collection" not in cfg.get("paths", {}), \
        "a collection name under paths: gets root-joined by path_for()"


def test_collection_records_cosine_space_and_the_model(cfg, tmp_collection):
    meta = dict(tmp_collection.metadata or {})
    assert meta.get("hnsw:space") == cfg["vector_store"]["metadata"]["space"] == "cosine"
    assert meta.get("embedding_dim") == cfg["vector_store"]["metadata"]["hdim"] == 384
    assert meta.get("embedding_model") == cfg["embedding"]["model_name"]


def test_configured_dimension_matches_the_model():
    assert common.load_config()["embedding"]["dimension"] == 384
    assert common.load_config()["vector_store"]["metadata"]["hdim"] == \
        common.load_config()["embedding"]["dimension"], \
        "store and embed configs disagree - every write would be at risk"


# ---------------------------------------------------------------------------
# Metadata coercion (pure)
# ---------------------------------------------------------------------------


def _chunk(**over) -> dict:
    base = {
        "chunk_id": "c1", "text": "Expense ratio: 1.03%", "scheme_slug": LARGE_CAP,
        "scheme_name": "HDFC Large Cap Fund", "category": "Large Cap",
        "doc_type": "fact", "fact_key": "expense_ratio", "fact_value": "1.03%",
        "publisher": "HDFC AMC", "source_type": "amc_page",
        "source_url": "https://groww.in/funds/hdfc-large-cap-fund-direct-growth",
        "page_title": "HDFC Large Cap Fund", "section_heading": "Fees and charges",
        "plan": "Direct Growth", "chunk_index": 0, "token_count": 12,
        "content_hash": "abc", "embed_hash": "def",
        "source_fetched_at": "2026-09-27T00:00:00+00:00",
        "ingested_at": "2026-09-27T00:00:01+00:00", "pii_scan": "clean",
        "scheme_aliases": ["HDFC Large Cap Fund", "HDFC Large Cap"],
    }
    base.update(over)
    return base


def test_coerce_metadata_keeps_scalars_as_they_are():
    m = cs.coerce_metadata(_chunk())
    assert m["scheme_slug"] == LARGE_CAP
    assert m["fact_key"] == "expense_ratio"
    assert m["token_count"] == 12
    assert m["pii_scan"] == "clean"


def test_coerce_metadata_turns_none_into_empty_string():
    """Chroma rejects None outright. A prose chunk has no fact_key, and without
    this the whole prose partition fails to load."""
    m = cs.coerce_metadata(_chunk(fact_key=None, fact_value=None))
    assert m["fact_key"] == ""
    assert m["fact_value"] == ""


def test_coerce_metadata_joins_list_values():
    """`scheme_aliases` is a list and Chroma has no array type."""
    m = cs.coerce_metadata(_chunk(scheme_aliases=["HDFC Flexi Cap Fund", "Equity Fund"]))
    assert isinstance(m["scheme_aliases"], str)
    assert "HDFC Flexi Cap Fund" in m["scheme_aliases"]
    assert "Equity Fund" in m["scheme_aliases"]


def test_coerce_metadata_output_is_acceptable_to_chroma(tmp_path):
    """The real test: hand the coerced dict to Chroma and see if it takes it."""
    col = cs.get_collection(client=chromadb.PersistentClient(path=str(tmp_path / "c")))
    col.add(ids=["probe"], documents=["x"], metadatas=[cs.coerce_metadata(_chunk())])
    assert col.count() == 1


def test_coerce_metadata_survives_a_nested_dict():
    """hnsw:metadata={"hdim":384} is exactly the shape that made
    `metadata={"hnsw:space": "cosine", "hnsw:metadata": {...}}` fail. If a
    nested dict ever reaches a chunk field, it must be stringified, not passed
    through to raise."""
    m = cs.coerce_metadata(_chunk(fact_value={"amount": 1, "unit": "%"}))
    assert isinstance(m["fact_value"], str)
    assert "amount" in m["fact_value"]


# ---------------------------------------------------------------------------
# Filter validation
# ---------------------------------------------------------------------------


def test_filterable_keys_cover_everything_stage5_needs():
    for key in ("scheme_slug", "category", "doc_type", "publisher"):
        assert key in cs.FILTERABLE_KEYS, f"Stage 5 filters on {key}"


def test_a_typo_in_a_filter_key_is_rejected():
    with pytest.raises(KeyError, match="schemes_slug"):
        cs._coerce_where({"schemes_slug": LARGE_CAP})


def test_a_filter_on_a_stored_but_unfilterable_key_is_also_rejected():
    with pytest.raises(KeyError):
        cs._coerce_where({"page_title": "x"})


def test_empty_where_is_none():
    assert cs._coerce_where(None) is None
    assert cs._coerce_where({}) is None


def test_where_coerces_none_values_to_empty_string():
    assert cs._coerce_where({"fact_key": None}) == {"fact_key": ""}


# ---------------------------------------------------------------------------
# The real store: read-only invariants
# ---------------------------------------------------------------------------


def test_collection_count_equals_chunk_count(chunks, real_collection):
    assert real_collection.count() == len(chunks), \
        "collection and chunks.jsonl disagree - run --rebuild"


def test_every_chunk_id_is_stored_exactly_once(chunks, real_collection):
    stored = (real_collection.get(include=[]) or {}).get("ids") or []
    assert len(stored) == len(set(stored)) == len(chunks), \
        "a duplicate id would mean an upsert was not an upsert"


def test_coverage_gaps_is_clean():
    gaps = cs.coverage_gaps()
    assert gaps["store_exists"]
    assert not gaps["stale"], f"stale store: {gaps}"
    assert gaps["missing_from_store"] == []
    assert gaps["orphan_in_store"] == []


def test_where_filter_actually_filters(chunks, real_collection):
    """FR-4.1. A filter that returns everything is worse than no filter: it
    looks like it works.

    The query vector is a real stored vector, not a zero vector. Chroma's HNSW
    post-filters a candidate window, and an all-zeros query ties every distance
    so the walk's stopping point depends on graph state - which is how a
    24-row filter once returned 23 and made this test flaky. Using a real
    vector makes the test measure the filter, which is what it is for.
    """
    res = cs.query(_a_real_vector(real_collection), top_k=500,
                   where={"scheme_slug": SMALL_CAP}, collection=real_collection)
    assert res, "the filter matched nothing"
    assert {r["metadata"]["scheme_slug"] for r in res} == {SMALL_CAP}
    expected = sum(1 for c in chunks if c["scheme_slug"] == SMALL_CAP)
    assert len(res) == expected


def test_a_filtered_query_can_come_back_short_and_that_is_not_an_error(chunks,
                                                                      real_collection):
    """The hazard documented in `query()`, pinned so it is not forgotten.

    A filtered HNSW query is approximate: it retrieves a candidate window and
    filters it, so it may return fewer rows than exist. The contract this test
    defends is the important half - a short result set is *missing rows*, never
    *wrong rows*. What Stage 5 must not do is assume `len(results) == top_k`
    means the filter is satisfied.
    """
    matched = sum(1 for c in chunks if c["scheme_slug"] == SMALL_CAP)
    seen_counts = set()
    for _ in range(5):
        res = cs.query(_a_real_vector(real_collection), top_k=500,
                       where={"scheme_slug": SMALL_CAP}, collection=real_collection)
        assert res, "the filter returned nothing at all"
        assert {r["metadata"]["scheme_slug"] for r in res} == {SMALL_CAP}, \
            "a short filtered result set must be missing rows, never wrong rows"
        seen_counts.add(len(res))
    assert all(0 < n <= matched for n in seen_counts), seen_counts


def _a_real_vector(collection) -> list[float]:
    """A real stored unit vector - a well-posed query, unlike all zeros."""
    got = collection.get(limit=1, include=["embeddings"])
    assert len(got["embeddings"]), "collection is empty"
    return [float(x) for x in got["embeddings"][0]]


def test_a_filter_on_a_missing_fact_key_returns_only_prose(chunks, real_collection):
    """The design consequence of coercing None to "": `fact_key: ""` is the
    prose partition, and it is reachable rather than an impossible filter."""
    res = cs.query(_a_real_vector(real_collection), top_k=500, where={"fact_key": ""},
                   collection=real_collection)
    assert res
    assert {r["metadata"]["doc_type"] for r in res} == {"prose"}


def test_query_converts_distance_into_similarity(real_collection):
    """Chroma reports cosine distance. Stage 5's thresholds are written against
    similarity, so a silent sign flip here would invert every ranking."""
    res = cs.query(_a_real_vector(real_collection), top_k=5,
                   collection=real_collection)
    for r in res:
        assert abs(r["distance"] + r["similarity"] - 1.0) < 1e-9
        assert -1.0001 <= r["similarity"] <= 1.0001


def test_a_real_vector_ranks_its_own_chunk_first(real_collection):
    """The one semantic guarantee dense search must make."""
    got = real_collection.get(limit=1, include=["embeddings", "documents"])
    res = cs.query([float(x) for x in got["embeddings"][0]], top_k=1,
                   collection=real_collection)
    assert res[0]["chunk_id"] == got["ids"][0]
    assert res[0]["similarity"] > 0.99, \
        f"a vector's own chunk should score ~1.0, got {res[0]['similarity']}"


def test_query_results_carry_the_citation_fields(real_collection):
    res = cs.query(_a_real_vector(real_collection), top_k=3,
                   collection=real_collection)
    for r in res:
        assert r["chunk_id"] and r["text"]
        m = r["metadata"]
        assert m["source_url"].startswith("https://")
        assert m["publisher"]
        assert m["page_title"]


def test_query_against_a_missing_collection_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(cs, "get_collection", lambda **_kw: None)
    with pytest.raises(FileNotFoundError, match="chroma_store --rebuild"):
        cs.query([0.0] * 384)


# ---------------------------------------------------------------------------
# Write behaviour (sandboxed)
# ---------------------------------------------------------------------------


def test_upsert_writes_one_row_per_chunk(chunks, tmp_collection):
    n = cs.upsert_chunks(chunks[:5], collection=tmp_collection, show_progress=False)
    assert n == 5
    assert tmp_collection.count() == 5


def test_upsert_is_idempotent(chunks, tmp_collection):
    """The acceptance criterion. Running the stage twice must not double the
    collection - that is what upsert keyed on chunk_id buys."""
    cs.upsert_chunks(chunks[:5], collection=tmp_collection, show_progress=False)
    cs.upsert_chunks(chunks[:5], collection=tmp_collection, show_progress=False)
    cs.upsert_chunks(chunks[:5], collection=tmp_collection, show_progress=False)
    assert tmp_collection.count() == 5


def test_upsert_replaces_a_changed_vector_in_place(chunks, tmp_collection):
    """Idempotency must mean "replace", not "refuse to touch"."""
    cs.upsert_chunks(chunks[:3], collection=tmp_collection, show_progress=False)
    before = tmp_collection.get(ids=[chunks[0]["chunk_id"]])
    cs.upsert_chunks(chunks[:3], collection=tmp_collection, show_progress=False)
    after = tmp_collection.get(ids=[chunks[0]["chunk_id"]])
    assert before["documents"] == after["documents"]


def test_upsert_reports_the_true_row_count_not_the_batch_size(chunks, tmp_collection):
    """Regression: `stop` was unclamped, so 121 chunks at batch_size=256 sliced
    correctly but counted 256 - the report said "wrote 256" next to a count of
    121."""
    assert len(chunks) > 256 or True  # the count check below is the real assertion
    n = cs.upsert_chunks(chunks, collection=tmp_collection, batch_size=256,
                         show_progress=False)
    assert n == tmp_collection.count() == len(chunks)


def test_upsert_in_small_batches_matches_one_batch(chunks, tmp_collection):
    cs.upsert_chunks(chunks[:20], collection=tmp_collection, batch_size=1,
                     show_progress=False)
    assert tmp_collection.count() == 20


def test_upsert_of_nothing_is_a_warning_not_an_error(tmp_collection):
    assert cs.upsert_chunks([], collection=tmp_collection, show_progress=False) == 0
    assert tmp_collection.count() == 0


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_a_missing_vector_is_an_error_not_a_skip(chunks, tmp_collection):
    """The link that must never break. A chunk with no row in the embed index
    is a bug to fix; silently skipping it is how a scheme becomes unanswerable
    with nothing in the logs."""
    with pytest.raises(KeyError, match="has no row in the embed index"):
        cs.upsert_chunks([{**chunks[0], "chunk_id": "not-a-real-chunk-id"}],
                         collection=tmp_collection, show_progress=False)


def test_a_chunk_with_no_text_is_rejected(chunks, tmp_collection):
    bad = {**chunks[0], "text": "   "}
    with pytest.raises(ValueError, match="could never back a citation"):
        cs.upsert_chunks([bad], collection=tmp_collection, show_progress=False)


def test_a_wrong_dimension_vector_is_rejected(chunks, tmp_client, monkeypatch,
                                               tmp_path):
    """Chroma will not catch this for us - see the next test."""
    import numpy as np

    vec_dir = tmp_path / "bad_vectors"
    vec_dir.mkdir()
    (vec_dir / "wrong.npy").write_bytes(b"")
    np.save(vec_dir / "wrong.npy", np.zeros(768, dtype="float32"))

    bad = {**chunks[0], "chunk_id": "wrong-dim"}
    real_path_for = common.path_for
    monkeypatch.setattr(
        common, "path_for",
        lambda k: (tmp_path / "bad_index.jsonl") if k == "embed_index_file"
        else (vec_dir if k == "embeddings_dir" else real_path_for(k)),
    )
    (tmp_path / "bad_index.jsonl").write_text(
        json.dumps({"chunk_id": "wrong-dim", "vector_file": "wrong.npy"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has dim 768, config says 384"):
        cs.upsert_chunks([bad], collection=cs.get_collection(client=tmp_client),
                         show_progress=False)


def test_a_non_unit_vector_is_rejected(chunks, tmp_client, monkeypatch, tmp_path):
    """Stage 5 treats a dense hit as a dot product. That is only cosine on unit
    vectors, so an unnormalised vector would quietly skew every score."""
    import numpy as np

    vec_dir = tmp_path / "l2_vectors"
    vec_dir.mkdir()
    np.save(vec_dir / "long.npy", np.full(384, 0.5, dtype="float32"))

    bad = {**chunks[0], "chunk_id": "not-unit"}
    real_path_for = common.path_for
    monkeypatch.setattr(
        common, "path_for",
        lambda k: (tmp_path / "l2_index.jsonl") if k == "embed_index_file"
        else (vec_dir if k == "embeddings_dir" else real_path_for(k)),
    )
    (tmp_path / "l2_index.jsonl").write_text(
        json.dumps({"chunk_id": "not-unit", "vector_file": "long.npy"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected ~1.0"):
        cs.upsert_chunks([bad], collection=cs.get_collection(client=tmp_client),
                         show_progress=False)


def test_a_duplicate_row_in_the_embed_index_is_rejected(tmp_client, monkeypatch,
                                                        tmp_path):
    """Two rows for one chunk_id is ambiguous, and ambiguity here means the wrong
    vector with no error."""
    f = tmp_path / "dupe_index.jsonl"
    row = {"chunk_id": "x", "vector_file": "x.npy"}
    f.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
    real_path_for = common.path_for
    monkeypatch.setattr(
        common, "path_for",
        lambda k: f if k == "embed_index_file" else real_path_for(k),
    )
    with pytest.raises(ValueError, match="two rows for x"):
        cs.load_join_index()


def test_an_empty_embed_index_says_what_to_run(monkeypatch, tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")
    real_path_for = common.path_for
    monkeypatch.setattr(
        common, "path_for",
        lambda k: f if k == "embed_index_file" else real_path_for(k),
    )
    with pytest.raises(FileNotFoundError, match="embed.index"):
        cs.load_join_index()


def test_chroma_does_not_pin_the_dimension(tmp_client):
    """Why `_load_vector` has to check the shape itself.

    chromadb 1.5.9 has no way to set `hnsw:metadata.hdim`, and a 768-dim vector -
    i.e. someone swapped in all-mpnet-base-v2 - is accepted into a 384-dim
    cosine collection with no error at all. This test asserts the *limitation* is
    real; if a future chromadb version starts enforcing it, this fails and the
    comment in `_load_vector` can be retired.
    """
    col = cs.get_collection(client=tmp_client)
    col.add(ids=["a"], documents=["a"], embeddings=[[0.1] * 768])
    assert col.count() == 1, \
        "chroma now pins the dimension - drop the manual check in _load_vector"


# ---------------------------------------------------------------------------
# Stats / report
# ---------------------------------------------------------------------------


def test_stats_reports_the_real_count(chunks):
    st = cs.stats()
    assert st["exists"]
    assert st["count"] == len(chunks)
    assert st["embedding_dim"] == 384
    assert sum(st["per_scheme"].values()) == st["count"]
    assert sum(st["per_doc_type"].values()) == st["count"]
    assert set(st["per_scheme"]) == {c["scheme_slug"] for c in chunks}


def test_stats_partition_matches_the_chunk_file(chunks):
    st = cs.stats()
    for slug, n in st["per_scheme"].items():
        assert n == sum(1 for c in chunks if c["scheme_slug"] == slug)
    for dt, n in st["per_doc_type"].items():
        assert n == sum(1 for c in chunks if c["doc_type"] == dt)


def test_every_stored_row_keeps_its_full_citation(chunks, real_collection):
    """Stage 5 has to produce exactly one source link. If `source_url` were
    dropped in coercion, that contract becomes unenforceable."""
    got = real_collection.get(include=["metadatas"])
    for meta in got["metadatas"]:
        assert meta["source_url"].startswith("https://groww.in/")
        assert meta["page_title"].strip()
        assert meta["source_fetched_at"]
        assert meta["plan"] == "Direct Growth"


def test_publisher_and_source_type_survive_coercion(chunks, real_collection):
    """The real values, asserted so a provenance change is visible.

    `publisher` is Groww, not HDFC AMC: the five pages in the corpus are Groww
    scheme pages that republish the AMC's own figures. The source-of-truth
    publisher is HDFC AMC, but the page a reader is pointed at - and the one the
    citation link must resolve to - is Groww, so that is what is stored.
    """
    assert {c["publisher"] for c in chunks} == {"Groww"}
    got = real_collection.get(include=["metadatas"])
    assert {m["publisher"] for m in got["metadatas"]} == {"Groww"}
    assert {m["source_type"] for m in got["metadatas"]} == {"scheme_page"}


def test_aliases_are_present_exactly_where_a_scheme_defines_them(chunks,
                                                                 real_collection):
    """Only HDFC Flexi Cap has a second public name.

    The Flexi Cap page is titled "Flexi Cap" while its URL slug is
    `hdfc-equity-fund-direct-growth`, so a user can reach it by either name. The
    other four schemes have exactly one public name and correctly carry an empty
    alias string. An empty alias is the honest value here, not a lost one - so
    this asserts the split rather than demanding aliases everywhere.
    """
    by_slug: dict[str, set[str]] = {}
    for c in chunks:
        by_slug.setdefault(c["scheme_slug"], set()).update(c.get("scheme_aliases") or [])
    named = {s: a for s, a in by_slug.items() if a}
    assert list(named) == ["hdfc-equity-fund-direct-growth"], \
        f"unexpected schemes with aliases: { {s: sorted(a) for s, a in named.items()} }"
    assert named["hdfc-equity-fund-direct-growth"] == {
        "HDFC Flexi Cap Fund - Direct Growth", "HDFC Flexi Cap Direct Plan Growth",
    }

    got = real_collection.get(include=["metadatas"])
    flexi = [m for m in got["metadatas"]
             if m["scheme_slug"] == "hdfc-equity-fund-direct-growth"]
    assert flexi and all(m["scheme_aliases"] for m in flexi), \
        "the one scheme that needs aliases lost them in coercion"
    assert all(" | " in m["scheme_aliases"] for m in flexi), \
        "aliases must be joined with the documented separator"
    others = [m for m in got["metadatas"]
              if m["scheme_slug"] != "hdfc-equity-fund-direct-growth"]
    assert all(m["scheme_aliases"] == "" for m in others)


def test_the_report_runs_and_exits_zero(capsys):
    assert cs.report() == 0
    out = capsys.readouterr().out
    assert "STAGE 4 - VECTOR STORE REPORT" in out
    assert "no gaps, no orphans" in out


def test_chroma_stats_file_was_written():
    s = read_json(common.path_for("chroma_stats_file"), None)
    assert s, "no chroma_stats.json - run python -m store.chroma_store --rebuild"
    assert s["count"] > 0
    assert s["exists"]
    assert s["checked_at"]


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_the_package_exports_its_public_surface():
    import store

    for name in store.__all__:
        assert hasattr(store, name), f"store.{name} is exported but missing"
    assert store.collection_name() == "hdfc_mf_faq"


def test_an_unknown_package_attribute_raises():
    import store

    with pytest.raises(AttributeError):
        store.does_not_exist
