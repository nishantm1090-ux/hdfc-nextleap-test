"""
store/chroma_store.py - STAGE 4: VECTOR STORE  (PRD FR-4, architecture.md §4 Stage 4)

Persists the 121 chunk vectors into a ChromaDB collection so Stage 5 can run
dense search AND filter by scheme_slug / category / doc_type / publisher.

Why the join goes through data/embed_index.jsonl
------------------------------------------------
Stage 3 discovered that a chunk's `content_hash` is NOT a usable join key: the
vector is of `prefix + text`, and four of our five schemes share the byte-
identical fact string "min. for sip: 100". The index file records one row per
chunk - `chunk_id -> vector_file` - so the join is explicit and unambiguous. This
module never infers a vector from a hash; it looks the file up by chunk_id and
fails loudly if the row is missing.

Idempotency
-----------
Every write is an UPSERT keyed on `chunk_id`, so running this twice cannot
duplicate a row. `rebuild()` is the only destructive path and it requires
`--rebuild` on the command line.

The one link that must never break
----------------------------------
A chunk in the index with no vector is not "skipped" - it is an ERROR. A silently
short index is how a scheme ends up unanswerable with nothing in the logs to
explain why.

Run:  python -m store.chroma_store            # upsert
      python -m store.chroma_store --rebuild  # wipe + recreate + upsert
      python -m store.chroma_store --stats
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

import common
from common import LOG, read_json, utc_now_iso, write_json

# Chroma's own telemetry prints a JSON blob on import and an emoji banner on
# get_or_create_collection. Neither belongs in a stage report, and the emoji
# crashes on a cp1252 console, so silence it here rather than at the call site.
os_env: dict[str, str] = {}
try:  # pragma: no cover - purely a side effect
    import os

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.impl.posthog.NoopProductTelemetryClient")
    os_env = {k: v for k, v in os.environ.items() if k.startswith("ANONYMIZED")}
except Exception:  # noqa: BLE001  # pragma: no cover
    pass


# The metadata keys Stage 5 filters on. Anything not in this list is not
# queryable, so it must not be relied on by retrieval.
FILTERABLE_KEYS = (
    "scheme_slug", "scheme_name", "category", "doc_type", "fact_key",
    "publisher", "source_type", "plan", "pii_scan",
)

# Everything attached to a row, for the citation and for debugging.
STORED_KEYS = FILTERABLE_KEYS + (
    "text", "source_url", "page_title", "section_heading", "fact_value",
    "chunk_index", "token_count", "content_hash", "embed_hash",
    "source_fetched_at", "ingested_at", "scheme_aliases",
)


# ---------------------------------------------------------------------------
# Metadata coercion
# ---------------------------------------------------------------------------


def coerce_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Flatten a chunk into a metadata dict Chroma will actually accept.

    Chroma allows only str / int / float / bool. Three things in a chunk break
    that, and each breaks differently:

      * ``None``  - a missing `fact_key` on a prose chunk is rejected outright.
        Coerced to "" so a `where={"fact_key": "nav"}` filter stays valid and a
        prose chunk is simply not in that partition.
      * ``list``  - `scheme_aliases` is a list. Chroma has no array type, so it
        is joined with " | ". BM25 tokenisation splits on non-alphanumerics
        anyway, so a separator is not even needed for matching - the join exists
        so a human reading `stats()` can see them.
      * ``bool`` is a subclass of ``int`` in Python, which is fine for Chroma but
        means `isinstance(x, int)` is not a safe way to reject it. The order of
        the isinstance chain below handles it explicitly.
    """
    out: dict[str, str | int | float | bool] = {}
    for key in STORED_KEYS:
        value = chunk.get(key)

        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = int(value)
        elif isinstance(value, (list, tuple, set)):
            out[key] = " | ".join(str(v).strip() for v in value if str(v).strip())
        elif value is None:
            out[key] = ""
        else:
            out[key] = str(value).strip()

    # A filter key that is empty is a landmine: `where={"fact_key": ""}` returns
    # every prose chunk and looks like a working filter. Name it explicitly so a
    # bad filter is visible in the row rather than silent.
    return out


# ---------------------------------------------------------------------------
# Client + collection
# ---------------------------------------------------------------------------


def collection_name() -> str:
    """The Chroma collection name, from `vector_store.collection_name`.

    Deliberately NOT resolved through `path_for`: that joins a `paths.*` value
    onto the project root, and a collection name is not a path. Getting this
    wrong is not silent - chroma rejects the result with a name-validation error
    that mentions the full filesystem path, which is at least a loud mistake.
    """
    return str(common.load_config()["vector_store"]["collection_name"])


def get_client():
    """A persistent Chroma client rooted at ``paths.chroma_dir``."""
    import chromadb  # noqa: PLC0415

    path = common.path_for("chroma_dir")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collection_kwargs() -> dict[str, Any]:
    """Collection creation args, pinned to cosine space at the configured dim.

    chromadb 1.5.9 accepts the legacy `metadata={"hnsw:space": "cosine"}` form but
    has **no way to set `hnsw:metadata.hdim`** - the modern
    `CreateHNSWConfiguration` exposes only space/ef_construction/ef_search/
    max_neighbors/num_threads/resize_factor/sync_threshold/batch_size.

    That matters more than it looks. Verified against 1.5.9: a 768-dim vector
    (i.e. someone swapped in all-mpnet-base-v2) is upserted into a 384-dim cosine
    collection with **no error at all**, and queries then return silently wrong
    rankings. So the dimension is recorded in the collection metadata for
    humans, and enforced in code by `_load_vector`, which is the only place it
    can actually be enforced.
    """
    meta_cfg = common.load_config()["vector_store"].get("metadata", {}) or {}
    return {
        "metadata": {
            "hnsw:space": meta_cfg.get("space", "cosine"),
            # Our own record of the invariant, not something Chroma enforces.
            "embedding_dim": int(meta_cfg.get("hdim", 384)),
            "embedding_model": common.load_config()["embedding"]["model_name"],
        },
    }


def get_collection(*, create: bool = True, client: Any = None):
    """The ``hdfc_mf_faq`` collection, with cosine space and hdim recorded.

    `_load_vector` is what actually pins the dimension; see `_collection_kwargs`.
    """
    client = client or get_client()
    name = collection_name()
    if create:
        return client.get_or_create_collection(name=name, **_collection_kwargs())
    try:
        return client.get_collection(name=name)
    except Exception:  # noqa: BLE001 - chroma raises its own collection-not-found type
        return None


# ---------------------------------------------------------------------------
# The chunk -> vector join
# ---------------------------------------------------------------------------


def load_join_index() -> dict[str, dict[str, Any]]:
    """`chunk_id -> {vector_file, embed_hash, dimension}` from Stage 3.

    Returns a mapping rather than trusting the caller's ordering, because a
    positional zip between chunks.jsonl and the vector directory is exactly the
    kind of thing that is correct until the corpus is re-chunked.
    """
    rows = common.read_jsonl(common.path_for("embed_index_file"))
    if not rows:
        raise FileNotFoundError(
            f"{common.path_for('embed_index_file')} is missing or empty. "
            f"Run: python -m embed.index"
        )
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row.get("chunk_id")
        if not cid:
            raise ValueError(f"embed index row has no chunk_id: {row}")
        if cid in index:
            raise ValueError(f"embed index has two rows for {cid}")
        index[cid] = row
    return index


def _load_vector(row: dict[str, Any], expected_dim: int) -> list[float]:
    import numpy as np  # noqa: PLC0415

    path = Path(common.path_for("embeddings_dir")) / row["vector_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"vector file {path.name} is missing for chunk {row['chunk_id']} - "
            f"run: python -m embed.index"
        )
    vec = np.load(path)
    if vec.ndim != 1:
        raise ValueError(f"{path.name} is {vec.ndim}-D, expected a 1-D vector")
    if vec.shape[0] != expected_dim:
        raise ValueError(
            f"{path.name} has dim {vec.shape[0]}, config says {expected_dim}. "
            f"Either the model changed or a stray vector is in data/embeddings/ "
            f"- re-run with --force after deleting data/embeddings/*.npy"
        )
    return [float(x) for x in vec]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def upsert_chunks(chunks: Sequence[dict[str, Any]], *, batch_size: int = 256,
                  collection: Any = None, show_progress: bool = True) -> int:
    """Upsert `chunks` into the collection, joining each to its vector.

    Returns the number of rows written. Raises on a missing or mismatched
    vector: a chunk that cannot be embedded is a bug to fix, not a row to skip.
    """
    import numpy as np  # noqa: PLC0415

    collection = collection or get_collection()
    expected_dim = int(common.load_config()["embedding"]["dimension"])
    index = load_join_index()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for chunk in chunks:
        cid = chunk["chunk_id"]
        row = index.get(cid)
        if row is None:
            raise KeyError(
                f"chunk {cid} has no row in the embed index. Stage 3 is stale - "
                f"run: python -m embed.index --force"
            )
        if not chunk.get("text", "").strip():
            raise ValueError(f"chunk {cid} has no text; it could never back a citation")
        vec = _load_vector(row, expected_dim)
        norm = float(np.linalg.norm(vec))
        if abs(norm - 1.0) > 1e-2:
            raise ValueError(
                f"chunk {cid}'s vector has norm {norm:.4f}, expected ~1.0. Stage 5 "
                f"computes cosine similarity as a dot product, which is only equal "
                f"to cosine on unit vectors."
            )

        ids.append(cid)
        documents.append(chunk["text"])
        metadatas.append(coerce_metadata(chunk))
        embeddings.append(vec)

    if not ids:
        LOG.warning("no chunks to upsert")
        return 0

    # Batching is done here, not passed to Chroma: `Collection.upsert()` takes no
    # batch_size (only the top-level bulk `Client.upsert()` does), and passing it
    # raises TypeError. With 121 rows this is one batch, but the loop keeps the
    # memory profile flat if the corpus grows, and makes the flush boundaries
    # explicit.
    written = 0
    batch_size = max(1, int(batch_size))
    for start in range(0, len(ids), batch_size):
        # Clamp: an unclamped `stop` over-reports on the final partial batch (121
        # chunks at batch_size=256 would slice correctly but count 256). The
        # collection count stayed right, so the stage report would have
        # disagreed with itself - 121 stored, "wrote 256".
        stop = min(start + batch_size, len(ids))
        collection.upsert(
            ids=ids[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
            embeddings=embeddings[start:stop],
        )
        written = stop
        if show_progress and len(ids) > batch_size:
            LOG.info("  upserted %d/%d chunk(s)", written, len(ids))

    if show_progress:
        LOG.info("  upserted %d chunk(s) into %s", written, collection_name())
    return written


def live_segment_ids() -> set[str]:
    """Segment directory ids the sqlite index still points at.

    Read from the `segments` table rather than inferred from the collection id.
    Those are two different UUIDs and conflating them is how a *live* segment
    looks like garbage - which is the mistake that makes this cleanup dangerous
    if it is ever done carelessly.
    """
    import sqlite3  # noqa: PLC0415

    db = Path(common.path_for("chroma_dir")) / "chroma.sqlite3"
    if not db.exists():
        return set()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id FROM segments").fetchall()
    except sqlite3.Error:
        return set()
    finally:
        con.close()
    return {str(r[0]) for r in rows}


def purge_orphan_segments(*, dry_run: bool = True) -> list[str]:
    """Remove HNSW segment directories that no collection references.

    `chromadb.Client.delete_collection()` drops the row from `segments` but does
    **not** remove the segment's `.bin` files from disk. Measured on this repo:
    after two `--rebuild` runs, `data/chroma/` held two identically-sized HNSW
    segment directories while the `segments` table listed only one - roughly
    168 KB leaked per rebuild, silently, forever.

    `rebuild()` is the only caller that passes `dry_run=False`, and only after it
    has just created a fresh collection. Two guards make this safe:

      * a directory is removed only if its name is a UUID and that UUID is absent
        from the `segments` table, so an unrecognised file is never touched;
      * the live collection's own segment is re-read from the table immediately
        before deleting, so the set cannot be stale.

    Defaults to `dry_run=True` because this is the one function here that removes
    files, and "show me what you would delete" is the right default for that.
    """
    root = Path(common.path_for("chroma_dir"))
    if not root.exists():
        return []

    live = live_segment_ids()
    orphans: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            uuid.UUID(entry.name)
        except ValueError:
            continue  # not a segment directory - leave it strictly alone
        if entry.name not in live:
            orphans.append(entry.name)

    if dry_run:
        return orphans

    import shutil  # noqa: PLC0415

    for name in orphans:
        # Re-read: a rebuild in another process could have adopted the directory.
        if name in live_segment_ids():
            continue
        shutil.rmtree(root / name, ignore_errors=True)
        LOG.info("  removed orphan segment %s", name)
    return orphans


def rebuild(*, show_progress: bool = True) -> int:
    """Wipe the collection and upsert every chunk from scratch.

    This is the only destructive operation, and it is the one the UI's "Rebuild
    index" button calls. A rebuild is a full re-derivation from the corpus, so
    it is safe by construction - the corpus in data/ is the source of truth.
    """
    client = get_client()
    name = collection_name()
    try:
        client.delete_collection(name=name)
        if show_progress:
            LOG.info("  deleted existing collection %s", name)
    except Exception as exc:  # noqa: BLE001
        LOG.info("  nothing to delete (%s: %s)", type(exc).__name__, exc)

    chunks = common.read_jsonl(common.path_for("chunks_file"))
    if not chunks:
        raise FileNotFoundError(
            f"{common.path_for('chunks_file')} is empty. Run: python -m ingest.chunk"
        )
    collection = get_collection(client=client)
    n = upsert_chunks(chunks, collection=collection, show_progress=show_progress)

    # Only now, with a freshly written collection in place, is it safe to drop the
    # segment directories the delete above orphaned.
    if show_progress:
        orphans = purge_orphan_segments(dry_run=False)
        if orphans:
            LOG.info("  purged %d orphan segment dir(s) left by delete_collection",
                     len(orphans))
    return n


def sync(*, show_progress: bool = True) -> int:
    """Upsert without wiping. Cheap, idempotent, and the default operation.

    The only thing it does NOT fix is a chunk that was removed from the corpus -
    upsert never deletes, so a stale row survives. `rebuild()` is the answer to
    "a scheme disappeared"; this is the answer to "a value changed".
    """
    chunks = common.read_jsonl(common.path_for("chunks_file"))
    if not chunks:
        raise FileNotFoundError(
            f"{common.path_for('chunks_file')} is empty. Run: python -m ingest.chunk"
        )
    return upsert_chunks(chunks, show_progress=show_progress)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def query(embedding: Sequence[float], *, top_k: int = 20,
          where: dict[str, Any] | None = None,
          collection: Any = None) -> list[dict[str, Any]]:
    """Dense search. Returns [{chunk_id, text, metadata, similarity}, ...].

    Chroma reports COSINE DISTANCE, which for our unit vectors is exactly
    `1 - cosine_similarity`. It is converted here so that callers deal in
    similarity - a bigger number meaning "more similar" - which is the only
    convention Stage 5's threshold makes sense in.

    A FILTERED query can come back SHORT - fewer than `top_k` rows, with no
    error. Measured on the real collection: `where={"scheme_slug":
    "hdfc-small-cap-fund-direct-growth"}` over 24 matching rows returned 23 on
    one run and 24 on the next twenty. Chroma's HNSW is an approximate index
    and a metadata filter is applied as a post-filter over the candidate window,
    so a walk that stops early loses rows. It is far more visible when the query
    vector is degenerate - an all-zeros vector ties every distance, which is why
    this only showed up in a test.

    Consequences for Stage 5, which is the only caller that filters:

      * never treat `len(results) == top_k` as "the filter found top_k matches".
        Absence of a row means "not retrieved", not "does not exist".
      * over-fetch when filtering - ask for several times what you need and
        truncate locally, rather than trusting n_results to be filled.
      * a scheme filter is a ranking refinement, not an existence check. The
        scheme is known before retrieval (it comes out of the question), so a
        short result set is recoverable; a hard filter on a fact_key is not.
    """
    collection = collection or get_collection(create=False)
    if collection is None:
        raise FileNotFoundError(
            f"collection {collection_name()} does not exist. "
            f"Run: python -m store.chroma_store --rebuild"
        )
    raw = collection.query(
        query_embeddings=[list(map(float, embedding))],
        n_results=int(top_k),
        where=_coerce_where(where),
        include=["documents", "metadatas", "distances"],
    )

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    results: list[dict[str, Any]] = []
    for i, cid in enumerate(ids):
        distance = float(dists[i]) if i < len(dists) else 1.0
        results.append({
            "chunk_id": cid,
            "text": docs[i] if i < len(docs) else "",
            "metadata": dict(metas[i] or {}) if i < len(metas) else {},
            "distance": distance,
            "similarity": 1.0 - distance,
        })
    return results


def _coerce_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate a metadata filter before it reaches Chroma.

    Two mistakes are caught here rather than as a confusing empty result set:
    filtering on a key that was never stored, and comparing a string field
    against a number.
    """
    if not where:
        return None
    for key in where:
        if key not in FILTERABLE_KEYS:
            raise KeyError(
                f"cannot filter on {key!r}. Filterable keys are: {', '.join(FILTERABLE_KEYS)}"
            )
    return {k: ("" if v is None else v) for k, v in where.items()}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def stats() -> dict[str, Any]:
    """Counts and coverage, for the sidebar and for eval/."""
    collection = get_collection(create=False)
    if collection is None:
        return {"exists": False, "collection": collection_name(),
                "count": 0, "path": str(common.path_for("chroma_dir"))}

    count = int(collection.count())

    per_scheme: dict[str, int] = {}
    per_doc_type: dict[str, int] = {}
    per_fact_key: dict[str, int] = {}
    sims: list[float] = []

    if count:
        got = collection.get(include=["metadatas", "documents"])
        for meta in got.get("metadatas") or []:
            meta = meta or {}
            slug = str(meta.get("scheme_slug") or "(none)")
            per_scheme[slug] = per_scheme.get(slug, 0) + 1
            dt = str(meta.get("doc_type") or "?")
            per_doc_type[dt] = per_doc_type.get(dt, 0) + 1
            fk = str(meta.get("fact_key") or "")
            if fk:
                per_fact_key[fk] = per_fact_key.get(fk, 0) + 1

    return {
        "exists": True,
        "collection": collection_name(),
        "path": str(common.path_for("chroma_dir")),
        "hnsw": _collection_kwargs(),
        "count": count,
        "per_scheme": dict(sorted(per_scheme.items(), key=lambda kv: -kv[1])),
        "per_doc_type": dict(sorted(per_doc_type.items(), key=lambda kv: -kv[1])),
        "per_fact_key": dict(sorted(per_fact_key.items(), key=lambda kv: -kv[1])),
        "embedding_dim": int(common.load_config()["embedding"]["dimension"]),
        "checked_at": utc_now_iso(),
    }


def coverage_gaps() -> dict[str, Any]:
    """Cross-check the store against the chunks. Catches a partial index.

    `upsert_chunks` raises on a missing vector, so a gap can only come from a
    stale store (chunks re-chunked, index not re-synced). This is the check that
    turns "the answer was wrong" into "the store was out of date".
    """
    chunks = common.read_jsonl(common.path_for("chunks_file"))
    chunk_ids = {c["chunk_id"] for c in chunks}
    collection = get_collection(create=False)
    if collection is None:
        return {"store_exists": False, "missing_from_store": sorted(chunk_ids),
                "orphan_in_store": [], "stale": True}

    stored = set((collection.get(include=[]) or {}).get("ids") or [])
    return {
        "store_exists": True,
        "chunks": len(chunk_ids),
        "stored": len(stored),
        "missing_from_store": sorted(chunk_ids - stored),
        "orphan_in_store": sorted(stored - chunk_ids),
        "stale": bool(chunk_ids - stored or stored - chunk_ids),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(*, full: bool = True) -> int:
    st = stats()
    gaps = coverage_gaps()
    print("=" * 78)
    print("  STAGE 4 - VECTOR STORE REPORT")
    print("=" * 78)
    print(f"  backend        : chromadb (persistent)")
    print(f"  path           : {st['path']}")
    print(f"  collection     : {st['collection']}")
    print(f"  hnsw           : {st.get('hnsw', {}).get('metadata', {})}")
    print(f"  vectors stored : {st['count']}")
    if not st["exists"]:
        print("  -> collection does not exist. Run: python -m store.chroma_store --rebuild")
        print("=" * 78)
        return 1

    if full and st["count"]:
        print()
        print("  per scheme:")
        for slug, n in st["per_scheme"].items():
            print(f"    {slug:<46} {n:>4}")
        print()
        print("  per doc_type:")
        for dt, n in st["per_doc_type"].items():
            print(f"    {dt:<46} {n:>4}")

    print()
    if gaps["stale"]:
        print(f"  COVERAGE GAP: {len(gaps['missing_from_store'])} chunk(s) are not in the "
              f"store and {len(gaps['orphan_in_store'])} stored row(s) are not in chunks.jsonl")
        for cid in gaps["missing_from_store"][:5]:
            print(f"    missing: {cid}")
        for cid in gaps["orphan_in_store"][:5]:
            print(f"    orphan : {cid}")
        print("  -> run: python -m store.chroma_store --rebuild")
    else:
        print(f"  coverage: {gaps['stored']}/{gaps['chunks']} chunks, no gaps, no orphans")
    print("=" * 78)
    return 1 if gaps["stale"] else 0


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4 - persist vectors in ChromaDB")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe the collection and re-upsert every chunk")
    ap.add_argument("--stats", action="store_true", help="print stats and exit")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()

    if args.stats:
        return report()

    started = time.perf_counter()
    common.ensure_dirs()

    if args.rebuild:
        n = rebuild()
    else:
        n = sync()

    print(f"wrote {n} row(s) in {time.perf_counter() - started:.2f}s")
    write_json(common.path_for("chroma_stats_file"),
               {"written": n, "elapsed_seconds": round(time.perf_counter() - started, 2),
                "rebuilt": bool(args.rebuild), "generated_at": utc_now_iso(),
                **stats()})
    return report()


if __name__ == "__main__":
    sys.exit(_main())
