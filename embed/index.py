"""
embed/index.py - STAGE 3: EMBEDDING  (PRD FR-3, architecture.md §4 Stage 3)

Turns data/chunks.jsonl into unit-norm 384-dim vectors, cached on disk by
`content_hash` so a re-run costs nothing and a single changed chunk re-embeds
exactly one vector.

THE LEAD-WINDOW RULE (do not "fix" this)
----------------------------------------
`all-MiniLM-L6-v2` was trained on short passages and its `max_seq_length` is
256. Feeding it a 1000-token chunk silently truncates at the model's own limit
and throws away the rest, so embedding the raw chunk would mean the vector
describes only the first ~256 tokens anyway - just without our knowing it.

So we embed an explicit, logged window: a metadata prefix (scheme, category,
section) plus the lead of the chunk, truncated to `embed_lead_tokens` using the
SAME tokenizer we counted with. Two consequences, both accepted deliberately:

  1. A prose vector describes the HEAD of its chunk, not all of it. That is
     exactly why Stage 5 runs BM25 alongside dense retrieval - the tail of a
     long chunk stays findable lexically, which a single dense vector cannot do.
  2. The prefix is what stops "Expense ratio: 1.03% p.a." from embedding almost
     identically to the other four schemes' expense-ratio rows. Without it the
     five rows differ by two tokens and the retriever cannot tell them apart.

Vectors are NOT written into chunks.jsonl. Stage 4 joins the two sides on
`content_hash`, so a re-chunk that changes prose never invalidates a cached
vector for an unchanged fact.

Run:  python -m embed.index
      python -m embed.index --force
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import common
from common import LOG, content_hash, read_json, truncate_to_tokens, utc_now_iso, write_json

# numpy is a hard dependency here - chromadb and sentence-transformers both pull
# it in, so there is no "embed without numpy" fallback worth maintaining.


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class EmbedStats:
    total_chunks: int
    computed: int
    cached: int
    failed: int
    dimension: int
    elapsed_seconds: float
    model: str
    cache_hit_pct: float
    max_seq_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# The lead-window text
# ---------------------------------------------------------------------------


def build_embed_text(chunk: dict[str, Any], cfg: dict[str, Any]) -> str:
    """prefix + lead(chunk text) + alias tail, truncated to `embed_lead_tokens`.

    The prefix is NOT decoration. A fact chunk is "TER: 1.03" - three words that
    are near-identical across all five schemes. Prefixing with the scheme name is
    what makes "expense ratio of HDFC Large Cap" retrieve the Large Cap row
    rather than whichever row the encoder happened to nudge.

    The fund's OTHER public names - HDFC's Flexi Cap page is reached at
    hdfc-equity-fund-direct-growth, and people still ask for it by the old name -
    are appended at the END rather than placed in the prefix. Measured, not
    assumed: in the leading position the three-name alias list outweighed the one
    current name, and "expense ratio of HDFC Flexi Cap Fund - Direct Growth"
    retrieved the Large Cap row (0.7601 against 0.7721), because "Cap" is shared
    with two rivals and "Flexi" had been averaged out of the vector. Trailing,
    the same query puts Flexi Cap first with a 0.0126 margin, and the old name is
    still in the embedded text. Trailing loses a little on every scheme; leading
    lost the one scheme that has aliases at all.
    """
    template = cfg.get("embed_prefix_template",
                       "{scheme_name} ({category}) - {section_heading}: ")
    alias_tpl = cfg.get("embed_alias_suffix", " (also called {aliases})")
    aliases = chunk.get("scheme_aliases") or []
    joined = ", ".join(aliases)
    try:
        prefix = template.format(
            scheme_name=chunk.get("scheme_name") or chunk.get("scheme_slug", ""),
            aliases=joined,
            category=chunk.get("category", ""),
            section_heading=chunk.get("section_heading", ""),
        )
    except (KeyError, IndexError):
        # A literal brace in a page heading must not crash the whole run.
        prefix = f"{chunk.get('scheme_name', '')} - "
    tail = ""
    if joined:
        try:
            tail = alias_tpl.format(aliases=joined)
        except (KeyError, IndexError):
            tail = f" (also called {joined})"

    lead = int(cfg.get("embed_lead_tokens", 200))
    return truncate_to_tokens(prefix + chunk.get("text", "") + tail, lead)


def embed_hash(chunk: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Cache key for a chunk's vector: a hash of the text ACTUALLY EMBEDDED.

    This must NOT be the chunk's `content_hash`. `content_hash` addresses the
    raw chunk text, but the embedded text is `prefix + text` and the prefix
    contains the scheme name. "Min. for SIP: Rs 100" is byte-identical across
    four of our five schemes, yet those four must NOT share a vector - sharing
    one is exactly how a question about HDFC Large Cap gets answered with HDFC
    Flexi Cap's row. Hashing what we embed is the only key that is correct.
    """
    return content_hash(build_embed_text(chunk, cfg))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


#: The repo-local, committed copy of the embedding model, shipped under models/
#: so a Render free-tier instance NEVER has to download ~90 MB on a cold start:
#: the deployed image already contains it and the load is fully offline. The
#: directory is present in git and absent only on a checkout that predates it.
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"


def _local_model_present() -> bool:
    return (LOCAL_MODEL_DIR / "config.json").is_file() and (
        LOCAL_MODEL_DIR / "model.safetensors").is_file()


@lru_cache(maxsize=4)
def load_model(model_name: str | None = None, *, local_files_only: bool | None = None
               ) -> Any:
    """Load MiniLM on CPU. Never auto-select the GPU (it OOMs on tiny corpora).

    Cached at module scope on (model_name, local_files_only). In the UI this is
    the difference between a 1-2s model reload on EVERY question and one load per
    process: `_embed_question` calls it per query, and a Streamlit rerun re-runs
    the script but not the module, so the cache survives across questions. The
    measured cost of the uncached reload was ~0.6-1s per question (and ~16s for
    the very first load, when torch and sentence-transformers are imported).

    With the default `model_name=None` (and for every caller while a committed
    copy exists) the repo-local ONNX export under models/ wins:

      * it is committed, so a fresh clone (or a Render image rebuilt from it)
        loads the weights with zero network access;
      * it runs on onnxruntime, NOT PyTorch. Importing torch costs ~560 MB peak
        RSS on a cold Render free-tier instance (512 MB cap), which OOM-kills
        the process mid-question; onnxruntime is ~80-100 MB and imports in ~1s.

    The torch-backed fallbacks below (`_local_model_present`, Hugging Face
    cache/hub) only run on a checkout that predates the committed copy.
    """
    # Any download that cannot reach the Hub must fail fast, not hang the first
    # question forever. huggingface_hub reads these env vars on every download.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    name = model_name or common.load_config()["embedding"]["model_name"]

    # Committed repo-local ONNX copy: no torch, no network, ~50 MB on disk.
    if (LOCAL_MODEL_DIR / "model.onnx").is_file():
        from embed.onnx_encoder import OnnxMiniLM  # noqa: PLC0415

        model = OnnxMiniLM(LOCAL_MODEL_DIR)
        LOG.info("  model %s (repo-local ONNX)  dim=%d  max_seq_length=%d  "
                 "offline=True  runtime=onnxruntime",
                 name, model.dimension, model.max_seq_length)
        return model

    # Committed repo-local torch copy (checkouts that predate the ONNX export).
    if model_name is None and _local_model_present():
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(str(LOCAL_MODEL_DIR), device="cpu",
                                    local_files_only=True)
        dim = (model.get_embedding_dimension()
               if hasattr(model, "get_embedding_dimension")
               else model.get_sentence_embedding_dimension())
        LOG.info("  model %s (repo-local copy)  dim=%d  max_seq_length=%d  offline=True",
                 name, dim, model.max_seq_length)
        return model

    if local_files_only is None:
        local_files_only = model_is_cached(name)

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    try:
        model = SentenceTransformer(name, device="cpu", local_files_only=local_files_only)
    except Exception as exc:  # noqa: BLE001
        if not local_files_only:
            raise
        # Cache says warm but the load failed - fall back to a networked attempt
        # so a partially-cached model still works, and report why.
        LOG.warning("  offline load of %s failed (%s: %s); retrying online",
                    name, type(exc).__name__, exc)
        model = SentenceTransformer(name, device="cpu", local_files_only=False)

    dim = (model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
           else model.get_sentence_embedding_dimension())
    LOG.info("  model %s  dim=%d  max_seq_length=%d  offline=%s",
             name, dim, model.max_seq_length, local_files_only)
    return model


def model_is_cached(model_name: str) -> bool:
    """Is this model already in the Hugging Face cache?

    Checks the cache directory rather than asking the Hub, so the answer is
    available with no network. A model cached under a different revision slug
    simply reports False and is fetched normally.
    """
    from huggingface_hub import constants as hf_constants  # noqa: PLC0415

    slug = "models--" + model_name.replace("/", "--")
    hub_dir = Path(hf_constants.HF_HUB_CACHE)
    return (hub_dir / slug / "snapshots").is_dir() and any(
        (hub_dir / slug / "snapshots").iterdir()
    )


# ---------------------------------------------------------------------------
# Disk cache: one .npy per content_hash
# ---------------------------------------------------------------------------


def load_cache(dirpath: Path | None = None) -> dict[str, Any]:
    """content_hash -> float32 vector, for every .npy in the cache directory."""
    import numpy as np  # noqa: PLC0415

    d = Path(dirpath or common.path_for("embeddings_dir"))
    cache: dict[str, Any] = {}
    if not d.exists():
        return cache
    for f in d.glob("*.npy"):
        try:
            vec = np.load(f)
        except (ValueError, OSError) as exc:   # truncated / corrupt file
            LOG.warning("  cache: dropping unreadable %s (%s)", f.name, exc)
            f.unlink(missing_ok=True)
            continue
        if vec.ndim != 1:
            LOG.warning("  cache: dropping %s, expected a 1-D vector, got %s",
                        f.name, tuple(vec.shape))
            f.unlink(missing_ok=True)
            continue
        cache[f.stem] = vec.astype("float32", copy=False)
    return cache


def save_vector(dirpath: Path | None, content_hash: str, vec: Any) -> None:
    import numpy as np  # noqa: PLC0415

    d = Path(dirpath or common.path_for("embeddings_dir"))
    d.mkdir(parents=True, exist_ok=True)
    # Write to a temp name and rename: a half-written .npy that a crash leaves
    # behind would be loaded as a valid cache entry next run and silently
    # produce a wrong answer.
    tmp = d / f".{content_hash}.tmp.npy"
    np.save(tmp, np.asarray(vec, dtype="float32"))
    tmp.replace(d / f"{content_hash}.npy")


# ---------------------------------------------------------------------------
# The embed pass
# ---------------------------------------------------------------------------


def embed_chunks(chunks: list[dict[str, Any]], *, force: bool = False,
                 model: Any = None, cache_dir: Path | None = None,
                 show_progress: bool = True) -> tuple[dict[str, EmbedStats], EmbedStats]:
    """Embed every chunk. Returns ({embed_hash: vector}, stats).

    Caching is keyed on `embed_hash` (a hash of the embedded text), so a chunk
    whose raw text is unchanged but whose metadata prefix changed - a renamed
    scheme, a re-worded heading - re-embeds, and a chunk whose prefix is
    unchanged but whose scheme is NOT cannot pick up a neighbour's vector.
    """
    import numpy as np  # noqa: PLC0415

    cfg = common.load_config()["embedding"]
    started = time.perf_counter()
    cdir = Path(cache_dir or common.path_for("embeddings_dir"))
    cache = load_cache(cdir)

    keys = {id(c): embed_hash(c, cfg) for c in chunks}
    todo: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {}
    cached = 0

    for c in chunks:
        h = keys[id(c)]
        if h in cache and not force:
            vectors[h] = cache[h]
            cached += 1
        else:
            todo.append(c)

    # Distinct embed texts are embedded once. Two chunks can embed identically
    # only if their prefix AND text match, in which case one vector is correct
    # for both - but each still gets its own index row in Stage 4.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in todo:
        h = keys[id(c)]
        if h in seen:
            continue
        seen.add(h)
        unique.append(c)

    failed = 0
    if unique:
        if model is None:
            model = load_model(cfg["model_name"])
        texts = [build_embed_text(c, cfg) for c in unique]
        LOG.info("  encoding %d unique embed text(s) for %d chunk(s) "
                 "(batch_size=%d, %d served from cache)",
                 len(texts), len(todo), int(cfg.get("batch_size", 64)), cached)
        try:
            batch = model.encode(
                texts,
                batch_size=int(cfg.get("batch_size", 64)),
                normalize_embeddings=bool(cfg.get("normalize", True)),
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
        except Exception as exc:  # noqa: BLE001
            # A model failure must be loud but not fatal to the cached half.
            LOG.error("  encoding failed (%s: %s)", type(exc).__name__, exc)
            return {}, _empty_stats(len(chunks), cached, len(unique), cfg, started)

        batch = np.asarray(batch, dtype="float32")
        expected = int(cfg.get("dimension", 384))
        if batch.ndim != 2 or batch.shape[1] != expected:
            LOG.error("  model returned %s, expected (N, %d) - wrong model? "
                      "(all-mpnet-base-v2 is 768, the brief mandates MiniLM at 384)",
                      tuple(batch.shape), expected)
            return {}, _empty_stats(len(chunks), cached, len(unique), cfg, started)

        for c, vec in zip(unique, batch):
            save_vector(cdir, keys[id(c)], vec)
            vectors[keys[id(c)]] = vec

    stats = EmbedStats(
        total_chunks=len(chunks),
        computed=len(unique),
        cached=cached,
        failed=failed,
        dimension=int(cfg.get("dimension", 384)),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        model=cfg["model_name"],
        cache_hit_pct=round(100.0 * cached / len(chunks), 2) if chunks else 0.0,
        max_seq_length=int(getattr(model, "max_seq_length", 0) or 0),
    )
    return vectors, stats


def _empty_stats(total: int, cached: int, failed: int,
                 cfg: dict[str, Any], started: float) -> EmbedStats:
    return EmbedStats(
        total_chunks=total, computed=0, cached=cached, failed=failed,
        dimension=int(cfg.get("dimension", 384)),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        model=cfg.get("model_name", ""), cache_hit_pct=0.0,
    )


def embed_all(*, force: bool = False) -> EmbedStats:
    """Embed data/chunks.jsonl -> data/embeddings/*.npy + data/embed_index.jsonl."""
    common.ensure_dirs()
    chunks = common.read_jsonl(common.path_for("chunks_file"))
    if not chunks:
        LOG.error("No chunks found. Run: python -m ingest.chunk")
        return EmbedStats(0, 0, 0, 0, 0, 0.0, "", 0.0)

    cfg = common.load_config()["embedding"]
    # Read the previous stats BEFORE overwriting them: on a warm cache the model
    # is never loaded, and this is where max_seq_length comes from.
    prev = read_json(common.path_for("embed_stats_file"), {}) or {}
    LOG.info("embedding %d chunk(s) with %s", len(chunks), cfg["model_name"])

    # Only pay the model-load cost when there is something to compute.
    model = None
    cache = load_cache()
    keys = [embed_hash(c, cfg) for c in chunks]
    if force or any(h not in cache for h in keys):
        model = load_model(cfg["model_name"])
        LOG.info("  model.max_seq_length = %d  (this is why we embed a lead window, "
                 "not the whole chunk)", model.max_seq_length)

    vectors, stats = embed_chunks(chunks, force=force, model=model)

    # The join surface for Stage 4. One row per CHUNK (not per vector), because
    # two chunks may legitimately share a vector while carrying different
    # metadata - "Min. for SIP: Rs 100" in four schemes embeds to four distinct
    # vectors, but "Rating: 5" in three schemes whose prefixes differ does too.
    # Keying the index on chunk_id makes that explicit and reversible.
    index_rows = []
    for c in chunks:
        h = embed_hash(c, cfg)
        vec = vectors.get(h)
        index_rows.append({
            "chunk_id": c["chunk_id"],
            "content_hash": c["content_hash"],
            "embed_hash": h,
            "vector_file": f"{h}.npy",
            "dimension": int(cfg.get("dimension", 384)),
            "norm": round(float((vec ** 2).sum() ** 0.5), 6) if vec is not None else None,
            "embed_lead_tokens": int(cfg.get("embed_lead_tokens", 200)),
        })
    common.write_jsonl(common.path_for("embed_index_file"), index_rows)

    norms = [r["norm"] for r in index_rows if r["norm"] is not None]
    payload = stats.to_dict()
    payload["max_seq_length"] = stats.max_seq_length or int(prev.get("max_seq_length", 0) or 0)
    payload.update({
        "generated_at": utc_now_iso(),
        "vectors_on_disk": len(list(Path(common.path_for("embeddings_dir")).glob("*.npy"))),
        "unique_embed_texts": len({r["embed_hash"] for r in index_rows}),
        "chunks_sharing_a_vector": len(index_rows) - len({r["embed_hash"] for r in index_rows}),
        "norm_mean": round(statistics.mean(norms), 6) if norms else 0.0,
        "norm_min": round(min(norms), 6) if norms else 0.0,
        "norm_max": round(max(norms), 6) if norms else 0.0,
        "embed_lead_tokens": int(cfg.get("embed_lead_tokens", 200)),
        "join_surface": str(common.path_for("embed_index_file")),
        "note": "vectors are NOT stored in chunks.jsonl - Stage 4 joins chunk_id "
                "-> vector via the embed index, never via content_hash",
    })
    write_json(common.path_for("embed_stats_file"), payload)
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(stats: EmbedStats) -> int:
    saved = read_json(common.path_for("embed_stats_file"), {}) or {}
    # On a warm cache the model is never loaded, so fall back to the last
    # recorded value rather than printing a misleading 0.
    maxseq = stats.max_seq_length or int(saved.get("max_seq_length", 0) or 0)
    print("=" * 78)
    print("  STAGE 3 - EMBEDDING REPORT")
    print("=" * 78)
    print(f"  model            : {stats.model}")
    print(f"  dimension        : {stats.dimension}")
    print(f"  max_seq_length   : {maxseq}   <- the reason for the lead window")
    print(f"  chunks           : {stats.total_chunks}")
    print(f"  unique to embed  : {stats.computed}  "
          f"(2 chunks share a vector only when prefix AND text both match)")
    print(f"  cached           : {stats.cached}   ({stats.cache_hit_pct}% cache hit)")
    print(f"  failed           : {stats.failed}")
    print(f"  elapsed          : {stats.elapsed_seconds}s")
    if saved.get("norm_mean"):
        print(f"  ||v||            : mean={saved['norm_mean']} "
              f"min={saved['norm_min']} max={saved['norm_max']}  (want ~1.0)")
    print("=" * 78)
    return 1 if stats.failed or (stats.total_chunks and not stats.computed + stats.cached) else 0


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3 - embed chunks into cached vectors")
    ap.add_argument("--force", action="store_true", help="re-embed even if cached")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()
    return report(embed_all(force=args.force))


if __name__ == "__main__":
    sys.exit(_main())
