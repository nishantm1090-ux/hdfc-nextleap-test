"""
ingest/chunk.py - STAGE 2: CHUNKING  (ADR-001, PRD §6 / §20)

Strategy: structure_first_hybrid.

    (A) FACTS  - one table_row / accordion / pill node becomes EXACTLY ONE chunk,
                 with a canonical fact_key. A fact's label and value can never be
                 split apart, which is the property that makes the citation in
                 every answer trustworthy.
    (B) PROSE  - remaining paragraphs/list items, concatenated per section and
                 split by a heading-aware recursive character splitter:
                   separators ["\\n\\n", "\\n", ". ", "? ", "! ", "; ", ", ", " "]
                   target 1000 tok, hard cap 1100, overlap 135 tok, with the
                   overlap snapped to a sentence boundary.
    (C) MERGE  - adjacent UNDERSIZED prose chunks are merged when their lead
                 sentences embed within cosine `semantic_merge.cosine_threshold`.
                 Skipped gracefully when sentence-transformers is unavailable.

Why not plain recursive splitting?  It slices a fee table apart, so the label
"Expense ratio" lands in one chunk and the value "1.03%" in another.
Why not pure semantic chunking?  3x the embedding cost, nondeterministic across
runs, and degenerate on short facts that have no sentence structure.

Everything here is deterministic: no timestamps, no randomness, no dict-order
dependence in any decision. Re-running on unchanged input therefore reproduces
the identical chunk_id sequence (PRD FR-2.4) - which is what makes the Chroma
upsert and the embedding cache safe. The only field that varies between runs is
the `ingested_at` stamp, which is metadata about the run, not about the text.

Run:  python -m ingest.chunk
      python -m ingest.chunk --stats
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import common
from common import (
    LOG,
    content_hash,
    count_tokens,
    read_json,
    short_hash,
    slugify,
    truncate_to_tokens,
    utc_now_iso,
    write_json,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Canonical fact vocabulary (PRD §6.1 fact_key)
# ---------------------------------------------------------------------------

#: canonical key -> regex matched against a node's label (and, failing that,
#: the head of its text). Order matters: the first match wins, so put the most
#: specific patterns first.
FACT_LABELS: list[tuple[str, str]] = [
    ("expense_ratio",     r"expense\s*ratio|\bter\b|total\s+expense|ongoing\s+charge"),
    ("exit_load",         r"exit\s*load|exit\s*charge|load\s*on\s*redemption|redemption\s*charge"),
    ("minimum_sip",       r"min\.?\s*for\s*sip|minimum\s*sip|min\s*sip|sip\s*amount"),
    ("minimum_investment", r"min\.?\s*for\s*\d|minimum\s*(lump|investment|amount)|min\.?\s*for\s*1st|min\.?\s*for\s*2nd"),
    ("lock_in_period",    r"lock[-\s]?in|lockin|minimum\s*holding\s*period|\b3\s*y\s*lock"),
    ("riskometer_level",  r"riskometer|risk\s*level|very\s+high\s+risk|high\s+risk|moderate\s+risk|low\s+risk"),
    ("benchmark",         r"benchmark|index\s*(tracked|tracked\s*by)"),
    ("aum",               r"\baum\b|fund\s*size|assets\s*under\s*management"),
    ("nav",               r"\bnav\b|net\s*asset\s*value"),
    ("statement_download", r"statement|capital\s*gains\s*statement|tax\s*report|transaction\s*report|download"),
    ("stamp_duty",        r"stamp\s*duty"),
    ("fund_objective",    r"investment\s*objective|\bobjective\b"),
    ("fund_category",     r"fund\s*category|scheme\s*type|\bcategory\b"),
    ("fund_managers",     r"fund\s*manager|fund\s*management"),
    ("taxation",          r"tax\s*implication|capital\s*gains\s*tax|\bltcg\b|\bstcg\b"),
    ("risk_profile",      r"very\s+high\s+risk|high\s+risk|moderate\s+risk|low\s+risk|asset\s*allocation"),
    ("rating",            r"^\s*rating\s*$|rating\s*\(groww\)|star\s*rating"),
    ("fund_launch",       r"date\s*of\s*(launch|inception)|since\s+its\s+inception|\blaunch(ed)?\b"),
    ("contact",           r"address|customer\s*care|helpline|toll\s*free"),
    ("how_to_invest",     r"how\s+to\s+invest|how\s+do\s+i\s+invest|invest\s+in\s+\d+\s+min"),
    ("how_to_redeem",     r"how\s+to\s+redeem|how\s+to\s+sell|redeem\s+your"),
    ("sip_vs_lumpsum",    r"sip\s+and\s+lump|either\s+sip\s+or\s+lump|sip\s+or\s+lumpsum"),
    ("pe_pb_ratio",       r"\bpe\b.*\bpb\b|pe\s*ratio|pb\s*ratio"),
]

_FACT_COMPILED = [(key, re.compile(pat, re.I)) for key, pat in FACT_LABELS]

SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

#: fact chunks get a much tighter cap than prose: a fact is short by nature
FACT_MAX_TOKENS = 120


def canonical_fact_key(label: str) -> str | None:
    """Map a node label (or the head of its text) to a canonical fact_key.

    Returns None when nothing matches, which keeps a node as prose instead of
    inventing a fact classification for it.
    """
    if not label:
        return None
    text = _squash(label)
    for key, pat in _FACT_COMPILED:
        if pat.search(text):
            return key
    return None


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


# ---------------------------------------------------------------------------
# Chunk container
# ---------------------------------------------------------------------------

#: The full PRD §6.1 metadata schema. validate_chunk() enforces every key.
REQUIRED_CHUNK_FIELDS: tuple[str, ...] = (
    "chunk_id", "text", "source_url", "page_title", "source_type", "publisher",
    "scheme_name", "scheme_slug", "scheme_aliases", "plan", "category", "doc_type",
    "fact_key", "fact_value", "section_heading", "chunk_index", "char_start",
    "char_end", "token_count", "content_hash", "source_fetched_at", "ingested_at",
    "pii_scan",
)


@dataclass
class Section:
    """A heading and the nodes beneath it."""

    heading: str
    level: int
    nodes: list[Any] = field(default_factory=list)


def build_chunk_id(scheme_slug: str, section: str, text: str) -> str:
    """Deterministic, content-addressed id: {scheme_slug}__{section}__{hash6}.

    No timestamps, no randomness. Identical input => identical id (FR-2.4).
    """
    scheme = scheme_slug or "corpus"
    sect = slugify(section)[:40] or "general"
    return f"{scheme}__{sect}__{short_hash(' '.join(text.split()), 6)}"


# ---------------------------------------------------------------------------
# (A) Facts
# ---------------------------------------------------------------------------


def split_facts(section: Section, doc: dict[str, Any], ck: dict[str, Any],
                start_index: int) -> tuple[list[dict[str, Any]], set[int]]:
    """One fact node -> exactly one chunk.

    Returns (chunks, consumed_node_orders). The caller passes the consumed set
    to split_prose so a fact's text is not ALSO embedded inside a prose chunk.

    A node whose `label: value` does not fit `fact_max_tokens` is DECLINED, not
    truncated. Truncating a fact to the cap is precisely the failure ADR-001
    exists to prevent: the label survives, the value is cut off mid-sentence, and
    the chunk reads like an answer but is not one. Declining routes the node to
    the prose splitter, which keeps whole sentences instead.
    """
    chunks: list[dict[str, Any]] = []
    consumed: set[int] = set()
    max_tokens = int(ck.get("fact_max_tokens", FACT_MAX_TOKENS))

    for node in section.nodes:
        if node.get("type") not in ("table_row", "accordion"):
            continue
        label = _squash(node.get("label") or "")
        value = _squash(node.get("value") or "")
        if not label:
            continue

        fact_key = canonical_fact_key(label) or canonical_fact_key(node.get("text", "")[:120])
        if fact_key is None:
            continue                      # not a fact we track -> stays prose

        text = _squash(node.get("text") or "")
        if not text:
            text = f"{label}: {value}" if value else label

        if count_tokens(text) > max_tokens:
            continue                      # not atomic -> prose, never truncated

        # A bare label with no value ("exit load") is chrome, not a fact.
        if not value and count_tokens(text) < int(ck.get("min_fact_tokens", 4)):
            continue

        consumed.add(int(node.get("order", -1)))
        chunks.append(
            _make_chunk(
                doc=doc,
                text=text,
                section=section.heading,
                doc_type="fact" if node.get("type") == "table_row" else "faq",
                fact_key=fact_key,
                fact_value=value or text,
                chunk_index=start_index + len(chunks),
            )
        )
    return chunks, consumed


# ---------------------------------------------------------------------------
# (B) Prose - heading-aware recursive character splitting
# ---------------------------------------------------------------------------


def _split_recursive(text: str, separators: Sequence[str], cap: int) -> list[str]:
    """Split until every piece fits `cap`, using the first separator that helps."""
    if not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_recursive(text, rest, cap)

    parts = text.split(sep)
    # Re-attach the separator to the preceding piece so no characters are lost.
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    if len(pieces) == 1:
        return _split_recursive(text, rest, cap)

    out: list[str] = []
    for piece in pieces:
        if piece and count_tokens(piece) > cap:
            out.extend(_split_recursive(piece, rest, cap))
        else:
            out.append(piece)
    return out


_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(\s|$)")


def _snap_to_sentence_start(text: str, min_prefix_tokens: int = 12) -> str:
    """Drop a short leading fragment so a chunk never opens mid-sentence.

    Overlap carried from the previous chunk usually begins partway through a
    sentence. If that fragment is longer than `min_prefix_tokens` it carries real
    meaning and is kept; if shorter, it is trimmed to the first sentence start.
    """
    if count_tokens(text) <= min_prefix_tokens:
        return text
    m = _SENTENCE_END_RE.search(text)
    if not m or m.end() >= len(text):
        return text
    return text[m.end():].lstrip()


def split_prose(section: Section, doc: dict[str, Any], ck: dict[str, Any],
                start_index: int, skip_orders: set[int] | None = None
                ) -> list[dict[str, Any]]:
    """Recursive character split with sentence-snapped overlap.

    `skip_orders` holds the node orders already emitted as atomic fact chunks.
    A fact is either its own chunk or part of the narrative, never both: the same
    sentence embedded twice makes retrieval scores depend on how many places a
    fact happens to be repeated, which is worse than having a slightly narrower
    prose window.
    """
    target = int(ck.get("prose_target_tokens", 1000))
    cap = int(ck.get("prose_max_tokens", 1100))
    overlap = int(ck.get("prose_overlap_tokens", 135))
    skip = skip_orders or set()
    floor = int(ck.get("min_chunk_tokens", 4))

    body = "\n\n".join(
        _squash(n.get("text", ""))
        for n in section.nodes
        if n.get("text") and int(n.get("order", -1)) not in skip
    ).strip()
    if not body:
        return []

    pieces = _split_recursive(body, SEPARATORS, cap)

    # Greedy merge up to target, carrying an overlap window forward.
    windows: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        pt = count_tokens(piece)
        if current and current_len + pt > target:
            windows.append(current)
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                prev_len = count_tokens(prev)
                if carry_len + prev_len > overlap:
                    break
                carry.insert(0, prev)
                carry_len += prev_len
            current, current_len = carry, carry_len
        current.append(piece)
        current_len += pt
    if current:
        windows.append(current)

    chunks: list[dict[str, Any]] = []
    for window in windows:
        text = "".join(window).strip()
        if not text:
            continue
        if chunks:
            text = _snap_to_sentence_start(text) or text
        text = _squash_keep(text)
        if count_tokens(text) > cap:
            text = truncate_to_tokens(text, cap)
        if not text or count_tokens(text) < floor:
            continue
        chunks.append(
            _make_chunk(
                doc=doc,
                text=text,
                section=section.heading,
                doc_type="prose",
                fact_key="",
                fact_value="",
                chunk_index=start_index + len(chunks),
            )
        )
    return chunks


def _squash_keep(text: str) -> str:
    """Collapse whitespace but keep sentence punctuation spacing readable."""
    return re.sub(r"[ \t]+", " ", text).strip()


# ---------------------------------------------------------------------------
# (C) Semantic merge of undersized chunks
# ---------------------------------------------------------------------------


def _lead_sentence(text: str, n: int = 220) -> str:
    return truncate_to_tokens(text, n)


def semantic_merge(chunks: list[dict[str, Any]], *, threshold: float,
                   max_tokens: int, min_tokens: int,
                   model: Any = None) -> tuple[list[dict[str, Any]], int]:
    """Merge adjacent UNDERSIZED prose chunks whose lead sentences are similar.

    This is an optimisation, never a correctness requirement, so it degrades to a
    no-op when the embedding model is unavailable (PRD FR-2.2, §6 step C).
    """
    prose_idx = [i for i, c in enumerate(chunks)
                 if c["doc_type"] == "prose" and c["token_count"] < min_tokens]
    if len(prose_idx) < 2:
        return chunks, 0

    if model is None:
        try:
            from embed.index import load_model  # noqa: PLC0415

            LOG.info("  semantic merge: loading embedding model")
            model = load_model()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("  semantic merge skipped (%s: %s) - it is optional, "
                        "not a correctness requirement", type(exc).__name__, exc)
            return chunks, 0

    import numpy as np  # noqa: PLC0415

    leads = [_lead_sentence(chunks[i]["text"]) for i in prose_idx]
    vectors = model.encode(leads, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    merged = 0
    drop: set[int] = set()
    carried: str = ""
    for k in range(len(prose_idx) - 1):
        i, j = prose_idx[k], prose_idx[k + 1]
        if i in drop or j in drop:
            continue
        if j != i + 1:
            continue                      # only merge true neighbours
        sim = float(np.dot(vectors[k], vectors[k + 1]))
        combined = carried + " " + chunks[j]["text"] if carried else f"{chunks[i]['text']} {chunks[j]['text']}"
        if sim >= threshold and count_tokens(combined) <= max_tokens:
            chunks[i]["text"] = _squash_keep(combined)
            chunks[i]["token_count"] = count_tokens(chunks[i]["text"])
            chunks[i]["content_hash"] = content_hash(chunks[i]["text"])
            drop.add(j)
            carried = chunks[i]["text"]
            merged += 1
        else:
            carried = ""

    if not merged:
        return chunks, 0

    kept = [c for idx, c in enumerate(chunks) if idx not in drop]
    for i, c in enumerate(kept):
        c["chunk_index"] = i
    return kept, merged


# ---------------------------------------------------------------------------
# Chunk assembly + validation
# ---------------------------------------------------------------------------


def _make_chunk(*, doc: dict[str, Any], text: str, section: str, doc_type: str,
                fact_key: str, fact_value: str, chunk_index: int) -> dict[str, Any]:
    """Build a chunk carrying the FULL PRD §6.1 metadata schema."""
    text = _squash_keep(text)
    return {
        "chunk_id": build_chunk_id(doc.get("scheme_slug") or doc.get("doc_id", "corpus"),
                                   section, text),
        "text": text,
        "source_url": doc.get("source_url", ""),
        "page_title": doc.get("page_title", ""),
        "source_type": doc.get("page_type", ""),
        "publisher": doc.get("publisher", ""),
        "scheme_name": doc.get("scheme_name", "") or "",
        "scheme_slug": doc.get("scheme_slug", "") or doc.get("doc_id", ""),
        "scheme_aliases": list(doc.get("aliases", []) or []),
        "plan": doc.get("plan", "") or "",
        "category": doc.get("category", "") or "",
        "doc_type": doc_type,
        "fact_key": fact_key,
        "fact_value": fact_value,
        "section_heading": section,
        "chunk_index": chunk_index,
        "char_start": 0,
        "char_end": len(text),
        "token_count": count_tokens(text),
        "content_hash": content_hash(text),
        "source_fetched_at": doc.get("source_fetched_at", ""),
        "ingested_at": utc_now_iso(),
        "pii_scan": doc.get("pii_scan", "clean"),
    }


def validate_chunk(chunk: dict[str, Any]) -> None:
    """Enforce the PRD §6.1 contract. Raises on violation.

    A malformed chunk that reaches Chroma becomes an uncitable answer, so this
    is a hard gate rather than a warning (architecture.md §4 Stage 2 step 5).
    """
    missing = [f for f in REQUIRED_CHUNK_FIELDS if f not in chunk]
    if missing:
        raise ValueError(f"chunk {chunk.get('chunk_id', '?')!r} missing fields: {missing}")
    if not chunk["text"].strip():
        raise ValueError(f"chunk {chunk['chunk_id']!r} has empty text")
    if not chunk["source_url"]:
        raise ValueError(f"chunk {chunk['chunk_id']!r} has no source_url - it could never be cited")
    if chunk["token_count"] <= 0:
        raise ValueError(f"chunk {chunk['chunk_id']!r} has token_count <= 0")
    recomputed = count_tokens(chunk["text"])
    if abs(recomputed - chunk["token_count"]) > 1:
        raise ValueError(
            f"chunk {chunk['chunk_id']!r} token_count {chunk['token_count']} "
            f"!= recount {recomputed}"
        )
    if chunk["content_hash"] != content_hash(chunk["text"]):
        raise ValueError(f"chunk {chunk['chunk_id']!r} content_hash does not match its text")


def validate_corpus(chunks: list[dict[str, Any]], ck: dict[str, Any]) -> None:
    cap = int(ck.get("prose_max_tokens", 1100))
    ids: set[str] = set()
    for c in chunks:
        validate_chunk(c)
        if c["chunk_id"] in ids:
            raise ValueError(f"duplicate chunk_id: {c['chunk_id']}")
        ids.add(c["chunk_id"])
        if c["token_count"] > cap:
            raise ValueError(
                f"chunk {c['chunk_id']!r} is {c['token_count']} tokens, over the {cap} cap"
            )


# ---------------------------------------------------------------------------
# Sectioning + driver
# ---------------------------------------------------------------------------


def _clean_heading(text: str) -> str:
    """Collapse whitespace but PRESERVE case.

    Headings are shown to the user in the citation line ("Source: Frequently
    Asked Questions"). Lower-casing them for matching convenience would put
    "frequently asked questions" in front of a reader. Matching is made
    case-insensitive where it matters instead, via canonical_fact_key().
    """
    return re.sub(r"\s+", " ", text or "").strip()


def group_sections(nodes: list[dict[str, Any]], doc_title: str) -> list[Section]:
    """A heading node starts a new section; nodes before the first heading fall
    into an implicit section named after the document."""
    sections: list[Section] = [Section(heading=_clean_heading(doc_title) or "Overview",
                                        level=1, nodes=[])]
    for node in nodes:
        if node.get("type") == "heading":
            sections.append(Section(heading=_clean_heading(node.get("text", "")) or "General",
                                    level=int(node.get("level", 2)), nodes=[]))
            continue
        if node.get("text"):
            sections[-1].nodes.append(node)
    return [s for s in sections if s.nodes]


def document_is_answerable(doc_chunks: list[dict[str, Any]], doc: dict[str, Any],
                           ck: dict[str, Any]) -> str:
    """Return "" if the document can answer anything, else the rejection reason.

    A page that yields zero fact chunks and whose nodes are all UI chrome - a
    search box, a footer nav, a heading with no body - is a JavaScript shell, not
    a document. Indexing it is actively harmful: it never wins a retrieval, but it
    does occupy prompt space and it dilutes the recall numbers in eval/.

    The test is deliberately narrow - "no facts AND thin nodes" - so a genuinely
    prose-only page (an AMFI investor-education article, say) is still kept.
    """
    n_facts = sum(1 for c in doc_chunks if c["doc_type"] in ("fact", "faq"))
    if n_facts:
        return ""
    node_chars = [len(re.sub(r"\s+", " ", n.get("text", "")))
                  for n in doc.get("nodes", []) if n.get("text")]
    if not node_chars:
        return "no text-bearing nodes at all"
    mean_chars = statistics.mean(node_chars)
    floor = float(ck.get("min_mean_node_chars", 80))
    if mean_chars < floor:
        return (f"no fact-bearing nodes and mean node is only {mean_chars:.0f} chars "
                f"(< {floor:.0f}) - navigation/search shell, not content")
    return ""


def _fact_separation_pct(facts: list[dict[str, Any]]) -> float:
    """ADR-001 acceptance test: % of facts whose VALUE is missing from its chunk.

    A fact is only separated if the value that the source node carried does not
    appear in the emitted chunk text. 0.0% is the pass condition.
    """
    if not facts:
        return 0.0
    separated = 0
    for c in facts:
        value = (c.get("fact_value") or "").strip().lower()
        if not value:
            continue                        # pill/badge facts state it inline
        if value not in c["text"].lower():
            separated += 1
    return round(100.0 * separated / len(facts), 3)


def chunk_all(*, force: bool = False) -> dict[str, Any]:
    """Chunk the whole corpus. Writes data/chunks.jsonl + data/chunk_stats.json."""
    common.ensure_dirs()
    corpus = read_json(common.path_for("corpus_file"), None)
    if not corpus or not corpus.get("documents"):
        LOG.error("No corpus found. Run: python -m ingest.fetch --all && python -m ingest.clean")
        return {}
    if common.path_for("chunks_file").exists() and not force:
        existing = common.read_jsonl(common.path_for("chunks_file"))
        stats = read_json(common.path_for("chunk_stats_file"), None)
        if existing and stats:
            LOG.info("Reusing %d existing chunks - pass --force to re-chunk",
                     len(existing))
            return stats

    ck = common.load_config()["chunking"]
    strategy = ck.get("strategy", "structure_first_hybrid")
    if strategy == "recursive":
        LOG.warning("strategy=recursive is the ADR-001 REJECTED baseline; "
                    "use structure_first_hybrid")

    all_chunks: list[dict[str, Any]] = []
    per_scheme: dict[str, Any] = {}
    all_facts: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}

    for doc in corpus["documents"]:
        doc_id = doc.get("doc_id", "corpus")
        doc_chunks: list[dict[str, Any]] = []
        for section in group_sections(doc.get("nodes", []), doc.get("page_title", "")):
            fact_chunks, consumed = split_facts(section, doc, ck, len(doc_chunks))
            doc_chunks.extend(fact_chunks)
            doc_chunks.extend(split_prose(section, doc, ck, len(doc_chunks),
                                          skip_orders=consumed))

        # Re-index so chunk_index is contiguous within the document.
        for i, c in enumerate(doc_chunks):
            c["chunk_index"] = i

        reason = document_is_answerable(doc_chunks, doc, ck)
        if reason:
            rejected[doc_id] = reason
            LOG.warning("corpus gate: dropping %s - %s", doc_id, reason)
            continue

        key = doc.get("scheme_slug") or doc.get("doc_id", "corpus")
        tokens = [c["token_count"] for c in doc_chunks]
        n_facts = sum(1 for c in doc_chunks if c["doc_type"] in ("fact", "faq"))
        per_scheme[key] = {
            "chunks": len(doc_chunks),
            "facts": n_facts,
            "prose": len(doc_chunks) - n_facts,
            "mean_tokens": round(statistics.mean(tokens), 1) if tokens else 0,
            "median_tokens": int(statistics.median(tokens)) if tokens else 0,
            "max_tokens": max(tokens) if tokens else 0,
            "min_tokens": min(tokens) if tokens else 0,
        }
        all_chunks.extend(doc_chunks)
        all_facts.extend(c for c in doc_chunks if c["doc_type"] in ("fact", "faq"))

    # --- (C) semantic merge over the whole corpus -----------------------
    sm = ck.get("semantic_merge", {}) or {}
    merges = 0
    if sm.get("enabled", True):
        all_chunks, merges = semantic_merge(
            all_chunks,
            threshold=float(sm.get("cosine_threshold", 0.62)),
            max_tokens=int(ck.get("prose_max_tokens", 1100)),
            min_tokens=int(ck.get("prose_min_tokens", 40)),
        )

    validate_corpus(all_chunks, ck)
    write_jsonl(common.path_for("chunks_file"), all_chunks)

    n_fact = sum(1 for c in all_chunks if c["doc_type"] in ("fact", "faq"))
    tokens = [c["token_count"] for c in all_chunks]
    prose_tokens = [c["token_count"] for c in all_chunks if c["doc_type"] == "prose"]
    min_tok = int(ck.get("prose_min_tokens", 40))
    fact_keys: dict[str, int] = {}
    for c in all_facts:
        if c.get("fact_key"):
            fact_keys[c["fact_key"]] = fact_keys.get(c["fact_key"], 0) + 1

    stats = {
        "generated_at": utc_now_iso(),
        "strategy": strategy,
        "tokenizer": common.tokenizer_backend(),
        "totals": {
            "chunks": len(all_chunks),
            "facts": n_fact,
            "prose": len(all_chunks) - n_fact,
            "faq": sum(1 for c in all_chunks if c["doc_type"] == "faq"),
            "quarantined": len(corpus.get("quarantined", [])),
            "documents_kept": len(per_scheme),
            "documents_rejected": len(rejected),
        },
        "tokens": {
            "mean": round(statistics.mean(tokens), 1) if tokens else 0,
            "median": int(statistics.median(tokens)) if tokens else 0,
            "min": min(tokens) if tokens else 0,
            "max": max(tokens) if tokens else 0,
            "cap": int(ck.get("prose_max_tokens", 1100)),
        },
        "per_scheme": per_scheme,
        "rejected_documents": rejected,
        "fact_key_histogram": dict(sorted(fact_keys.items(), key=lambda kv: -kv[1])),
        "adr001_evidence": {
            "fact_label_value_separated_pct": _fact_separation_pct(all_facts),
            # Measured over PROSE chunks only. A fact chunk is one label + one
            # value ("nav: 1189.08") and is short by definition; counting it
            # against the prose target reports 60%+ "undersized" while the actual
            # prose problem is near zero, which is a meaningless number.
            "prose_below_min_tokens_pct": round(
                100.0 * sum(1 for t in prose_tokens if t < min_tok) / len(prose_tokens), 3
            ) if prose_tokens else 0.0,
            "prose_median_tokens": int(statistics.median(prose_tokens)) if prose_tokens else 0,
            "semantic_merges_applied": merges,
            "note": "fact_label_value_separated_pct must be 0.0 - the ADR-001 "
                    "acceptance test (PRD S20). Non-zero means re-open ADR-001.",
        },
    }
    write_json(common.path_for("chunk_stats_file"), stats)
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(stats: dict[str, Any]) -> int:
    if not stats:
        print("!! no chunks produced - run ingest.fetch and ingest.clean first")
        return 1
    t, ev = stats["totals"], stats["adr001_evidence"]
    print("=" * 78)
    print("  STAGE 2 - CHUNKING REPORT  (ADR-001: structure_first_hybrid)")
    print("=" * 78)
    print(f"  strategy        : {stats['strategy']}")
    print(f"  tokenizer       : {stats['tokenizer']}")
    print(f"  chunks          : {t['chunks']}   (fact={t['facts']} faq={t['faq']} prose={t['prose']})")
    print(f"  tokens          : mean={stats['tokens']['mean']} median={stats['tokens']['median']} "
          f"min={stats['tokens']['min']} max={stats['tokens']['max']} cap={stats['tokens']['cap']}")
    print()
    print(f"  {'scheme':<44} {'chunks':>6} {'fact':>5} {'prose':>5} {'med_tok':>8} {'max_tok':>8}")
    print("  " + "-" * 76)
    for slug, s in stats["per_scheme"].items():
        print(f"  {slug[:44]:<44} {s['chunks']:>6} {s['facts']:>5} {s['prose']:>5} "
              f"{s['median_tokens']:>8} {s['max_tokens']:>8}")
    print()
    print("  Fact keys captured (this is the answerable question surface):")
    for k, n in stats["fact_key_histogram"].items():
        print(f"    {k:<22} {n:>3}")
    print()
    print("  ADR-001 evidence:")
    sep = ev["fact_label_value_separated_pct"]
    print(f"    fact_label_value_separated_pct : {sep}   "
          f"{'PASS' if sep == 0.0 else 'FAIL (re-open ADR-001)'}")
    print(f"    prose_below_min_tokens_pct     : {ev['prose_below_min_tokens_pct']}   "
          f"(prose median {ev['prose_median_tokens']} tok)")
    print(f"    semantic_merges_applied         : {ev['semantic_merges_applied']}")
    if stats.get("rejected_documents"):
        print()
        print("  Corpus gate dropped (indexed nothing, would only add noise):")
        for doc_id, why in stats["rejected_documents"].items():
            print(f"    - {doc_id}: {why}")
    print("=" * 78)
    return 0 if ev["fact_label_value_separated_pct"] == 0.0 else 1


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2 - ADR-001 hybrid chunking")
    ap.add_argument("--force", action="store_true", help="re-chunk even if chunks.jsonl exists")
    ap.add_argument("--stats", action="store_true", help="print the last stats without re-chunking")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()

    if args.stats:
        return report(read_json(common.path_for("chunk_stats_file"), {}) or {})
    return report(chunk_all(force=args.force))


if __name__ == "__main__":
    sys.exit(_main())
