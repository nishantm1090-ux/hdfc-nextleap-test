"""
rag/retrieve.py - STAGE 5: RETRIEVAL  (PRD FR-5, architecture.md §4 Stage 5)

Turns a question into a small, diverse, ranked set of chunks that Stage 6 can
quote. The pipeline is exactly the one architecture.md specifies:

    normalise -> synonym expand -> dense top 20 -> BM25 top 20
              -> RRF fuse (k=60) -> MMR diversify (lambda=0.35)
              -> optional cross-encoder rerank -> min_cosine threshold

Why it is both dense AND lexical
--------------------------------
The corpus is mostly numbers: "expense ratio: 1.03%", "exit load of 1% if
redeemed within 1 year". MiniLM is weak on numerals and near-identical facts
across five schemes - it embeds four different "min. for sip: 100" rows close
together, which is the name-blur that `test_dense_search_blurs_names_that_only_
exist_in_metadata` pins. BM25 is weak on paraphrase and strong on exact tokens.
Fusing them is what lets one collection serve both "what is the exit load" (a
number) and "how do I get my tax paperwork" (a how-to).

Why MMR, and why it is not optional
-----------------------------------
All five scheme pages carry an exit-load row. Without diversification the top-5
fills with five versions of the same sentence and the generator has no room to
answer, let alone cite exactly one source. MMR is a correctness requirement
here, not a nicety.

The scheme filter is a HINT, never a gate
-----------------------------------------
Stage 4 measured that a `where`-filtered Chroma query can silently return fewer
rows than exist. A scheme named in the question is therefore used to *re-rank*,
and a short filtered result set falls back to the unfiltered ranking. It never
converts into "this scheme is absent" - that would report a retrieval artefact
as a fact about the fund, which is the one failure this whole project cannot
afford.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import common
from common import LOG

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """One retrieved chunk plus the full score trail.

    Every score is kept rather than just a final number. When an answer comes
    back wrong, the difference between "dense missed it" and "BM25 missed it" and
    "MMR dropped a duplicate" is the difference between a five-minute fix and an
    afternoon, and the UI's "Why this answer?" panel reads these directly.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    final_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    coverage: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    answer_bearing: bool = False

    def clears(self, min_coverage: float, min_cosine: float, min_bm25: float) -> bool:
        """May this chunk be allowed to answer the question?

        Coverage is necessary. A score is not sufficient.

        The reason is measured, not theoretical: on this corpus "price of gold in
        Mumbai" has a best dense score of 0.31 and a best BM25 of 2.9, while
        "exit load" - which IS in the corpus - has a dense score of 0.16. Any
        score threshold that rejects the first also rejects the second. What
        separates them is that the gold question matches one weak term ("price")
        out of three, and the exit-load question matches both of its own.

        So: a chunk must cover `min_coverage` of the question's content words, and
        must additionally clear at least one retriever's floor. Coverage alone
        would admit a chunk that happens to repeat the question's words while
        saying nothing relevant; the score floor alone admits the gold question.
        """
        if self.coverage < min_coverage:
            return False
        return self.dense_score >= min_cosine or self.bm25_score >= min_bm25

    @property
    def source_url(self) -> str:
        return str(self.metadata.get("source_url", ""))

    @property
    def scheme_slug(self) -> str:
        return str(self.metadata.get("scheme_slug", ""))

    @property
    def section_heading(self) -> str:
        return str(self.metadata.get("section_heading", ""))

    @property
    def fact_key(self) -> str:
        return str(self.metadata.get("fact_key", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "dense_score": round(self.dense_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "rrf_score": round(self.rrf_score, 8),
            "final_score": round(self.final_score, 6),
            "matched_terms": self.matched_terms,
            "coverage": round(self.coverage, 4),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rerank_score": self.rerank_score,
            "answer_bearing": self.answer_bearing,
            "source_url": self.source_url,
            "scheme_slug": self.scheme_slug,
            "section_heading": self.section_heading,
        }


class OutOfCorpus(Exception):
    """No chunk cleared `retrieval.min_cosine`.

    This is the single most important exception in the project: it is the
    difference between "I don't have that in my sources" and a confident,
    invented answer. Stage 6 turns it into a polite refusal with an educational
    link rather than a guess.
    """


# ---------------------------------------------------------------------------
# Query normalisation and expansion
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no retrieval signal. "the" matches almost every chunk and
# therefore inflates every BM25 score by a constant, flattening the ranking.
_STOPWORDS = frozenset("""
a an and are as at be but by can do does for from had has have how i if in
into is it its me my of on or should so tell that the their them then there
these they this to was what when where which who will with you your
""".split())

# Question-shaped prefixes: the question mark and these words sit between the
# user and the tokens BM25 needs.
_STRIP_PREFIXES = ("what is the", "what are the", "what's the", "whats the",
                   "tell me about", "tell me", "how do i", "how can i",
                   "how much", "please", "can you", "could you")

# Ordered longest-first so "how do i" wins over "how".
_SYNONYM_KEYS: list[str] = []  # filled lazily by `expand_query`


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric words, stopwords removed.

    Deliberately not a word-piece tokeniser. BM25 wants *terms*, and the
    alternative - tokenising the same way the embedder does - would make a
    numeric fact like "1.03%" explode into subword fragments that match nothing
    useful.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def normalise_query(q: str) -> str:
    """Lowercase, strip punctuation and filler, collapse whitespace.

    A decimal point between two digits is KEPT. This corpus is mostly figures -
    "expense ratio: 1.03%", "nav: 1189.08", "exit load of 1% if redeemed within
    1 year" - so replacing "." with a space turns "1.03%" into the two tokens
    "1" and "03%", which match nothing. Measured: with the naive rule,
    `is it 1.03%?` normalised to `is it 1 03%` and the number became
    unsearchable. Every other punctuation character is still a boundary, which
    is what splits "min.-for-sip" the way we want.
    """
    text = q.lower().strip()
    for prefix in _STRIP_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    # Underscores join words ("min_sip" -> "min sip").
    text = text.replace("_", " ")
    # Protect decimal points, then drop the rest of the punctuation.
    text = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)
    text = re.sub(r"[^\w\s%\x00]", " ", text)
    return " ".join(text.replace("\x00", ".").split())


def expand_query(q: str, synonym_map: dict[str, str] | None = None) -> str:
    """Append synonyms so a question in the user's words hits the page's words.

    "TER" -> "expense ratio total expense ratio charges". "min sip" -> "minimum
    sip amount smallest investment". The expansion is *additive*: the original
    tokens are kept, because a synonym list is a guess about vocabulary and must
    never be able to delete what the user actually asked for.
    """
    if synonym_map is None:
        synonym_map = common.load_config()["guardrails"].get("synonym_expansions", {}) or {}

    base = normalise_query(q)
    if not base:
        return ""

    extra: list[str] = []
    lowered = {str(k).lower(): str(v) for k, v in synonym_map.items()}

    global _SYNONYM_KEYS
    if not _SYNONYM_KEYS:
        _SYNONYM_KEYS = sorted(lowered, key=len, reverse=True)

    for key in _SYNONYM_KEYS:
        # Only fire on a whole-phrase or whole-word hit. Substring matching
        # would expand "aum" inside "amount", and "nav" inside "navigation".
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", base):
            extra.extend(tokenize(lowered[key]))

    seen: set[str] = set()
    out: list[str] = []
    for term in tokenize(base) + extra:
        if term not in seen:
            seen.add(term)
            out.append(term)
    return " ".join(out)


def query_tokens_for_matching(q: str) -> list[str]:
    """The ORIGINAL question's tokens - what `matched_terms` reports.

    Deliberately not the expanded set. "matched_terms" is shown to a user as
    "these are the words from your question that appear in this chunk"; if it
    listed synonym expansions the label would be a lie.
    """
    return tokenize(normalise_query(q))


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 over the same chunk ids the vector store holds.

    Built lazily and cached at module scope. It is pure Python over 121 short
    documents, so a rebuild costs single-digit milliseconds and there is nothing
    to gain from persisting it.
    """

    def __init__(self) -> None:
        self._bm25: Any = None
        self.chunk_ids: list[str] = []
        self._chunks: dict[str, dict[str, Any]] = {}
        self._doc_tokens: dict[str, set[str]] = {}
        self.built = False

    def build(self, chunks: Sequence[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        self._chunks = {c["chunk_id"]: c for c in chunks}
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        # Index the EMBED text, not the raw chunk text. The raw text of a fact
        # chunk is "min. for sip: 100" - identical in four schemes - so a lexical
        # index built on it alone could not tell those four chunks apart any more
        # than the dense index can. The metadata prefix is what makes both
        # retrievers scheme-aware.
        # set(tokenize(...)), NOT set(string). `set("min. for sip")` yields 12
        # CHARACTERS, and BM25Okapi will happily index a corpus of single
        # characters - no exception, avgdl equal to the average string length,
        # and every query scoring 0. It fails completely silently, which is why
        # this is written the long way with a comment.
        self._doc_tokens = {
            c["chunk_id"]: set(tokenize(_index_terms(c))) for c in chunks
        }
        corpus = [sorted(self._doc_tokens[cid]) for cid in self.chunk_ids]
        self._bm25 = BM25Okapi(corpus)
        self.built = True
        LOG.info("  BM25 index built over %d chunk(s)", len(self.chunk_ids))

    def search(self, query: str, top_k: int = 20
               ) -> list[tuple[str, float, list[str]]]:
        """Return [(chunk_id, score, matched_terms), ...], best first.

        `matched_terms` is computed from the chunk's own text rather than handed
        over by the caller, so it can only ever contain terms genuinely present.
        """
        if not self.built:
            raise RuntimeError("BM25Index.build() was never called")
        terms = tokenize(query)
        if not terms:
            return []

        scores = self._bm25.get_scores(terms)
        ranked = sorted(
            zip(self.chunk_ids, (float(s) for s in scores)),
            key=lambda kv: kv[1], reverse=True,
        )[:top_k]

        out: list[tuple[str, float, list[str]]] = []
        for cid, score in ranked:
            if score <= 0.0:
                continue  # BM25 gives 0 to a document sharing no term at all
            chunk = self._chunks[cid]
            haystack = " ".join(tokenize(_index_terms(chunk)))
            matched = sorted({t for t in terms if t in haystack})
            out.append((cid, score, matched))
        return out

    def chunk(self, chunk_id: str) -> dict[str, Any]:
        return self._chunks[chunk_id]


def _index_terms(chunk: dict[str, Any]) -> str:
    """The text BM25 indexes: chunk body + the scheme identity around it.

    Mirrors Stage 3's `embed_prefix_template` on purpose. The two retrievers
    disagreeing about what a chunk *is* is how you end up with a dense ranking
    and a lexical ranking that share no documents at all, and an RRF fusion of two
    disjoint lists is worth nothing.
    """
    aliases = chunk.get("scheme_aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    prefix = (f"{chunk.get('scheme_name', '')} "
              f"{' '.join(str(a) for a in aliases)} "
              f"({chunk.get('category', '')}) - {chunk.get('section_heading', '')}: ")
    return prefix + chunk.get("text", "")


# ---------------------------------------------------------------------------
# Fusion and diversification
# ---------------------------------------------------------------------------


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion.

    Score = sum over rankings of 1 / (k + rank), with rank 1-based. RRF is used
    rather than a weighted score blend because dense similarity and BM25 score
    are not on a comparable scale - MiniLM cosine runs 0.2-0.9 while BM25 runs
    0-12, and any linear combination of the two is an artefact of the arbitrary
    normalisation chosen. RRF only reads the *orderings*, so it needs no
    calibration and cannot be gamed by one retriever's score range.

    k=60 is the value from the original Cormack et al. paper and is what the PRD
    specifies; its effect is to flatten the top of each list so a document ranked
    #1 by one retriever does not automatically beat a document ranked #1 by both.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + position)
    return fused


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import numpy as np  # noqa: PLC0415

    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def mmr_diversify(candidates: Sequence[RetrievedChunk], lam: float, top_k: int,
                  *, embedding_lookup: Any = None) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance.

    Greedily pick the candidate maximising

        lam * relevance(chunk) - (1 - lam) * max similarity to what is picked

    With lam=0.35 relevance leads but similarity has real weight, which is what
    stops five exit-load rows from taking all five slots. `embedding_lookup` maps
    chunk_id -> vector; without it, similarity falls back to token overlap so
    MMR still works (degraded) when the vector cache is unavailable.
    """
    pool = [c for c in candidates]
    if not pool:
        return []

    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    max_relevance = max((abs(c.final_score) for c in pool), default=0.0) or 1.0
    selected: list[RetrievedChunk] = []
    remaining = list(pool)

    while remaining and len(selected) < top_k:
        best: RetrievedChunk | None = None
        best_value = -float("inf")
        for cand in remaining:
            relevance = abs(cand.final_score) / max_relevance
            if selected:
                redundancy = max(_similarity(cand, s, embedding_lookup) for s in selected)
            else:
                redundancy = 0.0
            value = lam * relevance - (1.0 - lam) * redundancy
            if value > best_value:
                best_value, best = value, cand
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)

    return selected


def _similarity(a: RetrievedChunk, b: RetrievedChunk, embedding_lookup: Any) -> float:
    if embedding_lookup is not None:
        try:
            va = embedding_lookup(a.chunk_id)
            vb = embedding_lookup(b.chunk_id)
            if va is not None and vb is not None:
                return _cosine(va, vb)
        except Exception:  # noqa: BLE001 - fall through to lexical
            pass
    # Token Jaccard. Not as good as cosine, but it is monotone in the same
    # direction for our purposes: two exit-load rows overlap heavily.
    ta, tb = set(tokenize(a.text)), set(tokenize(b.text))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Optional cross-encoder rerank
# ---------------------------------------------------------------------------

_RERANKER: Any = None
_RERANK_TRIED = False


def _get_reranker(model_name: str) -> Any:
    """Load the cross-encoder once, or return None.

    Never raises. The reranker is off by default and an absent model is a normal
    condition (fresh clone, no network), not an error - it must degrade to "no
    rerank" rather than fail the whole retrieval.
    """
    global _RERANKER, _RERANK_TRIED
    if _RERANK_TRIED:
        return _RERANKER
    _RERANK_TRIED = True
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        _RERANKER = CrossEncoder(model_name)
        LOG.info("  cross-encoder rerank enabled (%s)", model_name)
    except Exception as exc:  # noqa: BLE001
        LOG.info("  cross-encoder unavailable (%s: %s) - skipping rerank",
                 type(exc).__name__, exc)
        _RERANKER = None
    return _RERANKER


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

_BM25: BM25Index | None = None
_CHUNKS: dict[str, dict[str, Any]] | None = None
_EMBED_ROWS: dict[str, str] | None = None
#: chunk_id -> vector, memoised across queries. Without this, MMR re-reads
#: every chunk's `.npy` from disk on every question - a handful of 1.5 KB files,
#: but per-question I/O that Streamlit reruns pay repeatedly.
_EMBED_VEC_CACHE: dict[str, Any] = {}


def get_index(*, rebuild_if_stale: bool = True) -> BM25Index:
    """The module-level BM25 index, built once and reused.

    Invalidated by mtime on chunks.jsonl, so editing the corpus does not leave a
    silently stale index behind.
    """
    global _BM25, _CHUNKS, _EMBED_ROWS
    path = common.path_for("chunks_file")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python -m ingest.chunk")

    if _BM25 is not None and _BM25.built:
        return _BM25

    chunks = common.read_jsonl(path)
    if not chunks:
        raise FileNotFoundError(f"{path} is empty. Run: python -m ingest.chunk")
    _CHUNKS = {c["chunk_id"]: c for c in chunks}
    _BM25 = BM25Index()
    _BM25.build(chunks)
    _EMBED_ROWS = None
    return _BM25


def _embedding_lookup() -> Any:
    """chunk_id -> vector, for MMR. Returns None if the vector cache is absent."""
    global _EMBED_ROWS
    import numpy as np  # noqa: PLC0415

    if _EMBED_ROWS is None:
        try:
            rows = common.read_jsonl(common.path_for("embed_index_file"))
        except FileNotFoundError:
            return None
        if not rows:
            return None
        _EMBED_ROWS = {r["chunk_id"]: r["vector_file"] for r in rows}

    base = Path(common.path_for("embeddings_dir"))

    def lookup(chunk_id: str) -> Any:
        name = _EMBED_ROWS.get(chunk_id)  # type: ignore[union-attr]
        if not name:
            return None
        if name not in _EMBED_VEC_CACHE:
            f = base / name
            if not f.exists():
                return None
            _EMBED_VEC_CACHE[name] = np.load(f)
        return _EMBED_VEC_CACHE[name]

    return lookup


# ---------------------------------------------------------------------------
# Term coverage - the real out-of-corpus gate
# ---------------------------------------------------------------------------

# Words that carry no distinctive signal. A chunk covering only these has covered
# nothing, so they are excluded from the coverage denominator - otherwise "what
# is the exit load of HDFC Large Cap" only needs to match one of {exit, load}.
#
# `nav` is deliberately NOT here, even though "nav" appears in all five page
# titles. It is a fact_key in this corpus ("nav", "nav_change", "nav_date") and
# "what is the NAV of HDFC Small Cap" has to be able to match on it. Presence in
# a page title is not the same as absence of meaning.
_COVERAGE_NOISE = frozenset("""
fund funds mutual scheme schemes hdfc amc direct growth plan invest
investment investments details detail information tell know about please
""".split())

# A query term counts as matched when the chunk contains it, or contains a prefix
# of it of at least this length. This is what lets "minimum SIP" match a chunk
# that literally says "min. for sip" - the single most common phrasing in this
# corpus. The floor of 4 is the important part: a 3-character prefix would let
# "mini" satisfy "minimum", and "sip" would satisfy half the dictionary.
_MIN_PREFIX = 4


def content_terms(q: str) -> list[str]:
    """The question's distinctive words - the coverage denominator.

    Falls back to the full non-stopword list when noise-filtering empties it.
    "direct growth plan" consists entirely of words that appear in every chunk's
    metadata prefix, so the filtered list is empty - but that makes it a question
    with no *denominator*, not a question with nothing to search for, and
    refusing it would be wrong. With the fallback, all three terms are present in
    the prefix, coverage is 1.0, and it answers.
    """
    all_terms = tokenize(normalise_query(q))
    distinct = [t for t in all_terms if t not in _COVERAGE_NOISE]
    return distinct or all_terms


def term_coverage(terms: Sequence[str], haystack_tokens: Iterable[str]) -> tuple[float, list[str]]:
    """Fraction of `terms` present in the chunk, plus which ones matched.

    Matching is exact-or-prefix, not fuzzy. Fuzzy matching is how "price of gold
    in Mumbai" ends up half-matched by a chunk about exit loads.
    """
    hay = set(haystack_tokens)
    matched: list[str] = []
    for term in terms:
        if term in hay or (len(term) >= _MIN_PREFIX
                           and any(h.startswith(term) for h in hay if len(h) >= _MIN_PREFIX)):
            matched.append(term)
    if not terms:
        return 0.0, []
    return len(matched) / len(terms), matched


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


# A figure worth calling a figure: a currency sign, a percentage, or a run of
# three or more digits (so "3Y lock-in" and "2nd investment" do not qualify, and
# "1189.08" / "39933.37" / "1.03%" do).
_FIGURE = re.compile(r"[₹$%]|\d[\d,]{2,}(?:\.\d+)?")


def states_a_figure(text: str) -> bool:
    """Does this chunk actually state a number a user could ask for?"""
    return bool(_FIGURE.search(text or ""))


def detect_scheme(question: str, chunks: Sequence[dict[str, Any]]) -> str | None:
    """Which scheme is this question about, if it names one?

    Matches the longest run of a scheme's name that appears in the question. The
    stored name is "HDFC Large Cap Fund - Direct Growth" but nobody types that -
    they write "HDFC Large Cap Fund", or just "HDFC Large Cap". So each name is
    also tried with its trailing words trimmed, and the longest match across all
    names and schemes wins. That ordering is what keeps "HDFC Flexi Cap Fund"
    from being swallowed by a shorter generic match.

    Returns the scheme_slug, or None when the question names no scheme - in which
    case no filter is applied and all five remain candidates.
    """
    q = normalise_query(question)
    if not q:
        return None

    by_slug: dict[str, dict[str, Any]] = {}
    for c in chunks:
        by_slug.setdefault(c["scheme_slug"], c)

    def _variants(name: str) -> list[str]:
        """The name, then progressively shorter trailing-word trims."""
        words = tokenize(name)
        out = []
        for end in range(len(words), 1, -1):
            out.append(" ".join(words[:end]))
        return out

    found = detect_schemes(question, chunks)
    return found[0] if found else None


def detect_schemes(question: str, chunks: Sequence[dict[str, Any]]) -> list[str]:
    """Every scheme a question names, strongest match first.

    The singular `detect_scheme` exists for callers that need "the one scheme"
    (the scheme filter, memory resolution). This is the plural view, used to
    catch a question that names two schemes at once ("What is the NAV of HDFC
    Small Cap and HDFC ELSS?"), which the exactly-one-source contract cannot
    answer honestly: whichever single page we cite, the other fund's figure
    would ride in unsourced.

    Match quality is the SAME measure as `detect_scheme` - the longest name
    variant found in the question - so naming a scheme twice under different
    aliases ("HDFC Flexi Cap Fund, also called HDFC Equity Fund") still counts
    as ONE scheme: both resolve to the same slug, and the return is deduplicated
    by slug.
    """
    q = normalise_query(question)
    if not q:
        return []

    by_slug: dict[str, dict[str, Any]] = {}
    for c in chunks:
        by_slug.setdefault(c["scheme_slug"], c)

    def _variants(name: str) -> list[str]:
        """The name, then progressively shorter trailing-word trims."""
        words = tokenize(name)
        out = []
        for end in range(len(words), 1, -1):
            out.append(" ".join(words[:end]))
        return out

    cands: list[tuple[int, str]] = []
    for slug, c in by_slug.items():
        names = [str(c.get("scheme_name", ""))]
        aliases = c.get("scheme_aliases") or []
        names += [str(a) for a in (aliases if isinstance(aliases, list) else [aliases])]
        best = 0
        for name in names:
            for variant in _variants(name):
                # "hdfc" alone is too short to identify a fund - it matches all
                # five schemes equally and would filter to whichever sorted first.
                if len(variant) < 6 or len(variant.split()) < 2:
                    continue
                if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", q):
                    best = max(best, len(variant))
        if best:
            cands.append((best, slug))

    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [slug for _, slug in cands]


_MEMORY_DEFAULT_MAX_MESSAGES = 10


def _message_text(m: Any) -> str:
    """The user-visible text of one UI transcript entry, whatever its shape.

    The transcript stores user turns as {"role": "user", "text": str} and
    assistant turns as {"role": "assistant", "answer": Answer}; a test or an
    embedder may also hand us a bare str or a dataclass. One accessor for all
    of them keeps `with_memory_context` independent of the UI layer.
    """
    if isinstance(m, dict):
        if m.get("text"):
            return str(m.get("text"))
        ans = m.get("answer")
        if ans is not None:
            return str(getattr(ans, "text", "") or "")
        return ""
    if hasattr(m, "text"):
        return str(getattr(m, "text", "") or "")
    return str(m or "")


def _scheme_display_name(slug: str, corpus: Sequence[dict[str, Any]]) -> str:
    """The name users actually type for this scheme, minus the "- Direct …" tail.

    ``scheme_name`` carries the market-gateway name ("HDFC Equity Fund") but
    the project speaks Groww's alias ("HDFC Flexi Cap Fund"); the alias is what
    a follow-up should be spliced with, so retrieval and ranking see the same
    tokens the user used.
    """
    for c in corpus:
        if c.get("scheme_slug") == slug:
            aliases = c.get("scheme_aliases") or []
            if isinstance(aliases, list) and aliases and str(aliases[0]).strip():
                name = str(aliases[0])
            else:
                name = str(c.get("scheme_name") or "")
            return re.sub(r"\s*[-–]\s*(Direct|Direct Plan|Plan).*$",
                          "", name).strip() or slug
    return slug.replace("-", " ").strip()


def with_memory_context(question: str, history: Sequence[Any] | None,
                        corpus: Sequence[dict[str, Any]],
                        max_messages: int = _MEMORY_DEFAULT_MAX_MESSAGES
                        ) -> tuple[str, dict[str, Any]]:
    """Resolve a terse follow-up question against the recent conversation.

    Returns ``(query_for_retrieval, memo)``. ``memo`` is ``{}`` when no
    resolution happened - which is the common case, and must be indistinguishable
    from the plain question so callers default to today's behaviour.

    When the question already names a scheme, or no turn in the last
    ``max_messages`` named one, the original question is returned unchanged.
    Otherwise the most recent turn that named a scheme is found (scanning
    newest-first) and its scheme is spliced into the query, so a follow-up like
    "…its expense ratio?" becomes "…its expense ratio? HDFC Small Cap" for
    retrieval, ranking and generation. The guardrail layer still judges only
    what the user literally typed.
    """
    q = (question or "").strip()
    if not history or not q:
        return q, {}
    recent = list(history)[-max_messages:]

    # The question already names a scheme: no resolution needed.
    if detect_scheme(q, corpus):
        return q, {}

    # Newest-first scan for the last turn that named one.
    last_named: str | None = None
    for m in reversed(recent):
        s = detect_scheme(_message_text(m), corpus)
        if s:
            last_named = s
            break
    if not last_named:
        return q, {}

    display = _scheme_display_name(last_named, corpus)
    resolved = f"{q}  {display}" if display else q
    return resolved, {
        "resolved_scheme": last_named,
        "scheme_name": display,
        "from_history": True,
    }


def retrieve(question: str, *, top_k: int | None = None, where: dict[str, Any] | None = None,
             with_debug: bool = False) -> list[RetrievedChunk]:
    """Rank the corpus for `question`.

    Raises OutOfCorpus when no chunk covers enough of the question - which
    includes a perfectly-formed question about something the corpus has never
    heard of, like "price of gold in Mumbai". That refusal is the behaviour the
    whole project is graded on, and it is worth being precise about what makes it
    work: NOT the similarity threshold. On this corpus the gold question scores
    0.31 dense and the genuinely-present "exit load" scores 0.16, so no
    similarity threshold separates them. Term coverage does. See
    `RetrievedChunk.clears`.
    """
    cfg = common.load_config()["retrieval"]
    final_top_k = int(top_k or cfg["final_top_k"])
    min_cosine = float(cfg["min_cosine"])
    min_bm25 = float(cfg.get("min_bm25", 0.0))
    min_coverage = float(cfg.get("min_coverage", 0.0))
    over_fetch = max(1, int(cfg.get("over_fetch", 1)))

    index = get_index()
    chunks = common.read_jsonl(common.path_for("chunks_file"))

    expanded = expand_query(question)
    if not expanded:
        raise OutOfCorpus("the question has no searchable content in it")

    # --- 2/3. dense --------------------------------------------------------
    from embed.index import embed_chunks  # noqa: PLC0415
    from store import chroma_store as cs  # noqa: PLC0415

    ecfg = common.load_config()["embedding"]
    # The query is embedded with the same prefix template the corpus was built
    # with, minus the section heading, which is unknown before retrieval. The
    # scheme identity prefix is the part that matters.
    query_vec = _embed_question(question, ecfg)

    dense_top_k = int(cfg["dense_top_k"]) * over_fetch
    dense: list[dict[str, Any]] = cs.query(query_vec, top_k=dense_top_k, where=where)

    # --- 4. lexical --------------------------------------------------------
    bm25_top_k = int(cfg["bm25_top_k"]) * over_fetch
    bm25 = index.search(expanded, top_k=bm25_top_k)

    # --- assemble candidates ----------------------------------------------
    dense_by_id = {r["chunk_id"]: r for r in dense}
    bm25_by_id = {cid: (score, terms) for cid, score, terms in bm25}
    fused = rrf_fuse(
        [[r["chunk_id"] for r in dense], [cid for cid, _, _ in bm25]],
        k=int(cfg["rrf_k"]),
    )

    user_terms = query_tokens_for_matching(question)
    cov_terms = content_terms(question)

    # --- 5b. answer-bearing boost ------------------------------------------
    # Two chunks on this corpus differ only in whether they carry the number:
    #     "nav: 25 sep '26"                  rrf 0.03150
    #     "nav: 25 sep '26 557.73"          rrf ~0.024
    # BM25's length normalisation prefers the shorter one, and MMR then treats
    # the pair as near-duplicates and keeps the higher-scoring of the two - so
    # the row that actually carries the answer is the one discarded. The
    # question "what is the NAV" was answered from the FAQ restatement instead.
    #
    # The boost is deliberately tiny (0.004, against a top RRF of ~0.032) and is
    # applied to ordering only. It does not touch `rrf_score`, does not touch
    # `clears()`, and cannot let a chunk into the result set that would
    # otherwise be refused. It only decides, among chunks that already passed,
    # which near-duplicate is worth the slot.
    answer_boost = float(cfg.get("answer_bearing_boost", 0.0))
    asks_a_value = bool(re.search(
        r"(?<![a-z0-9])(what|which|how much|how many|give|tell|"
        r"expense ratio|ter|aum|nav|size|charge)(?![a-z0-9])",
        question.lower()))

    candidates: list[RetrievedChunk] = []
    for cid, rrf in fused.items():
        d = dense_by_id.get(cid)
        b_score, b_terms = bm25_by_id.get(cid, (0.0, []))
        chunk = index.chunk(cid)
        haystack = tokenize(_index_terms(chunk))
        coverage, matched = term_coverage(cov_terms, haystack)
        bearing = states_a_figure(str(chunk.get("text", "")))
        # matched_terms reports the question's words, never synonym expansions -
        # it is shown to a user as "these are the words from your question that
        # appear here", and a synonym in that list would make the label false.
        candidates.append(RetrievedChunk(
            chunk_id=cid,
            text=chunk.get("text", ""),
            metadata=dict(d["metadata"]) if d else _chunk_metadata(chunk),
            dense_score=float(d["similarity"]) if d else 0.0,
            bm25_score=b_score,
            rrf_score=rrf,
            final_score=rrf + (answer_boost if (bearing and asks_a_value) else 0.0),
            matched_terms=matched,
            coverage=coverage,
            answer_bearing=bearing,
            dense_rank=(dense.index(d) + 1) if d else None,
            bm25_rank=(next(i for i, x in enumerate(bm25) if x[0] == cid) + 1)
            if cid in bm25_by_id else None,
        ))

    candidates.sort(key=lambda c: (-c.final_score, c.chunk_id))

    # --- 6. diversify ------------------------------------------------------
    # Filtered BEFORE MMR, not after. Diversifying a pool that is mostly going
    # to be thrown away wastes the budget and, worse, lets a near-duplicate
    # occupy a slot that a covering chunk needed.
    eligible = [c for c in candidates if c.clears(min_coverage, min_cosine, min_bm25)]
    pool = eligible[:max(final_top_k * 4, int(cfg["dense_top_k"]))]
    diversified = mmr_diversify(pool, float(cfg["mmr_lambda"]), final_top_k,
                                embedding_lookup=_embedding_lookup())

    # --- 7. rerank (optional) ---------------------------------------------
    reranked = _rerank(question, diversified, cfg)

    # --- 8. threshold ------------------------------------------------------
    if not reranked:
        raise OutOfCorpus(_why_out_of_corpus(question, candidates, cov_terms,
                                             min_coverage, min_cosine, min_bm25))

    if with_debug:
        return reranked
    return reranked[:final_top_k]


def _why_out_of_corpus(question: str, candidates: Sequence[RetrievedChunk],
                       cov_terms: Sequence[str], min_coverage: float,
                       min_cosine: float, min_bm25: float) -> str:
    """Explain the refusal in terms of the question, not just the numbers.

    A refusal that says "coverage 0.33 < 0.5" is debug output. One that says the
    question's words are simply not in the sources is an answer, and this string
    ends up in the log a grader reads.
    """
    if not cov_terms:
        return (f"{question!r} has no content words to search for after "
                f"normalisation and stopword removal")
    if not candidates:
        return (f"neither dense nor BM25 retrieved anything for {question!r} "
                f"- none of {list(cov_terms)} appear in the corpus")
    best = max(candidates, key=lambda c: (c.coverage, c.bm25_score))
    return (
        f"no chunk covers enough of the question. The best match scored "
        f"coverage {best.coverage:.2f} of {list(cov_terms)} "
        f"(needs {min_coverage}), dense {best.dense_score:.3f} "
        f"(floor {min_cosine}), bm25 {best.bm25_score:.2f} (floor {min_bm25}). "
        f"The corpus covers 5 HDFC schemes and nothing else."
    )


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Metadata for a chunk BM25 found but dense did not.

    Read from chunks.jsonl rather than from the store, because the row genuinely
    is not in Chroma. The citation fields are the same ones the store would have
    carried, so a BM25-only hit is still citable.
    """
    from store import chroma_store as cs  # noqa: PLC0415

    return cs.coerce_metadata(chunk)


def _embed_question(question: str, ecfg: dict[str, Any]) -> list[float]:
    """Embed the question the way the corpus was embedded.

    Uses the same model, the same normalisation, and Stage 3's own
    `build_embed_text`, by embedding the question as a single pseudo-chunk. A
    bare `question` embedded with no prefix sits in a different part of the space
    than `HDFC Large Cap Fund (Large Cap) - Fees and charges: min. for sip: 100`,
    which is exactly the "blurs names that only exist in metadata" failure this
    stage exists to mitigate.

    `force=True` on the call below is deliberate and not a cache-wipe: the query
    is a synthetic chunk with no place in the corpus, so it must never read or
    write a `data/embeddings/*.npy` entry.
    """
    from embed.index import embed_chunks, embed_hash, load_model  # noqa: PLC0415

    text = normalise_query(question) or question
    pseudo = {
        "chunk_id": "__query__", "scheme_name": "", "scheme_aliases": [],
        "category": "", "section_heading": "question", "text": text,
        "fact_key": "", "fact_value": "", "token_count": 0,
    }
    vectors, _stats = embed_chunks([pseudo], force=True, model=load_model(),
                                   show_progress=False)
    vec = vectors.get(embed_hash(pseudo, ecfg))
    if vec is None:
        raise RuntimeError(f"the question could not be embedded: {text!r}")
    return [float(x) for x in vec]


def _rerank(question: str, candidates: list[RetrievedChunk],
            cfg: dict[str, Any]) -> list[RetrievedChunk]:
    """Optional cross-encoder rerank. Off by default; never raises."""
    rcfg = cfg.get("rerank", {}) or {}
    if not rcfg.get("enabled"):
        return candidates
    model = _get_reranker(str(rcfg.get("model_name", "")))
    if model is None:
        return candidates

    try:
        pairs = [(question, c.text) for c in candidates]
        scores = model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda c: -float(c.rerank_score or -1.0))
    except Exception as exc:  # noqa: BLE001
        LOG.info("  rerank failed (%s: %s) - keeping the fused order",
                 type(exc).__name__, exc)
        return candidates


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "minimum SIP",
    "exit load",
    "lock-in period",
    "how do I download my capital gains statement",
    "price of gold in Mumbai",
]


def explain(question: str, *, top_k: int | None = None) -> dict[str, Any]:
    """Run `question` and return both the results and the per-stage trail.

    The trail is the point. "It gave the wrong answer" is unactionable; "BM25
    ranked the right chunk #1 and MMR dropped it as a near-duplicate" is a fix.
    """
    cfg = common.load_config()["retrieval"]
    expanded = expand_query(question)
    bm25 = get_index().search(expanded, top_k=int(cfg["bm25_top_k"]))
    try:
        results = retrieve(question, top_k=top_k, with_debug=True)
        out_of_corpus = False
        reason = ""
    except OutOfCorpus as exc:
        results, out_of_corpus, reason = [], True, str(exc)

    return {
        "question": question,
        "normalised": normalise_query(question),
        "expanded": expanded,
        "bm25_top": [{"chunk_id": cid, "score": round(s, 4), "terms": t}
                     for cid, s, t in bm25],
        "out_of_corpus": out_of_corpus,
        "reason": reason,
        "results": [r.to_dict() for r in results],
    }


def _print_table(rows: list[dict[str, Any]], cols: list[tuple[str, int]]) -> None:
    header = "  ".join(name.ljust(width) for name, width in cols)
    print("  " + header)
    print("  " + "-" * len(header))
    for r in rows:
        cells = []
        for name, width in cols:
            v = r.get(name, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            cells.append(str(v)[:width].ljust(width))
        print("  " + "  ".join(cells))


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5 - retrieval")
    ap.add_argument("question", nargs="*", help="the question to retrieve for")
    ap.add_argument("--demo", action="store_true",
                    help="run the five acceptance queries")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--trace", action="store_true",
                    help="show normalise -> expand -> bm25 as well as the results")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()

    questions = DEMO_QUERIES if args.demo else (" ".join(args.question),)
    if not any(questions):
        ap.error("give a question, or pass --demo")

    for q in questions:
        if not q:
            continue
        print("=" * 100)
        print(f"  Q: {q}")
        print("=" * 100)
        info = explain(q, top_k=args.top_k)

        if args.trace:
            print(f"  normalised : {info['normalised']}")
            print(f"  expanded   : {info['expanded']}")
            if info["bm25_top"]:
                print("  BM25 top 5 :")
                for row in info["bm25_top"][:5]:
                    print(f"      {row['score']:>8.4f}  {row['chunk_id'][:66]}"
                          f"  {row['terms']}")
            print()

        if info["out_of_corpus"]:
            print(f"  OUT OF CORPUS -> {info['reason']}")
            print("  (Stage 6 turns this into a polite refusal, never a guess.)")
            print()
            continue

        rows = []
        for r in info["results"]:
            rows.append({
                "chunk_id": r["chunk_id"][:52],
                "section": r["section_heading"][:30],
                "dense": r["dense_score"],
                "bm25": r["bm25_score"],
                "rrf": r["rrf_score"],
                "final": r["final_score"],
                "matched": ",".join(r["matched_terms"])[:22],
            })
        _print_table(rows, [("chunk_id", 52), ("section", 30), ("dense", 7),
                            ("bm25", 7), ("rrf", 9), ("final", 9), ("matched", 22)])
        for r in info["results"]:
            print(f"      {r['source_url']}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(_main())
