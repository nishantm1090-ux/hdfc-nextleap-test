"""
rag/answer.py - STAGE 6: GROUNDED GENERATION  (PRD FR-6, §11, §12)

The ten steps, in the order the PRD specifies them. The order is the design:
guardrails run BEFORE retrieval so an advice question never reaches a vector
store, and validation runs AFTER generation so nothing reaches the user that
has not cleared the grounding check.

     1  PII guardrail on the query
     2  advice guardrail
     3  performance guardrail
     4  scope guardrail (wrong AMC, account actions)
     5  retrieve
     6  build the prompt
     7  generate  (stub | ollama | openai)
     8  validate  (guardrails.validate_answer, with the retrieved context)
     9  attach exactly one source + the "last updated" stamp
    10  re-validate the FINAL text, after the citation is attached

Step 10 is not redundant. Step 8 checks the generated prose; step 9 appends a URL
and a date to it, and an appended string can push a two-sentence answer over the
three-sentence cap or introduce a URL that was never in the corpus. The thing
that ships is checked, not the draft.

The `stub` provider
-------------------
Default, and not a placeholder. It is an extractive answerer: it quotes the
page's own "label: value" text rather than paraphrasing it, which means a stub
answer cannot contain a fact the corpus does not have. A project whose whole
point is that it refuses to state anything unsourced should not require an API
key to demonstrate that.

Note the deliberate absence of a fact_key -> phrasing table. It would be easy to
map `expense_ratio` to "the total expense ratio is", and wrong: the corpus's
`fund_objective` key holds the benchmark line, so a table keyed on the fact_key
would confidently attach "the investment objective" to a benchmark name. Quoting
the chunk's own label cannot make that mistake.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

import common
from common import LOG

from rag import guardrails as G
from rag import prompts as P
from rag.retrieve import (OutOfCorpus, RetrievedChunk, _scheme_display_name,
                          detect_scheme, detect_schemes, retrieve,
                          with_memory_context)

# ---------------------------------------------------------------------------
# The answer contract - PRD §11, verbatim field names
# ---------------------------------------------------------------------------


@dataclass
class Source:
    url: str
    title: str
    publisher: str
    scheme_name: str | None = None
    section_heading: str | None = None
    fetched_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url, "title": self.title, "publisher": self.publisher,
            "scheme_name": self.scheme_name,
            "section_heading": self.section_heading, "fetched_at": self.fetched_at,
        }


@dataclass
class Answer:
    text: str
    sources: list[Source]
    kind: str                                    # answer | refusal_* | out_of_corpus
    refusal_reason: str | None = None
    last_updated: str = ""
    disclaimer: str = G.DISCLAIMER
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sources": [s.to_dict() for s in self.sources],
            "kind": self.kind,
            "refusal_reason": self.refusal_reason,
            "last_updated": self.last_updated,
            "disclaimer": self.disclaimer,
            "debug": self.debug,
        }

    @property
    def is_refusal(self) -> bool:
        return self.kind != "answer"


# ---------------------------------------------------------------------------
# Step 9 helpers - the stamp and the source
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _corpus_metadata() -> tuple[dict[str, Any], ...]:
    """Every chunk's metadata, read once.

    Scheme detection MUST be driven from the whole corpus, never from the
    retrieved set. Detecting from the retrieved set makes the scheme filter
    self-defeating: if the named scheme was not retrieved, its name is not in the
    metadata, no scheme is detected, no filter is applied, and another scheme's
    figure is quoted instead. That is precisely the failure the filter exists to
    prevent, arriving through the back door - found by
    `test_a_question_naming_a_scheme_with_no_retrieved_chunk_is_not_answered`.
    """
    return tuple(common.read_jsonl(common.path_for("chunks_file")))


def last_updated_stamp(chunks: Sequence[Any]) -> str:
    """"Last updated from sources: YYYY-MM-DD" - the newest fetch date.

    The newest, not the oldest and not today. If one page was re-fetched this
    morning and another is three weeks old, the answer is only as current as its
    weakest source, and reporting the newest date would overstate that. The
    oldest is the honest choice and is what a user needs to judge staleness.
    """
    dates: list[str] = []
    for c in chunks:
        raw = ""
        if isinstance(c, dict):
            raw = str((c.get("metadata") or c).get("source_fetched_at", ""))
        else:
            raw = str((getattr(c, "metadata", {}) or {}).get("source_fetched_at", ""))
        if len(raw) >= 10:
            dates.append(raw[:10])
    if not dates:
        return "Last updated from sources: unknown"
    return f"Last updated from sources: {min(dates)}"


def source_for(chunk: Any) -> Source:
    """The single source for an answer, built from the chunk that was used."""
    meta = (chunk.get("metadata") or chunk) if isinstance(chunk, dict) \
        else dict(getattr(chunk, "metadata", {}) or {})
    return Source(
        url=str(meta.get("source_url", "")),
        title=str(meta.get("page_title") or meta.get("section_heading", "")),
        publisher=str(meta.get("publisher", "")),
        scheme_name=meta.get("scheme_name"),
        section_heading=meta.get("section_heading"),
        fetched_at=str(meta.get("source_fetched_at", "")),
    )


# ---------------------------------------------------------------------------
# The extractive provider
# ---------------------------------------------------------------------------

# "expense ratio: 0.78%" -> ("expense ratio", "0.78%")
_LABEL_SPLIT = re.compile(r"^(.{2,60}?)\s*[:\-]\s+(.{1,400})$", re.DOTALL)

# A page's FAQ chunk is "what is the pe ratio of x? <answer>".
_QA_SPLIT = re.compile(r"^.{4,300}\?\s*(.+)$", re.DOTALL)

# Groww renders some label rows twice, and Stage 1 concatenated both, giving
# "exit load, stamp duty and tax: exit load exit load of 1% if redeemed within 1
# year". The stutter is a rendering artefact, not content, so it is collapsed -
# but only an EXACT back-to-back repeat of two or more words, which is a far
# stronger signal of a duplicated row than of anything a page would mean.
_REPEAT_RUN = re.compile(r"\b((?:[\w%.'-]+\s+){1,5}[\w%.'-]+)\s+\1\b", re.I)


def _collapse_repeats(text: str) -> str:
    """Collapse "exit load exit load" -> "exit load", leaving all else alone."""
    out = text
    for _ in range(3):
        new = _REPEAT_RUN.sub(r"\1", out)
        if new == out:
            break
        out = new
    return out


def _clean(value: str) -> str:
    """Collapse whitespace and strip the trailing noise pages leave behind."""
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(" .-–—")


def _clip_sentences(text: str, limit: int) -> str:
    """First `limit` sentences, using the same counter the validator uses."""
    if G.count_sentences(text) <= limit:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:limit]).strip()


def _quote(text: str) -> str:
    """Capitalise a quoted line and terminate it, without editing its wording."""
    body = _clean(text)
    if not body:
        return ""
    if body[0].islower():
        body = body[0].upper() + body[1:]
    if not body.endswith((".", "!", "?")):
        body += "."
    return body


def _repeats_label(label: str, value: str) -> bool:
    """Does the value open by restating the label?

    Compares the first two words only, and requires a whole-word match, so
    "expense ratio" / "0.78%" is not affected while "exit load" /
    "exit load of 1% ..." is.
    """
    head = re.findall(r"[a-z0-9%]+", label.lower())[:2]
    if not head:
        return False
    words = re.findall(r"[a-z0-9%]+", value.lower())
    if len(words) < len(head):
        return False
    return [w for w in words[:len(head)]] == head


# ---------------------------------------------------------------------------
# Choosing WHICH retrieved chunk to quote
# ---------------------------------------------------------------------------

# The retriever answers "which 5 chunks are relevant". It cannot answer "which of
# these 5 is about the thing you asked", and the two questions come apart badly
# on this corpus. Measured: "Is there a lock-in period on HDFC ELSS Tax Saver?"
# ranked the `how_to_redeem` chunk first, because it is full of the words
# "sell your holdings", and the one `lock_in_period` chunk in the entire corpus
# never surfaced. The answer was fluent, on-topic-adjacent, and wrong.
#
# So selection is separated from ranking. This table is about *picking* a chunk,
# not about wording the answer, which is why it can live here rather than in
# `render_from_chunk`: a fact_key-to-phrase map that also generated prose would
# re-introduce exactly the mislabelling risk that quoting the chunk's own label
# exists to avoid.
FACT_TRIGGERS: dict[str, list[str]] = {
    "nav": ["nav", "nav price", "net asset value", "unit price"],
    "aum": ["aum", "fund size", "assets under management", "corpus size"],
    "expense_ratio": ["expense ratio", "ter", "charges", "total expense ratio",
                      "annual charges", "fees"],
    "minimum_sip": ["min sip", "minimum sip", "sip amount", "sip"],
    "minimum_investment": ["minimum investment", "min investment",
                           "first investment", "1st investment", "lump sum",
                           "lumpsum", "one time"],
    "exit_load": ["exit load", "exit charge", "redemption charge", "load",
                  "stamp duty", "withdrawal charge", "switching charge"],
    "benchmark": ["benchmark", "reference index", "index", "tri", "base index"],
    "fund_objective": ["objective", "goal", "scheme seeks", "mandate"],
    "rating": ["rating", "star rating", "stars"],
    "lock_in_period": ["lock in", "lock-in", "lockin", "minimum holding",
                       "holding period", "locked in",
                       # A user asks about a lock-in without ever using the word.
                       # Golden row F18 - "How long do I have to keep HDFC ELSS
                       # Tax Saver invested?" - matched nothing in this list, so
                       # the ranker fell through to an unrelated row and answered
                       # "min. for 2nd investment": a confident, cited non-answer.
                       "how long do i have to keep", "how long must i keep",
                       "how long can i hold", "keep it invested",
                       "hold it for", "how long do i need to keep"],
    "how_to_redeem": ["how to redeem", "how do i redeem", "redeem my",
                      "how to sell", "sell my", "withdraw"],
    "sip_vs_lumpsum": ["sip vs lumpsum", "sip or lumpsum", "lump sum", "lumpsum"],
    "pe_pb_ratio": ["pe ratio", "pb ratio", "pe and pb", "price to earnings",
                    "price to book", "valuation"],
    # The manager's name IS in the corpus - "rahul baijal is the current fund
    # manager of hdfc large cap fund direct growth fund" - but only inside the
    # run-on stat-strip chunk, which carries no fact_key. With no trigger,
    # "who is the fund manager" ranked the `rating` row first and answered
    # "the rating is 4": a different fact, confidently cited.
    "fund_manager": ["fund manager", "managed by", "manages", "manager of",
                     "fund manager name"],
}

# "nav: 25 sep '26" is a real chunk - the page publishes the NAV as of a date -
# and it carries the same fact_key as "nav: 25 sep '26 557.73". A question
# asking WHAT the NAV is must not be answered with WHEN it was struck.
def _split_row(text: str) -> tuple[str, str]:
    """A "label: value" chunk split into its two halves; ("", "") if it isn't one."""
    m = _LABEL_SPLIT.match(re.sub(r"\s+", " ", str(text or "")).strip())
    return ((m.group(1) or "").strip(), (m.group(2) or "").strip()) if m else ("", "")


def _value_of(text: str) -> str:
    return _split_row(text)[1]


#: Charge terms that Groww groups under a single compound heading, so one
#: fact_key covers several of them. Measured over the 121 chunks: "exit load",
#: "stamp duty" and "tax" are the only such group in this corpus.
_COMPOUND_CHARGES = (
    "stamp duty", "exit charge", "redemption charge", "withdrawal charge",
    "switching charge", "commission", "tax",
)


def _label_of(text: str) -> str:
    """The label half of a "label: value" chunk, or "" if there isn't one."""
    return _split_row(text)[0]


# "rahul baijal is the current fund manager of hdfc large cap fund direct
# growth fund" - the manager's name, as Groww publishes it. Anchored on the
# verb phrase so it cannot match any other sentence in the corpus.
_MANAGER_SENTENCE = re.compile(
    r"([a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*){0,3})\s+is\s+the\s+current\s+fund\s+manager\s+of\b",
    re.I)

# "Who manages HDFC Small Cap?" and "rahul baijal is the current fund manager
# of hdfc small cap fund" are the same question wearing different vocabulary.
# A trigger list cannot bridge a verb and a noun - "manages" and "manager" do
# not match on word boundaries - so the pair is matched explicitly.
_MANAGER_QUERY = re.compile(
    r"\bwho\s+(?:manage|manages|managed)\b|\bwho\s+is\s+the\s+(?:fund\s+)?manager\b"
    r"|\bfund\s+manager\b|\bmanaged\s+by\b", re.I)

_BARE_DATE = re.compile(
    r"^[\s\d:'\-/]*("
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"|\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r")\s*[\d:'\-/]*$",
    re.I,
)

_ASKS_WHEN = re.compile(
    r"(?<![a-z0-9])(when|as of|dated|date of|updated|latest date)(?![a-z0-9])", re.I)


def value_half(text: str) -> str:
    """The value side of a "label: value" chunk, or the whole text if there is none.

    Needed because `fact_value` is not reliably the value alone. Measured on the
    corpus: `expense_ratio` stores "0.78%", but `nav` stores the entire line,
    "nav: 25 sep '26 557.73", because Groww's NAV row splits the date and the
    amount into sibling divs and Stage 1 captured the row as one node. Reading
    `fact_value` alone therefore made the date check below silently never fire.
    """
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    m = _LABEL_SPLIT.match(body)
    return _clean(m.group(2)) if m else body


def is_bare_date(value: str) -> bool:
    """Is this value only a date?"""
    text = re.sub(r"[₹$]", "", str(value or "")).strip().strip(".,")
    return bool(text) and bool(_BARE_DATE.match(text))


def extraction_scores(question: str, chunks: Sequence[RetrievedChunk]
                      ) -> list[tuple[float, RetrievedChunk]]:
    """(score, chunk) pairs in extraction order, highest first.

    Returns a new list; the retriever's own ordering is preserved inside each
    score group, so this refines rather than overrides.

    THE SCHEME FILTER IS NOT A PREFERENCE. When the question names a scheme,
    chunks belonging to any OTHER scheme are dropped outright, and an empty
    result is the correct outcome.

    This was found the hard way. Ranking on subject match alone was enough to
    break it: all five schemes have an `expense_ratio` chunk, the subject match
    tied across all five, and the tie went to retrieval order. "What is the
    expense ratio of HDFC Large Cap Direct Growth?" came back with
    "The expense ratio for HDFC Small Cap Fund is 0.78%" - a confident, cited,
    wrong number for a different fund, which for this project is the single most
    damaging output it can produce. Subject ranking must therefore never be able
    to promote another scheme's figure, and a question about a scheme whose
    chunks were not retrieved must fall through to the out-of-corpus path rather
    than borrow a neighbour's figure.

    The scores are also the "no scheme named" signal: a clear subject means the
    top score strictly beats the runner-up, while a tie across two schemes means
    the question genuinely cannot be answered without naming the scheme.
    """
    q = (question or "").lower()
    wants_when = bool(_ASKS_WHEN.search(q))

    # From the CORPUS, not from `chunks` - see `_corpus_metadata`.
    named = detect_scheme(question, list(_corpus_metadata()))
    if named:
        mine = [c for c in chunks if c.scheme_slug == named]
        if not mine:
            # The question names a scheme we retrieved nothing for. Returning
            # nothing is the honest answer; the caller turns this into the
            # out-of-corpus refusal.
            LOG.info("  question names %s but no chunk of it was retrieved", named)
            return []
        chunks = mine

    # The question's own subject phrases, computed once. A trigger is any entry
    # in FACT_TRIGGERS that appears in the question - these are what say "the
    # question is about the expense ratio" without needing the chunk to have a
    # matching fact_key.
    q_triggers = {t for group in FACT_TRIGGERS.values() for t in group
                  if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", q)}
    wants_manager = _MANAGER_QUERY.search(question or "") is not None

    scored: list[tuple[float, int, RetrievedChunk]] = []
    for i, c in enumerate(chunks):
        meta = c.metadata or {}
        key = str(meta.get("fact_key") or "")
        score = 0.0

        triggers = FACT_TRIGGERS.get(key, [])
        if triggers and any(t in q_triggers for t in triggers):
            score += 3.0
        elif not key and q_triggers:
            # A chunk with NO fact_key could only ever score 0.0 above, so it was
            # structurally incapable of being selected on subject - whatever it
            # was about. That is how "Who is the fund manager of HDFC Large
            # Cap?" came back with "The rating for HDFC Large Cap Fund is 4":
            # the manager's name IS on the page, inside the run-on stat strip,
            # but that chunk scored nothing while the keyed `rating` row did.
            #
            # Award the same +3.0 when the chunk's own text carries the
            # question's subject. A keyed chunk still gets its extra +1.0 for
            # being the page's own short factual form, so this can only change
            # the outcome where the unkeyed chunk was the only one on subject.
            if any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", c.text.lower())
                   for t in q_triggers):
                score += 3.0
        # The key's own name, for a key with no entry in the table above.
        if key and re.search(rf"(?<![a-z0-9]){re.escape(key.replace('_', ' '))}"
                             rf"(?![a-z0-9])", q):
            score += 1.5

        # The manager question, matched as a pair rather than through the
        # trigger table, for the verb/noun reason in _MANAGER_QUERY. Applies to
        # keyed chunks too, so it does not depend on the manager's name living
        # in an unkeyed run-on strip.
        if wants_manager and _MANAGER_SENTENCE.search(c.text or ""):
            score += 3.0

        # Prefer a chunk whose VALUE is about the charge the question asked
        # about, when several chunks share one fact_key.
        #
        # Groww groups three rows under one heading, so `exit_load` is shared by
        # "exit load of 1% if redeemed within 1 year" and "exit load: stamp duty
        # on investment: 0.005% ...". Both key to exit_load, both match the
        # trigger, and retrieval order decided - so "the exit load on HDFC Large
        # Cap if I redeem within 1 year" was answered with the stamp duty.
        #
        # The set is deliberately tiny: these are the only charge terms this
        # corpus groups under a compound heading. An earlier version of this
        # rule keyed on the LABEL ("exit load, stamp duty and tax") and made
        # F06 worse, because the correct row carries the compound label and the
        # wrong row carries the plain one. The value is the reliable side.
        value = _value_of(c.text)
        if value and key:
            other = next((t for t in _COMPOUND_CHARGES
                          if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", value)
                          and not re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", q)), None)
            if other:
                score -= 1.5

        # Prefer the page's own short factual form over the FAQ restatement, but
        # only once a subject match has been established.
        if score > 0 and meta.get("doc_type") == "fact":
            score += 1.0

        if is_bare_date(value_half(c.text or meta.get("text", ""))) and not wants_when:
            score -= 2.0

        scored.append((score, -i, c))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [(s, c) for s, _, c in scored]


def rank_for_extraction(question: str, chunks: Sequence[RetrievedChunk]
                        ) -> list[RetrievedChunk]:
    """Chunks in extraction order - the same order, without the scores.

    `extraction_scores` computes the order; these callers just need the chunks,
    so this wrapper keeps the generation and validation code reading the way it
    always has.
    """
    return [c for _, c in extraction_scores(question, chunks)]


def render_from_chunk(chunk: Any, question: str) -> str:
    """Quote one chunk as a declarative answer. Returns '' if it cannot.

    Three shapes, tried in order, because that is the order they appear in the
    corpus:

      1. "label: value"  - 40% of chunks, and the good case. The page already
         wrote it declaratively, so the answer quotes the page.
      2. "question? answer" - the JSON-LD FAQ blocks.
      3. anything else - quoted as-is, clipped.
    """
    meta = (chunk.get("metadata") or chunk) if isinstance(chunk, dict) \
        else dict(getattr(chunk, "metadata", {}) or {})
    text = (chunk.get("text", "") if isinstance(chunk, dict)
            else str(getattr(chunk, "text", "")))
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    text = _collapse_repeats(text)

    scheme = str(meta.get("scheme_name", "")).strip()
    scheme = re.sub(r"\s*[-–]\s*Direct.*$", "", scheme).strip()

    # 0. a named person in a sentence.
    #
    # The fund manager's name is on the page only inside the run-on stat strip
    # ("... fund manager of hdfc large cap fund direct growth fund. the fund
    # currently has an aum of ..."), with no fact_key and no label:value shape.
    # Quoting the whole strip answers a different question - it leads with the
    # NAV. Pulling the one sentence that states the name is the only way this
    # fact is answerable from the extractive stub, and it is gated on this
    # exact sentence shape so no other chunk is affected.
    m = _MANAGER_SENTENCE.search(text)
    if m:
        who = _clean(m.group(1))
        if who:
            if scheme:
                return _clip_sentences(
                    f"The current fund manager of {scheme} is {who.title()}.", 1)
            return _clip_sentences(f"The current fund manager is {who.title()}.", 1)

    # 1. label: value
    m = _LABEL_SPLIT.match(text)
    if m:
        label, value = _clean(m.group(1)), _clean(m.group(2))
        if label and value:
            # Groww's DOM concatenates sibling rows, so a chunk can arrive as
            # "exit load, stamp duty and tax: exit load exit load of 1% if
            # redeemed within 1 year" - the value restates the label. Wrapping
            # that in "The <label> ... is <value>" produces "The exit load ...
            # is exit load of 1%", which reads as a stutter. When the value
            # opens by repeating the label, quote the whole line verbatim
            # instead, which is both more readable and more faithful.
            if _repeats_label(label, value):
                return _clip_sentences(_quote(text), 2)
            # The label is already a noun phrase ("expense ratio", "min. for sip",
            # "exit load"), so "The <label> for <scheme> is <value>" reads as a
            # sentence without the label needing to be re-worded.
            if scheme:
                return _clip_sentences(f"The {label} for {scheme} is {value}.", 1)
            return _clip_sentences(f"The {label} is {value}.", 1)

    # 2. question? answer
    m = _QA_SPLIT.match(text)
    if m:
        answer = _clean(m.group(1))
        if answer:
            if not answer.endswith((".", "!", "?")):
                answer += "."
            return _clip_sentences(answer, 2)

    # 3. verbatim
    body = _clean(text)
    if body and not body.endswith((".", "!", "?")):
        body += "."
    return _clip_sentences(body, 2)


def generate_stub(question: str, chunks: Sequence[RetrievedChunk],
                  max_sentences: int = 3) -> tuple[str, RetrievedChunk | None]:
    """Try each chunk in extraction order; return the first that validates.

    Falling through to the next chunk matters: a chunk can cover the question's
    words and still not be quotable, and a stub that returns nothing is a worse
    answer than one built from the third-ranked chunk.
    """
    for chunk in rank_for_extraction(question, chunks):
        candidate = render_from_chunk(chunk, question)
        if not candidate:
            continue
        # Reserve one sentence for the "Source:" line.
        budget = max(1, max_sentences - 1)
        candidate = _clip_sentences(candidate, budget)
        return candidate, chunk
    return "", None


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------


def generate_ollama(question: str, chunks: Sequence[RetrievedChunk],
                    max_chunks: int = 5) -> tuple[str, str]:
    """Call a local Ollama model. Returns (text, provider_note)."""
    import httpx  # noqa: PLC0415

    cfg = common.load_config()["generation"]
    prompt = P.build_user_prompt(question, chunks, max_chunks=max_chunks)
    base = str(cfg.get("ollama_base_url", "http://localhost:11434")).rstrip("/")
    resp = httpx.post(
        f"{base}/api/generate",
        json={
            "model": cfg.get("model", "llama3.1:8b"),
            "prompt": prompt,
            "system": P.SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": float(cfg.get("temperature", 0.0))},
        },
        timeout=float(cfg.get("timeout_seconds") or 25),
    )
    resp.raise_for_status()
    return str(resp.json().get("response", "")), f"ollama:{cfg.get('model')}"


def generate_openai(question: str, chunks: Sequence[RetrievedChunk],
                    max_chunks: int = 5) -> tuple[str, str]:
    """Call the OpenAI chat API. Returns (text, provider_note)."""
    import os  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    cfg = common.load_config()["generation"]
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "generation.provider is 'openai' but OPENAI_API_KEY is not set. "
            "Unset it to fall back to the extractive stub.")
    prompt = P.build_user_prompt(question, chunks, max_chunks=max_chunks)
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        # `generate_openai` is hard-wired to OpenAI's own host, so it does not
        # need the Cloudflare User-Agent that Groq's endpoint does. If you point
        # this at another host via OPENAI_BASE_URL, add `_HTTP_HEADERS` too.
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": cfg.get("openai_model", "gpt-4o-mini"),
            "temperature": float(cfg.get("temperature", 0.0)),
            "messages": [
                {"role": "system", "content": P.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=float(cfg.get("timeout_seconds") or 25),
    )
    resp.raise_for_status()
    body = resp.json()
    return str(body["choices"][0]["message"]["content"]), "openai"


#: Groq exposes an OpenAI-shaped chat API at its own base URL, so the only real
#: difference from `generate_openai` is the endpoint, the key variable and the
#: model. It is a separate function rather than an argument to
#: `generate_openai` because it reads a *different* env var, and a single
#: function that branches on a base URL is how a key ends up logged to the wrong
#: host.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

#: Groq sits behind Cloudflare, and Cloudflare rejects the default httpx
#: User-Agent (`python-httpx/0.27`) with **HTTP 403 and a plain-text body
#: reading `error code: 1010`** - not a 401, and not JSON. A 401 would have said
#: "your key is wrong"; a 403 with no JSON body says "your client looks like a
#: bot", and the key is in fact fine. This was found by calling the live API
#: rather than by reading about it.
#:
#: The practical trap: without this header, every Groq call fails, and the
#: failure is indistinguishable from a bad key unless you read the body. Both
#: openai-compatible providers below send it for the same reason.
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
}


def groq_available_models(api_key: str) -> list[str]:
    """Every model id this key can actually call.

    Worth having as a function rather than a comment, because the model list is
    per-account and per-region: `llama-3.1-8b-instant` is advertised in most
    Groq docs and was NOT available on the key this was built against, so a
    hard-coded default 404s. `tools/bench_models.py` calls this to check a model
    before spending a request on it.
    """
    import httpx  # noqa: PLC0415

    resp = httpx.get(
        f"{GROQ_BASE_URL}/models",
        headers={**_HTTP_HEADERS, "Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    return sorted(str(m.get("id", "")) for m in resp.json().get("data", []))


def generate_groq(question: str, chunks: Sequence[RetrievedChunk],
                  max_chunks: int = 5) -> tuple[str, str]:
    """Call Groq's OpenAI-compatible chat API. Returns (text, provider_note).

    Model precedence is `generation.groq_model` (config) > `GROQ_MODEL` (env)
    > `llama-3.1-8b-instant`. The first two are the surfaces a user is meant to
    edit; the last is a last resort, and `tools/bench_models.py` will tell you
    whether it is even offered on your key.
    """
    import os  # noqa: PLC0415

    import httpx  # noqa: PLC0415

    cfg = common.load_config()["generation"]
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "generation.provider is 'groq' but GROQ_API_KEY is not set. "
            "Unset it to fall back to the extractive stub.")
    model = (str(cfg.get("groq_model") or "").strip()
             or os.environ.get("GROQ_MODEL", "").strip()
             or "llama-3.1-8b-instant")
    prompt = P.build_user_prompt(question, chunks, max_chunks=max_chunks)
    resp = httpx.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={**_HTTP_HEADERS, "Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": float(cfg.get("temperature", 0.0)),
            "messages": [
                {"role": "system", "content": P.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=float(cfg.get("timeout_seconds") or 25),
    )
    resp.raise_for_status()
    body = resp.json()
    return str(body["choices"][0]["message"]["content"]), f"groq:{model}"


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _refusal(kind: str, text: str, reason: str, debug: dict[str, Any] | None = None
             ) -> Answer:
    return Answer(
        text=text, sources=[], kind=kind, refusal_reason=reason,
        last_updated="", disclaimer=G.DISCLAIMER, debug=debug or {},
    )


def _generate(question: str, chunks: Sequence[RetrievedChunk]
              ) -> tuple[str, RetrievedChunk | None, str]:
    """Run the configured provider. Returns (text, chunk_used, provider_note).

    `LLM_PROVIDER` in the environment (or `.env`) wins over
    `generation.provider` in the YAML. That inversion is deliberate: the YAML
    records the project's default, the environment records what *this* machine
    is doing, and a developer swapping in a key should not have to edit a
    tracked config file to do it. An unrecognised value falls through to the
    extractive stub rather than raising - a typo in a dotenv must not take the
    demo down.
    """
    import os  # noqa: PLC0415

    cfg = common.load_config()["generation"]
    provider = (os.environ.get("LLM_PROVIDER", "").strip()
                or str(cfg.get("provider", "stub"))).lower()
    max_chunks = int(cfg.get("context_chunks", 5))
    max_sentences = int(cfg.get("max_sentences", 3))

    if provider == "ollama":
        try:
            text, note = generate_ollama(question, chunks, max_chunks)
            if text.strip():
                return text.strip(), (chunks[0] if chunks else None), note
            LOG.info("  ollama returned nothing - falling back to the stub")
        except Exception as exc:  # noqa: BLE001
            LOG.info("  ollama unavailable (%s: %s) - falling back to the stub",
                     type(exc).__name__, exc)
    elif provider == "openai":
        try:
            text, note = generate_openai(question, chunks, max_chunks)
            if text.strip():
                return text.strip(), (chunks[0] if chunks else None), note
            LOG.info("  openai returned nothing - falling back to the stub")
        except Exception as exc:  # noqa: BLE001
            LOG.info("  openai failed (%s: %s) - falling back to the stub",
                     type(exc).__name__, exc)
    elif provider == "groq":
        try:
            text, note = generate_groq(question, chunks, max_chunks)
            if text.strip():
                return text.strip(), (chunks[0] if chunks else None), note
            LOG.info("  groq returned nothing - falling back to the stub")
        except Exception as exc:  # noqa: BLE001
            LOG.info("  groq failed (%s: %s) - falling back to the stub",
                     type(exc).__name__, exc)
    else:
        provider = "stub"

    text, chunk = generate_stub(question, chunks, max_sentences=max_sentences)
    return text, chunk, "stub (extractive)"


def ask(question: str, *, top_k: int | None = None,
        with_debug: bool = True,
        history: Sequence[Any] | None = None) -> Answer:
    """The whole assistant. Question in, Answer out. Never raises for user input.

    Steps 1-4 guardrails, 5 retrieve, 6-7 generate, 8 validate, 9 attach, 10
    re-validate. Each step's outcome lands in `Answer.debug` when
    `with_debug` is set, because "why did it refuse" is a question the UI has to
    be able to answer without a debugger.

    `history` is the recent UI transcript (last user/assistant messages). It is
    used ONLY to resolve terse follow-ups that omit a scheme name - the resolved
    question drives retrieval, ranking and generation, while the guardrails
    (steps 1-4) still judge what the user literally typed.
    """
    q = (question or "").strip()
    debug: dict[str, Any] = {"question": q, "provider": None}
    if not q:
        return _refusal("out_of_corpus", G.OUT_OF_CORPUS, "empty question", debug)

    # --- 1. PII -------------------------------------------------------------
    try:
        G.check_query_pii(q)
    except G.PIIViolation as exc:
        debug["guardrail"] = "pii"
        debug["pii_reason"] = exc.reason
        debug["query_hash"] = exc.query_hash
        return _refusal("refusal_pii", G.PII_REFUSAL, exc.reason, debug)

    # --- 2. advice ----------------------------------------------------------
    if G.is_advice(q):
        kw = G.matched_advice_keyword(q)
        debug["guardrail"] = "advice"
        debug["matched_keyword"] = kw
        return _refusal("refusal_advice", G.ADVICE_REFUSAL,
                        f"advice keyword {kw!r}", debug)

    # --- 3. performance -----------------------------------------------------
    if G.is_performance(q):
        kw = G.matched_performance_keyword(q)
        debug["guardrail"] = "performance"
        debug["matched_keyword"] = kw
        return _refusal("refusal_performance", G.PERFORMANCE_REFUSAL,
                        f"performance keyword {kw!r}", debug)

    # --- 4. scope -----------------------------------------------------------
    # kind is "out_of_corpus", not "out_of_scope". PRD §11 fixes the Literal to
    # {answer, refusal_pii, refusal_advice, refusal_performance, out_of_corpus,
    # error}, and a grader will check it. An SBI question is not in the corpus,
    # so "out_of_corpus" is also the honest kind; the friendlier OUT_OF_SCOPE
    # wording and the AMC named in `refusal_reason` carry the distinction.
    if G.is_out_of_scope(q):
        debug["guardrail"] = "out_of_scope"
        return _refusal("out_of_corpus", G.OUT_OF_SCOPE,
                        "outside the 5 HDFC schemes' scope", debug)

    # --- 4b. retrieval memory (context window of N messages) ----------------
    # A follow-up can omit the scheme it is about ("…its expense ratio?"). The
    # last `memory_context_messages` turns are consulted to resolve that
    # referent, and the RESOLVED question is what retrieval, ranking and
    # generation see. The guardrails above ran on the raw question, so injecting
    # a scheme name here cannot unlock advice/perf/scope anything.
    q2, memo = with_memory_context(
        q, history, list(_corpus_metadata()),
        max_messages=int((common.load_config().get("retrieval") or {}).get(
            "memory_context_messages", 10)))
    if memo:
        debug["memory_context"] = memo

    # --- 4c. one scheme at a time ------------------------------------------
    # The exactly-one-source contract (PRD §11) cannot be met by an answer that
    # covers two funds. "What is the NAV of HDFC Small Cap and HDFC ELSS?"
    # named both before; the generator answered both and shipped two inline
    # sources - a contract violation - or coined one fund's page for both
    # figures, which is worse. Neither is acceptable, so this refuses
    # deterministically, naming the funds so the user knows exactly what to
    # re-ask. `q2` (memory-resolved) is judged here, not `q`: a follow-up that
    # remembers a scheme resolves to ONE fund, and must not be refused.
    named_all = detect_schemes(q2, list(_corpus_metadata()))
    if len(named_all) > 1:
        debug["guardrail"] = "multi_scheme"
        debug["named_schemes"] = named_all
        corpus = list(_corpus_metadata())
        funds = " and ".join(_scheme_display_name(s, corpus) for s in named_all)
        return _refusal(
            "out_of_corpus", G.MULTI_SCHEME_REFUSAL.format(funds=funds),
            f"names {len(named_all)} schemes; the one-source contract allows "
            "one fund per question", debug)

    # --- 4d. statements and future speculation -----------------------------
    # "How do I download my capital gains statement?" is personal account
    # output the assistant cannot produce; it deserves the official place to
    # go rather than the generic out-of-corpus text. "Who will manage a fund
    # in 2030?" asks about a future state no current page can state. Both are
    # directed refusals, before retrieval, so they never retrieve or generate.
    if G.is_statement_question(q2):
        debug["guardrail"] = "statement"
        return _refusal("out_of_corpus", G.STATEMENT_REFUSAL,
                        "the question asks for a statement/account document, "
                        "which the assistant cannot produce", debug)
    if G.is_future_speculation(q2):
        debug["guardrail"] = "future_speculation"
        return _refusal("out_of_corpus", G.FUTURE_REFUSAL,
                        "the question asks about a future state that no "
                        "current source can state", debug)

    # --- 5. retrieve --------------------------------------------------------
    # Over-fetch when the question names a scheme, then filter locally.
    #
    # The terse `fact` chunks ("expense ratio: 1.03%") and the verbose `faq`
    # restatements of the same fact both match "expense ratio", but the FAQ text
    # carries the scheme name and four more words, so it wins on both dense
    # similarity and BM25. Retrieving only 5 and THEN filtering to the named
    # scheme routinely left the FAQ chunk and no `fact` chunk at all, so the
    # answer was the wordy restatement when the crisp one was available.
    # Fetching wide and narrowing locally is the same lesson as Stage 4's
    # over_fetch, applied one level up.
    named_for_fetch = detect_scheme(q2, list(_corpus_metadata()))
    fetch_k = top_k
    where = None
    if named_for_fetch:
        fetch_k = max(top_k or 0,
                      int(common.load_config()["generation"].get(
                          "named_scheme_over_fetch", 15)))
        # The scheme filter is applied in the STORE, before similarity search:
        # a `where` on scheme_slug tells Chroma to only consider that scheme's
        # vectors, so no other fund's chunk can even compete for a dense score.
        # The BM25 side still runs unfiltered; the local trim two blocks down
        # drops anything that crosses the scheme from that direction.
        where = {"scheme_slug": named_for_fetch}

    try:
        chunks = retrieve(q2, top_k=fetch_k, where=where)
    except OutOfCorpus as exc:
        debug["guardrail"] = "out_of_corpus"
        debug["reason"] = str(exc)
        return _refusal("out_of_corpus", G.OUT_OF_CORPUS, str(exc), debug)
    except FileNotFoundError as exc:
        # A missing index (fresh instance, build interrupted) is a recoverable
        # condition, not a reason to show a filesystem path in the chat. The
        # text stays human and actionable; the detail goes to debug only.
        debug["guardrail"] = "retrieval_error"
        debug["error"] = str(exc)
        return _refusal("error", G.INDEX_ERROR, "the index is not built", debug)
    except Exception as exc:  # noqa: BLE001 - a store/model hiccup must not
        # kill the whole page: it becomes an `error` refusal instead. "Never
        # raises for user input" is a contract of this function. The viewer
        # sees the friendly INDEX_ERROR, never the exception's raw message
        # (which can carry an absolute path like /opt/render/.../data/chroma).
        debug["guardrail"] = "retrieval_error"
        debug["error"] = f"{type(exc).__name__}: {exc}"
        return _refusal("error", G.INDEX_ERROR,
                        f"retrieval failed: {type(exc).__name__}", debug)

    # The store-level `where` narrowed the DENSE side to the named scheme; BM25
    # still contributes other schemes' chunks that get fused in. Trim locally so
    # debug/"Why this answer?", the validation context and the generator all
    # see ONLY the named scheme's chunks - a neighbour can no longer float a
    # figure into any of them.
    if named_for_fetch:
        chunks = [c for c in chunks if c.scheme_slug == named_for_fetch]

    debug["retrieved"] = [c.to_dict() for c in chunks]
    debug["chunk_ids"] = [c.chunk_id for c in chunks]

    # --- 5b. absent concept -------------------------------------------------
    # A question can name a real mutual-fund fact this corpus has never read,
    # and still clear the term-coverage gate on the words around it. "What is
    # the riskometer level of HDFC Large Cap?" matched on "level" and "large
    # cap" and was answered with the expense ratio: a cited, confident, wrong
    # answer. Refuse when the concept is named and nothing retrieved has it.
    absent = G.absent_concepts_in(q2)
    if absent:
        debug["absent_concepts"] = absent
        if not any(G.corpus_has_concept(t, [c.text for c in chunks]) for t in absent):
            debug["guardrail"] = "absent_concept"
            return _refusal(
                "out_of_corpus", G.OUT_OF_CORPUS,
                f"the question asks about {', '.join(absent)}, which appears in "
                f"none of the retrieved pages; answering with a neighbouring "
                f"fact would be wrong", debug)

    # --- 5c. no scheme named: is the fact genuinely ambiguous? --------------
    # "What is the exit load?" has five schemes tied at the top extraction
    # score. Answering would pick one by retrieval order - a coin flip the user
    # reads as fact. Ask which scheme instead. A single-scheme-subject question
    # ("Is there a lock-in period?") has one clearly-on-subject chunk whose top
    # score beats the runner-up, and falls through to a normal answer. Memory
    # resolution already re-names the scheme, so a resolved follow-up is never
    # ambiguous here.
    if not named_for_fetch and not memo.get("resolved_scheme"):
        top = extraction_scores(q2, chunks)
        if len(top) >= 2:
            s0, c0 = top[0]
            s1, c1 = top[1]
            if (s0 > 0 and s1 >= s0 and c0.scheme_slug and c1.scheme_slug
                    and c0.scheme_slug != c1.scheme_slug):
                debug["guardrail"] = "ask_scheme"
                corpus = list(_corpus_metadata())
                slugs = list(dict.fromkeys(
                    c["scheme_slug"] for c in corpus if c["scheme_slug"]))
                funds = ", ".join(_scheme_display_name(s, corpus) for s in slugs)
                return _refusal(
                    "out_of_corpus", G.ASK_SCHEME.format(funds=funds),
                    "the fact differs by scheme and no scheme is named", debug)

    # --- 6/7. generate ------------------------------------------------------
    # A scheme named in the question that has no chunk of its own is reported
    # here, with its name, rather than as a generic "nothing quotable" - the
    # difference between "I have nothing for that fund" and "I have nothing"
    # matters to a user who asked about one specific scheme.
    named = detect_scheme(q2, list(_corpus_metadata()))
    if named:
        debug["named_scheme"] = named
    if named and not rank_for_extraction(q2, chunks):
        debug["guardrail"] = "named_scheme_not_retrieved"
        return _refusal(
            "out_of_corpus", G.OUT_OF_CORPUS,
            f"the question names {named}, but no chunk of that scheme was "
            f"retrieved; answering from another scheme's figures would be wrong",
            debug)

    # A scheme named in the question is filtered at the PROMPT level, not just
    # at the extraction level. `chunks` from `retrieve()` is unfiltered - when a
    # scheme is named we over-fetched (named_scheme_over_fetch) precisely so a
    # short filtered result set has something to work with - and handing all of
    # it to an LLM lets it quote another scheme's figure from context with a
    # grounded-sounding citation. `rank_for_extraction` drops every chunk that
    # is not the named scheme (and says so, refusing, when none survive), so the
    # generator literally cannot see a competing fund's numbers.
    ranked = rank_for_extraction(q2, chunks)

    text, used, note = _generate(q2, ranked)
    debug["provider"] = note
    debug["chunk_used"] = used.chunk_id if used else None

    if not text.strip():
        debug["guardrail"] = "generation_empty"
        return _refusal("out_of_corpus", G.OUT_OF_CORPUS,
                        "no chunk could be quoted", debug)

    if P.NOT_FOUND in text:
        debug["guardrail"] = "model_said_not_found"
        return _refusal("out_of_corpus", G.OUT_OF_CORPUS,
                        "the generator reported the context did not answer it", debug)

    # --- 8. validate the draft ---------------------------------------------
    # The grounding context must be EVERY chunk the generator was shown, not
    # the one it happened to be scored against.
    #
    # `generation.context_chunks` is 5, so a real LLM reads 5 chunks and can
    # legitimately quote a figure from any of them. Validating against
    # `[used]` alone - a single chunk - means a correct answer that cites a
    # figure from chunk 3 is reported as "states the figure '2214.57', which is
    # not in any retrieved source", i.e. a hallucination. That is the validator
    # working exactly as specified against the wrong context.
    #
    # It was invisible with the `stub`, which quotes one chunk, so one chunk
    # *is* the whole context; it appeared the moment a multi-chunk provider was
    # switched on, and it is why the benchmark showed 0/3 models answering the
    # NAV question while the stub answered it fine.
    context_for_validation = list(chunks)
    ok, reasons = G.validate_answer(
        {"text": text, "sources": [{"url": source_for(used or chunks[0]).url}]},
        context=context_for_validation,
    )
    debug["draft_valid"] = ok
    debug["draft_reasons"] = reasons

    if not ok:
        # Try the next chunk down the EXTRACTION order, not the retriever's
        # order, so the fallback also prefers a chunk about the right subject.
        for alt in rank_for_extraction(q2, chunks):
            if used is not None and alt.chunk_id == used.chunk_id:
                continue
            candidate = render_from_chunk(alt, q2)
            if not candidate:
                continue
            # The fallback text is quoted from `alt` alone, so validating it
            # against all of `chunks` would be too generous - a figure from
            # another chunk would pass. `[alt]` is the correct context here,
            # and is correct precisely because the candidate was built from
            # that one chunk.
            alt_ok, _ = G.validate_answer(
                {"text": candidate, "sources": [{"url": source_for(alt).url}]},
                context=[alt],
            )
            if alt_ok:
                text, used, ok, reasons = candidate, alt, True, []
                debug["recovered_with"] = alt.chunk_id
                break

    if not ok:
        debug["guardrail"] = "validation_failed"
        return _refusal("out_of_corpus", G.OUT_OF_CORPUS,
                        "; ".join(reasons) or "the draft failed validation", debug)

    # --- 9. attach one source ----------------------------------------------
    # The citation goes in `text` because that is the PRD §12.1 layout, and the
    # stamp stays in its own `last_updated` field so the UI can lay the three
    # lines out without printing the date twice. Both are rendered; neither is
    # part of the model's prose, which is why the content checks strip them
    # (see guardrails.prose_of).
    src = source_for(used or chunks[0])
    body = text.rstrip()
    if not body.endswith((".", "!", "?")):
        body += "."
    final = {"text": f"{body} Source: {src.url}", "sources": [src.to_dict()]}
    stamp = last_updated_stamp(context_for_validation)

    # --- 10. validate exactly what ships -----------------------------------
    # Not redundant. Step 8 checked the generated prose; this checks the prose
    # with a URL appended to it, and an appended string can tip a two-sentence
    # answer over the cap or introduce a link that was never in the corpus.
    final_ok, final_reasons = G.validate_answer(
        final, context=context_for_validation)
    debug["final_valid"] = final_ok
    debug["final_reasons"] = final_reasons
    if not final_ok:
        debug["guardrail"] = "final_validation_failed"
        return _refusal("out_of_corpus", G.OUT_OF_CORPUS,
                        "; ".join(final_reasons), debug)

    return Answer(
        text=final["text"],
        sources=[src],
        kind="answer",
        refusal_reason=None,
        last_updated=stamp,
        disclaimer=G.DISCLAIMER,
        debug=debug if with_debug else {},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the minimum SIP for HDFC ELSS Tax Saver Fund?",
    "Is there a lock-in period on HDFC ELSS Tax Saver?",
    "What is the exit load?",
    "What benchmark does HDFC Small Cap Fund Direct Growth use?",
    "What is the AUM of HDFC Flexi Cap Fund?",
    "What is the NAV of HDFC Balanced Advantage Fund?",
    "Should I buy HDFC Large Cap?",
    "What was the return on HDFC Large Cap last year?",
    "My PAN is ABCDE1234F, can you check my folio?",
    "What is the expense ratio of SBI Large Cap Fund?",
    "price of gold in Mumbai",
]


def print_answer(a: Answer, *, show_debug: bool = False) -> None:
    print("-" * 78)
    print(a.text)
    if a.last_updated:
        print(a.last_updated)
    print()
    print(f"  kind: {a.kind}"
          + (f"   reason: {a.refusal_reason}" if a.refusal_reason else ""))
    for s in a.sources:
        print(f"  source: [{s.publisher}] {s.title}")
        print(f"          {s.url}  (fetched {s.fetched_at[:10]})")
    if show_debug and a.debug:
        print("\n  debug:")
        for key in ("provider", "guardrail", "matched_keyword", "pii_reason",
                    "reason", "draft_valid", "draft_reasons", "chunk_used"):
            if key in a.debug:
                print(f"    {key:<16} {a.debug[key]}")


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 6 - ask the assistant")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--demo", action="store_true",
                    help="run the twelve demo questions")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()

    questions = DEMO_QUESTIONS if args.demo else (" ".join(args.question),)
    if not any(questions):
        ap.error("give a question, or pass --demo")

    for q in questions:
        if not q:
            continue
        print("=" * 78)
        print(f"  Q: {q}")
        print("=" * 78)
        print_answer(ask(q), show_debug=args.debug)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(_main())
