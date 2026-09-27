# ARCHITECTURE — HDFC Mutual Fund FAQ Assistant

> Companion to `PRD.md`. This document is the **design**: what the system is made
> of, how data flows, and *why* each piece exists. `implementation.md` is the
> **build order**: phase-by-phase instructions to execute this design.
>
> Traceability: every section cites the PRD section it implements.

---

## 1. Purpose and Scope

A facts-only RAG chatbot that answers factual questions about **5 HDFC mutual fund schemes (Direct Growth)**, grounded in a curated corpus of public pages, with **exactly one source link per answer** and a polite refusal for anything that smells like investment advice, performance reporting, or personal data.

**In scope:** HDFC AMC · 5 named schemes · Direct Growth plans only · English · public sources only · CPU-only · offline after ingest.

**Out of scope:** other AMCs · other plan/option combinations · live NAV · returns or scheme comparison · portfolio advice · user accounts · file uploads.

---

## 2. System Context

```
        ┌──────────────────────────── USER ────────────────────────────┐
        │  "What is the expense ratio of HDFC Large Cap Direct Growth?" │
        └───────────────────────────────┬───────────────────────────────┘
                                        │  (Streamlit chat widget)
                        ┌───────────────▼───────────────────────────────┐
                        │  UI LAYER  —  app.py                        │
                        │  welcome · 3 example chips · disclaimer      │
                        │  citation block · "Why this answer?"         │
                        └───────────────┬───────────────────────────────┘
                                        │
                        ┌───────────────▼───────────────────────────────┐
                        │  ORCHESTRATOR  —  rag/answer.py             │
                        │   1. guardrail gate   (PII → advice → perf)  │
                        │   2. retrieve         (Stage 5)              │
                        │   3. generate         (Stage 6)              │
                        │   4. validate output (citation? ≤3 sent?)   │
                        └────┬──────────────────────────────┬──────────┘
                             │                              │
              ┌──────────────▼───────────┐     ┌────────────▼────────────┐
              │  rag/retrieve.py         │     │  LLM                    │
              │  hybrid dense + BM25     │     │  ollama / openai / stub │
              │  RRF → MMR → rerank      │     │  temperature = 0        │
              └──────────────┬───────────┘     └─────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────────────────┐
              │  rag/guardrails.py   PII · advice · performance ·    │
              │                     out-of-corpus gates              │
              └──────────────┬──────────────────────────────────────┘
                             │
   ┌─────────────────────────┴──────────────────────────────────────┐
   │                    READ-ONLY ARTEFACTS                        │
   │  data/chroma/          Stage 4  vector store  (ChromaDB)      │
   │  data/embeddings/      Stage 3  MiniLM vectors + hash cache   │
   │  data/chunks.jsonl     Stage 2  chunked corpus + metadata     │
   │  data/corpus.jsonl     Stage 1  cleaned documents             │
   │  data/raw/*.html       Stage 1  cached source pages           │
   └───────────────────────────────────────────────────────────────┘
                             ▲
                             │  OFFLINE BUILD-TIME PIPELINE (run once / on re-ingest)
   ┌─────────────────────────┴──────────────────────────────────────┐
   │  ingest/fetch.py   Stage 1a  HTTP + cache + snapshot          │
   │  ingest/clean.py   Stage 1b  structure-aware extraction       │
   │  ingest/chunk.py   Stage 2   hybrid chunking (ADR-001)        │
   │  embed/index.py    Stage 3   all-MiniLM-L6-v2 + hash cache    │
   │  store/chroma_store.py  Stage 4  persist collection            │
   └───────────────────────────────────────────────────────────────┘
                             ▲
   ┌─────────────────────────┴──────────────────────────────────────┐
   │  config/sources.yaml   ← THE CORPUS GATE (only URLs listed    │
   │                           here can enter the index)           │
   │  config/app.yaml       ← every tunable number                 │
   └───────────────────────────────────────────────────────────────┘
```

**The single most important structural idea:** the build pipeline (Stages 1–4) is
**offline and write-once**; the query path (Stages 5–7) is **read-only and
online**. There is no network call at query time. This is what makes answers
reproducible, the demo reliable with no internet, and re-ingest safe.

---

## 3. Design Principles

| # | Principle | Consequence in the design |
|---|---|---|
| P1 | **Every stage is independently runnable** | Each stage is a module with a `if __name__ == "__main__"` runner that prints its own report and writes one file. A grader can run Stage 2 alone. |
| P2 | **Content-addressed, not position-addressed** | Chunks get `chunk_id = {scheme_slug}__{section_slug}__{hash6(text)}`. Re-running on unchanged text yields identical IDs → safe upsert into Chroma, and embeddings are cached by `content_hash`. |
| P3 | **Structure before statistics** | Headings/tables/accordions are extracted as *structure*, and chunking respects them. This is the direct consequence of ADR-001. |
| P4 | **Grounding is enforced, not requested** | The generator gets no tools, temperature 0, and its output passes a validator. If it fails validation, we degrade to an extractive answer, never to a guess. |
| P5 | **Refusal is a first-class path** | Advice / performance / PII / out-of-corpus each have a dedicated, tested branch with a fixed message. The app is not "an LLM that was told to be careful"; it is a gate chain. |
| P6 | **Degrade, never crash** | Missing chromadb, missing torch, missing API key, unreachable page — each has a defined fallback so the demo still runs. |
| P7 | **Privacy by default** | The corpus is scrubbed before chunking; the query gate refuses rather than logs; logs store masked/hashed values only. |
| P8 | **The corpus gate is data, not code** | `config/sources.yaml`. Adding a source must never require touching Python. |

---

## 4. Stage-by-Stage Design

### Stage 0 — Configuration & the corpus gate

| | |
|---|---|
| **Module** | `common.py` (`load_config`, `load_sources`, `load_schemes`, `load_supplementary`, `path_for`, `ensure_dirs`) |
| **Input** | `config/app.yaml`, `config/sources.yaml`, optional `.env` |
| **Output** | in-memory config; guaranteed-on-disk directories |
| **Design** | YAML, cached with `lru_cache`. `sources.yaml` holds the 5 seed scheme pages + supplementary SEBI/AMFI/AMC pages. **Nothing enters the corpus unless it is listed here.** |

**Why a gate:** the brief bans third-party blogs and screenshots. Making the allowed-source list a config file turns that rule from "a promise in the README" into "a mechanical constraint".

---

### Stage 1a — Fetch (`ingest/fetch.py`)

| | |
|---|---|
| **Input** | URLs from `sources.yaml` |
| **Output** | `data/raw/{doc_id}.html`, `data/source_snapshot.json`, `data/ingestion_log.jsonl` |
| **Design** | `httpx` with a descriptive User-Agent, 3 retries, exponential backoff, 2 s politeness delay between requests, 30 s timeout. Raw HTML is cached to disk and re-used if < 24 h old unless `--force`. |
| **Failure policy** | A failed page is **logged and skipped**, never fatal. `source_snapshot.json` records SHA-256 + byte size + fetch timestamp per URL so drift is detectable. |

**Key risk (PRD R1):** aggregator pages are JS-rendered. The fetcher therefore also records whether the extracted text is suspiciously short; if so, it raises a visible warning that the page needs a JS-rendered fallback or a source swap. This is a *reported* condition, not a silent one.

---

### Stage 1b — Clean (`ingest/clean.py`)

| | |
|---|---|
| **Input** | raw HTML |
| **Output** | `data/corpus.jsonl` — one `Document` per page |
| **Design** | BeautifulSoup + lxml. Three jobs: |

1. **Remove boilerplate.** Drop everything matching `chunking.strip_selectors` (nav, header, footer, aside, script, style, noscript, cookie banners, breadcrumbs, related-fund rails, app-store badges). Then re-hunt for now-orphaned short text nodes — removing the nav can leave stray text behind.
2. **Preserve structure.** Walk the DOM in document order and emit a typed node stream:
   - `heading` (with level) → establishes `section_heading` for everything after it
   - `paragraph` / `list_item` → prose
   - `table_row` → `(label, value)` pairs
   - `accordion` / `faq_item` → `(question, answer)`
   This is what makes ADR-001 possible in Stage 2.
3. **Scrub PII.** `common.scrub_pii` over every text node. Nodes that were high-confidence PII are redacted; anything still matching after scrubbing is quarantined to `data/quarantine.jsonl` and never chunked.

**Why the node stream, not a flat string:** a flat `get_text()` output destroys the table/accordion boundaries, and Stage 2 could not do anything smarter than recursive splitting. The node stream is the payload that makes the chunking decision worth having made.

---

### Stage 2 — Chunking (`ingest/chunk.py`) — ADR-001

| | |
|---|---|
| **Input** | `data/corpus.jsonl` (node stream) |
| **Output** | `data/chunks.jsonl`, `data/chunk_stats.json`, `data/quarantine.jsonl` |
| **Strategy** | `structure_first_hybrid` |

**The algorithm, in order:**

```
for each document:
    group nodes into sections           # a heading starts a new section
    for each section:
        (A) FACTS  - for every table_row / faq_item / labelled spec value
                      whose label maps to a canonical fact_key
                      (expense_ratio, exit_load, minimum_sip, lock_in_period,
                       riskometer_level, benchmark, ...):
                          emit ONE chunk  {label}: {value}
                          doc_type="fact", fact_key, fact_value
        (B) PROSE  - concatenate the remaining paragraph/list nodes, then
                      heading-aware recursive character split:
                        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
                        target 1000 tok, hard cap 1100, overlap 135 tok
                        overlap snapped to a sentence boundary
        (C) SEMANTIC MERGE - for each run of adjacent UNDERSIZED prose chunks
                      (< 40 tok), embed the lead sentence of each, and merge
                      neighbours whose cosine ≥ 0.62, up to 1100 tokens.
                      Re-split anything that overflows after merging.
    validate: every chunk has the full metadata schema (PRD S6.1)
    persist:  data/chunks.jsonl + data/chunk_stats.json
```

**Chunk families and why both are needed**

| | `fact` chunks | `prose` chunks |
|---|---|---|
| Shape | one label + one value, ≤ 120 tok | a coherent passage, ~1000 tok |
| Built from | table rows, FAQ accordions, labelled specs | paragraphs and list items |
| Answers | "what is the TER", "exit load", "lock-in", "min SIP" | "how do I download a statement", "what is this fund's objective" |
| Retrieval strength | strong on BM25 (exact label match) | strong on dense (semantic neighbours) |

Keeping them as separate families is what lets Stage 5 use a hybrid ranker effectively — a lexical hit nails "exit load" and a dense hit nails "how do I get my tax paperwork".

**Idempotency:** `chunk_id = {scheme_slug}__{section_slug}__{short_hash(normalised_text,6)}`. Identical input → identical IDs.

**Stats emitted** (`data/chunk_stats.json`): per-scheme chunk counts, mean/median/max tokens, % fact vs prose, quarantine count, and the ADR-001 validation evidence (see §9).

---

### Stage 3 — Embedding (`embed/index.py`)

| | |
|---|---|
| **Input** | `data/chunks.jsonl` |
| **Output** | `data/embeddings/{content_hash}.npy` (384-d, float32, L2-normalized), `data/chunks.jsonl` (unchanged — vectors are not inlined) |
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` · 384-d · mean pooling · cosine |

**The lead-window trick.** MiniLM is trained on sentences and its reliable
window is roughly 256 tokens; it degrades on 1000-token inputs. So the embedded
string is *not* the raw chunk:

```
embed_text = prefix + chunk_text
prefix     = "HDFC Large Cap Fund (Large Cap) - Fees and charges: "
input      = truncate(embed_text, embed_lead_tokens = 200)
```

The prefix injects scheme + aliases + category + section, which disambiguates the near-identical "expense ratio: 1.03%" style facts that appear on all five pages. The lead 200 tokens carry the fact-bearing sentence. **This is a deliberate trade-off** — the stored vector does not represent the tail of a long prose chunk — and it is documented in `implementation.md` so nobody "fixes" it later.

**Cache: keyed by `embed_hash`, NOT `content_hash`.** This was a real bug, found and fixed in Phase 3. `content_hash` addresses the raw chunk text, but the embedded text is `prefix + text` and the prefix carries the scheme name. `"min. for sip: ₹100"` is byte-identical across four of the five schemes, so caching on `content_hash` hands three of those four a vector belonging to the first one — and a question about HDFC Flexi Cap gets answered with HDFC Large Cap's row. Measured on the real corpus: 121 chunks, 84 distinct `content_hash` values, **37 chunks that would have been given the wrong scheme's vector.** The cache key is therefore `content_hash(build_embed_text(chunk))`, and `data/embed_index.jsonl` is the explicit chunk_id → vector join surface for Stage 4.

**Cost control:** `SentenceTransformer(..., device="cpu")`, batch 64, `show_progress_bar=True`.

---

### Stage 4 — Vector store (`store/chroma_store.py`)

| | |
|---|---|
| **Input** | `data/chunks.jsonl` + `data/embed_index.jsonl` + `data/embeddings/*.npy` |
| **Output** | `data/chroma/` — persistent collection `hdfc_mf_faq`, 121 rows |
| **Design** | `chromadb.PersistentClient(path=...)`, cosine space, 384-d. Every chunk stored as one document keyed by `chunk_id`, with the **full** metadata block from PRD §6.1, so Stage 5 can filter by `scheme_slug`, `category`, `doc_type`, `publisher`. |
| **Operations** | `sync()` (default upsert), `rebuild()` (`--rebuild`: wipe + recreate), `query()`, `stats()`, `coverage_gaps()`. |

**Why Chroma and not raw numpy:** the brief mandates it, it gives persistence and metadata filtering for free, and `data/chroma/` is a real on-disk artefact to show the grader.

**The join is by `chunk_id`, through the Stage 3 index file.** `upsert_chunks()` never infers a vector from a hash; it looks each chunk up in `data/embed_index.jsonl` and loads the file named there. A chunk with no row in that index raises rather than being skipped — a silently short index is how a scheme becomes unanswerable with nothing in the logs.

**Two Chroma 1.5.9 limits this stage had to design around.**

1. **`hnsw:metadata.hdim` cannot be set.** The modern `CreateHNSWConfiguration` exposes only `space` / `ef_construction` / `ef_search` / `max_neighbors` / `num_threads` / `resize_factor` / `sync_threshold` / `batch_size`, and the legacy `metadata={"hnsw:metadata": {...}}` form is rejected — Chroma metadata values must be scalars, and a nested dict is not one. Worse, **Chroma does not pin the dimension at all**: a 768-dim vector (i.e. someone swapped in `all-mpnet-base-v2`) is upserted into a 384-dim cosine collection with no error, and queries then return silently wrong rankings. Verified in `tests/test_store.py::test_chroma_does_not_pin_the_dimension`. So the expected dimension is recorded in the collection metadata *for humans*, and enforced in code by `_load_vector()`, which is the only place it can actually be enforced.
2. **A filtered query can come back short, with no error.** HNSW is approximate and the metadata filter is applied as a post-filter over the candidate window, so a walk that stops early loses rows. Measured on the real collection: `where={"scheme_slug": "hdfc-small-cap-fund-direct-growth"}` over 24 matching rows returned **23 on one run and 24 on the next twenty**. Consequences for Stage 5: never read `len(results) == top_k` as "the filter is satisfied", over-fetch and truncate locally, and treat a scheme filter as a ranking refinement rather than an existence check. What *is* guaranteed — and is tested — is that a short result set is missing rows, never wrong rows.

**Cosine distance is converted to similarity on the way out.** Chroma reports `1 - cosine_similarity`; Stage 5's thresholds are written against similarity, and a sign flip here would invert every ranking silently.

**Measured result (2026-09-27):** 121/121 rows, no gaps, no orphans. 25 ELSS / 24 each Flexi Cap, Large Cap, Small Cap, Balanced Advantage. 81 `fact` + 30 `faq` + 10 `prose`. Two `upsert` runs in a row leave the count at 121.

---

### Stage 5 — Retrieval (`rag/retrieve.py`)

| | |
|---|---|
| **Input** | user question (+ optional scheme filter) |
| **Output** | `list[RetrievedChunk]` ranked, each with `chunk_id`, `text`, metadata, `dense_score`, `bm25_score`, `rrf_score`, `final_score`, `coverage`, `matched_terms` |

**Pipeline:**

```
1. normalise      lowercase; KEEP decimal points (a NAV is not a sentence)
2. expand         synonym map, so a query in the user's words hits the page's
                  own words (TER→expense ratio, lock-in→lock in period, …)
3. dense search   Chroma query_embeddings  -> top 20
4. lexical search BM25 over metadata-prefix + body -> top 20
5. fuse           Reciprocal Rank Fusion, k=60
6. gate           term coverage >= 0.5  AND  (dense >= 0.20  OR  bm25 >= 1.0)
7. diversify      MMR, λ=0.35   (kills five near-duplicate fee rows)
8. scheme filter  a named scheme is a HARD DROP, not a preference
```

**Why hybrid, specifically:** the corpus is full of *numbers in tables*. Dense
retrieval on `all-MiniLM-L6-v2` is weak on numerals; BM25 nails
`"expense ratio"`. Conversely BM25 is useless for *"how do I get my tax
paperwork"*. Running both and fusing is what makes one collection serve both
the fact questions and the how-to questions.

**BM25 indexes the metadata prefix too**, not just the body, mirroring Stage 3's
`embed_prefix_template`. Half the discriminative words on a scheme page are the
scheme's own name, and an index that cannot see them cannot separate five pages
that differ by nothing else.

#### The central finding: coverage is the gate, not similarity

The plan was `min_cosine = 0.35`. That number does not work, and the reason is
worth recording because it is not a tuning problem:

| question | in corpus? | best dense |
|---|---|---|
| `exit load` | yes | **0.159** |
| `lock-in period` | yes | **0.144** |
| `price of gold in Mumbai` | no | **0.310** |
| `mumbai flat price` | no | **0.347** |

The two out-of-corpus questions score **twice as high** as the two in-corpus
ones. No threshold on cosine separates those groups: anything above ~0.15
refuses real questions, anything below ~0.35 admits nonsense. The cause is
structural — a short query about a foreign topic lands in a middling region of
the embedding space, while a terse query whose exact words are present is a weak
*global* match even though it is a perfect *local* one.

So the decision was moved to **term coverage** — the share of the question's own
content terms that appear in the chunk — and similarity was demoted to a floor
that only removes obvious non-matches:

```
min_coverage: 0.5      the actual accept/reject decision
min_cosine:   0.20     floor only
min_bm25:     1.0      floor only
```

`RetrievedChunk.clears()` requires coverage **and** (dense floor **or** BM25
floor), and coverage is applied **before** MMR so a rejected chunk never
displaces a good one. Measured effect: `"price of gold in Mumbai"` is rejected at
coverage 0.00 with a message naming the number, and `"exit load"` is accepted.

Two refinements the corpus forced:

- **`content_terms()` falls back to the full non-stopword list** when
  noise-filtering empties it. Otherwise `"direct growth plan"` reduces to nothing
  and every Direct Growth question would be unanswerable.
- **`match_prefix` coverage uses exact-or-4-char-prefix**, so a user typing
  `"minimum SIP"` matches the corpus's `"min. for SIP"`. Without it the most
  natural phrasing of a golden-set question scored 0.0.
- **`nav` was removed from `_COVERAGE_NOISE`** — it is a `fact_key` in this
  corpus, so treating it as a stopword removed the one term that identifies the
  NAV row.
- **`matched_terms` reports only the question's own tokens**, never synonym
  expansions. Reporting the expansion would show a user a "match" on a word they
  never typed.

#### The scheme filter is a hard drop

All five schemes publish an `expense_ratio` row. On subject match alone the five
tie, and the tie went to retrieval order — so *"What is the expense ratio of HDFC
Large Cap Direct Growth?"* came back with **"The expense ratio for HDFC Small Cap
Fund is 0.78%"**: a confident, cited, wrong number for a different fund. For this
project that is the single most damaging output it can produce.

So when a question names a scheme, chunks belonging to any *other* scheme are
dropped outright, and if nothing is left the answer is `out_of_corpus`. Never a
neighbour's figure. Detection matches progressively trimmed name variants,
longest-first, skipping variants under 2 words or 6 characters so a bare `"hdfc"`
cannot pick a scheme.

#### Other-AMC scope

`SBI Large Cap` retrieves HDFC's Large Cap row at BM25 8.9 — the two funds' fact
shapes are close to byte-identical, which is exactly why the mix-up is easy. A
list of ~45 other AMCs is refused pre-retrieval; `test_another_amc_is_out_of_scope`
pins twelve of them, so adding a scheme's support without adding its AMC is what
fails.

#### `OutOfCorpus` vs `out_of_scope`

Both mean "no", but they are different `kind` values and the distinction is
load-bearing. A question about SBI's Large Cap is `out_of_corpus` — HDFC-only is
a scope decision. A question about "the price of gold in Mumbai" is
`out_of_corpus` — retrieval found nothing. Neither is `out_of_scope`; that value
is not in the PRD §11 `Literal` and is never emitted.

---

### Stage 6 — Generation (`rag/answer.py`, `rag/guardrails.py`, `rag/prompts.py`)

| | |
|---|---|
| **Input** | question, `list[RetrievedChunk]` |
| **Output** | `Answer{text, sources, kind, refusal_reason, last_updated, disclaimer, debug}` |

**Order of operations.** Guardrails first, retrieval second, generation third,
validation fourth:

```
1. PII          query scan -> refusal_pii        (logs a HASH, never the text)
2. advice       keyword + regex patterns -> refusal_advice
3. performance  keyword + regex patterns -> refusal_performance
4. scope        other AMC / action request -> out_of_corpus
5. retrieve     -> out_of_corpus on failure
5b. absent concept  a named fact the corpus has never read -> out_of_corpus
6. generate     stub | ollama | openai
7. validate     grounding, length, PII echo, performance -> retry, then fall back
8. attach       exactly one Source, the stamp, the disclaimer
```

Steps 1–4 are pre-retrieval because a refusal must not need a retrieval to
justify itself. Step 5b is post-retrieval because whether the corpus can answer
depends on *what was retrieved*, not on what was asked.

#### Two vocabularies, and the answer-side check is groundedness

`guardrails.advice_keywords` stays question-phrased (`"should i"`) because it is
matched against what a user types. It matches **nothing** in declarative
generated text, so a second vocabulary — `generation.advice_vocabulary`, 22
assertion phrases — checks the answer side. Both were measured: 0 occurrences
across all 121 chunks.

The answer-side check that actually earns its keep is **groundedness**, not
vocabulary. Quoting each of the 121 real chunks against itself as its own
context produces **0 of 121 violations**, while hand-built answers claiming a
`24%`, `9%` and `12%` return are all rejected. A vocabulary list catches the
words you thought of; a grounding check catches the sentence you did not.

**Benchmark-index exemption.** Four of the five benchmarks are literally named
`… Total Return Index`, so a naive return-vocabulary check refuses *"which
benchmark does HDFC Large Cap use?"* — a golden-set question — on the index's own
name. `generation.benchmark_name_allow: ["total return index", "return index"]`
strips the legitimate cases, and it is applied **on the query side as well as the
answer side**, because the question contains the same words.

**Performance claims are refused by feature pairing, not by word list.** The
corpus itself contains one: an FAQ chunk says *"the average annual returns
provided by this fund is 18.28% since its inception"*. That figure is on the
page, so a grounding check passes it happily — grounding asks "did the source say
this?" and the source did. PRD §12.3 forbids reporting it regardless, so the rule
is the *pair* of return vocabulary with a percentage, which is a performance
claim wherever the number came from.

#### The stub quotes; it does not write

The default generator is extractive, and it takes the page's own
`label: value` text rather than paraphrasing. **There is no fact_key → phrase
table**, because the corpus's own labels are the phrasing — and the
`fund_objective` key, despite the name, holds the *benchmark* line, which is
precisely the kind of thing a hand-written phrase table would have got wrong.

A wrong answer then requires a bug rather than a hallucination, which is what
makes the eval's exact-figure assertions meaningful.

#### Extraction ranking, and the three ways it was wrong

`rank_for_extraction()` is scheme-first, then subject. The subject score is
keyed on `FACT_TRIGGERS`, and three real problems showed up:

1. **A chunk with an empty `fact_key` could only ever score 0.0**, so it was
   structurally incapable of being selected no matter what it was about. The
   fund manager's name is on the page only inside the run-on stat strip, which
   has no `fact_key` — so *"Who is the fund manager?"* ranked the `rating` row
   first and answered **"the rating is 4"**. An unkeyed chunk now scores on the
   question's subject words found in its own text; a keyed chunk keeps its extra
   `+1.0` for being the page's own short factual form, so this can only change
   the outcome where the unkeyed chunk was the only one on subject.
2. **Verb and noun do not meet.** *"Who manages HDFC Small Cap?"* and
   *"dhruv muchhal is the current fund manager of hdfc small cap fund"* share no
   word that matches on boundaries. The trigger table cannot bridge them, so the
   pair is matched explicitly, and the query-side synonym map was extended so the
   stat strip clears Stage 5's coverage gate at all.
3. **Several chunks share one `fact_key`.** Groww groups three rows under one
   heading, so `exit_load` covers both *"exit load of 1% if redeemed within 1
   year"* and *"stamp duty on investment: 0.005%"*. Both match the trigger;
   retrieval order decided, and an exit-load question got answered with the
   stamp duty. The rule now prefers the chunk whose **value** is about the charge
   that was asked about.

That third rule was written wrong first. It keyed on the **label**, and made
things worse, because the correct row carries the compound label
`"exit load, stamp duty and tax"` while the wrong one carries the plain
`"exit load"`. The value is the reliable side. The set of charge terms it
consults is deliberately tiny — the only compound groupings this corpus has.

#### A few smaller decisions that are load-bearing

- **`value_half()` exists because `fact_value` is not reliably the value alone.**
  `expense_ratio` stores `"0.78%"`; `nav` stores the whole line
  `"nav: 25 sep '26"`. Extracting digits is more robust than string-matching the
  field.
- **`_collapse_repeats()`** collapses only an *exact* back-to-back repeat of ≥2
  words. A looser rule eats legitimate repetition.
- **`prose_of()` strips the citation and the `Last updated` stamp before content
  checks.** The stamp lives only in `Answer.last_updated`, never in `Answer.text`
  — which is why the three-line layout cannot print the date twice.
- **`answer_bearing_boost: 0.004`** in `rag/retrieve.py` is an ordering-only
  nudge. It does not touch `rrf_score` or `clears()`, so it cannot let a
  non-answerable chunk through the gate.
- **Scheme detection in the answer layer reads the full corpus**, never the
  retrieved subset, so a scheme named in the question is recognised even when
  retrieval returned nothing for it.
- **`named_scheme_over_fetch: 15`.** Terse `fact` chunks lose to verbose `faq`
  restatements on *both* retrievers, because the restatement carries the scheme
  name and four more words. Fetching 5 and then filtering to the named scheme
  routinely left no `fact` chunk at all.

**Providers:** `stub` (default, extractive, offline), `ollama` (local), `openai`
(any OpenAI-compatible endpoint, which covers Groq/Together/OpenRouter by
setting `OPENAI_BASE_URL`), and `groq` (first-class, for `GROQ_API_KEY` +
`GROQ_MODEL`). Whatever the provider returns still passes `validate_answer` — a
real LLM does **not** get to bypass the grounding, length or performance checks.

---

### Stage 7 — UI (`app.py`)

Single-page Streamlit chat. Header carries the literal note
**"Facts-only. No investment advice."**; footer repeats it. Welcome line + 3
example question chips. Every answer renders: text, the one citation link, the
`Last updated from sources:` stamp, the disclaimer, and a *"Why this answer?"*
expander showing retrieved chunk IDs, scores, and excerpts. Sidebar: scope, the
5 source links, corpus stats, rebuild button. **No file-upload widget anywhere.**
Chat history is session-state only, never persisted.

The pure helpers (`citation`, `source_line`, `stamp_line`, `body_text`,
`why_rows`, `corpus_stats`, `run_rebuild`) are defined **above** the `st.*` calls
so they are testable without a server. That is the whole reason the UI has 36
tests that run in 21 seconds instead of needing a browser.

#### Four bugs this stage produced, all of them invisible to unit tests

1. **The third example chip was a question that correctly refuses.** The chip was
   the capital-gains question, which retrieval refuses as out-of-corpus. PRD
   FR-8.3 forbids a chip that trips a refusal: it is a bad first impression, and
   it teaches the user the demo is broken. Replaced with PRD §12's verbatim third
   chip. All three are now verified `kind="answer"` with one citation each; the
   capital-gains question stays in the golden set instead.
2. **Every source was rendered twice.** The stub appends a bare
   `Source: <url>` to `Answer.text` so the CLI has something to print; the UI
   also renders a labelled citation. Same single source, twice, violating FR-8.4.
   `body_text()` strips the inline one. The regex now lives in `common.py`
   because the generated `docs/SAMPLE_QA.md` renders the same three-line shape
   and had to not repeat the bug.
3. **`corpus_stats()["embeddings"]` read `computed` instead of `computed +
   cached`.** A warm rebuild is a 100% cache hit *by design*, so the Rebuild
   button printed `Embeddings: —` every single time.
4. **`run_rebuild()` shells out to a subprocess.** Stages 1–4 own module-level
   state — the embedding model, the Chroma client. Re-importing them inside the
   Streamlit session would leave the live session serving scores from a store it
   had already replaced.

#### How this was verified

`streamlit run` returning `200` on `/_stcore/health` proves nothing, because
Streamlit reports script errors in the browser rather than to the health
endpoint. The real check is `streamlit.testing.v1.AppTest`, which executes the
script: 0 exceptions, 3 chips, the Rebuild button, the sidebar, and a chat
exchange producing 1 expander, 1 dataframe, the figure, the stamp and
`kind="answer"`.

One AppTest gotcha worth recording: `at.markdown` flattens the sidebar, so the
"exactly one citation" assertion reads 6 instead of 1. `at.sidebar.markdown` is
the way to isolate it.

---

## 5. Data Contracts

### 5.1 `Document` (Stage 1 output, one line of `corpus.jsonl`)

```jsonc
{
  "doc_id": "hdfc-large-cap-fund-direct-growth",
  "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "page_title": "HDFC Large Cap Fund Direct Growth",
  "publisher": "Groww",
  "page_type": "scheme_page",
  "scheme_slug": "hdfc-large-cap-fund-direct-growth",
  "scheme_name": "HDFC Large Cap Fund - Direct Growth",
  "short_name": "HDFC Large Cap",
  "category": "Large Cap",
  "plan": "Direct Growth",
  "nodes": [ {"type":"heading","level":2,"text":"Fees and charges"}, ... ],
  "raw_sha256": "…",
  "source_fetched_at": "2026-09-27T10:12:03.412Z",
  "ingested_at": "2026-09-27T10:13:41.880Z",
  "pii_scan": "clean",
  "node_count": 412,
  "text_chars": 28410
}
```

### 5.2 `Chunk` (Stage 2 output, one line of `chunks.jsonl`)

Exactly the metadata schema in **PRD §6.1** — it is the contract between Stage 2,
Stage 3, Stage 4, Stage 5 and the UI citation block, so it is enforced at write
time by `validate_chunk()` in `ingest/chunk.py`.

### 5.3 `RetrievedChunk` (Stage 5 output)

```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict          # the full Stage-2 metadata block
    dense_score: float
    bm25_score: float
    rrf_score: float
    final_score: float
    coverage: float         # share of the question's content terms present
    matched_terms: list[str]

    def clears(self) -> bool:  # the actual accept/reject decision
        return self.coverage >= MIN_COVERAGE and (
            self.dense_score >= MIN_COSINE or self.bm25_score >= MIN_BM25)
```

`coverage` is here because it, not `dense_score`, is what decides the answer.
See Stage 5 — the measured table of why no similarity threshold works.

### 5.4 `Answer` (Stage 6 output)

Exactly the contract in **PRD §11**.

---

## 6. Module Map

| File | Stage | Responsibility | Runnable |
|---|---|---|---|
| `common.py` | 0 | config, paths, hashing, tokens, PII, JSONL, logging | `python common.py` |
| `ingest/fetch.py` | 1a | HTTP + cache + snapshot + log | `python -m ingest.fetch --all` |
| `ingest/clean.py` | 1b | boilerplate removal, node stream, PII scrub | `python -m ingest.clean` |
| `ingest/chunk.py` | 2 | ADR-001 hybrid chunking + stats | `python -m ingest.chunk` |
| `embed/index.py` | 3 | MiniLM + content-hash cache | `python -m embed.index` |
| `store/chroma_store.py` | 4 | Chroma upsert / rebuild / query / stats, join via `chunk_id` | `python -m store.chroma_store --rebuild` |
| `rag/retrieve.py` | 5 | dense + BM25 + RRF + coverage gate + MMR | `python -m rag.retrieve --demo` |
| `rag/answer.py` | 6 | gate → retrieve → generate → validate | `python -m rag.answer "exit load"` |
| `rag/prompts.py` | 6 | system rules (unit-tested) | — |
| `rag/guardrails.py` | 5/6 | PII, advice, performance, absent-concept, validator | `python -m rag.guardrails` |
| `app.py` | 7 | Streamlit UI | `streamlit run app.py` |
| `eval/run_eval.py` | 8 | 54-question golden set harness | `python -m eval.run_eval` |
| `tools/dump.py` | — | readable dumps of chunks / vectors / store / corpus | `python -m tools.dump` |
| `tools/make_sources.py` | 9 | generates `docs/SOURCES.{md,csv}` | `python -m tools.make_sources` |
| `tools/make_sample_qa.py` | 9 | generates `docs/SAMPLE_QA.md` from live output | `python -m tools.make_sample_qa` |
| `tools/make_disclaimer.py` | 9 | generates `docs/DISCLAIMER.md` from the code's string | `python -m tools.make_disclaimer` |

`eval/` and the three `tools/make_*` generators are build steps for documents
rather than pipeline stages, which is why they sit outside the six-stage chain.
The generators exist so the delivery docs cannot drift from the code: a
hand-typed disclaimer, source table or sample Q&A is a claim about the system
rather than evidence of it.

---

## 7. Failure Modes and Degradation

| Failure | Detection | Response |
|---|---|---|
| Source page unreachable / returns a JS shell | fetch status, or extracted text < 2000 chars | log it, continue with the other pages, print a loud warning in the corpus report. Never silently produce an empty index. |
| Page contains PII | `scrub_pii` post-check | redact; if it still matches, quarantine to `quarantine.jsonl` and skip. |
| `chromadb` not installed | import guard in `store/` | clear install message; Stage 3 still runs (vectors on disk). |
| `torch` / `sentence-transformers` not installed | import guard in `embed/` | clear install message; Stages 1–2 still run and are testable. |
| No LLM configured | `LLM_PROVIDER` unset or endpoint unreachable | `stub` extractive mode. |
| Nothing relevant retrieved | **term coverage** below `min_coverage` (0.5), plus the dense/BM25 floors | out-of-corpus refusal naming the best coverage score, not a guess. |
| Question names a real fact the corpus has never read | `known_absent_terms` in `config/app.yaml`, checked *after* retrieval | out-of-corpus refusal, rather than answering with a neighbouring fact. |
| A named scheme has no chunks of its own | scheme filter returns empty | out-of-corpus, naming the scheme. Never a neighbour's figure. |
| Generator output fails validation | `validate_answer` | one retry, then extractive fallback from the next chunk **in extraction order**. |
| User pastes PAN/Aadhaar/OTP/email/phone/account no. | `scan_pii` on the **query** | refuse, log only a hash + reason, never the text. |
| Golden set reports a PII identifier in its own output | `eval.run_eval.redact` | the report, the JSON and stdout are redacted before they are written. |

The last two are the ones worth dwelling on. The PII guardrail already refuses
those questions, so the pipeline is safe — but an **eval report that quotes the
PAN it was given** has copied that PAN into a file that gets handed to a class.
The redaction is about the evidence, not the system.

---

## 8. Security and Privacy

- **No PII ingestion.** Corpus scrubbed before chunking; quarantined chunks never embedded.
- **No PII logging.** `guardrails` logs `sha256(query)[:12]` + reason, never the query body. Logs are committed evidence, so they must be PII-free by construction.
- **No secrets in code.** `.env` only; `.env.example` documents the keys; `.gitignore` excludes `.env`.
- **No network at query time.** Stage 5–7 read only from `data/`.
- **No file uploads.** The UI has no uploader. The assistant's corpus is exactly `sources.yaml`.
- **Masked reporting.** `mask_value()` keeps 2 characters for human context and stars the rest — enough to debug, useless to an attacker.

---

## 9. Observability and the ADR-001 Validation

The design's most contestable decision is the chunking strategy (PRD §20), so the
pipeline must produce **evidence** for it, not just an assertion. Stage 2 writes
to `data/chunk_stats.json`:

```jsonc
// ACTUAL output of `python -m ingest.chunk` on 2026-09-27, not a mock-up.
{
  "generated_at": "…",
  "strategy": "structure_first_hybrid",
  "per_scheme": {
    "hdfc-large-cap-fund-direct-growth": {
      "chunks": 24, "facts": 22, "prose": 2,
      "mean_tokens": 41.0, "median_tokens": 15, "max_tokens": 282
    }
    // …one row per scheme
  },
  "totals": {
    "chunks": 121, "facts": 111, "prose": 10, "faq": 30, "quarantined": 0,
    "documents_kept": 5, "documents_rejected": 2
  },
  "rejected_documents": {
    "amfi-mutual-fund-basics":
      "corpus gate: no fact-bearing nodes and mean node is only 34 chars (< 80) - navigation/search shell, not content",
    "groww-help-statement-guides":
      "corpus gate: no fact-bearing nodes and mean node is only 55 chars (< 80) - navigation/search shell, not content"
  },
  "adr001_evidence": {
    "fact_label_value_separated_pct": 0.0,   // target 0%  (PRD S20 validation plan)
    "prose_below_min_tokens_pct": 0.0,
    "prose_median_tokens": 243,
    "semantic_merges_applied": 0
  }
}
```

The number that matters is `fact_label_value_separated_pct`. If a fact's label
ends up in a different chunk from its value, the whole strategy has failed its
own acceptance test and ADR-001 must be re-opened.

### 9.1 What the build actually found

Three defects that only a real run can surface. Each is now enforced by a test,
because each was invisible to inspection.

**1. The ADR-001 test failed at 4.27% on the first run.** The five
`how_to_invest` FAQ answers run ~130 tokens, over the 120-token fact cap, so they
were being *truncated* — label kept, value cut mid-sentence. The fix is not a
larger cap: it is that a node which does not fit the cap is **declined** and
routed to the prose splitter, which keeps whole sentences. Truncating a fact to
make it fit the schema is the exact failure the strategy exists to prevent.

**2. Prose was re-embedding every fact.** Each section's prose chunk contained
the same `expense ratio: 1.03%` string that already existed as its own atomic
chunk, so the same sentence was embedded twice and retrieval scores started
depending on how many places a fact happened to be repeated. Nodes consumed as
facts are now excluded from prose.

**3. Half the fetched corpus was noise.** `amfiindia.com/mutual-fund` and
`groww.in/help` both fetched cleanly and both cleaned without error, and both
turned out to be a search box and a footer nav. They are now caught by a corpus
quality gate (`document_is_answerable`: zero fact chunks **and** mean node under
80 chars ⇒ reject) and reported by name in `chunk_stats.json` rather than
edited out of `sources.yaml`, so the finding is re-proved on every run.

The knock-on effect is stated in the PRD rather than hidden: the answerable
corpus is 5 pages, so `riskometer` and `capital-gains-statement` questions
became out-of-corpus test cases (§15.1).

Every stage also prints a human-readable report, and the Streamlit sidebar shows
the same numbers, so a reviewer never has to open a JSON file to see that a stage
worked.

### 9.2 What only the end-to-end eval found

The three defects above were caught by a stage's own acceptance test. The eight
below were not caught by anything except asking 54 questions of the whole
pipeline at once — 420 unit tests, the demo questions, and reading the code all
passed while these were live. They are the argument for a golden set.

**Six were guardrail holes.** A04 *"is HDFC Large Cap a safe choice?"* — nothing
matched, because the advice list was all imperatives (`buy`, `invest`, `should
i`) and this is an adjective. A05 *"when is a good time to enter the market?"* —
market timing, again not imperative. P05 *"NAV growth since inception"* — neither
"cagr" nor "annual return", so the performance gate did not fire on a question
that is one by construction. X03, a 16-digit account number, was **not caught as
PII at all**: `common.py` and `rag/guardrails.py` each hold a PII pattern list,
the first for the corpus scrub and the second for the query, and they had
drifted — one matched `acc no.` and the other did not. O01 *riskometer* was
answered with the **expense ratio** (the `known_absent_terms` guard of §5b did
not exist yet). O05 was misclassified because the fund manager's name is in the
corpus while the PRD's out-of-corpus examples had never mentioned fund managers.

The PII drift is the one that should worry a reader most, because it is a
structural hazard and not a one-off: two copies of the same security logic is two
places for it to be wrong. They are now pinned together by
`test_the_two_pii_pattern_lists_still_agree`, over the exact probe strings that
exposed the drift. The `long_digits` backstop is only safe because the corpus has
**0** bare 12+ digit runs; a test fails loudly if a re-fetch ever changes that,
because a wrong high-confidence rule is worse than no rule.

**Two were extraction failures.** F18 asked how long the ELSS lock-in was, phrased
without the words "lock in" — six phrasings added to `FACT_TRIGGERS`. F09 and
F06 confused *stamp duty* with *exit load*, because both live under one
`fact_key` on the page; this is the compound-charge value rule of Stage 6, and
the label-based first attempt made F06 *worse*.

`tests/test_golden_regressions.py` keeps all eight in their own file, dated and
named, rather than folded into the suites that cover each stage — because a test
whose origin is invisible does not stop the bug coming back.

---

## 10. Testing Strategy

| Layer | File | What it proves |
|---|---|---|
| Foundation | `tests/test_common.py` | config shape, determinism, token fallback, **PII true positives and no false positives on financial figures**, JSONL integrity |
| Stage 1 | `tests/test_ingest.py` | boilerplate removal, node stream shape, table/accordion extraction, PII quarantine |
| Stage 2 | `tests/test_chunking.py` | budget compliance, sentence-boundary overlap, **fact label/value never separated**, idempotent IDs, metadata schema complete |
| Stage 3 | `tests/test_embedding.py` | dimensionality 384, L2-normalised, content-hash cache hit/miss, only-new-chunks-embedded |
| Stage 4 | `tests/test_store.py` | upsert idempotency, metadata filter, rebuild |
| Stages 5–6 | `tests/test_retrieval.py`, `tests/test_guardrails.py` | RRF/MMR maths, out-of-corpus threshold, **every refusal path**, validator rejects bad generations |
| End-to-end | `eval/run_eval.py` | the 30-question golden set + PRD §15.2 thresholds |

`eval/run_eval.py` is the release gate. Any change to chunk size, embedding
model, or prompt must ship with an updated `eval/report.md` diff.

---

## 11. Deviations from the PRD (and why)

| PRD said | This design does | Why |
|---|---|---|
| `ingest/`, `embed/`, `store/`, `rag/` with no shared module | adds root-level `common.py` | config/paths/hashing/PII are needed by all four stage groups; duplicating them would be worse. Keeps Stage 1–2 testable without torch/chromadb installed. |
| Stage 2 = "recursive / semantic, Cursor decides" | chose `structure_first_hybrid` and recorded it as ADR-001 | ADR-001 §1; required a *node stream* from Stage 1, which is why `clean.py` is more than `get_text()` |
| Rerank "optional" | default off, feature flag in config | a cross-encoder is ~90 MB and adds build time; the demo should work on first run |
| LangChain permitted only as a comparison branch | hand-rolled pipeline in the core path | P9 — this is a learning artefact; every stage should be readable in one sitting |

---

*End of ARCHITECTURE. Build order and per-phase instructions: `implementation.md`.*
