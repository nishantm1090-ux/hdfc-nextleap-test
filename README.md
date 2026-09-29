# HDFC Mutual Fund FAQ Assistant

**Live:** <https://hdfc-nextleap-test.onrender.com/>

A retrieval-augmented question-answering assistant over five HDFC Mutual Fund
(HDFC AMC) schemes. It answers factual questions — expense ratio, minimum SIP,
exit load, lock-in, benchmark, NAV, AUM, riskometer level — from the AMC's own
published scheme pages, cites the one page it used, and refuses everything else
politely, with a reason.

It **does not recommend, compare or rate schemes, and it does not compute or
report returns.** That is not a disclaimer bolted on at the end; it is the
constraint the design is built around.

> **Facts-only. No investment advice.** This assistant shares publicly available
> factual information about 5 HDFC mutual fund schemes (Direct Growth plans) from
> the sources linked in each answer. It does not recommend, compare, or rate
> schemes, and it does not compute or report returns. Mutual fund investments are
> subject to market risks; read all scheme-related documents carefully. Sources:
> HDFC Mutual Fund, SEBI and AMFI.

---

## The AMC and the schemes

**HDFC Mutual Fund** (HDFC Asset Management Company Limited, HDFC AMC) is a
mutual fund asset manager registered with SEBI. Every fact this assistant returns
comes from HDFC Mutual Fund's own site. The regulator (SEBI) and the industry
body (AMFI) are named in the disclaimer because they are the authorities a reader
can check against — not because either site is in the corpus.

The corpus is exactly these five schemes, **Direct Growth** option only:

| Scheme | Category | Benchmark (as published) | Riskometer |
|---|---|---|---|
| HDFC Large Cap Fund – Direct Growth | Large Cap | NIFTY 100 TRI | Very High |
| HDFC Flexi Cap Fund – Direct Growth | Flexi Cap | NIFTY 500 TRI | Very High |
| HDFC ELSS Tax Saver Fund – Direct Growth | Equity (ELSS) | NIFTY 500 TRI | Very High |
| HDFC Small Cap Fund – Direct Growth | Small Cap | BSE 250 SmallCap TRI | Very High |
| HDFC Balanced Advantage Fund – Direct Growth | Balanced Advantage | NIFTY 50 Hybrid Composite Debt 50:50 Index | Moderate |

A sixth page — HDFC Mutual Fund's Consolidated Account Statement page — is in
the index so that statement questions can be answered with a hand-off link rather
than a shrug. It carries no scheme facts.

## What it answers, and what it refuses

**Answers**, each with exactly one source link and a `Last updated from sources:`
date, in at most three sentences:

- expense ratio / TER, and whether it is charged on a direct plan
- minimum SIP, minimum lump-sum investment
- exit load, entry load
- ELSS lock-in period
- benchmark index
- NAV, AUM / fund size
- riskometer level
- how to download a consolidated account statement (as a hand-off, not as data)

**Refuses**, each with a stated reason and no citation:

| Refusal | Example |
|---|---|
| `refusal_advice` | "Should I buy HDFC Large Cap Fund?" |
| `refusal_performance` | "What return did HDFC Flexi Cap give last year?" |
| `refusal_pii` | any PAN, Aadhaar, account number or email |
| `refusal_statement` | "What were my capital gains?" |
| `refusal_future` | "What will NAV be next month?" |
| `out_of_corpus` | "What is the price of gold in Mumbai?" |
| `refusal_ask_scheme` | "What is the exit load?" with no scheme named |
| `refusal_error` | a genuine internal failure, shown as such |

The refusal the assistant uses when it simply cannot find the fact is
verbatim: **"I couldn't verify that from the available official sources."**

Two refusals deserve naming, because both used to be wrong answers:

- **Fund manager.** The AMC does not publish the fund manager's name on a scheme
  page, so a fund-manager question is refused rather than guessed at.
- **Portfolio P/E.** The AMC does not publish a portfolio P/E on a scheme page, so
  it is refused. Same for the Direct-vs-Regular distinction as a *fact*.

---

## How it works, in plain terms

A conventional RAG pipeline has five steps. This one has eight, and every step
writes a file the next step reads, so you can inspect and re-run any of them on
its own.

```
fetch → clean → chunk → embed → store → retrieve → answer → check
```

1. **fetch** — download the six HDFC Mutual Fund pages, hash each one, and record
   when it was read.
2. **clean** — throw away the navigation, menus and scripts, and keep only the
   content nodes. Any node that looks like it holds personal data is quarantined
   rather than indexed.
3. **chunk** — split each page into small, self-contained pieces: one chunk per
   `label: value` fact (expense ratio, NAV, benchmark…), plus a few longer
   passages for the prose sections. 68 chunks in total, 46 of them atomic facts.
4. **embed** — turn every chunk into a 384-dimension vector with a small
   sentence-transformers model, and cache the vectors by content hash so a
   re-index costs nothing.
5. **store** — write the vectors and their metadata into a local ChromaDB index.
6. **retrieve** — for a question: expand synonyms, search the vector index *and* a
   BM25 keyword index, fuse the two rankings, then **gate on term coverage** —
   a chunk is only admissible if enough of the question's own words actually
   appear in it. Near-duplicates are dropped with MMR.
7. **answer** — read the winning chunk and state its fact in a sentence. The
   default generator is *extractive*: it quotes the AMC's own `label: value`
   text rather than writing new prose, so a wrong answer requires a bug, not a
   hallucination.
8. **check** — every answer passes through guardrails that enforce the PII,
   advice, performance, groundedness and length rules, plus a named-scheme filter
   so a question about one scheme can never be answered with another's figure. A
   violation rejects the answer and the next chunk is tried.

Everything runs on CPU with no API key. An OpenAI-compatible or Ollama generator
is optional and is described below.

---

## Setup

Python 3.12 on Windows PowerShell.

```powershell
# 1. environment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. build the index. Each stage is idempotent; --force rebuilds from scratch.
.\.venv\Scripts\python.exe -m ingest.fetch     # 7 pages -> data/corpus.jsonl
.\.venv\Scripts\python.exe -m ingest.clean     # drop nav/menus/scripts
.\.venv\Scripts\python.exe -m ingest.chunk     # -> 68 chunks
.\.venv\Scripts\python.exe -m embed.index      # -> 68 vectors, 384-dim
.\.venv\Scripts\python.exe -m store.chroma_store --rebuild
```

> Always invoke `.\.venv\Scripts\python.exe`. A bare `python` picks up a
> different interpreter without the installed dependencies.

### Environment variables

There is **no required environment variable and no API key.** The defaults run
the whole system offline and reproducibly. Every variable below is optional.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `stub` | `stub` (extractive) \| `ollama` \| `openai` |
| `LLM_MODEL` | — | model name, for the `ollama` / `openai` providers |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | local Ollama server |
| `OPENAI_BASE_URL` | — | any OpenAI-compatible endpoint (Groq, Together, OpenRouter…) |
| `OPENAI_API_KEY` | — | **secret.** Put it in `.env`, which is git-ignored. Never commit it. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | override only if you re-export the ONNX graph |
| `CHROMA_DIR` | `data/chroma` | where the index lives |
| `TOP_K` | — | how many chunks to retrieve before gating |

Copy `.env.example` to `.env` if you want any of them. `.env` is git-ignored.

### Run it locally

```powershell
# the UI
.\.venv\Scripts\python.exe -m streamlit run app.py

# one question, from the command line
.\.venv\Scripts\python.exe -m rag.answer "What is the expense ratio of HDFC Large Cap Direct Growth?"

# retrieval only, with a full trace of the query pipeline
.\.venv\Scripts\python.exe -m rag.retrieve "What is the exit load of HDFC ELSS Tax Saver?" --trace

# both demos
.\.venv\Scripts\python.exe -m rag.answer --demo
.\.venv\Scripts\python.exe -m rag.retrieve --demo
```

## How to test the retrieval

Three commands, in increasing order of bluntness.

**1. The demos, ranked, with no assertions.**

```powershell
.\.venv\Scripts\python.exe -m rag.retrieve --demo    # 5 queries, retrieval only
.\.venv\Scripts\python.exe -m rag.answer --demo     # the full 12-question set, end to end
```

The first prints each query with its retrieved chunks, both scores, and the terms
that matched — the pipeline's view, before any answer is written. The second runs
the full 12-question set through every stage including the guardrails, and is the
one to read if you want to see what the assistant actually says.

**2. One question, with a full trace of the query pipeline.**

```powershell
.\.venv\Scripts\python.exe -m rag.retrieve "What is the exit load of HDFC ELSS Tax Saver?" --trace
```

The trace is the point. It shows the **normalised** query, the
**synonym-expanded** query, the **BM25 top 5** before fusion, and then a table of
every retrieved chunk with its `dense`, `bm25`, `rrf` and `final` scores and the
matched terms that earned it its place. "It gave the wrong answer" is
unactionable; "BM25 ranked the right chunk #1 and MMR dropped it as a
near-duplicate" is a fix.

**3. The tests — 63 of them, all on the retrieval layer.**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
```

### Using a real LLM (optional)

```powershell
Copy-Item .env.example .env    # then set LLM_PROVIDER and the matching key
```

The system prompt in `rag/prompts.py` carries 8 rules, every one of which a test
asserts is present, so a provider swap cannot quietly weaken the constraints.
Whatever a provider returns still passes through the guardrails — a real LLM does
**not** bypass the groundedness, length or performance checks. If it violates
one, the answer is rejected and the next chunk is tried.

---

## Reading the retrieval scores

The three commands above print this table. What each column is for:

| Column | Meaning | Role |
|---|---|---|
| `dense` | cosine similarity to the question's embedding | floor only (`min_cosine: 0.20`) |
| `bm25` | lexical overlap with the question's words | floor only (`min_bm25: 1.0`) |
| `matched` | the question's terms this chunk actually contains | **the real gate** |
| `rrf` / `final` | reciprocal-rank fusion, then MMR re-rank | ordering |

**Term coverage is the gate, not similarity.** This was the most important
finding of the build, and it is worth understanding before you judge a result:

```
"exit load"  (in corpus)   best dense 0.159
"lock-in"    (in corpus)   best dense 0.144
"price of gold in Mumbai"   best dense 0.310   <- out of corpus
"mumbai flat price"         best dense 0.347   <- out of corpus
```

**No similarity threshold can separate those two groups.** A question sharing no
words with a chunk can still be moderately similar to it, and a question whose
exact words *are* in the corpus can be weakly similar. Anything above ~0.15
refuses real questions; anything below ~0.35 admits nonsense. So `min_cosine` was
lowered to 0.20 and demoted to a *floor only*, and the accept/reject decision
moved to `min_coverage: 0.5` plus an explicit list of concepts the corpus is
known not to contain.

See the gate fire on a question that should be refused:

```powershell
.\.venv\Scripts\python.exe -m rag.retrieve "What is the price of gold in Mumbai?" --trace
# OutOfCorpus: no chunk covers enough of the question. The best match scored coverage 0.00.
```

---

## How it is put together

Hand-rolled — no LangChain, no agent framework.

| Module | What it owns |
|---|---|
| `ingest/fetch.py` | fetch 7 pages, SHA-256 each, cache by age |
| `ingest/clean.py` | DOM → content nodes, PII scrub, quarantine, corpus gate |
| `ingest/chunk.py` | atomic fact chunks + heading-aware prose split |
| `embed/index.py` | 68 vectors, content-hash cache, offline-capable ONNX runtime |
| `store/chroma_store.py` | upsert, persist, gap/orphan audit |
| `rag/retrieve.py` | hybrid search, coverage gate, MMR, scheme filter |
| `rag/answer.py` | extraction, generation, validation, the `Answer` object |
| `rag/guardrails.py` | every refusal rule and the self-test |
| `rag/prompts.py` | the system prompt and its 8 enforced rules |
| `app.py` | Streamlit UI |
| `eval/` | golden set, runner, report |

**Chunking** is `structure_first_hybrid`: atomic `label: value` facts capped at
120 tokens, plus a heading-aware recursive split (target 1000, cap 1100,
135-token overlap snapped to a sentence boundary) for prose.

**The corpus gate is real.** Seven pages are fetched; one is *rejected* by
`ingest/clean.py` for having no fact-bearing nodes — the AMFI "Mutual Fund Basics"
page turns out to be a navigation shell. It is recorded as rejected in
`data/chunk_stats.json` rather than quietly indexed, so a reader can see it was
considered and judged, not overlooked.

**Performance.** The app is deliberately lean:

- The page paints before the model loads. The embedding model runs as an ONNX
  graph on onnxruntime, not PyTorch: a cold process imports in ~0.3 s and peaks
  around 135 MB RSS instead of torch's ~560 MB, which OOM-killed the free tier.
- A background warm-up opens the model sessions and the Chroma client once per
  process, in parallel with the user reading the page. A warm-up failure is
  ignored and the lazy path simply pays it on the first question, so behaviour
  never changes.
- Config, the BM25 index, the Chroma client, the model and per-text query
  embeddings are all cached at process level across Streamlit reruns. A warm
  question costs ~0.1 s.
- No heavy imports at UI-import time.

Look at the data at any point:

```powershell
.\.venv\Scripts\python.exe -m tools.dump
```

writes `data/dump/{chunks,embeddings,store,corpus}.txt` — the raw chunks with
their metadata, the actual 384-float vectors, and the store contents. Read-only.

---

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m eval.run_eval
```

| Metric | Value | n |
|---|---|---|
| Answerable — answered at all | 100.0% | 31 |
| Answerable — exactly one citation | 100.0% | 31 |
| Answerable — correct source page | 100.0% | 31 |
| Answerable — correct figure (grounded) | 100.0% | 31 |
| Answerable — correct figure (strict) | 100.0% | 23 |
| Must-refuse — refused for the right reason | 100.0% | 27 |
| Mean latency | 114.5 ms | 58 |

**31/31 answerable** and **27/27 refused** for the right reason, out of 58
questions, with 100% groundedness. Per-question table in
[`eval/report.md`](eval/report.md).

The two numbers are never averaged. A system that answers everything scores 100%
on the first and 0% on the second; one that refuses everything does the reverse.
A refusal counts as a pass only when the refusal *kind* matches the expected one
exactly — refusing a performance question as PII is a different, and much worse,
failure than answering it.

**Strict** groundedness drops expected values with fewer than 4 significant
characters. `100` is a substring of `1,100` and `100.5`, so a substring check on
a three-character figure proves much less than on a nine-character one. Both
columns are reported; the strict one is the number to quote.

---

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 603 tests
```

| Suite | Tests | What it pins |
|---|---|---|
| `test_guardrails.py` | 86 | PII, advice, performance, groundedness |
| `test_answer.py` | 91 | the `Answer` contract, extraction order, every refusal |
| `test_retrieval.py` | 63 | the coverage gate, MMR, scheme filter, scope |
| `test_common.py` | 53 | config, paths, PII masking, logging, inline-source stripping |
| `test_store.py` | 45 | persistence, gap/orphan audit, segment purge |
| `test_chunking.py` | 42 | chunk shapes, idempotence, the corpus gate |
| `test_ui.py` | 36 | real Streamlit `AppTest` execution, 3 chips, one citation |
| `test_golden_regressions.py` | 34 | seven bugs the golden set caught |
| `test_embedding.py` | 29 | dimensionality, cache hits, offline mode |
| `test_delivery.py` | 27 | the documents against the live pipeline |
| `test_ingest.py` | 23 | fetch, clean, PII scrub, quarantine |
| `test_eval.py` | 23 | the harness, and the golden set against the corpus |
| `test_official_copy.py` | 16 | official-only copy, store-level scheme filter, ask-which-scheme, statements, future speculation |
| `test_multi_scheme.py` | 15 | one scheme per question, path-free error refusals, distinct-URL validation |
| `test_memory_context.py` | 12 | terse follow-ups resolve to the last-named scheme |
| `test_llm_providers.py` | 8 | provider wiring and fallbacks |

`test_golden_regressions.py` is worth knowing about. All seven bugs it covers
were invisible to the unit tests, to the demo questions, and to reading the code.
Nothing failed until the whole pipeline was asked all its questions at once — so
those tests are kept in their own file, dated, rather than folded in where their
origin would be invisible.

---

## Deliverables

| What | Where |
|---|---|
| **Live app** | <https://hdfc-nextleap-test.onrender.com/> |
| **Official source library** | [`sources.csv`](sources.csv) — 23 verified URLs, HDFC MF / SEBI / AMFI only |
| **Sample Q&A** | [`sample_qa.md`](sample_qa.md) |
| **Disclaimer** | [`disclaimer.md`](disclaimer.md) |
| PRD | [`PRD.md`](PRD.md) |
| Architecture & build findings | [`docs/architecture.md`](docs/architecture.md) |
| Phase-by-phase build log | [`docs/implementation.md`](docs/implementation.md) |
| Ingestion audit trail | [`docs/SOURCES.md`](docs/SOURCES.md) · [`docs/SOURCES.csv`](docs/SOURCES.csv) |
| Sample Q&A, generated live from the pipeline | [`docs/SAMPLE_QA.md`](docs/SAMPLE_QA.md) |
| Disclaimer, generated from the code's own string | [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md) |
| Eval report | [`eval/report.md`](eval/report.md) |

`docs/SAMPLE_QA.md`, `docs/SOURCES.md` and `docs/DISCLAIMER.md` are **generated**,
not hand-written, so they cannot drift from the code:

```powershell
.\.venv\Scripts\python.exe -m tools.make_sample_qa
.\.venv\Scripts\python.exe -m tools.make_sources
.\.venv\Scripts\python.exe -m tools.make_disclaimer
```

`sources.csv` at the repo root is the **source library** — 23 official URLs that
were opened and confirmed to resolve, including each scheme's Scheme Information
Document and Key Information Memorandum as published by HDFC Mutual Fund. It is
broader than the index on purpose: it is the reading list, not the ingestion
manifest. `docs/SOURCES.csv` is the ingestion manifest.

---

## Configuration

`config/app.yaml` holds every tunable — retrieval thresholds, guardrail
vocabularies, chunk sizes, generation settings. `config/sources.yaml` holds the
URLs, the scheme aliases, and the corpus gate. Adding a URL to `sources.yaml` is
the only way a page can enter the index.

Two conventions worth knowing, both learned the hard way:

- **A typo'd config key reads as its default and silently disables a feature.**
  `mmr_lambda` once read as `0.0` via `.get()` and silently turned MMR off.
  Load-bearing keys are asserted in tests.
- **Anything added to `known_absent_terms` or `_COMPOUND_CHARGES` must be
  measured against the corpus first**, and a test fails if a re-fetch makes the
  claim false.

---

## Known limitations

Stated plainly, because a demo that oversells itself is worse than one that does
not.

- **A scraped snapshot is not a scheme document.** 68 chunks from six HDFC Mutual
  Fund web pages, captured 29 September 2026. NAV, AUM and riskometer levels move
  without notice, and the AMC edits these pages in place — the URL never changes
  and nothing announces the edit. The `Last updated from sources:` stamp on every
  answer exists so a reader can see how old the snapshot is. For anything
  binding, the scheme's own factsheet governs.
- **The AMC's WAF blocks programmatic fetching.** Every `hdfcfund.com` page
  returns HTTP 403 to a plain HTTP client, so the six pages were captured through
  a real browser. That is a real limitation of the ingestion path, not a
  workaround that can be forgotten: a plain `python -m ingest.fetch` on a clean
  machine will get 403s, and the committed `data/` artefacts are what the
  deployed app runs on. The `sources.csv` URLs were each confirmed to load in a
  browser.
- **The AMC publishes no star rating and no portfolio P/E on a scheme page**, so
  the assistant cannot answer either. It refuses rather than substituting a
  neighbouring fact. `known_absent_terms` names what it is known to lack, each
  measured at 0 occurrences in the corpus.
- **No fund-manager name.** The AMC does not publish the fund manager's name on a
  scheme page, so a fund-manager question is refused. The AMC publishes no
  manager name anywhere in the captured pages, so there is nothing to retrieve.
- **`minimum_investment` exists for one scheme only.** HDFC Small Cap publishes a
  minimum lump-sum investment figure; the other four pages do not. Asking for the
  minimum investment of HDFC Large Cap is refused rather than answered with the
  minimum SIP, which is a different number and an easy thing to confuse.
- **HDFC Balanced Advantage's exit load is prose, not a fact row.** Its 15%
  free-exit-load slab is 129 tokens, over the 120-token cap for an atomic fact, so
  it lands in a prose chunk. The answer is still correct and cited, but it comes
  from a different extraction path than the other four schemes' exit loads.
- **Direct-vs-Regular is refused.** The pages do not state the plan variant as a
  fact, so "is this the Direct or the Regular plan?" is refused rather than
  answered.
- **No returns, ever.** The pages do publish historical returns, and the assistant
  deliberately does not read or report them. A return needs a start date, an end
  date and a price source, and choosing them is an act of authority. Performance
  questions are refused and hand off to the published factsheet.
- **Only HDFC, and only Direct Growth plans.** A question about another AMC's
  Large Cap fund is refused, even though the fact shapes are identical — which is
  exactly why the mix-up would be easy.
- **The generator quotes; it does not reason.** With the default `stub`, an answer
  is the AMC's own text shaped into a sentence. That is a feature for
  auditability and a limitation for fluency.
- **The app never writes to `data/`.** The index is built and committed, and the
  deployed instance serves from it. A UI action cannot change the corpus, so
  nothing a user types can reach the index.
- **One source per answer, always.** Every answer cites exactly one page. A
  question that genuinely spans two schemes is refused as ambiguous rather than
  answered from two pages.

---

## Product reference and independence

The assignment listed **Groww** as the product reference. This assistant is an
**independent rebuild** for that assignment. It is not affiliated with, endorsed
by, or connected to Groww in any way, and Groww's product, site and data are not
used anywhere in this project.

The corpus, every citation, every source link and the entire reference library in
`sources.csv` are **HDFC Mutual Fund, SEBI and AMFI only** — the official
publishers of the facts being answered. No broker, aggregator, blog or forum
contributes a single fact.
