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
.\.venv\Scripts\python.exe -m pytest -q          # 610 tests
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

## Source list

All 23 URLs are official publishers only — **HDFC AMC**, **SEBI**, **AMFI**. No
broker, aggregator, blog or forum contributes a single fact. Each was opened and
confirmed to load on 29 September 2026.

**In the corpus** (`in_corpus: yes`) — these 6 are what the assistant actually
retrieves from:

| Ref | Source | Publisher | Type | URL |
|---|---|---|---|---|
| S01 | HDFC Large Cap Fund — Direct Plan | HDFC AMC | scheme page | <https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct> |
| S02 | HDFC Flexi Cap Fund — Direct Plan | HDFC AMC | scheme page | <https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct> |
| S03 | HDFC ELSS — Tax Saver Fund Direct Plan | HDFC AMC | scheme page | <https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct> |
| S04 | HDFC Small Cap Fund — Direct Plan | HDFC AMC | scheme page | <https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct> |
| S05 | HDFC Balanced Advantage Fund Direct Plan | HDFC AMC | scheme page | <https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct> |
| S06 | Download Consolidated Account Statement | HDFC AMC | AMC service page | <https://www.hdfcfund.com/services/consolidated-account-statement> |

**Reading list** (`in_corpus: no`) — verified official, but not indexed. These are
the documents a reader should check a figure against:

| Ref | Source | Publisher | Type |
|---|---|---|---|
| S07 | Mutual Fund Glossary: A–Z Terms | HDFC AMC | glossary |
| S08 | HDFC Mutual Fund Short | HDFC AMC | explainer |
| S09 | Frequently Asked Questions for Investors | HDFC AMC | FAQ page |
| S10 | Grievances Redressal Mechanism | HDFC AMC | complaint page |
| S11 | Scheme Information Document — HDFC Large Cap Fund (21 Nov 2025) | HDFC AMC | SID (PDF) |
| S12 | Scheme Information Document — HDFC Flexi Cap Fund (21 Nov 2025) | HDFC AMC | SID (PDF) |
| S13 | Scheme Information Document — HDFC ELSS Tax Saver Fund (21 Nov 2025) | HDFC AMC | SID (PDF) |
| S14 | Scheme Information Document — HDFC Small Cap Fund (21 Nov 2025) | HDFC AMC | SID (PDF) |
| S15 | Scheme Information Document — HDFC Balanced Advantage Fund (21 Nov 2025) | HDFC AMC | SID (PDF) |
| S16 | Key Information Memorandum — HDFC Large Cap Fund (21 Nov 2025) | HDFC AMC | KIM (PDF) |
| S17 | Key Information Memorandum — HDFC Flexi Cap Fund (21 Nov 2025) | HDFC AMC | KIM (PDF) |
| S18 | Key Information Memorandum — HDFC ELSS Tax Saver Fund (21 Nov 2025) | HDFC AMC | KIM (PDF) |
| S19 | Key Information Memorandum — HDFC Small Cap Fund (21 Nov 2025) | HDFC AMC | KIM (PDF) |
| S20 | Key Information Memorandum — HDFC Balanced Advantage Fund (21 Nov 2025) | HDFC AMC | KIM (PDF) |
| S21 | Mutual Fund — AMFI | AMFI | industry body |
| S22 | NAVAll.txt — daily NAV feed for all schemes | AMFI | NAV feed |
| S23 | Mutual Funds — SEBI | SEBI | regulator |

Full machine-readable version with titles, schemes and per-URL purpose:
[`sources.csv`](sources.csv).

> **Why HDFC pages are not fetched programmatically.** The AMC's WAF returns
> HTTP 403 to non-browser clients, so `ingest/fetch.py` cannot read
> `hdfcfund.com`. These pages were captured in a real browser and written into
> `data/raw/`. This is a fetch-path constraint, not a broken page, and it is
> documented in `config/sources.yaml`.

---

## Sample Q&A

Ten questions against the running system. Every answer, source link and
`Last updated from sources` date below was produced by calling the pipeline
(`rag.answer.ask`) — not typed by hand. Full version with commentary:
[`sample_qa.md`](sample_qa.md).

**Sources: HDFC Mutual Fund, SEBI and AMFI.** Six of the ten are answered from
HDFC Mutual Fund's own scheme pages, and every cited link points there. The four
refusals hand off to SEBI and AMFI's investor-education material.

### 1. Minimum SIP

**Q:** What is the minimum SIP for HDFC Large Cap Direct Growth?

> The minimum SIP for HDFC Large Cap Fund is ₹ 100.

**Source:** <https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct>
**Last updated from sources:** 2026-09-29

### 2. Expense ratio

**Q:** What is the expense ratio of HDFC ELSS Tax Saver Direct Growth?

> The total expense ratio (TER) for HDFC ELSS Tax Saver Fund is 1.21%.

**Source:** <https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct>
**Last updated from sources:** 2026-09-29

### 3. Exit load

**Q:** What is the exit load on HDFC Large Cap if I redeem within 1 year?

> HDFC Large Cap Fund - Exit Load: In respect of each purchase/switch-in of
> Units, an Exit Load of 1.00% is payable if Units are redeemed/switched-out
> within 1 year from the date of allotment. No Exit Load is payable if Units are
> redeemed/switched-out after 1 year from the date of allotment.

**Source:** <https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct>
**Last updated from sources:** 2026-09-29

### 4. Benchmark

**Q:** What benchmark does HDFC Flexi Cap Direct Growth track?

> The Benchmark for HDFC Flexi Cap Fund is NIFTY 500 Total Returns Index.

**Source:** <https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct>
**Last updated from sources:** 2026-09-29

### 5. Riskometer

**Q:** What is the riskometer level of HDFC Small Cap Fund?

> The Riskometer for HDFC Small Cap Fund is Very High.

**Source:** <https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct>
**Last updated from sources:** 2026-09-29

### 6. ELSS lock-in

**Q:** What is the lock-in period on HDFC ELSS Tax Saver?

> The lock-in period for HDFC ELSS Tax Saver Fund is 3 years.

**Source:** <https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver-fund/direct>
**Last updated from sources:** 2026-09-29

### 7. Capital gains statement — refused

**Q:** How do I download my capital gains statement?

> I can't download or send your statements from here — that needs your account
> login and your personal data. In your HDFC Mutual Fund account
> (https://www.hdfcfund.com/) you can download your capital gains statement and
> Consolidated Account Statement. SEBI's investor-education pages
> (https://www.sebi.gov.in/) and AMFI (https://www.amfiindia.com/) explain how
> these statements work.

**Refused** — `out_of_corpus`. No citation, because nothing was retrieved. The
document exists but only behind the holder's own login, so the assistant names
*where* to get it rather than pretending to produce the data.

### 8. Investment advice — refused

**Q:** Should I buy HDFC Small Cap Fund for my retirement?

> I can share facts from the official pages I use, but I can't tell you whether
> to buy or sell a scheme — that's investment advice, and it's outside what this
> assistant does. Here's an official guide to understanding scheme risk instead:
> https://www.sebi.gov.in/ (investor education)

**Refused** — `refusal_advice`. No citation. The corpus holds all five schemes'
expense ratios, so it could rank them by cost — and that ranking would itself be
a recommendation. The numbers are available; the judgement is withheld.

### 9. Returns / comparison — refused

**Q:** How much did HDFC Small Cap Fund return last year?

> I don't compute or compare returns. For a scheme's official performance
> figures, please use the published factsheets from HDFC Mutual Fund
> (https://www.hdfcfund.com/) or AMFI (https://www.amfiindia.com/).

**Refused** — `refusal_performance`. No citation. A return needs a start date, an
end date and a price source, and choosing them is an act of authority this
assistant does not have. The refusal hands over to the published factsheet,
which does.

### 10. Personal data — refused

**Q:** My PAN is ABCDE1234F. What is the exit load on HDFC Flexi Cap?

> Please don't share personal identifiers like PAN, Aadhaar, account numbers,
> OTPs, or contact details here — I can't accept, store, or use them. Scheme
> facts are all public; ask me anything about expense ratio, exit load, SIP,
> lock-in, or statements.

**Refused** — `refusal_pii`. No citation. The identifier is not stored, not used
and not written to any file.

### Also refused — named gaps, not policy

HDFC Mutual Fund does not publish these on a scheme page, so there is nothing to
retrieve. All four return the same line, verbatim:

> I couldn't verify that from the available official sources.

| Asked | Why |
|---|---|
| Fund manager's name | no chunk in the corpus carries that `fact_key` |
| Portfolio P/E | no chunk in the corpus carries that `fact_key` |
| Direct vs Regular plan | the page does not state the plan variant as a fact |
| Minimum lump-sum (any scheme except Small Cap) | the only `minimum_investment` chunk belongs to HDFC Small Cap |

A **policy** refusal would happen even with a perfect corpus — the answer is
withheld on purpose. A **gap** refusal happens because the source does not carry
the fact, and `known_absent_terms` in `config/app.yaml` is what lets the
assistant say so instead of answering the fund-manager question with the expense
ratio.

### What the corpus can and cannot answer

68 chunks from 5 HDFC scheme pages plus the AMC's Consolidated Account Statement
page, captured 29 September 2026.

| Asked | Answered |
|---|---|
| expense ratio / TER | yes, all five schemes |
| minimum SIP | yes, all five schemes |
| exit load | yes, all five schemes |
| ELSS lock-in | yes |
| benchmark | yes, all five schemes |
| NAV, AUM | yes, all five schemes |
| riskometer level | yes, all five schemes |
| minimum lump-sum investment | Small Cap only — the only page that publishes it |
| fund manager, star rating, portfolio P/E | no — not published on a scheme page |
| Direct vs Regular plan | no — the page does not state it as a fact |
| returns, rankings, recommendations | no — withheld by policy |
| your holdings, statements, tax figures | no — requires your login |

**One source per answer, always.** A question that genuinely spans two schemes is
refused as ambiguous rather than answered from two pages.

---

## Disclaimer

This is the text exactly as the assistant shows it — under every answer, in the UI
and on the command line. Not a summary of it. Full version with the reasoning
behind each clause: [`disclaimer.md`](disclaimer.md).

**Facts-only. No investment advice.** This assistant shares publicly available factual information about 5 HDFC mutual fund schemes (Direct Growth plans) from the sources linked in each answer. It does not recommend, compare, or rate schemes, and it does not compute or report returns. Mutual fund investments are subject to market risks; read all scheme-related documents carefully. Sources: HDFC Mutual Fund, SEBI and AMFI.

Three sentences do the work, and each rules out one thing the assistant is built
not to do:

- **"It does not recommend, compare, or rate schemes."** The corpus holds the
  expense ratio of all five schemes, so it *could* rank them by cost — and that
  ranking would be a recommendation, because a lower TER is often the reason a
  person picks a scheme. The numbers are available. The comparison is withheld.
- **"It does not compute or report returns."** A return needs a start date, an
  end date and a price source. Choosing them is an act of authority, and the
  published factsheets are where that authority sits.
- **"read all scheme-related documents carefully."** Not boilerplate. The corpus is
  a snapshot of web pages captured on a single date. NAV, AUM and riskometer
  levels move without notice, and the AMC edits these pages in place without
  changing the URL, so a page can be stale with nothing announcing it. The
  `Last updated from sources:` date on every answer exists so you can see how old
  the snapshot is.

**This is an independent academic project.** It is not HDFC Mutual Fund, not
SEBI, not AMFI, and not affiliated with, endorsed by, or connected to any of
them. It is not a registered investment adviser and it is not a distributor.
Nothing here is investment advice, a recommendation, or an offer to buy or sell
any security. Read the scheme's own SID, KIM and monthly factsheet before
investing, and consider your own circumstances.

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
