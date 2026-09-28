# HDFC AMC Mutual Fund FAQ Assistant

A retrieval-augmented question-answering demo over **5 HDFC AMC mutual fund
schemes (Direct Growth)**. It answers factual questions from five published
pages, cites the one page it used, and refuses everything else — politely, with
a reason.

It **does not recommend, compare or rate schemes, and it does not compute or
report returns.** That is not a disclaimer bolted on at the end; it is the
constraint the design is built around. See [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md).

> **Facts-only. No investment advice.** This assistant shares publicly available
> factual information about 5 HDFC mutual fund schemes (Direct Growth plans) from
> the sources linked in each answer. It does not recommend, compare, or rate
> schemes, and it does not compute or report returns. Mutual fund investments are
> subject to market risks; read all scheme-related documents carefully. Sources:
> HDFC AMC, Groww, SEBI, AMFI.

---

## Quick start

```powershell
# 1. environment (Python 3.12)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. build the index  (each stage is idempotent; --force rebuilds)
.\.venv\Scripts\python.exe -m ingest.fetch          # 7 pages -> data/corpus.jsonl
.\.venv\Scripts\python.exe -m ingest.clean          # -> 204 nodes
.\.venv\Scripts\python.exe -m ingest.chunk --stats  # -> 121 chunks
.\.venv\Scripts\python.exe -m embed.index           # -> 121 vectors, 384-dim
.\.venv\Scripts\python.exe -m store.chroma_store --rebuild

# 3. ask something
.\.venv\Scripts\python.exe -m rag.answer "What is the expense ratio of HDFC Large Cap Direct Growth?"

# 4. the UI
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No API key is needed. The default generator is **extractive** — it quotes the
source page rather than writing prose — so the whole system runs offline and
reproducibly. See [Using a real LLM](#using-a-real-llm-optional) if you want one.

> Always invoke `.\.venv\Scripts\python.exe`. A bare `python` will pick up a
> different interpreter without the installed dependencies.

---

## How to test the retrieval

Three commands, in increasing order of bluntness.

**1. The demo — 5 questions, ranked, no assertions.**

```powershell
.\.venv\Scripts\python.exe -m rag.retrieve --demo
```

Prints each question with its retrieved chunks, both scores, and the terms that
matched. For the full 12-question set through the answer layer instead, use
`python -m rag.answer --demo`.

**2. One question, with a full trace of the query pipeline.**

```powershell
.\.venv\Scripts\python.exe -m rag.retrieve "What is the exit load of HDFC ELSS Tax Saver?" --trace
```

The trace is the point. It shows the **normalised** query, the **synonym-expanded**
query, the **BM25 top 5** before fusion, and then a table of every retrieved
chunk with `dense`, `bm25`, `rrf`, `final` and the `matched` terms that earned
it its place.

**3. The tests — 63 of them.**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
```

**Reading the numbers.**

| Column | Meaning | Role |
|---|---|---|
| `dense` | cosine similarity to the question's embedding | floor only (`min_cosine: 0.20`) |
| `bm25` | lexical overlap with the question's words | floor only (`min_bm25: 1.0`) |
| `matched` | the question's terms this chunk actually contains | the real gate |
| `rrf` / `final` | reciprocal-rank fusion, then MMR re-rank | ordering |

**Term coverage is the gate, not similarity.** This was the most important
finding in Stage 5, and it is worth understanding before you judge a result:

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
moved to term coverage plus an explicit list of concepts the corpus is known not
to contain.

See the gate fire on a question that should be refused:

```powershell
.\.venv\Scripts\python.exe -m rag.retrieve "What is the price of gold in Mumbai?" --trace
# OutOfCorpus: no chunk covers enough of the question. The best match scored coverage 0.00.
```

**And a case the gate alone cannot catch** — worth knowing, because it is the
most dangerous failure this system can produce:

```powershell
.\.venv\Scripts\python.exe -m rag.retrieve "What is the riskometer level of HDFC Large Cap?" --trace
```

This one is *retrieved successfully* — the right AMC, the right scheme, and
"level" and "large cap" give it coverage 0.67. But "riskometer" occurs **0 times**
in all 121 chunks, so there is no riskometer level on those pages. Left to
retrieval alone the system answered with the **expense ratio**: a confident,
cited, wrong number. It is stopped one layer later, by `known_absent_terms` in
`config/app.yaml` (a post-retrieval check in `rag/answer.py`), because answering
needs to know what was retrieved, not just what was asked.

---

## What it does

| | |
|---|---|
| **Corpus** | 5 indexable pages, 121 chunks, ~43 tokens each |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, unit-norm |
| **Vector store** | ChromaDB, cosine, persisted to `data/chroma/` |
| **Retrieval** | hybrid dense + BM25, RRF fusion, term-coverage gate, MMR diversity |
| **Generation** | extractive by default; OpenAI-compatible or Ollama optional |
| **Guardrails** | PII, advice, performance, out-of-scope, groundedness, length |
| **Eval** | 54 questions, answerable and must-refuse scored **separately** |

Current eval: **31/31 answerable** (100% cited, 100% correct source, 100%
grounded) and **23/23 refused for the right reason**. Full numbers and the
per-question table in [`eval/report.md`](eval/report.md).

The two numbers are never averaged. A system that answers everything scores 100%
on the first and 0% on the second; one that refuses everything does the reverse.
A refusal counts as a pass only when the `kind` matches the expected one exactly —
refusing a performance question as PII is a different, and much worse, failure
than answering it.

---

## How it is put together

Six stages, hand-rolled — no LangChain, no agent framework. Each stage writes a
file the next one reads, and each can be run and inspected on its own.

```
fetch → clean → chunk → embed → store → retrieve → generate → validate
 Stage 1  Stage 1   Stage 2  Stage 3   Stage 4   Stage 5    Stage 6    Stage 6
```

| Module | Stage | What it owns |
|---|---|---|
| `ingest/fetch.py` | 1 | fetch 7 pages, SHA-256 each, cache by age |
| `ingest/clean.py` | 1 | DOM → 204 text nodes, PII scrub, quarantine |
| `ingest/chunk.py` | 2 | atomic fact chunks + heading-aware prose split |
| `embed/index.py` | 3 | 121 vectors, content-hash cache, offline-capable |
| `store/chroma_store.py` | 4 | upsert, persist, gap/orphan audit |
| `rag/retrieve.py` | 5 | hybrid search, coverage gate, MMR, scheme filter |
| `rag/answer.py` | 6 | extraction, generation, validation, the `Answer` object |
| `rag/guardrails.py` | 6 | every refusal rule and the self-test |
| `rag/prompts.py` | 6 | the system prompt and its 8 enforced rules |
| `app.py` | 7 | Streamlit UI |
| `eval/` | 8 | golden set, runner, report |

**Chunking** is `structure_first_hybrid`: atomic `label: value` facts capped at
120 tokens, plus a heading-aware recursive split (target 1000, cap 1100,
135-token overlap snapped to a sentence boundary) for prose.

**Answers are extractive by default.** The stub quotes the page's own
`label: value` text rather than paraphrasing it, so a wrong answer requires a bug
rather than a hallucination. This is why the eval can assert exact figures.

### Looking at the data

Every stage's output is dumped as readable text:

```powershell
.\.venv\Scripts\python.exe -m tools.dump
```

writes `data/dump/{chunks,embeddings,store,corpus}.txt` — the raw chunks with
their metadata, the actual 384-float vectors, the store contents, and the
extracted corpus. Read-only; it never touches pipeline inputs.

---

## Using a real LLM (optional)

Copy `.env.example` to `.env` and set a provider. **`.env` is git-ignored and
must never be committed.**

```powershell
Copy-Item .env.example .env
```

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `stub` *(default)* | nothing | extractive, offline, fully reproducible |
| `ollama` | a local Ollama server | `OLLAMA_BASE_URL`, `LLM_MODEL` |
| `openai` | an OpenAI-compatible endpoint | `OPENAI_API_KEY` — set `OPENAI_BASE_URL` for Groq, Together, OpenRouter, etc. |

The system prompt in `rag/prompts.py` carries 8 rules, every one of which is
asserted present by a test, so a provider swap cannot quietly weaken the
constraints. Whatever the provider returns still passes through
`guardrails.validate_answer` — a real LLM does **not** get to bypass the
groundedness, length or performance checks. If it violates them, the answer is
rejected and the next chunk is tried.

---

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # 578 tests
```

| Suite | Tests | What it pins |
|---|---|---|
| `test_guardrails.py` | 86 | PII, advice, performance, groundedness |
| `test_answer.py` | 83 | the `Answer` contract, extraction order, every refusal |
| `test_retrieval.py` | 63 | the coverage gate, MMR, scheme filter, scope |
| `test_common.py` | 52 | config, paths, PII masking, logging, inline-source stripping |
| `test_store.py` | 45 | persistence, gap/orphan audit, segment purge |
| `test_chunking.py` | 42 | chunk shapes, idempotence, the corpus gate |
| `test_ui.py` | 36 | real `AppTest` script execution, 3 chips, one citation |
| `test_golden_regressions.py` | 34 | six bugs the golden set caught |
| `test_embedding.py` | 29 | dimensionality, cache hits, offline mode |
| `test_ingest.py` | 23 | fetch, clean, PII scrub, quarantine |
| `test_eval.py` | 23 | the harness, and the golden set against the corpus |
| `test_memory_context.py` | 12 | retrieval memory: terse follow-ups resolve to the last-named scheme |
| `test_multi_scheme.py` | 15 | one scheme per question (PRD §11: exactly one source), path-free error refusals, distinct-URL validation |

`test_golden_regressions.py` is worth knowing about. All six bugs it covers were
invisible to 420 unit tests, to the demo questions, and to reading the code.
Nothing failed until 54 questions were asked of the whole pipeline at once — so
the tests are kept in their own file, dated, rather than folded in where their
origin would be invisible.

---

## Deliverables

| What | Where |
|---|---|
| Prototype | `streamlit run app.py` |
| PRD | [`PRD.md`](PRD.md) |
| Architecture & build findings | [`docs/architecture.md`](docs/architecture.md) |
| Phase-by-phase build log | [`docs/implementation.md`](docs/implementation.md) |
| Source list | [`docs/SOURCES.md`](docs/SOURCES.md) · [`docs/SOURCES.csv`](docs/SOURCES.csv) |
| Sample Q&A (generated live) | [`docs/SAMPLE_QA.md`](docs/SAMPLE_QA.md) |
| Disclaimer | [`docs/DISCLAIMER.md`](docs/DISCLAIMER.md) |
| Eval report | [`eval/report.md`](eval/report.md) |

`docs/SAMPLE_QA.md`, `docs/SOURCES.md` and `docs/DISCLAIMER.md` are **generated**,
not hand-written, so they cannot drift from the code:

```powershell
.\.venv\Scripts\python.exe -m tools.make_sample_qa
.\.venv\Scripts\python.exe -m tools.make_sources
.\.venv\Scripts\python.exe -m tools.make_disclaimer
```

---

## Known limitations

Stated plainly, because a demo that oversells itself is worse than one that does
not.

- **A scraped snapshot is not a scheme document.** 121 chunks from five web
  pages. NAV, AUM and ratings move without notice. The `Last updated` stamp
  exists so a reader can see how old the snapshot is.
- **The corpus is thin by design.** It holds the facts those five pages publish.
  It does not hold riskometer levels, holdings, fund-of-funds detail, or
  anything requiring a computation. `known_absent_terms` in `config/app.yaml`
  names the concepts it is known to lack, each measured at 0 occurrences, so the
  assistant says "not in my pages" instead of answering with a neighbouring fact.
- **Only HDFC, and only Direct Growth plans.** A question about SBI's Large Cap
  is refused, even though the two funds' fact shapes are byte-identical — which
  is exactly why the mix-up would be easy.
- **The generator quotes; it does not reason.** With the default `stub`, an
  answer is the page's own text shaped into a sentence. That is a feature for
  auditability and a limitation for fluency.
- **The corpus pages are Groww's, not HDFC AMC's.** Groww is a broker publishing
  the AMC's figures. For anything binding, the scheme's own factsheet governs.

---

## Configuration

`config/app.yaml` holds every tunable — retrieval thresholds, guardrail
vocabularies, chunk sizes, generation settings. `config/sources.yaml` holds the
seven URLs, the scheme aliases, and the corpus gate.

Two conventions worth knowing, both learned the hard way:

- **A typo'd config key reads as its default and silently disables a feature.**
  `mmra_lambda` once read as `0.0` via `.get()` and silently turned MMR off.
  Load-bearing keys are asserted in tests.
- **Anything added to `known_absent_terms` or `_COMPOUND_CHARGES` must be
  measured against the corpus first**, and a test fails if a re-fetch makes the
  claim false.
