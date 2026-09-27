# IMPLEMENTATION GUIDE — HDFC Mutual Fund FAQ Assistant

> How to **build** the system described in `architecture.md`, in phases.
> Read `architecture.md` for *why*; read this for *what to type*.
>
> Each phase is self-contained and gives you:
> **Goal → Files → Signatures → Algorithm → Config → Acceptance criteria → Verify → Common failures → Cursor prompt**
>
> The **Cursor prompt** at the end of each phase is a copy-paste block. Use it to
> drive the implementation, then run the **Verify** commands yourself before
> moving on. Never trust a phase that has not been run.

---

## Phase Map

| Phase | Stage | Output artefact | Depends on | State |
|---|---|---|---|---|
| **0** | — | Python 3.12 + `.venv` + deps | — | ✅ done |
| **1** | Stage 0 | `common.py`, `config/*.yaml`, tests | 0 | ✅ done |
| **2** | Stages 1–2 | `data/raw/`, `data/corpus.jsonl`, `data/chunks.jsonl`, `data/chunk_stats.json` | 1 | ⬜ |
| **3** | Stage 3 | `data/embeddings/*.npy` | 2 | ⬜ |
| **4** | Stage 4 | `data/chroma/` | 3 | ⬜ |
| **5** | Stage 5 | `rag/retrieve.py` | 4 | ⬜ |
| **6** | Stage 6 | `rag/answer.py`, `rag/guardrails.py`, `rag/prompts.py` | 5 | ⬜ |
| **7** | Stage 7 | `app.py` | 6 | ⬜ |
| **8** | QA | `eval/golden_set.csv`, `eval/report.md` | 7 | ⬜ |
| **9** | Delivery | `README.md`, `SOURCES.md`, `SAMPLE_QA.md`, demo video | 8 | ⬜ |

Phases 1→3 are the ones implemented in this repo right now. Phases 4–9 are
specified here to the same level of detail so the build is reproducible end to end.

**Rule of the build: never let a phase produce an artefact it cannot be checked against.** Every phase writes a machine-readable report, and the Verify block is the contract.

---

# PHASE 0 — Environment ✅

**Goal:** a working Python environment on CPU.

**What was done**
```
winget install Python.Python.3.12
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Always invoke the interpreter as `.\.venv\Scripts\python.exe`** (Windows). If `python` on this machine is the Microsoft Store stub, it will silently do nothing — that cost an hour.

**Installed (verified):** Python 3.12.10 · torch 2.14.0 (CPU) · sentence-transformers 6.1.0 · chromadb 1.5.9 · rank-bm25 0.2.2 · numpy 2.5.3 · beautifulsoup4 · lxml · httpx · tiktoken · PyYAML · pytest

**Acceptance:** `.\.venv\Scripts\python.exe -c "import torch, chromadb, sentence_transformers; print('ok')"` prints `ok`.

---

# PHASE 1 — Bootstrap & shared foundation ✅

**Goal:** one module that owns config, paths, hashing, tokenisation, JSONL, PII, and logging — so that Stages 1–2 are testable **without** torch or chromadb installed.

**Files created**
```
common.py               # the foundation (see architecture.md §4 "Stage 0")
config/app.yaml         # every tunable number
config/sources.yaml     # THE corpus gate: 5 scheme pages + supplementary
requirements.txt
.env.example
.gitignore
tests/test_common.py
```

**Key design points**

| Decision | Reason |
|---|---|
| `common.py` at repo root, not inside a package | all four stage groups need it; keeps `import common` trivial from a notebook or a stage script |
| Config via YAML + `lru_cache` | the corpus must be changeable as data (principle P8) |
| `content_hash()` normalises whitespace | makes the embedding cache robust to reformatting while still catching real edits |
| `chunk_id` is content-addressed | idempotent re-runs (P2) — the whole Chroma upsert story depends on it |
| tiktoken **with a fallback** | tiktoken downloads BPE ranks on first use; an air-gapped demo must not die. The fallback only ever budgets chunk size, so ~15 % drift is harmless |
| PII scan is **overlap-aware** | patterns are tried high-priority first and each match *consumes* its span, so a phone number is never also reported as a "possible OTP" |
| PII has two confidences | a bare 4–6 digit run collides with `AUM is Rs 50000 crore`. `high` blocks; `low` is reported but never blocks |
| `mask_value()` keeps 2 chars | enough for a human to debug a log line, useless to anyone reading over their shoulder |

**Acceptance criteria — all met**
- [x] `python common.py` prints a runtime report: 5 schemes, 4 supplementary sources, model, chunk strategy, tokenizer
- [x] PII self-test: 8/8 true positives caught (PAN, spaced phone, bare 10-digit phone, Aadhaar, email, account no., IFSC, DP ID)
- [x] PII self-test: 6/6 financial facts **not** flagged (expense ratio, exit load, 3-year lock-in, min SIP, AUM, benchmark)
- [x] `pytest tests/test_common.py` → **48 passed**

**Verify**
```powershell
.\.venv\Scripts\python.exe common.py
.\.venv\Scripts\python.exe -m pytest tests/test_common.py -q
```

**Common failures**
| Symptom | Cause | Fix |
|---|---|---|
| `SyntaxError: positional argument follows keyword argument` | `field(default=0, (0,0))` | use a plain tuple default, not `field()` |
| PII self-test "FAIL" on a phone number | regex assumed no separators | allow `[\s.\-]?` between every digit group |
| PII self-test "FAIL" on Aadhaar | lookbehind `(?<![\d\s])` rejected the leading **space** | use `(?<!\d)` — digit-only |
| `tiktoken` hangs on first call | downloading BPE ranks | expected once; the fallback path exists for offline runs |

---

# PHASE 2 — Loading & Chunking

**Goal:** turn 9 public URLs into a chunked, citable corpus.

**Stage 1a — `ingest/fetch.py`**

```python
@dataclass
class FetchResult:
    doc_id: str; url: str; ok: bool; status: int | None
    html: str; raw_path: Path; sha256: str; bytes: int
    fetched_at: str; from_cache: bool; error: str | None
    looks_thin: bool          # <2000 chars of text => probably a JS shell

def fetch_url(url: str, doc_id: str, *, force: bool = False) -> FetchResult: ...
def fetch_all(*, force: bool = False, only: list[str] | None = None) -> list[FetchResult]: ...
def snapshot_path(raw_dir: Path) -> Path: ...          # data/source_snapshot.json
def _is_fresh(path: Path, max_age_hours: int) -> bool: ...
```

Algorithm
1. Raw cache path = `data/raw/{doc_id}.html`. If it exists and is < 24 h old and `--force` is absent → return it with `from_cache=True`.
2. `httpx.get(url, headers={"User-Agent": ...}, timeout=30, follow_redirects=True)`.
3. Retry up to `retries` (3) with exponential backoff (2 s, 4 s, 8 s).
4. `politeness_delay_seconds` (2 s) sleep **between** requests — never in a tight loop.
5. Write bytes to the cache, SHA-256 them, append to `data/ingestion_log.jsonl`.
6. Merge into `data/source_snapshot.json` so drift is diffable.
7. **Never raise** on a single page. Return `ok=False` and carry on.

> **PRD R1 warning:** Groww is a JS-rendered SPA. After extraction, if a page's visible text is under ~2000 chars, set `looks_thin=True` and print a loud warning. Do not silently ship an empty index. Escalation: switch that doc's `url` in `sources.yaml` to the HDFC AMC scheme page (the brief permits AMC pages) and keep Groww as the citation of record.

**Stage 1b — `ingest/clean.py`**

```python
NODE_TYPES = {"heading", "paragraph", "list_item", "table_row", "accordion", "quote"}

@dataclass
class Node:
    type: str; text: str
    level: int | None = None      # heading level
    label: str | None = None      # table_row key / accordion question
    value: str | None = None      # table_row value / accordion answer
    order: int = 0

@dataclass
class Document:
    doc_id: str; source_url: str; page_title: str; publisher: str
    page_type: str; scheme_slug: str | None; scheme_name: str | None
    short_name: str | None; category: str | None; plan: str | None
    nodes: list[Node]; raw_sha256: str
    source_fetched_at: str; ingested_at: str
    pii_scan: str; node_count: int; text_chars: int

def strip_boilerplate(soup: BeautifulSoup, selectors: list[str]) -> BeautifulSoup: ...
def extract_nodes(soup: BeautifulSoup) -> list[Node]: ...
def clean_document(html: str, meta: dict) -> tuple[Document, list[dict]]: ...  # (doc, quarantined)
def clean_all(*, force: bool = False) -> list[Document]: ...
```

Algorithm
1. Parse with `lxml`. Remove every element matching `chunking.strip_selectors`. **Then sweep again** for orphaned short text nodes — deleting a `<nav>` can strand its text.
2. Walk remaining elements **in document order**, emitting typed nodes:
   - `h1..h6` → `heading` with `level`; this sets `section_heading` for everything after it
   - `p` → `paragraph` (skip if < 40 chars and has no digits — likely a fragment)
   - `li` → `list_item`
   - `<tr>` → one `table_row` per row: first cell → `label`, remainder → `value`
   - elements whose class/id matches `faq|accordion|question|answer` → `accordion` with `label`=question, `value`=answer
3. `common.scrub_pii` every text field. If a node still matches high-confidence PII after scrubbing → it is **quarantined** to `data/quarantine.jsonl` and dropped.
4. Write `data/corpus.jsonl`, one `Document` per line, with `pii_scan`, counts, and timestamps.

**Stage 2 — `ingest/chunk.py`**

```python
FACT_LABELS = {           # label regex/keywords -> canonical fact_key
    "expense_ratio": r"expense\s*ratio|\bter\b|total expense",
    "exit_load": r"exit\s*load|exit\s*charge|load\s*on\s*redemption",
    "minimum_sip": r"minimum\s*sip|min\s*sip|sip\s*amount",
    "minimum_investment": r"minimum\s*(lump\s*sum|investment|amount)",
    "lock_in_period": r"lock[-\s]?in|holding\s*period",
    "riskometer_level": r"riskometer|risk\s*level",
    "benchmark": r"benchmark",
    "fund_objective": r"objective",
    "fund_category": r"category|scheme\s*type",
    "aum": r"\baum\b|assets\s*under\s*management",
    "nav": r"\bnav\b|net\s*asset\s*value",
    "statement_download": r"statement|tax|capital\s*gains|download",
}

def canonical_fact_key(label: str) -> str | None: ...
def split_facts(doc: Document) -> list[dict]: ...           # (A) one fact = one chunk
def split_prose(nodes: list[Node], *, target, cap, overlap) -> list[dict]: ...   # (B) recursive char split
def semantic_merge(chunks: list[dict], *, threshold, cap, model=None) -> tuple[list[dict], int]: ...  # (C)
def build_chunk_id(scheme_slug, section, text) -> str: ...   # {slug}__{section}__{hash6}
def validate_chunk(chunk: dict) -> None: ...                 # enforce PRD §6.1
def chunk_all() -> dict: ...                                 # writes + returns stats
```

Algorithm
1. Group nodes into **sections** — a `heading` node starts a new section. Nodes before the first heading land in an implicit section named after the document.
2. **(A) Facts.** For every `table_row` / `accordion` node whose `label` maps to a `FACT_LABELS` key, emit exactly one chunk:
   `"expense ratio: 1.03%"` with `doc_type="fact"`, `fact_key`, `fact_value`, `token_count ≤ fact_max_tokens (120)`.
   **A node that does not fit the cap is DECLINED, not truncated**, and falls through to the prose splitter. Truncating it is the exact failure ADR-001 exists to prevent: the label survives, the value is cut mid-sentence, and the chunk reads like an answer but is not one. (Measured: the 5 `how_to_invest` FAQ answers run ~130 tokens and were all being truncated before this rule existed.)
3. **(B) Prose.** Concatenate the remaining `paragraph`/`list_item` nodes per section, preserving paragraph breaks, then recursive-character-split with separators in order `["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]` until each piece fits `prose_target_tokens=1000`, hard-capping at `prose_max_tokens=1100`. **Overlap is snapped to a sentence boundary** — never cut mid-sentence — with `prose_overlap_tokens=135`. Nodes already emitted as fact chunks are **excluded** from prose: a fact is either its own chunk or part of the narrative, never both, or the same sentence is embedded twice and retrieval scores start depending on how many places a fact is repeated.
4. **(C) Semantic merge.** Group consecutive prose chunks smaller than `prose_min_tokens=40`. Embed each one's lead sentence; merge neighbours with cosine ≥ `0.62`, stopping at `max_merge_tokens=1100`; re-split anything that overflows. If a sentence-transformer model is unavailable, **skip this pass** and record `semantic_merges_applied: 0` — it is an optimisation, not a correctness requirement.
5. `validate_chunk()` on every chunk: all 22 metadata fields from PRD §6.1 present, `chunk_id` unique, `token_count` matches a recount. Raise on violation — a malformed chunk that reaches Chroma becomes an uncitable answer.
6. Write `data/chunks.jsonl` + `data/chunk_stats.json` including the **ADR-001 evidence** block (architecture.md §9).

**Config keys consumed:** `chunking.*` (all of it), `guardrails.pii.*`, `paths.*`

**Acceptance criteria**
- [ ] `data/ingestion_log.jsonl` has one row per URL; ≥ 5 rows with `ok=true`
- [ ] `data/corpus.jsonl` has ≥ 5 documents; `node_count > 50` for each scheme page
- [ ] Table rows and accordions are present in the node stream (inspect a few)
- [ ] `data/chunks.jsonl` has ≥ 150 chunks; **0** chunks over `prose_max_tokens`
- [ ] Every chunk has a valid `chunk_id`, `source_url`, `fact_key`/`section_heading`, and `content_hash`
- [ ] `fact_label_value_separated_pct == 0.0` in `chunk_stats.json` ← **the ADR-001 acceptance test**
- [ ] Running the phase twice produces **byte-identical** `chunks.jsonl` (idempotency)
- [ ] `pytest tests/test_ingest.py tests/test_chunking.py -q` passes

**Verify**
```powershell
.\.venv\Scripts\python.exe -m ingest.fetch  --all
.\.venv\Scripts\python.exe -m ingest.clean
.\.venv\Scripts\python.exe -m ingest.chunk
.\.venv\Scripts\python.exe -m pytest tests/test_ingest.py tests/test_chunking.py -q
# idempotency proof
Copy-Item data\chunks.jsonl data\chunks.run1.jsonl
.\.venv\Scripts\python.exe -m ingest.chunk
(Get-FileHash data\chunks.jsonl).Hash -eq (Get-FileHash data\chunks.run1.jsonl).Hash
```

**Common failures**
| Symptom | Cause | Fix |
|---|---|---|
| `node_count` is 0 or tiny | the page is a JS shell (PRD R1) | swap the URL to the HDFC AMC page in `sources.yaml`; do not lower the threshold and pretend |
| chunks are all `prose`, zero `fact` | `FACT_LABELS` didn't match the page's wording | add the page's real phrasing to the regex; do not lower the strategy to "prose only" — that is the decision ADR-001 rejected |
| `fact_label_value_separated_pct > 0` | a table row's label and value landed in different sections | a heading split the table; group table nodes **before** sectioning, or re-join rows whose label has no value |
| second run produced different IDs | a timestamp or random value leaked into `text` | timestamps belong in metadata only, never in `text` |
| imports fail on `parse_qsl` | `httpx` version drift | pin `httpx>=0.27,<0.29` |

**Cursor prompt**
```
Read PRD.md §4, §6, §6.1, §7 (FR-1, FR-2) and docs/architecture.md §4 (Stages 1a, 1b, 2). Use common.py for config, paths, logging, JSONL, tokenisation and PII — do not reimplement them.

Implement PHASE 2 only:
  1. ingest/__init__.py
  2. ingest/fetch.py  — FetchResult dataclass, fetch_url(), fetch_all(), raw HTML cache
     (data/raw/{doc_id}.html, 24h freshness, 3 retries, exponential backoff, 2s politeness
     delay), data/source_snapshot.json (sha256+size+timestamp per URL), and
     data/ingestion_log.jsonl. Read every URL from config/sources.yaml via common.load_schemes()
     and common.load_supplementary(). NEVER raise on a single page failure — log and continue.
     If extracted visible text < 2000 chars, set looks_thin=True and log a loud warning.
  3. ingest/clean.py  — strip_boilerplate() using config chunking.strip_selectors, then a
     second sweep for orphaned text; extract_nodes() walking the DOM IN DOCUMENT ORDER and
     emitting Node(type=heading|paragraph|list_item|table_row|accordion, text, level,
     label, value, order); clean_document() scrubbing PII with common.scrub_pii and
     quarantining anything that still matches; write data/corpus.jsonl one Document per line.
  4. ingest/chunk.py  — implement docs/architecture.md §4 Stage 2 EXACTLY, in this order:
     group nodes into sections (a heading starts one); (A) split_facts() emits ONE chunk per
     table_row/accordion whose label matches the FACT_LABELS regex map, with
     doc_type="fact", fact_key, fact_value, cap 120 tokens; (B) split_prose() does a
     heading-aware recursive character split, separators ["\n\n","\n",". ","? ","! ","; ",
     ", "," "], target 1000 tokens, hard cap 1100, overlap 135 tokens SNAPPED TO A SENTENCE
     BOUNDARY; (C) semantic_merge() merges adjacent prose chunks under 40 tokens when their
     lead-sentence embeddings have cosine >= 0.62, and SKIPS ITSELF GRACEFULLY if
     sentence-transformers is unavailable. build_chunk_id() =
     {scheme_slug}__{section_slug}__{short_hash(normalised_text,6)} — no timestamps, no
     randomness. validate_chunk() enforces every field of the PRD §6.1 metadata schema and
     RAISES on violation. Write data/chunks.jsonl and data/chunk_stats.json including the
     adr001_evidence block with fact_label_value_separated_pct computed for real.
  5. tests/test_ingest.py and tests/test_chunking.py covering: boilerplate removal, node-stream
     order, table/accordion extraction, PII quarantine, token budgets, sentence-boundary
     overlap, fact label/value never separated, metadata completeness, and byte-identical
     idempotency across two runs.

Every module needs an `if __name__ == "__main__"` runner that prints a human-readable report.
No LangChain. No new top-level dependencies. Then run the fetch, clean, chunk and pytest
commands and paste the real output.
```

---

# PHASE 3 — Embedding

**Goal:** turn `data/chunks.jsonl` into 384-dim vectors, cheaply and idempotently.

**File: `embed/index.py`**

```python
@dataclass
class EmbedStats:
    total_chunks: int; computed: int; cached: int; failed: int
    dimension: int; elapsed_seconds: float; model: str; cache_hit_pct: float

def build_embed_text(chunk: dict, cfg: dict) -> str:
    """prefix + truncate(chunk text) — see the lead-window note below."""

def load_model(model_name: str) -> SentenceTransformer: ...   # device="cpu"
def load_cache(dirpath: Path) -> dict[str, np.ndarray]: ...
def save_vector(dirpath: Path, content_hash: str, vec: np.ndarray) -> None: ...
def embed_chunks(chunks: list[dict], *, force: bool = False,
                 model=None) -> tuple[dict[str, np.ndarray], EmbedStats]: ...
def embed_all(*, force: bool = False) -> EmbedStats: ...
```

**The lead-window rule (do not "fix" this)**

MiniLM is trained on sentences with a practical window of ~256 tokens; it degrades
badly on 1000-token inputs. So we do **not** embed the raw chunk:

```python
prefix = cfg["embed_prefix_template"].format(
    scheme_name=chunk["scheme_name"], category=chunk["category"],
    section_heading=chunk["section_heading"],
)                                    # "HDFC Large Cap Fund (Large Cap) - Fees and charges: "
text  = truncate_to_tokens(prefix + chunk["text"], cfg["embed_lead_tokens"])   # 200 tokens
vec   = model.encode(text, normalize_embeddings=True)
```

Two consequences you must accept and document:
1. The stored vector represents the **head** of a prose chunk, not all of it. That is the deliberate trade-off in PRD §6 / ADR-001 — it is why BM25 runs alongside dense search in Stage 5, so the tail of a long chunk is still findable lexically.
2. The prefix is what stops `"expense ratio: 1.03%"` from being equally similar to all five schemes' expense-ratio rows.

**Algorithm**
1. `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")`. Log `model.max_seq_length` so the 256-token claim is visible in the run output.
2. Load `data/embeddings/*.npy` into a `content_hash -> vector` cache.
3. For each chunk: if `content_hash` is cached and `not force` → reuse (**cached**). Else compute (**computed**).
4. `batch_size=64`, `normalize_embeddings=True`, `convert_to_numpy=True`, `show_progress_bar=True`.
5. Save each new vector to `data/embeddings/{content_hash}.npy` as float32.
6. Write `data/embed_stats.json`; print a report. **Do not** write vectors into `chunks.jsonl` — Stage 4 joins the two by `content_hash`.

**Acceptance criteria**
- [ ] Every chunk in `chunks.jsonl` has a matching `.npy`
- [ ] Every vector is shape `(384,)` and `‖v‖₂ ≈ 1.0`
- [ ] Sanity check: cosine(`"expense ratio of HDFC Large Cap Fund"`, its own fact chunk) is the **highest** among the five schemes' expense-ratio chunks
- [ ] Sanity check: cosine(`"lock-in period"`, the ELSS lock-in chunk) is the highest among all lock-in chunks
- [ ] Second run reports `cached == total`, `computed == 0`
- [ ] Cold run of ≤ 600 chunks completes in under 10 minutes on CPU
- [ ] `pytest tests/test_embedding.py -q` passes

**Verify**
```powershell
.\.venv\Scripts\python.exe -m embed.index
.\.venv\Scripts\python.exe -m embed.index          # second run: expect 100% cached
.\.venv\Scripts\python.exe -m pytest tests/test_embedding.py -q
```

**Common failures**
| Symptom | Cause | Fix |
|---|---|---|
| `OSError: We couldn't connect to 'https://huggingface.co'` | first run needs network | run once online to populate the HF cache; afterwards set `HF_HUB_OFFLINE=1` |
| cosine similarity all ≈ 0.5 | forgot `normalize_embeddings=True` | the model default is un-normalised; cosine maths in Stage 5 assumes unit vectors |
| dimension ≠ 384 | wrong model | `all-mpnet-base-v2` is 768 — the brief mandates MiniLM |
| `computed` is not 0 on the second run | vectors not persisted, or `content_hash` computed after mutating text | cache on the **same** text that is embedded; never re-normalise between hashing and embedding |
| torch grabs the GPU and OOMs | device auto-selection | pass `device="cpu"` explicitly |

**Cursor prompt**
```
Read PRD.md §6, §7 (FR-3) and docs/architecture.md §4 (Stage 3). Use common.py for config,
paths, logging, JSONL and truncate_to_tokens().

Implement PHASE 3 ONLY: embed/__init__.py and embed/index.py.

  1. load_model() -> SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").
     Log model.max_seq_length in the runner so the ~256-token window is visible.
  2. build_embed_text(chunk, cfg) must build
       prefix = cfg["embedding"]["embed_prefix_template"].format(scheme_name, category, section_heading)
       return truncate_to_tokens(prefix + chunk["text"], cfg["embedding"]["embed_lead_tokens"])
     Do NOT embed the raw chunk and do NOT remove the prefix — both are deliberate
     (docs/architecture.md §4 Stage 3, "the lead-window trick").
  3. load_cache()/save_vector() persist float32 vectors at data/embeddings/{content_hash}.npy.
  4. embed_chunks() reuses a cached vector when content_hash is present and force=False;
     otherwise encodes with batch_size=64 and normalize_embeddings=True, counting
     computed vs cached vs failed. Use a tqdm progress bar.
  5. embed_all() reads data/chunks.jsonl, writes data/embed_stats.json, and prints a report.
     DO NOT write vectors into chunks.jsonl — Stage 4 joins on content_hash.
  6. tests/test_embedding.py: every chunk has a .npy; each vector is (384,) with
     norm ~= 1.0; the second run computes 0 new vectors; and two semantic sanity checks —
     a query naming a scheme must rank that scheme's expense-ratio chunk highest, and
     "lock-in period" must rank the ELSS lock-in chunk highest. Mark the model-loading
     tests with skipif so they skip cleanly if the model cannot be downloaded.

Then run: python -m embed.index, run it a SECOND time (expect computed=0), run pytest, and
paste the real output including the cache-hit percentage.
```

---

# PHASE 4 — Vector store (ChromaDB)

**Goal:** persist the vectors with full metadata so Stage 5 can filter.

**File: `store/chroma_store.py`**

```python
def get_client() -> chromadb.PersistentClient: ...          # path from config paths.chroma_dir
def get_collection(*, create: bool = True): ...            # hdfc_mf_faq, hnsw:space=cosine, hdim=384
def upsert_chunks(chunks, vectors, *, batch_size=256) -> int: ...
def rebuild() -> int: ...                                   # wipe + recreate + upsert
def query(embedding, *, top_k=20, where=None) -> list[dict]: ...
def stats() -> dict: ...
```

**Acceptance:** collection count == chunk count; `where={"scheme_slug": "..."}` filters correctly; a second `upsert_chunks` does not duplicate; `--rebuild` starts from zero.

**Verify:** `python -m store.chroma_store --rebuild` then `python -m store.chroma_store --stats`

**Cursor prompt**
```
Read docs/architecture.md §4 (Stage 4) and PRD §7 FR-4. Implement store/__init__.py and
store/chroma_store.py. chromadb 1.5.x is installed. Use chromadb.PersistentClient(path=...),
collection name and hnsw:space=cosine / hnsw:metadata={"hdim":384} from config. Store each
chunk as one document keyed by chunk_id (upsert, never duplicate), with the FULL PRD §6.1
metadata block attached so Stage 5 can filter by scheme_slug/category/doc_type/publisher.
chromadb metadata values must be str/int/float/bool only — coerce None to "" or 0 and add a
_coerce_metadata() helper. Provide upsert_chunks(), rebuild(), query(embedding, top_k, where)
and stats(). Verify with a temp collection: insert 3, query with a where filter, upsert the
same 3 again and assert count is still 3, then --rebuild. Paste the real output.
```

---

# PHASE 5 — Retrieval

**Goal:** rank the corpus for a question using dense + lexical + fusion + diversity.

**File: `rag/retrieve.py`**

```python
@dataclass
class RetrievedChunk:
    chunk_id: str; text: str; metadata: dict
    dense_score: float; bm25_score: float; rrf_score: float
    final_score: float; matched_terms: list[str]

class OutOfCorpus(Exception): ...

def normalise_query(q: str) -> str: ...
def expand_query(q: str, synonym_map: dict) -> str: ...
def dense_search(embedding, top_k, where=None) -> list[tuple[str, float]]: ...
class BM25Index:
    def build(self, chunks) -> None; def search(self, query, top_k) -> list[tuple[str, float, list[str]]]
def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]: ...
def mmr_diversify(candidates, lam: float, top_k: int) -> list[RetrievedChunk]: ...
def rerank(query, candidates, top_n_out) -> list[RetrievedChunk]: ...   # optional cross-encoder
def retrieve(question, *, top_k=None, where=None, with_debug=False) -> list[RetrievedChunk]:
    """Raises OutOfCorpus when no chunk clears retrieval.min_cosine."""
```

**Acceptance:** `"minimum SIP"` retrieves the min-SIP chunk; `"exit load"` retrieves exit-load rows across schemes but MMR keeps them diverse; a nonsense question raises `OutOfCorpus`; every result carries populated `dense_score`, `bm25_score`, `rrf_score`, `final_score`, `matched_terms`.

**Cursor prompt**
```
Read PRD §7 FR-5 and docs/architecture.md §4 (Stage 5). Implement rag/__init__.py and
rag/retrieve.py. rank_bm25 is installed. Implement the exact pipeline from architecture.md:
normalise -> synonym expand (config guardrails.synonym_expansions) -> dense search top 20 via
Chroma -> BM25 top 20 over the SAME chunk set -> RRF fuse with k=60 -> MMR diversify with
lambda=0.35 -> optional cross-encoder rerank (config retrieval.rerank.enabled, DEFAULT OFF,
skip silently if the model is absent) -> threshold. Raise OutOfCorpus when no chunk reaches
retrieval.min_cosine. BM25Index should be built once at module level and cached, and must
tokenise on lowercase alphanumeric words. Populate matched_terms with the query terms that
actually appear in the chunk. Then demo: run queries "minimum SIP", "exit load", "lock-in
period", "how do I download my capital gains statement", and "price of gold in Mumbai" and
paste a table of chunk_id, section_heading, dense, bm25, rrf, final for each.
```

---

# PHASE 6 — Guardrails + grounded generation

**Goal:** make the assistant honest — refusals, citations, and a length cap, enforced in code.

**Files: `rag/guardrails.py`, `rag/prompts.py`, `rag/answer.py`**

```python
# guardrails.py
class PIIViolation(Exception): ...
class AdviceViolation(Exception): ...
class PerformanceViolation(Exception): ...

def check_query_pii(q: str) -> list[PIIMatch]: ...          # refuse; log only a hash
def is_advice(q: str) -> bool: ...
def is_performance(q: str) -> bool: ...
def is_out_of_scope(q: str) -> bool: ...                    # e.g. "open my account"
def validate_answer(answer: dict, corpus_urls: set[str]) -> tuple[bool, list[str]]: ...
PII_REFUSAL / ADVICE_REFUSAL / PERFORMANCE_REFUSAL / OUT_OF_CORPUS : str   # fixed text, PRD §12

# answer.py
@dataclass
class Answer: text; sources: list[Source]; kind; refusal_reason; last_updated; disclaimer; debug
def answer_question(question: str, *, provider=None) -> Answer: ...
def _generate_stub(context, question) -> str: ...           # extractive fallback
def _generate_ollama(...)/_generate_openai(...) -> str: ...
```

**Order of operations inside `answer_question`**
```
1. common.load_dotenv_if_present()
2. check_query_pii          -> Answer(kind="refusal_pii")        # log sha256(query)[:12], NOT the text
3. is_advice                -> Answer(kind="refusal_advice",      src=SEBI/AMFI edu link)
4. is_performance           -> Answer(kind="refusal_performance", src=factsheet link)
5. is_out_of_scope          -> Answer(kind="out_of_corpus")
6. retrieve(...)            -> OutOfCorpus -> Answer(kind="out_of_corpus")
7. build prompt from <=5 chunks + the 7 system rules
8. generate at temperature 0
9. validate_answer(): needs a corpus URL, <=3 sentences, no PII echo, no advice/performance
   verbs -> retry once -> on second failure use _generate_stub()  (NEVER return a guess)
10. attach disclaimer + "Last updated from sources: max(fetched_at) of used chunks"
```

**Acceptance:** all four refusals produce the exact PRD §12 text; a good question yields ≤3 sentences + one real link; a bad generation is caught by the validator; no refusal logs the query text.

**Cursor prompt**
```
Read PRD §7 (FR-6, FR-7), §11 and §12 verbatim, and docs/architecture.md §4 (Stage 6).
Implement rag/prompts.py, rag/guardrails.py and rag/answer.py.

rag/prompts.py: put the 7 system rules from architecture.md Stage 6 into a SYSTEM_PROMPT
constant, plus build_context_block(chunks) and build_user_prompt(question). The 7 rules must
appear VERBATIM as listed in the architecture so a unit test can assert each one.

rag/guardrails.py: check_query_pii (raise PIIViolation, and when logging, log ONLY
sha256(query)[:12] plus the masked reason — never the query body), is_advice, is_performance,
is_out_of_scope, driven by the keyword lists in config guardrails. Export the four fixed
refusal message constants with the EXACT wording from PRD §12. Implement validate_answer()
returning (ok, reasons): fails if no corpus URL is cited, if the sentence count exceeds
config generation.max_sentences, if the output echoes PII, or if it contains advice or
performance vocabulary.

rag/answer.py: Answer/Source dataclasses exactly as PRD §11. answer_question() executes the
10-step order above. Providers: stub (extractive — take the best chunk, emit <=3 sentences
plus its link), ollama, openai, selected by LLM_PROVIDER env var falling back to config.
temperature 0. On OutOfCorpus return the out-of-corpus message. On a failed validation,
retry once, then fall back to _generate_stub(). Always attach the disclaimer and the
"Last updated from sources: YYYY-MM-DD" line derived from max(source_fetched_at) of the
chunks actually used.

tests/test_guardrails.py must cover: each advice keyword triggers, each performance keyword
triggers, all 4 PII formats trigger, valid financial questions trigger NOTHING,
validate_answer rejects a fabricated answer with no link and accepts a good one, and the PII
log path never contains the raw query. Then paste the real pytest output.
```

---

# PHASE 7 — Streamlit UI

**Goal:** the demo surface. Tiny, honest, citation-first.

**File: `app.py`** — header + literal note "Facts-only. No investment advice."; welcome line; exactly 3 example chips; chat history via `st.session_state`; per-answer text + one clickable citation + `Last updated from sources:` + disclaimer; a "Why this answer?" expander showing chunk IDs/scores/excerpts; sidebar with scope, the 5 source links, corpus stats and a rebuild button. **No `st.file_uploader` anywhere.**

**Cursor prompt**
```
Read PRD §7 FR-8, §11 and §12, and docs/architecture.md §4 (Stage 7). Implement app.py with
streamlit. Requirements: the literal string "Facts-only. No investment advice." must appear
in BOTH the header and the footer, taken from config ui.note, not hardcoded twice. Show
config ui.welcome_line and EXACTLY the three config ui.example_questions as clickable chips
that populate the input. Every answer renders: text, exactly ONE clickable citation URL from
Answer.sources, the "Last updated from sources:" line, and the disclaimer from
docs/DISCLAIMER.md. Add a "Why this answer?" expander showing Answer.debug (chunk ids,
scores, excerpts). Sidebar: AMC scope, the 5 scheme names with their URLs from
config/sources.yaml, corpus stats (documents, chunks, embeddings, last ingest), and a
Rebuild index button. Use st.session_state for chat history only — never persist to disk.
DO NOT add st.file_uploader or any file input. Keep the whole UI under ~200 lines.
```

---

# PHASE 8 — Evaluation

**Files: `eval/golden_set.csv`, `eval/run_eval.py`, `eval/report.md`**

`golden_set.csv` columns: `qid, category, question, expected_source_url, expect_kind`
Composition exactly as PRD §15.1 — 30 questions: 5 each of expense ratio / exit load / min SIP / ELSS lock-in / riskometer+benchmark, 4 statement, 1 plan nuance, 8 advice, 5 performance, 4 PII, 4 out-of-corpus.

`run_eval.py` reports: citation-present %, citation-correct %, groundedness, refusal accuracy, PII block rate, source-hit recall@5, p50/p95 latency, and a per-question table. It is the release gate for PRD §15.2.

**Cursor prompt**
```
Read PRD §15 in full. Create eval/golden_set.csv with exactly 30 rows in the composition
specified in §15.1, columns qid,category,question,expected_source_url,expect_kind. The 8
advice rows use expect_kind=refusal_advice, the 5 performance rows refusal_performance, the
4 PII rows refusal_pii, the 4 out-of-corpus rows out_of_corpus, the rest answer. The 4 PII
rows must contain synthetic PAN/Aadhaar/phone/email values — synthetic, never real. Then
implement eval/run_eval.py to run every row through rag.answer.answer_question() and write
eval/report.md containing citation-present %, citation-correct %, groundedness (judged by
substring match of the expected fact value in the answer), refusal accuracy, PII block rate,
source-hit recall@5, p50/p95 latency, and a per-question table. Exit non-zero if any PRD §15.2
threshold is missed. Run it and paste the report.
```

---

# PHASE 9 — Delivery

| Deliverable | File | Contents |
|---|---|---|
| Source list | `docs/SOURCES.md` | the 5 URLs + supplementary, with publisher, fetch date, SHA-256 |
| README | `README.md` | setup, scope, architecture summary, run commands, known limits (PRD §18) |
| Sample Q&A | `docs/SAMPLE_QA.md` | 8–10 real queries with the assistant's actual answers + links |
| Disclaimer | `docs/DISCLAIMER.md` | byte-identical to PRD §11 |
| Architecture | `docs/architecture.md` | ✅ done |
| Implementation | `docs/implementation.md` | ✅ this file |
| Evaluation | `eval/report.md` | Phase 8 output |
| Demo | `docs/demo_script.md` + video | ≤3 min, or the running local app |

**README must state explicitly:** HDFC AMC only · 5 named schemes · Direct Growth only · figures are "as on the last ingest date" · no advice, no returns, no comparison · MiniLM-L6-v2 for embeddings · ChromaDB for the vector store · the chunking strategy and its rationale.

---

## Cross-Phase Rules

1. **Every stage is runnable alone.** `python -m ingest.chunk` must not need a Chroma collection to exist. Stages 1–2 must work with torch and chromadb uninstalled.
2. **No timestamps or randomness inside `text`.** They belong in metadata. This is what makes `chunks.jsonl` byte-identical across runs.
3. **Validate at the boundary.** `validate_chunk()` (Stage 2) and `validate_answer()` (Stage 6) raise instead of passing bad data downstream.
4. **Degrade, never crash.** Missing optional dependency → skip that enhancement and say so in the report. Missing required thing → fail loudly with an install command.
5. **Never log PII.** Masked value + reason, or `sha256(text)[:12]`. Logs are committed evidence.
6. **Reports over silent success.** Each stage prints numbers. `data/chunk_stats.json` and `data/embed_stats.json` are how a reviewer confirms a stage worked without reading code.
7. **Config, not code.** New source, threshold, chunk size, or keyword → `config/*.yaml`. Never hardcode a tunable.
8. **Cite PRD and architecture in docstrings.** A reviewer should be able to trace any function to the requirement it implements.

---

*End of IMPLEMENTATION GUIDE. Phases 0–3 complete; Phases 4–9 specified above and ready to execute.*
