"""
common.py - Shared foundation for every RAG stage.

This module is deliberately dependency-light so that Stages 1 and 2 can be
tested without pulling in torch / chromadb. It owns:

  * config + source-list loading (cached)
  * filesystem layout guarantees
  * JSONL read / write / append helpers
  * ISO-8601 UTC timestamps
  * content hashing + stable slugs (idempotency, PRD FR-2.4)
  * token counting (tiktoken with a graceful offline fallback)
  * the PII scanner / scrubber (PRD FR-1.5, FR-7.1)
  * logging setup

Run as a script it prints a resolved runtime report - useful as a Phase 1
smoke test:

    python common.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent


def load_config() -> dict[str, Any]:
    """Load config/app.yaml (cached)."""
    return _load_yaml(ROOT / "config" / "app.yaml")


def load_sources() -> dict[str, Any]:
    """Load config/sources.yaml (cached). This is the corpus gate - PRD S4."""
    return _load_yaml(ROOT / "config" / "sources.yaml")


@lru_cache(maxsize=8)
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required config missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def path_for(key: str) -> Path:
    """Resolve a `paths.*` key from app.yaml to an absolute path."""
    cfg = load_config()
    try:
        rel = cfg["paths"][key]
    except KeyError as exc:  # pragma: no cover - config error
        raise KeyError(f"Unknown paths key: {key!r}") from exc
    return ROOT / rel


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to. Idempotent."""
    for rel in load_config()["paths"].values():
        if isinstance(rel, str) and ("/" in rel or "\\" in rel):
            (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "embeddings").mkdir(parents=True, exist_ok=True)


def load_dotenv_if_present() -> None:
    """Minimal .env loader - avoids a python-dotenv dependency."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables always win over the file.
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Full-precision ISO-8601 UTC, e.g. 2026-09-27T10:12:03.412Z"""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc_now().microsecond // 1000:03d}Z"


def utc_date() -> str:
    """Date only, used for the 'Last updated from sources: YYYY-MM-DD' stamp."""
    return utc_now().strftime("%Y-%m-%d")


#: The stub appends a bare `Source: <url>` to `Answer.text` so the CLI has
#: something to print. Anything that renders the citation separately - the UI,
#: the generated sample Q&A - must strip it, or the one source appears twice
#: (PRD FR-8.4: "the one source link"). It lives here rather than in app.py so
#: there is one definition instead of one per renderer.
#:
#: The `+` handles a generator that inlines the URL AND the pipeline appends
#: the citation: both trailing copies are stripped, so the rendered answer
#: shows the source exactly once - in the citation block.
_INLINE_SOURCE = re.compile(r"(?:\s*Source:\s*https?://\S+\.?)*\s*$", re.I)


def strip_inline_source(text: str) -> str:
    """`text` with a trailing inline `Source: <url>` removed."""
    return _INLINE_SOURCE.sub("", str(text or "")).strip()


def parse_iso(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Hashing + slugs
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """Whitespace-normalised hash - this is the embedding cache key (FR-3.3)."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 6) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# tiktoken needs to fetch its BPE ranks on first use. That breaks offline runs
# and air-gapped demos, so we fall back to a deterministic approximation.
# The fallback is only ever used for *budgeting* (chunk size), never for
# billing, so a ~15% drift is harmless.
_ENCODER: Any = None
_ENCODER_TRIED = False


def _get_encoder() -> Any:
    global _ENCODER, _ENCODER_TRIED
    if not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # offline, or tiktoken unavailable
            _ENCODER = None
    return _ENCODER


def _fallback_token_count(text: str) -> int:
    # Punctuation counts as its own token; words average ~1.3 subword tokens.
    return max(1, int(len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)) * 1.3))


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc is None:
        return _fallback_token_count(text)
    return len(enc.encode(text, disallowed_special=()))


def tokenize(text: str) -> list[str]:
    """Token list - used by the chunker to snap overlaps to token boundaries."""
    enc = _get_encoder()
    if enc is None:
        return re.findall(r"\S+", text)
    return enc.decode(enc.encode(text, disallowed_special=())).split()


def tokenizer_backend() -> str:
    return "tiktoken:cl100k_base" if _get_encoder() is not None else "fallback:regex"


# ---------------------------------------------------------------------------
# Source-text shapes that are not values
# ---------------------------------------------------------------------------

#: A body that is only an "as on" date - "As on 31 Aug 2026", "As of 30/09/2026".
#:
#: The AMC uses these as section furniture: a heading and nothing else. Promoting
#: one to a fact produced a stored value that carries no information, under
#: whatever fact_key the heading happened to key to. "Portfolio Allocation & Top
#: Holdings" became the fact "portfolio allocation & top holdings: as on 31 aug
#: 2026", and a link label ("Click here to view performance of other schemes
#: managed by the Fund Manager(s)") became `fund_managers: As on 31 Aug 2026`.
#: Both were then retrievable and quotable as though they were facts.
#:
#: Defined here rather than in either ingest stage because both need it: Stage 1
#: declines to emit the node at all, and Stage 2 refuses to make a chunk from a
#: value that has no content.
_AS_ON_ONLY_RE = re.compile(
    r"^\s*(?:as\s+(?:on|of)|as\s+at)\s+\d{1,2}\s*[a-z/-]{3,12}\.?,?\s*\d{2,4}\s*$",
    re.I,
)


def is_as_on_only(value: str) -> bool:
    """True if `value` is nothing but an as-on date, so carries no fact."""
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip(".")
    return bool(text) and bool(_AS_ON_ONLY_RE.match(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _get_encoder()
    if enc is None:
        # Word-proportional cut.
        if _fallback_token_count(text) <= max_tokens:
            return text
        words, out, running = text.split(), [], 0.0
        for w in words:
            running += _fallback_token_count(w) + 1
            if running > max_tokens:
                break
            out.append(w)
        return " ".join(out)
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# PII  (PRD FR-1.5 / FR-7.1 - no PAN, Aadhaar, account no., OTP, email, phone)
# ---------------------------------------------------------------------------


@dataclass
class PIIMatch:
    kind: str
    confidence: str          # high | low
    masked: str              # never the raw value
    span: tuple[int, int] = (0, 0)


# Order matters: more specific patterns run first, and a match CONSUMES its
# span, so a later pattern can never re-report a substring of an earlier match
# (e.g. "possible_otp" firing inside an already-detected phone number).
_PII_PATTERNS: list[tuple[str, str, str]] = [
    # (kind, regex, confidence)
    ("email",      r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "high"),
    ("pan",        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "high"),
    ("ifsc",       r"\b[A-Z]{4}0[A-Z0-9]{6}\b", "high"),
    # Indian mobile / landline: optional +91, optional leading 0, separators
    # allowed anywhere, national number must start 6-9 and be 10 digits.
    ("phone",      r"(?<![\w.])0?(?:\+?91[\s.\-]?)?[\s.\-]?\(?[6-9]\)?[\s.\-]?\d{4}[\s.\-]?\d{5}(?!\d)", "high"),
    # Lookbehind is digit-only (not \s) so a leading space is allowed, while
    # still refusing to match the tail of a longer digit run.
    ("aadhaar",    r"(?<!\d)[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}(?!\d)", "high"),
    # "is" / "was" / ":" between the label and the digits. Without this the
    # single most natural phrasing - "my account number is 1234567890123456" -
    # did not match, because the regex demanded digits the moment the label
    # ended. Golden row X03 is exactly that sentence.
    ("account_no", r"(?<![\w])(?:a/?c|acc(?:ount)?|acct)[\s.:#\-]*(?:no\.?|number)?[\s.:#\-]*(?:is|was|are|=|:)?[\s.:#\-]*\d{9,18}(?!\d)", "high"),
    ("demat",      r"(?i)(?<![\w])(?:dp|dem)\s*-?\s*(?:id)?\s*[.:#-]?\s*\d{8,16}(?!\d)", "high"),
    # Backstop for an account number stated with no label at all. A bare run of
    # 12+ digits is a demat/folio number in this domain: measured over the 121
    # real chunks there are 0 such runs, because every figure on the page is
    # either comma-grouped (1,13,606.47) or short (100, 1.03%). The lookarounds
    # exclude . and , so a grouped figure cannot reach it.
    ("long_digits", r"(?<![\d.,])\d{12,}(?![\d.,])", "high"),
    # A short secret stated next to its own name. The keyword is what makes this
    # unambiguous: bare 4-6 digit runs collide with expense ratios, AUM and
    # lock-in years, but nothing on a fund page reads "your OTP is 482913".
    # Placed above possible_otp so the match consumes the digits.
    ("otp",        r"(?i)(?<![\w])(?:otp|one[\s\-]?time\s+pass(?:word|code)|cvv|cvc|security\s+code|passcode|pass\s*code|pin)\s*(?:is|was|[:\-])?\s*[*x•\d]{4,8}(?![\w])", "high"),
    # 4-6 bare digits collide with expense ratios / AUM / lock-in years, so this
    # is LOW confidence only and never auto-quarantines a chunk.
    ("possible_otp", r"(?<![\w.%])\d{4,6}(?![\w.%])", "low"),
]


def mask_value(value: str) -> str:
    """Redact a PII value, keeping just enough for a human to recognise context."""
    if len(value) <= 4:
        return "*" * len(value)
    keep = 2
    return value[:keep] + "*" * (len(value) - keep)


def scan_pii(text: str) -> list[PIIMatch]:
    """Find PII in `text`. Returns masked matches only - never raw values.

    Overlap-aware: once a span is claimed by a higher-priority pattern, lower
    priority patterns are skipped there.
    """
    found: list[PIIMatch] = []
    occupied: list[tuple[int, int]] = []
    for kind, pattern, confidence in _PII_PATTERNS:
        for m in re.finditer(pattern, text):
            start, end = m.span()
            if any(start < e and s < end for s, e in occupied):
                continue
            occupied.append((start, end))
            found.append(
                PIIMatch(
                    kind=kind,
                    confidence=confidence,
                    masked=mask_value(m.group(0)),
                    span=(start, end),
                )
            )
    return found


def has_high_confidence_pii(text: str) -> bool:
    return any(m.confidence == "high" for m in scan_pii(text))


def scrub_pii(text: str, placeholder: str = "[REDACTED]") -> tuple[str, list[PIIMatch]]:
    """Replace high-confidence PII with a placeholder. Returns (clean_text, matches).

    This is the defence-in-depth pass applied to *corpus* text before chunking.
    The *query* path uses the gate in rag/guardrails.py which refuses instead
    of scrubbing.
    """
    matches = [m for m in scan_pii(text) if m.confidence == "high"]
    if not matches:
        return text, []

    # Work right-to-left so earlier spans stay valid.
    for m in sorted(matches, key=lambda x: x.span[0], reverse=True):
        text = text[: m.span[0]] + placeholder + text[m.span[1] :]
    return text, matches


def pii_report(matches: list[PIIMatch]) -> str:
    """Human-readable, raw-value-free summary for logs."""
    if not matches:
        return "clean"
    return ", ".join(f"{m.kind}:{m.masked}({m.confidence})" for m in matches)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def chunk_list(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_schemes(include_disabled: bool = False) -> list[dict[str, Any]]:
    """The 5 seed scheme pages from config/sources.yaml."""
    schemes = load_sources().get("schemes", [])
    return [s for s in schemes if include_disabled or s.get("enabled", True)]


def load_supplementary(include_disabled: bool = False) -> list[dict[str, Any]]:
    """Supplementary AMC/SEBI/AMFI pages.

    Includes both `fetch: true` pages (which enter the RAG index) and
    `fetch: false` link-only pages (used as educational hyperlinks in refusals).
    """
    docs = load_sources().get("supplementary", [])
    return [d for d in docs if include_disabled or d.get("enabled", True)]


def load_fetchable_supplementary() -> list[dict[str, Any]]:
    """Only the supplementary pages that are actually downloaded and embedded."""
    return [d for d in load_supplementary() if d.get("fetch", False)]


def load_educational_links() -> dict[str, str]:
    """publisher -> url, for the refusal messages in rag/guardrails.py.

    Link-only sources live here even though they are never fetched, so an advice
    refusal can still point a user at a real SEBI/AMFI page.
    """
    links: dict[str, str] = {}
    for d in load_supplementary():
        links.setdefault(d["publisher"], d["url"])
    return links


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_READY = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _LOG_READY
    logger = logging.getLogger("mfbot")
    if not _LOG_READY:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        _LOG_READY = True
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def setup_console() -> None:
    """Force UTF-8 on stdout/stderr.

    The Windows console defaults to cp1252, which raises UnicodeEncodeError on
    the rupee sign and other characters that appear constantly in Indian mutual
    fund data (Rs 1,189.08, TER 1.03%, etc.). errors="replace" means a stage
    report can never crash on a glyph.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - non-reconfigurable stream
            pass


setup_console()


LOG = setup_logging()


# ---------------------------------------------------------------------------
# Phase 1 smoke test
# ---------------------------------------------------------------------------


def _main() -> int:
    load_dotenv_if_present()
    setup_console()
    ensure_dirs()

    cfg = load_config()
    src = load_sources()
    schemes = load_schemes()
    supp = load_supplementary()

    print("=" * 68)
    print(f"  {cfg['project']['name']}  -  Phase 1 runtime check")
    print("=" * 68)
    print(f"  root              : {ROOT}")
    print(f"  python            : {sys.version.split()[0]}")
    print(f"  tokenizer         : {tokenizer_backend()}")
    print(f"  schemes           : {len(schemes)}")
    for s in schemes:
        print(f"      - [{s['category']:<20}] {s['short_name']:<24} {s['url']}")
    print(f"  supplementary     : {len(supp)}")
    for d in supp:
        flag = "" if d.get("enabled", True) else "  (disabled)"
        print(f"      - [{d['publisher']:<8}] {d['title']}{flag}")
    print(f"  embedding model   : {cfg['embedding']['model_name']} ({cfg['embedding']['dimension']}-d)")
    print(f"  vector store      : {cfg['vector_store']['backend']} -> {cfg['paths']['chroma_dir']}")
    print(f"  chunking strategy : {cfg['chunking']['strategy']}")
    print(f"  LLM provider      : {os.environ.get('LLM_PROVIDER', cfg['generation']['provider'])}")

    # Sanity-check the PII gate against labelled probes.
    print("-" * 68)
    print("  PII gate self-test:")
    probes: list[tuple[str, bool]] = [
        ("My PAN is ABCDE1234F, what is the TER?", True),
        ("call me on +91 98765 43210", True),
        ("my number is 9876543210", True),
        ("aadhaar 4567 8901 2345", True),
        ("email me at ravi.sharma@example.com", True),
        ("acct no 30123456789012", True),
        ("IFSC is HDFC0001234", True),
        ("DP ID 12345678", True),
        # Must NOT be flagged as high-confidence PII. The figures below are the
        # real live values, so this doubles as a guard: if the scanner ever starts
        # flagging a real corpus figure, the corpus is unindexable.
        ("what is the expense ratio of HDFC Large Cap?", False),
        ("Expense ratio: 1.03%", False),
        ("nav: ₹1,189.08", False),
        ("fund size (aum): ₹1,13,606.47 Cr", False),
        ("Exit load is 1.00% under 12 months", False),
        ("lock-in period is 3 years", False),
        ("min. for sip: ₹100", False),
        ("min. for sip: ₹500", False),
        ("AUM is Rs 50000 crore", False),
        ("benchmark is Nifty 50 TRI", False),
    ]
    all_ok = True
    for probe, expect_flag in probes:
        matches = scan_pii(probe)
        high = [m for m in matches if m.confidence == "high"]
        ok = bool(high) == expect_flag
        all_ok &= ok
        detail = pii_report(high) or "clean"
        print(f"      [{'PASS' if ok else 'FAIL'}] {detail:<40} <- {probe[:44]}")
    print("-" * 68)
    print("  PII GATE:", "ALL PASS" if all_ok else "SOME FAILED")
    print("  TOKENIZER:", count_tokens("HDFC Large Cap Fund - Direct Growth"))
    print("=" * 68)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
