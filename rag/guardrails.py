"""
rag/guardrails.py - STAGE 6: THE HONESTY LAYER  (PRD FR-7, §11, §12)

Everything in this module exists to make one guarantee: the assistant never
states something the sources do not support, and never goes looking for
personal data while doing it.

Four refusals, all verbatim from PRD §12:
    PII          §12.4   the user offered personal identifiers
    advice       §12.2   the user asked whether to buy or sell
    performance  §12.3   the user asked for returns or a comparison
    out-of-corpus §12.5  the corpus has nothing on it

and a post-generation validator, `validate_answer`, which is the only thing
standing between a confident LLM sentence and the user.

The logging rule
----------------
`check_query_pii` logs `sha256(query)[:12]` and the *masked* reason. Never the
query body. A user who pastes their PAN into a chat box must not find it in a
log file, and a test asserts the log path never contains the raw text. This is
the one place in the project where "log it for debugging" and "do not store PII"
actively conflict, and the second one wins.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Sequence

import common
from common import LOG

# ---------------------------------------------------------------------------
# Refusal text - PRD §12, verbatim. Do not reword; a test pins the wording.
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "**Facts-only. No investment advice.** This assistant shares publicly "
    "available factual information about 5 HDFC mutual fund schemes (Direct "
    "Growth plans) from the sources linked in each answer. It does not "
    "recommend, compare, or rate schemes, and it does not compute or report "
    "returns. Mutual fund investments are subject to market risks; read all "
    "scheme-related documents carefully. Sources: HDFC AMC, Groww, SEBI, AMFI."
)

EDUCATION_LINK = "https://www.sebi.gov.in/ (investor education)"

PII_REFUSAL = (
    "Please don't share personal identifiers like PAN, Aadhaar, account numbers, "
    "OTPs, or contact details here \u2014 I can't accept, store, or use them. Scheme "
    "facts are all public; ask me anything about expense ratio, exit load, SIP, "
    "lock-in, or statements."
)

ADVICE_REFUSAL = (
    "I can share facts from the official pages I use, but I can't tell you "
    "whether to buy or sell a scheme \u2014 that's investment advice, and it's "
    "outside what this assistant does. Here's an official guide to understanding "
    f"scheme risk instead: {EDUCATION_LINK}"
)

PERFORMANCE_REFUSAL = (
    "I don't compute or compare returns. For a scheme's official performance "
    "figures, please use the published factsheet: "
    "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
)

OUT_OF_CORPUS = (
    "I couldn't find that in the official pages I use (5 HDFC schemes, Direct "
    "Growth). Try: expense ratio \u00b7 exit load \u00b7 minimum SIP \u00b7 ELSS lock-in "
    "\u00b7 benchmark \u00b7 NAV and AUM \u00b7 investment objective."
)

OUT_OF_SCOPE = (
    "That one's outside what this assistant does \u2014 it only answers factual "
    "questions about 5 HDFC mutual fund schemes (Direct Growth) from the "
    "official pages it has read. Try: expense ratio \u00b7 exit load \u00b7 minimum "
    "SIP \u00b7 ELSS lock-in \u00b7 benchmark \u00b7 NAV and AUM \u00b7 investment objective."
)

# PRD §11 pins "exactly one source link" per answer. A question that names two
# schemes ("What is the NAV of HDFC Small Cap and HDFC ELSS?") cannot be
# answered honestly within that contract - citing either fund's page leaves the
# other fund's figure unsourced, and a comparison is what the disclaimer rules
# out. Refuse deterministically instead of letting the generator coin-flip.
MULTI_SCHEME_REFUSAL = (
    "That question names more than one scheme ({funds}). Every answer here "
    "cites exactly one source link, so I can only handle one fund per question "
    "\u2014 ask me about one scheme at a time."
)

# The one place a filesystem detail could reach a viewer. A store hiccup on the
# hosting instance must read as a helpful, actionable message - never as a raw
# `/opt/.../data/chroma` path in the chat.
INDEX_ERROR = (
    "I couldn't reach the search index just now. If this keeps happening, "
    "press **Rebuild index** in the sidebar, then ask again."
)

REFUSALS = {
    "pii": PII_REFUSAL,
    "advice": ADVICE_REFUSAL,
    "performance": PERFORMANCE_REFUSAL,
    "out_of_corpus": OUT_OF_CORPUS,
    "out_of_scope": OUT_OF_SCOPE,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PIIViolation(Exception):
    """The query contains something that looks like a personal identifier.

    Carries the MASKED description and the query hash, never the query.
    """

    def __init__(self, reason: str, query_hash: str) -> None:
        super().__init__(f"{reason} (query {query_hash})")
        self.reason = reason
        self.query_hash = query_hash


class AdviceViolation(Exception):
    """The user asked whether to buy, sell, or allocate."""


class PerformanceViolation(Exception):
    """The user asked for returns, rankings, or a comparison."""


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIIMatch:
    kind: str
    masked: str


def query_hash(q: str) -> str:
    """A stable, non-reversible id for a query. Safe to log."""
    return hashlib.sha256(q.encode("utf-8")).hexdigest()[:12]


# Each pattern is (kind, regex, mask_template). The regexes are deliberately
# tight: a PII scanner that fires on every 6-digit number refuses every NAV
# price, and a false-positive refusal is worse than a missed one here because it
# looks like the assistant is broken.
#
# The three that need care:
#   * PAN      - [A-Z]{5}[0-9]{4}[A-Z] with no word boundaries, because PANs are
#                written glued together.
#   * aadhaar  - 12 digits, but ONLY with the word "aadhaar" nearby. Bare 12
#                digits is a common way of writing an amount.
#   * otp/cvv/pin - keyword-anchored, since 4-6 bare digits is otherwise just a
#                number in a sentence about SIP amounts.
_PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "PAN-like {mask}"),
    ("aadhaar", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"), "Aadhaar-like number"),
    ("aadhaar_word", re.compile(r"\baadhaar\b|\baar\b", re.I), "the word 'Aadhaar'"),
    ("pan_word", re.compile(r"\bpan\b(?!\s*card\b(?!\s*of))", re.I), "the word 'PAN'"),
    # "is" / "was" between the label and the digits. Without it the most
    # natural phrasing of all - "my account number is 1234567890123456" -
    # slipped through, because the regex demanded digits the instant the
    # label ended. Golden row X03 is exactly that sentence.
    # "acc" as well as "acct": the corpus list in common.py spells it
    # `acc(?:ount)?`, which matches a bare "acc". Without it here the two lists
    # disagreed on "acc number: 1234567890" and the query check missed it.
    ("account", re.compile(
        r"\b(?:account|acc|acct|a\s?/?\s?c)\s*(?:no|number|#)?\s*(?:is|was|are)?\s*[:\-]?\s*[0-9]{9,}\b", re.I),
     "an account number"),
    # A bare 12+ digit run is a demat/folio number in this domain. Measured
    # over the 121 real chunks there are 0 such runs - every figure on the
    # page is comma-grouped or short - so this cannot fire on a real fact.
    ("long_digits", re.compile(r"(?<![\d.,])\d{12,}(?![\d.,])"),
     "a 12+ digit account-style number"),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "an IFSC code"),
    ("micr", re.compile(r"\b\d{9}\b"), "a 9-digit number"),
    ("otp", re.compile(r"\botp\b\s*(?:is|was|:)?\s*\d{4,6}\b", re.I), "an OTP"),
    ("cvv", re.compile(r"\bcvv\b\s*(?:is|was|:)?\s*\d{3,4}\b", re.I), "a CVV"),
    ("pin", re.compile(r"\bpin\b\s*(?:is|was|:)?\s*\d{4}\b", re.I), "a PIN"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "an email address"),
    ("phone", re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b"), "a phone number"),
    ("dob", re.compile(r"\b(?:dob|date of birth)\b\s*[:\-]?\s*\d", re.I), "a date of birth"),
]


def _mask(text: str) -> str:
    """Reduce a match to its shape, never its content.

    `ABCDE1234F` -> `A*******` and `9876543210` -> `9*********`. Enough to debug
    which rule fired; not enough to reconstruct anyone's PAN.
    """
    if not text:
        return ""
    if "@" in text:
        return "***@***"
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    if digits and not letters:
        return text[0] + "*" * (len(text) - 1)
    if letters and not digits:
        return text[0].upper() + "*" * (len(text) - 1)
    return "".join(ch if ch.isdigit() else "*" for ch in text[:4]) + "****"


def check_query_pii(q: str) -> list[PIIMatch]:
    """Raise PIIViolation if the query carries a personal identifier.

    Returns an empty list when clean. On a hit, logs the query HASH and the
    MASKED reason, then raises - so the caller never has to remember not to log
    the text.
    """
    cfg = common.load_config()["guardrails"]
    if not cfg.get("pii", {}).get("enabled", True):
        return []

    matches: list[PIIMatch] = []
    for kind, pattern, template in _PII_PATTERNS:
        for m in pattern.finditer(q):
            masked = _mask(m.group(0))
            matches.append(PIIMatch(kind=kind, masked=masked))
            LOG.warning("  PII guardrail: %s | query_hash=%s | matched=%s",
                        template.format(mask=masked), query_hash(q), masked)
            # One hit is enough to refuse; do not enumerate the user's data.
            raise PIIViolation(template.format(mask=masked), query_hash(q))
    return matches


# ---------------------------------------------------------------------------
# Advice / performance / scope
# ---------------------------------------------------------------------------


def _hit(q: str, keywords: Sequence[str]) -> str | None:
    """The first keyword present in q, matched on word boundaries.

    Substring matching is not safe here: "return" would fire on "returns",
    "returned" and - via "load" - nothing, but "profit" would fire on
    "profitable", and "best fund" would fire inside "best funds to buy". Word
    boundaries plus a phrase check keep it honest.
    """
    ql = q.lower()
    for kw in keywords:
        k = str(kw).lower().strip()
        if not k:
            continue
        if " " in k:
            if k in ql:
                return k
        elif re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", ql):
            return k
    return None


def is_advice(q: str) -> bool:
    """Does the user want to be told what to do?"""
    if _pattern_hit(q, "advice_patterns") is not None:
        return True
    return _hit(q, common.load_config()["guardrails"].get("advice_keywords", [])) is not None


def _pattern_hit(q: str, key: str) -> str | None:
    """First configured regex matching `q`, or None.

    A keyword list cannot express "two named things, joined by or, asking which is
    better" - the words "which" and "better" are each present in the list as
    "which is better", and a user who writes "Which fund is better, HDFC Large Cap
    or HDFC Small Cap?" puts five words between them. A pattern catches the shape;
    a keyword list can only catch an exact phrase.

    Patterns are still config, not code, so tuning them does not mean editing this
    module, and a pattern that fires on a golden question is caught by
    `test_benign_questions_match_no_advice_pattern`.
    """
    ql = q.lower()
    for raw in (common.load_config()["guardrails"].get(key, []) or []):
        try:
            if re.search(str(raw), ql, flags=re.I):
                return str(raw)
        except re.error:
            LOG.warning("  guardrails: %r is not a valid regex, skipping", raw)
    return None


def is_performance(q: str) -> bool:
    """Does the user want returns, rankings, or a comparison?

    Benchmark index NAMES are exempted, on the query side as well as the answer
    side. Four of the five benchmarks are literally named "... Total Return
    Index", so without this the assistant refused to answer "which benchmark
    does HDFC Large Cap use?" - a golden-set question - as a performance
    question, because the index's own name contains the word "return".
    """
    return matched_performance_keyword(q) is not None


def matched_advice_keyword(q: str) -> str | None:
    pat = _pattern_hit(q, "advice_patterns")
    if pat is not None:
        return pat
    return _hit(q, common.load_config()["guardrails"].get("advice_keywords", []))


# A percentage sitting next to return vocabulary is a performance claim, whatever
# words wrap it. This exists because the CORPUS itself contains one: the
# "how to invest in ..." FAQ chunk says "the average annual returns provided by
# this fund is 18.28% since its inception". That figure is on the page, so the
# grounding check passes it happily - grounding only asks "did the source say
# this?", and the source did. PRD §12.3 forbids reporting it regardless, so the
# pair of features is what has to be refused, not the number.
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")

# Return vocabulary, for the pairing test below. "return" is deliberately not
# here on its own - the index-name exemption in `_strip_allowed` has already
# removed the legitimate cases, so what reaches this test is a real claim.
_RETURN_VOCAB = re.compile(
    r"(?<![a-z0-9])(return|returns|cagr|xirr|yield|yielded|profit|gain)(?![a-z0-9])",
    re.I)


def performance_violations(text: str) -> list[str]:
    """Performance claims in generated text, by phrase and by feature pairing."""
    cfg = common.load_config()
    gcfg = cfg.get("generation", {})
    allow = gcfg.get("benchmark_name_allow", []) or []
    scannable = _strip_allowed(text, allow)

    found: list[str] = []
    claim = _hit(scannable, gcfg.get("performance_claim_phrases", []) or [])
    if claim:
        found.append(f"the answer makes a performance claim: {claim!r}")

    if _RETURN_VOCAB.search(scannable) and _PERCENT.search(scannable):
        found.append(
            "the answer pairs return vocabulary with a percentage, which is a "
            "performance claim even when the figure is quoted from the page")
    return found


def matched_performance_keyword(q: str) -> str | None:
    """The performance signal in a question: pattern first, then keyword.

    Patterns are checked first because the keyword list cannot express them
    without false positives. "growth" is in neither, deliberately: it occurs
    in "Direct Growth", in all five scheme names, in every one of the 121
    chunks, so a "growth" keyword would refuse golden questions F01-F30.
    """
    cfg = common.load_config()
    allow = (cfg.get("generation", {}) or {}).get("benchmark_name_allow", []) or []
    pat = _pattern_hit(q, "performance_patterns")
    if pat is not None:
        return pat
    return _hit(_strip_allowed(q, allow),
                cfg["guardrails"].get("performance_keywords", []))


def absent_concepts_in(q: str) -> list[str]:
    """Real mutual-fund concepts named in `q` that this corpus never contains.

    These are not refusals - the assistant simply has never read them. They need
    a post-retrieval check rather than a pre-retrieval one, because whether the
    corpus can answer depends on what was actually retrieved.

    Motivation: the Stage 5 gate is term-coverage based, so "What is the
    riskometer level of HDFC Large Cap?" cleared it on the words "level" and
    "large cap" and was answered with the expense ratio. A cited, confident,
    wrong answer is worse than a refusal, and that is the only shape of wrong
    this system can produce.
    """
    ql = q.lower()
    hits = []
    for term in (common.load_config()["guardrails"].get("known_absent_terms", []) or []):
        if re.search(rf"\b{re.escape(str(term).lower())}\b", ql):
            hits.append(str(term))
    return hits


def corpus_has_concept(term: str, texts: Sequence[str]) -> bool:
    """Does any retrieved chunk actually mention this term?"""
    t = term.lower()
    return any(t in str(x).lower() for x in texts)


# Other AMCs. The corpus is HDFC-only, so a question about SBI's Large Cap must
# be refused rather than answered with HDFC's Large Cap - the two have
# byte-identical fact shapes, which is exactly why the mix-up is easy.
#
# The list cannot be complete; there are hundreds of Indian AMCs. It is here to
# catch the large ones a user is most likely to type. `test_another_amc_is_out_of_scope`
# pins twelve of them, so adding a scheme's worth of support without adding its
# AMC is what makes that test fail.
_OTHER_AMCS = (
    "sbi", "icici", "axis", "kotak", "aditya birla", "birla", "dhirubhai",
    "mirae", "sundaram", "tata", "lic", "canara", "pnb", "indusind", "hsbc",
    "franklin", "templeton", "parag parikh", "nippon", "sbi mf", "icici pru",
    "quant", "ppf", "bandhan", "canara robo", "union amc", "jm financial",
    "motilal", "motilal oswal", "mahindra", "principal", "pgf", "shriram",
    "nippon india", "sundaram mf", "l&T", "lt", "groww", "zeta", "oak hill",
    "whitespark", "navi", "vijay kadia", "piramal", "sriram", "icici Lombard",
    "bajaj", "bajaj allianz", "bajaj finserv", "hdfc life",
)

# Actions that are not questions about a fund at all.
_OUT_OF_SCOPE_ACTIONS = (
    "open my account", "open an account", "create an account", "close my account",
    "withdraw money", "transfer money", "change my bank", "update my address",
    "update my phone", "update my email", "reset my password", "log in",
    "login", "sign up", "signup", "download app", "install app", "uninstall",
    "call me", "phone me", "email me", "contact support", "raise a ticket",
    "reset otp", "update pan", "link bank account", "start sip for me",
    "invest on my behalf", "place an order", "buy for me", "sell for me",
)


def is_out_of_scope(q: str) -> bool:
    """Is this about something the assistant was never built to do?

    Covers two cases: an account/transaction action it cannot perform, and a
    question about a different AMC. The second matters more than it looks: "SBI
    Large Cap expense ratio" retrieves HDFC's expense-ratio row at a BM25 of
    8.9, so without this the assistant would answer a question about SBI with
    HDFC's number and attribute it to a page about HDFC.
    """
    ql = q.lower()
    for phrase in _OUT_OF_SCOPE_ACTIONS:
        if phrase in ql:
            return True
    for amc in _OTHER_AMCS:
        if re.search(rf"(?<![a-z0-9]){re.escape(amc)}(?![a-z0-9])", ql):
            # ...unless the question is explicitly about HDFC, in which case the
            # AMC name is incidental ("is SBI better than HDFC?") and the advice
            # guardrail, not this one, should handle it.
            #
            # The exception is for OTHER companies that merely share the HDFC
            # name. HDFC Life is an insurer, not HDFC Asset Management, and
            # "what is the claim ratio in HDFC Life" must not be answered from a
            # mutual fund page just because the string "HDFC" is present.
            if amc.startswith("hdfc"):
                return True
            if not re.search(r"(?<![a-z0-9])hdfc(?![a-z0-9])", ql):
                return True
    return False


def is_out_of_corpus_reason(reason: str) -> bool:
    return bool(reason)


# ---------------------------------------------------------------------------
# Sentence counting
# ---------------------------------------------------------------------------

# Abbreviations that end in a period without ending a sentence. Without these,
# "The NAV is Rs 1,189.08. Min SIP is Rs 100." counts as three sentences and a
# perfectly good answer gets rejected for exceeding the three-sentence cap.
#
# Dotted abbreviations need BOTH the abbreviation and its trailing dot consumed,
# because "p.a." and "e.g." carry two. `count_sentences` therefore appends a
# sentinel after matching the abbreviation plus its dot, so a dotted form cannot
# leave a bare "." behind to be read as a sentence end. Added after measuring
# "It costs 1.03% p.a. for the direct plan." being counted as two sentences.
_ABBREV = (r"(?:mr|mrs|ms|dr|vs|etc|no|fig|approx|incl|excl|min|max|rs|inr|cr"
           r"|e\.g|i\.e|p\.a|p\.p|w\.e|f\.y|s\.d|asst|mgr|co|ltd|inc|avg)")


def count_sentences(text: str) -> int:
    """Sentences in `text`, tolerant of abbreviations and decimals.

    Deliberately conservative: it over-counts rather than under-counts, because
    the failure mode of under-counting is a long answer shipping, and the failure
    mode of over-counting is a short correct answer being retried and replaced by
    the extractive fallback - annoying, but safe.
    """
    if not text or not text.strip():
        return 0
    body = re.sub(rf"(?<![a-z0-9])({_ABBREV})\.?", r"\1<DOT>", text, flags=re.I)
    # A period between digits is a decimal, not a sentence end.
    body = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", body)
    # Strip URLs - a dot in a domain is not a sentence boundary.
    body = re.sub(r"https?://\S+", " <URL> ", body)
    parts = [p for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
    return max(len(parts), 1)


# ---------------------------------------------------------------------------
# Post-generation validator
# ---------------------------------------------------------------------------


def corpus_urls() -> set[str]:
    """Every URL the corpus was actually built from.

    The validator's job is to catch a citation that was invented. It can only do
    that by comparing against a real set, and the only trustworthy version of that
    set is the one in the built corpus - not a hardcoded list that drifts.
    """
    urls: set[str] = set()
    for row in common.read_jsonl(common.path_for("chunks_file")):
        u = row.get("source_url")
        if u:
            urls.add(str(u))
    try:
        import yaml  # noqa: PLC0415

        src = common.ROOT / "config" / "sources.yaml"
        if src.exists():
            data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
            for page in data.get("pages", []) or []:
                if page.get("url"):
                    urls.add(str(page["url"]))
    except Exception:  # noqa: BLE001 - a validator must not crash on its own inputs
        pass
    return urls


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Every number in `text`, comma-stripped.

    "₹1,13,606.47 Cr" -> {"113606.47"}. Commas are removed rather than kept
    because the corpus writes Indian grouping (1,13,606.47) while a language
    model may write either, and a grounding check that fails on formatting
    difference would reject correct answers.
    """
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text)}


def grounding_violations(answer_text: str, context: Sequence[Any]) -> list[str]:
    """Numbers the answer states that appear in NO retrieved chunk.

    This is the strongest check in the module and it is the one that actually
    catches a performance claim. "It returned 24% last year" contains no banned
    phrase and reads perfectly; it fails here, because 24 is not a figure on any
    of the five pages. Vocabulary lists rot as soon as a model finds new ways to
    say something, but "this number is not in my sources" does not.

    Every number is checked, including one- and two-digit ones. An earlier
    version skipped anything under three digits, to avoid firing on incidental
    integers, and in doing so waved through "returned 24%" - exactly the case
    PRD §12.3 exists to refuse. The filter was removed only after measuring
    whether it was needed:

        for each of the 121 real chunks, treat that chunk as the answer and the
        chunk itself as context  ->  0 of 121 produce a violation

    so on grounded content the strict version does not false-reject, while
    "returned 24%" / "returned 9%" / "gave 12% returns" are all three rejected.

    `context` accepts RetrievedChunk objects, plain strings, or dicts with a
    "text" key, because all three show up at the call site.
    """
    if not context:
        return []

    supplied: set[str] = set()
    for item in context:
        if isinstance(item, str):
            body = item
        elif isinstance(item, dict):
            body = str(item.get("text", ""))
        else:
            body = str(getattr(item, "text", ""))
        supplied |= numbers_in(body)

    claimed = numbers_in(answer_text)
    unsourced = sorted(claimed - supplied)
    return [f"states the figure {n!r}, which is not in any retrieved source"
            for n in unsourced]


_APPENDED = re.compile(
    r"(?:\s*Source:\s*https?://\S+)?"
    r"(?:\s*Last updated from sources:\s*\S+)?\s*$", re.I)


def prose_of(text: str) -> str:
    """The answer with the pipeline's own furniture removed.

    `ask()` appends two things to the model's prose: the citation and the
    "Last updated from sources: 2026-09-27" stamp. Both are metadata this
    pipeline wrote, not claims the model made, so the CONTENT checks - advice,
    performance, and above all grounding - must not see them.

    This was not hypothetical. Validating the stamped text sent every answer down
    the "drop the stamp" path, because a date is by construction absent from the
    chunk it annotates: "Last updated from sources: 2026-09-27" states 2026, 09
    and 27, none of which appear in "expense ratio: 1.03%". The grounding check
    was rejecting the pipeline's own timestamp as an unsourced figure, and every
    answer shipped with an empty `last_updated`.

    The sentence cap and the citation check still run on the FULL text - those
    two are about what ships, not about what the model claimed.
    """
    return _APPENDED.sub("", str(text or "")).strip()


def _strip_allowed(text: str, allow: Sequence[str]) -> str:
    """Remove allow-listed phrases so a word check cannot fire inside a name.

    "The benchmark is the NIFTY 500 Total Return Index" must survive the
    performance check, and the only reason it would not is the substring
    "total return index".
    """
    out = text
    for phrase in allow:
        out = re.sub(re.escape(str(phrase)), " ", out, flags=re.I)
    return out


def validate_answer(answer: dict[str, Any], corpus_urls_set: set[str] | None = None,
                    context: Sequence[Any] | None = None) -> tuple[bool, list[str]]:
    """Is this generated answer safe to show? Returns (ok, reasons).

    Six independent checks, in the order they matter:

      1. every figure it states appears in a retrieved chunk  - the grounding
         check, and the only one that reliably catches a fabricated number;
      2. it cites a URL that is actually in the corpus  - a citation to a page
         nobody read is a fabricated citation, which is worse than no citation;
      3. it is within the sentence cap;
      4. it contains no advice ASSERTION (`generation.advice_vocabulary`, not
         the question-phrased `guardrails.advice_keywords`);
      5. it contains no performance CLAIM, with benchmark index names exempted;
      6. it does not echo PII back at the user.

    Returning the REASONS rather than a bare bool is deliberate: a validator that
    cannot say what it caught is a validator nobody trusts into being strict.
    """
    cfg = common.load_config()
    gcfg = cfg["generation"]
    reasons: list[str] = []

    text = str(answer.get("text") or "")
    if not text.strip():
        return False, ["the generated text is empty"]

    known = corpus_urls_set if corpus_urls_set is not None else corpus_urls()
    cited = [str(s.get("url", "")) for s in (answer.get("sources") or [])]
    in_text = re.findall(r"https?://[^\s)\]]+", text)
    all_urls = cited + in_text

    if not all_urls:
        reasons.append("no citation: the answer names no source URL")
    else:
        bad = [u for u in all_urls if u.rstrip(".,)") not in known]
        if bad:
            reasons.append(f"cites a URL that is not in the corpus: {bad}")

    # PRD §11 is exactly ONE source. A generator that names two different URLs
    # (e.g. "…X is ₹1,447.38 … Source: A … Source: B") inherits the multi-source
    # failure no matter how well grounded each figure is. Deduplicate first: the
    # pipeline itself appends the citation to the prose, so the same URL appears
    # twice (once in text, once in sources) on every legitimately-sourced answer.
    distinct_urls = {u.rstrip(".,)") for u in all_urls if u.rstrip(".,)")}
    if len(distinct_urls) > 1:
        reasons.append("cites more than one distinct source URL; PRD §11 is exactly one source")

    max_sentences = int(gcfg.get("max_sentences", 3))
    n = count_sentences(text)
    if n > max_sentences:
        reasons.append(f"{n} sentences, over the cap of {max_sentences}")

    # --- advice: ASSERTION vocabulary, not question vocabulary ---------------
    prose = prose_of(text)
    advice = _hit(prose, gcfg.get("advice_vocabulary", []) or [])
    if advice:
        reasons.append(f"the answer contains advice vocabulary: {advice!r}")

    # --- performance: claims, with index names exempted ---------------------
    reasons.extend(performance_violations(prose))

    # --- grounding: every stated figure must be a sourced figure ------------
    if context is not None:
        reasons.extend(grounding_violations(prose, context))

    if re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text):
        reasons.append("the answer echoes something that looks like PII")
    for kind, pattern, _tpl in _PII_PATTERNS:
        if kind in ("aadhaar_word", "pan_word"):
            continue
        if pattern.search(text):
            reasons.append(f"the answer contains a {kind}-like string")
            break

    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

# Real questions, used by `python -m rag.guardrails` to show that a valid
# financial question triggers NOTHING. A guardrail suite that only tests its
# triggers proves nothing about its false-positive rate, and a false positive
# here looks to a user like the assistant is broken.
BENIGN_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the minimum SIP amount?",
    "Is there a lock-in period on HDFC ELSS Tax Saver Fund?",
    "What is the exit load?",
    "What benchmark does HDFC Small Cap Fund Direct Growth use?",
    "What is the NAV of HDFC Balanced Advantage Fund?",
    "What is the AUM of HDFC Flexi Cap Fund?",
    "What is the expense ratio (TER)?",
    "How much can I invest via SIP?",
    "What is the exit load on HDFC ELSS?",
    "What is the fund size?",
    "What is the investment objective?",
    "What are the stamp duty charges?",
    "What is the exit load nil date?",
]


def self_test() -> int:
    print("=" * 78)
    print("  STAGE 6 - GUARDRAILS SELF-TEST")
    print("=" * 78)

    print("\n  PII - every format must trigger:")
    pii_samples = [
        ("PAN", "my PAN is ABCDE1234F"),
        ("Aadhaar", "aadhaar number 2345 6789 0123"),
        ("account", "account number 123456789012"),
        ("IFSC", "IFSC code HDFC0001234"),
        ("OTP", "the otp is 482913"),
        ("CVV", "cvv: 731"),
        ("PIN", "pin is 1234"),
        ("email", "mail me at ravi.sharma@example.com"),
        ("phone", "call me on 9876543210"),
        ("DOB", "dob: 12/04/1990"),
    ]
    for label, sample in pii_samples:
        try:
            check_query_pii(sample)
            print(f"    MISS  {label:<9} {sample!r}")
        except PIIViolation as exc:
            print(f"    ok    {label:<9} {exc.reason}")

    print("\n  ADVICE - every keyword must trigger:")
    hits = 0
    for kw in common.load_config()["guardrails"]["advice_keywords"]:
        q = f"should I {kw}?" if " " not in kw else kw
        if is_advice(q):
            hits += 1
        else:
            print(f"    MISS  {kw!r}")
    print(f"    {hits}/{len(common.load_config()['guardrails']['advice_keywords'])} triggered")

    print("\n  PERFORMANCE - every keyword must trigger:")
    hits = 0
    kws = common.load_config()["guardrails"]["performance_keywords"]
    for kw in kws:
        if is_performance(kw):
            hits += 1
        else:
            print(f"    MISS  {kw!r}")
    print(f"    {hits}/{len(kws)} triggered")

    print("\n  FALSE POSITIVES - valid questions must trigger NOTHING:")
    bad = 0
    for q in BENIGN_QUESTIONS:
        problems = []
        if is_advice(q):
            problems.append("advice")
        if is_performance(q):
            problems.append("performance")
        if is_out_of_scope(q):
            problems.append("out_of_scope")
        try:
            check_query_pii(q)
        except PIIViolation as exc:
            problems.append(f"pii ({exc.reason})")
        if problems:
            bad += 1
            print(f"    FALSE POSITIVE  {q!r} -> {', '.join(problems)}")
    print(f"    {len(BENIGN_QUESTIONS) - bad}/{len(BENIGN_QUESTIONS)} clean")

    print("\n  OTHER AMC - must be out of scope, not answered with HDFC's number:")
    for q in ("What is the expense ratio of SBI Large Cap Fund?",
              "ICICI Prudential Large Cap exit load"):
        print(f"    {'ok   ' if is_out_of_scope(q) else 'MISS '} {q!r}")

    print("\n  SENTENCE COUNTING:")
    for t, want in [
        ("One sentence.", 1),
        ("One. Two.", 2),
        ("The NAV is Rs 1,189.08. Min SIP is Rs 100.", 2),
        ("See https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth", 1),
        ("A, B, and C. D.", 2),
    ]:
        got = count_sentences(t)
        print(f"    {'ok  ' if got == want else 'MISS'} {got} (want {want})  {t[:58]!r}")

    print("\n  VALIDATOR:")
    urls = corpus_urls()
    big = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    # Real context, so the grounding check has something to check against.
    ctx = [c for c in common.read_jsonl(common.path_for("chunks_file"))
           if c["scheme_slug"] == "hdfc-large-cap-fund-direct-growth"]

    cases: list[tuple[str, dict, bool]] = [
        ("a good answer passes", {
            "text": "The expense ratio (TER) for HDFC Large Cap Fund - Direct Growth "
                    "is 1.03%. Source: " + big,
            "sources": [{"url": big}]}, True),
        ("benchmark index name", {
            "text": "The benchmark is the NIFTY 100 Total Return Index. Source: " + big,
            "sources": [{"url": big}]}, True),
        ("no citation", {"text": "The expense ratio is 1.03%.", "sources": []}, False),
        ("fabricated citation", {
            "text": "See https://example.com/mutual-funds for details.",
            "sources": [{"url": "https://example.com/mutual-funds"}]}, False),
        ("too long", {
            "text": "One. Two. Three. Four. Five.", "sources": [{"url": big}]}, False),
        ("advice assertion", {
            "text": "You should buy this scheme. Source: " + big,
            "sources": [{"url": big}]}, False),
        ("advice: recommend", {
            "text": "We recommend this scheme for you. Source: " + big,
            "sources": [{"url": big}]}, False),
        ("performance claim", {
            "text": "It returned 24% last year. Source: " + big,
            "sources": [{"url": big}]}, False),
        ("performance claim (cagr)", {
            "text": "The scheme has a CAGR of 18.4% over five years. Source: " + big,
            "sources": [{"url": big}]}, False),
        ("unsourced figure", {
            "text": "The NAV is 4,999.99. Source: " + big,
            "sources": [{"url": big}]}, False),
    ]
    failures = 0
    for label, payload, should_pass in cases:
        ok, why = validate_answer(payload, urls, context=ctx)
        good = ok is should_pass
        if not good:
            failures += 1
        mark = "ok  " if good else "MISS"
        detail = "" if should_pass else f" -> {why}"
        print(f"    {mark} {label:<24}{detail}")

    print("\n  GROUNDING - the check that catches a fabricated figure:")
    for text, want_unsourced in [
        ("The NAV is 1,189.08.", False),
        ("The AUM is 39,933.37 Cr.", False),
        ("It returned 24% last year.", True),
        ("The expense ratio is 2.99%.", True),
    ]:
        got = grounding_violations(text, ctx)
        mark = "ok  " if bool(got) is want_unsourced else "MISS"
        if bool(got) is not want_unsourced:
            failures += 1
        print(f"    {mark} unsourced={bool(got)!s:<5} {text!r}")

    print("\n" + "=" * 78)
    return 1 if (bad or failures) else 0


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stage 6 - guardrail self-test")
    ap.add_argument("--test", action="store_true", default=True)
    ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()
    return self_test()


if __name__ == "__main__":
    sys.exit(_main())
