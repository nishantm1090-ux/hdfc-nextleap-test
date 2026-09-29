"""
ingest/clean.py - STAGE 1b: LOADING (content side)

Turns cached raw HTML into a *structure-preserving node stream*. This is the
payload that makes the Stage 2 chunking decision (ADR-001) possible: a flat
get_text() would destroy the label/value boundaries, and Stage 2 would be
reduced to naive recursive splitting.

What the real Groww pages actually look like (verified 2026-09-27):
  1. schema.org JSON-LD  <script type="application/ld+json"> @type FAQPage
     -> 8 clean question/answer pairs per scheme page
  2. "fund details" blocks   -> label div + value div   (NAV, Min. for SIP, AUM,
     Expense ratio, Rating)
  3. "Minimum investments"   -> label/value rows
  4. "Understand terms"      -> term/definition rows    (Expense ratio, Tax,
     Exit load, Stamp duty)
  5. "Fund benchmark"        -> label/value
  6. ELSS badge              -> "ELSS" / "3Y Lock-in"
  7. exit-load slabs         -> "Exit load of 1% if redeemed within 1 year"
  8. real <table>/<tr> rows  -> holdings, annualised/absolute returns
  9. h1..h6 headings         -> section boundaries
 10. <p> and <li>            -> prose

Rather than hardcoding those class names (they are hashed CSS-module names and
will change), the extractor uses GENERIC rules: a short leaf element followed
by another short leaf element whose text looks like a value. That survives
class renames and works on the AMFI and help pages too.

Run:  python -m ingest.clean
      python -m ingest.clean --force
      python -m ingest.clean --doc-id hdfc-large-cap-fund-direct-growth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

import common
from common import (
    LOG,
    append_jsonl,
    load_config,
    load_schemes,
    load_sources,
    pii_report,
    read_json,
    scan_pii,
    scrub_pii,
    short_hash,
    slugify,
    utc_now_iso,
    write_json,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

LEAF_MAX_CHARS = 80          # a "leaf" label/value candidate is this short
VALUE_LOOKAHEAD = 8          # how far to walk forward to find the paired value
MIN_PROSE_CHARS = 60         # paragraph/list noise floor
CONSUMED_ATTR = "data-mfconsumed"   # marks nodes already used as a label/value

_HEADING_RE = re.compile(r"^h[1-6]$")
_TRACKING_RE = re.compile(r"\b(utm_|gclid|fbclid|page=\d+)", re.I)
#: CSS/markup ellipsis. A value carrying one is an abbreviation, not the fact.
_TRUNCATED_RE = re.compile(r"\.{3}|…")
#: A trailing as-on date, e.g. the AMC's "NAV (28/09/2026)" / "AUM (31/08/2026)".
#: The date qualifies the value, it is not part of the label's identity, so it
#: must not decide whether the label reads like a value.
_AS_ON_SUFFIX_RE = re.compile(r"\s*\(\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\)\s*$")

#: Heading tags, for section-titled facts.
_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]

#: Glyph bullets the AMC types into the text itself rather than using markup.
#: They are presentational and would otherwise start a stored value with "●".
_BULLET_RE = re.compile(r"^[\s•●▪·‣⁃\-–—]+")
_WS_RE = re.compile(r"\s+")
# What counts as a fact VALUE when pairing it with a label beside it.
# The digits/rupee branch covers NAV, TER, dates, amounts and exit-load
# percentages. The second branch covers SEBI's riskometer scale, whose members
# are words: without it "Riskometer" skipped straight over "Very High" and
# paired with whatever number came next - reporting the minimum SIP as the
# riskometer level, which is exactly the kind of confident wrong answer this
# pipeline must never emit.
_VALUEISH_RE = re.compile(
    r"(₹|\bRs\.?\b|\d|%)"
    r"|\b(very\s+high|high|moderate|medium|low)\b",
    re.I,
)

# "NA" / "NIL" are real, quotable fact values on an AMC scheme page - they are
# how the page states "no entry load", "no lock-in" and, on the ELSS scheme,
# "Exit Load: NIL" - and neither carries a digit, rupee or percent, so the value
# test below would otherwise read them as labels and pair them with whatever
# came next ("NA: NAV (28/09/2026)").
_NOT_APPLICABLE_RE = re.compile(r"^(n/?a|nil|none|not\s+applicable)$", re.I)

# Labels that only ever appear inside a scheme's own fact grid (TER, minimum
# SIP, riskometer, lock-in, AUM, entry/exit load, benchmark). A container
# holding several of these is the scheme's fact card, not page chrome, so the
# boilerplate pass must never delete it - whatever the site's CSS-module
# happens to call the wrapper.
_FACT_GRID_LABELS = (
    "TER", "Min SIP", "Riskometer", "Lock in", "AUM",
    "Entry Load", "Exit Load", "Benchmark", "Inception Date",
)
_FACT_GRID_MIN_HITS = 3
_LABEL_HAS_LETTER_RE = re.compile(r"[A-Za-z]")

# A short leaf that states a fact outright, e.g. the ELSS badge "ELSS - 3Y
# Lock-in". Below the prose floor, but precisely what a user asks about.
# NOTE: deliberately does NOT include a bare `\d+Y` pattern. That matched the
# returns widget's period chips ("3Y", "5Y", "+12.47 % 3Y annualised") and
# dragged performance figures into the corpus the brief forbids us to report.
_FACT_BADGE_RE = re.compile(
    r"lock[-\s]?in|lockin|riskometer|benchmark|expense|exit\s*load|min\.?\s*for"
    r"|minimum|\bnav\b|\baum\b|stamp\s*duty",
    re.I,
)

# UI filter chips that must never be treated as a fact label. "All" was being
# paired with "NAV: 25 Sep '26" and emitted as a fact.
_UI_CHIP_RE = re.compile(
    r"^(all|custom|apply|reset|buy|redeem|invest|invest\s*more|exit|explore|view"
    r"|more|login|sign\s*up|download|shares|folio|watchlist|add|remove|1d|1w|1m"
    r"|3m|6m|1y|2y|3y|5y|10y|all time|ytd)$",
    re.I,
)

# Section headings whose content is dropped wholesale (config chunking.skip_sections).
_SKIP_SECTION_RE = re.compile(
    r"holdings?|portfolio|compare\s+similar|fund\s*manag|fund\s*house|fund\s*facts",
    re.I,
)

# Extra selectors beyond config chunking.strip_selectors. These match the
# hashed CSS-module class patterns verified on the real pages. Substring
# matching is used rather than the exact hashed names so the extractor
# survives a CSS-module hash bump.
#
# The holdings / compareSimilarFunds / fundManagement widgets are stripped here
# rather than via heading-proximity filtering, because on these pages they are
# sibling widgets - the nearest preceding heading does not reliably bound them
# (the fundDetails block's nearest heading is just the page h1).
EXTRA_STRIP_SELECTORS = [
    "svg", "img", "picture", "source", "iframe", "video", "audio", "form",
    "button", "[aria-hidden='true']", "[tabindex='-1']",
    "[class*='footer']", "[class*='Footer']",
    "[class*='navbar']", "[class*='NavBar']", "[class*='navBar']",
    "[class*='header']", "[class*='Header']",
    "[class*='cookie']", "[class*='Cookie']",
    "[class*='popup']", "[class*='Popup']", "[class*='modal']", "[class*='Modal']",
    "[class*='rodal']", "[class*='dialog']", "[class*='Dialog']",
    "[class*='banner']", "[class*='Banner']",
    "[class*='sticky']", "[class*='Sticky']", "[class*='bottomNav']",
    "[class*='sidebar']", "[class*='SideBar']",
    "[class*='relatedFund']", "[class*='relatedCard']",
    "[class*='appBanner']", "[class*='downloadApp']",
    "[class*='trust']", "[class*='Trust']",
    "[class*='rating']", "[class*='Rating']", "[class*='star']", "[class*='Star']",
    # --- the three high-noise widgets on a Groww scheme page -------------
    "[class*='holdings_']", "[class*='holdingsHeader']",
    "[class*='compareSimilarFunds']",
    "[class*='fundManagement']",
    "[class*='fundHouse']",
    "[class*='returnsAndRankings']", "[class*='returnsWrapper']",
    # The return CALCULATOR and the return STATS ticker. Both render performance
    # figures, which this assistant is forbidden to report (PRD FR-3.3), and the
    # calculator's only "fact" is its own default input of Rs 5,000. Keeping them
    # puts a return number one keyword away from a generated answer.
    "[class*='returnCalculator']",
    "[class*='returnStats']",
    # --- hover tooltips on an HDFC AMC scheme page -------------------------
    # Every fact label ("TER", "Entry Load", "Lock in", "Benchmark") wraps an
    # info icon whose tooltip holds a paragraph of generic explanation, e.g.
    #     <div class="...title">TER<div class="...riskotooltiptext">...The Total
    #     Expense Ratio (TER) represents the costs for...</div></div>
    # That made the label element longer than LEAF_MAX_CHARS, so the label was
    # not a leaf, the label/value pair was never formed, and the fact was
    # dropped. Tooltips are hover-only duplicates of prose that already exists
    # elsewhere on the page, so removing them loses nothing a reader can see.
    # Substring matching, so a CSS-module hash bump does not break this.
    "[class*='riskotooltiptext']",
    "[class*='risktooltipbench']",
    "[class*='tooltipcolor']",
]


# ---------------------------------------------------------------------------
# Node / Document
# ---------------------------------------------------------------------------

NODE_TYPES = {"heading", "paragraph", "list_item", "table_row", "accordion", "quote"}


@dataclass
class Node:
    """One typed piece of a page, in document order."""

    type: str
    text: str
    level: int | None = None       # heading level
    label: str | None = None       # table_row key / accordion question
    value: str | None = None       # table_row value / accordion answer
    order: int = 0
    origin: str = "dom"            # dom | jsonld | table

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != ""}


@dataclass
class Document:
    doc_id: str
    source_url: str
    page_title: str
    publisher: str
    page_type: str
    nodes: list[Node]
    raw_sha256: str
    source_fetched_at: str
    ingested_at: str
    pii_scan: str
    node_count: int
    text_chars: int
    scheme_slug: str | None = None
    scheme_name: str | None = None
    short_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    category: str | None = None
    plan: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["nodes"] = [n.to_dict() if isinstance(n, Node) else n for n in self.nodes]
        return {k: v for k, v in d.items() if v is not None and v != ""}


def expand_truncated_values(soup: BeautifulSoup) -> int:
    """Recover the unabbreviated text the AMC hides behind a truncated value.

    On hdfcfund.com a fact value that is too long for its box is rendered
    truncated with a CSS ellipsis, and the full string is moved into a hover
    bubble:

        <div class="...description ...tooltipbench">NIFTY 100 (Total Ret...
          <div class="...risktooltipbench" aria-labelledby="a8">
            <span>NIFTY 100 (Total Return Index)</span></div>
        </div>

    Taken as written, the value reads "NIFTY 100 (Total Ret..." - which is what
    a visitor sees, but it is a worse answer than the page can actually give,
    and an ellipsis in a cited fact looks like truncation on our side.

    Runs BEFORE boilerplate stripping, because the boilerplate pass is what
    removes the tooltip bubble. Only replaces the element when the hidden text
    is genuinely more complete, so nothing is ever made worse.
    """
    replaced = 0
    for el in soup.select("[class*='tooltipbench']"):
        hidden = el.find(class_=re.compile(r"risktooltipbench"))
        if hidden is None:
            continue
        full = _tidy(hidden.get_text(" ", strip=True))
        if not full:
            continue
        # What the visitor actually sees is the element's own text, i.e. its
        # direct strings; the hidden bubble is the rest. Only rewrite when that
        # visible part really is abbreviated.
        shown = _tidy(" ".join(str(s) for s in el.find_all(string=True, recursive=False)))
        if not _TRUNCATED_RE.search(shown):
            continue
        el.clear()
        el.append(NavigableString(full))
        replaced += 1
    return replaced


# ---------------------------------------------------------------------------
# Step 1 - boilerplate removal
# ---------------------------------------------------------------------------


def _tidy(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def rescue_tag_pills(soup: BeautifulSoup) -> list[tuple[str, str, str]]:
    """Harvest the scheme's attribute pills BEFORE boilerplate stripping.

    On a Groww scheme page the fund's own tags live in
    `div.pills_container__*` inside the page <header>:

        <header>
          <section class="...header_schemeNameContainer">
            <div class="...pills_container">
              <a href="/mutual-funds/filter?..."><span>ELSS - 3Y Lock-in</span></a>

    "ELSS - 3Y Lock-in" is the single most-asked fact on the corpus and the
    only source of the ELSS lock-in. But it sits inside a <header>, which the
    boilerplate pass decomposes wholesale - so the fact was silently lost.

    This pass runs first, takes the pill texts, then removes the container so
    the strip pass cannot double-count it.
    """
    out: list[tuple[str, str, str]] = []
    for container in soup.select("div[class*='pills_container']"):
        for pill in container.find_all(["span", "a", "div"]):
            txt = _tidy(pill.get_text(" ", strip=True))
            if not txt or len(txt) > 60:
                continue
            # Only take the innermost element that carries the text.
            if pill.find(["span", "a", "div"]) is not None:
                continue
            if not _FACT_BADGE_RE.search(txt):
                continue
            out.append((txt, "", "pill"))
        container.decompose()
    return out


def holds_fact_grid(el: Tag) -> bool:
    """True if this element carries a scheme's own fact card.

    Page chrome and the fact grid are not always distinguishable by class name.
    On hdfcfund.com the whole fact card (TER, minimum SIP, riskometer, lock-in,
    AUM, benchmark) sits inside a <div class="...bannerdiv">, so the blanket
    `[class*='banner']` strip deleted every scheme fact in the corpus at once -
    silently, because the stage reported success. The guard is content-based
    rather than class-based on purpose: a promo banner does not state the
    scheme's TER.
    """
    txt = _tidy(el.get_text(" ", strip=True))
    return sum(1 for m in _FACT_GRID_LABELS if m in txt) >= _FACT_GRID_MIN_HITS


def strip_boilerplate(soup: BeautifulSoup, selectors: list[str]) -> BeautifulSoup:
    """Remove nav/footer/ads/overlays, then sweep for orphaned leftovers.

    Deleting a <nav> can strand its text inside a parent, so a second pass looks
    for short text blocks that are now detached from any meaningful container.
    """
    combined = list(dict.fromkeys(list(selectors) + EXTRA_STRIP_SELECTORS))
    removed = 0
    for sel in combined:
        try:
            for el in soup.select(sel):
                if isinstance(el, Tag) and not holds_fact_grid(el):
                    el.decompose()
                    removed += 1
        except Exception:  # noqa: BLE001 - a bad selector must not kill the stage
            continue

    # Second sweep: leftover empty wrappers and tracking-param noise.
    for el in soup.find_all(["div", "span", "p", "section", "ul", "ol"]):
        txt = _tidy(el.get_text(" ", strip=True))
        if not txt:
            el.decompose()
        elif _TRACKING_RE.search(txt) and len(txt) < 120:
            el.decompose()

    # Drop comments - they carry no answerable content.
    for c in soup.find_all(string=lambda s: isinstance(s, NavigableString) and s.strip().startswith("<!--")):
        c.extract()

    LOG.debug("strip_boilerplate removed %d element(s)", removed)
    return soup


# ---------------------------------------------------------------------------
# Step 2 - leaf classification
# ---------------------------------------------------------------------------


def _has_substantive_descendant(el: Tag) -> bool:
    for d in el.find_all(True):
        if d is el:
            continue
        if len(_tidy(d.get_text(" ", strip=True))) > LEAF_MAX_CHARS:
            return True
    return False


def _is_leaf(el: Tag) -> bool:
    """A short element with no meaningfully-long descendant = label or value."""
    if not isinstance(el, Tag):
        return False
    if el.name in ("script", "style", "noscript", "br", "hr", "img", "svg"):
        return False
    txt = _tidy(el.get_text(" ", strip=True))
    if not txt or len(txt) > LEAF_MAX_CHARS:
        return False
    if _has_substantive_descendant(el):
        return False
    return _text_bearing_descendants(el) < 2


def _text_bearing_descendants(el: Tag) -> int:
    """How many descendants of `el` carry text of their own.

    An element that wraps two or more text-bearing children is a LAYOUT
    container, not a label or a value. The HDFC AMC fact card is built as
    `<div><p>Entry Load</p><p>NA</p></div>`, so without this the wrapper reads
    as one label "Entry Load NA", claims the value "NA" before the real label
    gets a turn, and the real label then pairs with the NEXT fact - which is
    how "Lock in" ended up quoting a NAV and "Entry Load" quoting a TER.
    """
    n = 0
    for d in el.find_all(True):
        if d is el:
            continue
        if _tidy(d.get_text(" ", strip=True)):
            n += 1
            if n >= 2:
                return n
    return n


def _is_valueish(text: str) -> bool:
    if _NOT_APPLICABLE_RE.match(text.strip()):
        return True
    return bool(_VALUEISH_RE.search(text))


def _is_labelish(text: str) -> bool:
    # An as-on date qualifies a label without being part of what the fact is
    # called. The AMC writes its NAV and AUM facts as "NAV (28/09/2026)" and
    # "AUM (31/08/2026)"; those dates made the labels read as values, so the
    # labels were skipped and the real values beside them were never paired.
    # The date is preserved in the label text, only ignored for this test.
    core = _AS_ON_SUFFIX_RE.sub("", text.strip()) or text.strip()
    if not _LABEL_HAS_LETTER_RE.search(core):
        return False
    if core[0].isdigit():
        return False
    return not _is_valueish(core)


def _mark(el: Tag) -> None:
    el[CONSUMED_ATTR] = "1"


def _mark_subtree(el: Tag) -> None:
    """Consume an element and everything inside it.

    Prevents the parent/child duplication this pipeline originally produced:
    a long <div> wrapper and the <p> inside it both looked like prose, so the
    same sentence was emitted twice. Marking the subtree means the first,
    outermost match wins.
    """
    _mark(el)
    for d in el.find_all(True):
        d[CONSUMED_ATTR] = "1"


def _contains_consumed(el: Tag) -> bool:
    return any(d is not el and CONSUMED_ATTR in d.attrs for d in el.find_all(True))


def _is_consumed(el: Tag) -> bool:
    return CONSUMED_ATTR in el.attrs


# ---------------------------------------------------------------------------
# Step 3 - generic label/value extraction
# ---------------------------------------------------------------------------


def extract_label_value_pairs(root: Tag) -> list[tuple[str, str, str]]:
    """Find (label, value, source) triples using structure, not class names.

    Walks the DOM in document order. When it meets a short leaf that reads like
    a label ("Expense ratio", "Min. for SIP", "Fund benchmark"), it looks a
    short distance forward for the paired value. This is what recovers the
    hashed-CSS-module fact blocks without hardcoding any Groww class name.
    """
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for el in root.find_all(True):
        if _is_consumed(el) or not _is_leaf(el):
            continue
        label = _tidy(el.get_text(" ", strip=True)).rstrip(":").strip()
        if not _is_labelish(label) or _UI_CHIP_RE.match(label):
            continue
        if _contains_consumed(el):
            continue

        # Look forward for the paired value.
        #
        # `find_all_next()` yields the label's OWN descendants before anything
        # that follows it, because BeautifulSoup walks in document order. The
        # AMC's date-qualified labels wrap their date: <div>NAV<span>(28/09/2026)
        # </span></div><p>₹ 1,170.11</p>. The inner date span is itself a
        # value-shaped leaf, so the scan stopped there and the fact was recorded
        # as the self-referential "NAV (28/09/2026): (28/09/2026)" while the
        # rupee amount beside it was never read. Candidates that live INSIDE
        # the label are part of the label, not its value.
        value: str | None = None
        value_el: Tag | None = None
        for i, nxt in enumerate(el.find_all_next()):
            if i >= VALUE_LOOKAHEAD or not isinstance(nxt, Tag):
                continue
            if nxt is el or el in nxt.parents:
                continue
            if not _is_leaf(nxt) or _is_consumed(nxt):
                continue
            cand = _tidy(nxt.get_text(" ", strip=True))
            if not cand or cand == label or not _is_valueish(cand):
                continue
            value, value_el = cand, nxt
            break

        if value is None or value_el is None:
            continue

        key = (label.lower(), value.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((label, value, "dom"))
        _mark(el)
        _mark(value_el)

    return pairs


def _section_body_text(block: Tag, heading: Tag) -> str:
    """The heading's own body text, in document order, on separate lines.

    Walks the block manually so that the heading's own subtree and any nested
    heading's subtree are dropped rather than concatenated: a sub-heading's
    title is a label, not body text, and letting it in corrupts the value.
    """
    lines: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        line = _tidy(" ".join(buf))
        buf.clear()
        if line:
            lines.append(line)

    def walk(el: Tag, skip: bool) -> None:
        for child in el.children:
            if isinstance(child, NavigableString):
                if not skip:
                    buf.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            if child.name == "br":
                flush()
                continue
            if child.name in _HEADING_TAGS:
                continue                    # a nested title, not body text
            if child is heading or skip:
                walk(child, True)
            else:
                walk(child, False)

    walk(block, False)
    flush()
    return " ".join(_BULLET_RE.sub("", ln).strip() for ln in lines if _BULLET_RE.sub("", ln).strip())


def extract_heading_section_pairs(root: Tag) -> list[tuple[str, str, str]]:
    """A heading plus the body of its own section -> one label/value pair.

    Some facts are not a value in a box; they are a titled section whose body is
    prose or a list. The AMC writes the exit load exactly that way, and writes
    it three different ways across five pages:

        <h2>Exit Load</h2><ul><li>...</li><li>...</li></ul>          (Small Cap)
        <h2>Exit Load</h2><p><span>● ...</span><br/><span>● ...</span></p>
                                                                        (Large Cap)
        <h2>Exit Load</h2><p><span>NIL</span></p>                    (ELSS)
        <h2>Exit Load</h2><p>..</p><p>..</p><ul><li>..</li></ul>     (Balanced Adv.)

    Handling only the first shape fixed two schemes and left the other three
    wrong: their bullets fell through to prose, prose chunking merged them with
    a performance disclaimer and a list of PDF filenames into one ~400-token
    block, and "what is the exit load?" retrieved that block and answered with
    the benchmark and a past-performance caveat. Worse, the ELSS page states
    "NIL" in a <p>, so its exit load was absent from the corpus entirely.

    So the whole body of the heading's own section is taken, whichever element
    carries it. Two guards keep this from over-reaching:

    * `find_parent` alone is too generous - it walks up until it finds *a*
      block, and on these pages the nearest block above the "Downloads" heading
      is a container that also holds the whole exit-load section. The block must
      therefore not contain another heading of the same or higher rank.
    * The value is capped, so a heading whose block is a large container is
      declined rather than turned into a fact that is really a page dump.
    """
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for h in root.find_all(_HEADING_TAGS):
        if _is_consumed(h):
            continue
        label = _tidy(h.get_text(" ", strip=True)).rstrip(":").strip()
        if not label or not _is_labelish(label):
            continue
        rank = int(h.name[1])
        block = h.find_parent(["div", "section", "article", "td", "li"])
        if block is None or block is h:
            continue
        if any(int(o.name[1]) <= rank
               for o in block.find_all(_HEADING_TAGS) if o is not h):
            continue
        value = _section_body_text(block, h)
        if not value or len(value) > 900:
            continue
        # A section whose whole body is an "as on" date states nothing. The AMC
        # renders "Portfolio Allocation & Top Holdings" as a title plus the
        # single line "As on 31 Aug 2026", which became the fact
        # "portfolio allocation & top holdings: as on 31 aug 2026" - a fact with
        # no content that then anchored a prose chunk and appeared in answers.
        if common.is_as_on_only(value):
            continue
        key = (label.lower(), value.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((label, value, "dom"))
        _mark(h)
        # The whole section body is now the value, so mark the subtree. Without
        # this the same sentences are emitted a second time as list_item/prose
        # nodes, and the corpus ends up holding each exit-load statement twice -
        # once cleanly paired with its heading, once buried in a prose blob.
        _mark_subtree(block)
    return pairs


def extract_table_rows(root: Tag, *, enabled: bool = False) -> list[tuple[str, str, str]]:
    """Real <table> rows: first cell is the label, the rest is the value.

    DISABLED BY DEFAULT (config chunking.include_table_rows: false). Verified on
    the real pages: every <table> here is either a returns table (which the brief
    forbids us to report) or a holdings table (~850 stock names). Keeping them
    adds ~1,700 nodes of retrieval noise and drags performance figures into
    context. Every fact we answer with lives in a div label/value block or in
    the JSON-LD FAQPage block instead.
    """
    if not enabled:
        # Still mark them consumed so they do not leak into prose extraction.
        for tr in root.find_all("tr"):
            _mark_subtree(tr)
        return []

    rows: list[tuple[str, str, str]] = []
    for tr in root.find_all("tr"):
        cells = [_tidy(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        label, value = cells[0], " | ".join(cells[1:])
        if len(label) > LEAF_MAX_CHARS:
            continue
        _mark_subtree(tr)
        rows.append((label, value, "table"))
    return rows


def extract_fact_badges(root: Tag, max_chars: int, exclude: set[str] | None = None) -> list[tuple[str, str, str]]:
    """Promote short fact-stating leaves (e.g. "ELSS - 3Y Lock-in") to facts.

    These are the nodes that would otherwise be lost: they are under the prose
    floor, they are not a label/value pair, but they are exactly what the user
    asks about. Emitted with the full text as the label and an empty value, so
    chunk.py matches them on vocabulary and keeps the whole phrase.

    `exclude` holds labels already captured as a label/value pair, so "Min. for
    SIP" is not re-emitted as a badge beside the pair "Min. for SIP" -> "Rs 500".
    """
    exclude = {e.lower() for e in (exclude or ())}
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for el in root.find_all(True):
        if _is_consumed(el) or not _is_leaf(el):
            continue
        # A wrapper whose children were already extracted as a label/value pair
        # would otherwise re-emit the same content as a badge
        # (e.g. "Min. for SIP Rs 500" duplicating the pair below it).
        if _contains_consumed(el):
            continue
        txt = _tidy(el.get_text(" ", strip=True))
        if not txt or len(txt) > max_chars:
            continue
        if txt.lower() in exclude:
            continue
        if _UI_CHIP_RE.match(txt) or not _FACT_BADGE_RE.search(txt):
            continue
        key = txt.lower()
        if key in seen:
            continue
        seen.add(key)
        _mark(el)
        out.append((txt, "", "badge"))
    return out


def extract_jsonld_faq(soup: BeautifulSoup) -> list[tuple[str, str, str]]:
    """schema.org FAQPage blocks -> (question, answer, 'jsonld').

    This is the cleanest source on the page: 8 structured Q&A per scheme,
    including AUM, NAV and the expense-ratio explanation, all with as-of dates.
    """
    out: list[tuple[str, str, str]] = []
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text() or ""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict) or item.get("@type") != "FAQPage":
                continue
            for qa in item.get("mainEntity", []) or []:
                if not isinstance(qa, dict):
                    continue
                q = _tidy(re.sub(r"<[^>]+>", " ", str(qa.get("name", ""))))
                ans_block = (qa.get("acceptedAnswer") or {})
                a = _tidy(re.sub(r"<[^>]+>", " ", str(ans_block.get("text", ""))))
                if q and a:
                    out.append((q, a, "jsonld"))
    return out


# ---------------------------------------------------------------------------
# Step 4 - headings and prose
# ---------------------------------------------------------------------------


def extract_headings(root: Tag) -> list[Node]:
    nodes: list[Node] = []
    for h in root.find_all(_HEADING_RE):
        if _is_consumed(h):
            continue
        txt = _tidy(h.get_text(" ", strip=True))
        if len(txt) < 2 or len(txt) > 160:
            continue
        if _has_substantive_descendant(h):
            continue
        nodes.append(Node(type="heading", text=txt, level=int(h.name[1])))
        _mark(h)
    return nodes


def extract_prose(root: Tag) -> list[Node]:
    """Paragraphs, list items and long free-standing text blocks.

    find_all returns document order, so a wrapping <div> is seen before the <p>
    inside it. Emitting the outer element and consuming its subtree means the
    outermost, most complete text wins and nothing is duplicated.

    "span" is in the tag list for a specific reason: the AMC renders its exit
    load as `<li><span>...</span></li>`. The <li> is correctly skipped as a
    wrapper whose descendant holds the real text, and with no "span" in the
    list that text was then never looked at by anything - so the exit load, one
    of the facts this assistant exists to report, silently vanished from the
    corpus. Document order means the <span> is still reached, because the <li>
    that skipped it did not mark anything consumed.
    """
    nodes: list[Node] = []
    for el in root.find_all(["p", "li", "blockquote", "div", "section", "td", "span"]):
        if _is_consumed(el):
            continue
        txt = _tidy(el.get_text(" ", strip=True))
        if not (MIN_PROSE_CHARS < len(txt) <= 1200):
            continue
        if _has_substantive_descendant(el):
            # A wrapper whose children hold the real text - let them speak.
            continue
        kind = {"li": "list_item", "blockquote": "quote", "span": "list_item"}.get(
            el.name, "paragraph")
        nodes.append(Node(type=kind, text=txt))
        _mark_subtree(el)
    return nodes


# ---------------------------------------------------------------------------
# Step 5 - PII
# ---------------------------------------------------------------------------


def _scrub_node(node: Node) -> tuple[Node, list[Any]]:
    """Scrub a node's text/label/value. Returns (node, remaining_high_pii)."""
    remaining: list[Any] = []
    for attr in ("text", "label", "value"):
        val = getattr(node, attr, None)
        if not isinstance(val, str) or not val:
            continue
        clean, _ = scrub_pii(val)
        setattr(node, attr, clean)
        remaining.extend(m for m in scan_pii(clean) if m.confidence == "high")
    return node, remaining


# ---------------------------------------------------------------------------
# Step 6 - assembly
# ---------------------------------------------------------------------------


def _meta_for(doc_id: str) -> dict[str, Any] | None:
    for s in load_schemes():
        if s["scheme_slug"] == doc_id:
            return {
                "source_url": s["url"],
                "scheme_slug": s["scheme_slug"],
                "scheme_name": s["scheme_name"],
                "short_name": s["short_name"],
                "aliases": s.get("aliases", []) or [],
                "category": s["category"],
                "plan": s["plan"],
                "publisher": s.get("publisher", "Groww"),
                "page_type": s.get("page_type", "scheme_page"),
            }
    for d in load_sources().get("supplementary", []):
        if d["doc_id"] == doc_id:
            return {
                "source_url": d["url"],
                "scheme_slug": None,
                "scheme_name": None,
                "short_name": None,
                "aliases": [],
                "category": None,
                "plan": None,
                "publisher": d.get("publisher", ""),
                "page_type": d.get("page_type", "amc_page"),
            }
    return None


def clean_document(html: str, meta: dict[str, Any]) -> tuple[Document | None, list[dict[str, Any]]]:
    """Clean one page. Returns (document, quarantined_nodes)."""
    import hashlib

    selectors = load_config()["chunking"]["strip_selectors"]
    ck = load_config()["chunking"]
    soup = BeautifulSoup(html, "lxml")
    page_title = _tidy(soup.title.get_text()) if soup.title else meta.get("short_name") or meta["doc_id"]

    # MUST run before strip_boilerplate, for two reasons:
    #   1. the scheme attribute pills live inside the page <header>, which the
    #      strip pass decomposes - "ELSS - 3Y Lock-in" is the only lock-in source
    #      on the ELSS page and it was being silently lost.
    #   2. "script" is in chunking.strip_selectors, so the JSON-LD FAQPage
    #      block is gone by the time the rest of the pipeline runs.
    faq_pairs = extract_jsonld_faq(soup)
    pill_pairs = rescue_tag_pills(soup)
    expanded = expand_truncated_values(soup)

    strip_boilerplate(soup, selectors)
    root = soup.body if soup.body is not None else soup

    # --- collect every structural fact, richest extractor first ---------
    # Order matters: label/value pairs are the richest signal, so they claim
    # nodes first. The badge pass then only sees leftovers - which is what stops
    # it from swallowing "Expense ratio" and losing the "1.03%" beside it.
    table_pairs = extract_table_rows(root, enabled=bool(ck.get("include_table_rows", False)))
    # Runs before the leaf-pair scan: the AMC's exit-load section is an <h2> plus
    # a <ul>, so the "Exit Load" heading is also a candidate label there. Claiming
    # the whole block here keeps the label from pairing with the bullet text as
    # if a bullet were a value.
    section_pairs = extract_heading_section_pairs(root)
    dom_pairs = extract_label_value_pairs(root)
    badge_pairs = extract_fact_badges(
        root,
        int(ck.get("fact_badge_max_chars", 40)),
        exclude={label for label, _v, _o in dom_pairs},
    )

    nodes: list[Node] = []

    for label, value, origin in (pill_pairs + section_pairs + dom_pairs
                                 + table_pairs + badge_pairs):
        if not label:
            continue
        nodes.append(
            Node(
                type="table_row",
                text=f"{label}: {value}" if value else label,
                label=label,
                value=value,
                origin=origin,
            )
        )

    # Headings and prose.
    nodes.extend(extract_headings(root))
    nodes.extend(extract_prose(root))

    # FAQ JSON-LD gets its own clearly-labelled section at the end.
    if faq_pairs:
        nodes.append(Node(type="heading", text="Frequently asked questions (page data)", level=3, origin="jsonld"))
        for q, a, origin in faq_pairs:
            nodes.append(Node(type="accordion", text=f"{q} {a}", label=q, value=a, origin=origin))

    # --- document order + de-duplication --------------------------------
    # Keep first occurrence of identical (type, text).
    seen: set[tuple[str, str]] = set()
    ordered: list[Node] = []
    for n in nodes:
        key = (n.type, n.text[:160])
        if key in seen:
            continue
        seen.add(key)
        ordered.append(n)

    # --- PII scrub + quarantine ------------------------------------------
    quarantined: list[dict[str, Any]] = []
    clean_nodes: list[Node] = []
    all_matches: list[Any] = []
    for n in ordered:
        n, remaining = _scrub_node(n)
        all_matches.extend(remaining)
        if remaining:
            # Still PII after scrubbing -> quarantine, never chunk it (FR-1.5).
            quarantined.append(
                {
                    "doc_id": meta["doc_id"],
                    "source_url": meta.get("source_url"),
                    "node_type": n.type,
                    "pii_report": pii_report(remaining),
                    "text_hash": short_hash(n.text, 12),
                    "quarantined_at": utc_now_iso(),
                }
            )
            continue
        clean_nodes.append(n)

    for i, n in enumerate(clean_nodes):
        n.order = i

    if not clean_nodes:
        LOG.error("  %s produced 0 nodes - page is probably JS-rendered", meta["doc_id"])
        return None, quarantined

    return (
        Document(
            doc_id=meta["doc_id"],
            source_url=meta.get("source_url", ""),
            page_title=page_title,
            publisher=meta.get("publisher", ""),
            page_type=meta.get("page_type", "scheme_page"),
            scheme_slug=meta.get("scheme_slug"),
            scheme_name=meta.get("scheme_name"),
            short_name=meta.get("short_name"),
            aliases=list(meta.get("aliases", []) or []),
            category=meta.get("category"),
            plan=meta.get("plan"),
            nodes=clean_nodes,
            raw_sha256="sha256:" + hashlib.sha256(html.encode("utf-8")).hexdigest(),
            source_fetched_at=meta.get("source_fetched_at", utc_now_iso()),
            ingested_at=utc_now_iso(),
            pii_scan=pii_report(all_matches) if all_matches else "clean",
            node_count=len(clean_nodes),
            text_chars=sum(len(n.text) for n in clean_nodes),
        ),
        quarantined,
    )


def clean_all(*, force: bool = False, only: list[str] | None = None) -> tuple[list[Document], list[dict]]:
    """Clean every cached raw page. Reuses data/corpus.jsonl unless --force."""
    common.ensure_dirs()
    corpus_path = common.path_for("corpus_file")
    quarantine_path = common.path_for("quarantine_file")

    if not force and corpus_path.exists():
        cached = read_json(corpus_path, None)
        if isinstance(cached, dict) and cached.get("documents"):
            LOG.info("Reusing existing %s (%d docs) - pass --force to re-clean",
                     corpus_path.name, len(cached["documents"]))
            docs = []
            for d in cached["documents"]:
                d = dict(d)
                d["nodes"] = [Node(**n) for n in d["nodes"]]
                docs.append(Document(**d))
            return docs, list(cached.get("quarantined", []))

    snapshot = read_json(common.path_for("snapshot_file"), {}) or {}
    docs: list[Document] = []
    quarantined: list[dict] = []

    for doc_id, snap in sorted(snapshot.items()):
        if doc_id.startswith("_"):
            continue
        if only and doc_id not in only:
            continue
        if not snap.get("sha256"):
            LOG.warning("  %s - not in snapshot or never fetched; skipping", doc_id)
            continue

        raw_file = common.path_for("raw_dir") / f"{doc_id}.html"
        if not raw_file.exists():
            LOG.warning("  %s - raw HTML missing at %s; skipping", doc_id, raw_file)
            continue

        meta = _meta_for(doc_id)
        if meta is None:
            LOG.warning("  %s - not listed in config/sources.yaml; skipping", doc_id)
            continue
        meta["doc_id"] = doc_id
        meta["source_fetched_at"] = snap.get("fetched_at", utc_now_iso())

        doc, quarantined_nodes = clean_document(raw_file.read_text(encoding="utf-8", errors="replace"), meta)
        if doc is None:
            continue
        docs.append(doc)
        quarantined.extend(quarantined_nodes)
        for q in quarantined_nodes:
            append_jsonl(quarantine_path, q)

    write_jsonl(corpus_path, [d.to_dict() for d in docs])
    write_json(corpus_path, {"generated_at": utc_now_iso(), "documents": [d.to_dict() for d in docs],
                             "quarantined": quarantined})
    return docs, quarantined


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(docs: list[Document], quarantined: list[dict]) -> int:
    print("=" * 78)
    print("  STAGE 1b - CLEAN / NODE STREAM REPORT")
    print("=" * 78)
    print(f"  {'doc_id':<44} {'nodes':>6} {'chars':>8}  types")
    print("  " + "-" * 74)
    for d in docs:
        counts: dict[str, int] = {}
        for n in d.nodes:
            counts[n.type] = counts.get(n.type, 0) + 1
        types = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {d.doc_id[:44]:<44} {d.node_count:>6} {d.text_chars:>8,}  {types}")
    print("  " + "-" * 74)
    print(f"  documents={len(docs)}  nodes={sum(d.node_count for d in docs)}  "
          f"chars={sum(d.text_chars for d in docs):,}  quarantined={len(quarantined)}")

    # Show the facts we actually care about, per scheme.
    #
    # The search terms are the AMC's own labels, because that is what the corpus
    # is now built from. hdfcfund.com states the total expense ratio as "TER"
    # and the SIP floor as "Min SIP"; an earlier version of this report looked
    # for "expense ratio" and "min. for sip", which are Groww's wording, and
    # therefore reported ABSENT for two facts that were present and correct.
    # A coverage check that cannot see the facts is worse than no check.
    wanted = ("ter", "min sip", "benchmark", "lock", "exit load", "aum", "nav",
              "riskometer", "entry load", "inception")
    always = ("ter", "min sip", "benchmark", "riskometer")
    # Word-bounded, or "ter" matches inside "riskometer" and the report claims
    # the expense ratio is "Very High".
    wanted_re = {w: re.compile(r"\b" + re.escape(w) + r"\b") for w in wanted}
    print()
    print("  Fact coverage (ADR-001 check - is each answerable fact a node?):")
    for d in docs:
        if not d.scheme_slug:
            continue
        found: dict[str, str] = {}
        for n in d.nodes:
            if n.type not in ("table_row", "accordion"):
                continue
            key = (n.label or n.text).lower()
            for w in wanted:
                if wanted_re[w].search(key) and w not in found:
                    # "-" means the fact is present but stated inline
                    # (e.g. the "ELSS - 3Y Lock-in" pill) rather than as a
                    # label/value split. ABSENT would mean the fact is missing.
                    found[w] = _tidy(n.value or "")[:30] or "(inline)"
        cells = " | ".join(
            f"{w}={found.get(w, 'ABSENT')}" for w in wanted if w in found or w in always
        )
        print(f"    {d.short_name:<24} {cells}")

    if quarantined:
        print()
        print(f"  !! {len(quarantined)} node(s) QUARANTINED for PII (never chunked):")
        for q in quarantined[:5]:
            print(f"     {q['doc_id']} {q['node_type']}: {q['pii_report']}")

    if not docs:
        print("\n  !! NO DOCUMENTS PRODUCED - check data/ingestion_log.jsonl")
    print("=" * 78)
    return 0 if docs else 1


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1b - clean raw HTML into a node stream")
    ap.add_argument("--force", action="store_true", help="re-clean even if corpus.jsonl exists")
    ap.add_argument("--doc-id", action="append", default=None, help="only this doc_id (repeatable)")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()
    docs, quarantined = clean_all(force=args.force, only=args.doc_id)
    return report(docs, quarantined)


if __name__ == "__main__":
    sys.exit(_main())
