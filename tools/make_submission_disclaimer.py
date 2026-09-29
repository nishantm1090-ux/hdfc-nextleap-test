"""Generate the root `disclaimer.md` submission file from the code's own string.

The brief asks for a short disclaimer whose wording matches what the user sees
in the app. "Matches" has to mean byte-for-byte or it means nothing, so this
imports `rag.guardrails.DISCLAIMER` - the same object `app.py` renders under
every answer and the same object `eval/run_eval.py` scores per row - and refuses
to write a file that does not contain it exactly once.

`docs/DISCLAIMER.md` is the long-form version: the same string, plus where it
appears and how to read it. This is the short one.

Regenerate with:

    python -m tools.make_submission_disclaimer
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "disclaimer.md"

PREAMBLE = """# Disclaimer

**Live app:** <https://hdfc-nextleap-test.onrender.com/>

## The disclaimer shown in the app

This is the text exactly as the assistant shows it - under every answer, in the
UI and in the command line. It is not a summary of it.

"""

POSTAMBLE = """

## Why it is worded this way

Three sentences do the work, and each rules out one thing the assistant is built
not to do.

**"It does not recommend, compare, or rate schemes."** The corpus holds the
expense ratio of all five schemes, so it *could* rank them by cost - and that
ranking would be a recommendation, because a lower TER is often the reason a
person picks a scheme. The numbers are available. The comparison is withheld.

**"It does not compute or report returns."** A return needs a start date, an end
date and a price source. Choosing them is an act of authority, and the published
factsheets are where that authority sits. Return questions are refused and handed
over rather than answered.

**"read all scheme-related documents carefully."** Not boilerplate. The corpus is
a snapshot of web pages captured on a single date. NAV, AUM and riskometer levels
move without notice, and the AMC edits these pages in place without changing the
URL, so a page can be stale with nothing announcing it. The
`Last updated from sources:` date on every answer exists so you can see how old
the snapshot is.

## Sources

**Sources: HDFC Mutual Fund, SEBI and AMFI.**

Every fact in this assistant comes from **HDFC Mutual Fund** (HDFC Asset
Management Company Limited) - its own scheme pages on `hdfcfund.com`, and its
own Scheme Information Documents and Key Information Memoranda. **SEBI** and
**AMFI** are named because they are the regulator and the industry body whose
material a reader can check against; the assistant refuses advice and return
questions by handing over to their investor-education pages.

No broker, aggregator, blog or forum contributes a single fact. The full list of
official sources, each one opened and confirmed to load, is in
[`sources.csv`](sources.csv).

## What this assistant is not

This is an independent academic project. It is not HDFC Mutual Fund, not SEBI,
not AMFI, and not affiliated with any of them. It is not a registered investment
adviser and it is not a distributor. Nothing here is investment advice, a
recommendation, or an offer to buy or sell any security.

Mutual fund investments are subject to market risks. Read the scheme's own
Scheme Information Document, Key Information Memorandum and monthly factsheet
before investing, and consider your own circumstances.
"""


def main() -> int:
    from rag.guardrails import DISCLAIMER

    text = PREAMBLE + DISCLAIMER + POSTAMBLE
    assert text.count(DISCLAIMER) == 1, "the disclaimer must appear exactly once"
    assert "Sources: HDFC Mutual Fund, SEBI and AMFI" in DISCLAIMER, (
        "the disclaimer no longer names the three official publishers")
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes, "
          f"disclaimer {len(DISCLAIMER)} chars, verbatim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
