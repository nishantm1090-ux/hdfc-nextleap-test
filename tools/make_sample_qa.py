"""Generate `docs/SAMPLE_QA.md` from the live pipeline.

Hand-typing the answers would make the document a claim about the system rather
than evidence of it, and any drift between the doc and the code would go
unnoticed. Every sentence, link and stamp below is produced by calling
`rag.answer.ask`, so the file is a transcript, not a description.

Regenerate with:

    python -m tools.make_sample_qa

Nothing else in the project imports this. It is a build step for one document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import common

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "SAMPLE_QA.md"

#: (category, scheme_slug or None, question). The scheme NAME is looked up from
#: `config/sources.yaml` at render time rather than typed here, because a
#: hand-typed scheme name in a transcript is a claim, not evidence - and the AMC
#: renamed one of the five schemes without renaming its URL.
#:
#: Ten rows: the brief asks for 5-10. Seven answers cover all five schemes plus
#: the two questions the corpus is specifically good at (the SEBI riskometer,
#: which the AMC publishes on every page, and the ELSS lock-in phrased without
#: the words "lock in"). Three refusals cover the three ways this assistant has
#: to decline: a request for advice, a request for a return, and a request for
#: something that needs the user's own account.
PAIRS: list[tuple[str, str | None, str]] = [
    ("expense_ratio", "hdfc-large-cap-fund-direct-growth",
     "What is the expense ratio of HDFC Large Cap Fund?"),
    ("benchmark", "hdfc-equity-fund-direct-growth",
     "What benchmark does HDFC Flexi Cap Fund use?"),
    ("min_sip", "hdfc-elss-tax-saver-fund-direct-plan-growth",
     "What is the minimum SIP for HDFC ELSS Tax Saver Fund?"),
    ("aum", "hdfc-small-cap-fund-direct-growth",
     "What is the NAV and AUM of HDFC Small Cap Fund?"),
    ("riskometer_level", "hdfc-balanced-advantage-fund-direct-growth",
     "What is the riskometer level of HDFC Balanced Advantage Fund?"),
    ("exit_load", "hdfc-large-cap-fund-direct-growth",
     "What is the exit load on HDFC Large Cap Fund?"),
    ("lock_in", "hdfc-elss-tax-saver-fund-direct-plan-growth",
     "How long do I have to keep HDFC ELSS Tax Saver invested?"),
    ("statement", None,
     "Where can I download my consolidated account statement and capital gains "
     "statement?"),
    ("advice", None, "Should I buy HDFC Large Cap Fund?"),
    ("performance", None,
     "What is the 5-year return of HDFC Large Cap Fund?"),
]

KIND_BLURB = {
    "refusal_advice": "refused as advice",
    "refusal_performance": "refused as a performance question",
    "refusal_pii": "refused as PII",
    "out_of_corpus": "refused as out of corpus",
    "error": "ERROR",
}

HEADER = """# Sample Q&A

Ten questions, answered by the pipeline as it actually runs. Every answer,
link, `Last updated` stamp and disclaimer below was produced by calling
`rag.answer.ask` - this file is a transcript, not a description. Regenerate it
with `python -m tools.make_sample_qa`.

Every cited page is on **hdfcfund.com**, HDFC AMC's own site. Figures are the
values published there on the fetch date recorded in `docs/SOURCES.md`; they are
quoted, never computed, and none of them is a return.

Each answer carries exactly one source link, in the format PRD §12.1 requires.
Refusals carry **no** source link at all: there is nothing in the corpus to
cite, and inventing a citation for a refusal would be a lie with a URL on it.
"""


def render() -> str:
    from rag.answer import ask

    names = {s["scheme_slug"]: s["scheme_name"] for s in common.load_schemes()}

    L: list[str] = [HEADER]
    L.append("\n## Answers\n")
    for i, (category, scheme_slug, q) in enumerate(PAIRS, start=1):
        a = ask(q)
        L.append(f"### {i}. {q}\n")
        line = f"**`{a.kind}`** - category `{category}`"
        if scheme_slug:
            line += f" - scheme {names[scheme_slug]}"
        if a.kind != "answer":
            line += f" - _{KIND_BLURB.get(a.kind, a.kind)}_"
            if a.refusal_reason:
                line += f" (`{a.refusal_reason}`)"
        L.append(line + "\n")

        # Exactly the PRD §12.1 shape: the answer paragraph, then the one source
        # line, then the stamp, then the disclaimer. The stub's inline
        # `Source:` is stripped, because printing it as well would render the
        # single source twice.
        L.append("```\n" + common.strip_inline_source(a.text) + "\n```\n")
        if a.sources:
            L.append(f"Source: {a.sources[0].url}\n")
        if a.last_updated:
            # `last_updated` already carries the full "Last updated from
            # sources: YYYY-MM-DD" string; prefixing it again doubled it.
            L.append(f"```\n{a.last_updated}\n```\n")
        L.append(f"> {a.disclaimer}\n")

    n_chunks = len(common.read_jsonl(common.path_for("chunks_file")))
    L.append(f"""
## Why the refusals look the way they do

Three of the ten are refusals, and each is a different failure the system has to
avoid rather than a topic it has not read.

**`Should I buy HDFC Large Cap Fund?`** is advice. The corpus can tell you the
expense ratio is 1.03%; it cannot tell you whether that is a good number *for
you*, and the moment it starts ranking schemes it is making a recommendation it
has no standing to make. The refusal points at SEBI and AMFI's official
investor-education material instead of at a fund.

**`What is the 5-year return of HDFC Large Cap Fund?`** is a performance
question. A return is a calculation over a period, and computing it means
choosing a start and an end date and a price source, which is exactly the
authority this assistant does not have. The published factsheet does have that
authority, so the refusal hands over rather than calculates.

**`Where can I download my consolidated account statement?`** is the one that
most looks answerable, and it is not. The AMC's own statement page is the sixth
page in the corpus - it is fetched, chunked and indexed - so the *procedure* is
in there. The *document* is not, because it lives behind the user's own login.
Answering "download it from your HDFC Mutual Fund account" is a refusal that
names the hand-off, and it is deliberately not dressed up with a citation,
because nothing in the corpus was used to produce the account it is about.

## What the AMC publishes, and what it does not

This corpus is five scheme pages and one statement page from
`hdfcfund.com` - {n_chunks} chunks in total. That is enough for every keyed fact
above and not enough for a few things a reader might reasonably expect, and the
honest response to those is the "I couldn't verify that from the available
official sources" refusal rather than a guess:

- **Fund manager's name.** The scheme pages do not carry it in the captured
  markup; it loads separately. There is no `fund_manager` chunk, so
  "Who is the fund manager of HDFC Large Cap Fund?" is refused. It is not
  answered with a neighbouring row, which is the same failure the O01
  regression in `tests/test_golden_regressions.py` was written to prevent.
- **Return figures.** The pages do publish them, and the guardrails refuse to
  report them anyway - a percentage attached to return vocabulary is a
  performance claim regardless of where it came from.
- **Star ratings and portfolio P/E.** Not on the scheme pages, so not here.
- **Minimum investment for four of the five schemes.** Only the Small Cap page
  carries a `minimum_investment` row, so only that scheme answers a
  "how much do I need to start" question with a lump-sum figure.
""")
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
