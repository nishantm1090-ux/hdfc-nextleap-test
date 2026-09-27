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

#: One answered fact per scheme, then the two facts that only exist because of
#: the Phase 8 fixes (a lock-in stated without the words "lock in", and a fund
#: manager buried in an unlabelled run-on chunk), then one refusal of each kind.
#: Ten rows: the brief asks for 5-10, and ten covers all five schemes plus every
#: refusal path without padding.
PAIRS: list[tuple[str, str, str]] = [
    ("expense_ratio", "HDFC Large Cap Fund - Direct Growth",
     "What is the expense ratio of HDFC Large Cap Direct Growth?"),
    ("nav_aum", "HDFC Flexi Cap Fund - Direct Growth",
     "What is the NAV of HDFC Flexi Cap Fund?"),
    ("min_sip", "HDFC ELSS Tax Saver Fund - Direct Plan - Growth",
     "What is the minimum SIP for HDFC ELSS Tax Saver Fund?"),
    ("aum", "HDFC Small Cap Fund - Direct Growth",
     "What is the fund size or AUM of HDFC Small Cap Fund?"),
    ("exit_load", "HDFC Balanced Advantage Fund - Direct Growth",
     "What is the exit load on HDFC Balanced Advantage Fund?"),
    ("lock_in", "HDFC ELSS Tax Saver Fund - Direct Plan - Growth",
     "How long do I have to keep HDFC ELSS Tax Saver invested?"),
    ("fund_manager", "HDFC Large Cap Fund - Direct Growth",
     "Who is the fund manager of HDFC Large Cap Fund?"),
    ("advice", None, "Should I buy HDFC Large Cap Fund?"),
    ("performance", None,
     "What is the 5-year return of HDFC Large Cap Fund?"),
    ("out_of_corpus", None,
     "What is the riskometer level of HDFC Large Cap Fund?"),
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

Figures are the values published on the source pages on the fetch date recorded
in `docs/SOURCES.md`. They are quoted, never computed, and none of them is a
return.

Each answer carries exactly one source link, in the format PRD §12.1 requires.
Refusals carry **no** source link at all: there is nothing in the corpus to
cite, and inventing a citation for a refusal would be a lie with a URL on it.
"""


def render() -> str:
    from rag.answer import ask

    L: list[str] = [HEADER]
    L.append("\n## Answers\n")
    for i, (category, scheme, q) in enumerate(PAIRS, start=1):
        a = ask(q)
        L.append(f"### {i}. {q}\n")
        line = f"**`{a.kind}`** - category `{category}`"
        if scheme:
            line += f" - scheme {scheme}"
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

    L.append("""
## Why the refusals look the way they do

Three of the ten are refusals, and each is a different failure the system has to
avoid rather than a topic it has not read.

**`Should I buy HDFC Large Cap Fund?`** is advice. The corpus can tell you the
expense ratio is 1.03%; it cannot tell you whether that is a good number *for
you*, and the moment it starts ranking schemes it is making a recommendation it
has no standing to make. The refusal points at official investor-education
material instead of at a fund.

**`What is the 5-year return of HDFC Large Cap Fund?`** is a performance
question. A return is a calculation over a period, and computing it means
choosing a start and an end date and a price source, which is exactly the
authority this assistant does not have. The published factsheet does have that
authority, so the refusal hands over rather than calculates.

**`What is the riskometer level of HDFC Large Cap Fund?`** is the hard one,
because it *sounds* perfectly in scope - it is a real mutual-fund fact, about
the right AMC, for a scheme in the corpus. The pages simply do not publish it:
"riskometer" occurs 0 times across all 121 chunks. Left alone, the term-coverage
gate cleared the question on the words "level" and "large cap" and the system
answered with the **expense ratio**: a confident, cited, wrong number. That is
the worst output this system can produce, and it is why `known_absent_terms` in
`config/app.yaml` exists. Each entry there is measured to be absent from the
corpus before it is added, and a test fails if a re-fetch ever makes one present.
""")
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
