"""Generate the root `sample_qa.md` submission file from the live pipeline.

The brief asks for 8-10 sample Q&A with, for each, the question, the assistant's
answer, the official source link and the date the sources were last read. Every
one of those four fields is produced here by calling `rag.answer.ask` and
rendering what comes back, so the file is a transcript of the running system and
not a description of it. A hand-typed answer in a submission document is a claim
about the system; this is evidence.

This is the *root* copy, shaped to the brief's four required fields.
`docs/SAMPLE_QA.md` (via `tools/make_sample_qa.py`) is the engineering copy, and
it carries the per-stage `kind` codes and the reasoning behind each refusal.

Regenerate with:

    python -m tools.make_submission_qa

Nothing else in the project imports this. It is a build step for one document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import common

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_qa.md"

#: (category, question). Ten rows: the brief asks for 8-10, and ten covers every
#: fact type the corpus holds plus every way the assistant has to decline.
#:
#: Six are answered. The facts are spread across four of the five schemes so a
#: reader can see the scheme filter working - HDFC Balanced Advantage is left
#: out of the answered set on purpose, because its exit load comes from a prose
#: chunk rather than a fact row and showing both paths is covered in
#: `docs/SAMPLE_QA.md` instead.
#:
#: Four are refused, one for each reason that is a *policy* rather than a gap:
#: a statement the assistant cannot produce, a request for advice, a request for
#: a return, and personal data.
PAIRS: list[tuple[str, str]] = [
    ("Minimum SIP", "What is the minimum SIP for HDFC Large Cap Direct Growth?"),
    ("Expense ratio", "What is the expense ratio of HDFC ELSS Tax Saver Direct Growth?"),
    ("Exit load", "What is the exit load on HDFC Large Cap if I redeem within 1 year?"),
    ("Benchmark", "What benchmark does HDFC Flexi Cap Direct Growth track?"),
    ("Riskometer", "What is the riskometer level of HDFC Small Cap Fund?"),
    ("ELSS lock-in", "What is the lock-in period on HDFC ELSS Tax Saver?"),
    ("Capital gains statement",
     "How do I download my capital gains statement?"),
    ("Investment advice", "Should I buy HDFC Small Cap Fund for my retirement?"),
    ("Returns / comparison",
     "How much did HDFC Small Cap Fund return last year?"),
    ("Personal data",
     "My PAN is ABCDE1234F. What is the exit load on HDFC Flexi Cap?"),
]

#: Questions the AMC does not publish answers for on a scheme page. Not counted
#: in the ten - they are here because "what does it refuse, and why" is half of
#: what a reviewer is checking, and the honest answer needs its own list.
#:
#: The third element overrides the reason the pipeline reports, and it exists
#: for one row. The automatic reason for a missing fact is "no chunk in the
#: corpus carries that fact_key", which is true for `fund_manager` and false for
#: `minimum_investment` - the Small Cap page *does* publish one. Rendering the
#: automatic reason there would put a false statement in a submission document to
#: save three words, so the row carries its own reason.
ALSO_REFUSED: list[tuple[str, str, str | None]] = [
    ("Fund manager's name", "Who is the fund manager of HDFC Large Cap Fund?", None),
    ("Portfolio P/E", "What is the PE ratio of HDFC Small Cap Fund?", None),
    ("Direct vs Regular plan",
     "Is HDFC Large Cap Fund Direct Growth the Direct plan or the Regular plan?",
     None),
    ("Minimum lump-sum investment (any scheme except Small Cap)",
     "What is the minimum lump sum needed to invest in HDFC Large Cap Fund?",
     "the only `minimum_investment` chunk in the corpus belongs to HDFC Small Cap, "
     "so asking for any other scheme's is a question the pages do not answer"),
]

#: What the refusal column says, in the assistant's own words. Kept here so the
#: prose explaining a refusal cannot drift away from the string the code returns.
UNVERIFIED = "I couldn't verify that from the available official sources."

#: Why each refusal kind is the right refusal, rather than a gap in reading.
WHY: dict[str, str] = {
    "Capital gains statement":
        "The document exists, but only behind the holder's own login. The AMC's "
        "statement page is in the corpus, so the assistant can name *where* to "
        "get it, and it hands off rather than pretending to produce the data.",
    "Investment advice":
        "The corpus holds all five schemes' expense ratios, so it could rank "
        "them by cost - and that ranking would itself be a recommendation. The "
        "numbers are available; the judgement is withheld.",
    "Returns / comparison":
        "A return needs a start date, an end date and a price source, and "
        "choosing them is an act of authority this assistant does not have. The "
        "refusal hands over to the published factsheet, which does.",
    "Personal data":
        "The identifier is not stored, not used and not written to any file. The "
        "refusal names the categories it will not accept so the user knows what "
        "to do instead.",
}

HEADER = """# Sample Q&A

**Live app:** <https://hdfc-nextleap-test.onrender.com/>

Ten questions against the running system. Every answer, source link and
`Last updated from sources` date below was produced by calling the pipeline
(`rag.answer.ask`), not typed by hand. Regenerate it with:

```powershell
.\\.venv\\Scripts\\python.exe -m tools.make_submission_qa
```

**Sources: HDFC Mutual Fund, SEBI and AMFI.** Six of the ten are answered from
HDFC Mutual Fund's own scheme pages on `hdfcfund.com`, and every cited link
points there. The four refusals hand off to SEBI and AMFI's investor-education
material, which is why those two are named in the disclaimer.

Figures are quoted from the source page and never computed. Each answer is at
most three sentences and carries exactly one citation - the one page it was read
from. The `Last updated from sources` date is the day that page was captured, and
it is on every answered question because a scraped page is a snapshot: NAV, AUM
and riskometer levels move without notice, and the AMC edits these pages in
place without changing the URL.
"""


def render() -> str:
    from rag.answer import ask

    L: list[str] = [HEADER]
    L.append("\n---\n")

    for i, (category, q) in enumerate(PAIRS, start=1):
        a = ask(q)
        L.append(f"\n## {i}. {category}\n")
        L.append(f"**Question:** {q}\n")
        L.append(f"**Assistant Answer:**\n")
        L.append("```\n" + common.strip_inline_source(a.text) + "\n```\n")

        if a.kind == "answer":
            L.append(f"**Official Source Link:** {a.sources[0].url}\n")
            # `last_updated` already carries the "Last updated from sources: "
            # prefix; printing it under a heading of the same name doubled it.
            stamp = a.last_updated.split(": ", 1)[-1]
            L.append(f"**Last Updated From Sources:** {stamp}\n")
        else:
            # A refusal is a policy response, not a sourced fact, so it gets no
            # citation. Attaching a URL to a refusal would be claiming the corpus
            # produced an answer it did not.
            L.append(f"**Refused.** `{a.kind}`"
                     + (f" - {a.refusal_reason}" if a.refusal_reason else "")
                     + "\n")
            L.append("**Official Source Link:** none - nothing was retrieved, so "
                     "there is nothing to cite.\n")
            L.append(f"\n{WHY[category]}\n")

    L.append("\n---\n")
    L.append(f"""
## Also refused, and why

These are not in the ten above. They are listed separately because the honest
answer to "what does it refuse?" needs its own list, and because the interesting
property of this assistant is that it refuses *named* gaps rather than filling
them with a neighbouring fact.

HDFC Mutual Fund does not publish these on a scheme page, so there is nothing in
the corpus to retrieve. All four return the same line, verbatim:

> {UNVERIFIED}
""")
    # The rows are collected first and joined once. Appending each with a trailing
    # newline and then `"\n".join`ing puts a blank line between every row, which
    # breaks a markdown table into a paragraph followed by unrendered pipes.
    rows = ["| Asked | The assistant says | Why |", "|---|---|---|"]
    for label, q, override in ALSO_REFUSED:
        a = ask(q)
        assert common.strip_inline_source(a.text).startswith(UNVERIFIED), (
            f"{q!r} no longer produces the unverified-facts refusal; it produced "
            f"{a.kind}. Regenerating this file would put a claim in it that the "
            f"code does not back.")
        rows.append(f"| {label} | `{a.kind}` | {override or a.refusal_reason} |")
    L.append("\n".join(rows) + "\n")

    n_chunks = len(common.read_jsonl(common.path_for("chunks_file")))
    n_schemes = len(common.load_schemes())
    L.append(f"""
The distinction that matters: a refusal is either a **policy** (the four in the
ten above - advice, returns, statements, personal data) or a **gap** (this
table). A policy refusal would happen even with a perfect corpus, because the
answer is withheld on purpose. A gap refusal happens because the source does not
carry the fact, and the point of `known_absent_terms` in `config/app.yaml` is
that the assistant can say so instead of answering the fund-manager question with
the expense ratio.

## Scope of this corpus

{n_chunks} chunks from {n_schemes} HDFC Mutual Fund scheme pages plus the AMC's
Consolidated Account Statement page, captured 29 September 2026.

| Asked | Answered |
|---|---|
| expense ratio / TER | yes, for all five schemes |
| minimum SIP | yes, for all five schemes |
| exit load | yes, for all five schemes |
| ELSS lock-in | yes |
| benchmark | yes, for all five schemes |
| NAV, AUM | yes, for all five schemes |
| riskometer level | yes, for all five schemes |
| minimum lump-sum investment | Small Cap only - the only page that publishes it |
| fund manager, star rating, portfolio P/E | no - not published on a scheme page |
| Direct vs Regular plan | no - the page does not state the plan variant as a fact |
| returns, rankings, recommendations | no - withheld by policy, see above |
| your personal holdings, statements, tax figures | no - requires your login |

**One source per answer, always.** A question that genuinely spans two schemes
is refused as ambiguous rather than answered from two pages.
""")
    return "\n".join(L) + "\n"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
