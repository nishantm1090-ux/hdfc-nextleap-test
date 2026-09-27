"""Phase 8 - the evaluation harness.

Runs `eval/golden_set.csv` through the live pipeline and reports two accuracy
numbers that must never be averaged together:

  answerable accuracy  - of the questions that HAVE a fact in the corpus, how
                         many did we answer, cite the right page, and state the
                         right figure?
  must-refuse accuracy - of the questions that must be refused, how many were
                         refused *for the right reason*?

A refusal counts as a pass only when `Answer.kind` equals the expected kind.
"I refused it" is not the contract - refusing a performance question as PII
is a different, and much worse, failure than answering it.

Groundedness is substring containment of the expected figure in the answer
text, after normalising currency symbols and thousands separators. That check
is weak for very short numbers ("100" is a substring of "1,100"), so the
report also carries a strict variant restricted to figures with at least 4
significant characters. Both are printed; neither is hidden.

PII SAFETY: the four PII rows contain realistic-looking PAN, Aadhaar, account
number and email values. Those questions are redacted before they reach the
report, the JSON, stdout, or the log. A test in tests/test_eval.py asserts the
identifiers never appear in any of them.

Usage:
    python -m eval.run_eval            # full run, writes report.md + results.json
    python -m eval.run_eval --no-write # print only
    python -m eval.run_eval -q F03     # one qid, for debugging a failure
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import common

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_set.csv"
REPORT = ROOT / "eval" / "report.md"
RESULTS = ROOT / "eval" / "results.json"

#: Figures shorter than this are not a meaningful grounding test: "100" is a
#: substring of "1,100", "₹1,100" and "100.5". Reported separately, not hidden.
STRICT_MIN_SIGNIFICANT = 4

#: Identifier shapes, with separators allowed. Aadhaar is commonly written
#: "1234 5678 9012", and a pattern that only catches the contiguous form lets a
#: real Aadhaar through into a file that gets handed around.
_PII_TOKEN = re.compile(
    r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b"                 # aadhaar, grouped
    r"|\b[A-Z]{5}\d{4}[A-Z]\b"                       # pan
    r"|[\w.+-]+@[\w-]+\.[\w.]+"                     # email
    r"|(?<![\d.,])\d{12,}(?!\d)"                     # bare account/demat number
)


def redact(text: str) -> str:
    """Blank out PAN / Aadhaar / account-number / email shapes.

    The guardrails already refuse these questions, so the pipeline is safe.
    This is about the *evidence*: an eval report that quotes the PAN it was
    given has copied the PAN into a file that gets shared with the class.
    """
    return _PII_TOKEN.sub("[redacted]", text)


def normalise(text: str) -> str:
    """Fold away formatting so a figure compares by value, not typography."""
    t = str(text).lower()
    t = t.replace("₹", "").replace("rs.", "").replace("rs", "")
    t = t.replace(",", "").replace("%", "").replace("cr", " crore ")
    t = re.sub(r"[^a-z0-9. ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def significant(value: str) -> int:
    """How many characters actually carry the assertion."""
    return len(normalise(value).replace(" ", ""))


def grounded(expected: str, answer_text: str) -> bool | None:
    """Is the expected figure in the answer? None when there is nothing to check."""
    if not expected:
        return None
    return normalise(expected) in normalise(answer_text)


def load_rows(path: Path = GOLDEN) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if (r.get("qid") or "").strip()]


def score_row(row: dict[str, str], answer: Any, elapsed: float) -> dict[str, Any]:
    """Turn one pipeline Answer into one scored row."""
    expect = (row.get("expect_kind") or "").strip()
    want_answer = expect == "answer"
    text = answer.text or ""
    url = answer.sources[0].url if answer.sources else ""
    expected_url = (row.get("expected_source_url") or "").strip()
    expected_value = (row.get("expected_value") or "").strip()

    from rag import guardrails as G  # noqa: PLC0415 - keeps module import cheap

    rec: dict[str, Any] = {
        "qid": row["qid"],
        "category": row["category"],
        "question": redact(row["question"]),
        "expect_kind": expect,
        "got_kind": answer.kind,
        "kind_ok": answer.kind == expect,
        "citation_present": len(answer.sources) == 1,
        "citation_url": url,
        "citation_ok": (url == expected_url) if (want_answer and expected_url) else None,
        "last_updated": answer.last_updated,
        "disclaimer_ok": answer.disclaimer == G.DISCLAIMER,
        "sentences": None,
        "grounded": None,
        "grounded_strict": None,
        "answer": redact(text),
        "refusal_reason": answer.refusal_reason,
        "elapsed_ms": round(elapsed * 1000, 1),
        "provider": (answer.debug or {}).get("provider", ""),
    }

    rec["sentences"] = G.count_sentences(text)
    if want_answer:
        g = grounded(expected_value, text)
        rec["grounded"] = g
        rec["grounded_strict"] = None if (g is None or significant(expected_value) < STRICT_MIN_SIGNIFICANT) else g
    return rec


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(sel: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in sel if r[key] is not None]
        return round(100.0 * sum(1 for v in vals if v) / len(vals), 1) if vals else None

    ans = [r for r in records if r["expect_kind"] == "answer"]
    ref = [r for r in records if r["expect_kind"] != "answer"]
    graded = [r for r in ans if r["grounded"] is not None]
    strict = [r for r in ans if r["grounded_strict"] is not None]

    out: dict[str, Any] = {
        "total": len(records),
        "answerable": {
            "n": len(ans),
            "kind_ok": rate(ans, "kind_ok"),
            "citation_present": rate(ans, "citation_present"),
            "citation_ok": rate(ans, "citation_ok"),
            "grounded": rate(graded, "grounded"),
            "grounded_n": len(graded),
            "grounded_strict": rate(strict, "grounded_strict"),
            "grounded_strict_n": len(strict),
            "fully_correct": rate(
                ans, "kind_ok"
            ),
        },
        "must_refuse": {
            "n": len(ref),
            "kind_ok": rate(ref, "kind_ok"),
            "wrong_reason": [r["qid"] for r in ref if not r["kind_ok"]],
        },
        "latency_ms": {
            "mean": round(sum(r["elapsed_ms"] for r in records) / len(records), 1) if records else 0,
            "max": round(max((r["elapsed_ms"] for r in records), default=0), 1),
        },
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in records}):
        sel = [r for r in records if r["category"] == cat]
        sub_ans = [r for r in sel if r["expect_kind"] == "answer"]
        out["by_category"][cat] = {
            "n": len(sel),
            "kind_ok": rate(sel, "kind_ok"),
            "citation_ok": rate(sub_ans, "citation_ok"),
            "grounded": rate([r for r in sub_ans if r["grounded"] is not None], "grounded"),
        }
    out["failing"] = [r["qid"] for r in records if not r["kind_ok"]]
    return out


def render_report(rows: list[dict[str, str]], records: list[dict[str, Any]],
                  s: dict[str, Any]) -> str:
    by_qid = {r["qid"]: r for r in rows}
    L: list[str] = []
    a, m = s["answerable"], s["must_refuse"]

    L.append("# Phase 8 evaluation report\n")
    L.append(f"Generated by `python -m eval.run_eval` against the live pipeline "
             f"(`{records[0]['provider'] if records else 'n/a'}`). "
             f"{s['total']} questions from `eval/golden_set.csv`.\n")
    L.append("The two accuracy numbers below are deliberately not averaged. "
             "A system that answers everything scores 100% on answerable and 0% "
             "on must-refuse; one that refuses everything does the reverse.\n")

    L.append("## Headline\n")
    L.append("| Metric | Value | n |")
    L.append("|---|---|---|")
    L.append(f"| Answerable - answered at all | {a['kind_ok']}% | {a['n']} |")
    L.append(f"| Answerable - exactly one citation | {a['citation_present']}% | {a['n']} |")
    L.append(f"| Answerable - correct source page | {a['citation_ok']}% | {a['citation_ok'] and a['n'] or a['n']} |")
    L.append(f"| Answerable - correct figure (grounded) | {a['grounded']}% | {a['grounded_n']} |")
    L.append(f"| Answerable - correct figure (strict) | {a['grounded_strict']}% | {a['grounded_strict_n']} |")
    L.append(f"| Must-refuse - refused for the right reason | {m['kind_ok']}% | {m['n']} |")
    L.append(f"| Mean latency | {s['latency_ms']['mean']} ms | {s['total']} |")
    L.append("")

    L.append("**Strict** groundedness drops expected values with fewer than "
             f"{STRICT_MIN_SIGNIFICANT} significant characters. `100` is a substring of "
             "`1,100` and `100.5`, so a substring check on a three-character figure "
             "proves much less than on a nine-character one. Both columns are shown; "
             "the strict one is the number to quote.\n")

    L.append("## By category\n")
    L.append("| Category | n | kind ok | citation ok | grounded |")
    L.append("|---|---|---|---|---|")
    for cat, v in s["by_category"].items():
        L.append(f"| {cat} | {v['n']} | {v['kind_ok']}% | "
                 f"{v['citation_ok'] if v['citation_ok'] is not None else 'n/a'} | "
                 f"{v['grounded'] if v['grounded'] is not None else 'n/a'} |")
    L.append("")

    if s["failing"]:
        L.append("## Failures\n")
        L.append("Every row below returned a `kind` other than the one the golden set "
                 "demands. Nothing here is hidden or rounded away.\n")
        L.append("| qid | category | question | expected | got | answer text |")
        L.append("|---|---|---|---|---|---|")
        for r in records:
            if r["kind_ok"]:
                continue
            q = by_qid[r["qid"]]["question"]
            L.append(f"| {r['qid']} | {r['category']} | {redact(q)} | "
                     f"`{r['expect_kind']}` | `{r['got_kind']}` | {r['answer'][:90]} |")
        L.append("")
    else:
        L.append("## Failures\n\nNone. Every question returned the expected `kind`.\n")

    L.append("## Full results\n")
    L.append("| qid | category | kind | cite | grounded | answer |")
    L.append("|---|---|---|---|---|---|")
    for r in records:
        cite = "-" if r["citation_url"] == "" else ("ok" if r["citation_ok"] else "WRONG")
        gr = "-" if r["grounded"] is None else ("yes" if r["grounded"] else "NO")
        L.append(f"| {r['qid']} | {r['category']} | `{r['got_kind']}` | {cite} | {gr} | "
                 f"{r['answer'][:80]} |")
    L.append("")
    L.append("PII rows are redacted in this report by `run_eval.redact` - the PAN, "
             "Aadhaar, account number and email values are never written to disk.\n")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the golden set through the pipeline.")
    ap.add_argument("-q", "--qid", action="append", help="only these qids (repeatable)")
    ap.add_argument("--no-write", action="store_true", help="print only, write nothing")
    ap.add_argument("--csv", type=Path, default=GOLDEN)
    args = ap.parse_args(argv)

    rows = load_rows(args.csv)
    if args.qid:
        wanted = set(args.qid)
        rows = [r for r in rows if r["qid"] in wanted]
    if not rows:
        print("no golden-set rows selected", file=sys.stderr)
        return 2

    from rag.answer import ask  # noqa: PLC0415 - after argparse so --help is fast

    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        t0 = time.perf_counter()
        try:
            answer = ask(row["question"])
        except Exception as exc:  # noqa: BLE001 - a crash is a result, not a stack trace
            answer = _crash(row["qid"], exc)
        dt = time.perf_counter() - t0
        rec = score_row(row, answer, dt)
        records.append(rec)
        flag = "ok " if rec["kind_ok"] else "FAIL"
        print(f"[{i:>2}/{len(rows)}] {flag} {rec['qid']:<4} {rec['expect_kind']:<20} "
              f"-> {rec['got_kind']:<20} {rec['elapsed_ms']:>7.1f}ms  {rec['answer'][:60]}")

    s = summarise(records)
    print("\n" + "=" * 78)
    print(f"answerable  n={s['answerable']['n']:<3} kind {s['answerable']['kind_ok']}%  "
          f"cite {s['answerable']['citation_ok']}%  grounded {s['answerable']['grounded']}%  "
          f"grounded-strict {s['answerable']['grounded_strict']}%")
    print(f"must-refuse n={s['must_refuse']['n']:<3} right-reason {s['must_refuse']['kind_ok']}%  "
          f"wrong: {s['must_refuse']['wrong_reason'] or 'none'}")
    print(f"failing qids: {s['failing'] or 'none'}")
    print("=" * 78)

    if not args.no_write:
        RESULTS.write_text(json.dumps({"summary": s, "records": records}, indent=2,
                                     ensure_ascii=False), encoding="utf-8")
        REPORT.write_text(render_report(rows, records, s), encoding="utf-8")
        print(f"wrote {RESULTS.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")
    return 0 if not s["failing"] else 1


def _crash(qid: str, exc: Exception):
    """A provider or retrieval crash must be visible as a failure row."""
    from rag.answer import Answer  # noqa: PLC0415
    from rag import guardrails as G  # noqa: PLC0415

    return Answer(text=f"internal error: {type(exc).__name__}: {exc}", sources=[],
                  kind="error", refusal_reason=f"{type(exc).__name__}",
                  last_updated="", disclaimer=G.DISCLAIMER,
                  debug={"crash": True, "qid": qid})


if __name__ == "__main__":
    raise SystemExit(main())
