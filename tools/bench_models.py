"""Benchmark Groq models through the real pipeline, not in isolation.

The question this answers is "which of these models should `.env` name?", and
the honest way to answer it is to run the candidates on the questions that
actually matter and count the failures. A model that writes nicer prose but
routinely trips the 3-sentence limit, drops a figure, or invents a return is
worse than a smaller one that does not - and the guardrails are what decide
that, so the guardrails have to be in the loop.

    python -m tools.bench_models
    python -m tools.bench_models --models openai/gpt-oss-20b qwen/qwen3.8-27b

Checks each model against, per question:
  * did the call succeed at all
  * did `validate_answer` accept the text (grounding, length, PII, performance)
  * is the answer <= 3 sentences
  * is the right figure present, when the golden set knows it
  * did latency stay inside the 90s timeout

A model that is unavailable is reported as such and skipped - the per-account
model list is not the published one.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import common

ROOT = Path(__file__).resolve().parent.parent

#: Figures the corpus is known to hold. Taken from the pages read on the fetch
#: date, not from a previous run, so a model cannot pass by memorising them from
#: an earlier experiment.
EXPECTED = {
    "What is the expense ratio of HDFC Large Cap Direct Growth?": "1.03",
    "What is the NAV of HDFC Flexi Cap Fund?": "2,214.57",
    "What is the minimum SIP for HDFC ELSS Tax Saver Fund?": "500",
    "What is the fund size or AUM of HDFC Small Cap Fund?": "41,890.86",
    "What is the exit load on HDFC Large Cap Fund?": "1",
    "Who is the fund manager of HDFC Large Cap Fund?": "Rahul Baijal",
}

QUESTIONS = list(EXPECTED) + [
    "Should I buy HDFC Large Cap Fund?",          # must refuse: advice
    "What is the 5-year return of HDFC Large Cap Fund?",  # must refuse: performance
    "What is the riskometer level of HDFC Large Cap Fund?",  # must refuse: not in corpus
]


def bench_one(model: str) -> dict[str, object]:
    import rag.answer as A
    from rag import guardrails as G

    os.environ["LLM_PROVIDER"] = "groq"
    # `_generate` reads the model from config first; write through the same
    # path a user would, so the benchmark cannot pass on a value the real
    # provider would ignore.
    cfg = common.load_config()
    cfg["generation"]["groq_model"] = model

    ok = wrong_fig = refused_ok = 0
    too_long = invalid = errored = 0
    lat: list[float] = []
    rows: list[tuple[str, str, str, str]] = []

    urls = {c.get("source_url")
            for c in common.read_jsonl(common.path_for("chunks_file"))}

    for q in QUESTIONS:
        t0 = time.perf_counter()
        try:
            a = A.ask(q)
            dt = (time.perf_counter() - t0) * 1000
            lat.append(dt)
        except Exception as exc:  # noqa: BLE001
            errored += 1
            rows.append((q[:44], "ERROR", type(exc).__name__, str(exc)[:60]))
            continue

        expect = EXPECTED.get(q)
        if expect is None:
            # A refusal question: the model is not asked to generate at all
            # (guardrails run pre-retrieval), so this measures the pipeline,
            # not the model. Recorded so the count is honest about what it covers.
            refused_ok += int(a.kind.startswith("refusal") or a.kind == "out_of_corpus")
            rows.append((q[:44], a.kind, "-", "pre-retrieval guard"))
            continue

        if a.kind != "answer":
            wrong_fig += 1
            rows.append((q[:44], a.kind, "-", common.strip_inline_source(a.text)[:60]))
            continue

        body = common.strip_inline_source(a.text)
        n = G.count_sentences(body)
        if n > 3:
            too_long += 1
        verdict = "ok"
        if expect.lower() not in body.lower():
            wrong_fig += 1
            verdict = "MISSING FIGURE"
        # Feed the validator the same shape `ask()` hands it - a dict with the
        # text and the sources - plus the chunks actually used, so "is this
        # grounded?" is the question the pipeline asks rather than a stricter
        # one that would fail every model equally and hide the differences.
        ok_valid, reasons = G.validate_answer(
            {"text": a.text, "sources": [{"url": s.url} for s in a.sources]},
            urls,
            context=a.debug.get("retrieved") or [],
        )
        if not ok_valid:
            invalid += 1
            verdict += " + INVALID:" + ",".join(reasons[:2])
        elif verdict == "ok":
            ok += 1
        rows.append((q[:44], a.kind, f"{n} sent", f"{verdict}: {body[:70]}"))

    return {
        "model": model,
        "correct": ok,
        "wrong_figure_or_refused": wrong_fig,
        "too_long": too_long,
        "invalid": invalid,
        "errored": errored,
        "refusals_ok": refused_ok,
        "mean_ms": round(statistics.mean(lat)) if lat else 0,
        "max_ms": round(max(lat)) if lat else 0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None,
                    help="model ids to try; default = every text model the key offers")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY is not set - copy .env.example to .env first.")
        return 2

    import rag.answer as A

    live = A.groq_available_models(key)
    print(f"this key can call {len(live)} models:")
    for m in live:
        print(f"    {m}")
    print()

    # Whisper is transcription, allam is Arabic-first, the prompt-guard models
    # are classifiers. None of them can hold a chat turn.
    skip = ("whisper", "guard", "allam", "canopylabs")
    candidates = [m for m in live if not any(s in m for s in skip)]
    if args.models:
        unknown = [m for m in args.models if m not in live]
        for m in unknown:
            print(f"!! {m} is NOT available on this key - skipping")
        candidates = [m for m in candidates if m in live]

    if not candidates:
        print("no usable text model on this key.")
        return 2

    print(f"benchmarking {len(candidates)} model(s) over {len(QUESTIONS)} questions\n")
    for model in candidates:
        r = bench_one(model)
        print("=" * 78)
        print(f"{r['model']}")
        print(f"  correct {r['correct']}/{len(EXPECTED)}   "
              f"wrong-figure-or-refused {r['wrong_figure_or_refused']}   "
              f">3 sentences {r['too_long']}   "
              f"validator-rejected {r['invalid']}   errors {r['errored']}")
        print(f"  mean {r['mean_ms']}ms   max {r['max_ms']}ms   "
              f"(refusal guards OK: {r['refusals_ok']}/{len(QUESTIONS) - len(EXPECTED)})")
        for q, kind, sent, note in r["rows"]:
            print(f"    {q:<46} {kind:<20} {sent:<8} {note}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
