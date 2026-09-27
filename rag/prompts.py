"""
rag/prompts.py - STAGE 6: the prompt, and the rules it encodes.

A prompt is not prose around a model, it is a control surface. Everything the
PRD promises - at most three sentences, exactly one source, no advice, no
computed returns, no figures that are not in the context - is stated here once
and enforced twice more downstream (grounding check in `guardrails`, and the
validator). Writing it in one place and pointing at the other two is the only
way to keep the three from drifting apart silently.

The system prompt
-----------------
Note what it does NOT contain: any scheme's name, any number, any answer. Those
belong in the user message, where they come from the corpus. A system prompt
containing facts is a system prompt that can leak them into a question the
corpus does not cover, which is precisely the failure the out-of-corpus path
exists to prevent.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

import common

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a factual assistant for 5 HDFC mutual fund schemes (Direct Growth plans).

Rules, in priority order:

1. Answer ONLY from the numbered CONTEXT below. It is the complete set of
   facts you have. Your training data is not a source. If the context does not
   contain the answer, say you could not find it - do not reason toward it,
   do not estimate it, do not recall it from elsewhere.
2. Report figures EXACTLY as the context writes them. Do not convert units,
   round, reformat, or recompute. If the context says "1,13,606.47 cr", that is
   the figure. Never calculate a derived number - no ratios, no growth, no
   totals, no differences between two funds.
3. Never state or imply a return, ranking, or comparison. No "performed well",
   no "outperformed", no "returns of X%". You may name a benchmark index if the
   context names it, and you may quote a figure the page publishes.
4. Never advise. No "you should", "we recommend", "worth buying", "ideal for",
   "safe to invest". You describe what a page says; you do not tell anyone what
   to do with money.
5. Use AT MOST 3 sentences. Use ONE source link: the single most relevant
   context block's URL. Do not list several.
6. If the context is absent, irrelevant, or does not answer the question,
   reply with exactly: NOT_FOUND

Never output anything except the answer prose.\
"""

NOT_FOUND = "NOT_FOUND"

USER_PROMPT_TEMPLATE = """\
QUESTION: {question}

CONTEXT:
{context}

Answer in at most 3 sentences, citing one source link. If the context does not \
answer the question, reply with exactly: {not_found}"""

REFUSAL_PROMPT_TEMPLATE = """\
QUESTION: {question}

The question was refused for this reason: {reason}

Do not answer it. Do not offer an alternative question as if it were an \
answer. Reply with exactly: {not_found}"""

# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------

# The citation is rendered once, from the block that was actually used, so the
# URL in the prose and the URL in `Answer.sources` cannot disagree. A model told
# to "cite a link" and a model told to "copy the URL after the tag" fail at very
# different rates.
BLOCK_TEMPLATE = """\
[BLOCK {n}]
Source: {url}
Scheme: {scheme}
Section: {section}
{fact_line}{text}"""


def format_context(chunks: Sequence[Any], max_chunks: int = 5) -> str:
    """Render retrieved chunks as the numbered CONTEXT block.

    Takes RetrievedChunk objects, dicts, or anything with `.metadata`/`.text`,
    because the same function is called from the live path and from tests that
    build chunks by hand.
    """
    lines: list[str] = []
    for i, c in enumerate(list(chunks)[:max_chunks], start=1):
        if isinstance(c, dict):
            text = str(c.get("text", ""))
            meta = c.get("metadata") or c
        else:
            text = str(getattr(c, "text", ""))
            meta = dict(getattr(c, "metadata", {}) or {})

        fact_line = ""
        if meta.get("fact_key"):
            fact_line = f"Fact: {meta['fact_key']} = {meta.get('fact_value', '')}\n"

        lines.append(BLOCK_TEMPLATE.format(
            n=i,
            url=meta.get("source_url", ""),
            scheme=meta.get("scheme_name", ""),
            section=meta.get("section_heading", ""),
            fact_line=fact_line,
            text=text.strip(),
        ))
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: Sequence[Any], max_chunks: int = 5
                      ) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        context=format_context(chunks, max_chunks=max_chunks),
        not_found=NOT_FOUND,
    )


def build_refusal_prompt(question: str, reason: str) -> str:
    return REFUSAL_PROMPT_TEMPLATE.format(
        question=question.strip(), reason=reason, not_found=NOT_FOUND,
    )


# ---------------------------------------------------------------------------
# The rules as data, so they can be asserted on
# ---------------------------------------------------------------------------

# Mirrors SYSTEM_PROMPT. The prompt text and this list are kept adjacent on
# purpose: a test asserts every rule here appears in SYSTEM_PROMPT, so the two
# cannot drift.
ENFORCED_RULES: list[dict[str, str]] = [
    {"id": "context_only",
     "text": "Answer ONLY from the numbered CONTEXT"},
    {"id": "exact_figures",
     "text": "Report figures EXACTLY as the context writes them"},
    {"id": "no_derivation",
     "text": "Never calculate a derived number"},
    {"id": "no_performance",
     "text": "Never state or imply a return, ranking, or comparison"},
    {"id": "no_advice",
     "text": "Never advise"},
    {"id": "three_sentences",
     "text": "Use AT MOST 3 sentences"},
    {"id": "one_source",
     "text": "Use ONE source link"},
    {"id": "not_found",
     "text": "reply with exactly: NOT_FOUND"},
]


def render_for_humans(chunks: Sequence[Any], question: str = "") -> str:
    """The full prompt as it would be sent. Used by `python -m rag.prompts`."""
    max_chunks = int(common.load_config()["generation"].get("context_chunks", 5))
    return (
        "=== SYSTEM ===\n"
        + SYSTEM_PROMPT
        + "\n\n=== USER ===\n"
        + build_user_prompt(question, chunks, max_chunks=max_chunks)
    )


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Show the Stage 6 prompt with real retrieved context")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--show-rules", action="store_true",
                    help="print the enforced-rule checklist")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()

    if args.show_rules:
        print("=" * 78)
        print("  ENFORCED RULES (each must appear in SYSTEM_PROMPT)")
        print("=" * 78)
        for rule in ENFORCED_RULES:
            present = rule["text"] in SYSTEM_PROMPT
            print(f"  {'ok  ' if present else 'MISS'} {rule['id']:<16} "
                  f"{rule['text']}")
        return 0

    question = " ".join(args.question) or "What is the expense ratio of HDFC Large Cap?"
    from rag.retrieve import retrieve  # noqa: PLC0415

    print("=" * 78)
    print(f"  PROMPT FOR: {question}")
    print("=" * 78)
    print(render_for_humans(retrieve(question), question))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
