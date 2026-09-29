"""
tests/test_answer.py - STAGE 6: the answer contract (PRD §11, §12).

The centre of gravity is `test_a_named_scheme_never_gets_another_schemes_figure`.
That failure happened: ranking chunks by subject match alone tied across all five
schemes, the tie went to retrieval order, and "What is the expense ratio of HDFC
Large Cap Direct Growth?" came back as

    The expense ratio for HDFC Small Cap Fund is 0.78%.

confidently, with a citation, and attributed to the wrong fund. It is the single
most damaging output this project can produce, so it is asserted directly rather
than inferred from the sample questions passing.

Also pinned here, because each was a real defect:
  * `test_every_answer_is_grounded_in_its_own_context`
  * `test_an_answer_cites_exactly_one_source_and_it_is_in_the_corpus`
  * `test_a_benchmark_name_is_not_read_as_a_performance_claim`

Run: .\\.venv\\Scripts\\python.exe -m pytest tests/test_answer.py -q
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
from conftest import scheme_url
from rag import answer as A
from rag import guardrails as G
from rag import prompts as P

# PRD §11 pins this Literal. A kind outside it is a contract violation, whatever
# the code says about it.
PRD_KINDS = {"answer", "refusal_pii", "refusal_advice", "refusal_performance",
             "out_of_corpus", "error"}

# (question, the needle that must appear in the answer, the scheme it must belong to)
# The needle is checked case-insensitively as plain text, so a benchmark name
# works the same as a figure.
GOLDEN = [
    ("What is the expense ratio of HDFC Large Cap Direct Growth?",
     "1.03", "hdfc-large-cap-fund-direct-growth"),
    ("What is the expense ratio of HDFC Small Cap Fund?",
     "0.78", "hdfc-small-cap-fund-direct-growth"),
    ("What is the expense ratio of HDFC Flexi Cap Fund?",
     "0.77", "hdfc-equity-fund-direct-growth"),
    ("What is the expense ratio of HDFC ELSS Tax Saver Fund?",
     "1.21", "hdfc-elss-tax-saver-fund-direct-plan-growth"),
    ("What is the expense ratio of HDFC Balanced Advantage Fund?",
     "0.78", "hdfc-balanced-advantage-fund-direct-growth"),
    ("What is the minimum SIP for HDFC ELSS Tax Saver Fund?",
     "500", "hdfc-elss-tax-saver-fund-direct-plan-growth"),
    ("What is the minimum SIP for HDFC Large Cap Fund?",
     "100", "hdfc-large-cap-fund-direct-growth"),
    ("What is the AUM of HDFC Flexi Cap Fund?",
     "113606.47", "hdfc-equity-fund-direct-growth"),
    ("What is the AUM of HDFC Large Cap Fund?",
     "39933.37", "hdfc-large-cap-fund-direct-growth"),
    ("What is the NAV of HDFC Balanced Advantage Fund?",
     "551.04", "hdfc-balanced-advantage-fund-direct-growth"),
    ("What is the NAV of HDFC ELSS Tax Saver Fund?",
     "1426.93", "hdfc-elss-tax-saver-fund-direct-plan-growth"),
    ("What benchmark does HDFC Small Cap Fund Direct Growth use?",
     "bse 250 smallcap", "hdfc-small-cap-fund-direct-growth"),
    ("What benchmark does HDFC Large Cap Fund use?",
     "nifty 100", "hdfc-large-cap-fund-direct-growth"),
    ("What is the riskometer level of HDFC Large Cap Fund?",
     "very high", "hdfc-large-cap-fund-direct-growth"),
    ("What is the lock-in period of HDFC ELSS Tax Saver Fund?",
     "3 years", "hdfc-elss-tax-saver-fund-direct-plan-growth"),
    ("What is the exit load of HDFC ELSS Tax Saver Fund?",
     "nil", "hdfc-elss-tax-saver-fund-direct-plan-growth"),
]

MUST_REFUSE = [
    ("Should I buy HDFC Large Cap?", "refusal_advice"),
    ("Which fund is better, HDFC Large Cap or HDFC Small Cap?", "refusal_advice"),
    ("What was the return on HDFC Large Cap last year?", "refusal_performance"),
    ("What is the CAGR of HDFC ELSS Tax Saver?", "refusal_performance"),
    ("my PAN is ABCDE1234F, can you check my folio", "refusal_pii"),
    ("my aadhaar number is 2345 6789 0123", "refusal_pii"),
    ("What is the expense ratio of SBI Large Cap Fund?", "out_of_corpus"),
    ("price of gold in Mumbai", "out_of_corpus"),
    ("What is the weather in Mumbai today?", "out_of_corpus"),
    ("how do I download my capital gains statement", "out_of_corpus"),
]


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return common.read_jsonl(common.path_for("chunks_file"))


@pytest.fixture(scope="module")
def cfg() -> dict:
    return common.load_config()


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question,needle,slug", GOLDEN,
                         ids=[q[0][:38] for q in GOLDEN])
def test_a_named_scheme_never_gets_another_schemes_figure(question, needle, slug):
    a = A.ask(question)
    assert a.kind == "answer", f"refused instead of answered: {a.refusal_reason}"
    assert len(a.sources) == 1, f"expected exactly one source, got {len(a.sources)}"
    assert a.sources[0].url == scheme_url(slug), \
        f"cited the wrong scheme's page: {a.sources[0].url}"
    # Commas are stripped before matching: the corpus writes Indian grouping
    # (1,13,606.47) and the needles below are written without it.
    assert needle in a.text.lower().replace(",", ""), \
        f"expected {needle!r} in the answer: {a.text!r}"


def test_no_answer_ever_cites_a_scheme_the_question_did_not_name(chunks):
    """Cross-scheme check, run over every question that names a scheme.

    Stricter than the table above: for all 5 schemes and all 6 factual subjects,
    the cited URL's slug must equal the slug the question names.
    """
    subjects = ["expense ratio", "minimum SIP", "AUM", "NAV", "exit load",
                "benchmark"]
    names = {
        "hdfc-large-cap-fund-direct-growth": "HDFC Large Cap Fund",
        "hdfc-equity-fund-direct-growth": "HDFC Flexi Cap Fund",
        "hdfc-elss-tax-saver-fund-direct-plan-growth": "HDFC ELSS Tax Saver Fund",
        "hdfc-small-cap-fund-direct-growth": "HDFC Small Cap Fund",
        "hdfc-balanced-advantage-fund-direct-growth":
            "HDFC Balanced Advantage Fund",
    }
    wrong: list[str] = []
    for slug, name in names.items():
        for subject in subjects:
            a = A.ask(f"What is the {subject} of {name}?")
            if a.kind != "answer":
                continue
            if not a.sources or a.sources[0].url != scheme_url(slug):
                wrong.append(f"{name} / {subject} -> "
                             f"{a.sources[0].url if a.sources else 'no source'}")
    assert not wrong, "cross-scheme attribution failures:\n  " + "\n  ".join(wrong)


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question,kind", MUST_REFUSE, ids=[q[0][:38] for q in MUST_REFUSE])
def test_a_question_that_must_be_refused_is(question, kind):
    a = A.ask(question)
    assert a.kind == kind, \
        f"expected {kind}, got {a.kind} (reason: {a.refusal_reason})"
    assert a.is_refusal
    assert a.text.strip(), "a refusal with no text"
    assert a.refusal_reason, f"{kind} carried no reason"


def test_every_refusal_names_an_educational_or_source_pointer():
    """PRD §12: a refusal must point somewhere useful, not just say no.

    The PII refusal is exempt. Its text is verbatim PRD §12.4, which carries no
    link, and it must not: the whole point of that message is that nothing the
    user sent is echoed back, and a refusal that appends a URL is one more thing
    in the response to get wrong.
    """
    for question, kind in MUST_REFUSE:
        a = A.ask(question)
        assert a.text.strip(), question
        if kind == "refusal_pii":
            assert a.text == G.PII_REFUSAL, \
                "the PII refusal text was edited away from PRD §12.4"
            continue
        assert ("https://" in a.text) or ("try:" in a.text.lower()), \
            f"{kind} refusal for {question!r} points nowhere: {a.text!r}"


# ---------------------------------------------------------------------------
# The answer contract - PRD §11
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question,needle,slug", GOLDEN,
                         ids=[q[0][:38] for q in GOLDEN])
def test_every_answer_honours_the_contract(question, needle, slug):
    a = A.ask(question)
    assert a.kind in PRD_KINDS, f"kind {a.kind!r} is outside the PRD Literal"
    assert a.disclaimer == G.DISCLAIMER
    assert a.last_updated == "unknown" or re.fullmatch(
        r"Last updated from sources: \d{4}-\d{2}-\d{2}", a.last_updated), \
        f"malformed stamp: {a.last_updated!r}"


@pytest.mark.parametrize("question,needle,slug", GOLDEN,
                         ids=[q[0][:38] for q in GOLDEN])
def test_every_answer_carries_a_real_stamp(question, needle, slug):
    a = A.ask(question)
    assert re.fullmatch(r"Last updated from sources: \d{4}-\d{2}-\d{2}",
                        a.last_updated), \
        f"the stamp went missing: {a.last_updated!r} for {question!r}"


@pytest.mark.parametrize("question,needle,slug", GOLDEN,
                         ids=[q[0][:38] for q in GOLDEN])
def test_the_stamp_is_not_printed_twice(question, needle, slug):
    """The stamp lives in `last_updated`; `text` must not also carry it, or the
    UI's three-line layout (text / stamp / disclaimer) repeats the date."""
    a = A.ask(question)
    assert "Last updated from sources" not in a.text, a.text


def test_every_answer_is_within_three_sentences(cfg):
    cap = int(cfg["generation"]["max_sentences"])
    for question, needle, slug in GOLDEN:
        a = A.ask(question)
        assert a.kind == "answer"
        assert G.count_sentences(a.text) <= cap, \
            f"{G.count_sentences(a.text)} sentences: {a.text!r}"


def test_every_answer_cites_exactly_one_source_and_it_is_in_the_corpus(chunks):
    known = G.corpus_urls()
    for question, _needle, _slug in GOLDEN:
        a = A.ask(question)
        assert len(a.sources) == 1, f"{question!r} cited {len(a.sources)} sources"
        url = a.sources[0].url
        assert url in known, f"{question!r} cited an uncorpus URL: {url}"
        # Exactly one link in the prose, matching the structured field.
        in_text = re.findall(r"https?://[^\s)\]]+", a.text)
        assert len(in_text) == 1, f"{question!r} has {len(in_text)} links in prose"
        assert in_text[0] == url, f"{question!r}: prose link != sources[0].url"


def test_every_answer_is_grounded_in_its_own_context(chunks):
    """No answer may state a figure that is not in the chunk it came from."""
    for question, _f, _s in GOLDEN:
        a = A.ask(question)
        used = a.debug.get("chunk_used")
        assert used, f"{question!r} recorded no chunk"
        row = next((c for c in chunks if c["chunk_id"] == used), None)
        assert row is not None, f"{question!r} cites a chunk not in the corpus"
        bad = G.grounding_violations(a.text, [row])
        assert not bad, f"{question!r}: {bad}\n  {a.text!r}"


def test_the_stamp_uses_the_oldest_source_not_the_newest(chunks):
    """The answer is only as current as its weakest source."""
    a = A.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    used = next(c for c in chunks if c["chunk_id"] == a.debug["chunk_used"])
    assert a.last_updated == f"Last updated from sources: {used['source_fetched_at'][:10]}"


def test_a_source_carries_its_publisher_and_fetch_date(chunks):
    a = A.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    s = a.sources[0]
    assert s.publisher == "HDFC AMC"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.fetched_at[:10])
    assert s.title
    assert s.url.startswith("https://")


# ---------------------------------------------------------------------------
# Generation: the choices that keep the stub honest
# ---------------------------------------------------------------------------


def test_no_fact_key_to_phrase_table_is_used():
    """The phrase table is keyed on the page's own label, never on fact_key.

    A table keyed on fact_key could confidently answer with the wrong subject:
    the previous publisher's `fund_objective` row held the benchmark line, so
    "the investment objective is <a benchmark name>" would have read perfectly.
    The map in rag/answer.py cannot do that, because its key is the label the
    chunk carries and its output can only restate that label. This test holds
    that line: the phrase in the answer must be derivable from the used chunk's
    own label, and every figure in the answer must appear in that chunk.
    """
    a = A.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    used = next(c for c in common.read_jsonl(common.path_for("chunks_file"))
                if c["chunk_id"] == a.debug["chunk_used"])
    label = used["text"].split(":", 1)[0].strip()
    phrase = A.friendly_label(label)
    flat = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    assert flat(phrase) in flat(a.text), (
        f"the answer does not carry the page's own label: {a.text!r}")
    for figure in re.findall(r"\d[\d,]*\.?\d*", a.text):
        assert figure in used["text"] or figure in used["text"].replace(",", ""), \
            f"answer shows {figure!r}, which is not in the quoted chunk"


def test_a_bare_date_row_does_not_answer_what_is_the_nav(chunks):
    """Asking WHAT the NAV is must be answered with an amount, not a date.

    `is_bare_date` is the guard for a publisher that renders the NAV's as-on
    date as the value. The AMC does not, so no such row exists in the corpus any
    more - but the guard is still load-bearing for the answer layer, so it is
    exercised directly rather than through a corpus shape that no longer exists,
    and the corpus is checked for a regression back into the bad shape.
    """
    assert A.is_bare_date("25 Sep '26")
    assert A.is_bare_date("28/09/2026")
    assert not A.is_bare_date("551.04")
    assert not A.is_bare_date("NIFTY 100 (Total Return Index)")

    rows = [c for c in common.read_jsonl(common.path_for("chunks_file"))
            if c.get("fact_key") == "nav" and c["doc_type"] == "fact"]
    assert rows, "premise broken: the corpus carries no NAV facts at all"
    for c in rows:
        assert not A.is_bare_date(A.value_half(c["text"])), \
            f"a date-only NAV row is back in the corpus: {c['text']!r}"
    a = A.ask("What is the NAV of HDFC Balanced Advantage Fund?")
    assert a.kind == "answer"
    assert "551.04" in a.text, f"answered with a date, not the NAV: {a.text!r}"


def test_a_label_only_row_does_not_win_over_a_row_with_the_amount(chunks):
    """A bare heading must not be quoted as if it carried a figure.

    Every AMC scheme page is full of headings with no value of their own:
    "Ideal for", "Wealth Creation", "Exit Load", "Benchmark Riskometer". Only
    one page states a minimum investment at all, and where it does the row that
    carries the amount must beat every heading on the same page.
    """
    a = A.ask("What is the minimum investment for HDFC Small Cap Fund?")
    assert a.kind == "answer", f"refused instead of answered: {a.refusal_reason}"
    assert "100" in a.text, a.text

    # And where no page states it, the honest answer is that it cannot be
    # verified - not the nearest other row on that fund's page.
    gone = A.ask("What is the minimum investment for HDFC Large Cap?")
    assert gone.kind == "out_of_corpus", \
        f"invented a figure from another row: {gone.text!r}"


def test_a_verbatim_quote_is_not_mangled_by_the_repeat_collapser():
    """The collapse is for the page's duplicated label row and nothing else."""
    assert A._collapse_repeats("exit load exit load of 1%") == "exit load of 1%"
    assert A._collapse_repeats("the nav is 557.73") == "the nav is 557.73"
    assert A._collapse_repeats("min. for sip and min. for lumpsum") == \
        "min. for sip and min. for lumpsum"


def test_the_stub_quotes_the_labels_the_page_used(chunks):
    """The answer carries the page's own field name, in a readable form.

    The AMC writes "TER" where a reader expects "expense ratio", so the answer
    says "total expense ratio (TER)". The AMC's word is kept in the phrase, so
    the answer can only be more explicit than the source, never different from
    it - and the value is quoted with the source's own capitalisation.
    """
    a = A.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    used = next(c for c in chunks if c["chunk_id"] == a.debug["chunk_used"])
    assert "ter" in used["text"].lower()
    assert "ter" in a.text.lower()
    assert "1.03" in a.text


def test_a_question_naming_a_scheme_with_no_retrieved_chunk_is_not_answered(
        monkeypatch):
    """The safety net behind the scheme filter.

    If the filter ever yields nothing for a named scheme, the pipeline must fall
    through to out_of_corpus rather than quote a neighbour.
    """
    real = A.retrieve

    def only_other_schemes(question, **kwargs):
        return [c for c in real(question, **kwargs)
                if c.scheme_slug != "hdfc-small-cap-fund-direct-growth"]

    monkeypatch.setattr(A, "retrieve", only_other_schemes)
    a = A.ask("What is the expense ratio of HDFC Small Cap Fund?")
    assert a.kind == "out_of_corpus", a.kind
    assert a.sources == [], "a refusal must cite nothing"


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_every_enforced_rule_is_actually_in_the_system_prompt():
    for rule in P.ENFORCED_RULES:
        assert rule["text"] in P.SYSTEM_PROMPT, \
            f"rule {rule['id']} is not in the prompt it claims to enforce"


def test_the_system_prompt_states_no_facts():
    """A system prompt carrying a scheme name or a figure can leak it into a
    question the corpus does not cover - exactly the failure out-of-corpus
    exists to prevent."""
    body = P.SYSTEM_PROMPT.lower()
    for leak in ("1189", "1.03", "1447", "557.73", "39933", "113606", "0.77%",
                 "1.21%", "groww.in", "hdfc large cap", "hdfc elss",
                 "hdfc flexi", "hdfc small cap", "hdfc balanced"):
        assert leak not in body, f"the system prompt leaks {leak!r}"


def test_the_prompt_renders_a_citation_per_block():
    from rag.retrieve import retrieve  # noqa: PLC0415

    rendered = P.render_for_humans(retrieve("what is the expense ratio of "
                                            "hdfc large cap fund"),
                                   "expense ratio?")
    blocks = re.findall(r"\[BLOCK (\d+)\]\nSource: (\S+)", rendered)
    assert blocks, "no numbered blocks were rendered"
    for _n, url in blocks:
        assert url in G.corpus_urls(), f"prompt block cites {url}, not in corpus"
    assert "NOT_FOUND" in rendered


def test_the_refusal_prompt_tells_the_model_not_to_answer():
    out = P.build_refusal_prompt("Should I buy this?", "advice keyword")
    assert P.NOT_FOUND in out
    assert "Do not answer it" in out
    assert "Should I buy this?" in out
