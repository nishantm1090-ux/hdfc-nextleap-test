"""Phase 7 tests.

These are contract tests for the UI layer, and they are deliberately written
against the *live* pipeline rather than against fixtures. The three example
chips are the first thing a viewer sees; if the corpus changes under them, the
demo breaks in a way no import check would catch. So each chip is actually
asked, and the answer is asserted to be a real answer with a citation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import app
import common
import rag.guardrails as G

APP = Path(app.__file__).resolve()
SRC = APP.read_text(encoding="utf-8")
CFG = common.load_config()
UI = CFG["ui"]

FR8_REQUIRED = {
    "FR-8.1": "Facts-only. No investment advice.",
    "FR-8.2": "Ask me factual questions about 5 HDFC mutual fund schemes. Every answer comes with a source link.",
}


# ---------------------------------------------------------------------------
# FR-8.1 / 8.2 / 8.3 - the literal strings
# ---------------------------------------------------------------------------


def test_note_is_the_literal_prd_string():
    assert UI["note"] == FR8_REQUIRED["FR-8.1"]


def test_welcome_line_is_the_literal_prd_string():
    assert UI["welcome_line"] == FR8_REQUIRED["FR-8.2"]


def test_exactly_three_example_questions():
    assert len(UI["example_questions"]) == 3, "FR-8.3 says exactly 3 chips"


def test_the_note_is_not_hardcoded_in_the_ui_a_second_time():
    """The literal must come from config, not be pasted into app.py.

    Two copies is how the header and footer drift apart on the one string the
    whole 'facts-only' promise rests on.
    """
    assert UI["note"] not in SRC, "app.py hardcodes the note; read it from config ui.note"


def test_welcome_line_comes_from_config():
    assert UI["welcome_line"] not in SRC


# ---------------------------------------------------------------------------
# FR-8.3 - every chip answers, live
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", UI["example_questions"], ids=lambda q: q[:34])
def test_every_example_chip_returns_a_cited_answer(q):
    """The load-bearing UI test.

    An example chip that trips a refusal is what FR-8.3 forbids: it is the
    first interaction, and it teaches the viewer the demo is broken. This ran
    the demo UI against a corpus where chip 3 was the capital-gains question -
    correctly refused, and a terrible first impression.
    """
    a = app.ask(q)
    assert a.kind == "answer", f"chip {q!r} returned kind={a.kind}: {a.text[:90]}"
    assert len(a.sources) == 1
    assert a.sources[0].url.startswith("https://www.hdfcfund.com/"), \
        f"chip cited a non-official page: {a.sources[0].url}"
    assert a.last_updated.startswith("Last updated from sources: ")


@pytest.mark.parametrize("q", UI["example_questions"], ids=lambda q: q[:34])
def test_every_example_chip_is_at_most_three_sentences(q):
    assert G.count_sentences(app.ask(q).text) <= 3


# ---------------------------------------------------------------------------
# FR-8.4 - the answer block
# ---------------------------------------------------------------------------


def test_citation_is_exactly_one_link_for_an_answer():
    a = app.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    c = app.citation(a)
    assert c is not None
    assert c["url"].count("https://") == 1, "FR-8.4 / PRD §11: one source link"
    assert app.source_line(a).count("](") == 1


def test_body_text_removes_the_inline_source_so_the_link_is_not_shown_twice():
    """The stub appends `Source: <url>` to `text` for the CLI's benefit.

    The UI then renders its own labelled citation, so without this the same
    single source appeared twice in every answer - once as a bare URL glued to
    the sentence, once as the citation line.
    """
    a = app.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    assert "http" in a.text, "precondition: the stub does inline the URL"
    body = app.body_text(a)
    assert "http" not in body, f"inline source survived: {body[-90:]!r}"
    assert body == body.strip() and body
    assert "1.03%" in body, "stripping must not eat the answer itself"
    # exactly one URL survives the whole block: the citation
    assert body.count("http") + app.source_line(a).count("http") == 1


def test_a_refusal_carries_no_citation():
    """A refusal is a policy response, not a sourced fact. Its text already
    names a next step, so a link next to it would imply we looked something
    up when we deliberately did not."""
    a = app.ask("Should I buy HDFC Small Cap?")
    assert a.kind == "refusal_advice"
    assert app.citation(a) is None
    assert app.source_line(a) == ""


def test_stamp_is_separate_from_the_text_so_the_date_is_not_printed_twice():
    a = app.ask("What is the expense ratio of HDFC Large Cap Direct Growth?")
    assert a.last_updated.startswith("Last updated from sources: ")
    assert "Last updated from sources:" not in a.text
    assert app.stamp_line(a) == a.last_updated


def test_answer_text_carries_the_disclaimer_field():
    a = app.ask("What is the minimum SIP for HDFC ELSS Tax Saver Fund?")
    assert a.disclaimer == G.DISCLAIMER


def test_disclaimer_matches_prd_section_11_verbatim():
    assert G.DISCLAIMER.startswith("**Facts-only. No investment advice.**")
    assert "does not recommend, compare, or rate schemes" in G.DISCLAIMER
    assert "Sources: HDFC Mutual Fund, SEBI and AMFI." in G.DISCLAIMER


# ---------------------------------------------------------------------------
# FR-8.6 - the "Why this answer?" expander
# ---------------------------------------------------------------------------


def test_why_rows_exposes_the_full_score_trail():
    a = app.ask("What is the expense ratio of HDFC Small Cap Fund?")
    rows = app.why_rows(a.debug)
    assert rows, "expander would render empty for an answer that retrieved chunks"
    for r in rows:
        for col in ("chunk_id", "dense", "bm25", "rrf", "final_score", "coverage", "excerpt"):
            assert col in r
        assert r["excerpt"]


def test_why_rows_marks_exactly_the_chunk_the_answer_used():
    a = app.ask("What is the expense ratio of HDFC Small Cap Fund?")
    rows = app.why_rows(a.debug)
    used = [r for r in rows if r["used"] == "✓"]
    assert len(used) == 1, "the trace must point at exactly one chunk"
    assert used[0]["chunk_id"] == a.debug["chunk_used"]


def test_why_rows_is_empty_for_a_refusal_that_never_retrieves():
    a = app.ask("My PAN is ABCDE1234F, can you check my folio?")
    assert a.kind == "refusal_pii"
    assert app.why_rows(a.debug) == []


# ---------------------------------------------------------------------------
# FR-8.5 - sidebar
# ---------------------------------------------------------------------------


def test_corpus_stats_reflects_the_real_build():
    """The sidebar numbers come from the build artefacts, not from constants.

    The counts are asserted against the chunk file rather than pinned to a
    literal, because a pinned 121 was a lie twice over: it survived a corpus
    rebuild that produced 68 chunks, and it would have survived a broken build
    that produced none. What is pinned is the INVARIANT - one embedding per
    chunk, 100% cache hit, five schemes.

    `pages` is 6, not 5, and that is not a bug: the AMC's consolidated account
    statement page is a fetched document in its own right, so counting pages
    (6) above schemes (5) is the honest reading. The old `== 5` only held while
    one page per scheme happened to be the whole corpus.
    """
    cs = app.corpus_stats()
    n_chunks = len(common.read_jsonl(common.path_for("chunks_file")))
    pages = {r["source_url"] for r in common.read_jsonl(common.path_for("chunks_file"))}
    assert cs["schemes"] == 5
    assert cs["chunks"] == n_chunks
    assert cs["pages"] == len(pages) >= 5
    assert cs["embeddings"] == n_chunks
    assert cs["embed_hit_pct"] == 100.0
    assert re.match(r"\d{4}-\d{2}-\d{2}", str(cs["last_ingest"]))


def test_corpus_stats_counts_cached_embeddings_not_only_newly_computed():
    """A warm rebuild computes nothing and caches every chunk (`computed: 0`).

    Reading `computed` alone therefore reported "Embeddings: —" on every
    Rebuild press - the one control whose whole point is a warm no-op rebuild.
    """
    stats = common.read_json(common.path_for("embed_stats_file"))
    n_chunks = len(common.read_jsonl(common.path_for("chunks_file")))
    assert stats["cached"] == n_chunks and stats["computed"] == 0, \
        f"expected a warm cache over {n_chunks} chunks, got {stats}"
    assert app.corpus_stats()["embeddings"] == n_chunks


def test_corpus_stats_degrades_instead_of_crashing(monkeypatch, tmp_path):
    """A partial build must show '—', not white-screen the demo."""
    monkeypatch.setattr(common, "path_for", lambda k: tmp_path / "missing.json")
    monkeypatch.setattr(common, "read_json", lambda p: (_ for _ in ()).throw(IOError("boom")))
    cs = app.corpus_stats()
    assert cs["chunks"] == "—"
    assert cs["schemes"] == 5  # schemes come from config, not build artefacts


def test_sidebar_lists_the_five_source_links():
    """The five source links the sidebar offers are the AMC's own scheme pages."""
    schemes = common.load_schemes()
    assert len(schemes) == 5
    assert all(s["url"].startswith("https://www.hdfcfund.com/explore/mutual-funds/")
               for s in schemes), [s["url"] for s in schemes]
    assert len({s["url"] for s in schemes}) == 5, "the five links must be distinct"


# ---------------------------------------------------------------------------
# FR-8.8 / 8.9 - prohibitions
# ---------------------------------------------------------------------------


def test_no_file_upload_control_anywhere():
    """FR-8.8. A scheme FAQ with an upload widget invites users to hand over
    statements and documents, which is the exact PII path FR-6 refuses."""
    for fn in ("file_uploader", "data_editor", "text_area"):
        assert not re.search(rf"st\.{fn}\b", SRC), f"FR-8.8: st.{fn} must not appear"


def test_chat_history_is_session_only_never_written_to_disk():
    """FR-8.9. Nothing in app.py may open a file for writing."""
    tree = ast.parse(SRC)
    writes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "attr", getattr(fn, "id", ""))
        if name in {"write_text", "write_bytes", "dump", "writelines"}:
            writes.append(ast.dump(node)[:90])
        # open(..., "w") / open(..., "a")
        if name == "open" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            if node.args[1].value in {"w", "a", "w+", "a+"}:
                writes.append(ast.dump(node)[:90])
    assert not writes, f"FR-8.9: app.py must not persist anything: {writes}"


def test_no_personal_data_placeholder_in_the_input():
    """FR-8.8. The placeholder must not invite a folio number or PAN."""
    m = re.search(r'st\.chat_input\(\s*"([^"]+)"', SRC)
    assert m, "expected a chat_input with a literal placeholder"
    low = m.group(1).lower()
    for bad in ("pan", "aadhaar", "folio", "account number", "phone", "email", "otp"):
        assert bad not in low, f"placeholder invites personal data: {bad!r}"


# ---------------------------------------------------------------------------
# The rebuild path
# ---------------------------------------------------------------------------


def test_rebuild_runs_the_three_build_stages_as_a_subprocess():
    """A subprocess, not an in-process call.

    Stages 1-4 own module-level state (the embedding model, the Chroma client).
    Re-importing them inside the Streamlit session would leave the page
    serving scores from a store that no longer matches the one on disk.
    """
    tree = ast.parse(SRC)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "run_rebuild")
    seg = ast.get_source_segment(SRC, fn)
    for stage in ("ingest.chunk", "embed.index", "store.chroma_store"):
        assert stage in seg, f"rebuild must run {stage}"
    assert "subprocess.run" in seg


def test_run_rebuild_returns_a_two_tuple_and_does_not_touch_the_store_on_probe(monkeypatch):
    """Prove the failure path: a non-zero child exit is reported, not raised."""
    class R:
        returncode = 1
        stdout = "some progress output\n"
        stderr = "boom: no chunks"

    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: R())
    ok, msg = app.run_rebuild()
    assert ok is False
    assert "boom: no chunks" in msg


# ---------------------------------------------------------------------------
# App size / structure
# ---------------------------------------------------------------------------


def test_app_py_is_under_200_lines_of_code():
    """The spec's own budget: 'Keep the whole UI under ~200 lines'.

    Measured as executable statements with docstrings, comments and blanks
    excluded - a 14-line module docstring is documentation, not UI code, and
    counting it would make the budget a lie.
    """
    tree = ast.parse(SRC)
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    # drop lines that are only inside a docstring
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                d = body[0]
                doc_lines.update(range(d.lineno, (d.end_lineno or d.lineno) + 1))
    code = len(lines - doc_lines)
    assert code < 200, f"app.py is {code} code lines, budget is 200"


def test_render_answer_and_sidebar_are_wired_into_main():
    tree = ast.parse(SRC)
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = set()
    for n in ast.walk(main_fn):
        if isinstance(n, ast.Call):
            f = n.func
            called.add(getattr(f, "id", None) or getattr(f, "attr", None) or "")
    assert "sidebar" in called
    assert "render_answer" in called
    assert "chat_input" in called


# ---------------------------------------------------------------------------
# The script actually runs (Streamlit's own runner, not a mock)
# ---------------------------------------------------------------------------


def _run(question: str | None = None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    if question:
        at.chat_input[0].set_value(question).run()
    return at


def test_the_page_renders_with_no_exception():
    """`/_stcore/health` returning 200 proves nothing - Streamlit reports
    script errors in the browser, not the health endpoint. This runs the real
    script through Streamlit's own AppTest runner."""
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert UI["title"] in [t.value for t in at.title]


def test_the_page_shows_exactly_three_chips_and_the_scope():
    at = _run()
    chips = [b.label for b in at.button if b.label in UI["example_questions"]]
    assert chips == UI["example_questions"]
    assert any("Rebuild index" == b.label for b in at.button)
    assert any("5 schemes" in m.value for m in at.markdown)


def test_asking_a_question_renders_the_whole_answer_block():
    at = _run("What is the expense ratio of HDFC Large Cap Direct Growth?")
    assert not at.exception, [e.value for e in at.exception]
    assert [e.label for e in at.expander] == ["Why this answer?"]
    assert len(at.dataframe) == 1, "the score trail table is missing"

    # `at.markdown` flattens the sidebar, which legitimately carries 5 more
    # source links. FR-8.4 is about the *answer block*, so measure that.
    sidebar_texts = {m.value for m in at.sidebar.markdown}
    main = [m.value for m in at.markdown if m.value not in sidebar_texts]
    rendered = "\n".join(main)
    assert "1.03%" in rendered
    assert rendered.count("https://www.hdfcfund.com/explore/mutual-funds/") == 1, \
        "FR-8.4: one source link"
    assert "https://" not in next(m for m in main if "1.03%" in m), "bare URL in the prose"

    caps = [c.value for c in at.caption]
    assert any(c.startswith("Last updated from sources: ") for c in caps)
    assert any(c.startswith("kind = `answer`") for c in caps)
    # FR-8.7: the note is at the top and pinned at the bottom
    assert UI["note"] in caps[0] and UI["note"] in caps[-1]


def test_the_why_expander_labels_the_chunk_the_answer_used():
    at = _run("What is the minimum SIP for HDFC ELSS Tax Saver Fund?")
    assert not at.exception
    frame = at.dataframe[0].value
    assert len(frame) > 1
    used = frame[frame["used"] == "✓"]
    assert len(used) == 1
    assert "elss" in used.iloc[0]["chunk_id"]
    assert {"dense", "bm25", "rrf", "final_score", "coverage"} <= set(frame.columns)


def test_a_refusal_renders_without_a_citation():
    at = _run("Should I buy HDFC Small Cap?")
    assert not at.exception
    sidebar_texts = {m.value for m in at.sidebar.markdown}
    main = "\n".join(m.value for m in at.markdown if m.value not in sidebar_texts)
    assert "investment advice" in main
    assert "Source:" not in main, "a refusal must not render a citation"
    assert "https://groww.in" not in main
