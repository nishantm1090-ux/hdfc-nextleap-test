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


# ---------------------------------------------------------------------------
# The theme actually reaches the page
#
# This is here because the theme silently did NOT reach the page for two
# separate reasons, and neither raised an error, a warning, or a failed test:
#
#   1. `st.html(css_path)` looks like the correct API. It is not - it runs its
#      input through DOMPurify, which strips `<style>`, so the tag never lands
#      in the DOM and the page renders as raw Streamlit defaults. Confirmed in
#      a real browser: zero `<style>` nodes, computed `font-family` still
#      "Source Sans". `st.markdown(..., unsafe_allow_html=True)` applies fully.
#   2. In Streamlit 1.64 the composer textarea CARRIES the `stChatInputTextArea`
#      testid rather than sitting inside it, so a descendant selector matches
#      nothing and quietly loses to Streamlit's own 14px rule.
#
# A regression test that only asserted "the file exists and is non-empty" would
# have passed through both failures. These assert the contract that was
# actually broken: the bytes are delivered by the API that survives the
# sanitiser, and the selectors match the DOM Streamlit really ships.
# ---------------------------------------------------------------------------


def test_the_stylesheet_is_delivered_through_the_api_that_survives_the_sanitiser():
    """Regression: `st.html` strips `<style>`. Assert we are not using it."""
    markup = app.stylesheet_markup()
    assert markup.startswith("<style>") and markup.rstrip().endswith("</style>")

    css = (APP.parent / "assets" / "style.css").read_text(encoding="utf-8")
    assert css.strip(), "assets/style.css is empty - the page would ship unthemed"
    # The real stylesheet's body must be inlined, not a path or a repr.
    assert css.strip()[:200] in markup
    assert "WindowsPath" not in markup and "PosixPath" not in markup

    # And it must actually be injected, through st.markdown + unsafe_allow_html.
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    rendered = [m.value for m in at.markdown if m.value.startswith("<style>")]
    assert rendered, "no <style> block was rendered - the theme is not reaching the page"
    assert "--ground" in rendered[0], "the rendered block is not our stylesheet"

    main_src = SRC.split("def main(")[-1]
    assert "st.markdown(stylesheet_markup(), unsafe_allow_html=True)" in main_src, (
        "main() must inject the sheet via st.markdown + unsafe_allow_html")
    assert "st.html" not in main_src, (
        "main() must not use st.html - DOMPurify strips <style>, so the theme "
        "would silently never reach the page")


def _css_rules() -> list[tuple[str, str]]:
    """`assets/style.css` as (selector-list, declarations) pairs, comments out.

    Comments are stripped first on purpose. Two of the regressions below were
    originally caught by naive substring checks that matched prose *inside a
    comment* and a selector inside a grouped list - both false positives, and
    both would have trained everyone to ignore the assertion.
    """
    css = (APP.parent / "assets" / "style.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)          # drop comments
    css = re.sub(r"@media[^{]*\{", "@media{", css)              # don't count at-rule heads
    return [(sels.strip(), body.strip())
            for sels, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


def _rule_for(testid: str) -> str | None:
    """Declarations of the first rule whose selector list targets `testid` exactly.

    Matches whole selector items, so `[data-testid="stBottom"]` never picks up
    `[data-testid="stBottom"]::before` or `[data-testid="stBottom"] > div`.
    """
    want = f'[data-testid="{testid}"]'
    for sels, body in _css_rules():
        items = [s.strip() for s in sels.split(",")]
        if any(i == want for i in items):
            return body
    return None


def test_the_palette_in_css_and_the_streamlit_theme_are_the_same_palette():
    """Streamlit paints some of its own widgets from `.streamlit/config.toml`,
    and 1.64 does not expose those colours as CSS variables - so the TOML and
    the stylesheet must agree by hand or the page renders half in each.

    A disagreement is what produced a light composer sitting on a dark panel."""
    import tomllib

    root = APP.parent
    css = (root / "assets" / "style.css").read_text(encoding="utf-8")
    with (root / ".streamlit" / "config.toml").open("rb") as fh:
        theme = tomllib.load(fh)["theme"]

    toml_colors = {theme["backgroundColor"].upper(),
                   theme["secondaryBackgroundColor"].upper(),
                   theme["primaryColor"].upper()}

    defs = dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*(#[0-9A-Fa-f]{6})", css))
    declared = {v.upper() for v in defs.values()}
    missing = {c for c in toml_colors if c not in declared}
    assert not missing, (f"config.toml paints {sorted(missing)} but assets/style.css "
                         f"does not declare them - the page will render in two palettes")

    # base must stay light: the stylesheet has no prefers-color-scheme branch,
    # so a dark base would leave Streamlit's own widgets dark.
    assert theme["base"] == "light"
    live = re.sub(r"/\*.*?\*/", "", css, flags=re.S)   # comments are not code
    assert "prefers-color-scheme" not in live, (
        "the stylesheet must not branch on prefers-color-scheme - Streamlit "
        "paints its own widgets from config.toml and would not follow")


def test_the_composer_selectors_match_the_dom_streamlit_actually_ships():
    """The textarea carries `stChatInputTextArea`; it is not wrapped by it.

    Asserted against the file so the selector and the DOM shape cannot drift
    apart silently again."""
    selectors = {s.strip() for sels, _ in _css_rules() for s in sels.split(",")}
    assert '[data-testid="stChatInput"] textarea' in selectors, (
        "no selector matching a textarea inside stChatInput")
    assert '[data-testid="stChatInputTextArea"]' in selectors, (
        "no selector matching the textarea that itself carries the testid")


def test_the_composer_is_not_a_light_slab_on_a_dark_panel():
    """The original complaint, as an assertion.

    `stBottom` is the fixed container Streamlit paints itself. If it is opaque,
    the composer reads as a light rectangle floating on the page instead of a
    card that belongs to it."""
    bottom = _rule_for("stBottom")
    assert bottom is not None, "assets/style.css no longer styles stBottom at all"
    assert re.search(r"background:\s*transparent", bottom), (
        f"stBottom must be transparent, otherwise the composer is a slab. Got: {bottom!r}")

    fade = _rule_for("stBottom")  # sanity: the bare selector is not the ::before rule
    assert fade is not None
    selectors = {s.strip() for sels, _ in _css_rules() for s in sels.split(",")}
    assert '[data-testid="stBottom"]::before' in selectors, (
        "stBottom needs the ground fade that ties the composer to the page")
    ground = re.findall(r"(--ground)\s*:\s*(#[0-9A-Fa-f]{6})", (APP.parent / "assets" / "style.css").read_text(encoding="utf-8"))
    assert ground, "--ground is not declared, so the fade cannot resolve"


def test_the_spinner_label_is_not_squeezed_into_one_letter_per_line():
    """The reported bug: "Retrieving and checking…" rendered vertically.

    `st.spinner("…")` was styled as `[data-testid="stSpinner"] > div` on the
    assumption that child was the spinner circle. In Streamlit 1.64 it is the
    flex ROW that contains the label, so pinning it to 18x18 crushed the label
    container to 15px wide and the sentence wrapped one character per line -
    measured at 419px tall for a single line of text.

    Two things must hold, and the second is the one that actually broke:
      1. the icon is sized via `stSpinnerIcon`
      2. neither `stSpinner` nor its row is given a fixed width
    """
    root = _rule_for("stSpinner")
    assert root is not None, "assets/style.css no longer styles stSpinner at all"
    assert not re.search(r"width:\s*\d", root), (
        f"stSpinner must not be given a fixed width - it holds the label. Got: {root!r}")

    rules = dict()
    for sels, body in _css_rules():
        for s in [x.strip() for x in sels.split(",")]:
            rules.setdefault(s, []).append(body)

    row_bodies = rules.get('[data-testid="stSpinner"] > div', [])
    assert row_bodies, "the spinner flex row is unstyled - the icon will not sit with the label"
    for body in row_bodies:
        assert not re.search(r"width:\s*\d", body), (
            f"the spinner's flex row must not be given a fixed width. Got: {body!r}")

    icon = rules.get('[data-testid="stSpinnerIcon"]', [])
    assert icon, ("size the icon via [data-testid=\"stSpinnerIcon\"], not via "
                  "`stSpinner > div` - the latter is the row that holds the label")

    label = rules.get('[data-testid="stSpinner"] p', [])
    assert label and any("nowrap" in b for b in label), (
        "the spinner label must be white-space: nowrap so it cannot reflow")


def test_markdown_body_text_is_not_left_in_streamlits_default_font():
    """Streamlit pins `font-family: "Source Sans"` on every markdown container.

    It is a class selector on the element itself, so it beat the font
    inherited from `stAppViewContainer` and every paragraph, list item, link
    and caption rendered in Source Sans while headings and buttons used the
    system stack. Measured before the fix: 12 of 17 <p> across 23 containers.
    """
    bodies = dict()
    for sels, body in _css_rules():
        for s in [x.strip() for x in sels.split(",")]:
            bodies.setdefault(s, []).append(body)

    md = bodies.get('[data-testid="stMarkdownContainer"]', [])
    assert md, ("assets/style.css must style stMarkdownContainer - Streamlit's "
                "emotion class pins Source Sans on it and nothing else overrides it")
    assert any("font-family" in b for b in md), (
        f"stMarkdownContainer must set font-family. Got: {md!r}")


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
