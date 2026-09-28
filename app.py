"""Phase 7 - the Streamlit UI.

Design rule: the retrieval/answering work already happened in `rag.answer`.
This file renders a result and does not make policy decisions. Anything that
looks like a judgement - which scheme, is this in corpus, should this be
refused - lives behind `ask()` and is tested there.

Everything below the `st.*` calls is a pure function so `tests/test_ui.py`
can drive the shaping logic without a running Streamlit server.

PRD FR-8: header note, welcome line, 3 chips, one citation per answer, the
"Last updated from sources:" stamp, a "Why this answer?" expander, a sidebar
with scope + links + stats + rebuild, no file upload, session-only history.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import streamlit as st

import common
from rag.answer import Answer, ask
from rag import guardrails as G

# The `.env` is read first, so a developer who set LLM_PROVIDER there is
# honoured; the setdefault is only the floor for a machine that has no `.env`
# at all. The demo must never *require* a paid LLM - the extractive stub is
# what every Phase 7 test asserts - but dropping a key into `.env` is the
# documented way to switch providers, and that only works if this does not
# shadow it.
common.load_dotenv_if_present()
os.environ.setdefault("LLM_PROVIDER", "stub")

CFG = common.load_config()
UI = CFG["ui"]
PY = Path(sys.executable)


def _warm_work() -> None:
    from embed.index import load_model
    from store import chroma_store as cs
    try:  # best-effort: a model/store hiccup must not take the page down
        load_model()
        cs.get_collection(create=False)
    except Exception: pass


@st.cache_resource(show_spinner=False)
def _warm() -> None:
    """Heavy singletons load on a daemon thread, so the first render is never
    blocked on a cold model load; the first question pays for it if it arrives
    before warmup lands."""
    threading.Thread(target=_warm_work, daemon=True).start()

# ---------------------------------------------------------------------------
# Pure helpers - no Streamlit state, fully unit-testable.
# ---------------------------------------------------------------------------


def body_text(answer: Answer) -> str:
    """The answer paragraph, with any inline `Source: <url>` removed.

    The stub appends the raw URL to `text` so the CLI has something to print.
    The UI renders the citation itself as a labelled link, so leaving it in
    would show the same single source twice - once as a bare URL inside the
    sentence, once as the citation line. PRD FR-8.4 is "the one source link".
    """
    return common.strip_inline_source(answer.text)


def citation(answer: Answer) -> dict[str, str] | None:
    """The single citation for an answer, or None if it refused.

    PRD 11 allows exactly one source per answer. A refusal is a policy
    response, not a sourced fact, so it gets no citation link - its text
    already names a next step.
    """
    if answer.is_refusal or not answer.sources:
        return None
    src = answer.sources[0]
    return {"url": src.url, "label": src.title or src.scheme_name or src.url}


def source_line(answer: Answer) -> str:
    """The one-link line rendered under the answer text."""
    c = citation(answer)
    return f"Source: [{c['label']}]({c['url']})" if c else ""


def stamp_line(answer: Answer) -> str:
    """The `Last updated from sources:` line, or '' when we have no date.

    The stamp lives only in `Answer.last_updated` and never in `Answer.text`,
    so the three-line layout cannot print the date twice.
    """
    return answer.last_updated or ""


def why_rows(debug: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for the "Why this answer?" table.

    `retrieved` is ordered as the answer layer saw it, so the table is a real
    trace rather than a re-sorted summary: the row the answer used is the one
    marked in the first column.
    """
    rows = []
    used = debug.get("chunk_used")
    for i, r in enumerate(debug.get("retrieved") or [], start=1):
        text = " ".join(str(r.get("text") or "").split())
        rows.append(
            {
                "#": i,
                "used": "✓" if r.get("chunk_id") == used else "",
                "chunk_id": r.get("chunk_id", ""),
                "scheme": r.get("scheme_slug", ""),
                "final_score": round(float(r.get("final_score") or 0.0), 5),
                "dense": round(float(r.get("dense_score") or 0.0), 4),
                "bm25": round(float(r.get("bm25_score") or 0.0), 3),
                "rrf": round(float(r.get("rrf_score") or 0.0), 5),
                "coverage": round(float(r.get("coverage") or 0.0), 3),
                "matched": ",".join(r.get("matched_terms") or []) or "-",
                "excerpt": text[:160] + ("…" if len(text) > 160 else ""),
            }
        )
    return rows


def corpus_stats() -> dict[str, Any]:
    """Sidebar numbers. Every read is guarded: a missing build artefact shows
    '—' rather than crashing the page, because a demo that white-screens on a
    partial build is worse than one that admits it is incomplete."""
    out: dict[str, Any] = {"schemes": 0, "pages": "—", "chunks": "—",
                           "embeddings": "—", "embed_hit_pct": None, "last_ingest": "—"}

    def grab(key: str, *path: str) -> Any:
        p = common.path_for(key)
        if not p.exists():
            return None
        try:
            cur: Any = common.read_json(p)
            for step in path:
                cur = (cur or {}).get(step)
            return cur
        except Exception:
            return None

    out["schemes"] = len(common.load_schemes())
    out["pages"] = grab("chunk_stats_file", "totals", "documents_kept") or "—"
    out["chunks"] = grab("chunk_stats_file", "totals", "chunks") or "—"

    # computed + cached, not computed. A warm rebuild is a 100% cache hit by
    # design (that is the idempotency guarantee), so reading `computed` alone
    # printed "Embeddings: —" on every single Rebuild press.
    computed, cached = grab("embed_stats_file", "computed"), grab("embed_stats_file", "cached")
    if isinstance(computed, int) and isinstance(cached, int):
        out["embeddings"] = computed + cached
        out["embed_hit_pct"] = grab("embed_stats_file", "cache_hit_pct")
    log = common.path_for("ingestion_log")
    if log.exists():
        try:
            rows = common.read_jsonl(log)
            if rows:
                out["last_ingest"] = str(rows[-1].get("fetched_at", ""))[:19]
        except Exception:
            pass
    return out


def run_rebuild() -> tuple[bool, str]:
    """Re-run the build in a child process.

    A subprocess, not an in-process call: Stage 1-4 own module-level state
    (the embedding model, the Chroma client) and re-importing them in the
    Streamlit session would leave the running page serving a store that no
    longer matches the one it loaded scores from.
    """
    root = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    steps = ["ingest.chunk", "embed.index", "store.chroma_store rebuild"]
    log: list[str] = []
    for stage in steps:
        argv = stage.split()
        p = subprocess.run([sys.executable, "-m", *argv], cwd=root, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = [ln for ln in (p.stdout or "").splitlines() if ln.strip()][-3:]
        log.append(f"$ python -m {stage}\n" + "\n".join(tail))
        if p.returncode != 0:
            return False, (p.stderr or p.stdout or "unknown error")[-1500:] + "\n\n" + "\n\n".join(log)
    return True, "\n\n".join(log)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_answer(answer: Answer) -> None:
    """The four-line answer block: text, one citation, stamp, disclaimer."""
    st.markdown(body_text(answer))
    src = citation(answer)
    if src:
        st.markdown(f":blue[**Source:** [{src['label']}]({src['url']})]")
    stamp = stamp_line(answer)
    if stamp:
        st.caption(stamp)
    st.caption(answer.disclaimer)

    debug = answer.debug or {}
    with st.expander("Why this answer?"):
        st.caption(f"kind = `{answer.kind}`  ·  provider = `{debug.get('provider', '—')}`  ·  "
                   f"named scheme = `{debug.get('named_scheme') or '—'}`  ·  guardrails = "
                   + ("pass" if debug.get("final_valid") else "fail"))
        rows = why_rows(debug)
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True,
                         column_config={"excerpt": st.column_config.TextColumn("excerpt", width="large")})
        elif debug.get("draft_reasons"):
            st.warning("Refused before retrieval: " + ", ".join(debug["draft_reasons"]))
        else:
            st.info("Nothing was retrieved for this question.")


def sidebar() -> None:
    st.sidebar.title("Scope")
    st.sidebar.markdown("**HDFC Asset Management Company** — 5 schemes, all **Direct Growth** plans.\n\n"
                        "Facts only. No recommendations, no returns, no comparisons.")

    st.sidebar.subheader("Sources")
    for s in common.load_schemes():
        st.sidebar.markdown(f"- [{s['short_name']}]({s['url']})  \n  <sub>{s['publisher']}</sub>")

    st.sidebar.subheader("Corpus")
    cs = corpus_stats()
    hit = f"  _(cache hit {cs['embed_hit_pct']}%)_" if cs.get("embed_hit_pct") is not None else ""
    st.sidebar.markdown(f"- Schemes: **{cs['schemes']}**\n- Pages indexed: **{cs['pages']}**\n- "
                        f"Chunks: **{cs['chunks']}**\n- Embeddings: **{cs['embeddings']}**{hit}\n- Last ingest: `{cs['last_ingest']}`")

    st.sidebar.subheader("Maintenance")
    if st.sidebar.button("Rebuild index", width="stretch"):
        with st.spinner("Rebuilding chunk → embed → store…"):
            ok, msg = run_rebuild()
        (st.sidebar.success if ok else st.sidebar.error)(msg)
        if ok:
            st.cache_clear()


def main() -> None:
    st.set_page_config(page_title=UI["title"], page_icon="📊", layout="wide")
    _warm()  # heavy singletons are loaded on page load, not on the first question
    sidebar()

    # FR-8.7 - disclaimer visible without scrolling, at the top and pinned at
    # the bottom. Both come from config ui.note / guardrails.DISCLAIMER, so the
    # literal string is written once, not twice.
    st.caption(UI["note"])

    st.title(UI["title"]); st.markdown(f"*{UI['welcome_line']}*")

    st.session_state.setdefault("messages", []); st.session_state.setdefault("pending", None)

    # FR-8.3 - exactly 3 chips, shown only while the transcript is empty.
    if not st.session_state["messages"]:
        cols = st.columns(len(UI["example_questions"]))
        for col, q in zip(cols, UI["example_questions"]):
            if col.button(q, width="stretch", key=f"chip::{q}"):
                st.session_state["pending"] = q

    question = (st.chat_input("Ask a factual question about a scheme (no personal data)")
                or st.session_state.pop("pending", None))

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_answer(m["answer"])
            else:
                st.markdown(m["text"])

    if question:
        st.session_state["messages"].append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and checking…"):
                answer = ask(question)
            render_answer(answer)
        st.session_state["messages"].append({"role": "assistant", "answer": answer})

    st.caption(UI["note"])
    st.caption(G.DISCLAIMER)


if __name__ == "__main__":
    main()
