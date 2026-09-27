"""
ingest/fetch.py - STAGE 1a: LOADING (network side)

Downloads every URL listed in config/sources.yaml into a raw HTML cache, records a
SHA-256 snapshot so source drift is detectable, and appends to an ingestion log.

Design rules (docs/architecture.md §4 Stage 1a, PRD §7 FR-1.1/1.2/1.6):
  * cache raw HTML to data/raw/{doc_id}.html; re-use if < 24h old unless --force
  * descriptive User-Agent, 3 retries, exponential backoff, 2s politeness delay
  * NEVER raise on a single page failure - log it, mark it, carry on
  * flag pages whose visible text is suspiciously short (JS-shell detection, PRD R1)

Run:  python -m ingest.fetch --all
      python -m ingest.fetch --all --force
      python -m ingest.fetch --doc-id hdfc-large-cap-fund-direct-growth
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

import common
from common import (
    LOG,
    append_jsonl,
    load_fetchable_supplementary,
    load_sources,
    load_schemes,
    read_json,
    utc_now_iso,
    write_json,
)

# Below this many visible characters a page is almost certainly a JS shell rather
# than the content we asked for (PRD R1). Loudly reported, never silently ignored.
THIN_PAGE_CHARS = 2000

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)


@dataclass
class FetchResult:
    """Outcome of one page fetch."""

    doc_id: str
    url: str
    ok: bool
    status: int | None = None
    raw_path: Path | None = None
    sha256: str = ""
    n_bytes: int = 0
    fetched_at: str = ""
    from_cache: bool = False
    error: str | None = None
    looks_thin: bool = False
    visible_chars: int = 0
    publisher: str = ""
    page_type: str = ""

    def to_log(self) -> dict[str, Any]:
        """Log-safe projection. Never contains page body content."""
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "publisher": self.publisher,
            "page_type": self.page_type,
            "ok": self.ok,
            "status": self.status,
            "bytes": self.n_bytes,
            "sha256": self.sha256,
            "from_cache": self.from_cache,
            "visible_chars": self.visible_chars,
            "looks_thin": self.looks_thin,
            "error": self.error,
            "fetched_at": self.fetched_at,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _visible_chars(html: str) -> int:
    """Crude visible-text length, used only as a JS-shell signal."""
    stripped = _SCRIPT_RE.sub(" ", html)
    stripped = _TAG_RE.sub(" ", stripped)
    return len(re.sub(r"\s+", " ", stripped).strip())


def raw_path_for(doc_id: str) -> Path:
    return common.path_for("raw_dir") / f"{doc_id}.html"


def _is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    age_h = (common.utc_now().timestamp() - path.stat().st_mtime) / 3600.0
    return age_h < max_age_hours


def _all_targets() -> list[dict[str, Any]]:
    """Every URL that may enter the corpus, flattened and normalised.

    This is the corpus gate in code: nothing is fetched that is not in
    config/sources.yaml (principle P8, PRD §4.2).
    """
    targets: list[dict[str, Any]] = []
    for s in load_schemes():
        targets.append(
            {
                "doc_id": s["scheme_slug"],
                "url": s["url"],
                "publisher": s.get("publisher", ""),
                "page_type": s.get("page_type", "scheme_page"),
            }
        )
    for d in load_fetchable_supplementary():
        targets.append(
            {
                "doc_id": d["doc_id"],
                "url": d["url"],
                "publisher": d.get("publisher", ""),
                "page_type": d.get("page_type", "amc_page"),
            }
        )
    return targets


# ---------------------------------------------------------------------------
# Stage 1a
# ---------------------------------------------------------------------------


def fetch_url(url: str, doc_id: str, *, force: bool = False, max_age_hours: int = 24) -> FetchResult:
    """Fetch one page into the raw cache. Never raises."""
    cfg = load_sources()["defaults"]
    dest = raw_path_for(doc_id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # --- cache hit -----------------------------------------------------
    if not force and _is_fresh(dest, max_age_hours):
        html = dest.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
        vis = _visible_chars(html)
        return FetchResult(
            doc_id=doc_id,
            url=url,
            ok=True,
            status=None,
            raw_path=dest,
            sha256=digest,
            n_bytes=len(html.encode("utf-8")),
            fetched_at=common.utc_now_iso(),
            from_cache=True,
            visible_chars=vis,
            looks_thin=vis < THIN_PAGE_CHARS,
        )

    # --- network fetch with retries ------------------------------------
    headers = {
        "User-Agent": cfg.get("user_agent", "MF-FAQ-RAG-Bot/1.0"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }
    retries = int(cfg.get("retries", 3))
    backoff = float(cfg.get("backoff_seconds", 2))
    timeout = float(cfg.get("timeout_seconds", 30))

    last_error: str | None = None
    status: int | None = None
    html: str = ""

    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                resp = client.get(url, headers=headers)
            status = resp.status_code
            resp.raise_for_status()
            html = resp.text
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                delay = backoff * (2 ** (attempt - 1))
                LOG.warning("  fetch %s attempt %d/%d failed (%s) - retrying in %.1fs",
                            doc_id, attempt, retries, last_error, delay)
                time.sleep(delay)

    if last_error is not None or not html:
        return FetchResult(
            doc_id=doc_id,
            url=url,
            ok=False,
            status=status,
            error=last_error or "empty response body",
            fetched_at=common.utc_now_iso(),
        )

    dest.write_text(html, encoding="utf-8")
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    vis = _visible_chars(html)

    return FetchResult(
        doc_id=doc_id,
        url=url,
        ok=True,
        status=status,
        raw_path=dest,
        sha256=digest,
        n_bytes=len(html.encode("utf-8")),
        fetched_at=common.utc_now_iso(),
        from_cache=False,
        visible_chars=vis,
        looks_thin=vis < THIN_PAGE_CHARS,
    )


def fetch_all(*, force: bool = False, only: list[str] | None = None,
              max_age_hours: int = 24) -> list[FetchResult]:
    """Fetch every corpus target, honouring the politeness delay between requests."""
    common.ensure_dirs()
    cfg = load_sources()["defaults"]
    delay = float(cfg.get("politeness_delay_seconds", 2))
    max_age = 24

    targets = _all_targets()
    if only:
        wanted = set(only)
        targets = [t for t in targets if t["doc_id"] in wanted]
        if not targets:
            LOG.error("No targets matched --doc-id %s", sorted(wanted))
            return []

    snapshot = read_json(common.path_for("snapshot_file"), {}) or {}
    results: list[FetchResult] = []

    LOG.info("Stage 1a - fetching %d page(s) (force=%s)", len(targets), force)
    for i, target in enumerate(targets, 1):
        LOG.info("[%d/%d] %s", i, len(targets), target["doc_id"])
        res = fetch_url(target["url"], target["doc_id"], force=force, max_age_hours=max_age_hours)
        res.publisher = target["publisher"]
        res.page_type = target["page_type"]
        results.append(res)

        if res.ok:
            snapshot[target["doc_id"]] = {
                "url": target["url"],
                "publisher": target["publisher"],
                "page_type": target["page_type"],
                "sha256": res.sha256,
                "bytes": res.n_bytes,
                "visible_chars": res.visible_chars,
                "fetched_at": res.fetched_at,
            }
        else:
            LOG.error("    FAILED: %s", res.error)

        append_jsonl(common.path_for("ingestion_log"), res.to_log())
        if i < len(targets):
            time.sleep(delay)          # politeness: never tight-loop the source

    snapshot["_updated_at"] = utc_now_iso()
    write_json(common.path_for("snapshot_file"), snapshot)
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(results: list[FetchResult]) -> int:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    thin = [r for r in ok if r.looks_thin]
    cached = [r for r in ok if r.from_cache]

    print("=" * 78)
    print("  STAGE 1a - LOADING / FETCH REPORT")
    print("=" * 78)
    print(f"  {'doc_id':<46} {'bytes':>9}  {'vis':>7}  flags")
    print("  " + "-" * 74)
    for r in results:
        flags = []
        if not r.ok:
            flags.append("FAILED")
        if r.looks_thin:
            flags.append("THIN/JS-SHELL?")
        if r.from_cache:
            flags.append("cached")
        print(f"  {r.doc_id[:46]:<46} {r.n_bytes:>9,}  {r.visible_chars:>7,}  {','.join(flags) or '-'}")
    print("  " + "-" * 74)
    print(f"  ok={len(ok)}  failed={len(failed)}  thin={len(thin)}  from_cache={len(cached)}")

    if thin:
        print()
        print("  !! WARNING - these pages have very little visible text:")
        for r in thin:
            print(f"     {r.doc_id}  ({r.visible_chars:,} chars)")
        print("     The page is probably JavaScript-rendered (PRD risk R1).")
        print("     Options: (a) switch this doc's url in config/sources.yaml to the")
        print("     HDFC AMC scheme page, or (b) fetch it through a JS renderer.")
        print("     Do NOT lower THIN_PAGE_CHARS and pretend the index is fine.")
    if failed:
        print()
        print("  !! FAILED PAGES (the pipeline continues without them):")
        for r in failed:
            print(f"     {r.doc_id}: {r.error}")

    link_only = [d for d in common.load_supplementary() if not d.get("fetch", False)]
    if link_only:
        print()
        print("  LINK-ONLY sources (never fetched; used as educational links in refusals):")
        for d in link_only:
            print(f"     {d['publisher']:<9} {d['url']}")
            print(f"               reason: {d.get('fetch_blocked_reason', 'n/a')}")
    print("=" * 78)
    return 0 if ok else 1


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1a - fetch the corpus into a raw HTML cache")
    ap.add_argument("--all", action="store_true", help="fetch every URL in config/sources.yaml")
    ap.add_argument("--doc-id", action="append", default=None, help="fetch only this doc_id (repeatable)")
    ap.add_argument("--force", action="store_true", help="ignore the 24h cache and re-download")
    ap.add_argument("--max-age-hours", type=int, default=24, help="cache freshness window")
    args = ap.parse_args(argv)

    common.load_dotenv_if_present()
    common.setup_console()

    if not args.all and not args.doc_id:
        ap.print_help()
        print("\nNothing to do - pass --all or --doc-id <id>.")
        return 2

    # Single code path: fetch_all owns the politeness delay, the raw cache, the
    # ingestion log AND source_snapshot.json. The CLI must not diverge from it,
    # or Stage 1b reads a snapshot that was never written.
    results = fetch_all(force=args.force, only=args.doc_id, max_age_hours=args.max_age_hours)
    return report(results)


if __name__ == "__main__":
    sys.exit(_main())
