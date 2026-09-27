"""ingest/ - STAGE 1 (Loading) and STAGE 2 (Chunking) for the RAG pipeline.

Modules
-------
fetch   Stage 1a - HTTP fetch, raw HTML cache, source snapshot, ingestion log
clean   Stage 1b - boilerplate removal, structure-preserving node stream, PII scrub
chunk   Stage 2  - ADR-001 structure-first hybrid chunking

Each module is independently runnable:

    python -m ingest.fetch --all
    python -m ingest.clean
    python -m ingest.chunk
"""

__all__ = ["fetch", "clean", "chunk"]
