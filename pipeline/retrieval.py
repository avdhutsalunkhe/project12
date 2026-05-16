"""
pipeline/retrieval.py — Reusable retrieval functions for semantic search.

This module provides a clean, importable API for semantic retrieval over
the ChromaDB vector store.  It is distinct from the test script
(test_retrieval.py) and is designed to be used by:
  - The FastAPI application (via RetrievalService)
  - Offline notebooks / scripts
  - Unit tests

Functions
---------
search(query, top_k, ...)
    Primary semantic search — returns ranked list of assessments.

search_with_scores(query, top_k, ...)
    Same as search but returns raw similarity scores alongside results.

search_by_id(assessment_id, top_k)
    "More like this" — finds assessments similar to a known one.

batch_search(queries, top_k, ...)
    Run multiple queries in one call; returns a list of result lists.

filter_by_type(results, test_type_code)
    Post-hoc type filter on an already-retrieved result list.

filter_by_duration(results, max_minutes)
    Post-hoc duration filter on a result list.

format_results(results, max_doc_chars)
    Convert raw result dicts into a human-readable string.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.services.retrieval_service import retrieval_service  # noqa: E402
from app.services.vector_store import vector_store  # noqa: E402
from pipeline.embedder import embed_query  # noqa: E402

logger = logging.getLogger(__name__)


# ── Type alias ──────────────────────────────────────────────────────────────
Result = Dict[str, Any]


# ── Core retrieval functions ─────────────────────────────────────────────────

def search(
    query: str,
    *,
    top_k: int = 10,
    test_type_code: Optional[str] = None,
    job_level: Optional[str] = None,
    max_duration: Optional[int] = None,
    min_score: float = 0.0,
) -> List[Result]:
    """
    Semantic similarity search over the SHL assessment catalog.

    Encodes *query* with all-MiniLM-L6-v2 and returns the top-k most
    similar assessments from ChromaDB, optionally filtered.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.
        test_type_code: Filter by SHL test type ('A','B','C','D','E','K','P','S').
        job_level: Filter by job level substring (e.g. 'Manager', 'Entry').
        max_duration: Exclude assessments longer than this many minutes.
        min_score: Drop results with cosine similarity below this threshold.

    Returns:
        List of result dicts (sorted by score descending):
        [
          {
            "id": "shl_some-slug",
            "score": 0.8432,
            "metadata": {
              "name": "...", "url": "...", "test_type": "...",
              "duration_minutes": 30, "job_levels": "...", ...
            },
            "document": "rich text used for embedding",
            "distance": 0.1568,
          },
          ...
        ]
    """
    start = time.time()
    results = retrieval_service.search(
        query,
        top_k=top_k,
        test_type_code=test_type_code,
        job_level=job_level,
        max_duration=max_duration,
        min_score=min_score,
    )
    elapsed_ms = (time.time() - start) * 1000
    logger.debug(
        "search('%s') → %d results in %.1fms",
        query[:60],
        len(results),
        elapsed_ms,
    )
    return results


def search_with_scores(
    query: str,
    *,
    top_k: int = 10,
    test_type_code: Optional[str] = None,
    max_duration: Optional[int] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper that surfaces score + rank alongside each result.

    Returns same structure as search() but each dict also has:
      - "rank": 1-indexed rank position
      - "score_pct": score expressed as percentage (0-100)
    """
    results = search(
        query,
        top_k=top_k,
        test_type_code=test_type_code,
        max_duration=max_duration,
        min_score=min_score,
    )
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        r["score_pct"] = round(r["score"] * 100, 1)
    return results


def search_by_id(
    assessment_id: str,
    *,
    top_k: int = 5,
) -> List[Result]:
    """
    "More like this" — find assessments similar to *assessment_id*.

    Args:
        assessment_id: The ChromaDB document ID (e.g. 'shl_verify-g-plus').
        top_k: How many similar assessments to return.

    Returns:
        List of result dicts (excluding the query document itself).
    """
    return retrieval_service.search_similar(assessment_id, top_k=top_k)


def batch_search(
    queries: List[str],
    *,
    top_k: int = 10,
    test_type_code: Optional[str] = None,
    max_duration: Optional[int] = None,
    min_score: float = 0.0,
) -> List[List[Result]]:
    """
    Run multiple queries in a single call.

    Args:
        queries: List of natural-language search strings.
        top_k: Top-k per query.
        (other args same as search())

    Returns:
        List of result lists — one per input query, in the same order.
    """
    all_results: List[List[Result]] = []
    for q in queries:
        results = search(
            q,
            top_k=top_k,
            test_type_code=test_type_code,
            max_duration=max_duration,
            min_score=min_score,
        )
        all_results.append(results)
    return all_results


def raw_vector_search(
    query: str,
    *,
    top_k: int = 10,
    where: Optional[Dict[str, Any]] = None,
) -> List[Result]:
    """
    Low-level vector search — bypasses RetrievalService to query
    ChromaDB directly with a custom `where` filter dict.

    Useful for advanced filtering scenarios not covered by the high-level
    search() function.

    Args:
        query: Query text (will be embedded).
        top_k: Number of results.
        where: Raw ChromaDB `where` filter dict.

    Returns:
        List of result dicts from VectorStore.query_texts().
    """
    query_vec = embed_query(query)
    return vector_store.query_texts(query_vec, top_k=top_k, where=where)


# ── Post-retrieval filters ────────────────────────────────────────────────────

def filter_by_type(
    results: List[Result],
    test_type_code: str,
) -> List[Result]:
    """Filter an already-retrieved result list by SHL test type code."""
    code = test_type_code.upper()
    return [
        r for r in results
        if r.get("metadata", {}).get("test_type_code", "").upper() == code
    ]


def filter_by_duration(
    results: List[Result],
    max_minutes: int,
) -> List[Result]:
    """Filter an already-retrieved result list by maximum duration."""
    return [
        r for r in results
        if (r.get("metadata", {}).get("duration_minutes") or 999) <= max_minutes
    ]


def filter_by_min_score(
    results: List[Result],
    min_score: float,
) -> List[Result]:
    """Drop results below a minimum similarity score."""
    return [r for r in results if r.get("score", 0) >= min_score]


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_results(
    results: List[Result],
    *,
    max_doc_chars: int = 200,
    show_document: bool = False,
) -> str:
    """
    Format a result list as a human-readable string (useful for CLI / logs).

    Args:
        results: Output from search() or batch_search().
        max_doc_chars: Truncate embedded document text at this length.
        show_document: Whether to include the document text in output.

    Returns:
        Multi-line formatted string.
    """
    if not results:
        return "  (no results)"

    lines: List[str] = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        name = meta.get("name", "Unknown")
        score = r.get("score", 0)
        test_type = meta.get("test_type", "N/A")
        duration = meta.get("duration_minutes", "N/A")
        levels = meta.get("job_levels", "N/A")
        url = meta.get("url", "")

        lines.append(f"  {i:>2}. {name}")
        lines.append(
            f"      Score: {score:.4f} | Type: {test_type} | Duration: {duration} min"
        )
        lines.append(f"      Levels: {levels}")
        if url:
            lines.append(f"      URL:    {url}")
        if show_document and r.get("document"):
            doc = r["document"]
            if len(doc) > max_doc_chars:
                doc = doc[:max_doc_chars] + "…"
            lines.append(f"      Doc:    {doc}")

    return "\n".join(lines)


def store_info() -> Dict[str, Any]:
    """Return metadata about the ChromaDB collection (count, name, path)."""
    return retrieval_service.collection_info()


def is_ready() -> bool:
    """Return True if the vector store has been populated."""
    return retrieval_service.is_ready
