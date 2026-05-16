"""
Retrieval testing script — validates the semantic search pipeline.

Runs a battery of test queries against the ChromaDB vector store and
prints ranked results with scores, metadata, and timing.

Usage:
    python -m pipeline.test_retrieval
    python -m pipeline.test_retrieval --top-k 5
    python -m pipeline.test_retrieval --query "Python developer assessment"
    python -m pipeline.test_retrieval --query "Java" --test-type K --max-duration 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# High-level reusable retrieval API
from pipeline.retrieval import (
    search,
    search_with_scores,
    format_results,
    store_info,
    is_ready,
)
# Low-level service (used for collection_info)
from app.services.retrieval_service import retrieval_service

logger = logging.getLogger(__name__)


# ── Test Queries ─────────────────────────────────────────────────
DEFAULT_TEST_QUERIES = [
    # Technical skills
    {
        "query": "Java developer assessment",
        "description": "Technical: Java programming",
    },
    {
        "query": "Python programming skills test",
        "description": "Technical: Python",
    },
    {
        "query": "data science and machine learning",
        "description": "Technical: Data Science / ML",
    },
    {
        "query": "cloud computing AWS DevOps",
        "description": "Technical: Cloud / DevOps",
    },
    # Job roles
    {
        "query": "entry level customer service representative",
        "description": "Role: Customer Service (entry)",
    },
    {
        "query": "mid-level manager leadership assessment",
        "description": "Role: Manager / Leadership",
    },
    {
        "query": "call center agent phone skills",
        "description": "Role: Call Center Agent",
    },
    # Assessment types
    {
        "query": "personality questionnaire behavioral assessment",
        "description": "Type: Personality / Behavioral",
    },
    {
        "query": "cognitive ability numerical reasoning aptitude",
        "description": "Type: Cognitive / Aptitude",
    },
    {
        "query": "situational judgement test for graduates",
        "description": "Type: SJT / Graduates",
    },
    # Industry-specific
    {
        "query": "bank operations accounting financial",
        "description": "Industry: Banking / Finance",
    },
    {
        "query": "sales transformation report",
        "description": "Industry: Sales",
    },
    # Filtered queries
    {
        "query": "software development",
        "description": "Filtered: Knowledge & Skills type only",
        "filters": {"test_type_code": "K"},
    },
    {
        "query": "leadership management skills",
        "description": "Filtered: Max 30 min duration",
        "filters": {"max_duration": 30},
    },
]


def run_single_query(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a single retrieval query and return results with timing.

    The ``filters`` dict may contain any combination of:
      - ``test_type_code`` (str)  — e.g. 'K', 'A'
      - ``max_duration``  (int)  — maximum duration in minutes
      - ``job_level``     (str)  — job level substring
      - ``min_score``     (float)— minimum similarity score
    """
    start = time.time()

    # Unpack supported filter keys explicitly — avoids the kwargs.update() bug
    # where unknown keys caused a TypeError in retrieval_service.search().
    filters = filters or {}
    results = search(
        query,
        top_k=top_k,
        test_type_code=filters.get("test_type_code"),
        job_level=filters.get("job_level"),
        max_duration=filters.get("max_duration"),
        min_score=filters.get("min_score", 0.0),
    )
    elapsed = time.time() - start

    return {
        "query": query,
        "top_k": top_k,
        "filters": filters or None,
        "num_results": len(results),
        "elapsed_ms": round(elapsed * 1000, 1),
        "results": results,
    }


def print_query_results(
    result: Dict[str, Any],
    description: str = "",
    verbose: bool = False,
) -> None:
    """Pretty-print a query's results."""
    print(f"\n{'=' * 70}")
    if description:
        print(f"  [{description}]")
    print(f"  Query:   \"{result['query']}\"")
    if result.get("filters"):
        print(f"  Filters: {result['filters']}")
    print(f"  Results: {result['num_results']} | Time: {result['elapsed_ms']}ms")
    print(f"{'=' * 70}")

    for i, r in enumerate(result["results"], 1):
        meta = r.get("metadata", {})
        name = meta.get("name", "Unknown")
        score = r.get("score", 0)
        test_type = meta.get("test_type", "N/A")
        duration = meta.get("duration_minutes", "N/A")
        levels = meta.get("job_levels", "N/A")
        url = meta.get("url", "")

        print(f"\n  {i}. {name}")
        print(f"     Score: {score:.4f} | Type: {test_type} | Duration: {duration} min")
        print(f"     Levels: {levels}")
        if url:
            print(f"     URL: {url}")

        if verbose and r.get("document"):
            doc_preview = r["document"][:200] + "..." if len(r["document"]) > 200 else r["document"]
            print(f"     Doc: {doc_preview}")


def run_test_suite(
    top_k: int = 5,
    verbose: bool = False,
    queries: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Run the full test suite and return aggregate statistics.
    """
    if queries is None:
        queries = DEFAULT_TEST_QUERIES

    # Check index health
    info = store_info()
    print(f"\nVector Store: {info['name']} ({info['count']} documents)")
    print(f"Persist Dir:  {info['persist_directory']}")

    if info["count"] == 0:
        print("\nERROR: Vector store is empty! Run ingestion first:")
        print("  python -m pipeline.ingest")
        return {"status": "error", "message": "Empty vector store"}

    total_queries = len(queries)
    total_results = 0
    total_time_ms = 0.0
    all_scores: List[float] = []

    print(f"\nRunning {total_queries} test queries (top_k={top_k})...")

    for test in queries:
        query = test["query"]
        description = test.get("description", "")
        filters = test.get("filters")

        result = run_single_query(query, top_k=top_k, filters=filters)
        print_query_results(result, description=description, verbose=verbose)

        total_results += result["num_results"]
        total_time_ms += result["elapsed_ms"]

        for r in result["results"]:
            all_scores.append(r.get("score", 0))

    # Aggregate stats
    avg_time = total_time_ms / total_queries if total_queries else 0
    avg_results = total_results / total_queries if total_queries else 0
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    max_score = max(all_scores) if all_scores else 0
    min_score = min(all_scores) if all_scores else 0

    summary = {
        "status": "success",
        "total_queries": total_queries,
        "total_results": total_results,
        "avg_results_per_query": round(avg_results, 1),
        "avg_latency_ms": round(avg_time, 1),
        "total_latency_ms": round(total_time_ms, 1),
        "score_stats": {
            "avg": round(avg_score, 4),
            "max": round(max_score, 4),
            "min": round(min_score, 4),
        },
    }

    print(f"\n{'=' * 70}")
    print("  TEST SUITE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Queries run:        {total_queries}")
    print(f"  Total results:      {total_results}")
    print(f"  Avg results/query:  {avg_results:.1f}")
    print(f"  Avg latency:        {avg_time:.1f}ms")
    print(f"  Score range:        {min_score:.4f} - {max_score:.4f}")
    print(f"  Avg score:          {avg_score:.4f}")
    print(f"{'=' * 70}")

    return summary


def main() -> None:
    """CLI entry-point for retrieval testing."""
    parser = argparse.ArgumentParser(
        description="Test the semantic retrieval pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results per query (default: 5).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single custom query instead of the test suite.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default=None,
        help="Filter results by test type code (e.g., K, A, P).",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=None,
        help="Filter results by max duration in minutes.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show document text in results.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    if args.query:
        # Single custom query
        filters = {}
        if args.test_type:
            filters["test_type_code"] = args.test_type
        if args.max_duration:
            filters["max_duration"] = args.max_duration

        result = run_single_query(
            args.query,
            top_k=args.top_k,
            filters=filters or None,
        )
        print_query_results(result, description="Custom Query", verbose=args.verbose)
    else:
        # Full test suite
        run_test_suite(top_k=args.top_k, verbose=args.verbose)


if __name__ == "__main__":
    main()
