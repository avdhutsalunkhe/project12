"""
Ingestion pipeline — loads catalog.json, generates embeddings, and
stores vectors + metadata in ChromaDB.

Usage:
    python -m pipeline.ingest                      # full ingestion
    python -m pipeline.ingest --reset              # reset collection first
    python -m pipeline.ingest --catalog data/catalog.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

# ── Default catalog path ─────────────────────────────────────────
DEFAULT_CATALOG = os.path.join(PROJECT_ROOT, "data", "catalog.json")


def _load_catalog(catalog_path: str) -> List[Dict[str, Any]]:
    """Load and validate the assessment catalog from JSON."""
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assessments = data.get("assessments", [])

    # Filter out errored entries
    valid = [
        a for a in assessments
        if not a.get("scrape_error") and a.get("name")
    ]

    logger.info(
        "Loaded catalog: %d total, %d valid assessments",
        len(assessments),
        len(valid),
    )
    return valid


def _build_document_text(assessment: Dict[str, Any]) -> str:
    """
    Build a rich text document for embedding from an assessment record.

    Combines name, description, test type, job levels, and other
    metadata into a single searchable string.
    """
    parts: List[str] = []

    # Assessment name (high weight through repetition)
    name = assessment.get("name", "")
    if name:
        parts.append(name)

    # Description
    desc = assessment.get("description", "")
    if desc:
        parts.append(desc)

    # Test type
    test_type = assessment.get("test_type", "")
    if test_type:
        parts.append(f"Test type: {test_type}")

    # Job levels
    levels = assessment.get("job_levels", [])
    if levels:
        parts.append(f"Job levels: {', '.join(levels)}")

    # Languages
    languages = assessment.get("languages", [])
    if languages:
        parts.append(f"Languages: {', '.join(languages[:5])}")

    return " | ".join(parts)


def _build_metadata(assessment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a ChromaDB-compatible metadata dict from an assessment.

    ChromaDB metadata values must be str, int, float, or bool.
    Lists are serialised as comma-separated strings.
    """
    meta: Dict[str, Any] = {}

    # Core fields
    meta["name"] = assessment.get("name", "")
    meta["url"] = assessment.get("url", "")
    meta["catalog_type"] = assessment.get("catalog_type", "")

    # Test type
    if assessment.get("test_type_code"):
        meta["test_type_code"] = assessment["test_type_code"]
    if assessment.get("test_type"):
        meta["test_type"] = assessment["test_type"]

    # Duration (as integer for filtering)
    duration = assessment.get("duration_minutes")
    if duration is not None:
        meta["duration_minutes"] = int(duration)

    # Remote testing
    remote = assessment.get("remote_testing")
    if remote is not None:
        meta["remote_testing"] = bool(remote)

    # Lists serialised as comma-separated strings
    levels = assessment.get("job_levels", [])
    if levels:
        meta["job_levels"] = ", ".join(levels)

    languages = assessment.get("languages", [])
    if languages:
        meta["languages"] = ", ".join(languages)

    # Fact sheet URLs (first one only, to stay within metadata size limits)
    fact_sheets = assessment.get("fact_sheet_urls", [])
    if fact_sheets:
        meta["fact_sheet_url"] = fact_sheets[0]

    return meta


def _build_document_id(assessment: Dict[str, Any], index: int) -> str:
    """
    Generate a stable, unique document ID for each assessment.

    Uses the URL slug when available, otherwise falls back to index.
    """
    url = assessment.get("url", "")
    if url:
        # Extract slug from URL: .../view/some-slug/ -> some-slug
        parts = url.rstrip("/").split("/")
        if parts:
            return f"shl_{parts[-1]}"

    return f"shl_{index:04d}"


def ingest_catalog(
    catalog_path: str = DEFAULT_CATALOG,
    reset: bool = False,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
      1. Load catalog.json
      2. Build document texts and metadata
      3. Generate embeddings via sentence-transformers
      4. Store in ChromaDB

    Args:
        catalog_path: Path to the catalog JSON file.
        reset: If True, delete and recreate the collection before ingesting.
        batch_size: Batch size for embedding generation.

    Returns:
        Summary dict with counts and timing.
    """
    start_time = time.time()

    # Step 1: Load catalog
    assessments = _load_catalog(catalog_path)
    if not assessments:
        return {"status": "error", "message": "No valid assessments found"}

    # Step 2: Build documents, metadata, and IDs
    logger.info("Building document corpus...")
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, assessment in enumerate(assessments):
        doc_id = _build_document_id(assessment, idx)
        doc_text = _build_document_text(assessment)
        meta = _build_metadata(assessment)

        ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append(meta)

    logger.info("Built %d documents for embedding", len(documents))

    # Step 3: Generate embeddings
    logger.info("Generating embeddings (model=%s)...", embedding_service.model_name)
    embed_start = time.time()

    embeddings = embedding_service.encode(
        documents,
        batch_size=batch_size,
        show_progress=True,
        normalize=True,
    )

    embed_time = time.time() - embed_start
    logger.info(
        "Embeddings generated: %d vectors (dim=%d) in %.1fs",
        len(embeddings),
        len(embeddings[0]) if embeddings else 0,
        embed_time,
    )

    # Step 4: Store in ChromaDB
    if reset:
        logger.info("Resetting vector store collection...")
        vector_store.reset()

    logger.info("Ingesting into ChromaDB...")
    ingest_start = time.time()

    count = vector_store.upsert_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        batch_size=100,
    )

    ingest_time = time.time() - ingest_start
    total_time = time.time() - start_time

    summary = {
        "status": "success",
        "total_assessments": len(assessments),
        "documents_indexed": count,
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "embedding_model": embedding_service.model_name,
        "embedding_time_seconds": round(embed_time, 2),
        "ingestion_time_seconds": round(ingest_time, 2),
        "total_time_seconds": round(total_time, 2),
        "collection_info": vector_store.collection_info(),
    }

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("  Documents indexed: %d", count)
    logger.info("  Embedding dim:     %d", summary["embedding_dim"])
    logger.info("  Embedding time:    %.1fs", embed_time)
    logger.info("  Ingestion time:    %.1fs", ingest_time)
    logger.info("  Total time:        %.1fs", total_time)
    logger.info("=" * 60)

    return summary


def main() -> None:
    """CLI entry-point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest SHL catalog into ChromaDB vector store.",
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG,
        help=f"Path to catalog.json (default: {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before ingesting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    try:
        summary = ingest_catalog(
            catalog_path=args.catalog,
            reset=args.reset,
            batch_size=args.batch_size,
        )
        print(f"\nResult: {json.dumps(summary, indent=2)}")

    except KeyboardInterrupt:
        logger.info("Ingestion interrupted")
        sys.exit(1)

    except Exception:
        logger.exception("Fatal error during ingestion")
        sys.exit(1)


if __name__ == "__main__":
    main()
