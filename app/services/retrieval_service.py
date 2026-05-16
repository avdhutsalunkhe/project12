"""
Retrieval service — high-level semantic search API.

Combines the EmbeddingService and VectorStore into a single interface
for the rest of the application to use. Handles:
  - Query embedding generation
  - Top-k similarity search
  - Metadata-filtered retrieval
  - Result enrichment with original catalog data
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Semantic retrieval over the SHL assessment catalog.

    Encodes user queries with all-MiniLM-L6-v2, then searches
    the ChromaDB vector store for the most similar assessments.
    """

    def __init__(self) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    @property
    def index_count(self) -> int:
        """Number of documents currently in the vector index."""
        return self._vector_store.count()

    @property
    def is_ready(self) -> bool:
        """Check if the vector store has been populated."""
        try:
            return self._vector_store.count() > 0
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────
    #  Core Retrieval
    # ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        test_type_code: Optional[str] = None,
        job_level: Optional[str] = None,
        max_duration: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for assessments matching a natural-language query.

        Args:
            query: Free-text search query (e.g., "Java developer assessment").
            top_k: Maximum number of results to return.
            test_type_code: Filter by test type code (A/B/C/D/E/K/P/S).
            job_level: Filter by job level substring match.
            max_duration: Maximum duration in minutes.
            min_score: Minimum similarity score threshold (0.0 - 1.0).

        Returns:
            List of result dicts sorted by similarity score (descending).
            Each dict contains: id, document, metadata, distance, score.
        """
        # Step 1: Encode the query
        query_embedding = self._embedding_service.encode_single(query)

        # Step 2: Build ChromaDB where filter
        where_filter = self._build_where_filter(
            test_type_code=test_type_code,
            max_duration=max_duration,
        )

        # Step 3: Query the vector store (fetch extra for post-filtering)
        fetch_k = top_k * 3 if (job_level or min_score > 0) else top_k
        raw_results = self._vector_store.query_texts(
            query_embedding,
            top_k=fetch_k,
            where=where_filter,
        )

        # Step 4: Post-filter results
        filtered = []
        for result in raw_results:
            # Score threshold
            if result["score"] < min_score:
                continue

            # Job level filter (not natively supported by ChromaDB on list fields)
            if job_level:
                meta_levels = result.get("metadata", {}).get("job_levels", "")
                if isinstance(meta_levels, str):
                    if job_level.lower() not in meta_levels.lower():
                        continue
                elif isinstance(meta_levels, list):
                    if not any(job_level.lower() in l.lower() for l in meta_levels):
                        continue

            filtered.append(result)

            if len(filtered) >= top_k:
                break

        logger.info(
            "Semantic search: query='%s' -> %d results (top_k=%d, filters=%s)",
            query[:80],
            len(filtered),
            top_k,
            where_filter,
        )

        return filtered

    def search_similar(
        self,
        assessment_id: str,
        *,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find assessments similar to a given assessment (by ID).

        Useful for "more like this" recommendations.
        """
        # Get the document's text to re-encode
        doc = self._vector_store.get_by_id(assessment_id)
        if not doc:
            logger.warning("Assessment not found: %s", assessment_id)
            return []

        return self.search(doc["document"], top_k=top_k + 1)[1:]  # Exclude self

    # ─────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_where_filter(
        test_type_code: Optional[str] = None,
        max_duration: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a ChromaDB `where` filter dict from optional parameters.

        ChromaDB supports: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin,
        and logical operators $and, $or.
        """
        conditions: List[Dict[str, Any]] = []

        if test_type_code:
            conditions.append({
                "test_type_code": {"$eq": test_type_code.upper()}
            })

        if max_duration is not None:
            conditions.append({
                "duration_minutes": {"$lte": max_duration}
            })

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def collection_info(self) -> Dict[str, Any]:
        """Return info about the underlying vector store."""
        return self._vector_store.collection_info()


# ── Module-level singleton ────────────────────────────────────
retrieval_service = RetrievalService()
