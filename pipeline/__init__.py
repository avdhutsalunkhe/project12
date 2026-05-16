"""
Pipeline package.

Exposes the most commonly used pipeline functions at the package level
so callers can do:

    from pipeline import ingest_catalog, search, embed_documents
"""

from pipeline.embedder import (
    embed_documents,
    embed_query,
    embed_batch,
    get_embedding_dim,
    get_model_name,
    warm_up,
    health_check as embedding_health_check,
)
from pipeline.retrieval import (
    search,
    search_with_scores,
    search_by_id,
    batch_search,
    raw_vector_search,
    filter_by_type,
    filter_by_duration,
    filter_by_min_score,
    format_results,
    store_info,
    is_ready,
)
from pipeline.ingest import ingest_catalog

__all__ = [
    # Embedding
    "embed_documents",
    "embed_query",
    "embed_batch",
    "get_embedding_dim",
    "get_model_name",
    "warm_up",
    "embedding_health_check",
    # Retrieval
    "search",
    "search_with_scores",
    "search_by_id",
    "batch_search",
    "raw_vector_search",
    "filter_by_type",
    "filter_by_duration",
    "filter_by_min_score",
    "format_results",
    "store_info",
    "is_ready",
    # Ingestion
    "ingest_catalog",
]
