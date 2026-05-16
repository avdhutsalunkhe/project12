"""
app/services/__init__.py

Exposes all service singletons at the package level.
"""

from app.services.embedding_service import embedding_service, EmbeddingService
from app.services.vector_store import vector_store, VectorStore
from app.services.retrieval_service import retrieval_service, RetrievalService
from app.services.comparison_engine import comparison_engine, ComparisonEngine
from app.services.refusal_engine import refusal_engine, RefusalEngine

__all__ = [
    "embedding_service",
    "EmbeddingService",
    "vector_store",
    "VectorStore",
    "retrieval_service",
    "RetrievalService",
    "comparison_engine",
    "ComparisonEngine",
    "refusal_engine",
    "RefusalEngine",
]
