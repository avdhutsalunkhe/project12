"""
pipeline/embedder.py — Reusable embedding helpers for the ingestion pipeline.

Provides high-level functions for:
  - Embedding text documents in batches
  - Single-query encoding for retrieval
  - Embedding dimension introspection
  - Warm-up / health-check utilities

All functions delegate to the shared EmbeddingService singleton so the
sentence-transformer model is loaded only once per process.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.services.embedding_service import embedding_service, EmbeddingService  # noqa: E402

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────────

def embed_documents(
    texts: List[str],
    *,
    batch_size: int = 64,
    show_progress: bool = True,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Embed a list of raw text documents.

    Args:
        texts: List of strings to embed (e.g., assembled document texts).
        batch_size: Number of texts per encoding batch (tune for VRAM/RAM).
        show_progress: Show a tqdm progress bar.
        normalize: L2-normalise the output vectors (required for cosine sim).

    Returns:
        List of embedding vectors (each a list of 384 floats for MiniLM-L6).
    """
    if not texts:
        return []

    start = time.time()
    logger.info(
        "Embedding %d documents (batch_size=%d, model=%s) ...",
        len(texts),
        batch_size,
        embedding_service.model_name,
    )

    vectors = embedding_service.encode(
        texts,
        batch_size=batch_size,
        show_progress=show_progress,
        normalize=normalize,
    )

    elapsed = time.time() - start
    logger.info(
        "Embedded %d docs → dim=%d in %.2fs (%.1f docs/s)",
        len(vectors),
        len(vectors[0]) if vectors else 0,
        elapsed,
        len(vectors) / elapsed if elapsed > 0 else 0,
    )
    return vectors


def embed_query(query: str, *, normalize: bool = True) -> List[float]:
    """
    Embed a single natural-language query for semantic retrieval.

    Args:
        query: Free-text search query.
        normalize: L2-normalise (must match how documents were embedded).

    Returns:
        A single embedding vector (list of floats).
    """
    return embedding_service.encode_single(query, normalize=normalize)


def embed_batch(
    texts: List[str],
    *,
    batch_size: int = 64,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Thin alias around embed_documents without the progress bar.
    Handy for programmatic use inside other pipeline steps.
    """
    return embedding_service.encode(
        texts,
        batch_size=batch_size,
        show_progress=False,
        normalize=normalize,
    )


def get_embedding_dim() -> int:
    """Return the embedding dimension for the active model (e.g. 384 for MiniLM-L6-v2)."""
    return embedding_service.dimension


def get_model_name() -> str:
    """Return the sentence-transformer model identifier."""
    return embedding_service.model_name


def warm_up(test_text: str = "warm up") -> Dict[str, Any]:
    """
    Force the embedding model to load and run one forward-pass.

    Call this at application startup so the first real request is fast.

    Returns:
        Dict with model_name, dimension, and latency_ms.
    """
    start = time.time()
    vec = embed_query(test_text)
    latency = (time.time() - start) * 1000

    info = {
        "model_name": embedding_service.model_name,
        "dimension": len(vec),
        "latency_ms": round(latency, 1),
    }
    logger.info("Warm-up complete: %s", info)
    return info


def health_check() -> Dict[str, Any]:
    """
    Validate the embedding service is operational.

    Returns:
        Dict with status ('ok' | 'error'), model_name, dimension, and message.
    """
    try:
        info = warm_up("health check probe")
        return {"status": "ok", **info}
    except Exception as exc:  # noqa: BLE001
        logger.error("Embedding health check failed: %s", exc)
        return {
            "status": "error",
            "model_name": embedding_service.model_name,
            "dimension": 0,
            "message": str(exc),
        }


def make_embedding_service(model_name: str) -> EmbeddingService:
    """
    Factory: create a *new* EmbeddingService with a custom model.

    Useful for experiments; does NOT replace the module-level singleton.
    """
    return EmbeddingService(model_name=model_name)
