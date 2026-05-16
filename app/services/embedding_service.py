"""
Embedding service — generates dense vector embeddings using sentence-transformers.

Model: all-MiniLM-L6-v2 (384-dim, fast, high quality for semantic search).
Singleton pattern with lazy loading to avoid loading the model until needed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# ── Default Configuration ────────────────────────────────────────
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384


class EmbeddingService:
    """
    Generates sentence embeddings using sentence-transformers.

    Lazily loads the model on first use to keep imports fast.
    Thread-safe after initialisation (the model is read-only).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._dimension: int = DEFAULT_EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return

        logger.info("Loading embedding model: %s ...", self._model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        # Update dimension from the actual model
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded: %s (dim=%d)",
            self._model_name,
            self._dimension,
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        *,
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> List[List[float]]:
        """
        Encode one or more texts into dense embeddings.

        Args:
            texts: A single string or list of strings to embed.
            batch_size: Batch size for encoding (controls memory usage).
            show_progress: Show a tqdm progress bar during encoding.
            normalize: L2-normalise the vectors (recommended for cosine similarity).

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        # Convert numpy array to list of lists for serialisation
        return embeddings.tolist()

    def encode_single(self, text: str, normalize: bool = True) -> List[float]:
        """Encode a single text and return a flat embedding vector."""
        return self.encode(text, normalize=normalize)[0]


# ── Module-level singleton ────────────────────────────────────
embedding_service = EmbeddingService()
