"""
Vector store — ChromaDB wrapper for SHL assessment embeddings.

Handles:
  - Collection creation and management
  - Document ingestion with embeddings + metadata
  - Semantic similarity search with top-k retrieval
  - Metadata filtering during retrieval

Uses ChromaDB's persistent storage so the index survives restarts.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# ── Default Configuration ────────────────────────────────────────
_DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "chroma_db",
)
DEFAULT_COLLECTION_NAME = "shl_assessments"


class VectorStore:
    """
    ChromaDB-backed vector store for semantic assessment retrieval.

    Supports persistent storage, metadata filtering, and top-k search.
    """

    def __init__(
        self,
        persist_directory: str = _DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._persist_dir = persist_directory
        self._collection_name = collection_name
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None

    @property
    def persist_directory(self) -> str:
        return self._persist_dir

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _ensure_client(self) -> None:
        """Initialise the ChromaDB client if not already done."""
        if self._client is not None:
            return

        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
        )

        logger.info(
            "ChromaDB client initialised (persist_dir=%s)",
            self._persist_dir,
        )

    def _ensure_collection(self) -> chromadb.Collection:
        """Get or create the assessment collection."""
        self._ensure_client()

        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine similarity
            )
            logger.info(
                "Collection '%s' ready (%d documents)",
                self._collection_name,
                self._collection.count(),
            )

        return self._collection

    # ─────────────────────────────────────────────────────────
    #  Ingestion
    # ─────────────────────────────────────────────────────────

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Add documents with pre-computed embeddings to the collection.

        Args:
            ids: Unique document IDs.
            documents: Raw text documents (stored alongside vectors).
            embeddings: Pre-computed embedding vectors.
            metadatas: Optional metadata dicts for each document.
            batch_size: ChromaDB ingestion batch size.

        Returns:
            Number of documents added.
        """
        collection = self._ensure_collection()
        total = len(ids)

        if not metadatas:
            metadatas = [{}] * total

        # ChromaDB has batch limits; chunk the data
        added = 0
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
            added += end - start
            logger.debug("Ingested batch %d-%d / %d", start, end, total)

        logger.info(
            "Added %d documents to collection '%s'",
            added,
            self._collection_name,
        )
        return added

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert documents (insert or update if ID already exists).
        Same interface as add_documents but uses upsert.
        """
        collection = self._ensure_collection()
        total = len(ids)

        if not metadatas:
            metadatas = [{}] * total

        upserted = 0
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
            upserted += end - start

        logger.info(
            "Upserted %d documents to collection '%s'",
            upserted,
            self._collection_name,
        )
        return upserted

    # ─────────────────────────────────────────────────────────
    #  Retrieval
    # ─────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Semantic similarity search using a query embedding vector.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            where: ChromaDB metadata filter (e.g., {"test_type_code": "K"}).
            where_document: ChromaDB document content filter.
            include: Fields to include in results. Default: all.

        Returns:
            ChromaDB query result dict with keys:
            ids, documents, metadatas, distances, embeddings.
        """
        collection = self._ensure_collection()

        if include is None:
            include = ["documents", "metadatas", "distances"]

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": include,
        }

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = collection.query(**kwargs)
        return results

    def query_texts(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: semantic search returning a clean list of results.

        Returns:
            List of dicts with keys: id, document, metadata, distance, score.
        """
        raw = self.query(
            query_embedding,
            top_k=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: List[Dict[str, Any]] = []
        if not raw["ids"] or not raw["ids"][0]:
            return results

        for i in range(len(raw["ids"][0])):
            distance = raw["distances"][0][i] if raw["distances"] else None
            # ChromaDB cosine distance = 1 - cosine_similarity
            score = 1.0 - distance if distance is not None else 0.0

            results.append({
                "id": raw["ids"][0][i],
                "document": raw["documents"][0][i] if raw["documents"] else "",
                "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                "distance": distance,
                "score": round(score, 4),
            })

        return results

    # ─────────────────────────────────────────────────────────
    #  Management
    # ─────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._ensure_collection().count()

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single document by ID."""
        collection = self._ensure_collection()
        result = collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "document": result["documents"][0] if result["documents"] else "",
            "metadata": result["metadatas"][0] if result["metadatas"] else {},
        }

    def delete_collection(self) -> None:
        """Delete the entire collection (destructive!)."""
        self._ensure_client()
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = None
            logger.info("Collection '%s' deleted", self._collection_name)
        except Exception as exc:
            logger.warning("Could not delete collection: %s", exc)

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.delete_collection()
        self._ensure_collection()
        logger.info("Collection '%s' reset", self._collection_name)

    def collection_info(self) -> Dict[str, Any]:
        """Return basic info about the collection."""
        collection = self._ensure_collection()
        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata,
            "persist_directory": self._persist_dir,
        }


# ── Module-level singleton ────────────────────────────────────
vector_store = VectorStore()
