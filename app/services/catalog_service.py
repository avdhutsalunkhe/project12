"""
Catalog service — loads, indexes, and searches the SHL assessment catalog.

Uses TF-IDF vectorisation for fast keyword-based similarity search
against assessment names and descriptions. Also supports metadata
filtering by test type, job level, duration, and language.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.logging_config import get_logger

logger = get_logger(__name__)

# ── Path to catalog data ─────────────────────────────────────────
_DEFAULT_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "catalog.json",
)


# ═══════════════════════════════════════════════════════════════
#  Lightweight TF-IDF Engine (no heavy deps like scikit-learn)
# ═══════════════════════════════════════════════════════════════

import math
from collections import Counter


def _tokenize(text: str) -> List[str]:
    """Lowercase and split into word tokens, stripping non-alphanum."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFIndex:
    """Minimal in-memory TF-IDF index for assessment search."""

    def __init__(self) -> None:
        self._documents: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        self._tfidf_vectors: List[Dict[str, float]] = []

    def build(self, texts: List[str]) -> None:
        """Index a list of document strings."""
        self._documents = [_tokenize(t) for t in texts]
        n = len(self._documents)

        # Document frequency
        df: Counter = Counter()
        for doc_tokens in self._documents:
            unique = set(doc_tokens)
            for tok in unique:
                df[tok] += 1

        # IDF = log(N / df)
        self._idf = {
            tok: math.log((n + 1) / (freq + 1)) + 1
            for tok, freq in df.items()
        }

        # TF-IDF vectors (normalised)
        self._tfidf_vectors = []
        for doc_tokens in self._documents:
            tf: Counter = Counter(doc_tokens)
            total = len(doc_tokens) or 1
            vec: Dict[str, float] = {}
            for tok, count in tf.items():
                vec[tok] = (count / total) * self._idf.get(tok, 1.0)
            # L2 normalise
            norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
            vec = {k: v / norm for k, v in vec.items()}
            self._tfidf_vectors.append(vec)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Return (doc_index, cosine_score) pairs sorted by relevance.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Build query TF-IDF
        qtf: Counter = Counter(query_tokens)
        total = len(query_tokens)
        qvec: Dict[str, float] = {}
        for tok, count in qtf.items():
            qvec[tok] = (count / total) * self._idf.get(tok, 1.0)
        qnorm = math.sqrt(sum(v ** 2 for v in qvec.values())) or 1.0
        qvec = {k: v / qnorm for k, v in qvec.items()}

        # Cosine similarity against all docs
        scores: List[Tuple[int, float]] = []
        for idx, dvec in enumerate(self._tfidf_vectors):
            dot = sum(qvec.get(tok, 0.0) * dvec.get(tok, 0.0) for tok in qvec)
            if dot > 0:
                scores.append((idx, dot))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════
#  Catalog Service
# ═══════════════════════════════════════════════════════════════

class CatalogService:
    """
    Loads the SHL catalog and provides search/filter capabilities.

    Thread-safe once initialised (read-only after load).
    """

    def __init__(self) -> None:
        self._assessments: List[Dict[str, Any]] = []
        self._index = TFIDFIndex()
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def total(self) -> int:
        return len(self._assessments)

    def load(self, catalog_path: str = _DEFAULT_CATALOG_PATH) -> None:
        """Load catalog from JSON and build the search index."""
        path = Path(catalog_path)
        if not path.exists():
            logger.error("Catalog file not found: %s", catalog_path)
            raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw = data.get("assessments", [])

        # Filter out entries with scrape errors or missing names
        self._assessments = [
            a for a in raw
            if not a.get("scrape_error") and a.get("name")
        ]

        # Build search corpus: combine name + description for each assessment
        corpus = []
        for a in self._assessments:
            parts = [a.get("name", "")]
            if a.get("description"):
                parts.append(a["description"])
            if a.get("test_type"):
                parts.append(a["test_type"])
            for level in a.get("job_levels", []):
                parts.append(level)
            corpus.append(" ".join(parts))

        self._index.build(corpus)
        self._loaded = True

        logger.info(
            "Catalog loaded: %d assessments indexed from %s",
            len(self._assessments),
            catalog_path,
        )

    def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        test_type: Optional[str] = None,
        job_level: Optional[str] = None,
        max_duration: Optional[int] = None,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search assessments by text query with optional metadata filters.

        Returns a list of assessment dicts enriched with `match_score`.
        """
        if not self._loaded:
            logger.info("Catalog not loaded — lazy loading now...")
            self.load()

        # Get raw TF-IDF results (fetch extra to compensate for filtering)
        raw_results = self._index.search(query, top_k=max_results * 5)

        results: List[Dict[str, Any]] = []
        for idx, score in raw_results:
            assessment = self._assessments[idx]

            # Apply metadata filters
            if test_type and assessment.get("test_type_code", "").upper() != test_type.upper():
                # Also check the full test_type name
                if test_type.lower() not in (assessment.get("test_type") or "").lower():
                    continue

            if job_level:
                levels = [l.lower() for l in assessment.get("job_levels", [])]
                if not any(job_level.lower() in l for l in levels):
                    continue

            if max_duration and assessment.get("duration_minutes"):
                if assessment["duration_minutes"] > max_duration:
                    continue

            if language:
                langs = [l.lower() for l in assessment.get("languages", [])]
                if not any(language.lower() in l for l in langs):
                    continue

            result = {**assessment, "match_score": round(score, 4)}
            results.append(result)

            if len(results) >= max_results:
                break

        return results

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Exact name lookup (case-insensitive)."""
        if not self._loaded:
            self.load()
        name_lower = name.lower()
        for a in self._assessments:
            if a.get("name", "").lower() == name_lower:
                return a
        return None

    def get_all_test_types(self) -> List[str]:
        """Return all unique test types in the catalog."""
        if not self._loaded:
            self.load()
        types = set()
        for a in self._assessments:
            if a.get("test_type"):
                types.add(a["test_type"])
        return sorted(types)

    def get_all_job_levels(self) -> List[str]:
        """Return all unique job levels in the catalog."""
        if not self._loaded:
            self.load()
        levels = set()
        for a in self._assessments:
            for level in a.get("job_levels", []):
                levels.add(level)
        return sorted(levels)


# ── Module-level singleton ────────────────────────────────────
catalog_service = CatalogService()
