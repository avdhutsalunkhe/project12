"""
tests/test_embedding_pipeline.py

Pytest unit tests for the embedding & vector database pipeline.

Test groups
-----------
EmbeddingService    — encode(), encode_single(), dim, model_name
VectorStore         — upsert, query, count, reset, get_by_id
RetrievalService    — search(), search_similar(), filters, edge cases
pipeline.embedder   — module-level helper functions
pipeline.retrieval  — reusable retrieval functions (post-filters, format)
pipeline.ingest     — document text / metadata / ID builders

All tests that touch ChromaDB use a **temporary in-memory collection**
(via a fixture) so they don't pollute the real persistent store.

Run with:
    pytest tests/test_embedding_pipeline.py -v
    pytest tests/test_embedding_pipeline.py -v -k "embedding"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure project root is importable ───────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def embedding_svc():
    """Real EmbeddingService using all-MiniLM-L6-v2."""
    from app.services.embedding_service import EmbeddingService
    return EmbeddingService()


@pytest.fixture(scope="module")
def sample_texts() -> List[str]:
    return [
        "Java developer assessment for software engineers",
        "Python programming skills test",
        "Personality questionnaire for managers",
        "Numerical reasoning aptitude test",
        "Customer service representative evaluation",
    ]


@pytest.fixture(scope="function")
def temp_vector_store(tmp_path):
    """
    VectorStore pointing at a fresh temp directory.
    Each test function gets its own isolated collection.
    """
    from app.services.vector_store import VectorStore
    store = VectorStore(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )
    yield store
    # Cleanup
    try:
        store.delete_collection()
    except Exception:
        pass


@pytest.fixture(scope="function")
def populated_store(temp_vector_store, embedding_svc, sample_texts):
    """VectorStore pre-loaded with sample_texts and their embeddings."""
    embeddings = embedding_svc.encode(sample_texts)
    ids = [f"doc_{i}" for i in range(len(sample_texts))]
    metadatas = [
        {
            "name": f"Assessment {i}",
            "test_type_code": ["K", "K", "P", "A", "B"][i],
            "test_type": ["Knowledge", "Knowledge", "Personality", "Ability", "Behavior"][i],
            "duration_minutes": [30, 45, 20, 60, 15][i],
            "job_levels": ["Mid-Level", "Entry", "Manager", "Graduate", "Entry"][i],
            "url": f"https://example.com/assessment-{i}",
        }
        for i in range(len(sample_texts))
    ]
    temp_vector_store.upsert_documents(
        ids=ids,
        documents=sample_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return temp_vector_store


# ════════════════════════════════════════════════════════════════════════════
#  1. EmbeddingService tests
# ════════════════════════════════════════════════════════════════════════════

class TestEmbeddingService:

    def test_model_name(self, embedding_svc):
        assert embedding_svc.model_name == "all-MiniLM-L6-v2"

    def test_encode_single_text(self, embedding_svc):
        vec = embedding_svc.encode(["hello world"])
        assert isinstance(vec, list)
        assert len(vec) == 1
        assert isinstance(vec[0], list)
        assert len(vec[0]) == 384  # MiniLM-L6-v2 dimension

    def test_encode_batch(self, embedding_svc, sample_texts):
        vecs = embedding_svc.encode(sample_texts)
        assert len(vecs) == len(sample_texts)
        for v in vecs:
            assert len(v) == 384

    def test_encode_single_helper(self, embedding_svc):
        vec = embedding_svc.encode_single("test query")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_dimension_property(self, embedding_svc):
        # Trigger model load
        embedding_svc.encode_single("probe")
        assert embedding_svc.dimension == 384

    def test_normalize_produces_unit_vectors(self, embedding_svc):
        import math
        vec = embedding_svc.encode_single("unit norm test", normalize=True)
        magnitude = math.sqrt(sum(x * x for x in vec))
        assert abs(magnitude - 1.0) < 1e-4

    def test_string_input_wraps_to_list(self, embedding_svc):
        """encode() must accept a bare string."""
        result = embedding_svc.encode("bare string input")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_embeddings_are_different(self, embedding_svc):
        """Semantically different texts should produce different embeddings."""
        v1 = embedding_svc.encode_single("Java developer")
        v2 = embedding_svc.encode_single("Personality questionnaire")
        # Dot product of unit vectors = cosine similarity
        cos = sum(a * b for a, b in zip(v1, v2))
        assert cos < 0.95, "Dissimilar texts produced near-identical embeddings"

    def test_similar_texts_higher_similarity(self, embedding_svc):
        """Semantically similar texts should be closer than unrelated ones."""
        import math
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb + 1e-9)

        v_java = embedding_svc.encode_single("Java developer test")
        v_python = embedding_svc.encode_single("Python programming assessment")
        v_personality = embedding_svc.encode_single("cooking recipe baking")

        sim_related = cosine(v_java, v_python)
        sim_unrelated = cosine(v_java, v_personality)
        assert sim_related > sim_unrelated


# ════════════════════════════════════════════════════════════════════════════
#  2. VectorStore tests
# ════════════════════════════════════════════════════════════════════════════

class TestVectorStore:

    def test_empty_collection_count(self, temp_vector_store):
        assert temp_vector_store.count() == 0

    def test_upsert_and_count(self, populated_store, sample_texts):
        assert populated_store.count() == len(sample_texts)

    def test_query_returns_top_k(self, populated_store, embedding_svc):
        q_vec = embedding_svc.encode_single("Python programming")
        results = populated_store.query_texts(q_vec, top_k=3)
        assert len(results) <= 3

    def test_query_result_structure(self, populated_store, embedding_svc):
        q_vec = embedding_svc.encode_single("developer assessment")
        results = populated_store.query_texts(q_vec, top_k=2)
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "document" in r
        assert "metadata" in r
        assert "score" in r
        assert "distance" in r

    def test_scores_in_valid_range(self, populated_store, embedding_svc):
        q_vec = embedding_svc.encode_single("assessment test")
        results = populated_store.query_texts(q_vec, top_k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0 + 1e-5

    def test_scores_descending(self, populated_store, embedding_svc):
        q_vec = embedding_svc.encode_single("Java developer")
        results = populated_store.query_texts(q_vec, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_by_id(self, populated_store):
        doc = populated_store.get_by_id("doc_0")
        assert doc is not None
        assert doc["id"] == "doc_0"
        assert doc["document"] != ""

    def test_get_by_id_missing(self, populated_store):
        doc = populated_store.get_by_id("nonexistent_id_xyz")
        assert doc is None

    def test_where_filter(self, populated_store, embedding_svc):
        q_vec = embedding_svc.encode_single("assessment")
        results = populated_store.query_texts(
            q_vec,
            top_k=10,
            where={"test_type_code": {"$eq": "K"}},
        )
        # All returned results must have test_type_code == "K"
        for r in results:
            assert r["metadata"]["test_type_code"] == "K"

    def test_upsert_idempotent(self, temp_vector_store, embedding_svc):
        """Upserting same IDs twice must not inflate the count."""
        texts = ["text one", "text two"]
        ids = ["id_1", "id_2"]
        embeddings = embedding_svc.encode(texts)
        metadatas = [{"name": "one"}, {"name": "two"}]
        temp_vector_store.upsert_documents(ids, texts, embeddings, metadatas)
        temp_vector_store.upsert_documents(ids, texts, embeddings, metadatas)
        assert temp_vector_store.count() == 2

    def test_reset_clears_collection(self, populated_store):
        assert populated_store.count() > 0
        populated_store.reset()
        assert populated_store.count() == 0

    def test_collection_info(self, populated_store, sample_texts):
        info = populated_store.collection_info()
        assert info["count"] == len(sample_texts)
        assert "name" in info
        assert "persist_directory" in info


# ════════════════════════════════════════════════════════════════════════════
#  3. RetrievalService tests (uses temp store via monkeypatching)
# ════════════════════════════════════════════════════════════════════════════

class TestRetrievalService:

    @pytest.fixture(autouse=True)
    def patch_singletons(self, populated_store, embedding_svc, monkeypatch):
        """
        Patch the module-level singletons used by RetrievalService
        so tests run against the isolated temp store.
        """
        import app.services.retrieval_service as rs_mod
        monkeypatch.setattr(rs_mod, "vector_store", populated_store)
        monkeypatch.setattr(rs_mod, "embedding_service", embedding_svc)
        # Rebuild the service with patched deps
        from app.services.retrieval_service import RetrievalService
        self.svc = RetrievalService()
        self.svc._embedding_service = embedding_svc
        self.svc._vector_store = populated_store

    def test_is_ready(self):
        assert self.svc.is_ready is True

    def test_index_count(self, sample_texts):
        assert self.svc.index_count == len(sample_texts)

    def test_search_returns_list(self):
        results = self.svc.search("Java developer")
        assert isinstance(results, list)

    def test_search_top_k(self):
        results = self.svc.search("assessment", top_k=2)
        assert len(results) <= 2

    def test_search_result_keys(self):
        results = self.svc.search("Python programming", top_k=1)
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "score" in r
        assert "metadata" in r

    def test_search_filter_test_type(self):
        results = self.svc.search("assessment", top_k=10, test_type_code="K")
        for r in results:
            assert r["metadata"]["test_type_code"] == "K"

    def test_search_filter_max_duration(self):
        results = self.svc.search("assessment", top_k=10, max_duration=30)
        for r in results:
            duration = r["metadata"].get("duration_minutes", 0)
            assert duration <= 30

    def test_search_min_score_filter(self):
        results = self.svc.search("assessment", top_k=10, min_score=0.99)
        # Very high threshold — might return 0 results (acceptable)
        for r in results:
            assert r["score"] >= 0.99

    def test_search_similar(self):
        results = self.svc.search_similar("doc_0", top_k=3)
        # Must not return doc_0 itself
        ids = [r["id"] for r in results]
        assert "doc_0" not in ids

    def test_empty_query_handled(self):
        results = self.svc.search("", top_k=5)
        assert isinstance(results, list)


# ════════════════════════════════════════════════════════════════════════════
#  4. pipeline.embedder module tests
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineEmbedder:

    def test_embed_documents(self, sample_texts):
        from pipeline.embedder import embed_documents
        vecs = embed_documents(sample_texts, show_progress=False)
        assert len(vecs) == len(sample_texts)
        assert len(vecs[0]) == 384

    def test_embed_query(self):
        from pipeline.embedder import embed_query
        vec = embed_query("Java developer test")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_batch(self, sample_texts):
        from pipeline.embedder import embed_batch
        vecs = embed_batch(sample_texts)
        assert len(vecs) == len(sample_texts)

    def test_get_embedding_dim(self):
        from pipeline.embedder import get_embedding_dim
        assert get_embedding_dim() == 384

    def test_get_model_name(self):
        from pipeline.embedder import get_model_name
        assert get_model_name() == "all-MiniLM-L6-v2"

    def test_warm_up(self):
        from pipeline.embedder import warm_up
        info = warm_up()
        assert info["dimension"] == 384
        assert info["latency_ms"] >= 0

    def test_health_check_ok(self):
        from pipeline.embedder import health_check
        result = health_check()
        assert result["status"] == "ok"
        assert result["dimension"] == 384

    def test_embed_empty_list(self):
        from pipeline.embedder import embed_documents
        result = embed_documents([])
        assert result == []


# ════════════════════════════════════════════════════════════════════════════
#  5. pipeline.retrieval module tests (patched singletons)
# ════════════════════════════════════════════════════════════════════════════

class TestPipelineRetrieval:

    @pytest.fixture(autouse=True)
    def patch_retrieval_module(self, populated_store, embedding_svc, monkeypatch):
        """Patch both retrieval and vector_store singletons in pipeline.retrieval."""
        import app.services.retrieval_service as rs_mod
        import pipeline.retrieval as ret_mod

        monkeypatch.setattr(rs_mod, "vector_store", populated_store)
        monkeypatch.setattr(rs_mod, "embedding_service", embedding_svc)

        # Build a fresh service pointing at temp store
        from app.services.retrieval_service import RetrievalService
        fresh_svc = RetrievalService()
        fresh_svc._embedding_service = embedding_svc
        fresh_svc._vector_store = populated_store

        monkeypatch.setattr(ret_mod, "retrieval_service", fresh_svc)
        monkeypatch.setattr(ret_mod, "vector_store", populated_store)

    def test_search_function(self):
        from pipeline.retrieval import search
        results = search("Python programming", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_search_with_scores(self):
        from pipeline.retrieval import search_with_scores
        results = search_with_scores("assessment", top_k=3)
        for r in results:
            assert "rank" in r
            assert "score_pct" in r
            assert 1 <= r["rank"] <= 3

    def test_batch_search(self, sample_texts):
        from pipeline.retrieval import batch_search
        queries = ["Java developer", "personality test"]
        all_results = batch_search(queries, top_k=3)
        assert len(all_results) == 2
        for res_list in all_results:
            assert isinstance(res_list, list)

    def test_filter_by_type(self, populated_store, embedding_svc):
        from pipeline.retrieval import search, filter_by_type
        results = search("assessment", top_k=10)
        filtered = filter_by_type(results, "K")
        for r in filtered:
            assert r["metadata"]["test_type_code"] == "K"

    def test_filter_by_duration(self, populated_store, embedding_svc):
        from pipeline.retrieval import search, filter_by_duration
        results = search("assessment", top_k=10)
        filtered = filter_by_duration(results, 30)
        for r in filtered:
            assert (r["metadata"].get("duration_minutes") or 0) <= 30

    def test_filter_by_min_score(self):
        from pipeline.retrieval import search, filter_by_min_score
        results = search("assessment", top_k=10)
        filtered = filter_by_min_score(results, 0.5)
        for r in filtered:
            assert r["score"] >= 0.5

    def test_format_results_empty(self):
        from pipeline.retrieval import format_results
        output = format_results([])
        assert "no results" in output

    def test_format_results_non_empty(self):
        from pipeline.retrieval import search, format_results
        results = search("developer", top_k=2)
        output = format_results(results)
        assert "Score:" in output

    def test_is_ready(self):
        from pipeline.retrieval import is_ready
        assert is_ready() is True

    def test_store_info(self):
        from pipeline.retrieval import store_info
        info = store_info()
        assert "count" in info
        assert info["count"] > 0


# ════════════════════════════════════════════════════════════════════════════
#  6. pipeline.ingest helper function tests (pure / no I/O)
# ════════════════════════════════════════════════════════════════════════════

class TestIngestHelpers:

    def _assessment(self, **overrides) -> Dict[str, Any]:
        base = {
            "name": "Verify – Numerical Ability",
            "description": "Tests numerical reasoning for graduates.",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical/",
            "test_type_code": "A",
            "test_type": "Ability & Aptitude",
            "duration_minutes": 17,
            "remote_testing": True,
            "job_levels": ["Graduate", "Mid-Professional"],
            "languages": ["English", "French", "Spanish", "German"],
            "fact_sheet_urls": ["https://example.com/fact.pdf"],
        }
        base.update(overrides)
        return base

    def test_build_document_text_includes_name(self):
        from pipeline.ingest import _build_document_text
        a = self._assessment()
        text = _build_document_text(a)
        assert "Verify" in text

    def test_build_document_text_includes_test_type(self):
        from pipeline.ingest import _build_document_text
        a = self._assessment()
        text = _build_document_text(a)
        assert "Ability" in text

    def test_build_document_text_includes_job_levels(self):
        from pipeline.ingest import _build_document_text
        a = self._assessment()
        text = _build_document_text(a)
        assert "Graduate" in text

    def test_build_document_text_no_empty_parts(self):
        from pipeline.ingest import _build_document_text
        # Minimal assessment
        a = {"name": "Simple Test"}
        text = _build_document_text(a)
        assert "Simple Test" in text
        assert "None" not in text

    def test_build_metadata_duration(self):
        from pipeline.ingest import _build_metadata
        a = self._assessment(duration_minutes=30)
        meta = _build_metadata(a)
        assert meta["duration_minutes"] == 30
        assert isinstance(meta["duration_minutes"], int)

    def test_build_metadata_remote_testing_bool(self):
        from pipeline.ingest import _build_metadata
        meta = _build_metadata(self._assessment(remote_testing=True))
        assert meta["remote_testing"] is True

    def test_build_metadata_job_levels_string(self):
        from pipeline.ingest import _build_metadata
        a = self._assessment(job_levels=["Graduate", "Mid-Professional"])
        meta = _build_metadata(a)
        assert isinstance(meta["job_levels"], str)
        assert "Graduate" in meta["job_levels"]

    def test_build_metadata_languages_truncated(self):
        from pipeline.ingest import _build_metadata
        # Only first 5 languages should appear in document text
        a = self._assessment()
        meta = _build_metadata(a)
        # All languages stored in metadata (no truncation there)
        assert "English" in meta["languages"]

    def test_build_metadata_fact_sheet_first_only(self):
        from pipeline.ingest import _build_metadata
        a = self._assessment(fact_sheet_urls=["https://a.com/1.pdf", "https://b.com/2.pdf"])
        meta = _build_metadata(a)
        assert meta["fact_sheet_url"] == "https://a.com/1.pdf"

    def test_build_document_id_from_url(self):
        from pipeline.ingest import _build_document_id
        a = self._assessment(url="https://www.shl.com/solutions/products/product-catalog/view/verify-numerical/")
        doc_id = _build_document_id(a, 0)
        assert doc_id == "shl_verify-numerical"

    def test_build_document_id_fallback(self):
        from pipeline.ingest import _build_document_id
        a = {"name": "No URL"}
        doc_id = _build_document_id(a, 7)
        assert doc_id == "shl_0007"

    def test_load_catalog_missing_file(self, tmp_path):
        from pipeline.ingest import _load_catalog
        with pytest.raises(FileNotFoundError):
            _load_catalog(str(tmp_path / "missing.json"))

    def test_load_catalog_filters_errored(self, tmp_path):
        from pipeline.ingest import _load_catalog
        catalog = {
            "assessments": [
                {"name": "Good One", "url": "https://example.com/good"},
                {"name": "", "scrape_error": "timeout"},
                {"scrape_error": "404", "url": "https://example.com/bad"},
            ]
        }
        p = tmp_path / "catalog.json"
        p.write_text(json.dumps(catalog), encoding="utf-8")
        loaded = _load_catalog(str(p))
        assert len(loaded) == 1
        assert loaded[0]["name"] == "Good One"
