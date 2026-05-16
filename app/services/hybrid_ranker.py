"""
Hybrid retrieval and ranking system for SHL assessments.

Combines multiple scoring signals with weighted reranking:
  - 0.45 × semantic similarity  (ChromaDB vector cosine)
  - 0.25 × keyword overlap      (Jaccard token overlap)
  - 0.20 × role matching        (extracted role vs. assessment metadata)
  - 0.10 × seniority matching   (job level alignment)

Also enforces hallucination prevention:
  - Every recommendation is validated against the canonical catalog
  - Invalid URLs are rejected
  - Assessments not in the catalog are filtered out
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services.conversation_state import ConversationState

logger = logging.getLogger(__name__)

# ── Scoring weights ──────────────────────────────────────────────
W_SEMANTIC = 0.45
W_KEYWORD = 0.25
W_ROLE = 0.20
W_SENIORITY = 0.10


# ═══════════════════════════════════════════════════════════════
#  Component Scorers
# ═══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> Set[str]:
    """Lowercase tokenisation for keyword overlap."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def score_keyword_overlap(query: str, document: str) -> float:
    """
    Jaccard-based keyword overlap between query and document.
    Returns 0.0 – 1.0.
    """
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(document)

    if not q_tokens or not d_tokens:
        return 0.0

    intersection = q_tokens & d_tokens
    union = q_tokens | d_tokens

    return len(intersection) / len(union)


def score_role_match(state: ConversationState, metadata: Dict[str, Any]) -> float:
    """
    Score how well the assessment matches the target role.
    Uses fuzzy token matching between the extracted role, skills,
    and the assessment name/description.

    Returns 0.0 – 1.0.
    """
    if not state.role and not state.technical_skills:
        return 0.0

    # Build a set of role-relevant tokens
    role_tokens: Set[str] = set()
    if state.role:
        role_tokens.update(_tokenize(state.role))
    for skill in state.technical_skills[:5]:
        role_tokens.update(_tokenize(skill))

    if not role_tokens:
        return 0.0

    # Build assessment tokens from name + test_type
    assess_tokens: Set[str] = set()
    assess_tokens.update(_tokenize(metadata.get("name", "")))
    assess_tokens.update(_tokenize(metadata.get("test_type", "")))

    # Also include document text if available (but truncated)
    doc_text = metadata.get("document", "")
    if doc_text:
        assess_tokens.update(_tokenize(doc_text[:300]))

    if not assess_tokens:
        return 0.0

    overlap = role_tokens & assess_tokens
    # Normalize against role tokens (recall-oriented)
    return len(overlap) / len(role_tokens)


def score_seniority_match(
    state: ConversationState,
    metadata: Dict[str, Any],
) -> float:
    """
    Score seniority alignment between state and assessment.

    Returns 1.0 for exact match, 0.5 for partial, 0.0 for none.
    """
    if not state.seniority:
        return 0.5  # Neutral if no seniority preference

    job_levels = metadata.get("job_levels", "")
    if isinstance(job_levels, list):
        job_levels = ", ".join(job_levels)

    if not job_levels:
        return 0.3  # Unknown levels — slight penalty

    target = state.seniority.lower()
    levels_lower = job_levels.lower()

    # Exact substring match
    if target in levels_lower:
        return 1.0

    # Partial match: check adjacent seniority levels
    _ADJACENT = {
        "entry-level": ["graduate", "general population"],
        "graduate": ["entry-level", "mid-professional"],
        "mid-professional": ["graduate", "professional individual contributor", "front line manager"],
        "professional individual contributor": ["mid-professional", "front line manager"],
        "front line manager": ["supervisor", "manager", "mid-professional"],
        "supervisor": ["front line manager", "manager"],
        "manager": ["supervisor", "director", "front line manager"],
        "director": ["manager", "executive"],
        "executive": ["director"],
    }

    adjacent = _ADJACENT.get(target, [])
    for adj in adjacent:
        if adj in levels_lower:
            return 0.6

    return 0.1


# ═══════════════════════════════════════════════════════════════
#  Hybrid Ranker
# ═══════════════════════════════════════════════════════════════

def hybrid_rank(
    semantic_results: List[Dict[str, Any]],
    tfidf_results: List[Dict[str, Any]],
    state: ConversationState,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Combine semantic and TF-IDF results with weighted reranking.

    Each candidate gets a hybrid score:
      hybrid = 0.45 × semantic + 0.25 × keyword + 0.20 × role + 0.10 × seniority

    Deduplicates by assessment name, keeping the higher-scored instance.

    Args:
        semantic_results: Results from ChromaDB vector search.
        tfidf_results: Results from TF-IDF keyword search.
        state: Reconstructed conversation state.
        max_results: Max number of results to return.

    Returns:
        Sorted list of assessment dicts with `hybrid_score`.
    """
    # Merge candidates into a deduplicated pool
    candidates: Dict[str, Dict[str, Any]] = {}

    for r in semantic_results:
        meta = r.get("metadata", {})
        name = meta.get("name", r.get("name", "")).lower().strip()
        if not name:
            continue

        key = name
        if key not in candidates or r.get("score", 0) > candidates[key].get("_semantic_score", 0):
            candidates[key] = {
                **meta,
                "name": meta.get("name", r.get("name", "")),
                "url": meta.get("url", r.get("url", "")),
                "document": r.get("document", ""),
                "description": r.get("document", meta.get("description", "")),
                "_semantic_score": r.get("score", 0.0),
                "_source": "semantic",
            }

    for r in tfidf_results:
        name = r.get("name", "").lower().strip()
        if not name:
            continue

        key = name
        tfidf_score = r.get("match_score", 0.0)

        if key in candidates:
            # Already have from semantic — enrich with TF-IDF score
            candidates[key]["_tfidf_score"] = tfidf_score
        else:
            candidates[key] = {
                **r,
                "document": r.get("description", ""),
                "_semantic_score": 0.0,
                "_tfidf_score": tfidf_score,
                "_source": "tfidf",
            }

    # Score each candidate
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for key, candidate in candidates.items():
        sem_score = candidate.get("_semantic_score", 0.0)
        kw_score = score_keyword_overlap(state.search_query, candidate.get("document", ""))
        role_score = score_role_match(state, candidate)
        sen_score = score_seniority_match(state, candidate)

        hybrid = (
            W_SEMANTIC * sem_score
            + W_KEYWORD * kw_score
            + W_ROLE * role_score
            + W_SENIORITY * sen_score
        )

        # Store component scores for debugging/transparency
        candidate["hybrid_score"] = round(hybrid, 4)
        candidate["match_score"] = round(hybrid, 4)
        candidate["_scores"] = {
            "semantic": round(sem_score, 4),
            "keyword": round(kw_score, 4),
            "role": round(role_score, 4),
            "seniority": round(sen_score, 4),
            "hybrid": round(hybrid, 4),
        }

        # Clean up internal keys from the output
        for internal_key in ("_semantic_score", "_tfidf_score", "_source"):
            candidate.pop(internal_key, None)

        scored.append((hybrid, candidate))

    # Sort by hybrid score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [item for _, item in scored[:max_results]]


# ═══════════════════════════════════════════════════════════════
#  Hallucination Prevention
# ═══════════════════════════════════════════════════════════════

class HallucinationGuard:
    """
    Validates recommendations against the canonical SHL catalog
    to prevent hallucinated or fabricated assessment data.
    """

    def __init__(self) -> None:
        self._valid_names: Set[str] = set()
        self._valid_urls: Set[str] = set()
        self._loaded = False

    def load_catalog(self) -> None:
        """Build the validation index from the catalog service."""
        from app.services.catalog_service import catalog_service

        if not catalog_service.loaded:
            logger.info("Lazy loading catalog service for hallucination guard...")
            catalog_service.load()

        for i in range(catalog_service.total):
            # Access assessments directly from the catalog
            pass

        # Rebuild from the raw assessments list
        import json
        import os
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "catalog.json",
        )
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for a in data.get("assessments", []):
                if a.get("name"):
                    self._valid_names.add(a["name"].lower().strip())
                if a.get("url"):
                    self._valid_urls.add(a["url"].strip().rstrip("/"))
            self._loaded = True
            logger.info(
                "Hallucination guard loaded: %d names, %d URLs",
                len(self._valid_names),
                len(self._valid_urls),
            )
        except Exception as exc:
            logger.error("Failed to load hallucination guard: %s", exc)

    def validate(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out any recommendation that doesn't exist in the catalog.

        Checks:
          1. Assessment name must exist in the catalog (fuzzy match)
          2. URL must be a valid SHL product catalog URL
          3. URL must exist in the canonical URL set
        """
        if not self._loaded:
            logger.info("Hallucination guard not loaded — lazy loading now...")
            self.load_catalog()

        validated: List[Dict[str, Any]] = []

        for rec in recommendations:
            name = rec.get("name", "").lower().strip()
            url = rec.get("url", "").strip().rstrip("/")

            # Check 1: Name exists in catalog
            if name and name not in self._valid_names:
                logger.warning(
                    "Hallucination rejected (name not in catalog): '%s'",
                    rec.get("name"),
                )
                continue

            # Check 2: URL format validation
            if url and not url.startswith("https://www.shl.com/"):
                logger.warning(
                    "Hallucination rejected (invalid URL domain): '%s'",
                    url,
                )
                continue

            # Check 3: URL exists in canonical set
            if url and self._valid_urls and url not in self._valid_urls:
                logger.warning(
                    "Hallucination rejected (URL not in catalog): '%s'",
                    url,
                )
                continue

            validated.append(rec)

        rejected = len(recommendations) - len(validated)
        if rejected > 0:
            logger.info(
                "Hallucination guard: %d/%d passed, %d rejected",
                len(validated),
                len(recommendations),
                rejected,
            )

        return validated


# ── Module-level singleton ────────────────────────────────────
hallucination_guard = HallucinationGuard()
