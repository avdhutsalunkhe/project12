"""
Chat / recommendation service -- business logic layer.

Full pipeline:
  1. Reconstruct ConversationState from messages[] (stateless)
  2. Search via semantic retrieval (ChromaDB) with TF-IDF fallback
  3. Build a rich natural-language reply with ranked recommendations

No session storage, no memory. Everything is derived from the
incoming messages[] array on every request.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger
from app.schemas.chat import (
    AssessmentRecommendation,
    ChatRequest,
    ChatResponse,
    MessageRole,
)
from app.services.conversation_state import ConversationState, reconstruct_state
from app.services.refusal_engine import refusal_engine

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Search Backend Abstraction
# ═══════════════════════════════════════════════════════════════

def _fetch_semantic(
    state: ConversationState,
    max_results: int,
) -> List[Dict[str, Any]]:
    """Fetch from ChromaDB semantic search (may return empty)."""
    try:
        from app.services.retrieval_service import retrieval_service
        if not retrieval_service.is_ready:
            return []
        return retrieval_service.search(
            query=state.search_query,
            top_k=max_results * 2,  # fetch extra for reranking pool
            test_type_code=state.preferred_test_type,
            max_duration=state.max_duration,
            min_score=0.10,
        )
    except Exception as exc:
        logger.warning("Semantic search failed: %s", exc)
        return []


def _fetch_tfidf(
    state: ConversationState,
    max_results: int,
) -> List[Dict[str, Any]]:
    """Fetch from TF-IDF keyword search (may return empty)."""
    try:
        from app.services.catalog_service import catalog_service
        if not catalog_service.loaded:
            return []
        return catalog_service.search(
            query=state.search_query,
            max_results=max_results * 2,
            test_type=state.preferred_test_type,
            job_level=state.seniority,
            max_duration=state.max_duration,
            language=state.language,
        )
    except Exception as exc:
        logger.warning("TF-IDF search failed: %s", exc)
        return []


def _search(
    state: ConversationState,
    max_results: int,
) -> List[Dict[str, Any]]:
    """
    Hybrid search: fetch from both backends, rerank, validate.
    """
    from app.services.hybrid_ranker import hybrid_rank, hallucination_guard

    semantic_results = _fetch_semantic(state, max_results)
    tfidf_results = _fetch_tfidf(state, max_results)

    logger.info(
        "Raw results: %d semantic, %d tfidf",
        len(semantic_results),
        len(tfidf_results),
    )

    # Hybrid reranking
    ranked = hybrid_rank(
        semantic_results=semantic_results,
        tfidf_results=tfidf_results,
        state=state,
        max_results=max_results,
    )

    # Hallucination prevention
    validated = hallucination_guard.validate(ranked)

    logger.info(
        "After ranking: %d candidates -> %d validated",
        len(ranked),
        len(validated),
    )

    return validated[:max_results]


# ═══════════════════════════════════════════════════════════════
#  Response Formatting
# ═══════════════════════════════════════════════════════════════

def _format_recommendation(
    assessment: Dict[str, Any], rank: int
) -> AssessmentRecommendation:
    """Convert a raw result dict into the API schema."""
    job_levels = assessment.get("job_levels", [])
    if isinstance(job_levels, str):
        job_levels = [l.strip() for l in job_levels.split(",") if l.strip()]

    languages = assessment.get("languages", [])
    if isinstance(languages, str):
        languages = [l.strip() for l in languages.split(",") if l.strip()]

    return AssessmentRecommendation(
        name=assessment.get("name", "Unknown"),
        url=assessment.get("url"),
        test_type=assessment.get("test_type_code") or assessment.get("test_type"),
    )


def _build_reply(
    recommendations: List[Dict[str, Any]],
    state: ConversationState,
) -> str:
    """Generate a natural-language response summarising recommendations."""
    if not recommendations:
        parts = ["I couldn't find any assessments matching your criteria."]
        if state.extraction_confidence > 0:
            parts.append(
                "Try broadening your search by removing some constraints "
                "(e.g., job level, duration, or test type)."
            )
        else:
            parts.append(
                "Could you provide more details about the role, skills, "
                "or assessment type you're looking for?"
            )
        return " ".join(parts)

    n = len(recommendations)

    # Context-aware intro
    intro_parts = [f"I found **{n}** relevant SHL assessment{'s' if n > 1 else ''}"]
    if state.role:
        intro_parts.append(f"for **{state.role}**")
    if state.seniority:
        intro_parts.append(f"({state.seniority})")
    intro = " ".join(intro_parts) + ":\n\n"

    # Format each result
    lines = []
    for i, rec in enumerate(recommendations, 1):
        name = rec.get("name", "Unknown")
        score = rec.get("match_score", 0)
        duration = rec.get("duration_minutes")
        test_type = rec.get("test_type", "")
        desc = rec.get("description", "")

        # Truncate long descriptions
        if desc and len(desc) > 160:
            desc = desc[:157] + "..."

        line = f"**{i}. {name}** (Match: {score:.0%})"
        details = []
        if test_type:
            details.append(f"Type: {test_type}")
        if duration:
            details.append(f"Duration: {duration} min")
        if details:
            line += f"\n   {' | '.join(details)}"
        if desc:
            line += f"\n   {desc}"
        lines.append(line)

    body = "\n\n".join(lines)

    # Active filters summary
    filter_parts = []
    if state.seniority:
        filter_parts.append(f"Level: {state.seniority}")
    if state.preferred_test_type:
        filter_parts.append(f"Type: {state.preferred_test_type}")
    if state.max_duration:
        filter_parts.append(f"Max Duration: {state.max_duration} min")
    if state.language:
        filter_parts.append(f"Language: {state.language}")

    filter_note = ""
    if filter_parts:
        filter_note = f"\n\n_Filters: {', '.join(filter_parts)}_"

    # Extracted context summary
    context_note = ""
    context_items = []
    if state.technical_skills:
        context_items.append(f"Skills: {', '.join(state.technical_skills[:5])}")
    if state.personality_requirements:
        context_items.append(f"Traits: {', '.join(state.personality_requirements[:3])}")
    if context_items:
        context_note = f"\n_Understood: {'; '.join(context_items)}_"

    tip = "\n\nYou can refine by specifying duration, job level, test type, or specific skills."

    return intro + body + filter_note + context_note + tip


# ═══════════════════════════════════════════════════════════════
#  ChatService
# ═══════════════════════════════════════════════════════════════

class ChatService:
    """Orchestrates conversational assessment recommendation."""

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Full pipeline:
          0. Run refusal check on the latest user message
          1. Reconstruct ConversationState from messages[]
          2. Search via semantic retrieval + TF-IDF fallback
          3. Build rich response with ranked recommendations
        """
        logger.info(
            "Processing chat request -- %d message(s), max_recs=%d",
            len(request.messages),
            request.max_recommendations,
        )

        # Step 0: Refusal check — must run before any search logic
        raw_messages = [
            {"role": m.role.value, "content": m.content}
            for m in request.messages
        ]
        refusal = refusal_engine.check_messages(raw_messages)
        if refusal.should_refuse:
            logger.info(
                "Refusal triggered: category=%s", refusal.category
            )
            return ChatResponse(
                reply=refusal.reply or "I can only help with SHL assessment recommendations.",
                recommendations=[],
                end_of_conversation=False,
            )

        # Step 1: Reconstruct state from full conversation
        state = reconstruct_state(raw_messages, context=request.context)

        # Handle empty / greeting messages
        if not state.search_query.strip():
            return ChatResponse(
                reply=(
                    "Hello! I'm the SHL Assessment Recommender. "
                    "Tell me about the role you're hiring for, the skills "
                    "you need to assess, or the type of test you're looking "
                    "for, and I'll suggest the best SHL assessments."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # Step 1.5: Clarification check for vague queries
        from app.services.clarification_engine import decide
        decision = decide(state)
        if decision.should_clarify:
            logger.info("Clarification needed (score=%.2f). Asking: %s", decision.sufficiency_score, decision.question)
            return ChatResponse(
                reply=decision.question,
                recommendations=[],
                end_of_conversation=False,
            )

        logger.info(
            "State: role=%s, seniority=%s, skills=%s, query='%s'",
            state.role,
            state.seniority,
            state.technical_skills[:3],
            state.search_query[:100],
        )

        # Step 2: Search
        results = _search(state, request.max_recommendations)

        # Step 3: Build response
        recommendations = [
            _format_recommendation(r, i + 1)
            for i, r in enumerate(results)
        ]

        reply = _build_reply(results, state)

        logger.info(
            "Returning %d recommendations (confidence=%.0f%%)",
            len(recommendations),
            state.extraction_confidence * 100,
        )

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=True,  # Providing recommendations means we reached the end of the flow.
        )


# Module-level singleton (stateless, so safe to reuse)
chat_service = ChatService()
