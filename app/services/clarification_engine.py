"""
Clarification decision engine.

Dynamically determines whether the current ConversationState has
enough information to produce quality recommendations, or whether
a single clarification question should be asked.

Design principles:
  - No hardcoded conversation flows or decision trees
  - Priority-weighted dimensions drive the decision
  - At most ONE question per turn to minimize friction
  - Vague queries get the highest-value missing question
  - As confidence rises, the bar to skip clarification drops
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.services.conversation_state import ConversationState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Information Dimension Definitions
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InfoDimension:
    """
    A single dimension of information needed for good recommendations.

    Attributes:
        name:       Internal identifier.
        weight:     How much this dimension improves result quality (0–1).
                    Higher = more valuable to ask about first.
        question:   Template for the clarification question.
        follow_ups: Alternative phrasings based on what IS known.
    """
    name: str
    weight: float
    question: str
    follow_ups: dict  # {condition_key: alternative_question}


# Dimensions ordered by typical information value.
# The engine picks the highest-weight MISSING dimension.
_DIMENSIONS: List[InfoDimension] = [

    InfoDimension(
        name="role",
        weight=0.95,
        question="What role or position are you looking to assess candidates for?",
        follow_ups={
            "has_skills": "Based on the skills you mentioned, are you hiring for a specific role like {skills_hint}?",
        },
    ),

    InfoDimension(
        name="seniority",
        weight=0.70,
        question="What seniority level is this role? (e.g., entry-level, mid-level, senior, manager)",
        follow_ups={
            "has_role": "What seniority level for the {role} position?",
        },
    ),

    InfoDimension(
        name="skills",
        weight=0.60,
        question="Are there specific technical skills or competencies you need to evaluate?",
        follow_ups={
            "has_role": "What specific skills should the {role} be tested on?",
        },
    ),

    InfoDimension(
        name="test_type",
        weight=0.40,
        question="Do you have a preference for assessment type? (e.g., cognitive/aptitude, personality, technical knowledge, simulation)",
        follow_ups={
            "has_role_and_skills": "Would you prefer a technical knowledge test, a personality assessment, or a cognitive aptitude test for this role?",
        },
    ),

    InfoDimension(
        name="constraints",
        weight=0.30,
        question="Any constraints on the assessment? (e.g., maximum duration, language, remote testing support)",
        follow_ups={
            "has_basics": "Any time or format constraints? For example, should the test be under a certain number of minutes?",
        },
    ),
]


# ═══════════════════════════════════════════════════════════════
#  Sufficiency Scoring
# ═══════════════════════════════════════════════════════════════

def _is_dimension_filled(dim: InfoDimension, state: ConversationState) -> bool:
    """Check if a dimension has been extracted from the conversation."""
    if dim.name == "role":
        return state.role is not None
    elif dim.name == "seniority":
        return state.seniority is not None
    elif dim.name == "skills":
        return len(state.technical_skills) > 0
    elif dim.name == "test_type":
        return state.preferred_test_type is not None
    elif dim.name == "constraints":
        # Filled if ANY constraint is specified
        return (
            state.max_duration is not None
            or state.language is not None
        )
    return False


def compute_sufficiency(state: ConversationState) -> float:
    """
    Compute how sufficient the current state is for producing
    quality recommendations.

    Returns a score from 0.0 (nothing known) to 1.0 (fully specified).
    The score is a weighted sum of filled dimensions.
    """
    total_weight = sum(d.weight for d in _DIMENSIONS)
    filled_weight = sum(
        d.weight for d in _DIMENSIONS
        if _is_dimension_filled(d, state)
    )

    # Bonus for personality/communication traits (not a separate dimension
    # but adds to overall understanding)
    bonus = 0.0
    if state.personality_requirements:
        bonus += 0.05
    if state.communication_requirements:
        bonus += 0.05

    score = (filled_weight + bonus) / total_weight
    return min(round(score, 3), 1.0)


# ═══════════════════════════════════════════════════════════════
#  Question Generation
# ═══════════════════════════════════════════════════════════════

def _generate_question(
    dim: InfoDimension,
    state: ConversationState,
) -> str:
    """
    Generate a contextual clarification question for a missing dimension.

    Uses follow_up phrasings when related dimensions ARE filled,
    so the question feels natural and connected to the conversation.
    """
    # Try context-aware follow-ups first
    if dim.follow_ups:
        if "has_skills" in dim.follow_ups and state.technical_skills:
            skills_hint = _infer_role_from_skills(state.technical_skills)
            if skills_hint:
                return dim.follow_ups["has_skills"].format(
                    skills_hint=skills_hint
                )

        if "has_role" in dim.follow_ups and state.role:
            return dim.follow_ups["has_role"].format(role=state.role)

        if "has_role_and_skills" in dim.follow_ups and state.role and state.technical_skills:
            return dim.follow_ups["has_role_and_skills"]

        if "has_basics" in dim.follow_ups and (state.role or state.technical_skills):
            return dim.follow_ups["has_basics"]

    # Fall back to the generic question
    return dim.question


def _infer_role_from_skills(skills: List[str]) -> Optional[str]:
    """
    Suggest a plausible role name from extracted skills.
    Used to make clarification questions more specific.
    """
    skills_lower = {s.lower() for s in skills}

    role_hints = [
        ({"java", "spring", "microservices"}, "a Java Developer"),
        ({"python", "django", "flask"}, "a Python Developer"),
        ({"python", "machine learning", "data science"}, "a Data Scientist"),
        ({"react", "angular", "vue.js", "javascript", "frontend"}, "a Frontend Developer"),
        ({"node.js", "backend", "rest api"}, "a Backend Developer"),
        ({"docker", "kubernetes", "devops", "ci/cd"}, "a DevOps Engineer"),
        ({"sql", "data analysis", "tableau", "power bi"}, "a Data Analyst"),
        ({"aws", "azure", "gcp"}, "a Cloud Engineer"),
        ({"full stack"}, "a Full Stack Developer"),
        ({"project management"}, "a Project Manager"),
        ({"salesforce"}, "a Salesforce Developer"),
        ({"accounting / finance"}, "an Accounting Professional"),
        ({"human resources"}, "an HR Professional"),
    ]

    for required_skills, role_name in role_hints:
        if skills_lower & required_skills:
            return role_name

    return None


# ═══════════════════════════════════════════════════════════════
#  Decision Engine (public API)
# ═══════════════════════════════════════════════════════════════

# Threshold: if sufficiency >= this, skip clarification and search
SUFFICIENCY_THRESHOLD = 0.55


@dataclass
class ClarificationDecision:
    """Result of the clarification engine."""

    should_clarify: bool
    """True if a clarification question should be asked."""

    question: Optional[str]
    """The single clarification question (None if should_clarify=False)."""

    sufficiency_score: float
    """Current information sufficiency (0.0 – 1.0)."""

    missing_dimensions: List[str]
    """Names of dimensions still missing."""

    filled_dimensions: List[str]
    """Names of dimensions already filled."""


def decide(
    state: ConversationState,
    threshold: float = SUFFICIENCY_THRESHOLD,
) -> ClarificationDecision:
    """
    Decide whether to ask a clarification question or proceed to search.

    Algorithm:
      1. Compute sufficiency score from filled dimensions
      2. If score >= threshold → proceed (no question)
      3. If score < threshold → pick the highest-weight missing dimension
         and generate a single contextual question

    Args:
        state: The reconstructed conversation state.
        threshold: Sufficiency threshold (default 0.55).

    Returns:
        ClarificationDecision with the verdict and optional question.
    """
    sufficiency = compute_sufficiency(state)

    filled = [d.name for d in _DIMENSIONS if _is_dimension_filled(d, state)]
    missing = [d.name for d in _DIMENSIONS if not _is_dimension_filled(d, state)]

    # If we have enough info, skip clarification
    if sufficiency >= threshold:
        logger.info(
            "Sufficiency %.1f%% >= %.1f%% threshold -- proceeding to search "
            "(filled=%s, missing=%s)",
            sufficiency * 100,
            threshold * 100,
            filled,
            missing,
        )
        return ClarificationDecision(
            should_clarify=False,
            question=None,
            sufficiency_score=sufficiency,
            missing_dimensions=missing,
            filled_dimensions=filled,
        )

    # Find the highest-weight missing dimension
    missing_dims = [
        d for d in _DIMENSIONS
        if not _is_dimension_filled(d, state)
    ]

    if not missing_dims:
        # Shouldn't happen, but handle gracefully
        return ClarificationDecision(
            should_clarify=False,
            question=None,
            sufficiency_score=sufficiency,
            missing_dimensions=[],
            filled_dimensions=filled,
        )

    # Pick the most valuable missing dimension
    target = missing_dims[0]  # Already ordered by weight (highest first)
    question = _generate_question(target, state)

    logger.info(
        "Sufficiency %.1f%% < %.1f%% -- asking about '%s' "
        "(filled=%s, missing=%s)",
        sufficiency * 100,
        threshold * 100,
        target.name,
        filled,
        missing,
    )

    return ClarificationDecision(
        should_clarify=True,
        question=question,
        sufficiency_score=sufficiency,
        missing_dimensions=missing,
        filled_dimensions=filled,
    )
