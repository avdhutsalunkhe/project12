"""
Schemas for the POST /chat endpoint.

Designed for a stateless conversational API — the client sends the full
conversation history on every request.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════

class MessageRole(str, Enum):
    """Who authored a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ═══════════════════════════════════════════════════════════════
#  Sub-models
# ═══════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: MessageRole
    content: str = Field(..., min_length=1, max_length=10_000)
    timestamp: Optional[datetime] = None


class AssessmentRecommendation(BaseModel):
    """A single SHL assessment recommendation returned by the system."""

    name: str
    url: Optional[str] = None
    test_type: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
#  Request / Response
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """
    Inbound payload for POST /chat.

    The client must include the full conversation history so the API
    remains completely stateless.
    """

    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Full conversation history, newest message last.",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra context (e.g. job role, industry).",
    )
    max_recommendations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of assessments to recommend.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "I need to assess Java developers for a mid-senior role.",
                        }
                    ],
                    "context": {"industry": "Technology", "job_level": "mid-senior"},
                    "max_recommendations": 5,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Outbound payload from POST /chat."""

    reply: str = Field(
        ...,
        description="Natural-language response from the assistant.",
    )
    recommendations: List[AssessmentRecommendation] = Field(
        default_factory=list,
        description="Ranked list of recommended assessments (may be empty).",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="Indicates whether the conversational flow has completed.",
    )
