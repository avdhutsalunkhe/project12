"""
Schemas for the POST /compare endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Inbound payload for POST /compare."""

    names: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="2–5 assessment names to compare (exact or approximate).",
        examples=[["Java (New)", "Python (New)"]],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"names": ["Java (New)", "Python (New)", "Agile Software Development"]}
            ]
        }
    }


class DimensionResult(BaseModel):
    """Comparison result for a single dimension."""

    dimension: str
    values: Dict[str, Any]   # assessment_name -> value
    notes: str = ""


class AssessmentProfileOut(BaseModel):
    """Public-facing profile of an assessed SHL product."""

    name: str
    url: Optional[str] = None
    test_type: Optional[str] = None
    test_type_code: Optional[str] = None
    duration_minutes: Optional[int] = None
    job_levels: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    purpose: str = ""
    skills: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    """Outbound payload from POST /compare."""

    assessments: List[str] = Field(
        description="Names of successfully resolved assessments."
    )
    profiles: List[AssessmentProfileOut] = Field(
        default_factory=list,
        description="Structured profile for each assessment.",
    )
    dimensions: List[DimensionResult] = Field(
        default_factory=list,
        description="Dimension-by-dimension comparison.",
    )
    summary: str = Field(
        description="Concise natural-language summary of the comparison."
    )
    not_found: List[str] = Field(
        default_factory=list,
        description="Assessment names that could not be resolved from the catalog.",
    )
