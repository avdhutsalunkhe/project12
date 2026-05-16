"""
POST /compare — Compare two or more SHL assessments side-by-side.

Uses catalog data only; no hallucination.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.logging_config import get_logger
from app.schemas.compare import (
    AssessmentProfileOut,
    CompareRequest,
    CompareResponse,
    DimensionResult,
)
from app.services.comparison_engine import comparison_engine

logger = get_logger(__name__)

router = APIRouter(prefix="/compare", tags=["compare"])


@router.post(
    "",
    response_model=CompareResponse,
    summary="Compare SHL assessments",
    description=(
        "Compare 2–5 SHL assessments across **purpose, type, duration, "
        "skills, and use cases** using catalog data only. "
        "Names are resolved with exact → prefix → fuzzy matching."
    ),
)
async def compare_assessments(request: CompareRequest) -> CompareResponse:
    """Side-by-side comparison of SHL assessments sourced from catalog data."""
    logger.info("Compare request: %s", request.names)

    result = comparison_engine.compare(request.names)

    if not result.profiles and result.not_found:
        raise HTTPException(
            status_code=404,
            detail=(
                f"None of the requested assessments were found in the catalog: "
                f"{', '.join(result.not_found)}"
            ),
        )

    return CompareResponse(
        assessments=result.assessments,
        profiles=[
            AssessmentProfileOut(
                name=p.name,
                url=p.url,
                test_type=p.test_type,
                test_type_code=p.test_type_code,
                duration_minutes=p.duration_minutes,
                job_levels=p.job_levels,
                languages=p.languages,
                purpose=p.purpose,
                skills=p.skills,
                use_cases=p.use_cases,
            )
            for p in result.profiles
        ],
        dimensions=[
            DimensionResult(
                dimension=d.dimension,
                values=d.values,
                notes=d.notes,
            )
            for d in result.dimensions
        ],
        summary=result.summary,
        not_found=result.not_found,
    )
