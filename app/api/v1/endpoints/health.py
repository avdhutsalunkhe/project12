"""
GET /health — lightweight liveness / readiness probe.
"""

from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_app_settings
from app.schemas.common import BaseResponse
from app.schemas.health import HealthCheckResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=BaseResponse[HealthCheckResponse],
    summary="Health check",
    description="Returns service health status, version, and environment.",
)
async def health_check(
    settings: Settings = Depends(get_app_settings),
) -> BaseResponse[HealthCheckResponse]:
    payload = HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        service=settings.APP_NAME,
    )
    return BaseResponse(data=payload, message="Service is healthy")
