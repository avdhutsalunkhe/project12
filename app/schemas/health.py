"""
Schemas for the GET /health endpoint.
"""

from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    """Payload returned by the health-check endpoint."""

    status: str = "healthy"
    version: str
    environment: str
    service: str
