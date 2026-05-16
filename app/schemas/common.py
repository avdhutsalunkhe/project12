"""
Common / shared schema primitives used across multiple endpoints.
"""

from datetime import datetime
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class BaseResponse(BaseModel, Generic[T]):
    """Standard API response envelope."""

    success: bool = True
    data: Optional[T] = None
    message: str = "OK"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorDetail(BaseModel):
    """Structured error detail inside an error response."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    success: bool = False
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationMeta(BaseModel):
    """Pagination metadata for list endpoints."""

    page: int = 1
    page_size: int = 20
    total_items: int = 0
    total_pages: int = 0
