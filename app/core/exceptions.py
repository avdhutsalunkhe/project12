"""
Custom exception hierarchy and global exception handlers.

All application-level exceptions inherit from AppException so they can
be caught in a single handler and translated to consistent HTTP responses.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Exception Classes
# ═══════════════════════════════════════════════════════════════

class AppException(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class BadRequestError(AppException):
    """400 — client sent an invalid request."""

    def __init__(self, message: str = "Bad request", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=400, error_code="BAD_REQUEST", details=details)


class NotFoundError(AppException):
    """404 — requested resource does not exist."""

    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=404, error_code="NOT_FOUND", details=details)


class ServiceUnavailableError(AppException):
    """503 — a downstream service is unreachable."""

    def __init__(self, message: str = "Service unavailable", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=503, error_code="SERVICE_UNAVAILABLE", details=details)


class RateLimitError(AppException):
    """429 — rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, status_code=429, error_code="RATE_LIMIT_EXCEEDED", details=details)


# ═══════════════════════════════════════════════════════════════
#  Global Exception Handlers
# ═══════════════════════════════════════════════════════════════

def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI application."""

    @app.exception_handler(AppException)
    async def app_exception_handler(_request: Request, exc: AppException) -> JSONResponse:
        logger.warning(
            "Application error: %s [%s]",
            exc.message,
            exc.error_code,
            extra={"status_code": exc.status_code, "details": exc.details},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                },
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {},
                },
            },
        )
