"""
FastAPI dependency injection providers.

Centralized so that endpoints remain thin and testable.
"""

from app.config import Settings, get_settings


def get_app_settings() -> Settings:
    """Dependency that provides the Settings singleton to endpoints."""
    return get_settings()


# ──────────────────────────────────────────────────────────────
# Add future dependencies here, e.g.:
#
# async def get_db_session() -> AsyncGenerator[AsyncSession, None]: ...
# async def get_llm_client() -> LLMClient: ...
# ──────────────────────────────────────────────────────────────
