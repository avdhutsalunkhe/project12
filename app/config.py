"""
Application configuration — loaded from environment variables / .env file.

Uses pydantic-settings for typed, validated configuration with sensible defaults.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for the entire application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    APP_NAME: str = "SHL Assessment Recommender"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"  # development | staging | production
    DEBUG: bool = True

    # ── Server ───────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ── CORS (JSON array in .env, e.g. ["http://localhost:3000"]) ─
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # ── Logging ──────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json | text

    # ── API Keys (extend as needed) ──────────────────────────────────────────
    # OPENAI_API_KEY: str | None = None
    # GEMINI_API_KEY: str | None = None

    # ── Embedding ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIM: int = 384  # MiniLM-L6-v2 output dimension

    # ── Vector Store (ChromaDB) ───────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "shl_assessments"
    RETRIEVAL_TOP_K: int = 10          # default top-k for API endpoints
    RETRIEVAL_MIN_SCORE: float = 0.0   # minimum cosine similarity threshold

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this instead of instantiating Settings directly."""
    return Settings()
