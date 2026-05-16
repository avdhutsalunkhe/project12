"""
Application factory — builds and configures the FastAPI instance.

Keeps main.py declarative; all wiring lives here so the app can be
imported by tests and ASGI servers without side-effects.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.middleware import register_middleware
from app.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle hook.

    Initialises the SHL catalog index on startup so search is
    available immediately when the first request arrives.
    """
    settings = get_settings()
    logger.info(
        "Starting %s v%s [env=%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
    )

    # ── Startup logic ────────────────────────────────────────────
    # For Render free tier, we completely bypass heavy startup initialization.
    # Vector DB, embeddings, and TF-IDF catalog are lazy-loaded on the first request.
    # This prevents OOM kills during boot and allows the /health endpoint to return instantly.
    logger.info("Startup complete. Resources will be lazy-loaded on first request to conserve memory.")

    yield

    # ── Shutdown logic ───────────────────────────────────────────
    logger.info("%s shutting down", settings.APP_NAME)


def create_app() -> FastAPI:
    """Build and return a fully-configured FastAPI application."""

    # Logging must be configured first
    setup_logging()

    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Conversational SHL Assessment Recommendation System — "
            "stateless API that suggests the best SHL assessments based "
            "on a natural-language conversation."
        ),
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom middleware ────────────────────────────────────────
    register_middleware(app)

    # ── Exception handlers ───────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ──────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Root Health Probes ───────────────────────────────────────
    # Render's default health check and port scanner sends a GET/HEAD request to the root path (/).
    # If this returns 404, Render assumes the application is not ready and times out the deployment.
    @app.get("/")
    @app.head("/")
    async def root_health_check():
        return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}

    return app


# ── Module-level ASGI application instance ────────────────────
# Used by: `uvicorn app.main:app` and the run.py entry-point.
app = create_app()
