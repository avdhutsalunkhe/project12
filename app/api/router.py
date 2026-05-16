"""
Central API router — aggregates all versioned endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.compare import router as compare_router
from app.api.v1.endpoints.health import router as health_router

api_router = APIRouter()

# ── v1 routes (prefix-free — versioning via /api/v1 is applied in main.py) ──
api_router.include_router(health_router)
api_router.include_router(chat_router)
api_router.include_router(compare_router)
