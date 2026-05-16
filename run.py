"""
Entry-point for running the application directly with `python run.py`.

For production use `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
"""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_development,
        workers=settings.WORKERS if not settings.is_development else 1,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
