import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shared.models.base import Base
from shared.utils.exceptions import AppException
from shared.utils.logging import setup_logging
from speech.config import settings
from speech.routers import (
    health,
    language_detection,
    models,
    stt,
    translation,
    transliteration,
    tts,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    if settings.USAGE_PRICING:
        os.environ["USAGE_PRICING"] = settings.USAGE_PRICING

    import shared.usage.model  # noqa: F401 — register usage_events on Base

    try:
        from speech.database.session import engine

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        logger.warning("database_init_skipped", error=str(exc))

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Employee Platform - Speech",
        version="0.1.0",
        description="Speech-to-Text and Text-to-Speech service powered by Sarvam AI",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail, "error_code": exc.error_code},
        )

    app.include_router(health.router, tags=["Health"])
    app.include_router(stt.router, tags=["Speech-to-Text"])
    app.include_router(tts.router, tags=["Text-to-Speech"])
    app.include_router(models.router, tags=["Voice Models"])
    app.include_router(translation.router, tags=["Translation"])
    app.include_router(language_detection.router, tags=["Language Detection"])
    app.include_router(transliteration.router, tags=["Transliteration"])

    return app
