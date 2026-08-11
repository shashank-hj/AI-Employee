from contextlib import asynccontextmanager
import os

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from rag.config import settings
from rag.database.session import engine
from rag.routers import health, documents
from shared.models.base import Base
from shared.utils.exceptions import AppException
from shared.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    if settings.USAGE_PRICING:
        os.environ["USAGE_PRICING"] = settings.USAGE_PRICING

    import rag.models  # noqa: F811 — register rag models on Base
    import shared.usage.model  # noqa: F401 — register usage_events on Base
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("database_init_skipped", error=str(e))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Employee Platform - RAG",
        version="0.1.0",
        description="Retrieval Augmented Generation Service",
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
    app.include_router(documents.router, tags=["Documents"])

    return app
