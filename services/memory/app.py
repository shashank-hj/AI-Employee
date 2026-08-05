from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from memory.config import settings
from memory.database.session import engine
from memory.routers import health, sessions, long_term, search, conversations, profiles
from shared.models.base import Base
from shared.utils.exceptions import AppException
from shared.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    import memory.models  # noqa: F811 — register all memory models on Base
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
        title="AI Employee Platform - Memory",
        version="0.1.0",
        description="Dual-backend Memory Service (Redis + PostgreSQL/pgvector)",
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
    app.include_router(sessions.router, tags=["Session Memory"])
    app.include_router(long_term.router, tags=["Long-Term Memory"])
    app.include_router(search.router, tags=["Search"])
    app.include_router(conversations.router, tags=["Conversations"])
    app.include_router(profiles.router, tags=["Profiles"])

    return app
