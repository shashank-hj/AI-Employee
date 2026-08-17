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
    try:
        await _run_backfill()
    except Exception as e:
        logger.warning("backfill_message_counts_skipped", error=str(e))
    yield


async def _run_backfill() -> None:
    """Reconcile Redis message_count with PostgreSQL after startup."""
    from memory.container import get_embedding_service, get_session_store, get_summarizer
    from memory.database.session import async_session
    from memory.repositories.conversation import ConversationRepository
    from memory.repositories.long_term import LongTermMemoryRepository
    from memory.repositories.profile import ProfileRepository
    from memory.services.memory_service import MemoryService

    async with async_session() as db:
        service = MemoryService(
            session_store=get_session_store(),
            long_term_repo=LongTermMemoryRepository(db),
            conversation_repo=ConversationRepository(db),
            profile_repo=ProfileRepository(db),
            embedding_service=get_embedding_service(),
            summarizer=get_summarizer(),
        )
        result = await service.backfill_message_counts()
        logger.info(
            "backfill_message_counts_startup",
            total=result.get("total", 0),
            errors=len(result.get("errors", [])),
        )


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

    @app.post("/sessions/{session_id}/messages")
    async def compat_add_messages(session_id: str, payload: dict):
        """Compatibility route for gateway which posts to old /sessions/ format."""
        from fastapi.responses import JSONResponse
        requests = payload.get("requests", [])
        from memory.container import get_memory_service
        service = get_memory_service()
        from memory.schemas.session import SessionMessage
        for req in requests:
            msg = SessionMessage(role=req.get("role", "user"), content=req.get("content", ""))
            await service.add_session_message(session_id, msg)
        sess = await service.get_session(session_id)
        return JSONResponse(content=sess.model_dump())

    return app
