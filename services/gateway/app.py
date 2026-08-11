from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import gateway.models  # noqa: F401 - register gateway models on Base
from gateway.routers import channels, health, proxy, webchat
from shared.auth.middleware import register_auth_middleware
from shared.models.base import Base
from shared.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    try:
        from gateway.database.session import engine

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_initialized")
    except Exception as exc:
        logger.warning("database_init_skipped", error=str(exc)[:200])
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Employee Platform - Gateway",
        version="0.1.0",
        description="API Gateway for the AI Employee Platform",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_auth_middleware(app)
    app.include_router(health.router, tags=["Health"])
    app.include_router(webchat.router, tags=["Web Chat"])
    app.include_router(channels.router, tags=["Channels"])
    app.include_router(proxy.router, tags=["Proxy"])
    return app
