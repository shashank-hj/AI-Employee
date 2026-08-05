from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from workflow.config import settings
from workflow.database.session import engine
from workflow.routers import health, workflows
from shared.models.base import Base
from shared.utils.exceptions import AppException
from shared.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    import workflow.models.workflow  # noqa: F811 — register models on Base
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("database_init_skipped", error=str(e))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Employee Platform - Workflow Engine",
        version="0.1.0",
        description="LangGraph-based Workflow Execution Engine",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail, "error_code": exc.error_code},
        )

    app.include_router(health.router, tags=["Health"])
    app.include_router(workflows.router, tags=["Workflows"])
    return app
