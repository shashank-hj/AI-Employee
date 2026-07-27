import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from orchestrator.config import settings
from orchestrator.container import _build_rag_client, _build_llm_provider
from orchestrator.routers import health, agent
from shared.utils.exceptions import AppException
from shared.utils.logging import setup_logging

logger = structlog.get_logger(__name__)

APP_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    rag_client = app.state.rag_client
    llm_provider = app.state.llm_provider

    _log_startup_config()

    results = await asyncio.gather(
        _check_rag(rag_client),
        _check_llm(llm_provider),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            logger.error("startup_health_check_error", error=str(result))

    yield


def _log_startup_config() -> None:
    logger.info(
        "startup_configuration",
        version=APP_VERSION,
        environment=settings.ENVIRONMENT,
        service_name=settings.SERVICE_NAME,
        rag_url=settings.RAG_URL,
        llm_provider="SarvamProvider" if settings.SARVAM_API_KEY else "MockPlanner",
        llm_model=settings.SARVAM_MODEL,
        log_level=settings.LOG_LEVEL,
        rag_timeout=settings.RAG_TIMEOUT,
    )


async def _check_rag(rag_client) -> None:
    healthy = await rag_client.health_check()
    if healthy:
        logger.info("rag_healthy", base_url=settings.RAG_URL)
    else:
        logger.warning(
            "rag_unhealthy",
            base_url=settings.RAG_URL,
            message="RAG service unavailable; SearchDocumentsTool will use mock fallback",
        )


async def _check_llm(llm_provider) -> None:
    if llm_provider is None:
        logger.warning("llm_disabled", message="No SARVAM_API_KEY configured; using MockPlanner")
        return
    healthy = await llm_provider.health_check()
    if healthy:
        logger.info("llm_healthy", model=settings.SARVAM_MODEL)
    else:
        logger.warning(
            "llm_unhealthy",
            model=settings.SARVAM_MODEL,
            message="Sarvam API unreachable; intent classification will fall back to 'general'",
        )


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Employee Platform - Orchestrator",
        version=APP_VERSION,
        description="Agent Orchestration Engine with LangGraph",
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
            content={
                "status": "error",
                "message": exc.detail,
                "error_code": exc.error_code,
            },
        )

    app.state.rag_client = _build_rag_client()
    app.state.llm_provider = _build_llm_provider()

    app.include_router(health.router, tags=["Health"])
    app.include_router(agent.router, tags=["Agent"])

    return app
