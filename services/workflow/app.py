from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from workflow.config import settings
from workflow.routers import health, workflows
from shared.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
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
    app.include_router(health.router, tags=["Health"])
    app.include_router(workflows.router, tags=["Workflows"])
    return app
