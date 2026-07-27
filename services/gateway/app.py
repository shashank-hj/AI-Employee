from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gateway.config import settings
from gateway.routers import health, proxy
from shared.utils.logging import setup_logging
from shared.auth.middleware import register_auth_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Employee Platform - Gateway",
        version="0.1.0",
        description="API Gateway for the AI Employee Platform",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )
    register_auth_middleware(app)
    app.include_router(health.router, tags=["Health"])
    app.include_router(proxy.router, tags=["Proxy"])
    return app
