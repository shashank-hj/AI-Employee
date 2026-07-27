from fastapi import APIRouter
from tool_registry.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="tool-registry", version="0.1.0")
