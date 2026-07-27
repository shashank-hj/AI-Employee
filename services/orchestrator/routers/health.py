from fastapi import APIRouter
from orchestrator.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="orchestrator", version="0.1.0")
