from fastapi import APIRouter
from memory.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="memory", version="0.1.0")
