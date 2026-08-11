from fastapi import APIRouter

from gateway.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="gateway", version="0.1.0")
