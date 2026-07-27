from fastapi import APIRouter, Depends

from memory.container import get_memory_service
from memory.schemas.session import SessionCreate, SessionResponse, SessionMessage
from memory.services.memory_service import MemoryService

router = APIRouter(prefix="/memory")


@router.post("/session", response_model=SessionResponse)
async def upsert_session(
    data: SessionCreate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.upsert_session(data)


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.get_session(session_id)


@router.delete("/session/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    await service.delete_session(session_id)


@router.post("/session/{session_id}/message", response_model=SessionResponse)
async def add_message(
    session_id: str,
    message: SessionMessage,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.add_session_message(session_id, message)
