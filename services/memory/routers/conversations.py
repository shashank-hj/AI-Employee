from fastapi import APIRouter, Depends

from memory.container import get_memory_service
from memory.schemas.conversation import ConversationMessageCreate, ConversationMessageResponse
from memory.services.memory_service import MemoryService

router = APIRouter(prefix="/memory")


@router.post("/conversation", response_model=ConversationMessageResponse, status_code=201)
async def store_message(
    data: ConversationMessageCreate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.store_message(data)


@router.get("/conversation/{session_id}", response_model=list[ConversationMessageResponse])
async def get_conversation(
    session_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.get_conversation(session_id)
