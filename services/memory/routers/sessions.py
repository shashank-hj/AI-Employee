
from fastapi import APIRouter, Depends, Query

from memory.container import get_memory_service
from memory.schemas.session import (
    SessionCreate,
    SessionMessage,
    SessionResponse,
    SessionSummaryResponse,
    SessionUpdate,
    SummaryRequest,
)
from memory.services.memory_service import MemoryService
from shared.utils.response import paginated_response

router = APIRouter(prefix="/memory")


@router.post("/session", response_model=SessionResponse)
async def upsert_session(
    data: SessionCreate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.upsert_session(data)


@router.get("/session", response_model=dict)
async def list_sessions(
    user_id: str | None = Query(None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: MemoryService = Depends(get_memory_service),
):
    items, total = await service.list_sessions(user_id=user_id, page=page, page_size=page_size)
    return paginated_response(
        items=[i.model_dump(mode="json", exclude={"messages"}) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/session/backfill-counts")
async def backfill_counts(
    service: MemoryService = Depends(get_memory_service),
):
    return await service.backfill_message_counts()


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.get_session(session_id)


@router.patch("/session/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    data: SessionUpdate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.update_session_state(
        session_id=session_id,
        context=data.context,
        metadata=data.metadata,
    )


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


@router.post("/session/{session_id}/clear", response_model=SessionResponse)
async def clear_messages(
    session_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.clear_session_messages(session_id)


@router.post("/session/{session_id}/expire", status_code=204)
async def expire_session(
    session_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    await service.expire_session(session_id)


@router.post("/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def summarize_session(
    session_id: str,
    data: SummaryRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.generate_session_summary(
        session_id=session_id,
        message_limit=data.message_limit,
        store=data.store,
    )
