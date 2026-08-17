from fastapi import APIRouter, Depends, Query
from typing import Optional

from memory.container import get_memory_service
from memory.schemas.long_term import LongTermMemoryCreate, LongTermMemoryResponse
from memory.services.memory_service import MemoryService
from shared.utils.response import paginated_response

router = APIRouter(prefix="/memory")


@router.post("/long-term", response_model=LongTermMemoryResponse, status_code=201)
async def store_long_term(
    data: LongTermMemoryCreate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.store_long_term(data)


@router.get("/long-term/{memory_id}", response_model=LongTermMemoryResponse)
async def get_long_term(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.get_long_term(memory_id)


@router.get("/long-term")
async def list_long_term(
    user_id: str = Query(default=None),
    memory_type: Optional[str] = Query(None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: MemoryService = Depends(get_memory_service),
):
    if not user_id:
        return paginated_response(items=[], total=0, page=page, page_size=page_size)
    items, total = await service.list_long_term(user_id, memory_type, page, page_size)
    return paginated_response(
        items=[i.model_dump(mode="json") for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/long-term/{memory_id}", status_code=204)
async def delete_long_term(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    await service.delete_long_term(memory_id)
