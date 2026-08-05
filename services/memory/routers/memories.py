from typing import Optional
from fastapi import APIRouter, Depends, Query

from memory.container import get_memory_service
from memory.schemas.memory import MemoryCreate, MemoryResponse, MemoryType
from memory.services.memory_service import MemoryService

router = APIRouter(prefix="/api")


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def store_memory(
    memory: MemoryCreate,
    service: MemoryService = Depends(get_memory_service),
):
    result = await service.store_long_term(
        user_id=memory.user_id or "system",
        content=memory.content,
        memory_type=(memory.memory_type.value if memory.memory_type else "fact"),
        importance=memory.importance or 0.5,
        metadata=memory.metadata,
        source=memory.source,
    )
    return MemoryResponse(
        id=result.get("id", "stub") if result else "stub",
        content=memory.content,
        memory_type=memory.memory_type or MemoryType.CONVERSATION,
        importance=memory.importance,
        metadata=memory.metadata,
        source=memory.source,
        user_id=memory.user_id,
    )


@router.get("/memories", response_model=list[MemoryResponse])
async def list_memories(
    memory_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    service: MemoryService = Depends(get_memory_service),
):
    result = await service.list_long_term(user_id=user_id, memory_type=memory_type)
    items = result.get("items", []) if isinstance(result, dict) else result
    responses = []
    for item in items:
        responses.append(MemoryResponse(
            id=item.get("id", ""),
            content=item.get("content", ""),
            memory_type=MemoryType(item.get("memory_type", "fact")),
            importance=item.get("importance", 0.5),
            metadata=item.get("metadata"),
            source=item.get("source"),
            user_id=item.get("user_id"),
            created_at=item.get("created_at"),
        ))
    return responses


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    result = await service.get_long_term(memory_id)
    return MemoryResponse(
        id=result.get("id", memory_id) if result else memory_id,
        content=result.get("content", "") if result else "",
        memory_type=MemoryType(result.get("memory_type", "fact")) if result else MemoryType.CONVERSATION,
        importance=result.get("importance", 0.5) if result else 0.5,
        metadata=result.get("metadata") if result else None,
        source=result.get("source") if result else None,
        user_id=result.get("user_id") if result else None,
        created_at=result.get("created_at") if result else None,
    )


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    await service.delete_long_term(memory_id)


@router.get("/memories/search", response_model=list[MemoryResponse])
async def search_memories(
    query: str = Query(..., min_length=1),
    service: MemoryService = Depends(get_memory_service),
):
    result = await service.search_long_term(query=query)
    items = result if isinstance(result, list) else result.get("results", [])
    responses = []
    for item in items:
        responses.append(MemoryResponse(
            id=item.get("id", ""),
            content=item.get("content", ""),
            memory_type=MemoryType(item.get("memory_type", "fact")),
            importance=item.get("importance", 0.5),
            metadata=item.get("metadata"),
            source=item.get("source"),
            user_id=item.get("user_id"),
            created_at=item.get("created_at"),
        ))
    return responses
