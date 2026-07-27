from typing import Optional
from fastapi import APIRouter, Query
from memory.schemas.memory import MemoryCreate, MemoryResponse, MemoryType

router = APIRouter(prefix="/api")


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def store_memory(memory: MemoryCreate):
    """Store a new memory."""
    from datetime import datetime
    return MemoryResponse(id="stub", created_at=datetime.utcnow(), **memory.model_dump())


@router.get("/memories", response_model=list[MemoryResponse])
async def list_memories(memory_type: Optional[MemoryType] = Query(None)):
    """List memories, optionally filtered by type."""
    return []


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str):
    """Get a memory by id."""
    from datetime import datetime
    return MemoryResponse(
        id=memory_id,
        content="stub",
        memory_type=MemoryType.CONVERSATION,
        created_at=datetime.utcnow(),
    )


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: str):
    """Delete a memory."""
    pass


@router.get("/memories/search", response_model=list[MemoryResponse])
async def search_memories(query: str = Query(..., min_length=1)):
    """Search memories by content."""
    return []
