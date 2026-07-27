from fastapi import APIRouter, Depends

from memory.container import get_memory_service
from memory.schemas.search import MemorySearchRequest, MemorySearchResult
from memory.services.memory_service import MemoryService

router = APIRouter(prefix="/memory")


@router.post("/search", response_model=list[MemorySearchResult])
async def search_memories(
    request: MemorySearchRequest,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.search_memories(request)
