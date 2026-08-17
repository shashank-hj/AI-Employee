from fastapi import APIRouter, Depends, Query

from memory.container import get_memory_service
from memory.schemas.profile import UserProfileCreate, UserProfileUpdate, UserProfileResponse
from memory.services.memory_service import MemoryService
from shared.utils.response import paginated_response

router = APIRouter(prefix="/memory")


@router.put("/profile", response_model=UserProfileResponse)
async def upsert_profile(
    data: UserProfileCreate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.upsert_profile(data)


@router.get("/profile")
async def list_profiles(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: MemoryService = Depends(get_memory_service),
):
    items, total = await service.list_profiles(page=page, page_size=page_size)
    return paginated_response(
        items=[i.model_dump(mode="json") for i in items],
        total=total, page=page, page_size=page_size,
    )


@router.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_profile(
    user_id: str,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.get_profile(user_id)


@router.patch("/profile/{user_id}", response_model=UserProfileResponse)
async def update_profile(
    user_id: str,
    data: UserProfileUpdate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.update_profile(user_id, data)
