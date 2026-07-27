from fastapi import APIRouter, Depends

from memory.container import get_memory_service
from memory.schemas.profile import UserProfileCreate, UserProfileUpdate, UserProfileResponse
from memory.services.memory_service import MemoryService

router = APIRouter(prefix="/memory")


@router.put("/profile", response_model=UserProfileResponse)
async def upsert_profile(
    data: UserProfileCreate,
    service: MemoryService = Depends(get_memory_service),
):
    return await service.upsert_profile(data)


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
