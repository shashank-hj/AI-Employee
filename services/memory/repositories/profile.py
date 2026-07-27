from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory.models.profile import UserProfileModel


class ProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_user_id(self, user_id: str) -> Optional[UserProfileModel]:
        stmt = select(UserProfileModel).where(UserProfileModel.user_id == user_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert(self, profile: UserProfileModel) -> UserProfileModel:
        existing = await self.get_by_user_id(profile.user_id)
        if existing:
            existing.display_name = profile.display_name or existing.display_name
            existing.preferences = {**existing.preferences, **profile.preferences}
            existing.metadata_ = profile.metadata_ or existing.metadata_
            await self._session.flush()
            await self._session.refresh(existing)
            return existing
        self._session.add(profile)
        await self._session.flush()
        await self._session.refresh(profile)
        return profile
