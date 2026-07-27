from typing import Optional
from uuid import UUID

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import ARRAY

from tool_registry.models.tool import ToolModel


class ToolRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, tool: ToolModel) -> ToolModel:
        self._session.add(tool)
        await self._session.flush()
        await self._session.refresh(tool)
        return tool

    async def get_by_id(self, tool_id: str) -> Optional[ToolModel]:
        stmt = select(ToolModel).where(ToolModel.id == tool_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[ToolModel]:
        stmt = select(ToolModel).where(func.lower(ToolModel.name) == name.lower())
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        category: Optional[str] = None,
        is_active: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[ToolModel], int]:
        conditions = []

        if category is not None:
            conditions.append(ToolModel.category == category)
        if is_active is not None:
            conditions.append(ToolModel.is_active == is_active)
        if tags:
            conditions.append(ToolModel.tags.overlap(tags))
        if search:
            ilike = f"%{search}%"
            conditions.append(
                or_(
                    ToolModel.name.ilike(ilike),
                    ToolModel.description.ilike(ilike),
                )
            )

        base = select(ToolModel)
        if conditions:
            base = base.where(*conditions)

        count_stmt = select(func.count()).select_from(base.subquery())
        total_result = await self._session.execute(count_stmt)
        total = total_result.scalar() or 0

        list_stmt = (
            base.order_by(ToolModel.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self._session.execute(list_stmt)
        items = list(result.scalars().all())

        return items, total

    async def update(self, tool: ToolModel, update_data: dict) -> ToolModel:
        for key, value in update_data.items():
            setattr(tool, key, value)
        await self._session.flush()
        await self._session.refresh(tool)
        return tool

    async def delete(self, tool: ToolModel) -> None:
        await self._session.delete(tool)
        await self._session.flush()
