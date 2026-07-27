from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from tool_registry.database.session import get_db
from tool_registry.repositories.tool_repository import ToolRepository
from tool_registry.services.tool_service import ToolService


async def get_tool_repository(db: AsyncSession = Depends(get_db)) -> ToolRepository:
    return ToolRepository(db)


async def get_tool_service(repo: ToolRepository = Depends(get_tool_repository)) -> ToolService:
    return ToolService(repo)
