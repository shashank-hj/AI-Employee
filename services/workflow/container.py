from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from workflow.database.session import get_db
from workflow.repositories.workflow_repo import WorkflowRepository
from workflow.services.workflow_service import WorkflowService


@lru_cache()
def get_repository(db: AsyncSession = Depends(get_db)) -> WorkflowRepository:
    return WorkflowRepository(db)


async def get_workflow_service(db: AsyncSession = Depends(get_db)) -> WorkflowService:
    repo = WorkflowRepository(db)
    return WorkflowService(repo)
