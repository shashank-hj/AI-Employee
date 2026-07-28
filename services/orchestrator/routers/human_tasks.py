from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from orchestrator.database.session import get_db
from orchestrator.schemas.human_task import HumanTaskCreate, HumanTaskResponse, HumanTaskResolve
from orchestrator.services.human_task_service import HumanTaskService

router = APIRouter(prefix="/api/human-tasks", tags=["Human Tasks"])
logger = structlog.get_logger(__name__)


@router.post("", response_model=HumanTaskResponse, status_code=201)
async def create_task(data: HumanTaskCreate, db: AsyncSession = Depends(get_db)):
    try:
        service = HumanTaskService(db)
        return await service.create(data)
    except Exception as exc:
        logger.error("human_task_create_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Task service unavailable")


@router.get("", response_model=list[HumanTaskResponse])
async def list_tasks(limit: int = 50, db: AsyncSession = Depends(get_db)):
    try:
        service = HumanTaskService(db)
        return await service.list_pending(limit)
    except Exception as exc:
        logger.warning("human_task_list_failed", error=str(exc))
        return []


@router.get("/count")
async def get_count(db: AsyncSession = Depends(get_db)):
    try:
        service = HumanTaskService(db)
        return {"pending": await service.get_count()}
    except Exception as exc:
        logger.warning("human_task_count_failed", error=str(exc))
        return {"pending": 0}


@router.post("/{task_id}/claim", response_model=HumanTaskResponse)
async def claim_task(task_id: str, assigned_to: str = "operator", db: AsyncSession = Depends(get_db)):
    try:
        service = HumanTaskService(db)
        task = await service.claim(task_id, assigned_to)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("human_task_claim_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Task service unavailable")


@router.post("/{task_id}/resolve", response_model=HumanTaskResponse)
async def resolve_task(task_id: str, data: HumanTaskResolve, db: AsyncSession = Depends(get_db)):
    try:
        service = HumanTaskService(db)
        task = await service.resolve(task_id, data)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("human_task_resolve_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Task service unavailable")
