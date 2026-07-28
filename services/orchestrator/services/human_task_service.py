import structlog
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.models.human_task import HumanTask
from orchestrator.schemas.human_task import HumanTaskCreate, HumanTaskResponse, HumanTaskResolve

logger = structlog.get_logger(__name__)


class HumanTaskService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, data: HumanTaskCreate) -> HumanTaskResponse:
        task = HumanTask(
            request_id=data.request_id,
            user_input=data.user_input,
            intent=data.intent,
            reason=data.reason,
            priority=data.priority,
            context=data.context,
            status="pending",
        )
        self._db.add(task)
        await self._db.commit()
        await self._db.refresh(task)
        logger.info("human_task_created", task_id=task.id, intent=data.intent)
        return HumanTaskResponse.model_validate(task)

    async def list_pending(self, limit: int = 50) -> list[HumanTaskResponse]:
        stmt = (
            select(HumanTask)
            .where(HumanTask.status == "pending")
            .order_by(HumanTask.created_at.desc())
            .limit(limit)
        )
        result = await self._db.execute(stmt)
        tasks = result.scalars().all()
        return [HumanTaskResponse.model_validate(t) for t in tasks]

    async def claim(self, task_id: str, assigned_to: str = "operator") -> HumanTaskResponse | None:
        task = await self._db.get(HumanTask, task_id)
        if task is None:
            return None
        task.status = "in_progress"
        task.assigned_to = assigned_to
        await self._db.commit()
        await self._db.refresh(task)
        logger.info("human_task_claimed", task_id=task_id, assigned_to=assigned_to)
        return HumanTaskResponse.model_validate(task)

    async def resolve(self, task_id: str, data: HumanTaskResolve) -> HumanTaskResponse | None:
        task = await self._db.get(HumanTask, task_id)
        if task is None:
            return None
        task.status = "resolved"
        task.resolved_at = datetime.now(timezone.utc)
        if data.resolution_note:
            task.resolution_note = data.resolution_note
        if data.assigned_to:
            task.assigned_to = data.assigned_to
        await self._db.commit()
        await self._db.refresh(task)
        logger.info("human_task_resolved", task_id=task_id)
        return HumanTaskResponse.model_validate(task)

    async def get_count(self) -> int:
        stmt = select(func.count()).where(HumanTask.status == "pending")
        result = await self._db.execute(stmt)
        return result.scalar_one()
