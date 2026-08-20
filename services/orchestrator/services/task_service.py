"""User task/reminder management (thin persistence layer).

The orchestrator had only an execution-task stub and the human-escalation queue
before this. User-facing tasks (create / update / complete / list / delete) are
backed by the ``user_tasks`` table so the Samvaad agent's ``tasks/manage``
webhook operates on real data.
"""

from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import delete, select

from orchestrator.database.session import async_session
from orchestrator.models.task import UserTask

logger = structlog.get_logger(__name__)

VALID_STATUSES = ("pending", "in_progress", "completed")


def _task_to_dict(task: UserTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "session_id": task.session_id,
        "user_id": task.user_id,
        "title": task.title,
        "description": task.description,
        "status": task.status,
        "priority": task.priority,
        "due_at": task.due_at.isoformat() if task.due_at else None,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


class TaskService:
    def __init__(self, session_factory: Any = async_session) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        *,
        title: str,
        session_id: str | None = None,
        user_id: str | None = None,
        description: str | None = None,
        priority: int = 0,
        due_at: datetime | None = None,
    ) -> dict[str, Any]:
        task = UserTask(
            title=title,
            session_id=session_id,
            user_id=user_id,
            description=description,
            priority=priority,
            due_at=due_at,
            status="pending",
        )
        async with self._session_factory() as db:
            db.add(task)
            await db.commit()
            await db.refresh(task)
            logger.info("task_created", task_id=task.id, user_id=user_id)
            return _task_to_dict(task)

    async def get(self, task_id: str) -> dict[str, Any] | None:
        async with self._session_factory() as db:
            task = await db.get(UserTask, task_id)
            return _task_to_dict(task) if task else None

    async def update(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        priority: int | None = None,
        due_at: datetime | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        async with self._session_factory() as db:
            task = await db.get(UserTask, task_id)
            if task is None:
                return None
            if title is not None:
                task.title = title
            if description is not None:
                task.description = description
            if priority is not None:
                task.priority = priority
            if due_at is not None:
                task.due_at = due_at
            if status is not None:
                task.status = status
                if status == "completed":
                    task.completed_at = datetime.now(UTC)
            task.updated_at = datetime.now(UTC)
            await db.commit()
            await db.refresh(task)
            logger.info("task_updated", task_id=task_id)
            return _task_to_dict(task)

    async def complete(self, task_id: str) -> dict[str, Any] | None:
        return await self.update(task_id, status="completed")

    async def delete(self, task_id: str) -> bool:
        async with self._session_factory() as db:
            result = await db.execute(
                delete(UserTask).where(UserTask.id == task_id)
            )
            await db.commit()
            return bool(result.rowcount and result.rowcount > 0)

    async def list(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        stmt = select(UserTask).order_by(UserTask.created_at.desc())
        if session_id:
            stmt = stmt.where(UserTask.session_id == session_id)
        if user_id:
            stmt = stmt.where(UserTask.user_id == user_id)
        if status:
            stmt = stmt.where(UserTask.status == status)
        stmt = stmt.limit(limit)
        async with self._session_factory() as db:
            result = await db.execute(stmt)
            return [_task_to_dict(t) for t in result.scalars().all()]
