
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from workflow.models.workflow import WorkflowModel


class WorkflowRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, wf: WorkflowModel) -> WorkflowModel:
        self._session.add(wf)
        await self._session.flush()
        return wf

    async def get_by_id(self, workflow_id: str) -> WorkflowModel | None:
        stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self, status: str | None = None, page: int = 1, page_size: int = 20
    ) -> tuple[list[WorkflowModel], int]:
        stmt = select(WorkflowModel)
        if status:
            stmt = stmt.where(WorkflowModel.status == status)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self._session.execute(count_stmt)).scalar() or 0
        list_stmt = stmt.order_by(WorkflowModel.created_at.desc()).offset(
            (page - 1) * page_size
        ).limit(page_size)
        result = await self._session.execute(list_stmt)
        return list(result.scalars().all()), total

    async def update(self, wf: WorkflowModel) -> WorkflowModel:
        await self._session.merge(wf)
        await self._session.flush()
        return wf

    async def delete(self, workflow_id: str) -> bool:
        wf = await self.get_by_id(workflow_id)
        if wf is None:
            return False
        await self._session.delete(wf)
        await self._session.flush()
        return True

    async def commit(self) -> None:
        """Persist pending changes; the response then reflects durable state."""
        await self._session.commit()
