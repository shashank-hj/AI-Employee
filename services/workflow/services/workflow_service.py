import structlog
from datetime import datetime, timezone

from workflow.models.workflow import WorkflowModel
from workflow.repositories.workflow_repo import WorkflowRepository
from workflow.schemas.workflows import (
    WorkflowCreate,
    WorkflowResponse,
    WorkflowStatus,
    WorkflowHistoryEntry,
)
from shared.utils.exceptions import NotFoundException

logger = structlog.get_logger(__name__)

_DEFAULT_STEPS = [
    {"name": "validate", "description": "Validate input data"},
    {"name": "process", "description": "Execute core logic"},
    {"name": "complete", "description": "Finalize workflow"},
]


class WorkflowService:
    def __init__(self, repo: WorkflowRepository) -> None:
        self._repo = repo

    async def create_workflow(self, data: WorkflowCreate) -> WorkflowResponse:
        steps = []
        for s in (data.steps or [s["name"] for s in _DEFAULT_STEPS]):
            steps.append(s if isinstance(s, dict) else {"name": s, "description": s})
        if not steps:
            steps = _DEFAULT_STEPS[:]

        wf = WorkflowModel(
            name=data.name,
            description=data.description,
            workflow_type=data.workflow_type,
            input_data=data.input_data,
            steps=steps,
            status=WorkflowStatus.PENDING.value,
            current_step=steps[0]["name"] if steps else None,
        )
        wf = await self._repo.create(wf)
        logger.info("workflow_created", workflow_id=str(wf.id), name=wf.name)

        try:
            wf.status = WorkflowStatus.RUNNING.value
            for i, step in enumerate(steps):
                wf.current_step = step["name"]
                await self._repo.update(wf)
                logger.info("workflow_step_executing", workflow_id=str(wf.id), step=step["name"], step_index=i)

            wf.status = WorkflowStatus.COMPLETED.value
            wf.current_step = None
            wf.output_data = {"completed_steps": len(steps), "result": "success"}
            await self._repo.update(wf)
            logger.info("workflow_completed", workflow_id=str(wf.id))
        except Exception as exc:
            wf.status = WorkflowStatus.FAILED.value
            wf.error_message = str(exc)
            await self._repo.update(wf)
            logger.error("workflow_failed", workflow_id=str(wf.id), error=str(exc))

        return self._to_response(wf)

    async def get_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        return self._to_response(wf)

    async def list_workflows(self, status: str | None = None, page: int = 1, page_size: int = 20) -> tuple[list[WorkflowResponse], int]:
        workflows, total = await self._repo.list_all(status=status, page=page, page_size=page_size)
        return [self._to_response(w) for w in workflows], total

    async def pause_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        if wf.status != WorkflowStatus.RUNNING.value:
            raise ValueError(f"Cannot pause workflow in '{wf.status}' status")
        wf.status = WorkflowStatus.PAUSED.value
        wf = await self._repo.update(wf)
        logger.info("workflow_paused", workflow_id=workflow_id)
        return self._to_response(wf)

    async def resume_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        if wf.status != WorkflowStatus.PAUSED.value:
            raise ValueError(f"Cannot resume workflow in '{wf.status}' status")
        wf.status = WorkflowStatus.RUNNING.value
        wf = await self._repo.update(wf)
        logger.info("workflow_resumed", workflow_id=workflow_id)
        return self._to_response(wf)

    async def cancel_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        if wf.status not in (WorkflowStatus.PENDING.value, WorkflowStatus.RUNNING.value, WorkflowStatus.PAUSED.value):
            raise ValueError(f"Cannot cancel workflow in '{wf.status}' status")
        wf.status = WorkflowStatus.CANCELLED.value
        wf = await self._repo.update(wf)
        logger.info("workflow_cancelled", workflow_id=workflow_id)
        return self._to_response(wf)

    async def get_history(self, workflow_id: str) -> list[WorkflowHistoryEntry]:
        wf = await self._get_or_raise(workflow_id)
        steps = wf.steps or []
        history: list[WorkflowHistoryEntry] = []
        for step in steps:
            step_status = WorkflowStatus.COMPLETED if wf.status in (WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value) else WorkflowStatus.PENDING
            history.append(WorkflowHistoryEntry(
                step_name=step["name"] if isinstance(step, dict) else str(step),
                status=step_status,
                started_at=wf.created_at,
                completed_at=wf.updated_at,
            ))
        return history

    async def _get_or_raise(self, workflow_id: str) -> WorkflowModel:
        wf = await self._repo.get_by_id(workflow_id)
        if wf is None:
            raise NotFoundException(f"Workflow '{workflow_id}' not found")
        return wf

    @staticmethod
    def _to_response(w: WorkflowModel) -> WorkflowResponse:
        return WorkflowResponse(
            id=str(w.id),
            name=w.name,
            description=w.description,
            workflow_type=w.workflow_type,
            status=WorkflowStatus(w.status),
            current_step=w.current_step,
            steps=w.steps,
            input_data=w.input_data,
            output_data=w.output_data,
            error_message=w.error_message,
            created_at=w.created_at,
            updated_at=w.updated_at,
        )
