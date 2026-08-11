import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import structlog
from langgraph.types import Command

from shared.utils.exceptions import NotFoundException
from workflow.graph.builder import build_workflow_graph
from workflow.graph.checkpointer import CheckpointEngine, get_checkpoint_engine
from workflow.graph.handlers import HANDLERS
from workflow.models.workflow import WorkflowModel
from workflow.repositories.workflow_repo import WorkflowRepository
from workflow.schemas.workflows import (
    WorkflowCreate,
    WorkflowHistoryEntry,
    WorkflowResponse,
    WorkflowRunResponse,
    WorkflowStatus,
)

logger = structlog.get_logger(__name__)

_TERMINAL = frozenset({
    WorkflowStatus.COMPLETED.value,
    WorkflowStatus.FAILED.value,
    WorkflowStatus.CANCELLED.value,
})

_DEFAULT_STEPS = [
    {
        "name": "validate", "type": "task", "handler": "echo",
        "params": {"stage": "validate"}, "next": "process",
    },
    {
        "name": "process", "type": "task", "handler": "echo",
        "params": {"stage": "process"}, "next": "complete",
    },
    {"name": "complete", "type": "task", "handler": "echo", "params": {"stage": "complete"}},
]


class WorkflowService:
    def __init__(
        self,
        repo: WorkflowRepository,
        handlers: dict | None = None,
        checkpointer: CheckpointEngine | None = None,
    ) -> None:
        self._repo = repo
        self._handlers = handlers or HANDLERS
        self._checkpointer = checkpointer or get_checkpoint_engine()
        self._graphs: dict[str, Any] = {}

    async def create_workflow(self, data: WorkflowCreate) -> WorkflowResponse:
        steps = data.steps or _DEFAULT_STEPS[:]
        self._validate_steps(steps)

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
        await self._repo.commit()
        logger.info("workflow_created", workflow_id=str(wf.id), name=wf.name, num_steps=len(steps))
        return self._to_response(wf)

    async def get_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        return self._to_response(wf)

    async def list_workflows(
        self, status: str | None = None, page: int = 1, page_size: int = 20
    ) -> tuple[list[WorkflowResponse], int]:
        workflows, total = await self._repo.list_all(status=status, page=page, page_size=page_size)
        return [self._to_response(w) for w in workflows], total

    async def pause_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        if wf.status != WorkflowStatus.RUNNING.value:
            raise ValueError(f"Cannot pause workflow in '{wf.status}' status")
        wf.status = WorkflowStatus.PAUSED.value
        wf = await self._repo.update(wf)
        await self._repo.commit()
        logger.info("workflow_paused", workflow_id=workflow_id)
        return self._to_response(wf)

    async def resume_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        if wf.status != WorkflowStatus.PAUSED.value:
            raise ValueError(f"Cannot resume workflow in '{wf.status}' status")
        wf.status = WorkflowStatus.RUNNING.value
        wf = await self._repo.update(wf)
        await self._repo.commit()
        logger.info("workflow_resumed", workflow_id=workflow_id)
        return self._to_response(wf)

    async def cancel_workflow(self, workflow_id: str) -> WorkflowResponse:
        wf = await self._get_or_raise(workflow_id)
        if wf.status not in (
            WorkflowStatus.PENDING.value,
            WorkflowStatus.RUNNING.value,
            WorkflowStatus.PAUSED.value,
        ):
            raise ValueError(f"Cannot cancel workflow in '{wf.status}' status")
        wf.status = WorkflowStatus.CANCELLED.value
        wf = await self._repo.update(wf)
        await self._repo.commit()
        logger.info("workflow_cancelled", workflow_id=workflow_id)
        return self._to_response(wf)

    async def get_history(self, workflow_id: str) -> list[WorkflowHistoryEntry]:
        wf = await self._get_or_raise(workflow_id)
        steps = wf.steps or []
        output_data = wf.output_data or {}
        history_raw = output_data.get("history", [])
        if history_raw:
            return [
                WorkflowHistoryEntry(
                    step_name=str(h.get("step", "")),
                    status=_status_from_run(str(h.get("status", "completed"))),
                    completed_at=datetime.now(UTC),
                    output_data=h.get("output"),
                )
                for h in history_raw
            ]
        # Fall back to static step list when the workflow has never run.
        return [
            WorkflowHistoryEntry(
                step_name=s["name"] if isinstance(s, dict) else str(s),
                status=WorkflowStatus.COMPLETED
                if wf.status in (WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value)
                else WorkflowStatus.PENDING,
                started_at=wf.created_at,
                completed_at=wf.updated_at,
            )
            for s in steps
        ]

    # ── Graph execution ────────────────────────────────────────────────────────

    async def run(
        self,
        workflow_id: str,
        input_data: dict | None,
        timeout_seconds: float | None = 300,
    ) -> WorkflowRunResponse:
        wf = await self._get_or_raise(workflow_id)
        await self._checkpointer.setup()
        await self._guard_new_run(wf)
        graph = await self._compile(wf)
        wf.status = WorkflowStatus.RUNNING.value
        await self._repo.update(wf)
        await self._repo.commit()

        config = {"configurable": {"thread_id": str(wf.id)}}
        initial: dict[str, Any] = {
            "workflow_id": str(wf.id),
            "input_data": input_data or {},
            "outputs": {},
            "history": [],
        }
        try:
            result = await asyncio.wait_for(
                graph.ainvoke(initial, config=config), timeout=timeout_seconds
            )
        except TimeoutError:
            await self._mark_failed(wf, f"Run timed out after {timeout_seconds}s")
            raise
        except Exception as exc:
            await self._mark_failed(wf, str(exc))
            raise

        return await self._finalize(wf, result)

    async def stream(
        self,
        workflow_id: str,
        input_data: dict | None,
        timeout_seconds: float | None = 300,
    ) -> AsyncIterator[dict[str, Any]]:
        wf = await self._get_or_raise(workflow_id)
        await self._checkpointer.setup()
        await self._guard_new_run(wf)
        graph = await self._compile(wf)
        wf.status = WorkflowStatus.RUNNING.value
        await self._repo.update(wf)
        await self._repo.commit()

        config = {"configurable": {"thread_id": str(wf.id)}}
        initial: dict[str, Any] = {
            "workflow_id": str(wf.id),
            "input_data": input_data or {},
            "outputs": {},
            "history": [],
        }
        try:
            iterator = graph.astream(initial, config=config, stream_mode="updates")
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(iterator), timeout=timeout_seconds)
                except StopAsyncIteration:
                    break
                yield {"type": "update", "data": chunk}
            snapshot = await graph.aget_state(config)
            result = dict(snapshot.values or {})
        except TimeoutError:
            await self._mark_failed(wf, f"Stream timed out after {timeout_seconds}s")
            raise
        except Exception as exc:
            await self._mark_failed(wf, str(exc))
            raise
        response = await self._finalize(wf, result)
        yield {"type": "done", "data": response.model_dump(mode="json")}

    async def resume(self, workflow_id: str, payload: dict) -> WorkflowRunResponse:
        wf = await self._get_or_raise(workflow_id)
        await self._checkpointer.setup()
        if wf.status != WorkflowStatus.PAUSED.value:
            raise ValueError(
                f"Cannot continue workflow in '{wf.status}' status; only paused workflows "
                "waiting for approval can be continued"
            )
        graph = await self._compile(wf)
        wf.status = WorkflowStatus.RUNNING.value
        await self._repo.update(wf)
        await self._repo.commit()

        config = {"configurable": {"thread_id": str(wf.id)}}
        try:
            result = await graph.ainvoke(Command(resume=payload), config=config)
        except Exception as exc:
            await self._mark_failed(wf, str(exc))
            raise
        return await self._finalize(wf, result)

    async def delete_workflow(self, workflow_id: str) -> None:
        deleted = await self._repo.delete(workflow_id)
        if not deleted:
            raise NotFoundException(f"Workflow '{workflow_id}' not found")
        await self._repo.commit()
        self._graphs.pop(str(workflow_id), None)
        try:
            await self._checkpointer.saver.adelete_thread(str(workflow_id))
        except Exception:  # noqa: BLE001
            logger.debug("checkpoint_thread_delete_failed", workflow_id=workflow_id)
        logger.info("workflow_deleted", workflow_id=workflow_id)

    async def _compile(self, wf: WorkflowModel) -> Any:
        key = str(wf.id)
        if key not in self._graphs:
            self._graphs[key] = build_workflow_graph(
                wf.steps or [], self._handlers, checkpointer=self._checkpointer.saver
            )
        return self._graphs[key]

    async def _finalize(self, wf: WorkflowModel, result: dict) -> WorkflowRunResponse:
        interrupted = bool(result.get("__interrupt__"))
        outputs = result.get("outputs", {})
        history = result.get("history", [])
        payload = self._interrupt_payload(result) if interrupted else None
        wf.output_data = {
            "outputs": outputs,
            "history": history,
            **({"pending_approval": payload} if payload else {}),
        }
        wf.current_step = self._current_step(result, interrupted)
        if interrupted:
            wf.status = WorkflowStatus.PAUSED.value
            wf.error_message = None
        else:
            wf.status = WorkflowStatus.COMPLETED.value
            wf.error_message = result.get("error")
        wf = await self._repo.update(wf)
        await self._repo.commit()
        logger.info(
            "workflow_run_finalized",
            workflow_id=str(wf.id),
            status=wf.status,
            interrupted=interrupted,
        )
        return WorkflowRunResponse(
            workflow=self._to_response(wf),
            interrupted=interrupted,
            current_step=wf.current_step,
            outputs=outputs,
            history=history,
        )

    @staticmethod
    def _interrupt_payload(result: dict) -> dict | None:
        interrupts = result.get("__interrupt__") or []
        if not interrupts:
            return None
        payload = interrupts[0]
        if hasattr(payload, "value"):
            payload = payload.value
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _current_step(cls, result: dict, interrupted: bool) -> str | None:
        if not interrupted:
            return result.get("current_step")
        payload = cls._interrupt_payload(result)
        if payload and payload.get("workflow_step"):
            return payload["workflow_step"]
        return result.get("current_step")

    async def _guard_new_run(self, wf: WorkflowModel) -> None:
        """Enforce run/stream state transitions and reset stale threads on re-run."""
        if wf.status == WorkflowStatus.RUNNING.value:
            raise ValueError("Workflow is already running")
        if wf.status == WorkflowStatus.PAUSED.value:
            raise ValueError(
                "Workflow is paused waiting for approval; use /continue to resume it"
            )
        if wf.status in _TERMINAL:
            await self._reset_thread(str(wf.id))

    async def _reset_thread(self, thread_id: str) -> None:
        try:
            await self._checkpointer.saver.adelete_thread(thread_id)
        except Exception:  # noqa: BLE001
            logger.debug("checkpoint_thread_reset_failed", thread_id=thread_id)

    async def _mark_failed(self, wf: WorkflowModel, error: str) -> None:
        wf.status = WorkflowStatus.FAILED.value
        wf.error_message = error[:2000]
        await self._repo.update(wf)
        await self._repo.commit()
        logger.error("workflow_run_failed", workflow_id=str(wf.id), error=error[:300])

    @staticmethod
    def _validate_steps(steps: list[dict]) -> None:
        if not steps:
            return
        names = [s.get("name") for s in steps]
        if any(not n for n in names):
            raise ValueError("Every workflow step must have a name")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate step names are not allowed")
        known = set(HANDLERS)
        for step in steps:
            step_type = step.get("type", "task")
            if step_type == "task":
                handler = step.get("handler")
                if handler not in known:
                    raise ValueError(f"Unknown handler '{handler}' for step '{step.get('name')}'")
            elif step_type == "fan_out":
                for handler in step.get("handlers", []):
                    if handler not in known:
                        raise ValueError(
                            f"Unknown handler '{handler}' in fan_out step '{step.get('name')}'"
                        )
            elif step_type not in ("branch",):
                raise ValueError(f"Unknown step type '{step_type}' for step '{step.get('name')}'")

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
            pending_approval=(w.output_data or {}).get("pending_approval"),
            created_at=w.created_at,
            updated_at=w.updated_at,
        )


def _status_from_run(status: str) -> WorkflowStatus:
    if status == "skipped":
        return WorkflowStatus.CANCELLED
    if status == "failed":
        return WorkflowStatus.FAILED
    return WorkflowStatus.COMPLETED


async def reconcile_stale_running(session: Any) -> int:
    """Mark workflows left in 'running' by a crash/restart as failed."""
    from sqlalchemy import update

    result = await session.execute(
        update(WorkflowModel)
        .where(WorkflowModel.status == WorkflowStatus.RUNNING.value)
        .values(
            status=WorkflowStatus.FAILED.value,
            error_message="Recovered after restart: execution was interrupted",
        )
    )
    await session.commit()
    return result.rowcount or 0
