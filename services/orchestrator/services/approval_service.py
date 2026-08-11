"""Human-in-the-loop approval gate (C4).

Tools listed in ``HITL_APPROVAL_TOOLS`` pause execution until a human approves
via the existing :class:`HumanTaskService` + ``/api/human-tasks`` endpoints.
Approval correlation uses a deterministic ``request_id`` derived from
session + tool + parameters, so once the human resolves the task the next agent
turn can proceed.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import structlog

from orchestrator.config import settings
from orchestrator.schemas.human_task import HumanTaskCreate, HumanTaskResolve
from orchestrator.services.human_task_service import HumanTaskService

logger = structlog.get_logger(__name__)

APPROVAL_PREFIX = "approval:"


@dataclass
class ApprovalDecision:
    tool_name: str
    required: bool
    approved: bool
    task_id: str | None = None
    message: str = ""


class ApprovalService:
    def __init__(
        self,
        approval_tools: list[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._approval_tools = set(approval_tools or _parse_approval_tools())
        self._enabled = settings.HITL_ENABLED if enabled is None else enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def requires_approval(self, tool_name: str) -> bool:
        return self._enabled and tool_name in self._approval_tools

    @staticmethod
    def _correlation_id(
        session_id: str | None,
        user_id: str | None,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> str:
        canonical = json.dumps(parameters, sort_keys=True, default=str)
        raw = f"{session_id or ''}|{user_id or ''}|{tool_name}|{canonical}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{APPROVAL_PREFIX}{tool_name}:{digest}"

    async def check_or_request(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        user_input: str,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ApprovalDecision:
        if not self.requires_approval(tool_name):
            return ApprovalDecision(tool_name=tool_name, required=False, approved=True)

        correlation_id = self._correlation_id(session_id, user_id, tool_name, parameters)

        async with _session() as db:
            service = HumanTaskService(db)
            approved = await _find_resolved(db, correlation_id)
            if approved:
                logger.info(
                    "approval_granted",
                    tool_name=tool_name,
                    task_id=approved,
                    session_id=session_id,
                )
                return ApprovalDecision(
                    tool_name=tool_name,
                    required=True,
                    approved=True,
                    task_id=approved,
                    message="Approval already granted.",
                )

            pending = await _find_pending(db, correlation_id)
            if pending is not None:
                logger.info(
                    "approval_already_pending",
                    tool_name=tool_name,
                    task_id=pending,
                    session_id=session_id,
                )
                return ApprovalDecision(
                    tool_name=tool_name,
                    required=True,
                    approved=False,
                    task_id=pending,
                    message="Approval is already pending for this action.",
                )

            task = await service.create(HumanTaskCreate(
                user_input=user_input[:1000],
                intent="tool_approval",
                reason=f"Approval required to execute tool '{tool_name}'",
                priority="HIGH",
                context={
                    "approval": True,
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "session_id": session_id,
                    "user_id": user_id,
                },
                request_id=correlation_id,
            ))
            logger.info(
                "approval_task_created",
                tool_name=tool_name,
                task_id=task.id,
                session_id=session_id,
            )
            return ApprovalDecision(
                tool_name=tool_name,
                required=True,
                approved=False,
                task_id=task.id,
                message="This action requires approval. A request has been sent to your team.",
            )

    async def approve(self, task_id: str, note: str | None = None) -> dict[str, Any] | None:
        async with _session() as db:
            service = HumanTaskService(db)
            task = await service.claim(task_id, assigned_to="operator")
            if task is None:
                return None
            resolved = await service.resolve(
                task_id,
                HumanTaskResolve(resolution_note=note or "Approved by operator"),
            )
            return {"approved": True, "task_id": task_id} if resolved else None


async def _find_resolved(db, correlation_id: str) -> str | None:
    from sqlalchemy import select

    from orchestrator.models.human_task import HumanTask

    result = await db.execute(
        select(HumanTask.id)
        .where(HumanTask.request_id == correlation_id)
        .where(HumanTask.status == "resolved")
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _find_pending(db, correlation_id: str) -> str | None:
    from sqlalchemy import select

    from orchestrator.models.human_task import HumanTask

    result = await db.execute(
        select(HumanTask.id)
        .where(HumanTask.request_id == correlation_id)
        .where(HumanTask.status.in_(("pending", "in_progress")))
        .limit(1)
    )
    return result.scalar_one_or_none()


def _session():
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from orchestrator.database.session import engine

    maker = async_sessionmaker(engine, expire_on_commit=False)
    return maker()


def _parse_approval_tools() -> list[str]:
    try:
        return json.loads(settings.HITL_APPROVAL_TOOLS)
    except (json.JSONDecodeError, TypeError):
        return ["send_email"]
