"""Async recorder that persists UsageRecords into the shared usage_events table.

Each service passes its own async session factory (from its ``database.session``
module) plus its service name, so this module stays decoupled from any one engine.
Recording failures are logged and swallowed — they must never break the caller
(the LLM / speech / embedding request path).
"""

import structlog

from shared.usage.context import get_usage_context
from shared.usage.model import UsageEvent
from shared.usage.pricing import compute_cost
from shared.usage.records import UsageRecord

logger = structlog.get_logger(__name__)


class UsageRecorder:
    def __init__(
        self,
        session_factory,
        service: str,
        enabled: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._service = service
        self._enabled = enabled

    async def __call__(self, record: UsageRecord) -> None:
        """Callable interface so the recorder can be passed directly as a usage hook."""
        if not self._enabled:
            return
        try:
            await self.record(record)
        except Exception as exc:
            logger.warning(
                "usage_record_failed",
                error=str(exc),
                service=record.service,
                category=record.category,
                operation=record.operation,
            )

    async def record(self, record: UsageRecord) -> None:
        if not self._enabled:
            return

        ctx = get_usage_context()

        if not record.service:
            record.service = self._service
        if not record.request_id:
            record.request_id = ctx.get("request_id")
        if not record.session_id:
            record.session_id = ctx.get("session_id")
        if not record.user_id:
            record.user_id = ctx.get("user_id")
        if ctx.get("operation"):
            record.operation = ctx["operation"]

        if record.cost_inr is None:
            record.cost_inr = compute_cost(record)

        async with self._session_factory() as session:
            session.add(UsageEvent.from_record(record))
            await session.commit()

    async def record_error(self, **kwargs) -> None:
        record = UsageRecord(**kwargs)
        record.status = "error"
        await self.record(record)
