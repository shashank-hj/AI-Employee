"""Channel event recording and aggregation for the gateway.

Two pieces:

- :class:`ChannelEventRecorder` persists one :class:`ChannelEvent` row per inbound
  channel-message outcome. Like :class:`shared.usage.UsageRecorder`, failures are
  logged and swallowed — recording must never break the message path.
- :class:`ChannelEventsService` runs the aggregation queries behind the
  ``GET /api/channels/stats`` and ``GET /api/channels/events`` endpoints that the
  dashboard widget consumes.
"""

from datetime import datetime

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models import ChannelEvent

logger = structlog.get_logger(__name__)


class ChannelEventRecorder:
    def __init__(self, session_factory, enabled: bool = True) -> None:
        self._session_factory = session_factory
        self._enabled = enabled

    async def record(
        self,
        channel: str,
        scope: str,
        status: str,
        violation_category: str | None = None,
        reason: str | None = None,
        redactions_count: int = 0,
        request_id: str | None = None,
        message_id: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        if not self._enabled:
            return
        try:
            async with self._session_factory() as session:
                session.add(
                    ChannelEvent(
                        channel=channel,
                        scope=scope,
                        status=status,
                        violation_category=violation_category,
                        reason=reason,
                        redactions_count=redactions_count,
                        request_id=request_id,
                        message_id=message_id,
                        duration_ms=duration_ms,
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.warning(
                "channel_event_record_failed",
                error=str(exc)[:200],
                channel=channel,
                status=status,
            )


def _range_conditions(start: datetime | None, end: datetime | None) -> list:
    conds = []
    if start:
        conds.append(ChannelEvent.recorded_at >= start)
    if end:
        conds.append(ChannelEvent.recorded_at <= end)
    return conds


class ChannelEventsService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def summary(self, start: datetime | None = None, end: datetime | None = None) -> dict:
        conds = _range_conditions(start, end)

        statuses = (
            await self._db.execute(
                select(
                    ChannelEvent.status,
                    func.count(ChannelEvent.id).label("count"),
                    func.coalesce(func.sum(ChannelEvent.redactions_count), 0).label("redactions"),
                )
                .where(*conds)
                .group_by(ChannelEvent.status)
            )
        ).all()

        by_channel = (
            await self._db.execute(
                select(
                    ChannelEvent.channel,
                    func.count(ChannelEvent.id).label("count"),
                )
                .where(*conds)
                .group_by(ChannelEvent.channel)
                .order_by(func.count(ChannelEvent.id).desc())
            )
        ).all()

        totals = {
            "accepted": 0,
            "blocked": 0,
            "rate_limited": 0,
            "redactions": 0,
            "calls": 0,
        }
        for row in statuses:
            count = int(row.count or 0)
            if row.status in totals:
                totals[row.status] = count
            totals["calls"] += count
            totals["redactions"] += int(row.redactions or 0)

        return {
            "totals": totals,
            "by_channel": [
                {"channel": row.channel, "calls": int(row.count or 0)} for row in by_channel
            ],
        }

    async def events(
        self,
        limit: int = 50,
        start: datetime | None = None,
        end: datetime | None = None,
        channel: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        conds = _range_conditions(start, end)
        if channel:
            conds.append(ChannelEvent.channel == channel)
        if status:
            conds.append(ChannelEvent.status == status)
        rows = (
            await self._db.execute(
                select(ChannelEvent)
                .where(*conds)
                .order_by(ChannelEvent.recorded_at.desc())
                .limit(limit)
            )
        ).scalars().all()
        return [_serialize_event(e) for e in rows]


def _serialize_event(event: ChannelEvent) -> dict:
    return {
        "id": event.id,
        "recorded_at": event.recorded_at.isoformat() if event.recorded_at else None,
        "channel": event.channel,
        "scope": event.scope,
        "status": event.status,
        "violation_category": event.violation_category,
        "reason": event.reason,
        "redactions_count": event.redactions_count,
        "request_id": event.request_id,
        "message_id": event.message_id,
        "duration_ms": event.duration_ms,
    }
