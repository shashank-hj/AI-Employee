"""Aggregation queries over the shared usage_events table for the usage dashboard."""

from datetime import datetime

import structlog
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.usage.model import UsageEvent

logger = structlog.get_logger(__name__)

LLM_CATEGORY = "llm"


def _float(value) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


class UsageService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def summary(self, start: datetime | None = None, end: datetime | None = None) -> dict:
        conds = _range_conditions(start, end)

        total_row = (
            await self._db.execute(
                select(
                    func.count(UsageEvent.id).label("calls"),
                    func.coalesce(func.sum(UsageEvent.cost_inr), 0).label("cost"),
                    func.coalesce(func.sum(UsageEvent.input_units), 0).label("input_units"),
                    func.coalesce(func.sum(UsageEvent.output_units), 0).label("output_units"),
                    func.coalesce(func.sum(UsageEvent.total_units), 0).label("total_units"),
                    func.coalesce(
                        func.sum(case((UsageEvent.status == "error", 1), else_=0)), 0
                    ).label("errors"),
                ).where(*conds)
            )
        ).one()

        llm_row = (
            await self._db.execute(
                select(
                    func.coalesce(func.sum(UsageEvent.input_units), 0).label("input_tokens"),
                    func.coalesce(func.sum(UsageEvent.output_units), 0).label("output_tokens"),
                    func.count(UsageEvent.id).label("calls"),
                ).where(*conds, UsageEvent.category == LLM_CATEGORY)
            )
        ).one()

        by_model = await self._model_breakdown(conds)
        by_operation = await self._operation_breakdown(conds)
        by_category = await self._category_breakdown(conds)

        calls = int(total_row.calls or 0)
        cost = _float(total_row.cost)
        avg_cost_per_call = round(cost / calls, 6) if calls else 0.0

        return {
            "totals": {
                "calls": calls,
                "cost": cost,
                "errors": int(total_row.errors or 0),
                "input_units": int(total_row.input_units or 0),
                "output_units": int(total_row.output_units or 0),
                "total_units": int(total_row.total_units or 0),
                "avg_cost_per_call": avg_cost_per_call,
                "llm_input_tokens": int(llm_row.input_tokens or 0),
                "llm_output_tokens": int(llm_row.output_tokens or 0),
                "llm_calls": int(llm_row.calls or 0),
            },
            "by_model": by_model,
            "by_operation": by_operation,
            "by_category": by_category,
        }

    async def trend(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        granularity: str = "day",
    ) -> list[dict]:
        bucket = "hour" if granularity == "hour" else "day"
        conds = _range_conditions(start, end)
        rows = (
            await self._db.execute(
                select(
                    func.date_trunc(bucket, UsageEvent.recorded_at).label("bucket"),
                    func.count(UsageEvent.id).label("calls"),
                    func.coalesce(func.sum(UsageEvent.cost_inr), 0).label("cost"),
                    func.coalesce(func.sum(UsageEvent.input_units), 0).label("input_units"),
                    func.coalesce(func.sum(UsageEvent.output_units), 0).label("output_units"),
                )
                .where(*conds)
                .group_by("bucket")
                .order_by("bucket")
            )
        ).all()
        return [
            {
                "bucket": str(row.bucket or ""),
                "calls": int(row.calls or 0),
                "cost": _float(row.cost),
                "input_units": int(row.input_units or 0),
                "output_units": int(row.output_units or 0),
            }
            for row in rows
        ]

    async def events(
        self,
        limit: int = 50,
        start: datetime | None = None,
        end: datetime | None = None,
        service: str | None = None,
    ) -> list[dict]:
        conds = _range_conditions(start, end)
        if service:
            conds.append(UsageEvent.service == service)
        rows = (
            await self._db.execute(
                select(UsageEvent)
                .where(*conds)
                .order_by(UsageEvent.recorded_at.desc())
                .limit(limit)
            )
        ).scalars().all()
        return [_serialize_event(e) for e in rows]

    async def _model_breakdown(self, conds) -> list[dict]:
        rows = (
            await self._db.execute(
                select(
                    UsageEvent.model,
                    func.count(UsageEvent.id).label("calls"),
                    func.coalesce(func.sum(UsageEvent.input_units), 0).label("input_units"),
                    func.coalesce(func.sum(UsageEvent.output_units), 0).label("output_units"),
                    func.coalesce(func.sum(UsageEvent.cost_inr), 0).label("cost"),
                )
                .where(*conds)
                .group_by(UsageEvent.model)
                .order_by(func.sum(UsageEvent.cost_inr).desc(), func.count(UsageEvent.id).desc())
            )
        ).all()
        return [
            {
                "model": row.model,
                "calls": int(row.calls or 0),
                "input_units": int(row.input_units or 0),
                "output_units": int(row.output_units or 0),
                "cost": _float(row.cost),
            }
            for row in rows
        ]

    async def _operation_breakdown(self, conds) -> list[dict]:
        rows = (
            await self._db.execute(
                select(
                    UsageEvent.operation,
                    UsageEvent.category,
                    func.count(UsageEvent.id).label("calls"),
                    func.coalesce(func.sum(UsageEvent.input_units), 0).label("input_units"),
                    func.coalesce(func.sum(UsageEvent.output_units), 0).label("output_units"),
                    func.coalesce(func.sum(UsageEvent.cost_inr), 0).label("cost"),
                )
                .where(*conds)
                .group_by(UsageEvent.operation, UsageEvent.category)
                .order_by(func.sum(UsageEvent.cost_inr).desc(), func.count(UsageEvent.id).desc())
            )
        ).all()
        return [
            {
                "operation": row.operation,
                "category": row.category,
                "calls": int(row.calls or 0),
                "input_units": int(row.input_units or 0),
                "output_units": int(row.output_units or 0),
                "cost": _float(row.cost),
            }
            for row in rows
        ]

    async def _category_breakdown(self, conds) -> list[dict]:
        rows = (
            await self._db.execute(
                select(
                    UsageEvent.category,
                    func.count(UsageEvent.id).label("calls"),
                    func.coalesce(func.sum(UsageEvent.cost_inr), 0).label("cost"),
                    func.coalesce(func.sum(UsageEvent.input_units), 0).label("input_units"),
                    func.coalesce(func.sum(UsageEvent.output_units), 0).label("output_units"),
                )
                .where(*conds)
                .group_by(UsageEvent.category)
                .order_by(func.sum(UsageEvent.cost_inr).desc(), func.count(UsageEvent.id).desc())
            )
        ).all()
        return [
            {
                "category": row.category,
                "calls": int(row.calls or 0),
                "input_units": int(row.input_units or 0),
                "output_units": int(row.output_units or 0),
                "cost": _float(row.cost),
            }
            for row in rows
        ]


def _range_conditions(start: datetime | None, end: datetime | None) -> list:
    conds = []
    if start:
        conds.append(UsageEvent.recorded_at >= start)
    if end:
        conds.append(UsageEvent.recorded_at <= end)
    return conds


def _serialize_event(event: UsageEvent) -> dict:
    return {
        "id": event.id,
        "recorded_at": event.recorded_at.isoformat() if event.recorded_at else None,
        "service": event.service,
        "category": event.category,
        "operation": event.operation,
        "model": event.model,
        "unit": event.unit,
        "input_units": event.input_units,
        "output_units": event.output_units,
        "total_units": event.total_units,
        "cost": _float(event.cost_inr),
        "request_id": event.request_id,
        "session_id": event.session_id,
        "user_id": event.user_id,
        "status": event.status,
        "error": event.error,
        "duration_ms": event.duration_ms,
    }

