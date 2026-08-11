from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from orchestrator.database.session import get_db
from orchestrator.services.usage_service import UsageService
from shared.usage.pricing import get_pricing

router = APIRouter(prefix="/api/usage", tags=["Usage"])
logger = structlog.get_logger(__name__)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.get("/summary")
async def usage_summary(
    start: str | None = None,
    end: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        service = UsageService(db)
        return await service.summary(_parse_dt(start), _parse_dt(end))
    except Exception as exc:
        logger.error("usage_summary_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Usage service unavailable")


@router.get("/trend")
async def usage_trend(
    start: str | None = None,
    end: str | None = None,
    granularity: str = "day",
    db: AsyncSession = Depends(get_db),
):
    try:
        service = UsageService(db)
        return await service.trend(_parse_dt(start), _parse_dt(end), granularity)
    except Exception as exc:
        logger.error("usage_trend_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Usage service unavailable")


@router.get("/events")
async def usage_events(
    limit: int = 50,
    start: str | None = None,
    end: str | None = None,
    service: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    limit = max(1, min(limit, 200))
    try:
        usage = UsageService(db)
        return await usage.events(limit, _parse_dt(start), _parse_dt(end), service)
    except Exception as exc:
        logger.error("usage_events_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Usage service unavailable")


@router.get("/pricing")
async def usage_pricing():
    return get_pricing()
