"""Channel router (CH1): single normalized entrypoint for inbound messages.

All connectors (web chat, WhatsApp, email, CRM, ...) post their native payload
as a canonical :class:`ChannelMessage` here. Edge guardrails (O4) run before the
message is forwarded: rate limiting, PII redaction, sanitization, and content
filtering. Every outcome is recorded into ``channel_events`` for the dashboard
widget via :class:`ChannelEventRecorder`.
"""

import time
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.config import settings
from gateway.database.session import async_session, get_db
from gateway.services.channel_events import ChannelEventRecorder, ChannelEventsService
from gateway.services.channel_service import ChannelService
from gateway.services.guardrails import get_guardrails_service, get_rate_limiter
from shared.guardrails import GuardrailsService, RedisRateLimiter
from shared.schemas.channels import ChannelMessage, ChannelResponse, ChannelType

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/channels", tags=["Channels"])

SUPPORTED_CHANNELS = {c.value for c in ChannelType}


def get_channel_service() -> ChannelService:
    return ChannelService()


def get_channel_event_recorder() -> ChannelEventRecorder:
    return ChannelEventRecorder(async_session, enabled=settings.CHANNEL_EVENTS_ENABLED)


def _scope_for(request: Request, message: ChannelMessage) -> str:
    if message.sender and (message.sender.user_id or message.sender.external_id):
        return f"user:{message.sender.user_id or message.sender.external_id}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.post("/{channel}", response_model=ChannelResponse, summary="Ingest a message for a channel")
async def inbound_message(
    channel: str,
    message: ChannelMessage,
    request: Request,
    service: ChannelService = Depends(get_channel_service),
    guardrails: GuardrailsService = Depends(get_guardrails_service),
    rate_limiter: RedisRateLimiter = Depends(get_rate_limiter),
    recorder: ChannelEventRecorder = Depends(get_channel_event_recorder),
) -> ChannelResponse:
    if channel not in SUPPORTED_CHANNELS:
        raise HTTPException(status_code=404, detail=f"Unsupported channel: {channel}")
    message.channel = ChannelType(channel)
    scope = _scope_for(request, message)
    started = time.perf_counter()

    if not await rate_limiter.allowed(scope):
        await recorder.record(
            channel=message.channel.value,
            scope=scope,
            status="rate_limited",
            message_id=message.message_id,
        )
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    result = guardrails.apply(message.text)
    if not result.allowed:
        await recorder.record(
            channel=message.channel.value,
            scope=scope,
            status="blocked",
            violation_category=result.violation.category,
            reason=result.violation.reason,
            redactions_count=len(result.redactions),
            message_id=message.message_id,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Message blocked ({result.violation.category}): {result.violation.reason}",
        )
    if result.text != message.text:
        logger.info(
            "message_guardrailed",
            channel=message.channel.value,
            redactions=len(result.redactions),
        )
    message.text = result.text

    response = await service.process(message)
    if isinstance(response, dict):
        request_id = response.get("request_id")
        message_id = response.get("message_id")
    else:
        request_id = response.request_id
        message_id = response.message_id
    await recorder.record(
        channel=message.channel.value,
        scope=scope,
        status="accepted",
        redactions_count=len(result.redactions),
        request_id=request_id,
        message_id=message_id,
        duration_ms=(time.perf_counter() - started) * 1000,
    )
    return response


@router.get("/stats", summary="Channel traffic summary for the dashboard")
async def channel_stats(
    start: str | None = None,
    end: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        service = ChannelEventsService(db)
        return await service.summary(_parse_dt(start), _parse_dt(end))
    except Exception as exc:
        logger.error("channel_stats_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Channel events service unavailable") from None


@router.get("/events", summary="Recent channel events for the dashboard")
async def channel_events(
    limit: int = 50,
    start: str | None = None,
    end: str | None = None,
    channel: str | None = None,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    limit = max(1, min(limit, 200))
    try:
        service = ChannelEventsService(db)
        return await service.events(
            limit,
            _parse_dt(start),
            _parse_dt(end),
            channel,
            status,
        )
    except Exception as exc:
        logger.error("channel_events_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Channel events service unavailable") from None
