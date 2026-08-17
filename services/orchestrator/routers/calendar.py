"""Calendar API routes backed by the CalendarService."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query

from orchestrator.config import settings
from orchestrator.container import get_calendar_service
from orchestrator.services.calendar.base import CalendarEventDraft
from shared.utils.response import success_response

router = APIRouter(prefix="/api/calendar", tags=["Calendar"])


def _get_service():
    return get_calendar_service()


def _parse_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid datetime: {value}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@router.get("/health", summary="Check Calendar integration status")
async def calendar_health():
    service = _get_service()
    health = await service.health()
    return {
        "enabled": service.enabled,
        "provider": health.get("provider"),
        "healthy": health.get("healthy"),
        "timezone": settings.CALENDAR_TIMEZONE,
    }


@router.get("/events", summary="List upcoming calendar events")
async def list_events(
    max_results: int = Query(default=10, ge=1, le=50),
    session_id: str | None = Query(default=None),
):
    service = _get_service()
    try:
        result = await service.list_meetings(
            session_id=session_id,
            start=datetime.now(timezone.utc),
            end=datetime.now(timezone.utc) + timedelta(days=30),
            status="scheduled",
            limit=max_results,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Calendar error: {str(exc)}")
    if not result.get("success"):
        raise HTTPException(status_code=502, detail=f"Calendar error: {result.get('error')}")
    events = result.get("meetings", [])
    return success_response(data={"events": events, "count": len(events)})


@router.post("/events", summary="Create a calendar event")
async def create_event(
    title: str = Query(...),
    start_at: str = Query(...),
    end_at: str | None = Query(default=None),
    attendees: str | None = Query(default=None),
    timezone: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
):
    service = _get_service()
    parsed_start = _parse_dt(start_at)
    parsed_end = _parse_dt(end_at) if end_at else parsed_start + timedelta(minutes=30)
    attendee_list = [a.strip() for a in attendees.split(",")] if attendees else []
    draft = CalendarEventDraft(
        title=title,
        start_at=parsed_start,
        end_at=parsed_end,
        timezone=timezone or settings.CALENDAR_TIMEZONE,
        attendees=attendee_list,
    )
    try:
        result = await service.create_meeting(draft, session_id=session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Calendar error: {str(exc)}")
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=f"Calendar error: {result.get('error')}")
    return success_response(data={"event": result["meeting"]})


@router.get("/events/{event_id}", summary="Get a calendar event by id")
async def get_event(event_id: str):
    service = _get_service()
    try:
        result = await service.get_meeting(event_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Calendar error: {str(exc)}")
    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Calendar event not found")
    return success_response(data={"event": result["meeting"]})


@router.patch("/events/{event_id}", summary="Update/reschedule a calendar event")
async def update_event(
    event_id: str,
    title: str | None = Query(default=None),
    start_at: str | None = Query(default=None),
    end_at: str | None = Query(default=None),
    timezone: str | None = Query(default=None),
):
    service = _get_service()
    try:
        existing = await service.get_meeting(event_id)
        if not existing.get("success"):
            raise HTTPException(status_code=404, detail="Calendar event not found")
        meeting = existing["meeting"]
        new_start = _parse_dt(start_at) if start_at else datetime.fromisoformat(meeting["start_at"])
        new_end = (
            _parse_dt(end_at)
            if end_at
            else new_start + timedelta(minutes=30)
            if start_at
            else datetime.fromisoformat(meeting["end_at"])
        )
        draft = CalendarEventDraft(
            title=title or meeting.get("title", "Meeting"),
            start_at=new_start,
            end_at=new_end,
            timezone=timezone or meeting.get("timezone") or settings.CALENDAR_TIMEZONE,
            attendees=meeting.get("attendees", []),
        )
        result = await service.update_meeting(event_id, draft)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Calendar error: {str(exc)}")
    if not result.get("success"):
        raise HTTPException(status_code=409, detail=f"Calendar error: {result.get('error')}")
    return success_response(data={"event": result["meeting"]})


@router.delete("/events/{event_id}", summary="Cancel/delete a calendar event")
async def delete_event(event_id: str):
    service = _get_service()
    try:
        result = await service.cancel_meeting(event_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Calendar error: {str(exc)}")
    if not result.get("success"):
        raise HTTPException(status_code=404, detail="Calendar event not found")
    return success_response(data={"status": "cancelled", "event_id": event_id})
