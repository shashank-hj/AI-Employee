"""Production calendar service.

Orchestrates the active :class:`CalendarProvider`, PostgreSQL persistence, and
the confirm-before-booking proposal flow. Replaces the mock implementation in
the production path.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from orchestrator.config import settings
from orchestrator.models.calendar_meeting import CalendarMeeting
from orchestrator.services.calendar.base import (
    CalendarEventDraft,
    _to_local,
    slot_to_dict,
)
from orchestrator.services.calendar.ics_provider import IcsEmailProvider
from orchestrator.services.calendar.google_provider import GoogleCalendarProvider
from orchestrator.services.calendar.repository import (
    CalendarMeetingRepository,
    PendingBookingRepository,
)
from orchestrator.services.interfaces import CalendarServiceProtocol

logger = structlog.get_logger(__name__)


def resolve_calendar_provider(
    provider_mode: str = "auto",
) -> GoogleCalendarProvider | IcsEmailProvider:
    """Pick the production provider.

    ``auto`` → Google when valid OAuth credentials are configured, otherwise
    fall back to the ICS email provider. ``google`` / ``ics`` force a backend.
    """
    google = GoogleCalendarProvider()
    if provider_mode == "google" and not google.enabled:
        raise ValueError("CALENDAR_PROVIDER=google but Google OAuth credentials are not configured")
    if provider_mode == "google" or (provider_mode == "auto" and google.enabled):
        return google
    return IcsEmailProvider()


class CalendarService:
    def __init__(
        self,
        provider: GoogleCalendarProvider | IcsEmailProvider | None = None,
        repository: CalendarMeetingRepository | None = None,
        pending_repository: PendingBookingRepository | None = None,
        timezone: str | None = None,
        duplicate_window_minutes: int | None = None,
    ) -> None:
        self._provider = provider if provider is not None else resolve_calendar_provider()
        self._repo = repository or CalendarMeetingRepository()
        self._pending = pending_repository or PendingBookingRepository()
        self._timezone = timezone or settings.CALENDAR_TIMEZONE
        self._duplicate_window = (
            duplicate_window_minutes
            if duplicate_window_minutes is not None
            else settings.CALENDAR_DUPLICATE_WINDOW_MINUTES
        )

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def enabled(self) -> bool:
        return self._provider.enabled

    async def health(self) -> dict[str, Any]:
        healthy = await self._provider.health_check()
        return {
            "provider": self._provider.name,
            "enabled": self.enabled,
            "healthy": healthy,
            "timezone": self._timezone,
        }

    # ── Availability + proposal flow ──

    async def check_availability(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        timezone: str | None = None,
        duration_minutes: int = 30,
    ) -> dict[str, Any]:
        tz = timezone or self._timezone
        slots = await self._provider.get_availability(start_at, end_at, tz, duration_minutes)
        requested_available = any(s.start == start_at for s in slots) or bool(slots)
        return {
            "requested_available": requested_available,
            "slots": [slot_to_dict(s, tz) for s in slots],
            "provider": self._provider.name,
            "timezone": tz,
        }

    async def propose_booking(
        self,
        *,
        session_id: str,
        user_id: str | None,
        draft: CalendarEventDraft,
        duration_minutes: int = 30,
    ) -> dict[str, Any]:
        tz = draft.timezone or self._timezone
        slots = await self._provider.get_availability(
            draft.start_at, draft.end_at, tz, duration_minutes
        )
        free = self._slot_free(slots, draft.start_at, draft.end_at)
        proposal = None
        if free or self._provider.name == "ics":
            proposal = await self._pending.upsert(
                session_id=session_id,
                user_id=user_id,
                title=draft.title,
                start_at=draft.start_at,
                end_at=draft.end_at,
                timezone=tz,
                attendees=draft.attendees,
                description=draft.description,
            )
        else:
            slots = await self._alternative_slots(
                draft.start_at, tz, duration_minutes
            )
        return {
            "proposed": proposal is not None,
            "proposal_id": proposal.id if proposal else None,
            "available": free,
            "slots": [slot_to_dict(s, tz) for s in slots],
            "provider": self._provider.name,
            "timezone": tz,
        }

    async def _alternative_slots(
        self,
        requested_start: datetime,
        timezone: str,
        duration_minutes: int = 30,
    ) -> list[Any]:
        """Find free slots on the same day as ``requested_start`` when the exact
        requested window is unavailable, so the agent can suggest alternatives."""
        try:
            local = requested_start.astimezone(timezone)
        except (TypeError, ValueError):
            local = requested_start
        day_start = local.replace(hour=9, minute=0, second=0, microsecond=0)
        day_end = local.replace(hour=18, minute=0, second=0, microsecond=0)
        return await self._provider.get_availability(
            day_start, day_end, timezone, duration_minutes
        )

    async def confirm_booking(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        proposal = await self._pending.get_active(session_id)
        if proposal is None:
            return {"success": False, "error": "no_pending_booking"}
        draft = CalendarEventDraft(
            title=proposal.title,
            start_at=proposal.start_at,
            end_at=proposal.end_at,
            timezone=proposal.timezone,
            attendees=list(proposal.attendees or []),
            description=proposal.description,
        )
        result = await self._create_meeting_internal(draft, session_id, user_id)
        await self._pending.clear(session_id)
        return result

    async def decline_booking(self, *, session_id: str) -> bool:
        await self._pending.clear(session_id)
        return True

    async def has_pending_booking(self, session_id: str) -> bool:
        return await self._pending.get_active(session_id) is not None

    # ── Meeting lifecycle ──

    async def create_meeting(
        self,
        draft: CalendarEventDraft,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._create_meeting_internal(draft, session_id, user_id)

    async def _create_meeting_internal(
        self,
        draft: CalendarEventDraft,
        session_id: str | None,
        user_id: str | None,
    ) -> dict[str, Any]:
        duplicate = await self._repo.find_duplicate(
            draft.start_at, draft.attendees, self._duplicate_window
        )
        if duplicate is not None:
            logger.info(
                "calendar_duplicate_prevented",
                meeting_id=duplicate.id,
                title=draft.title,
            )
            return {
                "success": False,
                "error": "duplicate_meeting",
                "meeting": _meeting_to_dict(duplicate),
            }

        try:
            created = await self._provider.create_event(draft)
        except Exception as exc:
            logger.error("calendar_create_failed", error=str(exc), title=draft.title)
            return {"success": False, "error": f"Calendar error: {str(exc)}"}

        meeting = CalendarMeeting(
            session_id=session_id,
            user_id=user_id,
            title=draft.title,
            description=draft.description,
            start_at=draft.start_at,
            end_at=draft.end_at,
            timezone=draft.timezone or self._timezone,
            attendees=draft.attendees,
            status="scheduled",
            provider=self._provider.name,
            provider_event_id=created.get("provider_event_id"),
            provider_link=created.get("link"),
            calendar_id=settings.GOOGLE_CALENDAR_ID if self._provider.name == "google" else None,
        )
        await self._repo.create(meeting)
        logger.info(
            "calendar_meeting_created",
            meeting_id=meeting.id,
            provider=self._provider.name,
            title=draft.title,
        )
        return {"success": True, "meeting": _meeting_to_dict(meeting)}

    async def get_meeting(self, meeting_id: str) -> dict[str, Any]:
        meeting = await self._repo.get(meeting_id)
        if meeting is None:
            return {"success": False, "error": "not_found"}
        return {"success": True, "meeting": _meeting_to_dict(meeting)}

    async def list_meetings(
        self,
        *,
        session_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = "scheduled",
        limit: int = 50,
    ) -> dict[str, Any]:
        meetings = await self._repo.list_meetings(
            session_id=session_id,
            start=start,
            end=end,
            status=status,
            limit=limit,
        )
        return {
            "success": True,
            "meetings": [_meeting_to_dict(m) for m in meetings],
            "provider": self._provider.name,
        }

    async def update_meeting(
        self,
        meeting_id: str,
        draft: CalendarEventDraft,
    ) -> dict[str, Any]:
        meeting = await self._repo.get(meeting_id)
        if meeting is None:
            return {"success": False, "error": "not_found"}
        if meeting.status == "cancelled":
            return {"success": False, "error": "meeting_cancelled"}

        try:
            updated = await self._provider.update_event(
                meeting.provider_event_id or "", draft
            )
        except Exception as exc:
            logger.error("calendar_update_failed", error=str(exc), meeting_id=meeting_id)
            return {"success": False, "error": f"Calendar error: {str(exc)}"}

        await self._repo.update(
            meeting_id,
            title=draft.title,
            description=draft.description,
            start_at=draft.start_at,
            end_at=draft.end_at,
            timezone=draft.timezone or meeting.timezone,
            attendees=draft.attendees,
            provider_event_id=updated.get("provider_event_id") or meeting.provider_event_id,
            provider_link=updated.get("link") or meeting.provider_link,
        )
        refreshed = await self._repo.get(meeting_id)
        logger.info(
            "calendar_meeting_updated",
            meeting_id=meeting_id,
            provider=self._provider.name,
        )
        return {"success": True, "meeting": _meeting_to_dict(refreshed) if refreshed else {}}

    async def cancel_meeting(self, meeting_id: str) -> dict[str, Any]:
        meeting = await self._repo.get(meeting_id)
        if meeting is None:
            return {"success": False, "error": "not_found"}

        try:
            await self._provider.cancel_event(meeting.provider_event_id or "")
        except Exception as exc:
            logger.error("calendar_cancel_failed", error=str(exc), meeting_id=meeting_id)
            return {"success": False, "error": f"Calendar error: {str(exc)}"}

        await self._repo.cancel(meeting_id)
        refreshed = await self._repo.get(meeting_id)
        logger.info("calendar_meeting_cancelled", meeting_id=meeting_id)
        return {"success": True, "meeting": _meeting_to_dict(refreshed) if refreshed else {}}

    # ── Matching for natural-language cancel/update ──

    async def match_meetings(
        self,
        *,
        session_id: str | None = None,
        meeting_id: str | None = None,
        ref_date: datetime | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        if meeting_id:
            meeting = await self._repo.get(meeting_id)
            if meeting is None:
                return {"success": False, "error": "not_found", "matches": []}
            return {"success": True, "matches": [_meeting_to_dict(meeting)]}

        start = ref_date - timedelta(days=1) if ref_date else datetime.now(timezone.utc) - timedelta(days=7)
        end = ref_date + timedelta(days=2) if ref_date else None
        meetings = await self._repo.list_meetings(
            session_id=session_id,
            start=start,
            end=end,
            status="scheduled",
            limit=limit,
        )
        return {"success": True, "matches": [_meeting_to_dict(m) for m in meetings]}

    def _slot_free(self, slots: list[Any], start: datetime, end: datetime) -> bool:
        for slot in slots:
            if getattr(slot, "available", True) and slot.start == start:
                return True
        return bool(slots)


def _meeting_to_dict(meeting: CalendarMeeting) -> dict[str, Any]:
    tz = meeting.timezone or settings.CALENDAR_TIMEZONE
    return {
        "id": meeting.id,
        "title": meeting.title,
        "description": meeting.description,
        "start_at": _to_local(meeting.start_at, tz).isoformat() if meeting.start_at else None,
        "end_at": _to_local(meeting.end_at, tz).isoformat() if meeting.end_at else None,
        "timezone": meeting.timezone,
        "attendees": list(meeting.attendees or []),
        "status": meeting.status,
        "provider": meeting.provider,
        "provider_event_id": meeting.provider_event_id,
        "link": meeting.provider_link,
        "session_id": meeting.session_id,
        "created_at": meeting.created_at.isoformat() if meeting.created_at else None,
        "updated_at": meeting.updated_at.isoformat() if meeting.updated_at else None,
    }


CalendarServiceProtocol.register(CalendarService)  # type: ignore[attr-defined]