"""ICS email provider (fallback backend).

Used automatically when Google Calendar OAuth credentials are not configured.
Builds an ``.ics`` invitation and sends it to attendees through the existing
Gmail SMTP integration. There is no server-side calendar to check, so
availability treats requested slots as open.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import icalendar
import structlog

from orchestrator.config import settings
from orchestrator.services.calendar.base import CalendarEventDraft, CalendarSlot
from orchestrator.services.gmail_client import EmailClient

logger = structlog.get_logger(__name__)


class IcsEmailProvider:
    name = "ics"

    def __init__(
        self,
        email_client: EmailClient | None = None,
        ics_invites_enabled: bool | None = None,
    ) -> None:
        self._email = email_client if email_client is not None else EmailClient()
        self._ics_enabled = (
            settings.CALENDAR_ICS_INVITES_ENABLED if ics_invites_enabled is None else ics_invites_enabled
        )

    @property
    def enabled(self) -> bool:
        return bool(self._email.enabled and self._ics_enabled)

    async def health_check(self) -> bool:
        return self.enabled

    async def get_availability(
        self,
        start: datetime,
        end: datetime,
        timezone: str,
        duration_minutes: int = 30,
    ) -> list[CalendarSlot]:
        return [CalendarSlot(start=start, end=end, available=True)]

    async def create_event(self, draft: CalendarEventDraft) -> dict[str, Any]:
        event_id = f"ics-{uuid.uuid4().hex[:12]}"
        await asyncio.to_thread(_send_invite, self._email, draft, event_id)
        logger.info(
            "calendar_event_created",
            provider="ics",
            title=draft.title,
            event_id=event_id,
            attendees=draft.attendees,
        )
        return {"provider_event_id": event_id, "link": None}

    async def update_event(
        self, provider_event_id: str, draft: CalendarEventDraft
    ) -> dict[str, Any]:
        await asyncio.to_thread(_send_invite, self._email, draft, provider_event_id, True)
        logger.info("calendar_event_updated", provider="ics", event_id=provider_event_id)
        return {"provider_event_id": provider_event_id, "link": None}

    async def cancel_event(self, provider_event_id: str) -> None:
        logger.info("calendar_event_cancelled", provider="ics", event_id=provider_event_id)

    async def list_events(
        self, start: datetime, end: datetime, timezone: str
    ) -> list[dict[str, Any]]:
        return []


def _build_ics(draft: CalendarEventDraft, event_id: str, method: str = "REQUEST") -> bytes:
    cal = icalendar.Calendar()
    cal.add("prodid", "-//AI Employee//Calendar//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("method", method)

    event = icalendar.Event()
    event.add("uid", f"{event_id}@ai-employee")
    event.add("summary", draft.title)
    if draft.description:
        event.add("description", draft.description)
    event.add("dtstart", draft.start_at)
    event.add("dtend", draft.end_at)
    event.add("dtstamp", datetime.now(timezone.utc))
    organizer = draft.attendees[0] if draft.attendees else settings.EMAIL_ADDRESS
    event.add("organizer", f"mailto:{organizer}")
    for attendee in draft.attendees:
        event.add("attendee", f"mailto:{attendee}")
    cal.add_component(event)
    return cal.to_ical()


def _send_invite(
    email: EmailClient,
    draft: CalendarEventDraft,
    event_id: str,
    update: bool = False,
) -> None:
    subject = f"Updated invitation: {draft.title}" if update else f"Invitation: {draft.title}"
    body = (
        f"You have been invited to '{draft.title}'.\n\n"
        f"When: {_fmt_local(draft.start_at)} - {_fmt_local(draft.end_at)}\n"
        f"Timezone: {draft.timezone}\n\n"
        "A calendar invitation (.ics) is attached. Accept it to add the event to your calendar.\n"
    )
    to = draft.attendees[0] if draft.attendees else settings.EMAIL_ADDRESS
    ics_bytes = _build_ics(draft, event_id, method="REQUEST")
    email.send_message(
        to,
        subject,
        body,
        attachments=[("invite.ics", ics_bytes, "text/calendar")],
    )


def _fmt_local(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.strftime("%A, %d %b %Y %H:%M %Z")


def make_calendar_slots(start: datetime, end: datetime, duration_minutes: int = 30) -> list[CalendarSlot]:
    """Generate half-hour candidate slots in a business window for availability UIs."""
    slots: list[CalendarSlot] = []
    cursor = start
    step = timedelta(minutes=max(duration_minutes, 30))
    while cursor + timedelta(minutes=duration_minutes) <= end and len(slots) < 20:
        slots.append(
            CalendarSlot(
                start=cursor,
                end=cursor + timedelta(minutes=duration_minutes),
                available=True,
            )
        )
        cursor += step
    return slots