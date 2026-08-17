"""Google Calendar API provider (primary production backend).

Authenticates with an OAuth2 refresh token (see ``scripts/google_calendar_auth.py``
to obtain one). Uses the ``calendar`` scope and writes to the configured calendar.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from orchestrator.config import settings
from orchestrator.services.calendar.base import CalendarEventDraft, CalendarSlot

logger = structlog.get_logger(__name__)

TOKEN_URI = "https://oauth2.googleapis.com/token"


class GoogleCalendarProvider:
    name = "google"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        calendar_id: str | None = None,
        scopes: list[str] | None = None,
    ) -> None:
        self._client_id = client_id if client_id is not None else settings.GOOGLE_CLIENT_ID
        self._client_secret = (
            client_secret if client_secret is not None else settings.GOOGLE_CLIENT_SECRET
        )
        self._refresh_token = (
            refresh_token if refresh_token is not None else settings.GOOGLE_REFRESH_TOKEN
        )
        self._calendar_id = calendar_id or settings.GOOGLE_CALENDAR_ID or "primary"
        self._scopes = scopes or [s for s in settings.GOOGLE_CALENDAR_SCOPES.split() if s]

    @property
    def enabled(self) -> bool:
        return bool(self._client_id and self._client_secret and self._refresh_token)

    def _service(self) -> Any:
        creds = Credentials(
            token=None,
            refresh_token=self._refresh_token,
            token_uri=TOKEN_URI,
            client_id=self._client_id,
            client_secret=self._client_secret,
            scopes=self._scopes,
        )
        creds.refresh(Request())
        return build("calendar", "v3", credentials=creds, cache_discovery=False)

    async def health_check(self) -> bool:
        if not self.enabled:
            return False
        try:
            service = await asyncio.to_thread(self._service)
            _list = await asyncio.to_thread(
                lambda: service.calendarList().get(calendarId=self._calendar_id).execute()
            )
            return bool(_list)
        except Exception as exc:
            logger.warning("calendar_health_failed", error=str(exc))
            return False

    async def get_availability(
        self,
        start: datetime,
        end: datetime,
        timezone: str,
        duration_minutes: int = 30,
    ) -> list[CalendarSlot]:
        if not self.enabled:
            return []
        service = await asyncio.to_thread(self._service)
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": timezone,
            "items": [{"id": self._calendar_id}],
        }
        try:
            resp = await asyncio.to_thread(
                lambda: service.freebusy().query(body=body).execute()
            )
        except Exception as exc:
            logger.error("calendar_freebusy_failed", error=str(exc))
            return []

        busy = (
            resp.get("calendars", {})
            .get(self._calendar_id, {})
            .get("busy", [])
        )
        busy_ranges = [
            (
                _parse_dt(period["start"], timezone),
                _parse_dt(period["end"], timezone),
            )
            for period in busy
            if period.get("start") and period.get("end")
        ]

        slots: list[CalendarSlot] = []
        cursor = start
        step = timedelta(minutes=max(duration_minutes, 30))
        while cursor + timedelta(minutes=duration_minutes) <= end and len(slots) < 20:
            slot_end = cursor + timedelta(minutes=duration_minutes)
            if not _overlaps_busy(cursor, slot_end, busy_ranges):
                slots.append(CalendarSlot(start=cursor, end=slot_end, available=True))
            cursor += step
        return slots

    async def create_event(self, draft: CalendarEventDraft) -> dict[str, Any]:
        service = await asyncio.to_thread(self._service)
        body: dict[str, Any] = {
            "summary": draft.title,
            "start": {"dateTime": draft.start_at.isoformat(), "timeZone": draft.timezone},
            "end": {"dateTime": draft.end_at.isoformat(), "timeZone": draft.timezone},
            "reminders": {"useDefault": True},
        }
        if draft.description:
            body["description"] = draft.description
        if draft.attendees:
            body["attendees"] = [{"email": a} for a in draft.attendees]

        created = await asyncio.to_thread(
            lambda: service.events()
            .insert(calendarId=self._calendar_id, body=body, sendUpdates="all")
            .execute()
        )
        logger.info(
            "calendar_event_created",
            provider="google",
            title=draft.title,
            event_id=created.get("id"),
        )
        return {
            "provider_event_id": created.get("id") or "",
            "link": created.get("htmlLink") or "",
        }

    async def update_event(
        self, provider_event_id: str, draft: CalendarEventDraft
    ) -> dict[str, Any]:
        service = await asyncio.to_thread(self._service)
        body: dict[str, Any] = {
            "summary": draft.title,
            "start": {"dateTime": draft.start_at.isoformat(), "timeZone": draft.timezone},
            "end": {"dateTime": draft.end_at.isoformat(), "timeZone": draft.timezone},
        }
        if draft.description:
            body["description"] = draft.description
        if draft.attendees:
            body["attendees"] = [{"email": a} for a in draft.attendees]
        updated = await asyncio.to_thread(
            lambda: service.events()
            .update(
                calendarId=self._calendar_id,
                eventId=provider_event_id,
                body=body,
                sendUpdates="all",
            )
            .execute()
        )
        logger.info(
            "calendar_event_updated",
            provider="google",
            event_id=provider_event_id,
        )
        return {
            "provider_event_id": updated.get("id") or provider_event_id,
            "link": updated.get("htmlLink") or "",
        }

    async def cancel_event(self, provider_event_id: str) -> None:
        service = await asyncio.to_thread(self._service)
        await asyncio.to_thread(
            lambda: service.events()
            .delete(calendarId=self._calendar_id, eventId=provider_event_id, sendUpdates="all")
            .execute()
        )
        logger.info("calendar_event_cancelled", provider="google", event_id=provider_event_id)

    async def list_events(
        self, start: datetime, end: datetime, timezone: str
    ) -> list[dict[str, Any]]:
        service = await asyncio.to_thread(self._service)
        result = await asyncio.to_thread(
            lambda: service.events()
            .list(
                calendarId=self._calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                showDeleted=False,
            )
            .execute()
        )
        items: list[dict[str, Any]] = []
        for item in result.get("items", []):
            start_info = item.get("start", {})
            start_dt = start_info.get("dateTime") or _dt_to_iso(start, timezone)
            items.append({
                "provider_event_id": item.get("id", ""),
                "title": item.get("summary", ""),
                "start_at": start_dt,
                "end_at": (
                    item.get("end", {}).get("dateTime")
                    or _dt_to_iso(end, timezone)
                ),
                "attendees": [
                    a.get("email", "") for a in item.get("attendees", []) if a.get("email")
                ],
                "link": item.get("htmlLink", ""),
                "status": "cancelled" if item.get("status") == "cancelled" else "scheduled",
            })
        return items


def _parse_dt(value: str, timezone: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
    return parsed


def _overlaps_busy(
    start: datetime, end: datetime, busy_ranges: list[tuple[datetime, datetime]]
) -> bool:
    for busy_start, busy_end in busy_ranges:
        if start < busy_end and end > busy_start:
            return True
    return False


def _dt_to_iso(dt: datetime, timezone: str) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone))
    return dt.isoformat()
