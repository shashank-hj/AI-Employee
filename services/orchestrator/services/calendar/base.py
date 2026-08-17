"""Calendar provider abstraction.

Production backends implement :class:`CalendarProvider` and can be swapped
without touching :class:`orchestrator.services.calendar_service.CalendarService`
or the tools that consume it.
"""

from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Any, Protocol
from zoneinfo import ZoneInfo


@dataclass
class CalendarSlot:
    start: datetime
    end: datetime
    available: bool = True


@dataclass
class CalendarEventDraft:
    title: str
    start_at: datetime
    end_at: datetime
    timezone: str = "UTC"
    description: str | None = None
    attendees: list[str] = field(default_factory=list)


@dataclass
class CalendarEvent:
    provider: str
    title: str
    start_at: datetime
    end_at: datetime
    timezone: str
    attendees: list[str]
    status: str = "scheduled"
    id: str | None = None
    provider_event_id: str | None = None
    link: str | None = None
    description: str | None = None


class CalendarProvider(Protocol):
    """A calendar backend. ``enabled`` gates whether it may be used."""

    name: str

    @property
    def enabled(self) -> bool: ...

    async def health_check(self) -> bool:
        """Return True when the backend is reachable and authorized."""

    async def get_availability(
        self,
        start: datetime,
        end: datetime,
        timezone: str,
        duration_minutes: int = 30,
    ) -> list[CalendarSlot]:
        """Return candidate free slots of ``duration_minutes`` in ``[start, end]``."""

    async def create_event(self, draft: CalendarEventDraft) -> dict[str, Any]:
        """Create an event. Returns ``{"provider_event_id": str, "link": str|None}``."""

    async def update_event(
        self, provider_event_id: str, draft: CalendarEventDraft
    ) -> dict[str, Any]:
        """Update an existing event by its provider event id."""

    async def cancel_event(self, provider_event_id: str) -> None:
        """Delete/cancel an event by its provider event id."""

    async def list_events(
        self, start: datetime, end: datetime, timezone: str
    ) -> list[dict[str, Any]]:
        """List events overlapping ``[start, end]`` as lightweight dicts."""


def _to_local(dt: datetime, timezone: str | None) -> datetime:
    """Shift a naive/UTC-aware datetime into ``timezone`` for display."""
    if dt.tzinfo is None or not timezone:
        return dt
    try:
        tz = timezone if isinstance(timezone, tzinfo) else ZoneInfo(timezone)
        return dt.astimezone(tz)
    except (TypeError, ValueError, KeyError):
        return dt


def slot_to_dict(slot: CalendarSlot, timezone: str | None = None) -> dict[str, str | bool]:
    return {
        "start": _to_local(slot.start, timezone).isoformat(),
        "end": _to_local(slot.end, timezone).isoformat(),
        "available": slot.available,
    }
