from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.services.calendar.base import CalendarEventDraft, CalendarSlot
from orchestrator.services.calendar_service import CalendarService


class _FakeProvider:
    name = "google"
    enabled = True

    def __init__(self, busy: bool = False):
        self._busy = busy

    async def get_availability(self, start, end, timezone, duration_minutes=30):
        if self._busy:
            # Exact requested window is busy, but a broader day window is free.
            if end - start <= timedelta(minutes=60):
                return []
            return [
                CalendarSlot(
                    start=datetime(2026, 8, 20, 10, 0, tzinfo=UTC),
                    end=datetime(2026, 8, 20, 10, 30, tzinfo=UTC),
                    available=True,
                )
            ]
        return [CalendarSlot(start=start, end=end, available=True)]

    async def health_check(self):
        return True

    async def create_event(self, draft):
        return {"provider_event_id": "evt-1", "link": "https://cal.example/evt-1"}

    async def update_event(self, provider_event_id, draft):
        return {"provider_event_id": provider_event_id, "link": ""}

    async def cancel_event(self, provider_event_id):
        return None

    async def list_events(self, start, end, timezone):
        return []


class _FakePending:
    def __init__(self):
        self._rows = {}

    async def upsert(self, **kwargs):
        import uuid
        class _P:
            id = str(uuid.uuid4())
            status = "proposed"
        self._rows[kwargs["session_id"]] = _P()
        return self._rows[kwargs["session_id"]]

    async def get_active(self, session_id):
        return self._rows.get(session_id)

    async def clear(self, session_id):
        self._rows.pop(session_id, None)


def _draft(start_at=None):
    start = start_at or datetime(2026, 8, 20, 9, 30, tzinfo=UTC)
    return CalendarEventDraft(
        title="Demo",
        start_at=start,
        end_at=start + timedelta(minutes=30),
        timezone="Asia/Kolkata",
        attendees=["ash@example.com"],
    )


class TestProposeBookingAlternatives:
    @pytest.mark.asyncio
    async def test_busy_slot_returns_same_day_alternatives(self):
        provider = _FakeProvider(busy=True)
        pending = _FakePending()
        svc = CalendarService(provider=provider, pending_repository=pending)
        result = await svc.propose_booking(
            session_id="s1",
            user_id=None,
            draft=_draft(),
        )
        assert result["available"] is False
        assert result["proposed"] is False
        # Alternative slots from the broad same-day window must be present
        assert isinstance(result["slots"], list)
        assert len(result["slots"]) >= 1

    @pytest.mark.asyncio
    async def test_free_slot_proposes_booking(self):
        provider = _FakeProvider(busy=False)
        pending = _FakePending()
        svc = CalendarService(provider=provider, pending_repository=pending)
        result = await svc.propose_booking(
            session_id="s1",
            user_id=None,
            draft=_draft(),
        )
        assert result["available"] is True
        assert result["proposed"] is True
        assert result["proposal_id"] is not None
