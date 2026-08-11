"""Tests for the gateway channel stats/events endpoints (dashboard widget)."""

from datetime import UTC, datetime

import pytest

from gateway.database.session import get_db
from gateway.models import ChannelEvent
from gateway.routers.channels import router
from gateway.services.channel_events import ChannelEventsService, _serialize_event


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def one(self):
        return self._rows[0]

    def scalars(self):
        return self


class FakeSession:
    """Responds to the two summary group-bys and the events select by column shape."""

    def __init__(self, status_rows, channel_rows, event_rows):
        self._status_rows = status_rows
        self._channel_rows = channel_rows
        self._event_rows = event_rows

    async def execute(self, query):
        keys = [getattr(c, "key", None) for c in query.selected_columns.values()]
        if len(keys) == 3:
            return FakeResult(self._status_rows)
        if len(keys) == 2:
            return FakeResult(self._channel_rows)
        return FakeResult(self._event_rows)


def _ns(**kwargs):
    return type("Row", (), kwargs)()


def _event(channel="web", status="accepted", reason=None, category=None, request_id="req-1"):
    return ChannelEvent(
        recorded_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        channel=channel,
        scope="user:user-1",
        status=status,
        violation_category=category,
        reason=reason,
        redactions_count=2,
        request_id=request_id,
        message_id="m-1",
        duration_ms=12.5,
    )


@pytest.fixture
def fake_session():
    return FakeSession(
        status_rows=[
            _ns(status="accepted", count=10, redactions=4),
            _ns(status="blocked", count=2, redactions=0),
            _ns(status="rate_limited", count=1, redactions=0),
        ],
        channel_rows=[
            _ns(channel="web", count=9),
            _ns(channel="whatsapp", count=4),
        ],
        event_rows=[_event(), _event(status="blocked", reason="blocked word", category="toxicity")],
    )


def test_router_registered():
    paths = {route.path for route in router.routes}
    assert "/api/channels/stats" in paths
    assert "/api/channels/events" in paths


class TestChannelStatsService:
    @pytest.mark.asyncio
    async def test_summary(self, fake_session):
        result = await ChannelEventsService(fake_session).summary()
        assert result["totals"]["accepted"] == 10
        assert result["totals"]["blocked"] == 2
        assert result["totals"]["rate_limited"] == 1
        assert result["totals"]["calls"] == 13
        assert result["totals"]["redactions"] == 4
        assert result["by_channel"] == [
            {"channel": "web", "calls": 9},
            {"channel": "whatsapp", "calls": 4},
        ]

    @pytest.mark.asyncio
    async def test_events_serialized(self, fake_session):
        result = await ChannelEventsService(fake_session).events(limit=10)
        assert len(result) == 2
        first = result[0]
        assert first["channel"] == "web"
        assert first["status"] == "accepted"
        assert first["redactions_count"] == 2
        assert first["request_id"] == "req-1"
        assert first["recorded_at"] is not None

    def test_serialize_event(self):
        event = _event(status="blocked", reason="blocked word", category="injection")
        data = _serialize_event(event)
        assert data["status"] == "blocked"
        assert data["violation_category"] == "injection"
        assert data["reason"] == "blocked word"


class TestChannelStatsEndpoints:
    @pytest.mark.asyncio
    async def test_stats_returns_summary(self, app, client, fake_session):
        app.dependency_overrides[get_db] = lambda: fake_session
        try:
            response = await client.get("/api/channels/stats?start=2026-01-01T00:00:00Z")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["totals"]["accepted"] == 10
        assert len(data["by_channel"]) == 2

    @pytest.mark.asyncio
    async def test_events_returns_rows(self, app, client, fake_session):
        app.dependency_overrides[get_db] = lambda: fake_session
        try:
            response = await client.get("/api/channels/events?limit=5&channel=web")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_stats_db_error_503(self, app, client):
        class BrokenSession:
            async def execute(self, query):
                raise RuntimeError("pg down")

        app.dependency_overrides[get_db] = lambda: BrokenSession()
        try:
            response = await client.get("/api/channels/stats")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 503
