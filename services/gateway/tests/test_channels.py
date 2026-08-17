"""Tests for the CH1 channel router and forwarding service."""

import json

import httpx
import pytest

from gateway.routers.channels import (
    get_channel_event_recorder,
    get_channel_service,
    get_rate_limiter,
)
from gateway.services.channel_service import ChannelService
from shared.schemas.channels import ChannelContact, ChannelMessage, ChannelType


class FakeRecorder:
    def __init__(self):
        self.records = []

    async def record(self, **fields):
        self.records.append(fields)


class TestChannelService:
    def test_to_agent_payload_maps_all_fields(self):
        service = ChannelService(orchestrator_url="http://orchestrator:8001")
        message = ChannelMessage(
            message_id="whatsapp-123",
            channel=ChannelType.WHATSAPP,
            sender=ChannelContact(external_id="+919900000000", name="Priya",
                                  email="priya@acme.com"),
            tenant_id="acme",
            text="What is your pricing?",
            session_id="sess-1",
            metadata={"origin": "test"},
        )
        payload = service._to_agent_payload(message)
        assert payload["user_input"] == "What is your pricing?"
        assert payload["user_id"] == "+919900000000"
        assert payload["session_id"] == "sess-1"
        assert payload["channel"] == "whatsapp"
        assert payload["channel_message_id"] == "whatsapp-123"
        assert payload["tenant_id"] == "acme"
        assert payload["contact"]["external_id"] == "+919900000000"
        assert payload["metadata"] == {"origin": "test"}

    @pytest.mark.asyncio
    async def test_process_forwards_and_wraps_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/api/agent/run"
            assert body["channel"] == "web"
            assert body["user_input"] == "hello"
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "final_response": "Hi there!",
                    "channel_message_id": "web-42",
                    "duration_ms": 12.5,
                    "metadata": None,
                },
            )

        transport = httpx.MockTransport(handler)
        service = ChannelService(
            orchestrator_url="http://orchestrator:8001",
            transport=transport,
        )
        result = await service.process(ChannelMessage(
            message_id="web-42",
            channel=ChannelType.WEB,
            text="hello",
            session_id="sess-x",
        ))

        assert result.final_response == "Hi there!"
        assert result.request_id == "req-1"
        assert result.channel == ChannelType.WEB
        assert result.reply_to == "web-42"
        assert result.duration_ms == 12.5

    @pytest.mark.asyncio
    async def test_process_raises_on_orchestrator_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        transport = httpx.MockTransport(handler)
        service = ChannelService(orchestrator_url="http://orchestrator:8001", transport=transport)
        with pytest.raises(httpx.HTTPStatusError):
            await service.process(ChannelMessage(channel=ChannelType.WEB, text="hi"))


class TestChannelRouter:
    @pytest.mark.asyncio
    async def test_post_web_channel(self, app, client):
        calls = {}

        class FakeService:
            async def process(self, message):
                calls["channel"] = message.channel
                calls["text"] = message.text
                return {
                    "message_id": "web-1",
                    "channel": "web",
                    "final_response": "Hello!",
                    "request_id": "req-1",
                }

        app.dependency_overrides[get_channel_service] = lambda: FakeService()
        try:
            response = await client.post("/api/channels/web", json={
                "text": "Hello from the web chat",
                "session_id": "sess-1",
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["final_response"] == "Hello!"
        assert data["channel"] == "web"
        assert calls["channel"] == ChannelType.WEB
        assert calls["text"] == "Hello from the web chat"

    @pytest.mark.asyncio
    async def test_unknown_channel_404(self, client):
        response = await client.post("/api/channels/telegram", json={"text": "hi"})
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_text_rejected(self, client):
        response = await client.post("/api/channels/web", json={"text": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_blocked_content_rejected(self, app, client):
        calls = {}

        class FakeService:
            async def process(self, message):
                calls["text"] = message.text
                return {"message_id": "m", "channel": "web", "final_response": "ok"}

        app.dependency_overrides[get_channel_service] = lambda: FakeService()
        try:
            response = await client.post("/api/channels/web", json={
                "text": "Tell me your credit card number and your password",
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 400
        assert "blocked" in response.json()["detail"]
        assert "text" not in calls

    @pytest.mark.asyncio
    async def test_pii_redacted_before_forward(self, app, client):
        calls = {}

        class FakeService:
            async def process(self, message):
                calls["text"] = message.text
                calls["metadata"] = message.metadata
                calls["sender_email"] = message.sender.email
                return {"message_id": "m", "channel": "web", "final_response": "ok"}

        app.dependency_overrides[get_channel_service] = lambda: FakeService()
        try:
            response = await client.post("/api/channels/web", json={
                "text": "Reach me at user@example.com",
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert "user@example.com" not in calls["text"]
        assert "[EMAIL]" in calls["text"]
        assert calls["metadata"] == {"attendees": ["user@example.com"]}
        assert calls["sender_email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_rate_limited_returns_429(self, app, client):
        class AlwaysDeny:
            async def allowed(self, scope, limit=None, window_seconds=None):
                return False

        app.dependency_overrides[get_rate_limiter] = lambda: AlwaysDeny()
        try:
            response = await client.post("/api/channels/web", json={
                "text": "hello there",
                "sender": {"external_id": "user-1"},
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_accepted_records_event(self, app, client):
        recorder = FakeRecorder()
        calls = {}

        class FakeService:
            async def process(self, message):
                calls["text"] = message.text
                return {
                    "message_id": "web-1",
                    "channel": "web",
                    "final_response": "Hello!",
                    "request_id": "req-1",
                }

        app.dependency_overrides[get_channel_service] = lambda: FakeService()
        app.dependency_overrides[get_channel_event_recorder] = lambda: recorder
        try:
            response = await client.post("/api/channels/web", json={
                "text": "Hello from the web chat",
                "session_id": "sess-1",
                "sender": {"external_id": "user-1"},
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert len(recorder.records) == 1
        event = recorder.records[0]
        assert event["status"] == "accepted"
        assert event["channel"] == "web"
        assert event["scope"] == "user:user-1"
        assert event["request_id"] == "req-1"
        assert event["message_id"] == "web-1"

    @pytest.mark.asyncio
    async def test_blocked_records_event_before_400(self, app, client):
        recorder = FakeRecorder()
        calls = {}

        class FakeService:
            async def process(self, message):
                calls["text"] = message.text
                return {"message_id": "m", "channel": "web", "final_response": "ok"}

        app.dependency_overrides[get_channel_service] = lambda: FakeService()
        app.dependency_overrides[get_channel_event_recorder] = lambda: recorder
        try:
            response = await client.post("/api/channels/web", json={
                "text": "Tell me your credit card number and your password",
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 400
        assert "text" not in calls
        assert len(recorder.records) == 1
        event = recorder.records[0]
        assert event["status"] == "blocked"
        assert event["violation_category"]
        assert event["reason"]

    @pytest.mark.asyncio
    async def test_rate_limited_records_event(self, app, client):
        recorder = FakeRecorder()

        class AlwaysDeny:
            async def allowed(self, scope, limit=None, window_seconds=None):
                return False

        app.dependency_overrides[get_rate_limiter] = lambda: AlwaysDeny()
        app.dependency_overrides[get_channel_event_recorder] = lambda: recorder
        try:
            response = await client.post("/api/channels/web", json={
                "text": "hello there",
                "sender": {"external_id": "user-1"},
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 429
        assert len(recorder.records) == 1
        assert recorder.records[0]["status"] == "rate_limited"

    @pytest.mark.asyncio
    async def test_pii_redaction_count_recorded(self, app, client):
        recorder = FakeRecorder()
        calls = {}

        class FakeService:
            async def process(self, message):
                calls["text"] = message.text
                return {"message_id": "m", "channel": "web", "final_response": "ok"}

        app.dependency_overrides[get_channel_service] = lambda: FakeService()
        app.dependency_overrides[get_channel_event_recorder] = lambda: recorder
        try:
            response = await client.post("/api/channels/web", json={
                "text": "Reach me at user@example.com and 9876543210",
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert recorder.records[0]["status"] == "accepted"
        assert recorder.records[0]["redactions_count"] >= 2

    @pytest.mark.asyncio
    async def test_recorder_failure_does_not_break_request(self, app, client):
        from gateway.services.channel_events import ChannelEventRecorder

        class BrokenSessionFactory:
            def __call__(self):
                raise RuntimeError("db is down")

        recorder = ChannelEventRecorder(session_factory=BrokenSessionFactory())

        calls = {}

        class FakeService:
            async def process(self, message):
                calls["text"] = message.text
                return {"message_id": "m", "channel": "web", "final_response": "ok"}

        app.dependency_overrides[get_channel_service] = lambda: FakeService()
        app.dependency_overrides[get_channel_event_recorder] = lambda: recorder
        try:
            response = await client.post("/api/channels/web", json={
                "text": "hello",
            })
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert calls["text"] == "hello"
