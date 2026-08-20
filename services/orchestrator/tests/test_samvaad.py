import base64

import httpx
import pytest

from orchestrator.container import get_samvaad_session_manager
from orchestrator.services.samvaad_client import (
    SamvaadError,
    SamvaadSession,
    SamvaadSessionManager,
    sdk_available,
)
from orchestrator.services.samvaad_usage import (
    SamvaadUsageClient,
    estimate_session_cost,
)


def _configured_manager(**overrides) -> SamvaadSessionManager:
    params = {
        "api_key": "test-key",
        "agent_id": "AI-Employee-33c6c05a-c14f",
        "org_id": "org_ai",
        "workspace_id": "workspace_1",
        "base_url": "https://apps.sarvam.ai/api/app-runtime/",
        "sample_rate": 16000,
        "default_language": "English",
        "enabled": True,
    }
    params.update(overrides)
    return SamvaadSessionManager(**params)


class TestSamvaadSessionManager:
    def test_enabled_when_configured(self):
        manager = _configured_manager()
        assert manager.enabled is True
        assert manager.unavailable_reason() is None

    def test_disabled_when_key_missing(self):
        manager = _configured_manager(api_key="")
        assert manager.enabled is False
        assert "SAMVAAD_API_KEY" in (manager.unavailable_reason() or "")

    def test_disabled_when_flag_off(self):
        manager = _configured_manager(enabled=False)
        assert manager.enabled is False
        assert "SAMVAAD_ENABLED" in (manager.unavailable_reason() or "")

    def test_available_sdk_fake_or_real(self):
        assert sdk_available() is True

    @pytest.mark.asyncio
    async def test_open_chat_session(self):
        manager = _configured_manager()
        session = await manager.open_session(user_identifier="user-1", mode="chat")
        assert session.session_id.startswith("samvaad-")
        assert session.mode == "chat"
        assert session.connected is True
        await session.send_text("hello")
        sent = session._agent.sent
        assert ("text", "hello") in sent
        assert manager.get(session.session_id) is session
        await manager.close_session(session.session_id)
        assert manager.get(session.session_id) is None

    @pytest.mark.asyncio
    async def test_interaction_id_set_via_connected_event(self):
        manager = _configured_manager()
        session = await manager.open_session(user_identifier="u", mode="chat")
        assert session.interaction_id is None
        event = type(
            "ConnectedEvent",
            (),
            {
                "interaction_id": "ix-123",
                "type": type("T", (), {"value": "server.action.interaction_connected"}),
            },
        )()
        await manager._on_event(session, event)
        assert session.interaction_id == "ix-123"

    @pytest.mark.asyncio
    async def test_open_call_session_sends_audio(self):
        manager = _configured_manager()
        session = await manager.open_session(user_identifier="user-1", mode="call")
        assert session.mode == "call"
        await session.send_audio(b"\x00\x01" * 100)
        assert ("audio", b"\x00\x01" * 100) in session._agent.sent
        await session.close()

    @pytest.mark.asyncio
    async def test_open_session_rejects_bad_mode(self):
        manager = _configured_manager()
        with pytest.raises(ValueError):
            await manager.open_session(user_identifier="u", mode="pigeon")

    @pytest.mark.asyncio
    async def test_open_session_unavailable_when_disabled(self):
        manager = _configured_manager(api_key="")
        with pytest.raises(SamvaadError):
            await manager.open_session(user_identifier="u", mode="chat")

    @pytest.mark.asyncio
    async def test_callbacks_drain_to_outbox(self):
        manager = _configured_manager()
        session = await manager.open_session(user_identifier="u", mode="chat")
        status = type("S", (), {"value": "completed"})
        text_msg = type("T", (), {"text": "hi", "status": status})()
        await manager._on_text(session, text_msg)
        role = type("R", (), {"value": "bot"})
        transcript = type("X", (), {"role": role, "content": "ok"})()
        await manager._on_transcript(session, transcript)
        msgs = session.drain()
        assert {"type": "text", "text": "hi", "status": "completed"} in msgs
        assert {"type": "transcript", "role": "bot", "content": "ok"} in msgs


class TestSamvaadEndpoints:
    @pytest.mark.asyncio
    async def test_status_disabled_by_default(self, client):
        response = await client.get("/api/samvaad/status")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_status_enabled_when_configured(self, app, client, monkeypatch):
        manager = _configured_manager()
        app.dependency_overrides[get_samvaad_session_manager] = lambda: manager
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_AGENT_ID",
            "AI-Employee-33c6c05a-c14f",
        )
        response = await client.get("/api/samvaad/status")
        assert response.status_code == 200
        assert response.json()["enabled"] is True

    @pytest.mark.asyncio
    async def test_open_and_text_session(self, app, client):
        manager = _configured_manager()
        app.dependency_overrides[get_samvaad_session_manager] = lambda: manager
        response = await client.post(
            "/api/samvaad/sessions",
            json={"user_id": "user-1", "mode": "chat"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["session_id"]
        assert data["status"] == "connected"

        session_id = data["session_id"]
        response = await client.post(
            f"/api/samvaad/sessions/{session_id}/text",
            json={"text": "hello"},
        )
        assert response.status_code == 200
        session = manager.get(session_id)
        assert session is not None
        assert ("text", "hello") in session._agent.sent

    @pytest.mark.asyncio
    async def test_audio_session(self, app, client):
        manager = _configured_manager()
        app.dependency_overrides[get_samvaad_session_manager] = lambda: manager
        response = await client.post(
            "/api/samvaad/sessions",
            json={"user_id": "user-1", "mode": "call"},
        )
        session_id = response.json()["session_id"]
        audio = base64.b64encode(b"\x00\x01" * 100).decode("utf-8")
        response = await client.post(
            f"/api/samvaad/sessions/{session_id}/audio",
            json={"audio_base64": audio},
        )
        assert response.status_code == 200
        session = manager.get(session_id)
        assert ("audio", b"\x00\x01" * 100) in session._agent.sent

    @pytest.mark.asyncio
    async def test_poll_messages(self, app, client):
        manager = _configured_manager()
        app.dependency_overrides[get_samvaad_session_manager] = lambda: manager
        response = await client.post(
            "/api/samvaad/sessions",
            json={"user_id": "u", "mode": "chat"},
        )
        session_id = response.json()["session_id"]
        session = manager.get(session_id)
        session.outbox.put_nowait({"type": "text", "text": "hi", "status": "completed"})
        response = await client.get(f"/api/samvaad/sessions/{session_id}/messages")
        assert response.status_code == 200
        assert response.json()["messages"] == [
            {"type": "text", "text": "hi", "status": "completed"}
        ]

    @pytest.mark.asyncio
    async def test_close_session(self, app, client):
        manager = _configured_manager()
        app.dependency_overrides[get_samvaad_session_manager] = lambda: manager
        response = await client.post(
            "/api/samvaad/sessions",
            json={"user_id": "u", "mode": "chat"},
        )
        session_id = response.json()["session_id"]
        response = await client.post(f"/api/samvaad/sessions/{session_id}/close")
        assert response.status_code == 200
        assert response.json()["status"] == "closed"
        assert manager.get(session_id) is None

    @pytest.mark.asyncio
    async def test_missing_session_404(self, app, client):
        manager = _configured_manager()
        app.dependency_overrides[get_samvaad_session_manager] = lambda: manager
        response = await client.get("/api/samvaad/sessions/nope")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_ws_disconnect_cleans_up_session(self, app, client, monkeypatch):
        manager = _configured_manager()
        monkeypatch.setattr(
            "orchestrator.routers.samvaad.get_samvaad_session_manager",
            lambda: manager,
        )
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/api/samvaad/ws") as ws:
                ws.send_json({"type": "init", "user_id": "u", "mode": "chat"})
                data = ws.receive_json()
                assert data["type"] == "session"
                session_id = data["session_id"]
                assert manager.active_sessions() == 1
                assert manager.get(session_id) is not None
            assert manager.active_sessions() == 0
            assert manager.get(session_id) is None

    @pytest.mark.asyncio
    async def test_ws_close_releases_session(self, app, client, monkeypatch):
        manager = _configured_manager()
        monkeypatch.setattr(
            "orchestrator.routers.samvaad.get_samvaad_session_manager",
            lambda: manager,
        )
        app.dependency_overrides[get_samvaad_session_manager] = lambda: manager
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/api/samvaad/ws") as ws:
                ws.send_json({"type": "init", "user_id": "u", "mode": "chat"})
                data = ws.receive_json()
                session_id = data["session_id"]
                response = tc.post(f"/api/samvaad/sessions/{session_id}/close")
                assert response.status_code == 200
            assert manager.active_sessions() == 0


class TestSamvaadTools:
    _AUTH = {"X-API-Key": "test-key"}
    _ALL_TOOLS = (
        '["on-start/context","on-end/record","calendar/availability",'
        '"calendar/schedule","calendar/update","email/send","email/search",'
        '"search/documents","orders/lookup","pricing/search","human/transfer",'
        '"tasks/manage"]'
    )

    @pytest.fixture(autouse=True)
    def _tools_allowlist(self, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOLS_ALLOWLIST", self._ALL_TOOLS
        )

    @pytest.mark.asyncio
    async def test_tools_require_auth(self, client):
        response = await client.post("/api/samvaad/tools/on-start/context", json={})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_on_start_context(self, client):
        response = await client.post(
            "/api/samvaad/tools/on-start/context",
            json={"user_identifier": "user-1"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert "agent_variables" in response.json()
        assert response.json()["agent_variables"]["user_identifier"] == "user-1"

    @pytest.mark.asyncio
    async def test_search_documents(self, client):
        response = await client.post(
            "/api/samvaad/tools/search/documents",
            json={"query": "onboarding"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_orders_lookup(self, client):
        response = await client.post(
            "/api/samvaad/tools/orders/lookup",
            json={"order_id": "ORD-1"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.asyncio
    async def test_pricing_search(self, client):
        response = await client.post(
            "/api/samvaad/tools/pricing/search",
            json={"query": "enterprise"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.asyncio
    async def test_human_transfer(self, client):
        response = await client.post(
            "/api/samvaad/tools/human/transfer",
            json={"reason": "customer asked", "user_input": "talk to human"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "ticket_id" in response.json()["result"]

    @pytest.mark.asyncio
    async def test_on_end_record(self, client):
        response = await client.post(
            "/api/samvaad/tools/on-end/record",
            json={
                "session_id": "s1",
                "user_id": "user-1",
                "transcript": [{"role": "user", "text": "hi"}, {"role": "bot", "text": "hello"}],
            },
headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["stored_messages"] == 2

    @pytest.mark.asyncio
    async def test_email_send_success_false_when_unconfigured(self, client, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.EMAIL_ADDRESS", "")
        monkeypatch.setattr("orchestrator.config.settings.EMAIL_PASSWORD", "")
        response = await client.post(
            "/api/samvaad/tools/email/send",
            json={"to": "a@b.com", "subject": "s", "body": "b"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["success"] is False

    @pytest.mark.asyncio
    async def test_action_tool_blocked_by_default(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOLS_ALLOWLIST", ""
        )
        response = await client.post(
            "/api/samvaad/tools/human/transfer",
            json={"reason": "x", "user_input": "talk to human"},
            headers=self._AUTH,
        )
        assert response.status_code == 403
        assert "SAMVAAD_TOOLS_ALLOWLIST" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_email_send_403_when_not_allowlisted(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOLS_ALLOWLIST",
            '["search/documents"]',
        )
        monkeypatch.setattr("orchestrator.config.settings.EMAIL_ADDRESS", "")
        monkeypatch.setattr("orchestrator.config.settings.EMAIL_PASSWORD", "")
        response = await client.post(
            "/api/samvaad/tools/email/send",
            json={"to": "a@b.com", "subject": "s", "body": "b"},
            headers=self._AUTH,
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_token_fallback_accepts_secret(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOL_SECRET", "tok-123"
        )
        response = await client.post(
            "/api/samvaad/tools/on-start/context?token=tok-123",
            json={"user_identifier": "user-1"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_token_fallback_rejects_wrong_secret(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOL_SECRET", "tok-123"
        )
        response = await client.post(
            "/api/samvaad/tools/on-start/context?token=wrong",
            json={"user_identifier": "user-1"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_token_fallback_rejects_missing_token(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOL_SECRET", "tok-123"
        )
        response = await client.post(
            "/api/samvaad/tools/on-start/context",
            json={"user_identifier": "user-1"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_secret_header_still_works_when_set(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOL_SECRET", "tok-123"
        )
        response = await client.post(
            "/api/samvaad/tools/on-start/context",
            json={"user_identifier": "user-1"},
            headers={"X-Samvaad-Secret": "tok-123"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_status_reports_tools_gate(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOLS_ALLOWLIST",
            '["email/send","search/documents"]',
        )
        response = await client.get("/api/samvaad/status")
        assert response.status_code == 200
        data = response.json()
        assert "email/send" in data["tools"]["allowed"]
        assert "search/documents" in data["tools"]["allowed"]
        assert "calendar/schedule" in data["tools"]["blocked"]
        assert "human/transfer" in data["tools"]["blocked"]

    @pytest.mark.asyncio
    async def test_new_action_tools_blocked_by_default(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOLS_ALLOWLIST", ""
        )
        for path in ("calendar/update", "tasks/manage"):
            response = await client.post(
                f"/api/samvaad/tools/{path}",
                json={},
                headers=self._AUTH,
            )
            assert response.status_code == 403
            assert "SAMVAAD_TOOLS_ALLOWLIST" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_email_search_available_by_default(self, client, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_TOOLS_ALLOWLIST",
            '["email/send"]',
        )

        class FakeEmailClient:
            enabled = True

            @staticmethod
            def list_messages(max_results, query):
                return [
                    {
                        "id": "1",
                        "from": "Boss <boss@example.com>",
                        "subject": "Report",
                        "date": "2026-01-01",
                        "snippet": "",
                        "label_ids": [],
                    }
                ]

            @staticmethod
            def get_message(message_id):
                return {
                    "id": message_id,
                    "snippet": "Full body snippet",
                    "body": "Full body",
                }

        monkeypatch.setattr(
            "orchestrator.services.gmail_client.EmailClient", FakeEmailClient
        )
        response = await client.post(
            "/api/samvaad/tools/email/search",
            json={"query": 'FROM "boss@example.com"', "max_results": 5, "with_body": True},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert data["messages"][0]["subject"] == "Report"

    @pytest.mark.asyncio
    async def test_email_search_success_false_when_unconfigured(self, client, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.EMAIL_ADDRESS", "")
        monkeypatch.setattr("orchestrator.config.settings.EMAIL_PASSWORD", "")
        response = await client.post(
            "/api/samvaad/tools/email/search",
            json={},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["success"] is False

    @pytest.mark.asyncio
    async def test_calendar_update_reschedule(self, client, monkeypatch):
        class FakeCalendarService:
            def __init__(self):
                self._timezone = "Asia/Kolkata"

            async def match_meetings(
                self, *, session_id=None, meeting_id=None, ref_date=None, limit=10
            ):
                return {
                    "success": True,
                    "matches": [
                        {"id": "m-1", "title": "Existing", "start_at": "2026-01-01T10:00:00+05:30"}
                    ],
                }

            async def update_meeting(self, meeting_id, draft):
                return {
                    "success": True,
                    "meeting": {
                        "id": meeting_id,
                        "title": draft.title,
                        "start_at": draft.start_at.isoformat(),
                        "end_at": draft.end_at.isoformat(),
                    },
                }

            async def cancel_meeting(self, meeting_id):
                return {"success": True, "meeting": {"id": meeting_id}}

        monkeypatch.setattr(
            "orchestrator.routers.samvaad_tools.get_calendar_service",
            lambda: FakeCalendarService(),
        )
        response = await client.post(
            "/api/samvaad/tools/calendar/update",
            json={
                "meeting_id": "m-1",
                "action": "reschedule",
                "new_start_at": "2026-01-02T10:00:00+05:30",
                "new_end_at": "2026-01-02T11:00:00+05:30",
            },
            headers=self._AUTH,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["meeting"]["title"] == "Existing"

    @pytest.mark.asyncio
    async def test_calendar_update_cancel(self, client, monkeypatch):
        class FakeCalendarService:
            async def match_meetings(
                self, *, session_id=None, meeting_id=None, ref_date=None, limit=10
            ):
                return {
                    "success": True,
                    "matches": [
                        {"id": "m-1", "title": "Existing", "start_at": "2026-01-01T10:00:00+05:30"}
                    ],
                }

            async def cancel_meeting(self, meeting_id):
                return {"success": True, "meeting": {"id": meeting_id}}

        monkeypatch.setattr(
            "orchestrator.routers.samvaad_tools.get_calendar_service",
            lambda: FakeCalendarService(),
        )
        response = await client.post(
            "/api/samvaad/tools/calendar/update",
            json={"meeting_id": "m-1", "action": "cancel"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["cancelled"] is True

    @pytest.mark.asyncio
    async def test_calendar_update_requires_new_times(self, client, monkeypatch):
        class FakeCalendarService:
            async def match_meetings(
                self, *, session_id=None, meeting_id=None, ref_date=None, limit=10
            ):
                return {
                    "success": True,
                    "matches": [{"id": "m-1", "title": "Existing",
                                 "start_at": "2026-01-01T10:00:00+05:30"}],
                }

        monkeypatch.setattr(
            "orchestrator.routers.samvaad_tools.get_calendar_service",
            lambda: FakeCalendarService(),
        )
        response = await client.post(
            "/api/samvaad/tools/calendar/update",
            json={"meeting_id": "m-1", "action": "reschedule"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["success"] is False

    @pytest.mark.asyncio
    async def test_tasks_manage_create_and_list(self, client, monkeypatch):
        fake_tasks = {"tasks": []}

        class FakeTaskService:
            async def create(self, *, title, session_id=None, user_id=None,
                             description=None, priority=0, due_at=None):
                task = {
                    "id": "t-1",
                    "title": title,
                    "session_id": session_id,
                    "user_id": user_id,
                    "description": description,
                    "priority": priority,
                    "due_at": due_at.isoformat() if due_at else None,
                    "status": "pending",
                }
                fake_tasks["tasks"].append(task)
                return task

            async def list(self, *, session_id=None, user_id=None, status=None, limit=50):
                return fake_tasks["tasks"]

            async def complete(self, task_id):
                return {"id": task_id, "status": "completed"}

            async def update(self, task_id, **fields):
                return {"id": task_id, "status": fields.get("status") or "pending"}

            async def delete(self, task_id):
                return True

        monkeypatch.setattr(
            "orchestrator.routers.samvaad_tools.get_task_service",
            lambda: FakeTaskService(),
        )
        response = await client.post(
            "/api/samvaad/tools/tasks/manage",
            json={
                "action": "create",
                "session_id": "s1",
                "user_id": "u1",
                "title": "Write report",
                "due_at": "2026-01-05T10:00:00+05:30",
            },
            headers=self._AUTH,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task"]["title"] == "Write report"
        assert data["task"]["status"] == "pending"

        response = await client.post(
            "/api/samvaad/tools/tasks/manage",
            json={"action": "list", "session_id": "s1"},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["count"] == 1

    @pytest.mark.asyncio
    async def test_tasks_manage_requires_action(self, client):
        response = await client.post(
            "/api/samvaad/tools/tasks/manage",
            json={},
            headers=self._AUTH,
        )
        assert response.status_code == 200
        assert response.json()["success"] is False


class TestSamvaadCostEstimate:
    def test_zero_turns_zero_cost(self):
        est = estimate_session_cost(0, 0)
        assert est["turns"] == 0
        assert est["cost_105b_rs"] == 0
        assert est["cost_glm_rs"] == 0

    def test_cost_grows_with_turns_and_duration(self):
        small = estimate_session_cost(5, 30)
        large = estimate_session_cost(50, 300)
        assert large["input_tokens"] > small["input_tokens"]
        assert large["cost_105b_rs"] > small["cost_105b_rs"]
        assert large["cost_glm_rs"] > small["cost_glm_rs"]
        # GLM is strictly more expensive than 105B for identical usage.
        assert large["cost_glm_rs"] > large["cost_105b_rs"]

    @pytest.mark.asyncio
    async def test_usage_client_parses_items(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "interaction_id": "ix-1",
                            "num_messages": 10,
                            "duration_in_seconds": 60,
                            "start_datetime": "2026-08-19T00:00:00Z",
                        },
                        {
                            "interaction_id": "ix-2",
                            "num_messages": 3,
                            "duration_in_seconds": 20,
                            "start_datetime": "2026-08-19T01:00:00Z",
                        },
                    ],
                    "total": 2,
                },
            )

        client = SamvaadUsageClient(
            api_key="k", transport=httpx.MockTransport(handler)
        )
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_ORG_ID", "org_x"
        )
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_WORKSPACE_ID", "ws_x"
        )
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_AGENT_ID", "app_x"
        )
        result = await client.estimate_window(days=14)
        assert result["available"] is True
        assert result["session_count"] == 2
        assert result["sessions"][0]["interaction_id"] == "ix-1"
        assert result["total_105b_rs"] > 0
        await client.aclose()

    @pytest.mark.asyncio
    async def test_usage_client_handles_missing_config(self):
        client = SamvaadUsageClient(
            api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        )
        result = await client.estimate_window(days=14)
        assert result["available"] is False
        assert result["reason"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_usage_endpoint_disabled_when_unconfigured(self, client):
        response = await client.get("/api/samvaad/usage?days=14")
        assert response.status_code == 200
        assert response.json()["available"] is False


class TestSamvaadTurnCap:
    @pytest.mark.asyncio
    async def test_limit_reached_after_max_turns(self, monkeypatch):
        monkeypatch.setattr(
            "orchestrator.config.settings.SAMVAAD_MAX_TURNS", 3
        )
        manager = _configured_manager(max_turns=3)
        session = await manager.open_session(user_identifier="u", mode="chat")
        role = type("R", (), {"value": "user"})
        bot_role = type("R", (), {"value": "bot"})
        for i in range(3):
            await manager._on_transcript(
                session, type("X", (), {"role": role, "content": f"u{i}"})()
            )
            await manager._on_transcript(
                session, type("X", (), {"role": bot_role, "content": "b"})()
            )
        assert session.turn_count == 3
        assert session.limit_reached == "max_turns"
        await manager.close_session(session.session_id)

    @pytest.mark.asyncio
    async def test_limit_not_reached_under_cap(self):
        manager = _configured_manager(max_turns=10)
        session = await manager.open_session(user_identifier="u", mode="chat")
        role = type("R", (), {"value": "user"})
        await manager._on_transcript(
            session, type("X", (), {"role": role, "content": "hi"})()
        )
        assert session.turn_count == 1
        assert session.limit_reached is None
        await manager.close_session(session.session_id)
