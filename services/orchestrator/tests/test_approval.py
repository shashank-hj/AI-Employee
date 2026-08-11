from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.services.approval_service import (
    APPROVAL_PREFIX,
    ApprovalService,
)

_AS = "orchestrator.services.approval_service"


class TestApprovalServiceUnit:
    def test_requires_approval_when_enabled_and_listed(self, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.HITL_ENABLED", True)
        service = ApprovalService(approval_tools=["send_email"])
        assert service.requires_approval("send_email") is True
        assert service.requires_approval("calculator") is False

    def test_requires_approval_when_disabled(self, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.HITL_ENABLED", False)
        service = ApprovalService(approval_tools=["send_email"])
        assert service.requires_approval("send_email") is False

    def test_enabled_flag_explicit_override(self):
        service = ApprovalService(approval_tools=["send_email"], enabled=False)
        assert service.enabled is False
        assert service.requires_approval("send_email") is False

    def test_correlation_id_is_deterministic(self):
        a = ApprovalService._correlation_id("s1", "u1", "send_email", {"to": "a@b.com"})
        b = ApprovalService._correlation_id("s1", "u1", "send_email", {"to": "a@b.com"})
        assert a == b
        assert a.startswith(f"{APPROVAL_PREFIX}send_email:")

    def test_correlation_id_differs_on_params(self):
        a = ApprovalService._correlation_id("s1", "u1", "send_email", {"to": "a@b.com"})
        b = ApprovalService._correlation_id("s1", "u1", "send_email", {"to": "c@d.com"})
        assert a != b

    def test_correlation_id_differs_on_tool(self):
        a = ApprovalService._correlation_id("s1", "u1", "send_email", {"to": "a@b.com"})
        b = ApprovalService._correlation_id("s1", "u1", "calendar", {"to": "a@b.com"})
        assert a != b

    def test_correlation_id_handles_none(self):
        a = ApprovalService._correlation_id(None, None, "send_email", {})
        assert a.startswith(f"{APPROVAL_PREFIX}send_email:")


class TestApprovalServiceGate:
    @pytest.mark.asyncio
    async def test_check_or_request_skips_unlisted_tool(self, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.HITL_ENABLED", True)
        service = ApprovalService(approval_tools=["send_email"])
        decision = await service.check_or_request(
            session_id="s1",
            user_id="u1",
            user_input="calc",
            tool_name="calculator",
            parameters={},
        )
        assert decision.required is False
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_check_or_request_creates_task(self, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.HITL_ENABLED", True)
        service = ApprovalService(approval_tools=["send_email"])

        fake_task = MagicMock()
        fake_task.id = "task-1"
        fake_service = MagicMock()
        fake_service.create = AsyncMock(return_value=fake_task)

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session

        none = AsyncMock(return_value=None)
        with patch(f"{_AS}._session", return_value=mock_session), \
             patch(f"{_AS}._find_resolved", new=none), \
             patch(f"{_AS}._find_pending", new=none), \
             patch(f"{_AS}.HumanTaskService", return_value=fake_service):
            decision = await service.check_or_request(
                session_id="s1",
                user_id="u1",
                user_input="Send the report",
                tool_name="send_email",
                parameters={"to": "a@b.com"},
            )

        assert decision.required is True
        assert decision.approved is False
        assert decision.task_id == "task-1"
        fake_service.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_or_request_returns_pending(self, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.HITL_ENABLED", True)
        service = ApprovalService(approval_tools=["send_email"])

        none = AsyncMock(return_value=None)
        pending = AsyncMock(return_value="task-pending")
        with patch(f"{_AS}._find_resolved", new=none), \
             patch(f"{_AS}._find_pending", new=pending):
            decision = await service.check_or_request(
                session_id="s1",
                user_id="u1",
                user_input="Send email",
                tool_name="send_email",
                parameters={},
            )

        assert decision.approved is False
        assert decision.task_id == "task-pending"

    @pytest.mark.asyncio
    async def test_check_or_request_returns_granted_when_resolved(self, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.HITL_ENABLED", True)
        service = ApprovalService(approval_tools=["send_email"])

        resolved = AsyncMock(return_value="task-resolved")
        with patch(f"{_AS}._find_resolved", new=resolved):
            decision = await service.check_or_request(
                session_id="s1",
                user_id="u1",
                user_input="Send email",
                tool_name="send_email",
                parameters={},
            )

        assert decision.approved is True
        assert decision.task_id == "task-resolved"

    @pytest.mark.asyncio
    async def test_approve_claims_and_resolves(self, monkeypatch):
        monkeypatch.setattr("orchestrator.config.settings.HITL_ENABLED", True)
        service = ApprovalService(approval_tools=["send_email"])

        fake_task = MagicMock()
        fake_task.id = "task-9"
        fake_service = MagicMock()
        fake_service.claim = AsyncMock(return_value=fake_task)
        fake_service.resolve = AsyncMock(return_value={"approved": True})

        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session

        with patch(f"{_AS}._session", return_value=mock_session), \
             patch(f"{_AS}.HumanTaskService", return_value=fake_service):
            result = await service.approve("task-9", note="Looks good")

        assert result == {"approved": True, "task_id": "task-9"}
        fake_service.claim.assert_awaited_once_with("task-9", assigned_to="operator")

    @pytest.mark.asyncio
    async def test_approve_missing_task_returns_none(self, monkeypatch):
        service = ApprovalService(approval_tools=["send_email"])
        fake_service = MagicMock()
        fake_service.claim = AsyncMock(return_value=None)
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session

        with patch(f"{_AS}._session", return_value=mock_session), \
             patch(f"{_AS}.HumanTaskService", return_value=fake_service):
            result = await service.approve("task-missing")

        assert result is None
