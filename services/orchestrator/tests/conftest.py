import enum
import sys
import types

import pytest


# ── Samvaad SDK fake ──
# The orchestrator lazily imports sarvam-conv-ai-sdk in
# orchestrator/services/samvaad_client.py. Install a minimal stand-in BEFORE
# importing orchestrator.app so the Samvaad bridge tests run deterministically
# against a fake agent regardless of whether the real SDK is installed in the
# environment (no network, no committed-version requirement).
def _install_fake_sarvam_sdk() -> None:
    pkg = types.ModuleType("sarvam_conv_ai_sdk")
    msgs = types.ModuleType("sarvam_conv_ai_sdk.messages")
    mtypes = types.ModuleType("sarvam_conv_ai_sdk.messages.types")

    class SarvamToolLanguageName(enum.StrEnum):
        BENGALI = "Bengali"
        GUJARATI = "Gujarati"
        HINDI = "Hindi"
        ENGLISH = "English"

    class InteractionType(enum.StrEnum):
        CALL = "call"
        CHAT = "chat"

    class UserIdentifierType(enum.StrEnum):
        CUSTOM = "custom"
        EMAIL = "email"
        PHONE_NUMBER = "phone_number"
        UNKNOWN = "unknown"

    class InteractionConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAgent:
        def __init__(self, **kwargs):
            self._interaction_id = kwargs.pop("_interaction_id", "ix-fake")
            self._connected = True
            self.kwargs = kwargs
            self.sent: list[tuple[str, object]] = []

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            self._connected = False

        async def wait_for_connect(self, timeout=None) -> bool:
            return True

        def is_connected(self) -> bool:
            return self._connected

        def get_interaction_id(self):
            return self._interaction_id

        async def send_text(self, text: str) -> None:
            self.sent.append(("text", text))

        async def send_audio(self, audio: bytes) -> None:
            self.sent.append(("audio", audio))

    pkg.AsyncSamvaadAgent = FakeAgent
    pkg.InteractionConfig = InteractionConfig
    pkg.InteractionType = InteractionType
    pkg.SarvamToolLanguageName = SarvamToolLanguageName
    mtypes.UserIdentifierType = UserIdentifierType
    msgs.types = mtypes
    pkg.messages = msgs
    sys.modules["sarvam_conv_ai_sdk"] = pkg
    sys.modules["sarvam_conv_ai_sdk.messages"] = msgs
    sys.modules["sarvam_conv_ai_sdk.messages.types"] = mtypes


_install_fake_sarvam_sdk()


from httpx import ASGITransport, AsyncClient  # noqa: E402

from orchestrator.app import create_app  # noqa: E402
from orchestrator.config import get_settings, settings  # noqa: E402
from orchestrator.container import (  # noqa: E402
    _build_classify_llm,
    _build_generate_llm,
    _build_rag_client,
    get_agent_service,
    get_approval_service,
    get_planner,
    get_samvaad_session_manager,
    get_tool_registry,
)


@pytest.fixture(scope="function", autouse=True)
def _force_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "")
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "")
    monkeypatch.setattr(settings, "HITL_ENABLED", False)
    monkeypatch.setattr(settings, "SAMVAAD_ENABLED", False)
    monkeypatch.setattr(settings, "SAMVAAD_API_KEY", "")
    monkeypatch.setattr(settings, "SAMVAAD_AGENT_ID", "")
    monkeypatch.setattr(settings, "SAMVAAD_ORG_ID", "")
    monkeypatch.setattr(settings, "SAMVAAD_WORKSPACE_ID", "")
    monkeypatch.setattr(settings, "SAMVAAD_TOOL_SECRET", "")
    monkeypatch.setattr(settings, "SAMVAAD_TOOLS_ALLOWLIST", "")
    get_settings.cache_clear()
    get_agent_service.cache_clear()
    get_approval_service.cache_clear()
    get_planner.cache_clear()
    get_tool_registry.cache_clear()
    get_samvaad_session_manager.cache_clear()
    _build_classify_llm.cache_clear()
    _build_generate_llm.cache_clear()
    _build_rag_client.cache_clear()


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
