import pytest
from httpx import AsyncClient, ASGITransport

from orchestrator.app import create_app
from orchestrator.config import settings, get_settings
from orchestrator.container import (
    get_agent_service,
    get_approval_service,
    get_planner,
    get_tool_registry,
    _build_classify_llm,
    _build_generate_llm,
    _build_rag_client,
)


@pytest.fixture(scope="function", autouse=True)
def _force_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "")
    monkeypatch.setattr(settings, "SARVAM_API_KEY", "")
    monkeypatch.setattr(settings, "HITL_ENABLED", False)
    get_settings.cache_clear()
    get_agent_service.cache_clear()
    get_approval_service.cache_clear()
    get_planner.cache_clear()
    get_tool_registry.cache_clear()
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
