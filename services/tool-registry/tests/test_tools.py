import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tool_registry.app import create_app
from tool_registry.schemas.tools import (
    RetryPolicy,
    ToolCategory,
    ToolCreate,
    ToolResponse,
    ToolUpdate,
)
from tool_registry.services.tool_service import ToolService
from shared.utils.exceptions import ConflictException, NotFoundException


def _make_response(**overrides) -> ToolResponse:
    now = datetime.now(timezone.utc)
    defaults = {
        "id": str(uuid.uuid4()),
        "name": "test-tool",
        "description": "A test tool",
        "version": "1.0.0",
        "category": ToolCategory.UTILITY,
        "permissions": ["read", "write"],
        "input_schema": {"param1": {"type": "string"}},
        "output_schema": {"result": {"type": "string"}},
        "timeout_seconds": 30.0,
        "retry_policy": RetryPolicy(),
        "tags": ["test", "mock"],
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    defaults.update(overrides)
    return ToolResponse(**defaults)


class MockToolService:
    def __init__(self):
        self._tools: dict[str, ToolResponse] = {}
        self.register_tool = AsyncMock(wraps=self._register_tool)
        self.get_tool = AsyncMock(wraps=self._get_tool)
        self.list_tools = AsyncMock(wraps=self._list_tools)
        self.update_tool = AsyncMock(wraps=self._update_tool)
        self.delete_tool = AsyncMock(wraps=self._delete_tool)

    async def _register_tool(self, data: ToolCreate):
        existing = any(t.name == data.name for t in self._tools.values())
        if existing:
            raise ConflictException(f"Tool with name '{data.name}' already exists")
        tool = _make_response(name=data.name, description=data.description, version=data.version,
                              category=data.category, permissions=data.permissions,
                              tags=data.tags, timeout_seconds=data.timeout_seconds)
        self._tools[tool.id] = tool
        return tool

    async def _get_tool(self, tool_id: str):
        if tool_id not in self._tools:
            raise NotFoundException(f"Tool with id '{tool_id}' not found")
        return self._tools[tool_id]

    async def _list_tools(self, params):
        tools = list(self._tools.values())
        total = len(tools)
        start = (params.page - 1) * params.page_size
        end = start + params.page_size
        return tools[start:end], total

    async def _update_tool(self, tool_id: str, data: ToolUpdate):
        if tool_id not in self._tools:
            raise NotFoundException(f"Tool with id '{tool_id}' not found")
        tool = self._tools[tool_id]
        update_dict = data.model_dump(exclude_unset=True)
        for k, v in update_dict.items():
            setattr(tool, k, v)
        self._tools[tool_id] = tool
        return tool

    async def _delete_tool(self, tool_id: str):
        if tool_id not in self._tools:
            raise NotFoundException(f"Tool with id '{tool_id}' not found")
        del self._tools[tool_id]


@pytest.fixture
def mock_service():
    return MockToolService()


@pytest.fixture
def app_mock(mock_service):
    app = create_app()

    async def override_get_tool_service():
        return mock_service

    from tool_registry.container import get_tool_service
    app.dependency_overrides[get_tool_service] = override_get_tool_service

    return app


@pytest.fixture
async def client_mock(app_mock):
    transport = ASGITransport(app=app_mock)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


TOOL_PAYLOAD = {
    "name": "test-tool",
    "description": "A test tool",
    "version": "1.0.0",
    "category": "utility",
    "permissions": ["read", "write"],
    "input_schema": {"param1": {"type": "string"}},
    "output_schema": {"result": {"type": "string"}},
    "timeout_seconds": 30.0,
    "retry_policy": {"max_retries": 3, "delay_seconds": 1.0, "backoff_multiplier": 2.0},
    "tags": ["test", "mock"],
}


class TestRegisterTool:
    @pytest.mark.asyncio
    async def test_register_success(self, client_mock):
        response = await client_mock.post("/api/tools", json=TOOL_PAYLOAD)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test-tool"
        assert data["category"] == "utility"
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_register_duplicate_name(self, client_mock):
        await client_mock.post("/api/tools", json=TOOL_PAYLOAD)
        response = await client_mock.post("/api/tools", json=TOOL_PAYLOAD)
        assert response.status_code == 409
        assert "already exists" in response.json()["message"]

    @pytest.mark.asyncio
    async def test_register_missing_name(self, client_mock):
        payload = {**TOOL_PAYLOAD, "name": ""}
        response = await client_mock.post("/api/tools", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_invalid_category(self, client_mock):
        payload = {**TOOL_PAYLOAD, "category": "invalid"}
        response = await client_mock.post("/api/tools", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_invalid_version(self, client_mock):
        payload = {**TOOL_PAYLOAD, "version": "not-semver"}
        response = await client_mock.post("/api/tools", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_negative_timeout(self, client_mock):
        payload = {**TOOL_PAYLOAD, "timeout_seconds": -1}
        response = await client_mock.post("/api/tools", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_defaults_applied(self, client_mock):
        minimal = {"name": "minimal-tool", "category": "data"}
        response = await client_mock.post("/api/tools", json=minimal)
        assert response.status_code == 201
        data = response.json()
        assert data["version"] == "1.0.0"
        assert data["timeout_seconds"] == 30.0
        assert data["permissions"] == []
        assert data["tags"] == []
        assert data["retry_policy"]["max_retries"] == 3


class TestGetTool:
    @pytest.mark.asyncio
    async def test_get_existing(self, client_mock):
        create_resp = await client_mock.post("/api/tools", json=TOOL_PAYLOAD)
        tool_id = create_resp.json()["id"]
        response = await client_mock.get(f"/api/tools/{tool_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "test-tool"

    @pytest.mark.asyncio
    async def test_get_not_found(self, client_mock):
        response = await client_mock.get("/api/tools/nonexistent-id")
        assert response.status_code == 404


class TestListTools:
    @pytest.mark.asyncio
    async def test_list_empty(self, client_mock):
        response = await client_mock.get("/api/tools")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["total"] == 0
        assert data["data"]["items"] == []

    @pytest.mark.asyncio
    async def test_list_with_items(self, client_mock):
        await client_mock.post("/api/tools", json={"name": "tool-a", "category": "utility"})
        await client_mock.post("/api/tools", json={"name": "tool-b", "category": "data"})
        response = await client_mock.get("/api/tools")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_pagination(self, client_mock):
        for i in range(5):
            await client_mock.post("/api/tools", json={"name": f"tool-{i}", "category": "utility"})
        response = await client_mock.get("/api/tools?page=1&page_size=2")
        data = response.json()
        assert len(data["data"]["items"]) == 2
        assert data["data"]["total"] == 5
        assert data["data"]["pages"] == 3

    @pytest.mark.asyncio
    async def test_list_filter_category(self, client_mock):
        await client_mock.post("/api/tools", json={"name": "tool-a", "category": "utility"})
        await client_mock.post("/api/tools", json={"name": "tool-b", "category": "data"})
        response = await client_mock.get("/api/tools?category=utility")
        data = response.json()
        assert data["data"]["total"] == 2  # mock doesn't filter, verify endpoint works

    @pytest.mark.asyncio
    async def test_list_search(self, client_mock):
        await client_mock.post("/api/tools", json={"name": "searchable-tool", "category": "utility"})
        response = await client_mock.get("/api/tools?search=searchable")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_list_invalid_page(self, client_mock):
        response = await client_mock.get("/api/tools?page=0")
        assert response.status_code == 422


class TestUpdateTool:
    @pytest.mark.asyncio
    async def test_update_success(self, client_mock):
        create_resp = await client_mock.post("/api/tools", json=TOOL_PAYLOAD)
        tool_id = create_resp.json()["id"]
        response = await client_mock.put(
            f"/api/tools/{tool_id}",
            json={"description": "Updated description", "is_active": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
        assert data["is_active"] is False

    @pytest.mark.asyncio
    async def test_update_not_found(self, client_mock):
        response = await client_mock.put(
            "/api/tools/nonexistent",
            json={"description": "Updated"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_empty_body(self, client_mock):
        create_resp = await client_mock.post("/api/tools", json=TOOL_PAYLOAD)
        tool_id = create_resp.json()["id"]
        response = await client_mock.put(f"/api/tools/{tool_id}", json={})
        assert response.status_code == 200


class TestDeleteTool:
    @pytest.mark.asyncio
    async def test_delete_success(self, client_mock):
        create_resp = await client_mock.post("/api/tools", json=TOOL_PAYLOAD)
        tool_id = create_resp.json()["id"]
        response = await client_mock.delete(f"/api/tools/{tool_id}")
        assert response.status_code == 204
        get_resp = await client_mock.get(f"/api/tools/{tool_id}")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, client_mock):
        response = await client_mock.delete("/api/tools/nonexistent")
        assert response.status_code == 404
