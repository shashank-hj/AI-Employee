import asyncio
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from tool_registry.app import create_app
from tool_registry.schemas.tools import (
    RetryPolicy,
    ToolCategory,
    ToolCreate,
    ToolInvokeResponse,
    ToolResponse,
    ToolUpdate,
)
from tool_registry.services.tool_executor import ToolExecutor

from shared.utils.exceptions import ConflictException, NotFoundException


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _make_response(**overrides) -> ToolResponse:
    now = datetime.now(UTC)
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
        "execution_type": "native",
        "execution_config": {},
        "created_at": now,
        "updated_at": now,
    }
    defaults.update(overrides)
    return ToolResponse(**defaults)


class MockToolService:
    def __init__(self):
        self._tools: dict[str, ToolResponse] = {}
        self._executor = ToolExecutor()
        self.register_tool = AsyncMock(wraps=self._register_tool)
        self.get_tool = AsyncMock(wraps=self._get_tool)
        self.list_tools = AsyncMock(wraps=self._list_tools)
        self.update_tool = AsyncMock(wraps=self._update_tool)
        self.delete_tool = AsyncMock(wraps=self._delete_tool)
        self.invoke_tool = AsyncMock(wraps=self._invoke_tool)

    async def _register_tool(self, data: ToolCreate):
        existing = any(t.name == data.name for t in self._tools.values())
        if existing:
            raise ConflictException(f"Tool with name '{data.name}' already exists")
        tool = _make_response(name=data.name, description=data.description, version=data.version,
                              category=data.category, permissions=data.permissions,
                              tags=data.tags, timeout_seconds=data.timeout_seconds,
                              input_schema=data.input_schema, output_schema=data.output_schema,
                              execution_type=data.execution_type,
                              execution_config=data.execution_config)
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

    async def _invoke_tool(self, tool_id: str, request):
        if tool_id not in self._tools:
            raise NotFoundException(f"Tool with id '{tool_id}' not found")
        tool = self._tools[tool_id]
        from tool_registry.models.tool import ToolModel
        model = ToolModel(
            name=tool.name,
            category=tool.category.value,
            input_schema=tool.input_schema,
            timeout_seconds=tool.timeout_seconds,
            retry_policy=tool.retry_policy.model_dump(),
            execution_type=tool.execution_type.value,
            execution_config=tool.execution_config.model_dump(exclude_none=True),
        )
        success, data, error = await self._executor.execute(
            model, request.parameters, request.context
        )
        return ToolInvokeResponse(
            success=success,
            data=data,
            error=error,
            tool_id=tool_id,
            tool_name=tool.name,
        )


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
