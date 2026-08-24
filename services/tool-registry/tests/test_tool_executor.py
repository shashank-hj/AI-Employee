from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tool_registry.models.tool import ToolModel
from tool_registry.services.tool_executor import ToolExecutor


def _make_tool(**overrides) -> ToolModel:
    defaults = {
        "id": None,
        "name": "echo",
        "description": "Echo tool",
        "version": "1.0.0",
        "category": "utility",
        "permissions": [],
        "input_schema": {},
        "output_schema": {},
        "timeout_seconds": 30.0,
        "retry_policy": {},
        "tags": [],
        "is_active": True,
        "execution_type": "native",
        "execution_config": {},
    }
    defaults.update(overrides)
    tool = ToolModel(**defaults)
    tool.id = tool.id or __import__("uuid").uuid4()
    return tool


class TestSchemaValidator:
    def test_valid_parameters(self):
        from tool_registry.services.schema_validator import validate_parameters

        schema = {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string", "minLength": 1}},
        }
        assert validate_parameters({"text": "hello"}, schema) == []

    def test_missing_required(self):
        from tool_registry.services.schema_validator import validate_parameters

        schema = {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        }
        errors = validate_parameters({}, schema)
        assert errors and "required" in errors[0]

    def test_wrong_type(self):
        from tool_registry.services.schema_validator import validate_parameters

        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        errors = validate_parameters({"count": "not-an-int"}, schema)
        assert errors and "integer" in errors[0]

    def test_enum(self):
        from tool_registry.services.schema_validator import validate_parameters

        schema = {
            "type": "object",
            "properties": {"level": {"type": "string", "enum": ["low", "high"]}},
        }
        errors = validate_parameters({"level": "medium"}, schema)
        assert errors and "one of" in errors[0]


class TestNativeExecution:
    @pytest.mark.asyncio
    async def test_echo(self):
        executor = ToolExecutor()
        tool = _make_tool(name="echo")
        ok, data, error = await executor.execute(tool, {"text": "hi"})
        assert ok and data == {"echo": "hi"} and error is None

    @pytest.mark.asyncio
    async def test_calculator(self):
        executor = ToolExecutor()
        tool = _make_tool(name="calculator")
        ok, data, error = await executor.execute(tool, {"expression": "2 + 3 * 4"})
        assert ok and data == {"result": 14}

    @pytest.mark.asyncio
    async def test_uppercase(self):
        executor = ToolExecutor()
        tool = _make_tool(name="uppercase")
        ok, data, _ = await executor.execute(tool, {"text": "hello"})
        assert ok and data == {"uppercased": "HELLO"}

    @pytest.mark.asyncio
    async def test_unknown_native_handler(self):
        executor = ToolExecutor()
        tool = _make_tool(name="no-such-handler")
        ok, data, error = await executor.execute(tool, {})
        assert not ok and error and "no native handler" in error.lower()

    @pytest.mark.asyncio
    async def test_validation_failure_blocks_execution(self):
        executor = ToolExecutor()
        tool = _make_tool(
            name="echo",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        )
        ok, data, error = await executor.execute(tool, {})
        assert not ok and error and "required" in error


class TestHTTPExecution:
    @pytest.mark.asyncio
    async def test_http_success(self):
        executor = ToolExecutor()
        tool = _make_tool(
            name="remote-calc",
            execution_type="http",
            execution_config={"url": "https://example.com/calc", "method": "POST"},
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": 42}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            ok, data, error = await executor.execute(tool, {"expression": "40+2"})

        assert ok and data == {"result": 42}
        args = mock_client.request.call_args
        assert args.args[0] == "POST"
        assert args.kwargs["json"]["parameters"] == {"expression": "40+2"}

    @pytest.mark.asyncio
    async def test_http_missing_url(self):
        executor = ToolExecutor()
        tool = _make_tool(name="remote", execution_type="http", execution_config={})
        ok, data, error = await executor.execute(tool, {})
        assert not ok and error and "url" in error.lower()

    @pytest.mark.asyncio
    async def test_http_retries_then_fails(self):
        executor = ToolExecutor()
        tool = _make_tool(
            name="remote",
            execution_type="http",
            execution_config={"url": "https://example.com/x"},
            retry_policy={"max_retries": 1, "delay_seconds": 0},
        )
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=Exception("boom"))

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            ok, data, error = await executor.execute(tool, {})

        assert not ok and error and "retries" in error
        assert mock_client.request.call_count == 2


class TestMCPExecution:
    @pytest.mark.asyncio
    async def test_mcp_call(self):
        executor = ToolExecutor()
        tool = _make_tool(
            name="mcp-tool",
            execution_type="mcp",
            execution_config={"mcp_server_url": "https://mock.local/mcp"},
        )

        calls = {
            "initialize": {"protocolVersion": "2024-11-05"},
            "tools/call": {"content": [{"type": "text", "text": "done"}], "isError": False},
        }

        async def fake_post(url, json, headers=None):
            method = json["method"]
            body = calls.get(method, {"content": [], "isError": True})
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"jsonrpc": "2.0", "id": json["id"], "result": body}
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=fake_post)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value = mock_client
            ok, data, error = await executor.execute(tool, {})

        assert ok and data == {"result": "done"}
        assert mock_client.post.await_count >= 2


class TestInvokeEndpoint:
    @pytest.mark.asyncio
    async def test_invoke_native_tool(self, client_mock):
        create = await client_mock.post("/api/tools", json={
            "name": "echo",
            "category": "utility",
            "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        })
        tool_id = create.json()["id"]
        response = await client_mock.post(f"/api/tools/{tool_id}/invoke", json={
            "parameters": {"text": "hello world"},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == {"echo": "hello world"}
        assert data["tool_name"] == "echo"

    @pytest.mark.asyncio
    async def test_execute_alias(self, client_mock):
        create = await client_mock.post(
            "/api/tools", json={"name": "uppercase", "category": "utility"}
        )
        tool_id = create.json()["id"]
        response = await client_mock.post(f"/api/tools/{tool_id}/execute", json={
            "parameters": {"text": "abc"},
        })
        assert response.status_code == 200
        assert response.json()["data"] == {"uppercased": "ABC"}

    @pytest.mark.asyncio
    async def test_invoke_validation_error(self, client_mock):
        create = await client_mock.post("/api/tools", json={
            "name": "echo",
            "category": "utility",
            "input_schema": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        })
        tool_id = create.json()["id"]
        response = await client_mock.post(f"/api/tools/{tool_id}/invoke", json={"parameters": {}})
        data = response.json()
        assert data["success"] is False
        assert data["error"]

    @pytest.mark.asyncio
    async def test_invoke_not_found(self, client_mock):
        response = await client_mock.post("/api/tools/nonexistent/invoke", json={"parameters": {}})
        assert response.status_code == 404


class TestMCPEndpoint:
    @pytest.mark.asyncio
    async def test_mcp_initialize(self, client_mock):
        response = await client_mock.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "initialize",
            "params": {},
        })
        assert response.status_code == 200
        assert response.json()["result"]["protocolVersion"]

    @pytest.mark.asyncio
    async def test_mcp_tools_list(self, client_mock):
        await client_mock.post("/api/tools", json={"name": "echo", "category": "utility"})
        response = await client_mock.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/list",
            "params": {},
        })
        assert response.status_code == 200
        tools = response.json()["result"]["tools"]
        assert any(t["name"] == "echo" for t in tools)

    @pytest.mark.asyncio
    async def test_mcp_tools_call(self, client_mock):
        await client_mock.post("/api/tools", json={"name": "echo", "category": "utility"})
        response = await client_mock.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": "3",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hi"}},
        })
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["isError"] is False
        assert "hi" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_mcp_unknown_tool(self, client_mock):
        response = await client_mock.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": "4",
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        })
        assert response.status_code == 200
        assert "error" in response.json()

    @pytest.mark.asyncio
    async def test_mcp_unknown_method(self, client_mock):
        response = await client_mock.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": "5",
            "method": "bogus",
            "params": {},
        })
        assert response.status_code == 200
        assert response.json()["error"]["code"] == -32601
