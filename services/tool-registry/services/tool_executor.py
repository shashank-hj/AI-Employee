"""Executes registered tools at invoke time.

Dispatches by ``execution_type``:
- ``native`` → a built-in handler from the registry;
- ``http`` → a remote HTTP call described by ``execution_config``;
- ``mcp`` → a tool call on an external MCP server.

Parameters are validated against the tool's ``input_schema`` before execution.
"""

import asyncio
import time
from typing import Any

import httpx
import structlog
from tool_registry.models.tool import ToolModel
from tool_registry.services.mcp_client import MCPClient
from tool_registry.services.native_handlers import resolve_native_handler
from tool_registry.services.schema_validator import validate_parameters

logger = structlog.get_logger(__name__)


class ToolExecutionError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class ToolExecutor:
    async def execute(
        self,
        tool: ToolModel,
        parameters: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, Any, str | None]:
        """Execute a tool, returning ``(success, data, error)``."""
        start = time.perf_counter()
        errors = validate_parameters(parameters, tool.input_schema or {})
        if errors:
            return False, None, "; ".join(errors)

        try:
            execution_type = tool.execution_type or "native"
            if execution_type == "native":
                data = await self._execute_native(tool.name, parameters)
            elif execution_type == "http":
                data = await self._execute_http(tool, parameters, context or {})
            elif execution_type == "mcp":
                data = await self._execute_mcp(tool, parameters)
            else:
                return False, None, f"Unsupported execution_type: {execution_type}"

            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "tool_executed",
                tool=tool.name,
                execution_type=execution_type,
                duration_ms=duration_ms,
            )
            return True, data, None
        except ToolExecutionError as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.warning(
                "tool_execution_failed",
                tool=tool.name,
                error=str(exc)[:300],
                duration_ms=duration_ms,
            )
            return False, None, str(exc)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error("tool_execution_error", tool=tool.name, error=str(exc)[:300])
            return False, None, f"Execution failed: {str(exc)[:500]}"

    async def _execute_native(self, name: str, parameters: dict[str, Any]) -> Any:
        handler = resolve_native_handler(name)
        if handler is None:
            raise ToolExecutionError(
                f"No native handler registered for tool '{name}'", status_code=404
            )
        return handler(parameters)

    async def _execute_http(
        self,
        tool: ToolModel,
        parameters: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        config = tool.execution_config or {}
        url = config.get("url")
        if not url:
            raise ToolExecutionError(f"Tool '{tool.name}' has no execution url")

        method = (config.get("method") or "POST").upper()
        headers = dict(config.get("headers") or {})
        body = {"parameters": parameters, "context": context}

        timeout = httpx.Timeout(tool.timeout_seconds)
        retries = max(0, int((tool.retry_policy or {}).get("max_retries", 0)))
        delay = float((tool.retry_policy or {}).get("delay_seconds", 1.0))

        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(retries + 1):
                try:
                    response = await client.request(method, url, json=body, headers=headers)
                    response.raise_for_status()
                    try:
                        return response.json()
                    except ValueError:
                        return {"content": response.text}
                except Exception as exc:
                    last_exc = exc
                    if attempt < retries:
                        await asyncio.sleep(delay)
        raise ToolExecutionError(f"HTTP call failed after {retries} retries: {str(last_exc)[:300]}")

    async def _execute_mcp(self, tool: ToolModel, parameters: dict[str, Any]) -> Any:
        config = tool.execution_config or {}
        server_url = config.get("mcp_server_url")
        if not server_url:
            raise ToolExecutionError(f"Tool '{tool.name}' has no mcp_server_url")

        client = MCPClient(server_url=server_url, timeout=tool.timeout_seconds)
        try:
            await client.initialize()
            result = await client.call_tool(tool.name, parameters)
        finally:
            await client.aclose()

        if result.get("isError"):
            raise ToolExecutionError("MCP tool returned an error")
        content = result.get("content", [])
        if len(content) == 1:
            return {"result": content[0]}
        return {"results": content}
