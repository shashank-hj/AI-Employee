"""Minimal MCP (Model Context Protocol) JSON-RPC client.

Implements just enough of the MCP spec for tool execution: ``initialize``,
``tools/list`` and ``tools/call`` over HTTP JSON-RPC. Not a full MCP
implementation (no stdio/SSE transports, resources, or prompts).
"""

import json
import uuid

import httpx

_PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    pass


class MCPClient:
    def __init__(self, server_url: str, timeout: float = 30.0) -> None:
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def _request(self, method: str, params: dict) -> dict:
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        response = await self._client.post(
            f"{self._server_url}/mcp",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body, dict) and body.get("jsonrpc") == "2.0" and "result" in body:
            return body["result"]
        if isinstance(body, dict) and "error" in body:
            error = body["error"]
            raise MCPError(error.get("message", "MCP request failed"))
        raise MCPError(f"Unexpected MCP response: {json.dumps(body)[:200]}")

    async def initialize(self) -> dict:
        return await self._request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "ai-employee-tool-registry", "version": "0.1.0"},
        })

    async def list_tools(self) -> list[dict]:
        result = await self._request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        if isinstance(result, dict):
            content = result.get("content", [])
            texts = [
                entry.get("text", "")
                for entry in content
                if isinstance(entry, dict) and entry.get("type") == "text"
            ]
            return {"content": texts, "isError": bool(result.get("isError"))}
        return {"content": [], "isError": True}

    async def aclose(self) -> None:
        await self._client.aclose()
