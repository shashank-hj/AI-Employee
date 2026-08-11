"""Minimal MCP server adapter.

Exposes registered tools to external MCP clients over HTTP JSON-RPC at
``POST /mcp``. Supports ``initialize``, ``tools/list`` and ``tools/call``.
This makes the tool-registry itself an MCP server endpoint (not a full MCP
spec implementation).
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from tool_registry.container import get_tool_service
from tool_registry.schemas.tools import ToolInvokeRequest, ToolListParams
from tool_registry.services.tool_service import ToolService

router = APIRouter(tags=["MCP"])

_PROTOCOL_VERSION = "2024-11-05"
_LIST_PARAMS = ToolListParams()


@router.post("/mcp")
async def mcp_endpoint(request: Request, service: ToolService = Depends(get_tool_service)):
    """JSON-RPC 2.0 MCP endpoint: initialize / tools/list / tools/call."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            },
        )

    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "initialize":
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {
            "protocolVersion": params.get("protocolVersion", _PROTOCOL_VERSION),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ai-employee-tool-registry", "version": "0.1.0"},
        }})

    if method == "notifications/initialized":
        return JSONResponse({})

    if method == "tools/list":
        tools, _ = await service.list_tools(_LIST_PARAMS)
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {
            "tools": [_to_mcp_tool(t) for t in tools],
        }})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        tool = await _find_tool_by_name(service, tool_name)
        if tool is None:
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {
                "code": -32602, "message": f"Unknown tool: {tool_name}",
            }})

        response = await service.invoke_tool(tool.id, ToolInvokeRequest(
            parameters=arguments,
            context={},
        ))
        result = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        json.dumps(response.data)
                        if response.success
                        else (response.error or "")
                    ),
                }
            ],
            "isError": not response.success,
        }
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})

    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {
        "code": -32601, "message": f"Method not found: {method}",
    }})


async def _find_tool_by_name(service: ToolService, name: str):
    tools, _ = await service.list_tools(_LIST_PARAMS)
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _to_mcp_tool(tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "inputSchema": tool.input_schema or {"type": "object", "properties": {}},
    }
